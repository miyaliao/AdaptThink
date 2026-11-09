#!/usr/bin/env python3
"""Route E / Experiment F: direct and margin-aware incremental token-cost learning.

Three frozen-P1, CPU-head experiments on the ORIGINAL strict 3000/450 split:
 A0: identical historical P2 Ridge separate log-cost heads (no retraining).
 A1: Ridge regression predicting raw (t_thinking - t_nothinking)/scale.
 A2: P2 separate log-cost Ridge heads + differentiable token-space delta MSE.

A2 beta selection is TRAIN-ONLY K-fold OOF Spearman; validation/test are never
used to tune beta. All routing ranks use PRE-GENERATION predictions only.
Observed validation lengths appear only in DESCRIPTIVE cost-matched replay.

Usage (from the AdaptThink repository root):
  python scripts/route_e_cost_difference.py self-test
  python scripts/route_e_cost_difference.py run
  python scripts/route_e_cost_difference.py run --no-plot --bootstrap-reps 0

Dependencies: numpy, torch and existing AdaptThink scripts/cached P1 features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/route_e/cost_difference/experiment_f'
DEFAULT_BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)
DEFAULT_BETAS = (0., 0.1, 1.)
MAX_TOKENS = 16384.


def finite_array(a, name):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: nonfinite values')
    return a


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def delta_target(rows, scale):
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError('Positive finite scale required')
    return np.asarray([(r['t_t'] - r['t_n'])/scale for r in rows], dtype=np.float64)


def direct_ridge_fit(X, rows, l2, scale):
    """Same feature normalization/regularization definition as historical P2."""
    if l2 <= 0:
        raise ValueError('l2 must be strictly positive')
    X = finite_array(X, 'X')
    y = delta_target(rows, scale)
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-4)
    Z = (X - mean)/std
    n, d = Z.shape
    gram = (Z.T @ Z)/n
    gram.flat[::d+1] += l2
    weights = np.linalg.solve(gram, (Z.T @ (y - y.mean()))/n)
    bias = float(y.mean() - (Z.mean(axis=0) @ weights))
    return {'mean': mean, 'scale': std, 'weight': weights,
            'intercept': bias, 'l2': l2, 'delta_scale': scale}


def direct_ridge_predict(model, X):
    X = finite_array(X, 'X')
    p = (((X-model['mean'])/model['scale']) @ model['weight'] + model['intercept']) * model['delta_scale']
    return np.clip(p, -MAX_TOKENS, MAX_TOKENS)


def margin_predict(model, X):
    """Retransform the same way as P2; smearing is fixed from fold-train A0."""
    X = finite_array(X, 'X')
    Z = (X-model['mean'])/model['scale']
    logits = Z @ model['weight'] + model['intercept']
    logits = np.clip(logits, 0., np.log1p(MAX_TOKENS))
    token_pred = np.clip(np.exp(logits)*model['smear'][None, :] - 1., 0., MAX_TOKENS)
    return token_pred[:, 0]-token_pred[:, 1], token_pred


def fit_margin(X, rows, l2, delta_scale, beta, max_iter, *, base_fit=None):
    """Train-only nonlinear margin loss; beta=0 EXACTLY returns historical P2.

    Loss uses log-token MSE + ridge + beta * MSE(pred_delta/scale, true_delta/scale).
    The predicted tokens use P2's log clipping and fixed train-derived smearing.
    P2's 3-column closed-form fit (delta_p, log_t, log_n) initializes log cost
    heads; quality head is never trained/used by A2.
    """
    import torch
    if beta < 0 or not math.isfinite(beta):
        raise ValueError('Invalid beta')
    if base_fit is None:
        import route_e_token_aware as p2
        base_fit = p2.ridge_fit(X, p2.target_matrix(rows), l2)
    model = {'mean': base_fit['mean'].copy(),
             'scale': base_fit['scale'].copy(),
             'weight': base_fit['weight'][:, 1:3].copy(),
             'intercept': base_fit['intercept'][1:3].copy(),
             'smear': base_fit['smear'].copy(),
             'l2': l2, 'beta': float(beta), 'delta_scale': delta_scale,
             'smearing_rule': 'fixed train-only P2 baseline correction'}
    if beta == 0:
        return model, {'optimized': False, 'reason': 'exact original ridge fit'}

    torch.set_num_threads(min(4, torch.get_num_threads()))
    dtype = torch.float64
    X = finite_array(X, 'features')
    Z = torch.tensor((X - model['mean'])/model['scale'], dtype=dtype)
    yt = torch.tensor([[np.log1p(r['t_t']), np.log1p(r['t_n'])] for r in rows], dtype=dtype)
    target_dt = torch.tensor(delta_target(rows, delta_scale), dtype=dtype)
    smear = torch.tensor(model['smear'], dtype=dtype)
    w = torch.nn.Parameter(torch.tensor(model['weight'].copy(), dtype=dtype))
    b = torch.nn.Parameter(torch.tensor(model['intercept'].copy(), dtype=dtype))
    optim = torch.optim.LBFGS([w,b], lr=1., max_iter=max_iter, history_size=25,
                               tolerance_grad=1e-10, tolerance_change=1e-12,
                               line_search_fn='strong_wolfe')
    state = {'closure_calls': 0}

    def closure():
        optim.zero_grad()
        pred_log = Z @ w + b
        # 1/2 on both data and weight regularizer matches P2's independent
        # mean-MSE + l2 ridge normal equations when beta=0.
        log_mse = (pred_log-yt).square().mean()
        reg = 0.5*l2*w.square().sum()
        log_clipped = torch.clamp(pred_log, 0., math.log1p(MAX_TOKENS))
        tokens = torch.clamp(torch.exp(log_clipped)*smear-1., 0., MAX_TOKENS)
        dt_scaled = (tokens[:,0]-tokens[:,1])/delta_scale
        diff_mse = (dt_scaled-target_dt).square().mean()
        objective = log_mse + reg + beta*diff_mse
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite A2 objective')
        objective.backward()
        state['closure_calls'] += 1
        return objective

    initial = float(closure().detach())
    optim.step(closure)
    final = float(closure().detach())
    new_w, new_b = w.detach().numpy().copy(), b.detach().numpy().copy()
    if not np.isfinite(new_w).all() or not np.isfinite(new_b).all():
        raise FloatingPointError('Nonfinite A2 parameters')
    model['weight'], model['intercept'] = new_w, new_b
    if final > initial + 1e-7:
        raise RuntimeError(f'Optimizer made loss worse: {initial} -> {final}')
    return model, {'optimized': True, 'objective_initial': initial,
                   'objective_final': final, 'closure_calls': state['closure_calls'],
                   'max_iter': max_iter}


def fold_assignments(rows, nfolds, seed):
    """Stable train-only folds stratified by dataset and train-only cost rank."""
    n = len(rows)
    if not 2 <= nfolds <= n:
        raise ValueError('Invalid CV fold count')
    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=np.int64)
    ds = np.asarray([r['dataset'] for r in rows])
    cost = np.asarray([r['t_t'] - r['t_n'] for r in rows])
    for group in np.unique(ds):
        ind = np.flatnonzero(ds==group)
        # Split ranked train cost into roughly equally populated local strata.
        sorted_inds = ind[np.argsort(cost[ind], kind='mergesort')]
        for block in np.array_split(sorted_inds, min(5, len(sorted_inds))):
            block = rng.permutation(block)
            for j, sample in enumerate(block):
                fold[sample] = j % nfolds
    if any(np.sum(fold==k)==0 for k in range(nfolds)):
        raise RuntimeError('Empty CV fold')
    return fold


def select_beta_cv(X, rows, betas, folds, seed, l2, scale, max_iter):
    """ONLY TRAIN data and train labels may flow into this function."""
    import route_e_token_aware as p2
    X = finite_array(X, 'CV train X')
    assign = fold_assignments(rows, folds, seed)
    truth = scale*delta_target(rows, scale)
    oof = {beta: np.full(len(rows), np.nan) for beta in betas}
    per_fold=[]
    for k in range(folds):
        tr_idx = np.flatnonzero(assign != k)
        va_idx = np.flatnonzero(assign == k)
        tr_rows = [rows[i] for i in tr_idx]
        base = p2.ridge_fit(X[tr_idx], p2.target_matrix(tr_rows), l2)
        for beta in betas:
            model, train_meta = fit_margin(X[tr_idx], tr_rows, l2, scale, beta, max_iter, base_fit=base)
            pred, _ = margin_predict(model, X[va_idx])
            oof[beta][va_idx] = pred
            diag = p2.regression_metrics(pred, truth[va_idx])
            per_fold.append({'fold': k, 'beta': beta, 'train_n': len(tr_idx),
                             'oof_n': len(va_idx), 'spearman': diag['spearman'],
                             'mae': diag['mae'], 'train_meta': train_meta})
    metrics={}
    for beta,preds in oof.items():
        if not np.isfinite(preds).all():
            raise AssertionError('Incomplete CV predictions')
        metrics[beta]=p2.regression_metrics(preds, truth)
    # No test or held-out validation outcomes enter beta choice.
    # Deterministic preference for smaller beta on a tied OOF Spearman.
    selected = sorted(betas, key=lambda b: (-(metrics[b]['spearman'] if metrics[b]['spearman'] is not None else -2), b))[0]
    return selected, metrics, per_fold


def auc_cost_sorted(curve, low, high):
    """Matches scripts/compute_nauc_ac.py's 451-cost-sorted trapezoid method."""
    from bisect import bisect_left
    points=sorted((float(r['tokens']), float(r['accuracy'])) for r in curve)
    merged=[]
    for x,y in points:
        if merged and merged[-1][0]==x:
            merged[-1][1]+=y
            merged[-1][2]+=1
        else:
            merged.append([x,y,1])
    xs=[z[0] for z in merged]
    ys=[z[1]/z[2] for z in merged]
    if xs[0]>low+1e-7 or xs[-1]<high-1e-7:
        raise ValueError('Curve does not cover endpoint costs')
    def at(x):
        j=bisect_left(xs,x)
        if j<len(xs) and abs(xs[j]-x)<1e-9: return ys[j]
        if j==0 or j==len(xs): raise ValueError('Interpolation out of range')
        a,b=xs[j-1],xs[j]
        return ys[j-1]+(ys[j]-ys[j-1])*(x-a)/(b-a)
    grid=[low]+[x for x in xs if low<x<high]+[high]
    values=[at(x) for x in grid]
    return sum((values[i]+values[i+1])*0.5*(grid[i+1]-grid[i]) for i in range(len(grid)-1))


def nauc_summary(curves, val):
    n=len(val)
    low=sum(r['t_n'] for r in val)/n
    high=sum(r['t_t'] for r in val)/n
    a_low=sum(r['p_n'] for r in val)/n
    a_high=sum(r['p_t'] for r in val)/n
    if low>=high: raise ValueError('Unexpected NT/T cost endpoints')
    baseline=0.5*(a_low+a_high)*(high-low)
    out={}
    for method,curve in curves.items():
        auc=auc_cost_sorted(curve,low,high)
        backtracks=sum(curve[i+1]['tokens']<curve[i]['tokens']-1e-9 for i in range(len(curve)-1))
        out[method]={'auc_ac': auc, 'nauc_ac': auc-baseline,
                     'mean_gain_pp': 100*(auc-baseline)/(high-low),
                     'cost_backtracks': backtracks}
    return {'baseline_auc': baseline, 'lower_tokens': low, 'upper_tokens': high,
            'methods': out}


def validate_source_a0(val, score_map, expected_path, p2_report_path, budgets):
    """Exact A0 reproduction against the historic cost-only source report."""
    import route_e_cost_only as old_cost
    rows = old_cost.get_validation()
    if [r['sample_id'] for r in val] != [r['sample_id'] for r in rows]:
        raise RuntimeError('Canonical validation ID mismatch')
    maps, pred, report, _ = old_cost.read_sources(rows, expected_path, p2_report_path)
    old_cost.validate_original(rows, maps, report, budgets)
    src = maps['Cost_only']
    if set(score_map) != set(src):
        raise RuntimeError('A0 predicted ID mismatch')
    abs_err=max(abs(score_map[s]-src[s]) for s in src)
    if abs_err > 1e-5:
        raise RuntimeError(f'A0 differs from saved historical Cost-only scores, max={abs_err:g}')
    return {'prediction_max_abs_error': abs_err,
            'saved_prediction_sha256': hash_file(expected_path),
            'original_p2_report_sha256': hash_file(p2_report_path),
            'source_ranking_matches': True}


def predicted_only_budget(val, baseline_lengths, score, budgets):
    """Prospective diagnostic: SHARED original A0 length forecasts for every arm.

    The top-k ordering uses each arm's own predicted delta; budget bracket is
    selected from common A0 predicted lengths, NEVER observed validation cost.
    This intentionally isolates score ordering, NOT native cost calibration of A1.
    """
    import route_e_token_aware as p2
    out=[]
    for budget in budgets:
        q=p2.prospective_budget_policy(val, baseline_lengths, -np.asarray(score), float(budget))
        out.append(q)
    return out


def run(args):
    import route_e_frozen as base
    import route_e_token_aware as p2
    from route_e_equal_budget import dense_curve, linear_bracket, random_at_budget

    out=Path(args.out_dir)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f'{out} already exists: pass --overwrite for NEW experiment F only')
    if args.l2!=0.02:
        raise ValueError('Keep historical A0 l2=0.02 for controlled comparison')
    if len(args.betas)!=len(set(args.betas)) or 0. not in args.betas or any(b<0 for b in args.betas):
        raise ValueError('Betas must be unique, nonnegative, and include 0')
    if len(args.budgets)!=len(set(args.budgets)) or any(b<=0 for b in args.budgets):
        raise ValueError('Invalid budgets')
    if args.bootstrap_reps<0: raise ValueError('Invalid bootstrap repetitions')

    train, val, Xtr, Xva, tm, vm, overlaps = p2.read_protocol_and_cache(False)
    if len(train)!=3000 or len(val)!=450:
        raise AssertionError('Strict 3000/450 protocol required')
    srcpred=base.ROOT/'outputs/route_e/token_aware/full/validation_predictions.json'
    srcp2=base.ROOT/'results/route_e/token_aware/full/validation.json'
    if not srcpred.is_file() or not srcp2.is_file():
        raise FileNotFoundError('Run original P2 token-aware full stage before Experiment F')

    # A0 is refit deterministically only for exact numeric auditing; the frozen
    # historical source is the benchmark, never overwritten/replaced.
    a0_fit=p2.ridge_fit(Xtr,p2.target_matrix(train),args.l2)
    a0_pred=p2.predict_heads(a0_fit,Xva)
    a0_delta=a0_pred['t_t']-a0_pred['t_n']
    score_maps={'A0_existing_cost_only': {r['sample_id']:float(z) for r,z in zip(val,a0_delta)}}
    baseline_check=validate_source_a0(val,score_maps['A0_existing_cost_only'],srcpred,srcp2,
                                      [float(b) for b in args.budgets])
    print(f'COST_DIFF_A0_REPRO_PASS max_abs_diff={baseline_check["prediction_max_abs_error"]:.3g}',flush=True)

    a1_fit=direct_ridge_fit(Xtr,train,args.l2,args.delta_scale)
    a1_delta=direct_ridge_predict(a1_fit,Xva)
    score_maps['A1_direct_delta']={r['sample_id']:float(z) for r,z in zip(val,a1_delta)}

    selected,cv_metrics,cv_folds=select_beta_cv(Xtr,train,args.betas,args.cv_folds,
                                                args.seed,args.l2,args.delta_scale,args.lbfgs_steps)
    print('COST_DIFF_TRAIN_ONLY_CV '+ ' '.join(f'beta={b:g}:rho={cv_metrics[b]["spearman"]:.4f}' for b in args.betas),flush=True)
    print(f'COST_DIFF_SELECTED_BETA {selected:g}',flush=True)
    a2_fit,train_details=fit_margin(Xtr,train,args.l2,args.delta_scale,selected,
                                    args.lbfgs_steps,base_fit=a0_fit)
    a2_delta,a2_lengths=margin_predict(a2_fit,Xva)
    score_maps['A2_margin_selected']={r['sample_id']:float(z) for r,z in zip(val,a2_delta)}
    if selected==0 and not np.allclose(a0_delta,a2_delta,rtol=0,atol=1e-6):
        raise AssertionError('A2 beta=0 should reproduce original A0')

    truth=np.asarray([r['t_t']-r['t_n'] for r in val])
    diagnostic_rows=[]
    for name,scores in score_maps.items():
        pred=np.asarray([scores[r['sample_id']] for r in val])
        for ds in ['overall']+sorted(set(r['dataset'] for r in val)):
            mask=np.ones(len(val),dtype=bool) if ds=='overall' else np.array([r['dataset']==ds for r in val])
            diag=p2.regression_metrics(pred[mask],truth[mask])
            diagnostic_rows.append({'method':name,'dataset':ds,'n':int(mask.sum()), **diag})

    curves={name:dense_curve(val,scores) for name,scores in score_maps.items()}
    area=nauc_summary(curves,val)
    budgets=[]; csv_budget=[]; csv_bootstrap=[]
    order=list(score_maps)
    for budget in args.budgets:
        record={'budget':float(budget),'methods':{},'paired_bootstrap':{}}
        rand=random_at_budget(val,budget)
        pn_map={}
        for name in order:
            curve=curves[name]
            bracket=linear_bracket(curve,budget)
            pn=p2.nt_prob_at_bracket(val,score_maps[name],bracket)
            groups=p2.group_metrics(val,pn)
            if abs(groups['overall']['actual_tokens']-budget)>1e-4:
                raise AssertionError('Measured-cost replay mismatch')
            pn_map[name]=pn
            record['methods'][name]={'mix':bracket,'groups':groups,
                                     'uplift_vs_global_random_pp':100*(groups['overall']['accuracy']-rand['accuracy'])}
            for ds,g in groups.items():
                csv_budget.append({'budget':budget,'method':name,'dataset':ds,
                                   'accuracy_pct':100*g['accuracy'],'actual_tokens':g['actual_tokens'],
                                   'nothinking_fraction':g['nothinking_fraction'],
                                   'uplift_vs_random_pp':100*(g['accuracy']-rand['accuracy']) if ds=='overall' else ''})
        for left,right in ((order[1],order[0]),(order[2],order[0]),(order[2],order[1])):
            label=left+'_minus_'+right
            boot=p2.paired_bootstrap(val,pn_map[left],pn_map[right],args.bootstrap_reps,args.seed+int(budget))
            record['paired_bootstrap'][label]=boot
            if boot:
                csv_bootstrap.append({'budget':budget,'comparison':label,
                                      'difference_pp':boot['difference_pp'],
                                      'ci95_low_pp':boot['ci_95_pp'][0],
                                      'ci95_high_pp':boot['ci_95_pp'][1],
                                      'bootstrap_repetitions':boot['repetitions']})
        budgets.append(record)
        print('COST_DIFF_BUDGET '+str(budget)+' '+ ' '.join(f'{name}={100*record["methods"][name]["groups"]["overall"]["accuracy"]:.2f}%' for name in order),flush=True)

    # Predicted-cost-only diagnostic. One COMMON original-P2 length forecast
    # is used across A0/A1/A2, so this compares ordering, NOT recalibration.
    prospective={}
    for name,scores in score_maps.items():
        score=np.asarray([scores[r['sample_id']] for r in val],dtype=np.float64)
        prospective[name]=predicted_only_budget(val,{'t_t':a0_pred['t_t'],'t_n':a0_pred['t_n']},
                                               score,args.budgets)

    if out.exists() and args.overwrite:
        # Only overwrite defined Experiment F outputs; don't delete other files.
        pass
    out.mkdir(parents=True,exist_ok=True)
    config={'experiment':'Route E / Experiment F: Cost Difference Supervision',
            'base_encoder':'Frozen DeepSeek-R1-Distill-Qwen-1.5B P1 assistant-only, final norm, last valid prompt token',
            'train_n':len(train),'val_n':len(val),'train_sha256':base.TRAIN_SHA256,
            'val_sha256':base.VAL_SHA256,'train_feature_meta':tm,'val_feature_meta':vm,
            'l2':args.l2,'delta_scale':args.delta_scale,'betas_cv':args.betas,
            'selected_beta':selected,'cv_folds':args.cv_folds,'lbfgs_steps':args.lbfgs_steps,
            'budgets':args.budgets,'bootstrap_reps':args.bootstrap_reps,'seed':args.seed,
            'a0_reproduction':baseline_check,'overlap_normalized_texts':overlaps,
            'validation_used_for_hyperparameter_selection':False,
            'test_accessed':False,'responses_regenerated':False,
            'cost_matching_uses_observed_validation_lengths':True,
            'prospective_cost_forecast':'shared historical P2 A0 lengths, transductive unlabeled validation features'}
    save_json(out/'config.json',config)
    save_json(out/'cv_beta_selection.json',{'selected_beta':selected,
             'metrics':{str(k):v for k,v in cv_metrics.items()},'folds':cv_folds})
    save_csv(out/'cost_diagnostics.csv', ['method','dataset','n','mae','rmse','spearman',
                                         'pred_mean','actual_mean','bias_pred_minus_actual'],diagnostic_rows)
    save_csv(out/'equal_budget.csv',['budget','method','dataset','accuracy_pct','actual_tokens',
                                      'nothinking_fraction','uplift_vs_random_pp'],csv_budget)
    save_csv(out/'paired_bootstrap.csv',['budget','comparison','difference_pp','ci95_low_pp',
                                         'ci95_high_pp','bootstrap_repetitions'],csv_bootstrap)
    save_json(out/'validation_pre_generation_scores.json',{'strict_val_sha256':base.VAL_SHA256,
            'scores_are_predictions_only':True,'no_actual_validation_labels_or_tokens':True,
            'score_direction':'higher = NoThinking first',
            'methods':score_maps})
    np.savez_compressed(out/'trained_heads.npz',
                        a1_mean=a1_fit['mean'],a1_scale=a1_fit['scale'],
                        a1_weight=a1_fit['weight'],a1_intercept=np.array(a1_fit['intercept']),
                        a2_mean=a2_fit['mean'],a2_scale=a2_fit['scale'],
                        a2_weight=a2_fit['weight'],a2_intercept=a2_fit['intercept'],
                        a2_smear=a2_fit['smear'],a2_beta=np.array(selected))
    save_json(out/'prospective_shared_a0_lengths.json',{
             'note':'Predicted-only batch threshold; same original A0 length predictions across methods. Actual costs assessed afterwards, not a deployable guarantee.',
             'results':prospective})
    summary={'nauc_ac':area,'selected_beta':selected,
             'a2_training':train_details,'a0_reproduction':baseline_check,
             'cost_diagnostics':diagnostic_rows,'budget_comparison':budgets,
             'warnings':['exploratory validation: multiple prior experiments examined same split',
                         'observed-token interpolation is descriptive, not deployable',
                         'mean generated tokens omit encoder/router overhead',
                         'train/validation token-length distribution shift',
                         'nominal paired bootstrap CIs condition on fixed fitted policies and ignore selection uncertainty']}
    save_json(out/'summary.json',summary)

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(9,5.4))
            for name in order:
                ax.plot(args.budgets,[100*r['methods'][name]['groups']['overall']['accuracy'] for r in budgets],
                        marker='o',label=name)
            ax.plot(args.budgets,[100*random_at_budget(val,b)['accuracy'] for b in args.budgets],
                    linestyle='--',label='Global Random')
            ax.set(xlabel='Mean generated tokens / problem (observed-cost offline replay)',
                   ylabel='Strict validation accuracy (%)',title='Experiment F: Cost Difference Supervision')
            ax.legend(); ax.grid(alpha=.25);fig.tight_layout()
            for ext in ('svg','png'):
                fig.savefig(out/f'accuracy_token_curve.{ext}',dpi=160)
            plt.close(fig)
        except ImportError:
            print('COST_DIFF_PLOT_SKIPPED matplotlib unavailable',flush=True)
    print('COST_DIFF_NAUC '+' '.join(f'{name}={area["methods"][name]["nauc_ac"]:+.4f}' for name in order),flush=True)
    print(f'COST_DIFF_DONE output={out}',flush=True)


def self_test():
    """No cache/GPU/real data required. Checks loss=0 parity & direct head."""
    import route_e_token_aware as p2
    from route_e_equal_budget import dense_curve
    rng=np.random.default_rng(42)
    n,d=48,7
    X=rng.normal(size=(n,d))
    tt=np.maximum(np.exp(4.+X[:,0]*.3+rng.normal(0,.2,n))-1,0)
    tn=np.maximum(np.exp(3.+X[:,1]*.3+rng.normal(0,.2,n))-1,0)
    rows=[{'sample_id':f'test:{i}','dataset':'test','t_t':float(tt[i]),
           't_n':float(tn[i]),'p_t':0.8,'p_n':0.6,'delta':0.2} for i in range(n)]
    base=p2.ridge_fit(X,p2.target_matrix(rows),.02)
    a0=p2.predict_heads(base,X)
    z,_=margin_predict(fit_margin(X,rows,.02,1000.,0.,30,base_fit=base)[0],X)
    assert np.allclose(z,a0['t_t']-a0['t_n'],rtol=0,atol=1e-9)
    dmodel=direct_ridge_fit(X,rows,.02,1000.)
    direct=direct_ridge_predict(dmodel,X)
    assert np.isfinite(direct).all()
    m,log=fit_margin(X,rows,.02,1000.,0.1,35,base_fit=base)
    p,_=margin_predict(m,X)
    assert np.isfinite(p).all() and log['objective_final']<=log['objective_initial']+1e-7
    order={r['sample_id']:float(v) for r,v in zip(rows,p)}
    assert dense_curve(rows,order)[0]['n_nothinking']==0
    assert dense_curve(rows,order)[-1]['n_nothinking']==n
    k=fold_assignments(rows,3,42)
    assert all((k==i).sum()>0 for i in range(3))
    demo=[{'tokens':10,'accuracy':.2},{'tokens':15,'accuracy':.5},{'tokens':20,'accuracy':.8}]
    assert abs(auc_cost_sorted(demo,10,20)-5)<1e-10
    print('COST_DIFF_SELFTEST_PASS (raw-delta ridge, beta0 parity, nonlinear margin, ranking, folds, nAUC)',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sp=p.add_subparsers(dest='command',required=True)
    sp.add_parser('self-test')
    r=sp.add_parser('run')
    r.add_argument('--out-dir',type=Path,default=DEFAULT_OUT)
    r.add_argument('--l2',type=float,default=.02)
    r.add_argument('--delta-scale',type=float,default=1000.)
    r.add_argument('--betas',type=float,nargs='+',default=list(DEFAULT_BETAS))
    r.add_argument('--cv-folds',type=int,default=3)
    r.add_argument('--lbfgs-steps',type=int,default=80)
    r.add_argument('--budgets',type=float,nargs='+',default=list(DEFAULT_BUDGETS))
    r.add_argument('--bootstrap-reps',type=int,default=2000)
    r.add_argument('--seed',type=int,default=42)
    r.add_argument('--overwrite',action='store_true')
    r.add_argument('--no-plot',action='store_true')
    args=p.parse_args()
    if args.command=='self-test': self_test()
    else: run(args)


if __name__=='__main__':
    main()
