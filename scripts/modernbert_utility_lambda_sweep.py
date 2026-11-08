#!/usr/bin/env python3
"""Full-FT ModernBERT Utility lambda sweep (strict val 450), one GPU sequentially.

Runs the ORIGINAL server-local scripts/train_modernbert_utility.py for training;
never changes its optimizer, labels, architectures, checkpoint criterion, or
prediction format. Hard lambda=0 reuses Classification; soft lambda=0 reuses
Soft-NoCost, so existing checkpoints/score files are not overwritten.

Commands from repo root:
  python scripts/modernbert_utility_lambda_sweep.py self-test
  python scripts/modernbert_utility_lambda_sweep.py plan
  # For training, use Slurm one-GPU allocation, NOT login node:
  python scripts/modernbert_utility_lambda_sweep.py run
  # To recompute results from existing scores without GPU:
  python scripts/modernbert_utility_lambda_sweep.py evaluate

This is exploratory repeated use of 450 validation outcomes. Freeze λ and
checkpoint before independent held-out test. Router/prefill costs excluded.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / 'scripts'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

LAMBDAS = (0., .005, .01, .02, .05, .10)
BUDGETS = (900., 1100., 1300., 1500., 1700., 1900., 2100.)
OUTPUT = ROOT / 'results/route_e/modernbert_utility_lambda_sweep'
ORIGINAL = SCRIPT_DIR / 'train_modernbert_utility.py'


def jobs(lambdas, tau):
    """Each tuple: display objective, λ, source trainer objective, source λ."""
    result = [('classification', 0., 'classification', 0.),
              ('soft_nocost', 0., 'soft_nocost', 0.)]
    for objective in ('utility_hard', 'utility_soft'):
        for lam in lambdas:
            if objective == 'utility_hard' and lam == 0.:
                src, src_lam = 'classification', 0.
            elif objective == 'utility_soft' and lam == 0.:
                src, src_lam = 'soft_nocost', 0.
            else:
                src, src_lam = objective, lam
            result.append((objective, float(lam), src, float(src_lam)))
    return result


def load_existing_module():
    if not ORIGINAL.is_file():
        raise FileNotFoundError(
            f'Missing original training script {ORIGINAL}. Restore the same '
            'script used in the completed ModernBERT Utility ablation; '
            'do not silently substitute another training implementation.')
    import train_modernbert_utility as trainer
    for name in ('BASE', 'read_scores', 'train_name', 'target'):
        if not hasattr(trainer, name):
            raise RuntimeError(f'Original trainer missing {name}; incompatible version')
    return trainer


def source_dir(trainer, objective, lam, tau):
    return Path(trainer.BASE) / trainer.train_name(objective, lam, tau)


def verify_artifacts(folder, objective, lam, tau, strict_train, strict_val, rows):
    pred_path = folder / 'validation_predictions.jsonl'
    meta_path = folder / 'run_metadata.json'
    if not pred_path.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    def check(name, expected):
        if meta.get(name) != expected:
            raise RuntimeError(f'{folder}: {name}={meta.get(name)!r} expected {expected!r}')
    check('objective', objective)
    check('train_sha256', strict_train)
    check('val_sha256', strict_val)
    check('n_train', 3000)
    check('n_validation', 450)
    check('smoke', False)
    check('seed', 42)
    check('batch_size', 16)
    check('max_length', 512)
    if not math.isclose(float(meta.get('learning_rate', -1)), 2e-5,
                        rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f'{folder}: learning rate differs from original 2e-5')
    if not math.isclose(float(meta.get('lambda_cost', -1)), lam, abs_tol=1e-12):
        raise RuntimeError(f'{folder}: lambda metadata mismatch')
    if not math.isclose(float(meta.get('temperature', -1)), tau, abs_tol=1e-12):
        raise RuntimeError(f'{folder}: temperature metadata mismatch')
    if not math.isclose(float(meta.get('epochs', -1)), 5., abs_tol=1e-8):
        raise RuntimeError(f'{folder}: expected 5 training epochs')
    trainer = load_existing_module()
    scores = trainer.read_scores(pred_path)
    if len(scores) != len(rows) or set(scores) != {r['sample_id'] for r in rows}:
        raise ValueError(f'{folder}: expected exactly 450 strict validation IDs')
    return scores


def required_sources(config, trainer):
    unique = {}
    for obj,lam,src,src_lam in jobs(config.lambdas, config.tau):
        name = trainer.train_name(src, src_lam, config.tau)
        unique[name] = (src, src_lam)
    return unique


def preflight(config):
    import route_e_frozen as strict
    train, val, _ = strict.load_protocol()
    trainer = load_existing_module()
    if Path(trainer.BASE) != ROOT / 'outputs/external_modernbert/utility_ablation':
        raise RuntimeError(f'Unexpected training output root: {trainer.BASE}')
    if any(not math.isfinite(x) or x < 0 for x in config.lambdas):
        raise ValueError('Lambdas must be finite and >=0')
    if len(set(config.lambdas)) != len(config.lambdas) or not config.lambdas:
        raise ValueError('Lambda grid must be nonempty, unique')
    if not math.isfinite(config.tau) or config.tau <= 0:
        raise ValueError('tau must be finite and positive')
    if len(train) != 3000 or len(val) != 450:
        raise ValueError('Unexpected strict train/val lengths')
    from route_e_frozen import TRAIN_SHA256, VAL_SHA256
    return trainer, train, val, TRAIN_SHA256, VAL_SHA256


def plan(config, show=True):
    trainer, train, val, trsha, vsha = preflight(config)
    to_train = []
    for name,(obj,lam) in required_sources(config,trainer).items():
        folder=source_dir(trainer,obj,lam,config.tau)
        state=verify_artifacts(folder,obj,lam,config.tau,trsha,vsha,val)
        if state is None:
            if folder.exists() and any(folder.iterdir()):
                raise RuntimeError(f'Incomplete existing run {folder}; inspect manually before retry')
            to_train.append((name,obj,lam,folder))
            status='NEEDS_GPU'
        else:
            status='REUSE_COMPLETE'
        if show:
            print(f'MODERNBERT_LAMBDA_PLAN {name} {status} path={folder}',flush=True)
    if show:
        print(f'MODERNBERT_LAMBDA_PREFLIGHT train={len(train)} val={len(val)} '
              f'gpu_runs_remaining={len(to_train)}',flush=True)
    return trainer, val, trsha, vsha, to_train


def train(config):
    trainer, val, trsha, vsha, todo=plan(config)
    if not todo:
        print('MODERNBERT_LAMBDA_ALL_MODELS_REUSED',flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training needs one GPU: submit ntu_modernbert_utility_lambda_sweep.sbatch')
    for name,obj,lam,folder in todo:
        cmd=[sys.executable,str(ORIGINAL),
             '--objective',obj,'--lambda-cost',str(lam),'--tau',str(config.tau),
             '--model','answerdotai/ModernBERT-base',
             '--lr','2e-5','--epochs','5','--batch-size','16',
             '--max-length','512','--seed','42','--output-dir',str(folder)]
        print(f'MODERNBERT_LAMBDA_TRAIN_START {name}',flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
        if verify_artifacts(folder,obj,lam,config.tau,trsha,vsha,val) is None:
            raise RuntimeError(f'Training did not save valid predictions: {folder}')
        print(f'MODERNBERT_LAMBDA_TRAIN_DONE {name}',flush=True)


def write_csv(path, records, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def evaluate(config):
    trainer, val, trsha, vsha, missing=plan(config,show=False)
    if missing:
        names=', '.join(z[0] for z in missing)
        raise FileNotFoundError(f'Run GPU training first; missing complete runs: {names}')
    from route_e_equal_budget import dense_curve,linear_bracket
    from compute_nauc_ac import calculate
    sampled={}
    curves={}
    all_rows=[]
    budget_rows=[]
    summary=[]
    for obj,lam,src,src_lam in jobs(config.lambdas,config.tau):
        folder=source_dir(trainer,src,src_lam,config.tau)
        scores=verify_artifacts(folder,src,src_lam,config.tau,trsha,vsha,val)
        assert scores is not None
        label=f'{obj} lambda={lam:g}' if obj.startswith('utility_') else obj
        dense=dense_curve(val,scores)
        curves[label]={k:(float(dense[450-k]['tokens']),float(dense[450-k]['accuracy']))
                       for k in range(451)}
        sampled[label]={'scores':scores,'source':str(folder),'objective':obj,'lambda':lam}
        for k in range(451):
            tok,acc=curves[label][k]
            all_rows.append({'method':label,'objective':obj,'lambda':lam,
                             'n_thinking':k,'accuracy':acc,'mean_output_tokens':tok})
        for budget in BUDGETS:
            p=linear_bracket(dense,budget)
            budget_rows.append({'method':label,'objective':obj,'lambda':lam,
                'budget':int(budget),'accuracy_pct':100*p['accuracy'],
                'nothinking_fraction':p['expected_nothinking_fraction']})
    met={item['method']:item for item in calculate(curves)}
    cls_nauc=met['classification']['nAUC_AC']
    for label,info in sampled.items():
        r=met[label]
        summary.append({'method':label,'objective':info['objective'],
                        'lambda':info['lambda'],'AUC_AC':r['AUC_AC'],
                        'nAUC_AC':r['nAUC_AC'],
                        'delta_vs_classification_nAUC_AC':r['nAUC_AC']-cls_nauc,
                        'mean_gain_pp':r['mean_gain_pp'],
                        'cost_backtracks':r['cost_backtracks'],
                        'score_source':info['source']})
        print(f'MODERNBERT_LAMBDA_NAUC {label} nAUC={r["nAUC_AC"]:+.4f} '
              f'delta_vs_classification={r["nAUC_AC"]-cls_nauc:+.4f}',flush=True)
    legacy=met.get('utility_hard lambda=0.02')
    if legacy is not None:
        delta=legacy['nAUC_AC']-9.2095
        if abs(delta)>.025:
            raise RuntimeError(
                f'Historical ModernBERT Utility-Hard λ=.02 nAUC mismatch: '
                f'now={legacy["nAUC_AC"]:+.4f}, expected=+9.2095 '
                '(check strict labels, scores, and original training run)')
        print('MODERNBERT_UTILITY_002_REPRODUCTION_PASS',flush=True)
    selected={}
    for obj in ('utility_hard','utility_soft'):
        subset=[r for r in summary if r['objective']==obj]
        best=min(subset,key=lambda r:(-r['nAUC_AC'],r['lambda']))
        selected[obj]={'lambda':best['lambda'],'nAUC_AC':best['nAUC_AC']}
        print(f'MODERNBERT_LAMBDA_DEV_BEST {obj} lambda={best["lambda"]:g} '
              f'nAUC={best["nAUC_AC"]:+.4f} (exploratory validation)',flush=True)
    out=config.output
    if out.exists() and any(out.iterdir()) and not config.overwrite:
        raise FileExistsError(f'Result output exists: {out}; use --overwrite (does not affect checkpoints)')
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'nauc_ac_lambda_sweep.csv',summary,list(summary[0]))
    write_csv(out/'fixed_budget_accuracy.csv',budget_rows,list(budget_rows[0]))
    write_csv(out/'all_451_threshold_policies.csv',all_rows,list(all_rows[0]))
    (out/'protocol_and_selection.json').write_text(json.dumps({
        'train_sha256':trsha,'val_sha256':vsha,'n_train':3000,'n_val':450,
        'lambda_grid':config.lambdas,'tau':config.tau,
        'trained_encoder':'ModernBERT-base full fine-tuning',
        'training_backend':'original scripts/train_modernbert_utility.py',
        'training_hyperparameters':{'lr':2e-5,'epochs':5,'batch_size':16,
                                    'max_length':512,'seed':42,'fp16':True},
        'metric':'451-policy cost-sorted nAUC-AC minus random expectation',
        'random_reference_AUC':met['classification']['random_reference_AUC'],
        'selected_on_validation_only':selected,'heldout_test_used':False,
        'limitations':'repeated validation selection, measured token-cost interpolation; router overhead not included',
    },indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    if not config.no_plot:
        plot(config,summary)
    print(f'MODERNBERT_LAMBDA_SWEEP_DONE {out}',flush=True)


def plot(config,summary):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('MODERNBERT_LAMBDA_PLOT_SKIPPED matplotlib unavailable',flush=True)
        return
    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for obj,marker in (('utility_hard','o'),('utility_soft','s')):
        sub=sorted((r for r in summary if r['objective']==obj),key=lambda x:x['lambda'])
        ax.plot([r['lambda'] for r in sub],[r['nAUC_AC'] for r in sub],
                marker=marker,linewidth=1.7,label=obj)
    cls=next(r['nAUC_AC'] for r in summary if r['objective']=='classification')
    ax.axhline(cls,color='gray',linestyle='--',label='Classification control')
    ax.axhline(38.3885,color='black',linestyle=':',label='P1 Cost-only (external)')
    ax.set(title='ModernBERT Full FT: Utility lambda sensitivity',
           xlabel='Lambda (cost penalty per 1000 tokens)',ylabel='nAUC-AC')
    ax.grid(alpha=.22)
    ax.legend()
    fig.tight_layout()
    for ext in ('png','svg'):
        fig.savefig(config.output/f'nauc_vs_lambda.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def self_test():
    assert len(jobs(LAMBDAS,.25))==14
    assert len(set(src for _,_,src,_ in jobs(LAMBDAS,.25)))==4
    h=[x for x in jobs(LAMBDAS,.25) if x[0]=='utility_hard' and x[1]==0.][0]
    s=[x for x in jobs(LAMBDAS,.25) if x[0]=='utility_soft' and x[1]==0.][0]
    assert h[2:] == ('classification',0.)
    assert s[2:] == ('soft_nocost',0.)
    assert abs(sum(LAMBDAS)-.185)<1e-9
    print('MODERNBERT_LAMBDA_SWEEP_SELFTEST_PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('self-test')
    for mode in ('plan','train','evaluate','run'):
        q=sub.add_parser(mode)
        q.add_argument('--lambdas',nargs='+',type=float,default=list(LAMBDAS))
        q.add_argument('--tau',type=float,default=.25)
        q.add_argument('--output',type=Path,default=OUTPUT)
        q.add_argument('--overwrite',action='store_true')
        q.add_argument('--no-plot',action='store_true')
    args=p.parse_args()
    if args.command=='self-test':
        self_test()
    elif args.command=='plan':
        plan(args)
    elif args.command=='train':
        train(args)
    elif args.command=='evaluate':
        evaluate(args)
    else:
        train(args)
        evaluate(args)


if __name__=='__main__':
    main()
