#!/usr/bin/env python3
"""Independent test evaluation for Route E Experiment F (A0/A1/A2).

One-off test protocol. No retraining, test-based beta/threshold selection, or
regeneration of model answers. The test-set grades are accessed only by
'evaluate', AFTER the validation-derived thresholds and test scores exist.

  python scripts/route_e_cost_difference_test.py self-test
  python scripts/route_e_cost_difference_test.py freeze
  python scripts/route_e_cost_difference_test.py check
  python scripts/route_e_cost_difference_test.py extract  # GPU; only test questions
  python scripts/route_e_cost_difference_test.py score    # CPU; only test features
  python scripts/route_e_cost_difference_test.py evaluate # CPU; reads test grades

Outputs are separate from validation and historical P2 artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import route_e_cost_difference as f

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT/'data/splits/adaptthink_original_baseline_seed42.json'
VALIDATION = ROOT/'results/val_450_labels.jsonl'
VALIDATION_SCORES = f.DEFAULT_OUT/'validation_pre_generation_scores.json'
VALIDATION_CONFIG = f.DEFAULT_OUT/'config.json'
TRAINED = f.DEFAULT_OUT/'trained_heads.npz'
P2_HEADS = ROOT/'outputs/route_e/token_aware/full/ridge_heads.npz'
P1_TRAIN = ROOT/'outputs/route_e/prompt_ablation/assistant/full/train.npz'
P1_VAL = ROOT/'outputs/route_e/prompt_ablation/assistant/full/val.npz'
GRADES = {
    'thinking': ROOT/'results/baseline_1p5b/thinking_grades.json',
    'nothinking': ROOT/'results/baseline_1p5b/nothinking_grades.json',
}
OUT = ROOT/'results/route_e/cost_difference/test'
FEATURES = ROOT/'outputs/route_e/cost_difference/test/test_p1_features.npz'
METHODS = ('A0_existing_cost_only', 'A1_direct_delta', 'A2_margin_selected')
EXPECTED_COUNTS = {'gsm8k':330, 'math':125, 'aime':30}


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return f.hash_file(path)


def safe_write(path, content, overwrite=False):
    path=Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f'{path}: exists; refusing to change test artifacts')
    f.save_json(path,content)


def check_source_blob(path, expected):
    raw=Path(path).read_bytes()
    blob_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode('ascii')+raw).hexdigest()
    if blob_sha!=expected:
        raise ValueError(f'Git blob hash mismatch {path}: {blob_sha} != {expected}')
    return hashlib.sha256(raw).hexdigest()


def test_questions():
    """No test grades here. Enforce the fixed 485-question canonical split."""
    split=load_json(SPLIT)
    if split['split_name']!='adaptthink_original_baseline_seed42' or split['schema_version']!=2:
        raise ValueError('Wrong canonical test split')
    rows=[]; sources={}
    for ds in ('gsm8k','math','aime'):
        src=split['source_files'][ds]
        path=ROOT/src['path']
        sources[ds]=check_source_blob(path,src['git_blob_sha1'])
        data=load_json(path)
        if len(data)!=src['num_rows']:
            raise ValueError(f'{ds}: unexpected source length')
        inds=src['selected_indices']
        if len(inds)!=EXPECTED_COUNTS[ds] or len(set(inds))!=len(inds):
            raise ValueError(f'{ds}: wrong split indices')
        if inds != sorted(inds):
            raise ValueError(f'{ds}: split indices are not in ascending order')
        for idx in inds:
            if not 0<=idx<len(data) or not isinstance(data[idx]['problem'],str) or not data[idx]['problem'].strip():
                raise ValueError(f'Invalid test question: {ds}:{idx}')
            rows.append({'sample_id':f'{ds}:{idx}','dataset':ds,'problem':data[idx]['problem']})
    if len(rows)!=485 or len(set(z['sample_id'] for z in rows))!=485:
        raise AssertionError('Test must have exactly 485 unique questions')
    return rows, {'split_sha256':sha(SPLIT),'source_sha256':sources}


def test_grades(questions):
    """Call ONLY in evaluate: AIME 16 repeats are collapsed per question."""
    expected={r['sample_id']:r for r in questions}
    by_mode={}
    grade_hashes={}
    for mode,path in GRADES.items():
        data=load_json(path)
        grade_hashes[mode]=sha(path)
        seen=set(); groups=defaultdict(dict)
        for record in data:
            if record.get('mode')!=mode:
                raise ValueError(f'{mode}: mode field mismatch')
            raw_id=str(record['id'])
            if raw_id in seen:
                raise ValueError(f'Duplicate generation {raw_id}')
            seen.add(raw_id)
            a=raw_id.split(':')
            if len(a)!=3:
                raise ValueError(f'Bad generation ID: {raw_id}')
            ds,index,repetition=a
            sid=f'{ds}:{index}'
            if sid not in expected or record.get('dataset')!=ds:
                raise ValueError(f'Generation outside fixed split: {raw_id}')
            r=int(repetition)
            if r in groups[sid]:
                raise ValueError(f'Duplicate repeat {raw_id}')
            acc=float(record['accuracy'])
            tokens=float(record['num_output_tokens'])
            if not math.isfinite(acc) or not math.isfinite(tokens) or not (0<=acc<=1) or tokens<0:
                raise ValueError(f'Invalid grade fields: {raw_id}')
            groups[sid][r]=(acc,tokens)
        if len(groups)!=485 or len(seen)!=935:
            raise ValueError(f'{mode}: expected 485 questions / 935 generations, got {len(groups)}/{len(seen)}')
        means={}
        for sid,q in expected.items():
            want=set(range(16)) if q['dataset']=='aime' else {0}
            if set(groups[sid])!=want:
                raise ValueError(f'{mode}: incomplete repetitions: {sid}')
            a=np.asarray(list(groups[sid].values()),dtype=np.float64)
            means[sid]=(float(a[:,0].mean()),float(a[:,1].mean()))
        by_mode[mode]=means
    rows=[]
    for q in questions:
        sid=q['sample_id']
        at,tt=by_mode['thinking'][sid]
        an,tn=by_mode['nothinking'][sid]
        rows.append({**q,'p_t':at,'p_n':an,'t_t':tt,'t_n':tn,'delta':at-an})
    return rows,grade_hashes


def train_feature_metadata():
    with np.load(P1_TRAIN,allow_pickle=False) as npz:
        train=load_metadata(npz)
    with np.load(P1_VAL,allow_pickle=False) as npz:
        val=load_metadata(npz)
    fields=('model','chat_template_sha256','prompt_mode','layer','pooling','dtype','feature_dim')
    for k in fields:
        if train[k]!=val[k]:
            raise ValueError(f'Train/val feature metadata differs: {k}')
    if train['prompt_mode']!='assistant' or train['layer']!='last_after_final_norm' or train['pooling']!='last_valid_prompt_token' or train['feature_dim']!=1536:
        raise ValueError('Not a P1 frozen last-layer 1536-d feature cache')
    return train


def load_metadata(npz):
    return json.loads(str(npz['metadata_json'].item()))


def frozen_threshold(sorted_scores,k):
    """Convert a VALIDATION top-k choice into a deployable numerical cutoff."""
    sorted_scores=np.asarray(sorted_scores,dtype=np.float64)
    n=len(sorted_scores)
    if not 0<=k<=n:
        raise ValueError('Invalid cutoff index')
    if k==0:
        return float(sorted_scores[0]+max(1.,abs(sorted_scores[0]))*1e-7)
    if k==n:
        return float(sorted_scores[-1]-max(1.,abs(sorted_scores[-1]))*1e-7)
    x,y=sorted_scores[k-1],sorted_scores[k]
    if x<=y:
        raise ValueError('Ambiguous tied scores at threshold boundary')
    z=float(x+(y-x)*0.5)
    if not y<z<x:
        raise ValueError('Numerically unseparable score cutoff')
    return z


def freeze(args):
    """Freeze seven validation-selected thresholds; DOES NOT open test files."""
    import route_e_frozen as base
    from route_e_equal_budget import dense_curve,linear_bracket
    import route_e_token_aware as p2
    from train_modernbert_delta import read_labels
    cfg=load_json(VALIDATION_CONFIG)
    if cfg['train_sha256']!=base.TRAIN_SHA256 or cfg['val_sha256']!=base.VAL_SHA256 or cfg['train_n']!=3000 or cfg['val_n']!=450:
        raise ValueError('Invalid original Experiment F provenance')
    if not math.isclose(float(cfg['selected_beta']),0.,abs_tol=1e-12):
        raise ValueError('Selected beta changed: freeze should be reviewed before test')
    if sha(VALIDATION)!=base.VAL_SHA256:
        raise ValueError('Strict validation hash mismatch')
    val=read_labels(VALIDATION)
    scores_file=load_json(VALIDATION_SCORES)
    if scores_file['strict_val_sha256']!=base.VAL_SHA256 or not scores_file['scores_are_predictions_only']:
        raise ValueError('Invalid validation predictions')
    scores=scores_file['methods']
    ids=[r['sample_id'] for r in val]
    report={'source':'validation-frozen cutoff mixture (NOT test-cost calibration)',
            'validation_score_sha256':sha(VALIDATION_SCORES),
            'validation_config_sha256':sha(VALIDATION_CONFIG),
            'validation_labels_sha256':base.VAL_SHA256,
            'selected_beta':cfg['selected_beta'],'methods':{}}
    for method in METHODS:
        if set(scores[method])!=set(ids):
            raise ValueError(f'{method}: validation score ID mismatch')
        rank=sorted(ids,key=lambda sid:(-scores[method][sid],sid))
        arr=[float(scores[method][sid]) for sid in rank]
        curve=dense_curve(val,scores[method])
        anchors=[]
        for budget in cfg['budgets']:
            bracket=linear_bracket(curve,float(budget))
            kl=int(round(bracket['policy_left_fraction']*len(val)))
            kr=int(round(bracket['policy_right_fraction']*len(val)))
            left=frozen_threshold(arr,kl)
            right=frozen_threshold(arr,kr)
            w=float(bracket['mix_right_probability'])
            # Verify frozen numerical score cutoff reproduces the VAL policy.
            v=np.array([scores[method][sid] for sid in ids])
            pn=(1-w)*(v>=left)+w*(v>=right)
            prev=p2.nt_prob_at_bracket(val,scores[method],bracket)
            if not np.allclose(pn,prev,rtol=0,atol=1e-12):
                raise RuntimeError('Frozen validation threshold does not reproduce original policy')
            anchors.append({'requested_validation_budget':float(budget),
                            'val_accuracy':float(bracket['accuracy']),
                            'left_cutoff':left,'right_cutoff':right,
                            'mix_right_probability':w,
                            'validation_k_left':kl,'validation_k_right':kr})
        report['methods'][method]=anchors
    safe_write(OUT/'frozen_validation_policies.json',report,args.overwrite)
    print('COST_DIFF_TEST_FREEZE_PASS validation_only=1 beta=0 methods=3',flush=True)


def check(_):
    rows,manifest=test_questions()
    meta=train_feature_metadata()
    print(f'COST_DIFF_TEST_CHECK_PASS n={len(rows)} gsm8k=330 math=125 aime=30 template_sha={meta["chat_template_sha256"][:12]} split_sha={manifest["split_sha256"][:12]}',flush=True)


def extract(args):
    """GPU extraction of P1 features; reads only question texts, never grades."""
    import torch
    from transformers import AutoModel,AutoTokenizer
    import route_e_prompt_ablation as prefix
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for DeepSeek prefill extraction')
    if FEATURES.exists():
        raise FileExistsError(f'Refusing to overwrite pre-existing test feature cache: {FEATURES}')
    rows,manifest=test_questions()
    meta=train_feature_metadata()
    model_path=Path(args.model or meta['model'])
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if str(model_path)!=str(meta['model']):
        raise ValueError('Test checkpoint path differs from training P1 checkpoint path')
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False)
    tokenizer.padding_side='right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    th=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
    if th!=meta['chat_template_sha256']:
        raise ValueError('Test tokenizer template differs from train P1')
    sequences=prefix.tokenize_ablations(tokenizer,[r['problem'] for r in rows],'assistant',args.max_length)
    model=AutoModel.from_pretrained(str(model_path),torch_dtype=torch.float16,
                                    local_files_only=True,trust_remote_code=False).to('cuda').eval()
    for param in model.parameters():param.requires_grad_(False)
    pads=tokenizer.pad_token_id
    xs=[];t0=time.perf_counter()
    with torch.inference_mode():
        for st in range(0,len(rows),args.batch_size):
            batch=sequences[st:st+args.batch_size]
            width=max(len(q) for q in batch)
            inp=torch.full((len(batch),width),pads,dtype=torch.long,device='cuda')
            mask=torch.zeros_like(inp)
            for j,seq in enumerate(batch):
                inp[j,:len(seq)]=torch.tensor(seq,device='cuda')
                mask[j,:len(seq)]=1
            hidden=model(input_ids=inp,attention_mask=mask,return_dict=True,use_cache=False).last_hidden_state
            loc=torch.tensor([len(seq)-1 for seq in batch],device='cuda')
            X=hidden[torch.arange(len(batch),device='cuda'),loc].float().cpu().numpy().astype(np.float16)
            if X.shape!=(len(batch),1536) or not np.isfinite(X).all():
                raise ValueError('Invalid extracted test embeddings')
            xs.append(X)
            if st==0 or st+args.batch_size>=len(rows):
                print(f'COST_DIFF_TEST_EXTRACT {min(st+args.batch_size,len(rows))}/{len(rows)}',flush=True)
    combined=np.concatenate(xs,axis=0)
    report={'schema':'route_e_cost_difference_test_p1_v1',
            'layer':meta['layer'],'pooling':meta['pooling'],
            'model':str(model_path),'prompt_mode':'assistant',
            'chat_template_sha256':th,'source_manifest':manifest,
            'train_feature_cache_sha256':sha(P1_TRAIN),
            'feature_dim':1536,'max_length':args.max_length,'dtype':'float16',
            'elapsed_seconds':time.perf_counter()-t0,
            'sample_count':485,'grade_files_read':False}
    FEATURES.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(FEATURES,X=combined,sample_id=np.array([r['sample_id'] for r in rows]),
                        metadata_json=np.array(json.dumps(report)))
    print(f'COST_DIFF_TEST_FEATURES_SAVED {FEATURES}',flush=True)


def load_test_features():
    rows,manifest=test_questions()
    with np.load(FEATURES,allow_pickle=False) as d:
        X=d['X'].astype(np.float64)
        ids=d['sample_id'].astype(str).tolist()
        meta=load_metadata(d)
    if ids != [r['sample_id'] for r in rows] or X.shape!=(485,1536) or not np.isfinite(X).all():
        raise ValueError('Test features / canonical 485 IDs mismatch')
    if meta['schema']!='route_e_cost_difference_test_p1_v1' or meta['source_manifest']!=manifest or meta['sample_count']!=485:
        raise ValueError('Wrong P1 test metadata or source hashes')
    reference=train_feature_metadata()
    for k in ('model','chat_template_sha256','layer','pooling','prompt_mode'):
        if meta[k]!=reference[k]:
            raise ValueError(f'Test feature provenance mismatch on {k}')
    if meta['train_feature_cache_sha256']!=sha(P1_TRAIN):
        raise ValueError('Original train feature cache changed')
    return rows, X, manifest


def score(args):
    """Only test questions/hidden states/heads; no test grades."""
    import route_e_token_aware as p2
    cfg=load_json(VALIDATION_CONFIG)
    if cfg['selected_beta'] != 0:
        raise ValueError('Expected locked Experiment F beta=0 from train CV')
    rows,X,manifest=load_test_features()
    with np.load(P2_HEADS,allow_pickle=False) as d:
        a0={k:d[k] for k in ('mean','scale','weight','intercept','smear')}
    with np.load(TRAINED,allow_pickle=False) as d:
        saved={k:d[k] for k in d.files}
    if saved['a2_beta'].item()!=cfg['selected_beta']:
        raise ValueError('Stored A2 beta differs from locked experiment config')
    a0_pred=p2.predict_heads(a0,X)
    t0=a0_pred['t_t']-a0_pred['t_n']
    a1={'mean':saved['a1_mean'],'scale':saved['a1_scale'],
        'weight':saved['a1_weight'],'intercept':float(saved['a1_intercept']),
        'delta_scale':cfg['delta_scale']}
    t1=f.direct_ridge_predict(a1,X)
    a2={'mean':saved['a2_mean'],'scale':saved['a2_scale'],'weight':saved['a2_weight'],
        'intercept':saved['a2_intercept'],'smear':saved['a2_smear']}
    t2,_=f.margin_predict(a2,X)
    if not np.allclose(t0,t2,rtol=0,atol=1e-4):
        raise RuntimeError('A2 beta=0 MUST equal A0 on independent test features')
    out={'schema':'route_e_cost_difference_test_scores_v1',
         'test_source_manifest':manifest,'test_features_sha256':sha(FEATURES),
         'experiment_f_config_sha256':sha(VALIDATION_CONFIG),
         'trained_heads_sha256':sha(TRAINED),'original_p2_heads_sha256':sha(P2_HEADS),
         'selected_beta':cfg['selected_beta'],
         'scores_are_pre_generation_only':True,
         'modes':{name:{r['sample_id']:float(z) for r,z in zip(rows,arr)}
                  for name,arr in zip(METHODS,(t0,t1,t2))}}
    safe_write(OUT/'test_pre_generation_scores.json',out,args.overwrite)
    print('COST_DIFF_TEST_SCORE_PASS test_grades_read=0 n=485 beta=0 A2_equals_A0=1',flush=True)


def frozen_pn(scores, cut):
    s=np.asarray(scores,dtype=np.float64)
    w=float(cut['mix_right_probability'])
    return (1-w)*(s>=float(cut['left_cutoff']))+w*(s>=float(cut['right_cutoff']))


def evaluate(args):
    import route_e_token_aware as p2
    from route_e_equal_budget import dense_curve,linear_bracket,random_at_budget
    questions,manifest=test_questions()
    pol=load_json(OUT/'frozen_validation_policies.json')
    score_file=load_json(OUT/'test_pre_generation_scores.json')
    cfg=load_json(VALIDATION_CONFIG)
    if pol['validation_score_sha256']!=sha(VALIDATION_SCORES) or pol['validation_config_sha256']!=sha(VALIDATION_CONFIG):
        raise ValueError('Frozen validation policy provenance drifted')
    if score_file['experiment_f_config_sha256']!=sha(VALIDATION_CONFIG) or score_file['test_source_manifest']!=manifest:
        raise ValueError('Test scores are stale/mismatched')
    if score_file['test_features_sha256']!=sha(FEATURES) or score_file['trained_heads_sha256']!=sha(TRAINED) or score_file['original_p2_heads_sha256']!=sha(P2_HEADS):
        raise ValueError('Prediction source weights/features changed')
    if pol['selected_beta']!=cfg['selected_beta'] or cfg['selected_beta']!=0:
        raise ValueError('A2 beta changed between validation and test')
    scores=score_file['modes']
    ids=[q['sample_id'] for q in questions]
    if any(set(scores[m])!=set(ids) for m in METHODS):
        raise ValueError('Test predicted score IDs mismatch')
    if (OUT/'test_summary.json').exists() and not args.overwrite:
        raise FileExistsError('Test already evaluated; refusing to overwrite')
    # Test grades are opened ONLY below, after confirming selected heads,
    # validation-frozen thresholds, split hashes, and pre-generation scores.
    rows,grade_hashes=test_grades(questions)
    curves={m:dense_curve(rows,scores[m]) for m in METHODS}
    area=f.nauc_summary(curves,rows)
    frozen_rows=[]; descriptive_rows=[]; bootstrap=[]; regression=[]
    datasets=np.array([r['dataset'] for r in rows])
    truth=np.array([r['t_t']-r['t_n'] for r in rows])
    for m in METHODS:
        pred=np.array([scores[m][sid] for sid in ids])
        for ds in ('overall','gsm8k','math','aime'):
            mask=np.ones(len(rows),dtype=bool) if ds=='overall' else (datasets==ds)
            diag=p2.regression_metrics(pred[mask],truth[mask])
            regression.append({'method':m,'dataset':ds,'n':int(mask.sum()), **diag})
    for j,budget in enumerate(cfg['budgets']):
        old={'frozen':{},'descriptive':{}}
        frozen_policy={}; descr_policy={}
        for m in METHODS:
            vals=np.array([scores[m][sid] for sid in ids])
            spec=pol['methods'][m][j]
            if spec['requested_validation_budget']!=budget:
                raise ValueError('Frozen validation budget differs from locked config')
            pn=frozen_pn(vals,spec)
            gm=p2.group_metrics(rows,pn)
            frozen_policy[m]=pn
            for ds,g in gm.items():
                frozen_rows.append({'validation_budget_target':budget,'method':m,'dataset':ds,
                    'test_accuracy_pct':100*g['accuracy'],'test_actual_tokens':g['actual_tokens'],
                    'test_nt_fraction':g['nothinking_fraction'],
                    'test_budget_error':g['actual_tokens']-budget if ds=='overall' else ''})
            bracket=linear_bracket(curves[m],budget)
            pn=p2.nt_prob_at_bracket(rows,scores[m],bracket)
            gm=p2.group_metrics(rows,pn)
            descr_policy[m]=pn
            rnd=random_at_budget(rows,budget)
            for ds,g in gm.items():
                descriptive_rows.append({'test_actual_token_budget':budget,'method':m,'dataset':ds,
                    'test_accuracy_pct':100*g['accuracy'],'test_nt_fraction':g['nothinking_fraction'],
                    'uplift_vs_global_random_pp':100*(g['accuracy']-rnd['accuracy']) if ds=='overall' else ''})
        for method_a,method_b in ((METHODS[1],METHODS[0]),(METHODS[2],METHODS[0])):
            for kind,policies in (('frozen_validation_threshold',frozen_policy),('descriptive_test_cost_match',descr_policy)):
                ci=p2.paired_bootstrap(rows,policies[method_a],policies[method_b],args.bootstrap_reps,42+int(budget))
                if ci is not None:
                    bootstrap.append({'budget':budget,'protocol':kind,'comparison':method_a+' - '+method_b,
                          'difference_pp':ci['difference_pp'],'ci_95_low_pp':ci['ci_95_pp'][0],
                          'ci_95_high_pp':ci['ci_95_pp'][1], 'n_bootstrap':ci['repetitions']})
        print('COST_DIFF_TEST_BUDGET '+str(budget)+' '+ ' '.join(
            f'{m}:frozen={100*p2.group_metrics(rows,frozen_policy[m])["overall"]["accuracy"]:.2f}%@{p2.group_metrics(rows,frozen_policy[m])["overall"]["actual_tokens"]:.0f}tok descriptive={100*p2.group_metrics(rows,descr_policy[m])["overall"]["accuracy"]:.2f}%'
            for m in METHODS),flush=True)
