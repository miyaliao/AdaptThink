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
