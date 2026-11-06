#!/usr/bin/env python3
"""Produce held-out test ROUTER SCORES only; never decode LLM answers.

Uses frozen P1 DeepSeek representations, the original P1 logistic probe,
the frozen cost-only Ridge heads, and a full-fine-tuned Utility-Hard ModernBERT.
Original validation scores are checked first to catch wrong heads/checkpoints.

python adaptthink_test_predict.py --stage verify
python adaptthink_test_predict.py --stage deepseek
python adaptthink_test_predict.py --stage modernbert --utility-model PATH
python adaptthink_test_predict.py --stage all --utility-model PATH
"""
import argparse
import gc
import json
import math
import time
from pathlib import Path

import numpy as np
from adaptthink_test_report import read_rows, read_scores, file_sha, curve, VAL_SHA

ROOT=Path.cwd()
DEFAULT_P1_HEAD=Path('outputs/route_e/prompt_ablation/assistant/full/linear_probe.npz')
DEFAULT_P1_VAL_FEATURES=Path('outputs/route_e/prompt_ablation/assistant/full/val.npz')
DEFAULT_COST_HEAD=Path('outputs/route_e/token_aware/full/ridge_heads.npz')
DEFAULT_COST_VAL=Path('outputs/route_e/token_aware/full/validation_predictions.json')
DEFAULT_P1_VAL=Path('outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json')
DEFAULT_UTILITY_VAL=Path('outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/validation_predictions.jsonl')
MAX_OUTPUT_TOKENS=16384


def load_npz(path,required):
    with np.load(path,allow_pickle=False) as f:
        if not set(required).issubset(set(f.files)):
            raise ValueError(f'Missing NPZ keys in {path}: {set(required)-set(f.files)}')
        return {k:f[k] for k in f.files}


def p1_probability(X,head):
    values=np.asarray(X,dtype=np.float32)
    logits=((values-head['mean'])/head['scale'])@head['weight']+head['bias'].item()
    return 1.0/(1.0+np.exp(-np.clip(logits,-60,60)))


def cost_predictions(X,heads):
    values=np.asarray(X,dtype=np.float64)
    z=(values-heads['mean'])/heads['scale']
    raw=z@heads['weight']+heads['intercept']
    if raw.shape[1]!=3:raise ValueError(f'Wrong cost head output width: {raw.shape}')
    length=np.clip(np.exp(np.clip(raw[:,1:3],0.,np.log1p(MAX_OUTPUT_TOKENS)))*heads['smear'][None,:]-1.,0.,MAX_OUTPUT_TOKENS)
    return length[:,0],length[:,1]


def verify_heads(args, val):
    p1=load_npz(args.p1_head,('weight','bias','mean','scale'))
    cost=load_npz(args.cost_head,('mean','scale','weight','intercept','smear'))
    with np.load(args.p1_val_features,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata_json'].item()))
        ids=f['sample_id'].astype(str).tolist()
        X=f['X'].astype(np.float32)
    if meta.get('schema')!='route_e_prefix_features_v1' or meta.get('prompt_mode')!='assistant':
        raise ValueError('P1 validation cache has wrong prompt mode or schema')
    if meta.get('labels_sha256')!=VAL_SHA or meta.get('layer')!='last_after_final_norm':
        raise ValueError('P1 validation feature cache does not use frozen strict val/latest layer')
    if ids != [r['sample_id'] for r in val] or X.shape!=(450,1536):
        raise ValueError('P1 validation feature cache ID/order/size mismatch')
    expected_p1=read_scores(args.p1_val,'p1',set(ids),VAL_SHA)
    predicted=p1_probability(X,p1)
    p1_error=max(abs(float(x)-expected_p1[sid]) for sid,x in zip(ids,predicted))
    if p1_error>args.verify_tolerance:
        raise ValueError(f'P1 checkpoint mismatch, max error={p1_error:.6g} > {args.verify_tolerance}')
    scores=dict(zip(ids,map(float,predicted)))
    c=curve(val,scores)
    if abs(c['accuracy'][225]*100-76.4444)>.06 or abs(c['tokens'][225]-1427.05)>2.0:
        raise ValueError('Historical P1 50%-NoThinking replication failed')
    cost_source=read_scores(args.cost_val,'cost',set(ids),VAL_SHA)
    tt,tn=cost_predictions(X,cost)
    cost_error=max(abs(float(a-b)-cost_source[sid]) for sid,a,b in zip(ids,tt,tn))
    if cost_error>max(args.verify_tolerance,1e-3):
        raise ValueError(f'Cost-only head mismatch, max error={cost_error:.6g}')
    print(f'VERIFY_FROZEN_HEADS_PASS P1_error={p1_error:.3g} '
          f'Cost_delta_t_error={cost_error:.3g} P1_50pct_acc={c["accuracy"][225]*100:.4f}% '
          f'P1_50pct_tok={c["tokens"][225]:.3f}',flush=True)
    return p1,cost,meta


def make_prompts(tokenizer,rows,max_input):
    sequences=[]
    for row in rows:
        messages=[{'role':'user','content':row['problem']}]
        user=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        generated=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        expected=user+'<｜Assistant｜><think>\n'
        if generated!=expected:raise ValueError(f'Chat-template P1 mismatch for {row["sample_id"]}')
        prompt=user+'<｜Assistant｜>'
        ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
        if not ids or len(ids)>max_input:
            raise ValueError(f'P1 prompt length invalid: {row["sample_id"]} length={len(ids)} max={max_input}')
        sequences.append(ids)
    return sequences


def deepseek(args, test, val, heads, meta):
    from transformers import AutoModel,AutoTokenizer
    import torch
    if Path(meta["model"]).resolve()!=args.deepseek_model.resolve():
        raise ValueError("DeepSeek checkpoint path does not match training P1 feature metadata")
    if args.test_features.exists() and not args.overwrite:
        with np.load(args.test_features,allow_pickle=False) as f:
            ids=f['sample_id'].astype(str).tolist();X=f['X'].astype(np.float32)
            stored=json.loads(str(f['metadata_json'].item()))
        if ids!=[r['sample_id'] for r in test] or X.shape!=(485,1536):
            raise ValueError('Reused P1 test feature ID/order/size mismatch')
        if stored.get('test_labels_sha256')!=file_sha(args.test_labels):
            raise ValueError('Reused P1 features generated on wrong test-label file')
        if stored.get('chat_template_sha256')!=meta['chat_template_sha256']:
            raise ValueError('Reused P1 features from a different tokenizer template')
        if Path(stored.get('model','')).resolve()!=args.deepseek_model.resolve():
            raise ValueError('Reused P1 test features from a different DeepSeek model path')
        print(f'REUSE_TEST_P1_FEATURES {args.test_features}',flush=True)
    else:
        if not torch.cuda.is_available():raise RuntimeError('One CUDA GPU required for DeepSeek prefill')
        if not args.deepseek_model.is_dir():raise FileNotFoundError(args.deepseek_model)
        tokenizer=AutoTokenizer.from_pretrained(str(args.deepseek_model),local_files_only=True,trust_remote_code=False)
        tokenizer.padding_side='right'
        if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
        import hashlib
        template_sha=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        if template_sha!=meta['chat_template_sha256']:
            raise ValueError('Test DeepSeek tokenizer/template differs from the training P1 cache')
        sequences=make_prompts(tokenizer,test,args.deepseek_max_input)
        print(f'DEEPSEEK_PREFLIGHT_PASS n={len(sequences)} max_input_tokens={max(map(len,sequences))}',flush=True)
        model=AutoModel.from_pretrained(str(args.deepseek_model),torch_dtype=torch.float16,local_files_only=True,trust_remote_code=False).to('cuda').eval()
        if model.config.hidden_size!=1536:raise ValueError('DeepSeek model hidden size mismatch')
        pad_id=tokenizer.pad_token_id
        features=[];start=time.perf_counter()
        with torch.inference_mode():
            for begin in range(0,len(test),args.deepseek_batch):
                batch=sequences[begin:begin+args.deepseek_batch];length=max(map(len,batch))
                ids=torch.full((len(batch),length),pad_id,dtype=torch.long,device='cuda')
                mask=torch.zeros_like(ids)
                for i,x in enumerate(batch):
                    ids[i,:len(x)]=torch.tensor(x,device='cuda');mask[i,:len(x)]=1
                out=model(input_ids=ids,attention_mask=mask,return_dict=True,use_cache=False)
                pos=torch.tensor([len(x)-1 for x in batch],device='cuda')
                picked=out.last_hidden_state[torch.arange(len(batch),device='cuda'),pos]
                features.append(picked.float().cpu().numpy().astype(np.float16))
                if begin==0 or begin+len(batch)==len(test) or begin%(args.deepseek_batch*20)==0:
                    print(f'DEEPSEEK_TEST_EXTRACT {begin+len(batch)}/{len(test)}',flush=True)
        X=np.concatenate(features).astype(np.float32)
        if X.shape!=(485,1536) or not np.isfinite(X).all():raise ValueError('Invalid DeepSeek features')
        args.test_features.parent.mkdir(parents=True,exist_ok=True)
        report={"schema":"route_e_test_features_v1","test_labels_sha256":file_sha(args.test_labels),
                "chat_template_sha256":template_sha,"prompt_mode":"assistant",
                "layer":"last_after_final_norm","pooling":"last_valid_prompt_token",
                "model":str(args.deepseek_model),"seconds":time.perf_counter()-start,
                "max_input_tokens":max(map(len,sequences))}
        np.savez_compressed(args.test_features,X=X.astype(np.float16),
                            sample_id=np.asarray([r['sample_id'] for r in test]),
                            metadata_json=np.asarray(json.dumps(report)))
        del model;gc.collect();torch.cuda.empty_cache()
        print(f'DEEPSEEK_TEST_FEATURES_SAVED {args.test_features}',flush=True)
    scores=p1_probability(X,heads[0]);tt,tn=cost_predictions(X,heads[1])
    ids=[r['sample_id'] for r in test]; sha=file_sha(args.test_labels)
    p1={"test_labels_sha256":sha,"method":"Frozen DeepSeek P1 Classification (original frozen head)",
        "p_nt":{sid:float(x) for sid,x in zip(ids,scores)}}
    cost={"test_labels_sha256":sha,"method":"Frozen DeepSeek P1 Cost-only (trained Ridge length heads)",
          "predictions":{sid:{"thinking_tokens_pred":float(a),"nothinking_tokens_pred":float(b),
                              "delta_t_pred":float(a-b)} for sid,a,b in zip(ids,tt,tn)}}
    args.out.mkdir(parents=True,exist_ok=True)
    for dest,obj in ((args.out/'p1_predictions.json',p1),(args.out/'cost_predictions.json',cost)):
        if dest.exists() and not args.overwrite:raise FileExistsError(f'Use --overwrite: {dest}')
        dest.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
        print('TEST_PREDICTIONS_SAVED',dest,flush=True)


def get_utility_p(logits):
    """Support a 1-logit BCE or 2-logit softmax CE binary classifier."""
    logits=np.asarray(logits,dtype=np.float64)
    if logits.ndim==1:logits=logits[:,None]
    if logits.ndim!=2:raise ValueError('Utility-Hard logits must have shape N x {1,2}')
    if logits.shape[1]==1:
        x=np.clip(logits[:,0],-60,60);return 1/(1+np.exp(-x))
    if logits.shape[1]==2:
        z=logits-logits.max(axis=1,keepdims=True);v=np.exp(z)
        return v[:,1]/v.sum(axis=1)
    raise ValueError(f'Unexpected Utility-Hard number of classes {logits.shape[1]}')


def modernbert(args,test,val):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    if not torch.cuda.is_available():raise RuntimeError('One CUDA GPU required for ModernBERT full FT inference')
    if not args.utility_model.is_dir():raise FileNotFoundError(f'Pass --utility-model to actual BEST checkpoint: {args.utility_model}')
    tokenizer=AutoTokenizer.from_pretrained(str(args.utility_model),local_files_only=True,trust_remote_code=False)
    model=AutoModelForSequenceClassification.from_pretrained(str(args.utility_model),torch_dtype=torch.float16,local_files_only=True,trust_remote_code=False).to('cuda').eval()
    def predict(rows):
        preds=[]
        with torch.inference_mode():
            for start in range(0,len(rows),args.utility_batch):
                batch=rows[start:start+args.utility_batch]
                tokens=tokenizer([r['problem'] for r in batch],truncation=True,
                                 max_length=args.utility_max_input,padding=True,return_tensors='pt')
                out=model(**{k:v.to('cuda') for k,v in tokens.items()})
                logits=out.logits.float().cpu().numpy()
                preds.extend(get_utility_p(logits).tolist())
        return np.asarray(preds,dtype=float)
    val_p=predict(val)
    val_ids=[r['sample_id'] for r in val]
    expected=read_scores(args.utility_val,'modernbert',set(val_ids),VAL_SHA)
    errors=np.asarray([abs(float(a)-expected[sid]) for sid,a in zip(val_ids,val_p)])
    print(f'UTILITY_REPRODUCTION maE={np.mean(errors):.6f} maxE={np.max(errors):.6f}',flush=True)
    if np.max(errors)>args.utility_tolerance:
        raise ValueError('ModernBERT validation scores do not match trained Utility-Hard; '
                         'check best_model path, sigmoid-vs-softmax, tokenizer, and max_length')
    test_p=predict(test)
    if not np.isfinite(test_p).all() or np.any((test_p<0)|(test_p>1)):
        raise ValueError('Invalid test ModernBERT prediction scores')
    dest=args.out/'utility_predictions.jsonl'
    args.out.mkdir(parents=True,exist_ok=True)
    if dest.exists() and not args.overwrite:raise FileExistsError(f'Use --overwrite: {dest}')
    with dest.open('w',encoding='utf-8') as f:
        for row,p in zip(test,test_p):
            f.write(json.dumps({'sample_id':row['sample_id'],'predicted_p_nothinking':float(p)})+'\n')
    del model;gc.collect();torch.cuda.empty_cache()
    print(f'UTILITY_TEST_DONE count={len(test)} {dest}',flush=True)


def self_test():
    X=np.asarray([[1.,2.,3.],[4.,5.,6.]])
    H={'mean':np.zeros(3),'scale':np.ones(3),'weight':np.array([0.,0.,1.]),'bias':np.array(0.)}
    assert abs(p1_probability(X,H)[0]-(1/(1+np.exp(-3))))<1e-8
    C={'mean':np.zeros(3),'scale':np.ones(3),'weight':np.zeros((3,3)),
       'intercept':np.asarray([0.,np.log(1001),np.log(501)]),'smear':np.ones(2)}
    a,b=cost_predictions(X,C)
    assert np.allclose(a,[1000,1000]) and np.allclose(b,[500,500])
    assert np.allclose(get_utility_p([[0.,0.],[0.,3.]]),[.5,1/(1+np.exp(-3))])
    print('THREE_ROUTER_TEST_PREDICT_SELFTEST_PASS')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('self-test','verify','deepseek','modernbert','all'),default='verify')
    p.add_argument('--test-labels',type=Path,default=Path('results/test_485_labels.jsonl'))
    p.add_argument('--val-labels',type=Path,default=Path('results/val_450_labels.jsonl'))
    p.add_argument('--deepseek-model',type=Path,default=Path('model/DeepSeek-R1-Distill-Qwen-1.5B'))
    p.add_argument('--utility-model',type=Path,default=Path('outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/best_model'))
    p.add_argument('--p1-head',type=Path,default=DEFAULT_P1_HEAD)
    p.add_argument('--cost-head',type=Path,default=DEFAULT_COST_HEAD)
    p.add_argument('--p1-val-features',type=Path,default=DEFAULT_P1_VAL_FEATURES)
    p.add_argument('--p1-val',type=Path,default=DEFAULT_P1_VAL)
    p.add_argument('--cost-val',type=Path,default=DEFAULT_COST_VAL)
    p.add_argument('--utility-val',type=Path,default=DEFAULT_UTILITY_VAL)
    p.add_argument('--test-features',type=Path,default=Path('outputs/test_485/deepseek_p1_features.npz'))
    p.add_argument('--out',type=Path,default=Path('outputs/test_485'))
    p.add_argument('--deepseek-batch',type=int,default=8)
    p.add_argument('--utility-batch',type=int,default=16)
    p.add_argument('--deepseek-max-input',type=int,default=1024)
    p.add_argument('--utility-max-input',type=int,default=512)
    p.add_argument('--verify-tolerance',type=float,default=0.001)
    p.add_argument('--utility-tolerance',type=float,default=0.025)
    p.add_argument('--overwrite',action='store_true')
    args=p.parse_args()
    if args.stage=='self-test':self_test();return
    if min(args.deepseek_batch,args.utility_batch,args.deepseek_max_input,args.utility_max_input)<=0:
        p.error('Batch sizes and input limits must be positive')
    val=read_rows(args.val_labels);test=read_rows(args.test_labels)
    if file_sha(args.val_labels)!=VAL_SHA or len(val)!=450:
        raise ValueError('Wrong strict validation labels')
    if len(test)!=485:raise ValueError('Wrong canonical 485-question test labels')
    H=verify_heads(args,val)
    if args.stage in ('deepseek','all'):deepseek(args,test,val,H[:2],H[2])
    if args.stage in ('modernbert','all'):modernbert(args,test,val)

if __name__=='__main__':main()
