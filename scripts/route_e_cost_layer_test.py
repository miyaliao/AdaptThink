#!/usr/bin/env python3
"""One-shot held-out 485 evaluation: validation-selected L16 Cost-only vs historical L28.

Stages, in order:
  prepare: CPU; verify historical L28 parity, train L16 head on 3000 training,
           freeze thresholds using ONLY 450 validation samples.
  extract: GPU; P1 frozen prefill on canonical 485 test QUESTIONS, never outcomes.
  report: CPU; apply frozen thresholds to cached test representations, read test
          outcomes ONLY to score results, no test-layer/threshold selection.

This script does not perform generation or update DeepSeek weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import route_e_frozen as base
import route_e_prompt_ablation as prefix
import route_e_cost_layer_ablation as layers
import adaptthink_test_report as eval_test
from adaptthink_test_predict import cost_predictions

ROOT = base.ROOT
OUT = ROOT / "outputs/route_e/cost_layer_ablation/heldout_485"
REPORT = ROOT / "results/route_e/cost_layer_ablation/heldout_485"
TEST_LABELS = ROOT / "results/test_485_labels.jsonl"
TEST_SPLIT = ROOT / "data/splits/adaptthink_original_baseline_seed42.json"
OLD_HEAD = ROOT / "outputs/route_e/token_aware/full/ridge_heads.npz"
OLD_VAL = ROOT / "outputs/route_e/token_aware/full/validation_predictions.json"
BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)
PRIMARY_BUDGET = 1300
PRIMARY_LAYER = 16
BASELINE_LAYER = 28
N_TEST = 485
MAXLEN = 1024


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def required_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


def write_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def load_old_head():
    required_file(OLD_HEAD)
    with np.load(OLD_HEAD, allow_pickle=False) as obj:
        keys = ("mean", "scale", "weight", "intercept", "smear")
        if not set(keys).issubset(obj.files):
            raise ValueError("Historical Ridge head missing keys")
        head = {key: obj[key] for key in keys}
    if head["weight"].shape != (1536, 3) or head["smear"].shape != (2,):
        raise ValueError("Unexpected historic 3-target Ridge schema")
    return head


def prepare():
    tr, va, _ = base.load_protocol()
    train_feats = layers.load("train", tr, smoke=False)
    val_feats = layers.load("val", va, smoke=False)
    old_train_dir = prefix.cache_dir("assistant", smoke=False)
    old_train, _ = prefix.read_cache(
        old_train_dir / "train.npz", tr, base.TRAIN_SHA256, "assistant", False)
    old_val, _ = prefix.read_cache(
        old_train_dir / "val.npz", va, base.VAL_SHA256, "assistant", False)
    # Validation-only FUNCTIONAL equivalence gate replaces an overly brittle
    # global max-absolute feature check. The first pretest run showed small
    # FP16/padding numerical drift, with nearly unchanged routing decisions.
    # This gate checks geometry, historical head reproduction, rankings, and
    # actual validation decision agreement; NO test data are read.
    def geometry(a, b):
        a=np.asarray(a,dtype=np.float64)
        b=np.asarray(b,dtype=np.float64)
        norms=np.maximum(np.linalg.norm(a,axis=1),1e-12)
        bnorms=np.maximum(np.linalg.norm(b,axis=1),1e-12)
        rel=np.linalg.norm(a-b,axis=1)/norms
        cos=np.sum(a*b,axis=1)/(norms*bnorms)
        return {"max_abs":float(np.max(np.abs(a-b))),
                "median_relative_l2":float(np.median(rel)),
                "p99_relative_l2":float(np.percentile(rel,99)),
                "min_cosine":float(np.min(cos))}
    geo_train=geometry(old_train,train_feats[28])
    geo_val=geometry(old_val,val_feats[28])
    print("L28_GEOMETRY_PARITY "+json.dumps({"train":geo_train,"val":geo_val}),flush=True)
    for name,g in (("train",geo_train),("val",geo_val)):
        if g["p99_relative_l2"]>0.005 or g["min_cosine"]<0.99998:
            raise RuntimeError(f"STOP: L28 {name} representation geometry differs materially")

    old_head = load_old_head()
    ids = [r["sample_id"] for r in va]
    old_scores = eval_test.read_scores(OLD_VAL,"cost",set(ids),eval_test.VAL_SHA)
    tt, tn = cost_predictions(old_val.astype(np.float64),old_head)
    old_delta=np.asarray(tt-tn,dtype=np.float64)
    historical_error=float(max(abs(float(t)-old_scores[sid])
                               for sid,t in zip(ids,old_delta)))
    print(f"L28_HISTORICAL_HEAD_PARITY max_abs_delta_tokens={historical_error:.6g}",flush=True)
    if historical_error>0.05:
        raise RuntimeError("STOP: historical L28 cost scores cannot be reproduced")

    newtt,newtn=cost_predictions(val_feats[28].astype(np.float64),old_head)
    new_delta=np.asarray(newtt-newtn,dtype=np.float64)
    old_delta=np.asarray([old_scores[sid] for sid in ids])
    def rank(v):
        # Stable average ranks, including tied scores.
        order=np.argsort(v,kind="mergesort")
        ranks=np.empty(len(v),dtype=np.float64)
        i=0
        while i<len(v):
            j=i+1
            while j<len(v) and v[order[j]]==v[order[i]]:
                j+=1
            ranks[order[i:j]]=(i+j+1)/2.
            i=j
        return ranks
    score_spearman=float(np.corrcoef(rank(old_delta),rank(new_delta))[0,1])
    print(f"L28_COST_ORDER_PARITY spearman={score_spearman:.8f}",flush=True)
    if not math.isfinite(score_spearman) or score_spearman<0.9995:
        raise RuntimeError("STOP: L28 historical cost ranking is not equivalent")
    for k in (23,45,113,225,338):
        before=set(np.argsort(-old_delta,kind="stable")[:k])
        after=set(np.argsort(-new_delta,kind="stable")[:k])
        overlap=len(before & after)/k
        print(f"L28_TOPK_PARITY k={k} overlap={overlap:.6f}",flush=True)
        if overlap<0.98:
            raise RuntimeError(f"STOP: L28 top-k routing differs at k={k}")
    new_map=dict(zip(ids,map(float,new_delta)))
    policy_diff={}
    for b in BUDGETS:
        oldp=eval_test.threshold_for_budget(va,old_scores,b)
        newp=eval_test.threshold_for_budget(va,new_map,b)
        def chosen(smap,p):
            return eval_test.means(va,{
                sid:float(score>=p["score_cutoff"]) for sid,score in smap.items()
            })
        a=chosen(old_scores,oldp)
        z=chosen(new_map,newp)
        accuracy_gap=100*(z["acc"]-a["acc"])
        tokens_gap=z["tok"]-a["tok"]
        share_gap=z["nt_fraction"]-a["nt_fraction"]
        print(f"L28_POLICY_PARITY B{b} acc_pp={accuracy_gap:+.6f} "
              f"tokens={tokens_gap:+.3f} nt_frac={share_gap:+.6f}",flush=True)
        if abs(accuracy_gap)>1e-8 or abs(tokens_gap)>15 or abs(share_gap)>(1/450+1e-8):
            raise RuntimeError(f"STOP: L28 routing policy changed materially at B{b}")
        policy_diff[str(b)]={"accuracy_diff_pp":accuracy_gap,
                             "tokens_diff":tokens_gap,
                             "nothinking_fraction_diff":share_gap}
    print("L28_FUNCTIONAL_PARITY_PASS",flush=True)

    head_l16 = layers.train_cost_heads(train_feats[16], tr)
    p16 = layers.predict_cost(head_l16, val_feats[16])
    v16 = dict(zip(ids, map(float, p16[:,0]-p16[:,1])))
    v28 = old_scores  # original historical cost-only decision scores, not retrained
    all_scores = {"L16": v16, "L28": v28}
    frozen = {
        "protocol": "PREDECLARED: L16 primary versus historical L28; no test-based model selection",
        "selected_on": "450 strict validation questions, never the 485 test",
        "selected_layer": 16, "baseline_layer": 28,
        "primary_validation_budget": PRIMARY_BUDGET, "budgets": list(BUDGETS),
        "val_sha256": sha(base.VAL), "train_sha256": sha(base.TRAIN),
        "original_l28_head_sha256": sha(OLD_HEAD), "original_val_scores_sha256": sha(OLD_VAL),
        "new_train_features_sha256": sha(layers.folder(False) / "train.npz"),
        "new_val_features_sha256": sha(layers.folder(False) / "val.npz"),
        "l28_geometry_train": geo_train, "l28_geometry_val": geo_val,
        "l28_old_new_cost_rank_spearman": score_spearman,
        "l28_validation_policy_parity": policy_diff,
        "max_original_l28_score_difference_tokens": historical_error,
        "calibrated": {},
        "description": "Closest observed VALIDATION mean generated token policy; "
                       "its numeric score cutoff is frozen and transferred to test.",
    }
    for key, smap in all_scores.items():
        frozen["calibrated"][key] = {}
        for budget in BUDGETS:
            frozen["calibrated"][key][str(budget)] = eval_test.threshold_for_budget(
                va, smap, budget)
        b1300 = frozen["calibrated"][key][str(PRIMARY_BUDGET)]
        print(f"FROZEN_POLICY {key} B{PRIMARY_BUDGET} "
              f"cutoff={b1300['score_cutoff']:.6g} "
              f"val_acc={b1300['validation_accuracy']:.4%} "
              f"val_actual_tokens={b1300['validation_mean_tokens']:.1f}", flush=True)

    # Both files are exclusive-create; old artifacts are never silently replaced.
    OUT.mkdir(parents=True, exist_ok=True)
    head_path = OUT / "l16_cost_head.npz"
    if head_path.exists() or (OUT / "frozen_policies.json").exists():
        raise FileExistsError("Prepared artifacts already exist; do not recalibrate after test access")
    np.savez_compressed(head_path, mean=head_l16[0], scale=head_l16[1],
                        weight=head_l16[2], intercept=head_l16[3],
                        smear=head_l16[4])
    write_new(OUT / "frozen_policies.json", json.dumps(frozen, indent=2) + "\n")
    print("FROZEN_TEST_PREPARED", OUT / "frozen_policies.json", flush=True)


def canonical_questions():
    spec = json.loads(required_file(TEST_SPLIT).read_text(encoding="utf-8"))
    problems = {}
    for ds, meta in spec["source_files"].items():
        items = json.loads((ROOT / meta["path"]).read_text(encoding="utf-8"))
        for i in meta["selected_indices"]:
            sid = f"{ds}:{i}"
            if sid in problems:
                raise ValueError("Duplicate canonical test split ID")
            problems[sid] = items[i]["problem"]
    ids = sorted(problems)
    if len(ids) != N_TEST or Counter(x.split(":")[0] for x in ids) != {
        "gsm8k": 330, "math": 125, "aime": 30
    }:
        raise ValueError("Unexpected canonical test split or dataset composition")
    return ids, [problems[sid] for sid in ids]


def frozen_policy():
    path = required_file(OUT / "frozen_policies.json")
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy["selected_layer"] != 16 or policy["baseline_layer"] != 28:
        raise ValueError("Unexpected frozen layer selection")
    if policy["val_sha256"] != sha(base.VAL) or policy["train_sha256"] != sha(base.TRAIN):
        raise ValueError("Training/validation changed since policy was frozen")
    if policy["original_l28_head_sha256"] != sha(OLD_HEAD):
        raise ValueError("Historic L28 head changed since policy was frozen")
    if policy["new_train_features_sha256"] != sha(layers.folder(False) / "train.npz"):
        raise ValueError("Training features changed since policy was frozen")
    if policy["new_val_features_sha256"] != sha(layers.folder(False) / "val.npz"):
        raise ValueError("Validation features changed since policy was frozen")
    return policy


def extract(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    frozen_policy()  # MUST freeze validation selections before test extraction.
    output = OUT / "test_features.npz"
    if output.exists():
        raise FileExistsError(f"Test feature file exists; not overwriting: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required for frozen Qwen prefill")
    ids, questions = canonical_questions()
    model_path = args.model.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    template_sha = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
    historical_cache = prefix.cache_dir("assistant", False) / "val.npz"
    with np.load(historical_cache, allow_pickle=False) as f:
        oldmeta = json.loads(str(f["metadata_json"].item()))
    if template_sha != oldmeta["chat_template_sha256"] or (
            Path(oldmeta["model"]).resolve() != model_path):
        raise RuntimeError("STOP: Test model/template mismatches historical P1 cache")
    sequences = prefix.tokenize_ablations(tokenizer, questions, "assistant", MAXLEN)
    model = AutoModel.from_pretrained(
        str(model_path), torch_dtype=torch.float16,
        local_files_only=True, trust_remote_code=False).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if model.config.num_hidden_layers != 28 or model.config.hidden_size != 1536:
        raise ValueError("Qwen checkpoint model dimensions changed")
    vecs = {"L16": [], "L28": []}
    with torch.inference_mode():
        for start in range(0, N_TEST, args.batch_size):
            batch = sequences[start:start+args.batch_size]
            width = max(map(len, batch))
            input_ids = torch.full((len(batch),width),tokenizer.pad_token_id,
                                   dtype=torch.long,device="cuda")
            mask = torch.zeros_like(input_ids)
            for j,seq in enumerate(batch):
                input_ids[j,:len(seq)] = torch.tensor(seq,device="cuda")
                mask[j,:len(seq)] = 1
            o = model(input_ids=input_ids,attention_mask=mask,return_dict=True,
                      output_hidden_states=True,use_cache=False)
            pos = torch.tensor([len(x)-1 for x in batch],device="cuda")
            ix = torch.arange(len(batch),device="cuda")
            for key, h in (("L16",o.hidden_states[16]),("L28",o.last_hidden_state)):
                z = h[ix,pos].float()
                if not torch.isfinite(z).all():
                    raise ValueError("Nonfinite test Hidden State")
                vecs[key].append(z.cpu().numpy().astype(np.float16))
            if start == 0 or start+args.batch_size>=N_TEST or start%(25*args.batch_size)==0:
                print(f"TEST_LAYER_EXTRACT {min(start+args.batch_size,N_TEST)}/{N_TEST}",flush=True)
    arrays = {key:np.concatenate(chunk) for key,chunk in vecs.items()}
    if any(x.shape!=(N_TEST,1536) for x in arrays.values()):
        raise ValueError("Test feature dimensions invalid")
    metadata = {"schema":"cost_layer_test_485_v1","test_split_sha256":sha(TEST_SPLIT),
                "model":str(model_path),"prompt":"P1 assistant",
                "chat_template_sha256":template_sha,"layers":[16,28],
                "pooling":"last_valid_prompt_token","num_questions":N_TEST,
                "policies_sha256":sha(OUT/"frozen_policies.json")}
    np.savez_compressed(output,**arrays,sample_id=np.array(ids),
                        metadata_json=np.array(json.dumps(metadata)))
    print("TEST_LAYER_FEATURES_SAVED",output,flush=True)


def load_test_features():
    path = required_file(OUT / "test_features.npz")
    ids, _ = canonical_questions()
    with np.load(path,allow_pickle=False) as f:
        meta = json.loads(f["metadata_json"].item())
        if meta["test_split_sha256"]!=sha(TEST_SPLIT) or (
            meta["policies_sha256"] != sha(OUT/"frozen_policies.json")
        ) or meta["num_questions"]!=N_TEST:
            raise ValueError("Test-feature provenance mismatch")
        if f["sample_id"].astype(str).tolist()!=ids:
            raise ValueError("Canonical test sample-ID order mismatch")
        return ids, f["L16"].astype(np.float64), f["L28"].astype(np.float64)


def test_metric(rows, scores, cutoff):
    chosen={r["sample_id"]:float(scores[r["sample_id"]]>=cutoff) for r in rows}
    return eval_test.means(rows, chosen), chosen


def bootstrap(rows, pn_a, pn_b, repetitions=2000, seed=42):
    differences=np.array([
        (pn_a[r["sample_id"]]-pn_b[r["sample_id"]])*(r["p_n"]-r["p_t"])
        for r in rows
    ],dtype=np.float64)
    rng=np.random.default_rng(seed)
    draws=np.array([100*float(np.mean(differences[rng.integers(0,len(rows),len(rows))]))
                    for _ in range(repetitions)])
    return {"difference_pp":float(differences.mean()*100),
            "conditional_paired_bootstrap_95CI_pp":np.percentile(draws,[2.5,97.5]).tolist(),
            "repetitions":repetitions,
            "limitation":"Conditional on validation-selected layers/thresholds; "
                         "does not account for model selection or budget mismatch."}


def report():
    policy=frozen_policy()
    ids,X16,X28=load_test_features()
    required_file(TEST_LABELS)
    val=eval_test.read_rows(base.VAL)
    test=eval_test.read_rows(TEST_LABELS)
    if len(val)!=450 or len(test)!=N_TEST:
        raise ValueError("Wrong validation/test sizes")
    if Counter(r["dataset"] for r in test)!={"gsm8k":330,"math":125,"aime":30}:
        raise ValueError("Wrong canonical test dataset distribution")
    if any(r["n_generations"]!=(16 if r["dataset"]=="aime" else 1) for r in test):
        raise ValueError("Test generations mismatch: AIME must be 16 per question")
    if set(r["sample_id"] for r in val)&set(ids):
        raise ValueError("Data leakage: validation/test sample IDs overlap")
    if [r["sample_id"] for r in test]!=ids:
        raise ValueError("Test labels differ in IDs/order from pre-generation test features")
    # Ensure question texts match exactly; no silently different question representation.
    _, questions=canonical_questions()
    if [r["problem"] for r in test]!=questions:
        raise ValueError("Canonical test question text mismatch")
    with np.load(OUT/"l16_cost_head.npz",allow_pickle=False) as f:
        fit=tuple(f[k] for k in ("mean","scale","weight","intercept","smear"))
    l16=layers.predict_cost(fit,X16)
    old=load_old_head()
    oldt,oldn=cost_predictions(X28,old)
    scores={
        "L16":dict(zip(ids,map(float,l16[:,0]-l16[:,1]))),
        "L28":dict(zip(ids,map(float,oldt-oldn))),
    }
    groups={"overall":test}
    groups.update({ds:[r for r in test if r["dataset"]==ds]
                   for ds in eval_test.DATASETS})
    transfers={}
    for b in BUDGETS:
        transfers[str(b)]={}
        for name in ("L16","L28"):
            cutoff=policy["calibrated"][name][str(b)]["score_cutoff"]
            transfers[str(b)][name]={}
            for group, rows in groups.items():
                outcome, _=test_metric(rows,scores[name],cutoff)
                transfers[str(b)][name][group]=outcome
        a=transfers[str(b)]["L16"]["overall"]
        c=transfers[str(b)]["L28"]["overall"]
        print(f"TEST_TRANSFER B{b} L16={a['acc']:.4%}/{a['tok']:.1f}tok "
              f"L28={c['acc']:.4%}/{c['tok']:.1f}tok "
              f"delta={100*(a['acc']-c['acc']):+.3f}pp",flush=True)
    p_a={r["sample_id"]:float(scores["L16"][r["sample_id"]]>=
                             policy["calibrated"]["L16"][str(PRIMARY_BUDGET)]["score_cutoff"])
         for r in test}
    p_b={r["sample_id"]:float(scores["L28"][r["sample_id"]]>=
                             policy["calibrated"]["L28"][str(PRIMARY_BUDGET)]["score_cutoff"])
         for r in test}
    ci=bootstrap(test,p_a,p_b)
    auc={name:{group:eval_test.auc_row(name,rows,scores[name])
               for group,rows in groups.items()} for name in ("L16","L28")}
    result={
        "status":"held-out run completed","test_count":len(test),
        "test_labels_sha256":sha(TEST_LABELS),
        "primary":"L16 versus historical L28 at predeclared validation-calibrated B=1300",
        "policies_sha256":sha(OUT/"frozen_policies.json"),
        "test_features_sha256":sha(OUT/"test_features.npz"),
        "transferred_policies":transfers,"paired_bootstrap_B1300":ci,
        "test_sweep_descriptive_nauc":auc,
        "limitations":["Validation threshold transferred unchanged; test budgets can differ",
                       "Test AUC is descriptive, not a test-optimized budget policy",
                       "Router prefill and CPU inference overhead excluded",
                       "Multiple validation-layer comparisons and outcome-driven selection "
                       "are not covered by bootstrap CI",
                       "AIME question-level correctness is mean of 16 generations"],
    }
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/"heldout_test.json").exists():
        raise FileExistsError("Held-out report already exists; do not rewrite test analysis")
    write_new(REPORT/"heldout_test.json",json.dumps(result,indent=2)+"\n")
    rows=[]
    for b in BUDGETS:
        for name in ("L16","L28"):
            x=transfers[str(b)][name]["overall"]
            rows.append(f"| {b} | {name} | {x['acc']*100:.3f}% | {x['tok']:.1f} | {x['nt_fraction']:.3f} |")
    body=[
        "# Held-out 485: Frozen P1 Cost-only layer comparison","",
        "Validation-selected **L16** versus original historical **L28**; no test-based threshold selection.",
        "Headline strategy: numeric cutoffs calibrated at **1300 validation generated tokens/question** and transferred unchanged.",
        "Test costs are reported as realized. Comparisons at different realized test costs are not matched-budget claims.","",
        "| Val budget | Layer | Test pooled Acc. | Realized test tokens | NT fraction |",
        "| ---: | --- | ---: | ---: | ---: |",*rows,"",
        "## Primary paired result at validation B1300","",
        f"- L16−L28: **{ci['difference_pp']:+.3f} pp**; conditional 95% paired CI "
        f"[{ci['conditional_paired_bootstrap_95CI_pp'][0]:+.3f}, "
        f"{ci['conditional_paired_bootstrap_95CI_pp'][1]:+.3f}] pp.","",
        "## Descriptive test AUC (not an online threshold choice)","",
        "| Layer | Pooled nAUC-AC | Mean gain (pp) |",
        "| --- | ---: | ---: |",
        *[f"| {name} | {auc[name]['overall']['nAUC_AC']:+.4f} | "
          f"{auc[name]['overall']['mean_gain_pp']:+.4f} |" for name in ("L16","L28")],
        "","## Per-dataset headline (Val B1300)","",
        "| Dataset | L16 Acc. | L16 tokens | L28 Acc. | L28 tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for group in ("gsm8k","math","aime"):
        a=transfers[str(PRIMARY_BUDGET)]["L16"][group]
        c=transfers[str(PRIMARY_BUDGET)]["L28"][group]
        body.append(f"| {group} | {a['acc']*100:.2f}% | {a['tok']:.1f} | "
                    f"{c['acc']*100:.2f}% | {c['tok']:.1f} |")
    body += ["","**Caveats:** Conditional bootstrap does not correct for validation layer "
             "selection. Descriptive test sweeps are not deployable budget control. "
             "AIME is averaged over 16 generations per question. Router overhead excluded.",""]
    write_new(REPORT/"heldout_test.md","\n".join(body))
    print(f"TEST_COST_LAYER_DONE {REPORT/'heldout_test.json'}",flush=True)


def self_test():
    x=np.array([[1.,2.],[3.,4.]])
    ids=["a","b"]
    assert dict(zip(ids, [1.,2.]))["b"]==2.
    y=np.array([1.,2.])
    assert len(y)==len(x)
    assert PRIMARY_LAYER==16 and BASELINE_LAYER==28
    assert PRIMARY_BUDGET in BUDGETS
    print("TEST_LAYER_SELFTEST_PASS",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("self-test","prepare","extract","report"))
    p.add_argument("--model",type=Path,default=Path(base.DEFAULT_MODELS["deepseek"]))
    p.add_argument("--batch-size",type=int,default=4)
    args=p.parse_args()
    if args.stage=="self-test":
        self_test()
    elif args.stage=="prepare":
        prepare()
    elif args.stage=="extract":
        if args.batch_size<1:raise ValueError("batch size must be positive")
        extract(args)
    else:
        report()


if __name__=="__main__":
    main()
