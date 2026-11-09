#!/usr/bin/env python3
"""CPU-only PRE-TEST diagnostics for new L28 vs historical frozen P1.

Consumes only original strict 3000 train / 450 validation caches and
saved validation Cost-only head. Does NOT open the canonical test set.
"""
import json
from pathlib import Path

import numpy as np

import route_e_frozen as base
import route_e_prompt_ablation as prefix
import route_e_cost_layer_ablation as layers
from adaptthink_test_predict import cost_predictions
from adaptthink_test_report import read_scores, threshold_for_budget, means

ROOT = base.ROOT
HIST_HEAD = ROOT / "outputs/route_e/token_aware/full/ridge_heads.npz"
HIST_SCORE = ROOT / "outputs/route_e/token_aware/full/validation_predictions.json"
OUT = ROOT / "results/route_e/cost_layer_ablation/l28_pretest_parity.json"
BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)


def compare_features(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    err = a - b
    per_row_norm = np.linalg.norm(err, axis=1)
    norms = np.maximum(np.linalg.norm(a, axis=1), 1.e-12)
    cos = np.sum(a*b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1.e-12)
    return {
        "max_abs": float(np.max(np.abs(err))),
        "mean_abs": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "relative_l2_median": float(np.median(per_row_norm/norms)),
        "relative_l2_p99": float(np.percentile(per_row_norm/norms, 99)),
        "cosine_min": float(np.min(cos)),
        "cosine_mean": float(np.mean(cos)),
        "cosine_p01": float(np.percentile(cos, 1)),
        "exact_rows": int(np.sum(np.all(a==b, axis=1))),
        "rows": int(len(a))
    }


def rankdata(v):
    v = np.asarray(v, np.float64)
    order = np.argsort(v,kind="mergesort")
    ranks=np.empty(len(v),np.float64)
    i=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[order[j]]==v[order[i]]:
            j+=1
        ranks[order[i:j]]=(i+j+1)/2.
        i=j
    return ranks


def cost_head(head_path):
    with np.load(head_path, allow_pickle=False) as f:
        return {k: f[k] for k in ("mean","scale","weight","intercept","smear")}


def describe_scores(val, old, new):
    ids=[r["sample_id"] for r in val]
    old=np.asarray(old,np.float64)
    new=np.asarray(new,np.float64)
    vdiff=new-old
    rankcorr=float(np.corrcoef(rankdata(old),rankdata(new))[0,1])
    out = {
      "abs_delta_max_tokens": float(np.max(np.abs(vdiff))),
      "abs_delta_median_tokens": float(np.median(np.abs(vdiff))),
      "abs_delta_p95_tokens": float(np.percentile(np.abs(vdiff),95)),
      "spearman_old_vs_new": rankcorr,
      "pearson_old_vs_new": float(np.corrcoef(old,new)[0,1]),
      "topk_overlap": {},
      "validation_calibrated_policy": {}
    }
    for k in (23, 45, 113, 225, 338):
        o=set(np.argsort(-old,kind="stable")[:k])
        n=set(np.argsort(-new,kind="stable")[:k])
        out["topk_overlap"][str(k)] = {"overlap":len(o&n),"fraction":len(o&n)/k}
    oldmap=dict(zip(ids,map(float,old)))
    newmap=dict(zip(ids,map(float,new)))
    for budget in BUDGETS:
        oldp=threshold_for_budget(val,oldmap,budget)
        newp=threshold_for_budget(val,newmap,budget)
        def actual(scores, cutoff):
            probs={sid: float(s>=cutoff) for sid,s in scores.items()}
            return means(val,probs)
        po=actual(oldmap,oldp["score_cutoff"])
        pn=actual(newmap,newp["score_cutoff"])
        out["validation_calibrated_policy"][str(budget)]={
          "old_accuracy":po["acc"], "new_accuracy":pn["acc"],
          "old_tokens":po["tok"], "new_tokens":pn["tok"],
          "acc_diff_pp":100*(pn["acc"]-po["acc"]),
          "token_diff":pn["tok"]-po["tok"],
          "nt_fraction_diff":pn["nt_fraction"]-po["nt_fraction"],
        }
    return out


def main():
    train,val,_=base.load_protocol()
    new_tr=layers.load("train",train,False)[28]
    new_va=layers.load("val",val,False)[28]
    old_tr,_=prefix.read_cache(prefix.cache_dir("assistant",False)/"train.npz",
                               train,base.TRAIN_SHA256,"assistant",False)
    old_va,_=prefix.read_cache(prefix.cache_dir("assistant",False)/"val.npz",
                               val,base.VAL_SHA256,"assistant",False)
    oldhead=cost_head(HIST_HEAD)
    ids=[r["sample_id"] for r in val]
    cached=read_scores(HIST_SCORE,"cost",set(ids),base.VAL_SHA256)
    cached=np.array([cached[s] for s in ids],np.float64)
    oldt,oldn=cost_predictions(old_va,oldhead)
    newt,newn=cost_predictions(new_va,oldhead)
    oldscore=oldt-oldn
    newscore=newt-newn
    report={
        "status":"CPU-only validation diagnostic; NO test data accessed",
        "old_vs_new_feature_train":compare_features(old_tr,new_tr),
        "old_vs_new_feature_val":compare_features(old_va,new_va),
        "saved_history_vs_old_features_max_delta_tokens":
          float(np.max(np.abs(cached-oldscore))),
        "old_head_on_new_L28_vs_saved_original":describe_scores(
            val,cached,newscore)
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():raise FileExistsError(f"Refusing overwrite {OUT}")
    OUT.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("L28_PARITY_DIAGNOSTIC",json.dumps(report,indent=2),flush=True)
    print("L28_PARITY_DIAGNOSTIC_DONE",OUT,flush=True)


if __name__=="__main__":
    main()
