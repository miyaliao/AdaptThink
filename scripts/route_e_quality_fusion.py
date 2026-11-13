#!/usr/bin/env python3
"""P2 Experiment 3 — replace Ridge quality score with frozen P1 Logistic score.

No model fitting, new generation, GPU extraction or sealed test reads.
The original frozen P2 cost predictions and lambda remain IDENTICAL.
The original P1 Logistic model is used as a ranking score, NOT as a
calibrated mode-success probability.

The P1 logit is negated so larger means Thinking preference, then its
training-set mean/std are matched to the already-fitted Ridge quality head's
TRAIN prediction mean/std. This is an intentionally fixed, TRAIN-ONLY
score-scale control; it is NOT calibrated probability Delta p.

Methods:
  E Cost_only             : NT priority = +predicted extra Thinking cost
  D Ridge_plus_cost       : NT priority = -(Ridge delta - lambda*cost/1000)
  F Logistic_plus_cost    : NT priority = -(train-scaled P1 logit - lambda*cost/1000)
  P1 Logistic_only        : NT priority = +P(NoThinking class)

Evaluation: strict 450 validation, offline observed-token equal-budget replay,
nominal paired bootstrap, per-dataset outputs, and a distinct predicted-budget
diagnostic (not a deployable online budget guarantee).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import route_e_frozen as base
import route_e_token_aware as p2
import route_e_cost_only as exp1
import route_e_cost_calibration as exp2
from route_e_equal_budget import dense_curve, linear_bracket, load_scores, random_at_budget

ROOT = base.ROOT
PROBE_PATH = ROOT/"outputs/route_e/prompt_ablation/assistant/full/linear_probe.npz"
P1_SCORES_PATH = ROOT/"outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json"
P2_HEAD_PATH = ROOT/"outputs/route_e/token_aware/full/ridge_heads.npz"
P2_PREDICTIONS_PATH = ROOT/"outputs/route_e/token_aware/full/validation_predictions.json"
P2_REPORT_PATH = ROOT/"results/route_e/token_aware/full/validation.json"
OUT = ROOT/"results/route_e/quality_head/experiment3"
METHODS = ("Cost_only", "Ridge_plus_cost", "Logistic_plus_cost",
           "P1_Logistic_only")
TOL = 2e-5


def load_probe(path, width=1536):
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {"weight", "bias", "mean", "scale"}:
            raise ValueError("Unexpected frozen P1 Logistic probe schema")
        fit = {key: np.asarray(source[key], dtype=np.float32)
               for key in source.files}
    if (fit["weight"].shape != (width,) or fit["mean"].shape != (width,)
            or fit["scale"].shape != (width,) or fit["bias"].shape != ()):
        raise ValueError("Wrong frozen P1 Logistic probe dimensions")
    if (not all(np.isfinite(v).all() for v in fit.values())
            or (fit["scale"] <= 0).any()):
        raise ValueError("Invalid P1 Logistic probe")
    return fit


def frozen_logits(X, fit):
    """Raw NoThinking-class logit; never interpreted as p_nothinking_success."""
    return np.asarray(((X-fit["mean"])/fit["scale"]) @ fit["weight"]
                      + fit["bias"], dtype=np.float64)


def train_only_scale(logit_train, logit_validation, ridge_quality_train):
    """Match TRAIN mean/std of Ridge quality predictions (no val outcomes).

    Negative Logistic logit means higher predicted Thinking preference.
    This matches scales for a controlled signal swap; it does not turn
    classification probabilities into empirical success probabilities.
    """
    source = -np.asarray(logit_train, dtype=np.float64)
    query = -np.asarray(logit_validation, dtype=np.float64)
    target = np.asarray(ridge_quality_train, dtype=np.float64)
    if (len(source) != len(target) or len(source) < 2 or
            not all(np.isfinite(v).all() for v in (source,query,target))):
        raise ValueError("Invalid train-only score-scale data")
    sigma_source = float(source.std())
    sigma_target = float(target.std())
    if sigma_source < 1e-10 or sigma_target < 1e-10:
        raise ValueError("Degenerate frozen training scores")
    scale = sigma_target/sigma_source
    offset = float(target.mean()-scale*source.mean())
    scaled_train = offset+scale*source
    scaled_val = offset+scale*query
    if (abs(scaled_train.mean()-target.mean())>1e-10 or
            abs(scaled_train.std()-sigma_target)>1e-10):
        raise AssertionError("Failed train-only moment matching")
    return scaled_val, {
        "scale": float(scale), "offset": offset,
        "train_logistic_thinking_score_mean": float(source.mean()),
        "train_logistic_thinking_score_std": sigma_source,
        "train_ridge_quality_mean": float(target.mean()),
        "train_ridge_quality_std": sigma_target,
        "note": "Train-only affine normalization of negative class LOGIT; "
                "NOT a calibrated expected success gain",
    }


def construct_new_scores(rows, saved, scaled_quality, lam):
    """Returns high-is-NT score maps, from saved pre-generation predictions."""
    if len(scaled_quality) != len(rows):
        raise ValueError("Quality score / validation count mismatch")
    maps = {k: {} for k in ("Cost_only", "Ridge_plus_cost",
                            "Logistic_plus_cost")}
    forecast = {"delta": [], "t_t": [], "t_n": []}
    for i,r in enumerate(rows):
        sid = r["sample_id"]
        v = saved[sid]
        q_old = float(v["delta_p_pred"])
        t_t = float(v["thinking_tokens_pred"])
        t_n = float(v["nothinking_tokens_pred"])
        cost = float(v["delta_t_pred"])
        if not all(math.isfinite(z) for z in (q_old,t_t,t_n,cost,scaled_quality[i])):
            raise ValueError(f"Nonfinite existing P2 scores for {sid}")
        if abs(cost-(t_t-t_n))>1e-8:
            raise ValueError(f"Original predicted cost difference changed at {sid}")
        if abs(q_old-lam*cost/1000.-float(v["score_predicted_cost"]))>1e-8:
            raise ValueError(f"Original P2 D score changed at {sid}")
        maps["Cost_only"][sid]=cost
        maps["Ridge_plus_cost"][sid]=-float(v["score_predicted_cost"])
        maps["Logistic_plus_cost"][sid]=-(float(scaled_quality[i])-lam*cost/1000.)
        forecast["delta"].append(q_old)
        forecast["t_t"].append(t_t)
        forecast["t_n"].append(t_n)
    for key in forecast:
        forecast[key]=np.asarray(forecast[key], dtype=np.float64)
    return maps, forecast


def check_p1_reproduction(rows, fit, Xva, source_path):
    """Reconstruct and verify saved P1 Logistic probabilities and 50%-NT row."""
    logits = frozen_logits(Xva, fit)
    derived = 1./(1.+np.exp(-np.clip(logits,-60.,60.)))
    old = load_scores(source_path, [r["sample_id"] for r in rows])
    ref = np.asarray([old[r["sample_id"]] for r in rows], dtype=np.float64)
    maximum = float(np.max(np.abs(derived-ref)))
    if maximum > TOL:
        raise ValueError(f"Frozen P1 probability reproduction failed: {maximum}")
    curve=dense_curve(rows,old)
    if len(curve)!=451 or abs(curve[225]["accuracy"]-344/450)>1e-8 or (
            abs(curve[225]["tokens"]-1427.05)>2.0):
        raise ValueError("Frozen P1 historical 50% NT reproduction failed")
    return logits,old,{
        "max_abs_p_nt_reproduction_error":maximum,
        "50pct_accuracy":curve[225]["accuracy"],
        "50pct_tokens":curve[225]["tokens"],
    }


def strict_data_and_head(args):
    train, rows, Xtr, Xva, tm, vm, overlaps = p2.read_protocol_and_cache(False)
    original_maps, source, old, lineage = exp1.read_sources(
        rows,args.p2_predictions,args.p2_report)
    exp1.validate_original(rows,original_maps,old,list(args.budgets))
    head=exp2.load_model(args.p2_head)
    q_train=p2.predict_heads(head,Xtr)["delta"]
    preds=p2.predict_heads(head,Xva)
    for i,row in enumerate(rows):
        recorded=source["predictions"][row["sample_id"]]
        for k,v in (("delta_p_pred",preds["delta"][i]),
                    ("thinking_tokens_pred",preds["t_t"][i]),
                    ("nothinking_tokens_pred",preds["t_n"][i])):
            if abs(float(recorded[k])-float(v))>TOL:
                raise ValueError(f"Saved P2 head predictions changed at {row['sample_id']} ({k})")
    probe=load_probe(args.probe)
    val_logit,p1,verification=check_p1_reproduction(rows,probe,Xva,args.p1_scores)
    train_logit=frozen_logits(Xtr,probe)
    quality,scale=train_only_scale(train_logit,val_logit,q_train)
    maps,forecast=construct_new_scores(
        rows,source["predictions"],quality,lineage["frozen_lambda"])
    # Strong controls: E and D must exactly retain Experiment 1 order/scores.
    for old_key,new_key in (("Cost_only","Cost_only"),
                            ("Delta_predicted_cost","Ridge_plus_cost")):
        for sid in original_maps[old_key]:
            if abs(original_maps[old_key][sid]-maps[new_key][sid])>1e-8:
                raise AssertionError(f"Changed frozen control scores: {old_key} at {sid}")
    maps["P1_Logistic_only"]=p1
    return rows,maps,forecast,quality,scale,{
        "train_sha256":base.TRAIN_SHA256,"val_sha256":base.VAL_SHA256,
        "p2_predictions_sha256":base.sha256(args.p2_predictions),
        "p2_report_sha256":base.sha256(args.p2_report),
        "p2_head_sha256":base.sha256(args.p2_head),
        "p1_probe_sha256":base.sha256(args.probe),
        "p1_score_sha256":base.sha256(args.p1_scores),
        "p2_control_reproduced":True,
        "p1_control_reproduced":verification,
        "lambda_frozen":lineage["frozen_lambda"],
        "train_count":len(train),"val_count":len(rows),
        "feature_source_metadata":vm,
        "train_only_normalization":scale,
    }


def one_budget(rows,maps,budget,reps,seed):
    data={"budget":float(budget),"methods":{},"paired_bootstrap":{}}
    pn={}
    rand=random_at_budget(rows,budget)
    for name in METHODS:
        curve=dense_curve(rows,maps[name])
        try:
            bracket=linear_bracket(curve,budget)
        except ValueError as e:
            data["methods"][name]={"available":False,"reason":str(e)}
            continue
        prob=p2.nt_prob_at_bracket(rows,maps[name],bracket)
        groups=p2.group_metrics(rows,prob)
        if abs(groups["overall"]["actual_tokens"]-budget)>1e-5:
            raise AssertionError(f"Cost mismatch in {name}")
        pn[name]=prob
        data["methods"][name]={
            "available":True,"groups":groups,"mix":bracket,
            "uplift_vs_global_random_pp":float(
                100*(groups["overall"]["accuracy"]-rand["accuracy"])),
        }
    data["Global_Random"]={"nt_probability":rand["nt_probability"],
                            "accuracy":rand["accuracy"]}
    for other in ("Ridge_plus_cost","Cost_only","P1_Logistic_only"):
        if "Logistic_plus_cost" in pn and other in pn:
            key=f"Logistic_plus_cost_minus_{other}"
            data["paired_bootstrap"][key]=p2.paired_bootstrap(
                rows,pn["Logistic_plus_cost"],pn[other],reps,seed+int(budget))
    return data


def write_csv(out,report):
    with (out/"equal_budget.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["budget","method","dataset","accuracy_pct","actual_tokens",
                    "nt_fraction","uplift_vs_random_pp"])
        for row in report["dense_equal_budget"]:
            for name,method in row["methods"].items():
                if not method.get("available"):
                    continue
                for ds,g in method["groups"].items():
                    w.writerow([row["budget"],name,ds,100*g["accuracy"],
                                g["actual_tokens"],g["nothinking_fraction"],
                                method["uplift_vs_global_random_pp"] if ds=="overall" else ""])
    with (out/"paired_bootstrap.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["budget","comparison","difference_pp","CI_low_pp","CI_high_pp"])
        for row in report["dense_equal_budget"]:
            for name,ci in row["paired_bootstrap"].items():
                if ci:
                    w.writerow([row["budget"],name,ci["difference_pp"],*ci["ci_95_pp"]])
    with (out/"predicted_budget.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["target_budget","method","feasible","predicted_budget",
                    "actual_budget","actual_accuracy_pct","nt_fraction"])
        for rec in report["predicted_only_budget"]:
            for name,policy in rec["methods"].items():
                w.writerow([rec["target_budget"],name,policy["feasible"],
                            policy.get("predicted_budget",""),policy.get("actual_budget",""),
                            (100*policy["actual_accuracy"] if policy["feasible"] else ""),
                            policy.get("expected_nothinking_fraction","")])


def figure(out,rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("QUALITY_FUSION_PLOT_SKIPPED matplotlib unavailable")
        return
    fig,ax=plt.subplots(figsize=(9,5))
    for method in METHODS:
        points=[(rec["budget"],100*rec["methods"][method]["groups"]["overall"]["accuracy"])
                for rec in rows if rec["methods"][method]["available"]]
        if points:
            ax.plot(*zip(*points),marker="o",label=method.replace("_"," "))
    ax.set(xlabel="Mean generated output tokens (offline observed-cost matching)",
           ylabel="Strict validation accuracy (%)",
           title="Experiment 3: Frozen Logistic quality + unchanged cost")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ("svg","png"):
        fig.savefig(out/f"equal_budget.{extension}",dpi=180)
    plt.close(fig)


def evaluate(args):
    budgets=[float(x) for x in args.budgets]
    if (not budgets or len(budgets)!=len(set(budgets)) or
            not all(math.isfinite(x) and x>0 for x in budgets) or
            args.bootstrap_reps<0):
        raise ValueError("Invalid budgets or Bootstrap reps")
    out=Path(args.out_dir)
    if (out/"comparison.json").exists() and not args.overwrite:
        raise FileExistsError("Experiment 3 report exists; use --overwrite for Experiment 3 only")
    rows,maps,forecast,quality,scale,lineage=strict_data_and_head(args)
    dense=[one_budget(rows,maps,b,args.bootstrap_reps,args.seed) for b in budgets]
    predictions={"strict_val_sha256":base.VAL_SHA256,
                 "note":"PRE-GENERATION scores only; classification score NOT success probability",
                 "scores":{r["sample_id"]:{
                     "scaled_logistic_thinking_score":float(quality[i]),
                     "unchanged_pred_delta_tokens":float(
                         forecast["t_t"][i]-forecast["t_n"][i]),
                     "new_combined_thinking_score":-maps["Logistic_plus_cost"][r["sample_id"]]}
                            for i,r in enumerate(rows)}}
    predicted=[]
    for budget in budgets:
        entry={"target_budget":budget,"methods":{}}
        for name in ("Logistic_plus_cost","Ridge_plus_cost","Cost_only"):
            if name=="Cost_only":
                # Highest cost priority -> smallest Thinking preference
                thinking_score=np.asarray([-maps[name][r["sample_id"]] for r in rows])
            else:
                thinking_score=np.asarray([-maps[name][r["sample_id"]] for r in rows])
            entry["methods"][name]=p2.prospective_budget_policy(rows,forecast,thinking_score,budget)
        predicted.append(entry)
    report={
        "method":"route_e_p2_experiment3_logistic_quality_fusion",
        "lineage":lineage,
        "same_original_frozen_P2_cost_predictions":True,
        "same_original_lambda":True,
        "no_model_retraining":True,
        "validation_true_labels_used_for_score":False,
        "logistic_class_probability_is_not_answer_success_probability":True,
        "quality_scale_source":"TRAIN score distribution only",
        "dense_cost_matching_uses_validation_observed_costs":True,
        "dense_equal_budget":dense,
        "predicted_only_budget":predicted,
        "warning":"Dense cost curves are exploratory and use actual validation "
                  "lengths only AFTER score ranking; predicted-budget selection "
                  "uses unlabeled validation features and is a transductive "
                  "batch diagnostic, not an online deployment guarantee. "
                  "Bootstrap CIs are nominal and not corrected for multiple budgets.",
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"validation_pre_generation_scores.json").write_text(
        json.dumps(predictions,indent=2)+"\n",encoding="utf-8")
    (out/"comparison.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    write_csv(out,report)
    if not args.no_plot:
        figure(out,dense)
    print(f"QUALITY_FUSION_SOURCE_PASS strict_val=450 lambda={lineage['lambda_frozen']:.4f} "
          f"max_p1_score_error={lineage['p1_control_reproduced']['max_abs_p_nt_reproduction_error']:.3g}",
          flush=True)
    print(f"QUALITY_FUSION_TRAIN_SCALE scale={scale['scale']:.6f} "
          f"offset={scale['offset']:.6f} "
          f"ridge_std={scale['train_ridge_quality_std']:.6f}",
          flush=True)
    for row in dense:
        entries=[]
        for m in METHODS:
            v=row["methods"][m]
            if v["available"]:
                entries.append(f"{m}={v['groups']['overall']['accuracy']:.2%}")
            else:
                entries.append(f"{m}=UNREACHABLE")
        ci=row["paired_bootstrap"].get("Logistic_plus_cost_minus_Ridge_plus_cost")
        if ci is not None:
            entries.append(f"new_minus_old={ci['difference_pp']:+.2f}pp")
            entries.append(f"95CI={ci['ci_95_pp']}")
        print(f"QUALITY_FUSION_BUDGET {row['budget']:.0f} | "+" | ".join(entries),flush=True)
    for row in predicted:
        values=[]
        for m,v in row["methods"].items():
            values.append(f"{m}="+
                (f"actual={v['actual_budget']:.1f}/acc={v['actual_accuracy']:.2%}"
                 if v["feasible"] else "UNREACHABLE"))
        print(f"QUALITY_FUSION_PREDICTED_BUDGET {row['target_budget']:.0f} | "+
              " | ".join(values),flush=True)
    print(f"QUALITY_FUSION_DONE report={out/'comparison.json'}",flush=True)


def self_test():
    rng=np.random.default_rng(31)
    X=rng.normal(size=(50,7)).astype(np.float32)
    logit=rng.normal(size=50)
    y=rng.normal(size=50)/10
    val=rng.normal(size=5)
    scaled,meta=train_only_scale(logit,val,y)
    assert abs(meta["train_ridge_quality_std"]-y.std())<1e-12
    assert scaled.shape==(5,)
    changed_y=y.copy()
    # More likely NoThinking => LOWER Thinking preference after sign flip.
    q,_=train_only_scale(np.array([-1.,1.,0.]),np.array([-1.,1.]),
                         np.array([.4,-.4,0.]))
    assert q[0]>q[1]
    rows=[{"sample_id":f"x:{i}","dataset":"x","p_t":1. if i==0 else 0.,
           "p_n":0. if i==0 else 1.,"t_t":100.,"t_n":20.} for i in range(2)]
    saved={}
    for r in rows:
        saved[r["sample_id"]]={
            "delta_p_pred":.2,"thinking_tokens_pred":100.,
            "nothinking_tokens_pred":20.,"delta_t_pred":80.,
            "score_predicted_cost":.2-.05*80/1000.,
        }
    maps,_=construct_new_scores(rows,saved,np.array([.1,.3]),.05)
    assert maps["Logistic_plus_cost"]["x:0"]>maps["Logistic_plus_cost"]["x:1"]
    assert maps["Cost_only"]["x:0"]==maps["Cost_only"]["x:1"]
    check=one_budget(rows,{**maps,"P1_Logistic_only":{"x:0":.5,"x:1":.2}},
                     60.,10,42)
    assert check["methods"]["Logistic_plus_cost"]["available"]
    # Predicted-cost ranking must not depend on observed validation labels.
    for r in rows:
        r["p_t"]=.3
        r["p_n"]=.7
    new_maps,_=construct_new_scores(rows,saved,np.array([.1,.3]),.05)
    assert new_maps==maps
    print("QUALITY_FUSION_SELFTEST_PASS (train-only score matching, orientation, "
          "frozen cost scores, replay, no outcome leakage)",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    run=sub.add_parser("run")
    run.add_argument("--probe",type=Path,default=PROBE_PATH)
    run.add_argument("--p1-scores",type=Path,default=P1_SCORES_PATH)
    run.add_argument("--p2-head",type=Path,default=P2_HEAD_PATH)
    run.add_argument("--p2-predictions",type=Path,default=P2_PREDICTIONS_PATH)
    run.add_argument("--p2-report",type=Path,default=P2_REPORT_PATH)
    run.add_argument("--out-dir",type=Path,default=OUT)
    run.add_argument("--budgets",type=float,nargs="+",default=list(p2.DEFAULT_BUDGETS))
    run.add_argument("--bootstrap-reps",type=int,default=2000)
    run.add_argument("--seed",type=int,default=42)
    run.add_argument("--overwrite",action="store_true")
    run.add_argument("--no-plot",action="store_true")
    args=p.parse_args()
    if args.command=="self-test":
        self_test()
    else:
        evaluate(args)


if __name__=="__main__":
    main()
