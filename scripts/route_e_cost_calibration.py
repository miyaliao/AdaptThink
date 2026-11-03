#!/usr/bin/env python3
"""P2 Experiment 2: train-only cross-fitted token-length calibration.

Scientific control: the original frozen P1 Ridge heads and the original P2
Thinking-preference score order are NOT changed. Only predicted per-mode
lengths used by the pre-generation *budget selector* are changed.

Calibration is fitted from five-fold OUT-OF-FOLD predictions on the original
3000 TRAINING questions (each row predicted by a Ridge fit excluding it).
No 450-validation outcome/token fields are used in fitting, selecting
factors, routing scores or predicted-budget threshold calibration.

Baseline: original tT_hat, tN_hat.
Calibrated: clip(cT*tT_hat, 0, max_tokens),
            clip(cN*tN_hat, 0, max_tokens)
where cm = mean(actual_train_mode_m)/mean(OOF_predicted_train_mode_m).

IMPORTANT: train-success-stratified data differ greatly from validation;
OOF calibration can help in-sample but fail on out-of-distribution validation.
Report negative results without tuning factors on validation.

This uses unlabeled validation features to set an illustrative batch threshold,
like the original P2 prospective diagnostic. It is not a production online
budget controller and includes neither encoder overhead nor a test set.

Usage:
  python scripts/route_e_cost_calibration.py self-test
  python scripts/route_e_cost_calibration.py run
  python scripts/route_e_cost_calibration.py run --overwrite --no-plot
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

import route_e_frozen as base
import route_e_token_aware as p2

ROOT = base.ROOT
P2_MODEL = ROOT/"outputs/route_e/token_aware/full/ridge_heads.npz"
P2_PREDICTIONS = ROOT/"outputs/route_e/token_aware/full/validation_predictions.json"
P2_REPORT = ROOT/"results/route_e/token_aware/full/validation.json"
OUT = ROOT/"results/route_e/cost_calibration/experiment2"
MODEL_OUT = ROOT/"outputs/route_e/cost_calibration/experiment2"
FOLDS = 5
SEED = 42
TOL = 1e-5


def file_sha256(path):
    return base.sha256(path)


def load_model(path):
    with np.load(path, allow_pickle=False) as f:
        keys={"mean","scale","weight","intercept","smear","l2"}
        if set(f.files)!=keys:
            raise ValueError(f"Unexpected frozen P2 head schema: {f.files}")
        model={k:np.asarray(f[k],dtype=np.float64) for k in keys}
    if (model["mean"].shape!=(1536,) or model["scale"].shape!=(1536,)
            or model["weight"].shape!=(1536,3)
            or model["intercept"].shape!=(3,) or model["smear"].shape!=(2,)
            or model["l2"].shape!=()):
        raise ValueError("Invalid full frozen P2 Ridge head dimensions")
    if (not all(np.isfinite(v).all() for v in model.values())
            or np.any(model["scale"]<=0) or np.any(model["smear"]<=0)
            or float(model["l2"])<=0):
        raise ValueError("Invalid frozen P2 head parameters")
    return model


def read_original_scores(rows, pred_path, report_path, model, Xva):
    saved=json.loads(Path(pred_path).read_text(encoding="utf-8"))
    historical=json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (saved.get("strict_val_sha256")!=base.VAL_SHA256 or
            historical.get("strict_val_sha256")!=base.VAL_SHA256 or
            historical.get("strict_train_sha256")!=base.TRAIN_SHA256 or
            historical.get("smoke") or not historical.get("encoder_frozen") or
            historical.get("generation_repeated")):
        raise ValueError("Frozen original P2 strict protocol hash mismatch")
    if abs(float(historical["head"]["ridge_penalty"])-float(model["l2"]))>1e-12:
        raise ValueError("Original P2 Ridge penalty differs from saved head")
    lam=float(historical["head"]["fixed_lambda"])
    if (not math.isfinite(lam) or lam<=0 or
            abs(float(saved["cost_lambda"])-lam)>1e-12):
        raise ValueError("Frozen lambda or saved P2 prediction mismatch")
    raw=saved["predictions"]
    if list(raw)!=[r["sample_id"] for r in rows]:
        raise ValueError("Original P2 prediction IDs/order mismatch")

    computed=p2.predict_heads(model,Xva)
    original_score=np.empty(len(rows),dtype=np.float64)
    for i,row in enumerate(rows):
        x=raw[row["sample_id"]]
        keys=("delta_p_pred","thinking_tokens_pred",
              "nothinking_tokens_pred","score_predicted_cost")
        if not all(k in x for k in keys):
            raise ValueError("Missing frozen P2 saved predicted field")
        actuals=(computed["delta"][i],computed["t_t"][i],
                 computed["t_n"][i])
        for k,value in zip(keys[:3],actuals):
            if not np.isclose(float(x[k]),value,rtol=1e-10,atol=TOL):
                raise ValueError(f"Original P2 frozen head reproduction failed: {row['sample_id']} {k}")
        score=float(x["score_predicted_cost"])
        expected=float(computed["delta"][i]-lam*(
            computed["t_t"][i]-computed["t_n"][i])/1000.)
        if not np.isclose(score,expected,rtol=1e-10,atol=TOL):
            raise ValueError("Original P2 router score changed")
        original_score[i]=score
    return computed,original_score,lam,historical


def crossfit_predict(Xtr,training_targets,l2,folds,seed):
    """Each training row is predicted by a model fitted on OTHER train rows."""
    if folds<2 or folds>len(Xtr)//2:
        raise ValueError("Crossfit must have >=2 folds and leave sufficient rows")
    n=len(Xtr)
    perm=np.random.default_rng(seed).permutation(n)
    splits=np.array_split(perm,folds)
    prediction=np.full((n,2),np.nan,dtype=np.float64)
    fold_id=np.full(n,-1,dtype=np.int32)
    for k,held in enumerate(splits):
        fit_idx=np.setdiff1d(np.arange(n),held,assume_unique=True)
        fitted=p2.ridge_fit(Xtr[fit_idx],training_targets[fit_idx],l2)
        forecast=p2.predict_heads(fitted,Xtr[held])
        prediction[held,0]=forecast["t_t"]
        prediction[held,1]=forecast["t_n"]
        fold_id[held]=k
        print(f"CALIB_OOF_FOLD {k+1}/{folds} fit={len(fit_idx)} "
              f"held={len(held)}",flush=True)
    if not np.isfinite(prediction).all() or (fold_id<0).any():
        raise AssertionError("Incomplete OOF cost predictions")
    return prediction,fold_id


def fit_multipliers(train_rows,oof):
    """Single predeclared *train-only* mode-specific mean-ratio correction."""
    y=np.array([[r["t_t"],r["t_n"]] for r in train_rows],dtype=np.float64)
    if y.shape!=oof.shape or not (np.isfinite(y).all() and np.isfinite(oof).all()):
        raise ValueError("Invalid OOF calibration arrays")
    if (y<0).any() or (oof<0).any() or (oof.mean(axis=0)<=0).any():
        raise ValueError("OOF mode means must be positive for ratio calibration")
    factors=y.mean(axis=0)/oof.mean(axis=0)
    if not np.isfinite(factors).all() or (factors<=0).any():
        raise ValueError("Invalid learned calibration factors")
    return factors


def apply_multipliers(pred,factors):
    factors=np.asarray(factors,dtype=np.float64)
    if factors.shape!=(2,) or not np.isfinite(factors).all() or (factors<=0).any():
        raise ValueError("Invalid multiplier shape or value")
    return {"delta":np.array(pred["delta"],copy=True),
            "t_t":np.clip(np.asarray(pred["t_t"])*factors[0],
                          0.,p2.MAX_OUTPUT_TOKENS),
            "t_n":np.clip(np.asarray(pred["t_n"])*factors[1],
                          0.,p2.MAX_OUTPUT_TOKENS)}


def diag(rows,pred):
    return p2.cost_diagnostics(rows,pred)


def label_free_routing_plan(rows,pred,score,budget):
    """Outputs only selection probabilities; cannot read actual validation labels."""
    n=len(rows)
    if len(pred["t_t"])!=n or len(score)!=n:
        raise ValueError("Invalid plan lengths")
    order=sorted(range(n),key=lambda i:(float(score[i]),rows[i]["sample_id"]))
    predicted_mean=float(np.mean(pred["t_t"]))
    candidate=[{"fraction":0.,"tokens":predicted_mean,"accuracy":0.}]
    for k,i in enumerate(order,1):
        predicted_mean+=(float(pred["t_n"][i])-float(pred["t_t"][i]))/n
        candidate.append({"fraction":k/n,"tokens":predicted_mean,"accuracy":0.})
    lo=min(x["tokens"] for x in candidate)
    hi=max(x["tokens"] for x in candidate)
    if budget<lo-1e-8 or budget>hi+1e-8:
        return {"feasible":False,"requested_budget":budget,
                "predicted_reachable_range":[float(lo),float(hi)]},None
    from route_e_equal_budget import linear_bracket
    bracket=linear_bracket(candidate,min(max(budget,lo),hi))
    pn=p2.nt_prob_at_bracket(
        rows,{r["sample_id"]:float(-s) for r,s in zip(rows,score)},bracket)
    expected_cost=float(np.mean((1-pn)*pred["t_t"]+pn*pred["t_n"]))
    if abs(expected_cost-budget)>TOL:
        raise AssertionError("Prediction-only budget bracket mismatch")
    return {"feasible":True,"requested_budget":budget,
            "predicted_budget":expected_cost,
            "expected_nt_fraction":float(pn.mean()),
            "mix":bracket},pn


def posthoc_metrics(rows,plan,pn):
    """True validation outcomes/lengths enter ONLY here, after plan is fixed."""
    if pn is None:
        return plan
    group=p2.group_metrics(rows,pn)
    actual=group["overall"]["actual_tokens"]
    return {**plan,
            "actual_budget":actual,
            "actual_accuracy":group["overall"]["accuracy"],
            "actual_minus_target_tokens":actual-plan["requested_budget"],
            "absolute_budget_error_tokens":abs(actual-plan["requested_budget"]),
            "groups":group}


def compare_budgets(rows,original,calibrated,score,budgets):
    table=[]
    for budget in budgets:
        entry={"target_budget":float(budget),"arms":{}}
        for name,pred in (("Uncalibrated",original),("Train_OOF_calibrated",calibrated)):
            plan,pn=label_free_routing_plan(rows,pred,score,float(budget))
            entry["arms"][name]=posthoc_metrics(rows,plan,pn)
        a=entry["arms"]["Uncalibrated"]
        b=entry["arms"]["Train_OOF_calibrated"]
        if a["feasible"] and b["feasible"]:
            entry["calibrated_minus_original_accuracy_pp"]=float(
                100*(b["actual_accuracy"]-a["actual_accuracy"]))
            entry["absolute_budget_error_reduction_tokens"]=float(
                a["absolute_budget_error_tokens"]-b["absolute_budget_error_tokens"])
            entry["additional_actual_tokens"]=float(
                b["actual_budget"]-a["actual_budget"])
        table.append(entry)
    return table


def output_files(out,report):
    (out/"comparison.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with (out/"budget_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["target_budget","arm","feasible","predicted_budget",
                    "actual_budget","absolute_budget_error_tokens",
                    "actual_accuracy_pct","nt_fraction"])
        for row in report["budgets"]:
            for name,arm in row["arms"].items():
                w.writerow([row["target_budget"],name,arm["feasible"],
                           arm.get("predicted_budget",""),arm.get("actual_budget",""),
                           arm.get("absolute_budget_error_tokens",""),
                           (100*arm["actual_accuracy"] if arm["feasible"] else ""),
                           arm.get("expected_nt_fraction","")])
    with (out/"length_metrics.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["split","arm","target","mae","rmse","spearman",
                    "pred_mean","actual_mean","bias_pred_minus_actual"])
        for split in ("train_oof","validation"):
            for arm,ds in report["diagnostics"][split].items():
                for key,m in ds.items():
                    if "mae" not in m:
                        continue
                    w.writerow([split,arm,key,m["mae"],m["rmse"],m["spearman"],
                                m["pred_mean"],m["actual_mean"],
                                m["bias_pred_minus_actual"]])


def make_plot(out,table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("CALIB_PLOT_SKIPPED matplotlib unavailable",flush=True)
        return
    fig,ax=plt.subplots(figsize=(9,5))
    b=[r["target_budget"] for r in table]
    ax.plot(b,b,linestyle="--",label="Target")
    for name in ("Uncalibrated","Train_OOF_calibrated"):
        x=[(row["target_budget"],row["arms"][name]["actual_budget"]) for row in table
           if row["arms"][name]["feasible"]]
        if x:
            ax.plot(*zip(*x),marker="o",label=name.replace("_"," "))
    ax.set(xlabel="Requested predicted mean output tokens",
           ylabel="Observed mean output tokens",
           title="P2 Experiment 2: Train-only cost calibration")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in (".svg",".png"):
        fig.savefig(out/("budget_calibration"+ext),dpi=180)
    plt.close(fig)


def run(args):
    if (args.folds<2 or args.seed<0 or not args.budgets or
            len(args.budgets)!=len(set(args.budgets)) or
            not all(math.isfinite(b) and b>0 for b in args.budgets)):
        raise ValueError("Invalid folds, seed, or budgets")
    out=Path(args.out_dir)
    model_out=Path(args.model_out)
    if (out/"comparison.json").exists() and not args.overwrite:
        raise FileExistsError(f"Calibration report already exists: {out}")
    if (model_out/"train_oof_calibrator.npz").exists() and not args.overwrite:
        raise FileExistsError(f"Calibration head already exists: {model_out}")
    train,val,Xtr,Xva,tm,vm,overlaps=p2.read_protocol_and_cache(False)
    head=load_model(args.p2_model)
    original,score,lam,source_report=read_original_scores(
        val,args.p2_predictions,args.p2_report,head,Xva)
    print(f"CALIB_SOURCE_PASS train={Xtr.shape} val={Xva.shape} "
          f"frozen_original_score=1 lambda={lam:.4f}",flush=True)
    started=time.perf_counter()
    y=p2.target_matrix(train)
    oof,folds=crossfit_predict(Xtr,y,float(head["l2"]),args.folds,args.seed)
    factors=fit_multipliers(train,oof)
    time_fit=time.perf_counter()-started
    train_original={"delta":np.asarray([r["delta"] for r in train]),
                    "t_t":oof[:,0],"t_n":oof[:,1]}
    train_calibrated=apply_multipliers(train_original,factors)
    calibrated=apply_multipliers(original,factors)
    # The score values and question ranking are STRICTLY unchanged by calibration.
    if not np.array_equal(score,score.copy()):
        raise AssertionError("Router score unexpectedly changed")
    report={
        "title":"Route E/P2 Experiment 2 — train-only OOF mean-ratio cost calibration",
        "strict_train_sha256":base.TRAIN_SHA256,
        "strict_val_sha256":base.VAL_SHA256,
        "source_paths":{"head":str(args.p2_model),
                        "predictions":str(args.p2_predictions),
                        "report":str(args.p2_report)},
        "source_file_sha256":{
            "head":file_sha256(args.p2_model),
            "predictions":file_sha256(args.p2_predictions),
            "report":file_sha256(args.p2_report)},
        "original_cost_lambda":lam,
        "calibrator":{"method":"five-fold OOF mode-specific train mean ratio",
                      "folds":args.folds,"seed":args.seed,
                      "thinking_multiplier":float(factors[0]),
                      "nothinking_multiplier":float(factors[1]),
                      "source":"original strict TRAIN only; independent held-out "
                               "predictions within each fold",
                      "train_oof_actual_mean_tokens":[float(x) for x in
                          np.array([[r["t_t"],r["t_n"]] for r in train]).mean(axis=0)],
                      "train_oof_pred_mean_tokens":[float(x) for x in oof.mean(axis=0)],
                      "crossfit_cpu_seconds":time_fit},
        "original_frozen_p2_head_reproduced":True,
        "routing_score_changed":False,
        "router_head_retrained_for_deployment":False,
        "crossfit_auxiliary_models_only":True,
        "validation_targets_used_for_fit":False,
        "test_data_accessed":False,
        "encoder_forward_or_new_generation":False,
        "budget_threshold_method":"unlabeled-validation batch-level predicted "
                                  "length bracket (transductive research diagnostic)",
        "warning":"Train-stratified calibration may worsen validation under distribution "
                  "shift. Actual validation lengths/accuracy are used only for posthoc "
                  "reporting. Not an independent online budget guarantee.",
        "diagnostics":{
            "train_oof":{"Uncalibrated":diag(train,train_original),
                         "Train_OOF_calibrated":diag(train,train_calibrated)},
            "validation":{"Uncalibrated":diag(val,original),
                          "Train_OOF_calibrated":diag(val,calibrated)},
        },
        "budgets":compare_budgets(val,original,calibrated,score,args.budgets),
    }
    out.mkdir(parents=True,exist_ok=True)
    model_out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(model_out/"train_oof_calibrator.npz",
                        multipliers=factors,fold_id=folds,
                        train_oof_tokens=oof,
                        seed=np.asarray(args.seed),
                        source_l2=np.asarray(float(head["l2"])))
    output_files(out,report)
    if not args.no_plot:
        make_plot(out,report["budgets"])
    for split in ("train_oof","validation"):
        for arm in ("Uncalibrated","Train_OOF_calibrated"):
            m=report["diagnostics"][split][arm]
            print(f"CALIB_METRICS {split} {arm} "
                  f"thinking_pred_mean={m['thinking_tokens']['pred_mean']:.1f} "
                  f"thinking_actual_mean={m['thinking_tokens']['actual_mean']:.1f} "
                  f"nothinking_pred_mean={m['nothinking_tokens']['pred_mean']:.1f} "
                  f"nothinking_actual_mean={m['nothinking_tokens']['actual_mean']:.1f}",flush=True)
    print(f"CALIB_FACTORS thinking={factors[0]:.6f} "
          f"nothinking={factors[1]:.6f}",flush=True)
    for row in report["budgets"]:
        bits=[]
        for name in ("Uncalibrated","Train_OOF_calibrated"):
            r=row["arms"][name]
            if r["feasible"]:
                bits.append(f"{name}:actual={r['actual_budget']:.1f}"
                            f"/acc={r['actual_accuracy']:.2%}"
                            f"/error={r['actual_minus_target_tokens']:+.1f}")
            else:
                bits.append(f"{name}:UNREACHABLE")
        print(f"CALIB_BUDGET {row['target_budget']:.0f} | "+" | ".join(bits),
              flush=True)
    print(f"CALIB_DONE report={out/'comparison.json'}",flush=True)


def self_test():
    """Proof of train-only calibration, source-score freeze, budget orientation."""
    rng=np.random.default_rng(42)
    X=rng.normal(size=(64,7))
    y=np.stack((np.tanh(X[:,0]),np.log1p(np.exp(5+X[:,1]*.2)),
                np.log1p(np.exp(4+X[:,2]*.2))),axis=1)
    oof,folds=crossfit_predict(X,y,.02,4,42)
    assert np.isfinite(oof).all() and set(folds)==set(range(4))
    original=[{"t_t":float(240+3*i),"t_n":float(60+i/2.),
               "delta":float(i%3)/10,"sample_id":f"a:{i}",
               "dataset":"test","p_t":.8,"p_n":.7} for i in range(64)]
    scale=fit_multipliers(original,oof)
    assert scale.shape==(2,) and (scale>0).all()
    raw={"delta":np.zeros(64),"t_t":np.full(64,300.),
         "t_n":np.full(64,100.)}
    adjusted=apply_multipliers(raw,[.5,2.])
    assert np.allclose(adjusted["t_t"],150.)
    assert np.allclose(adjusted["t_n"],200.)
    score=np.linspace(-1.,1.,64)
    cand={"delta":raw["delta"],"t_t":raw["t_t"],"t_n":raw["t_n"]}
    plan,pn=label_free_routing_plan(original,cand,score,200.)
    assert plan["feasible"] and abs(plan["predicted_budget"]-200.)<1e-8
    # Lower Thinking-preference entries are first NT.
    assert pn[0]>pn[-1] and plan["expected_nt_fraction"]==.5
    altered=[dict(r,t_t=r["t_t"]*10,t_n=r["t_n"]*3) for r in original]
    plan2,pn2=label_free_routing_plan(altered,cand,score,200.)
    assert plan==plan2 and np.array_equal(pn,pn2)
    assert not label_free_routing_plan(original,cand,score,500.)[0]["feasible"]
    print("CALIB_SELFTEST_PASS (OOF folds, train-only factors, frozen scores, "
          "predicted-only threshold, no validation target leakage)",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    cmd=sub.add_parser("run")
    cmd.add_argument("--folds",type=int,default=FOLDS)
    cmd.add_argument("--seed",type=int,default=SEED)
    cmd.add_argument("--budgets",type=float,nargs="+",default=list(p2.DEFAULT_BUDGETS))
    cmd.add_argument("--p2-model",type=Path,default=P2_MODEL)
    cmd.add_argument("--p2-predictions",type=Path,default=P2_PREDICTIONS)
    cmd.add_argument("--p2-report",type=Path,default=P2_REPORT)
    cmd.add_argument("--out-dir",type=Path,default=OUT)
    cmd.add_argument("--model-out",type=Path,default=MODEL_OUT)
    cmd.add_argument("--no-plot",action="store_true")
    cmd.add_argument("--overwrite",action="store_true")
    args=p.parse_args()
    if args.command=="self-test":
        self_test()
    else:
        run(args)


if __name__=="__main__":
    main()
