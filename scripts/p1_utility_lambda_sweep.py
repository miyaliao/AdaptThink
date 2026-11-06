#!/usr/bin/env python3
"""Frozen DeepSeek P1 Utility lambda sensitivity sweep, CPU only.

Reuse the original strict train-3000/val-450 P1 hidden-state caches.
Train Logistic heads for Utility-Hard / Utility-Soft at a predefined lambda grid,
and compute nAUC-AC from all 451 threshold policies (NOT budget-table points).

    python scripts/p1_utility_lambda_sweep.py self-test
    python scripts/p1_utility_lambda_sweep.py run

No DeepSeek generation, no GPU, no sealed test labels.
Validation-picked lambda is exploratory; confirm only on untouched test.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "outputs/route_e/prompt_ablation/assistant/full"
COST = ROOT / "outputs/route_e/token_aware/full/validation_predictions.json"
OUT = ROOT / "results/route_e/p1_utility_lambda_sweep"
LAMBDAS = (0., 0.005, 0.01, 0.02, 0.05, 0.10)
BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)
OBJECTIVES = ("utility_hard", "utility_soft")


def make_targets(rows, lam, tau, objective):
    """Train-only target; positive = prefer NoThinking."""
    dp = np.asarray([r["p_n"] - r["p_t"] for r in rows], dtype=np.float64)
    dt = np.asarray([r["t_t"] - r["t_n"] for r in rows], dtype=np.float64)
    gap = dp + lam*dt/1000.
    if objective == "utility_hard":
        y = (gap >= 0).astype(np.float32)
    elif objective == "utility_soft":
        y = (1/(1+np.exp(-np.clip(gap/tau, -60, 60)))).astype(np.float32)
    else:
        raise ValueError(objective)
    if not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise ValueError("Invalid training targets")
    return y


def check_scores(rows, scores, method):
    if set(scores) != {r["sample_id"] for r in rows} or len(scores) != len(rows):
        raise ValueError(f"{method}: validation IDs do not match")
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError(f"{method}: nonfinite score")


def read_p1(rows):
    obj = json.loads((P1/"validation_p_nt.json").read_text(encoding="utf-8"))
    scores = {str(k):float(v) for k,v in obj["p_nt"].items()}
    check_scores(rows, scores, "P1 historical")
    return scores


def read_cost(rows, path, sha):
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("strict_val_sha256") != sha:
        raise ValueError("Cost-only validation SHA mismatch")
    scores = {str(k):float(v["delta_t_pred"]) for k,v in obj["predictions"].items()}
    check_scores(rows, scores, "P1 Cost-only")
    return scores


def make_curve(rows, scores):
    """k=0 all NoThinking; k=450 all Thinking, matching compute_nauc_ac."""
    from route_e_equal_budget import dense_curve
    original = dense_curve(rows, scores)  # j = # NoThinking, 0..450
    n = len(rows)
    if len(original) != n+1:
        raise ValueError("Expected 451 dense policies")
    return {k:(float(original[n-k]["tokens"]),float(original[n-k]["accuracy"]))
            for k in range(n+1)}


def verify_historical(rows, Xval, scores):
    import route_e_frozen as base
    with np.load(P1/"linear_probe.npz", allow_pickle=False) as src:
        params = {key:np.array(src[key], copy=True)
                  for key in ("weight","bias","mean","scale")}
    values = base.predict_linear(Xval, params)
    diff = max(abs(float(v)-scores[r["sample_id"]]) for v,r in zip(values,rows))
    if diff > 1e-5:
        raise RuntimeError(f"P1 saved probe/prediction mismatch {diff:g}")
    t,a = make_curve(rows,scores)[225]
    if abs(a-344/450)>1e-6 or abs(t-1427.05)>2.:
        raise RuntimeError(f"Historical P1 50% NT reproduction failed: {a}, {t}")
    print(f"P1_HISTORICAL_BASELINE_PASS max_score_diff={diff:.2g} "
          f"accuracy={100*a:.4f}% tokens={t:.2f}",flush=True)
    return {"max_score_difference":diff,"50pct_accuracy":a,"50pct_tokens":t}


def csv_write(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def draw_plots(out, curves, results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("P1_LAMBDA_PLOT_SKIPPED matplotlib unavailable",flush=True)
        return

    fig,ax=plt.subplots(figsize=(8.5,4.8))
    for objective,marker in (("utility_hard","o"),("utility_soft","s")):
        rows=sorted((r for r in results if r["objective"]==objective),
                    key=lambda r:r["lambda"])
        ax.plot([r["lambda"] for r in rows],[r["nAUC_AC"] for r in rows],
                marker=marker,linewidth=1.6,label=objective)
    a=next(r["nAUC_AC"] for r in results if r["objective"]=="classification")
    ax.axhline(a,ls="--",linewidth=1.2,label="P1 Classification")
    cost=next((r["nAUC_AC"] for r in results if r["objective"]=="cost_only"),None)
    if cost is not None:
        ax.axhline(cost,ls=":",linewidth=1.4,label="P1 Cost-only")
    ax.set(xlabel="Lambda (token penalty per 1,000 output tokens)",
           ylabel="nAUC-AC",
           title="Frozen DeepSeek P1 | Utility lambda sensitivity")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(out/f"nauc_vs_lambda.{ext}",dpi=210,bbox_inches="tight")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,5),sharex=True,sharey=True)
    for ax,objective in zip(axes,OBJECTIVES):
        subset=sorted((r for r in results if r["objective"]==objective),
                      key=lambda r:r["lambda"])
        colors=plt.get_cmap("viridis")(np.linspace(.05,.90,len(subset)))
        for color,r in zip(colors,subset):
            pts=[curves[r["method"]][k] for k in range(451)]
            ax.plot([x[0] for x in pts],[100*x[1] for x in pts],
                    linewidth=1.1,color=color,label=f"λ={r['lambda']:g}")
        for name,ls in (("P1 Classification","--"),("P1 Cost-only",":")):
            if name in curves:
                pts=[curves[name][k] for k in range(451)]
                ax.plot([x[0] for x in pts],[100*x[1] for x in pts],
                        ls=ls,color="black",linewidth=1.2,label=name)
        ax.set(title=objective,xlabel="Observed mean output tokens")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel("Strict validation accuracy (%)")
    fig.suptitle("P1 Utility λ sweep | 451 threshold policies per λ")
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(out/f"threshold_curves_by_lambda.{ext}",dpi=195,bbox_inches="tight")
    plt.close(fig)


def run(args):
    import torch
    import route_e_frozen as base
    import route_e_prompt_ablation as prefix
    from route_e_equal_budget import dense_curve,linear_bracket
    from compute_nauc_ac import calculate

    if not args.lambdas or len(set(args.lambdas))!=len(args.lambdas):
        raise ValueError("Lambda grid must be nonempty and unique")
    if any(not math.isfinite(lam) or lam<0 for lam in args.lambdas):
        raise ValueError("Lambda grid must contain nonnegative finite numbers")
    if not math.isfinite(args.tau) or args.tau<=0:
        raise ValueError("tau must be finite and positive")
    if abs(args.l2-.001)>1e-12 or args.max_iter!=200:
        raise ValueError("Keep original P1 L2=1e-3 and LBFGS max_iter=200")
    if not args.budgets or any(not math.isfinite(b) or b<=0 for b in args.budgets):
        raise ValueError("Budgets must be finite and positive")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; pass --overwrite")

    train,val,overlap=base.load_protocol()  # strict 3000/450 hashes and IDs
    Xtr,mt=prefix.read_cache(P1/"train.npz",train,base.TRAIN_SHA256,
                             "assistant",False)
    Xva,mv=prefix.read_cache(P1/"val.npz",val,base.VAL_SHA256,
                             "assistant",False)
    for field in ("model","prompt_mode","chat_template_sha256","pooling",
                  "layer","dtype","max_length"):
        if mt[field]!=mv[field]:
            raise ValueError(f"P1 train/val metadata mismatch: {field}")
    if Xtr.shape!=(3000,1536) or Xva.shape!=(450,1536):
        raise ValueError("Expected strict P1 features 3000x1536 and 450x1536")
    if mt["layer"]!="last_after_final_norm" or mt["pooling"]!="last_valid_prompt_token":
        raise ValueError("Wrong P1 representation")

    original=read_p1(val)
    historical=verify_historical(val,Xva,original)
    all_scores={"P1 Classification":original}
    descriptions={"P1 Classification":("classification",None,"historical frozen probe")}
    if not args.skip_cost_only:
        all_scores["P1 Cost-only"]=read_cost(val,args.cost_scores,base.VAL_SHA256)
        descriptions["P1 Cost-only"]=("cost_only",None,str(args.cost_scores))

    args.output.mkdir(parents=True,exist_ok=True)
    preds_dir=args.output/"predictions"
    preds_dir.mkdir(exist_ok=True)
    class_targets=np.asarray([r["p_n"]>=r["p_t"] for r in train],
                             dtype=np.float32)
    torch.manual_seed(args.seed)
    fits=[]
    for obj in OBJECTIVES:
        for lam in args.lambdas:
            target=make_targets(train,lam,args.tau,obj)
            name=f"P1 {obj} lambda={lam:g}"
            file_stem=f"{obj}_lambda_{lam:g}"
            if obj=="utility_hard" and lam==0:
                if not np.array_equal(target,class_targets):
                    raise RuntimeError("Lambda-zero hard target != Classification")
                scores=original
                loss=None
                source="reused historical P1 probe"
            else:
                fit=base.train_logistic(Xtr,target,args.l2,args.max_iter)
                values=base.predict_linear(Xva,fit)
                scores={r["sample_id"]:float(value) for r,value in zip(val,values)}
                np.savez_compressed(preds_dir/f"{file_stem}_probe.npz",
                    weight=fit["weight"],bias=np.asarray(fit["bias"]),
                    mean=fit["mean"],scale=fit["scale"])
                loss=float(fit["train_objective"])
                source="new CPU LBFGS on frozen P1"
            check_scores(val,scores,name)
            (preds_dir/f"{file_stem}_validation_p_nt.json").write_text(
                json.dumps({"p_nt":scores},indent=2)+"\n",encoding="utf-8")
            all_scores[name]=scores
            descriptions[name]=(obj,float(lam),source)
            flips=int(np.sum(target!=class_targets)) if obj=="utility_hard" else None
            fits.append({"method":name,"objective":obj,"lambda":float(lam),
                         "target_mean":float(target.mean()),
                         "hard_flips_vs_classification":flips,
                         "train_loss":loss,"source":source})
            print(f"P1_LAMBDA_FIT {obj} lambda={lam:g} "
                  f"target_mean={target.mean():.4f} flips={flips} "
                  f"loss={loss}",flush=True)

    curves={name:make_curve(val,s) for name,s in all_scores.items()}
    metrics={r["method"]:r for r in calculate(curves)}
    if abs(metrics["P1 Classification"]["nAUC_AC"]-23.4831)>.025:
        raise RuntimeError("P1 nAUC reproduction FAIL (expected +23.4831)")
    if "P1 Cost-only" in metrics and abs(metrics["P1 Cost-only"]["nAUC_AC"]-38.3885)>.025:
        raise RuntimeError("Cost-only nAUC reproduction FAIL (expected +38.3885)")
    rows_summary=[]
    rows_budget=[]
    rows_curve=[]
    fits_by_name={r["method"]:r for r in fits}
    for name,scores in all_scores.items():
        obj,lam,_=descriptions[name]
        m=metrics[name]
        rows_summary.append({
            "method":name,"objective":obj,"lambda":lam,
            "AUC_AC":m["AUC_AC"],"nAUC_AC":m["nAUC_AC"],
            "mean_gain_pp":m["mean_gain_pp"],
            "cost_backtracks":m["cost_backtracks"],
            "hard_flips_vs_classification":
                fits_by_name.get(name,{}).get("hard_flips_vs_classification")})
        # Supplementary fixed-budget points. NOT used to calculate nAUC-AC.
        dense=dense_curve(val,scores)
        for budget in args.budgets:
            b=linear_bracket(dense,budget)
            rows_budget.append({
                "method":name,"objective":obj,"lambda":lam,"budget":budget,
                "accuracy_pct":100*b["accuracy"],
                "nothinking_fraction":b["expected_nothinking_fraction"]})
        for k in range(451):
            t,a=curves[name][k]
            rows_curve.append({
                "method":name,"objective":obj,"lambda":lam,
                "n_thinking":k,"thinking_fraction":k/450,
                "accuracy":a,"mean_output_tokens":t})
        print(f"P1_LAMBDA_NAUC {name}: AUC={m['AUC_AC']:.4f} "
              f"nAUC={m['nAUC_AC']:+.4f} "
              f"mean_gain={m['mean_gain_pp']:+.3f}pp "
              f"backwards={m['cost_backtracks']}",flush=True)

    selected={}
    for obj in OBJECTIVES:
        subset=[r for r in rows_summary if r["objective"]==obj]
        best=min(subset,key=lambda r:(-r["nAUC_AC"],r["lambda"]))
        selected[obj]={"lambda":best["lambda"],"nAUC_AC":best["nAUC_AC"],
                       "selection":"validation exploratory only"}
        print(f"P1_LAMBDA_DEV_BEST {obj} lambda={best['lambda']:g} "
              f"nAUC={best['nAUC_AC']:+.4f} "
              "(NOT confirmed on held-out test)",flush=True)

    out=args.output
    csv_write(out/"nauc_ac_lambda_sweep.csv",rows_summary,
              ["method","objective","lambda","AUC_AC","nAUC_AC",
               "mean_gain_pp","cost_backtracks","hard_flips_vs_classification"])
    csv_write(out/"fixed_budget_accuracy.csv",rows_budget,
              ["method","objective","lambda","budget","accuracy_pct","nothinking_fraction"])
    csv_write(out/"all_451_threshold_policies.csv",rows_curve,
              ["method","objective","lambda","n_thinking",
               "thinking_fraction","accuracy","mean_output_tokens"])
    csv_write(out/"fit_metadata.csv",fits,
              ["method","objective","lambda","target_mean",
               "hard_flips_vs_classification","train_loss","source"])
    report={
        "status":"validation-only sensitivity; independent test not used",
        "train_count":len(train),"val_count":len(val),
        "train_sha256":base.TRAIN_SHA256,"val_sha256":base.VAL_SHA256,
        "train_val_text_overlap_count":overlap,
        "encoder":"frozen DeepSeek P1 last-after-final-norm, last prompt token",
        "utility":"(p_N-p_T)+lambda*(t_T-t_N)/1000",
        "lambda_grid":args.lambdas,"temperature":args.tau,
        "head":"LBFGS Logistic","l2":args.l2,"max_iter":args.max_iter,"seed":args.seed,
        "historic_P1_reproduction":historical,
        "cost_only_included":not args.skip_cost_only,
        "metric_convention":"451 observed-token policies sorted by token cost; clipped shared endpoints; minus Random trapezoid",
        "budget_sweep_is_supplementary":args.budgets,
        "random_reference_AUC":metrics["P1 Classification"]["random_reference_AUC"],
        "token_interval":[metrics["P1 Classification"]["min_cost"],
                          metrics["P1 Classification"]["max_cost"]],
        "encoder_prefill_and_router_overhead_included":False,
        "selected_lambda_on_validation_only":selected,
    }
    (out/"protocol_and_selection.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if not args.no_plot:
        draw_plots(out,curves,rows_summary)
    print(f"P1_LAMBDA_SWEEP_DONE {out}",flush=True)


def self_test():
    row=[{"p_n":.6,"p_t":.7,"t_n":500,"t_t":2500}]
    assert make_targets(row,0,.25,"utility_hard")[0]==0.
    assert make_targets(row,.1,.25,"utility_hard")[0]==1.
    assert make_targets(row,0,.25,"utility_soft")[0]<.5
    assert make_targets(row,.1,.25,"utility_soft")[0]>.5
    assert np.isclose(make_targets(row,0,.25,"utility_soft")[0],
                      1/(1+math.exp(.1/.25)),atol=1e-7)
    assert len(set(LAMBDAS))==6
    print("P1_LAMBDA_SWEEP_SELFTEST_PASS")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    r=sub.add_parser("run")
    r.add_argument("--lambdas",nargs="+",type=float,default=list(LAMBDAS))
    r.add_argument("--tau",type=float,default=.25)
    r.add_argument("--l2",type=float,default=.001)
    r.add_argument("--max-iter",type=int,default=200)
    r.add_argument("--budgets",nargs="+",type=float,default=list(BUDGETS))
    r.add_argument("--cost-scores",type=Path,default=COST)
    r.add_argument("--skip-cost-only",action="store_true")
    r.add_argument("--output",type=Path,default=OUT)
    r.add_argument("--overwrite",action="store_true")
    r.add_argument("--seed",type=int,default=42)
    r.add_argument("--no-plot",action="store_true")
    args=p.parse_args()
    if args.command=="self-test":
        self_test()
    else:
        run(args)


if __name__=="__main__":
    main()
