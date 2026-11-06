#!/usr/bin/env python3
"""AdaptThink sealed-test report for three pre-selected external routers.

Scores are pre-generation predictions; test outcomes are used ONLY for offline
reporting.  The headline table uses validation-calibrated thresholds (never
recalibrated on test).  Threshold-swept test AUC is explicitly descriptive.

Install: numpy matplotlib (also part of AdaptThink's environment).
Usage: python adaptthink_test_report.py --help
       python adaptthink_test_report.py --self-test
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

METHODS = (
    ("ModernBERT + Utility-Hard", "utility", "modernbert"),
    ("Frozen DeepSeek P1 + Classification", "p1", "p1"),
    ("Frozen DeepSeek P1 + Cost-only", "cost", "cost"),
)
DATASETS = ("gsm8k", "math", "aime")
DISPLAY = {"gsm8k": "GSM8K", "math": "MATH-500", "aime": "AIME 2024"}
COLORS = ("#2563EB", "#E97827", "#159957")
VAL_SHA = "4c1e984d3744cc089dcc963a30af0ef3d1ed9808c4bfd245a97e48787d501461"
DEFAULT_BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    result, seen = [], set()
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            sid = str(obj["sample_id"])
            if sid in seen:
                raise ValueError(f"Duplicate label ID in {path}:{lineno}: {sid}")
            seen.add(sid)
            p_t, p_n = float(obj["p_thinking"]), float(obj.get("p_nothinking_strict", obj["p_nothinking"]))
            t_t, t_n = float(obj["mean_tokens_thinking"]), float(obj["mean_tokens_nothinking"])
            assert all(math.isfinite(x) for x in (p_t,p_n,t_t,t_n))
            if not (0<=p_t<=1 and 0<=p_n<=1 and t_t>=0 and t_n>=0):
                raise ValueError(f"Invalid outcomes for {sid}")
            result.append({"sample_id":sid, "dataset":sid.split(":")[0],
                           "problem":obj["problem"], "p_t":p_t,"p_n":p_n,"t_t":t_t,"t_n":t_n,
                           "n_generations":int(obj.get("n_generations",1))})
    return result


def read_scores(path, kind, ids, split_sha=None):
    path=Path(path)
    if kind=="modernbert":
        items=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        scores={}
        for item in items:
            sid=str(item["sample_id"])
            if sid in scores: raise ValueError(f"Duplicate ModernBERT prediction: {sid}")
            scores[sid]=float(item["predicted_p_nothinking"])
        if len(scores)!=len(items): raise ValueError("Duplicate score IDs")
    elif kind=="p1":
        obj=json.loads(path.read_text(encoding="utf-8"))
        if split_sha and obj.get("test_labels_sha256") and obj["test_labels_sha256"]!=split_sha:
            raise ValueError(f"Wrong P1 test-label SHA for {path}")
        scores={str(k):float(v) for k,v in obj["p_nt"].items()}
    elif kind=="cost":
        obj=json.loads(path.read_text(encoding="utf-8"))
        if obj.get("strict_val_sha256") and split_sha==VAL_SHA and obj["strict_val_sha256"]!=VAL_SHA:
            raise ValueError("Cost validation SHA mismatch")
        if split_sha and obj.get("test_labels_sha256") and obj["test_labels_sha256"]!=split_sha:
            raise ValueError("Cost test-label SHA mismatch")
        scores={str(k):float(v["delta_t_pred"]) for k,v in obj["predictions"].items()}
    else: raise ValueError(kind)
    if len(scores)!=len(ids) or set(scores)!=set(ids):
        raise ValueError(f"Scores {path}: missing={sorted(set(ids)-set(scores))[:4]} extra={sorted(set(scores)-set(ids))[:4]}")
    if not all(math.isfinite(v) for v in scores.values()): raise ValueError(f"Nonfinite scores: {path}")
    return scores


def means(rows, probs):
    """Per-question means; AIME repeated generations must be pre-aggregated."""
    n=len(rows)
    if not n: raise ValueError("Empty group")
    q=np.asarray([probs[r["sample_id"]] for r in rows],float)
    pn=np.asarray([r["p_n"] for r in rows]);pt=np.asarray([r["p_t"] for r in rows])
    tn=np.asarray([r["t_n"] for r in rows]);tt=np.asarray([r["t_t"] for r in rows])
    if np.any((q<0)|(q>1)):raise ValueError("Invalid NT probabilities")
    return {"acc":float(np.mean(q*pn+(1-q)*pt)),
            "tok":float(np.mean(q*tn+(1-q)*tt)),
            "nt_fraction":float(q.mean()),"n":n}


def fixed_modes(rows):
    ids=[r["sample_id"] for r in rows]
    return means(rows,dict.fromkeys(ids,1.)),means(rows,dict.fromkeys(ids,0.))


def curve(rows, scores):
    """k=0 all NoThinking; k=N all Thinking; scores high -> NoThinking first."""
    ordered=sorted(rows,key=lambda r:(-scores[r["sample_id"]],r["sample_id"]))[::-1]
    n=len(rows)
    nt,t=fixed_modes(rows)
    gains=np.array([r["p_t"]-r["p_n"] for r in ordered])/n
    costs=np.array([r["t_t"]-r["t_n"] for r in ordered])/n
    acc=nt["acc"]+np.r_[0.,np.cumsum(gains)]
    tok=nt["tok"]+np.r_[0.,np.cumsum(costs)]
    if abs(acc[-1]-t["acc"])>1e-8 or abs(tok[-1]-t["tok"])>1e-6:
        raise AssertionError("Curve endpoint check failed")
    return {"accuracy":acc,"tokens":tok,"n_thinking":np.arange(n+1),"n":n}


def cost_auc(points,low,high):
    """Same convention as scripts/compute_nauc_ac.py (sorted measured cost)."""
    if not high>low:raise ValueError("Invalid endpoints")
    grouped={}
    for x,y in sorted(points):grouped.setdefault(float(x),[]).append(float(y))
    xs=sorted(grouped); ys=[float(np.mean(grouped[x])) for x in xs]
    if xs[0]>low+1e-7 or xs[-1]<high-1e-7:raise ValueError("Missing baseline endpoints")
    grid=[low]+[x for x in xs if low<x<high]+[high]
    interpolated=np.interp(grid,xs,ys)
    return float(np.trapezoid(interpolated,grid) if hasattr(np,"trapezoid") else np.trapz(interpolated,grid))


def auc_row(name, rows, scores):
    c=curve(rows,scores)
    nt,t=fixed_modes(rows)
    span=t["tok"]-nt["tok"]
    baseline=.5*(t["acc"]+nt["acc"])*span
    area=cost_auc(zip(c["tokens"],c["accuracy"]),nt["tok"],t["tok"])
    return {"method":name,"n":len(rows),"nAUC_AC":area-baseline,
            "AUC_AC":area,"random_reference_AUC_AC":baseline,
            "mean_gain_pp":100*(area-baseline)/span,"nt_tokens":nt["tok"],"t_tokens":t["tok"],
            "backward_token_steps":int((np.diff(c["tokens"])<0).sum())}


def threshold_for_budget(rows,scores,budget):
    """Choose closest measured validation token cost, NOT best accuracy.

    The numeric prediction cutoff is frozen and transferred unchanged to test.
    NB: This calibration uses observed VALIDATION lengths; it does not guarantee
    the realized test token budget.
    """
    c=curve(rows,scores)
    nt_idx=int(np.argmin(np.abs(c["tokens"]-budget)))
    n_thinking=nt_idx
    n_nt=len(rows)-n_thinking
    ranked=sorted(rows,key=lambda r:(-scores[r["sample_id"]],r["sample_id"]))
    if n_nt==0:
        cutoff=float("inf")
    elif n_nt==len(rows):
        cutoff=float("-inf")
    else:
        hi=scores[ranked[n_nt-1]["sample_id"]]
        lo=scores[ranked[n_nt]["sample_id"]]
        if not hi>lo:
            raise ValueError("Validation threshold tied; implement deterministic tie control")
        cutoff=lo+.5*(hi-lo)
    return {"budget_target":float(budget),"score_cutoff":cutoff,
            "validation_mean_tokens":float(c["tokens"][nt_idx]),
            "validation_accuracy":float(c["accuracy"][nt_idx]),
            "validation_nt_fraction":n_nt/len(rows),"validation_n_thinking":n_thinking}


def apply_threshold(rows,scores,threshold):
    return means(rows,{r["sample_id"]:float(scores[r["sample_id"]]>=threshold)
                       for r in rows})


def make_figures(out, scopes, test_scores, val, val_scores, budgets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    def decorate(ax,rows,ylabel):
        nt,t=fixed_modes(rows)
        ax.plot([nt["tok"],t["tok"]],[100*nt["acc"],100*t["acc"]],
                "--",color="#A2AAB6",linewidth=1.4,label="Global Random (expected)")
        ax.scatter([nt["tok"]],[100*nt["acc"]],marker="*",s=110,color="#E05A47",label="Always NoThinking",zorder=4)
        ax.scatter([t["tok"]],[100*t["acc"]],marker="*",s=110,color="#2A9D65",label="Always Thinking",zorder=4)
        ax.set_xlabel("Mean generated tokens / question")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}"))
    def save(fig,base):
        fig.savefig(out/(base+".png"),bbox_inches="tight",dpi=230)
        fig.savefig(out/(base+".svg"),bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.2,5.4))
    rows=scopes["overall"]
    for (name,_,_),color in zip(METHODS,COLORS):
        c=curve(rows,test_scores[name]);ax.plot(c["tokens"],100*c["accuracy"],color=color,linewidth=1.65,label=name)
    decorate(ax,rows,"Strict test accuracy (%)")
    ax.set_title("Held-out test | Accuracy–Token threshold sweeps")
    ax.legend(fontsize=8,loc="best",frameon=False)
    fig.tight_layout();save(fig,"test_threshold_overlay")

    fig,axes=plt.subplots(1,3,figsize=(16,4.8))
    for ax,dataset in zip(axes,DATASETS):
        rr=scopes[dataset]
        for (name,_,_),color in zip(METHODS,COLORS):
            c=curve(rr,test_scores[name]);ax.plot(c["tokens"],100*c["accuracy"],color=color,linewidth=1.5,label=name)
        decorate(ax,rr,"Strict test accuracy (%)")
        ax.set_title(f"{DISPLAY[dataset]} (n={len(rr)})",fontsize=11)
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=3,fontsize=8,frameon=False)
    fig.suptitle("Held-out test | Per-dataset Accuracy–Token curves",fontsize=13)
    fig.tight_layout(rect=(0,.1,1,.94));save(fig,"test_per_dataset_threshold_curves")

    fig,ax=plt.subplots(figsize=(9.2,5.4))
    for (name,_,_),color in zip(METHODS,COLORS):
        values=[]
        for budget in budgets:
            policy=threshold_for_budget(val,val_scores[name],budget)
            result=apply_threshold(rows,test_scores[name],policy["score_cutoff"])
            values.append(result)
        ax.plot([v["tok"] for v in values],[v["acc"]*100 for v in values],
                "-o",color=color,markersize=4,linewidth=1.65,label=name)
        for b,v in zip(budgets,values):
            ax.annotate(str(b), (v["tok"],100*v["acc"]),xytext=(2,4),textcoords="offset points",fontsize=6,color=color)
    decorate(ax,rows,"Strict test accuracy (%)")
    ax.set_title("Validation-calibrated budgets transferred to held-out test")
    ax.legend(loc="best",fontsize=8,frameon=False)
    fig.tight_layout();save(fig,"test_validation_budget_transfer")


def write_csv(path, rows, fieldnames):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)


def report(args):
    # ThinkSwitcher convention: Random (x/y) = x% Thinking and y% NoThinking.
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    val=read_rows(args.val_labels);test=read_rows(args.test_labels)
    val_ids={r["sample_id"] for r in val};test_ids={r["sample_id"] for r in test}
    if args.strict_val_hash and file_sha(args.val_labels)!=VAL_SHA:
        raise ValueError("Validation-450 labels SHA differs from frozen protocol")
    if len(val)!=450 or Counter(r["dataset"] for r in val)!={"gsm8k":300,"math":150}:
        raise ValueError("Wrong validation 450 composition")
    if len(test)!=485 or Counter(r["dataset"] for r in test)!={"gsm8k":330,"math":125,"aime":30}:
        raise ValueError("Wrong held-out test 485 composition")
    if val_ids & test_ids:raise ValueError("Validation and test ID intersection; abort")
    if any(r["n_generations"]!=(16 if r["dataset"]=="aime" else 1) for r in test):
        raise ValueError("AIME must represent 16 generations PER QUESTION")
    if args.baseline_summary:
        reference=json.loads(args.baseline_summary.read_text(encoding="utf-8"))["results"]
        for ds in DATASETS:
            items=[r for r in test if r["dataset"]==ds]
            nt,t=fixed_modes(items)
            for mode,data in (("nothinking",nt),("thinking",t)):
                ref=reference[mode][ds]
                if abs(data["acc"]-ref["accuracy"])>1e-7 or abs(data["tok"]-ref["mean_response_tokens"])>1e-5:
                    raise ValueError(f"Baseline cache mismatch: {ds}/{mode}")
    sha=file_sha(args.test_labels)
    val_scores={};test_scores={};sources={}
    for name,key,kind in METHODS:
        vp=getattr(args,key+"_val");tp=getattr(args,key+"_test")
        val_scores[name]=read_scores(vp,kind,val_ids,VAL_SHA)
        test_scores[name]=read_scores(tp,kind,test_ids,sha)
        sources[name]={"validation":str(vp),"test":str(tp),"validation_sha":file_sha(vp),"test_sha":file_sha(tp)}
    scopes={"overall":test};scopes.update({ds:[r for r in test if r["dataset"]==ds] for ds in DATASETS})
    results=[]
    for name,_,_ in METHODS:
        rows=[auc_row(name,scopes[ds],test_scores[name])|{"dataset":ds} for ds in ("overall",)+DATASETS]
        macro_raw=sum(x["nAUC_AC"] for x in rows[1:])/3
        macro_norm=sum(x["mean_gain_pp"] for x in rows[1:])/3
        results.extend(rows)
        results.append({"method":name,"dataset":"Macro average (3 datasets)","n":485,
                        "nAUC_AC":macro_raw,"mean_gain_pp":macro_norm,
                        "AUC_AC":"","random_reference_AUC_AC":"","nt_tokens":"","t_tokens":"","backward_token_steps":""})
    auc_cols=["method","dataset","n","AUC_AC","random_reference_AUC_AC","nAUC_AC","mean_gain_pp","nt_tokens","t_tokens","backward_token_steps"]
    write_csv(out/"nauc_ac_test.csv",results,auc_cols)
    print("TEST_NAUC (raw acc x tokens; normalized mean gain in pp)")
    for r in results:
        print(f"  {r['method']:<40} {r['dataset']:<24} nAUC={r['nAUC_AC']:+9.2f} gain={r['mean_gain_pp']:+7.3f}pp")

    table=[]
    def table_row(label,per_group):
        row={"Method":label}
        for ds in DATASETS:
            z=per_group[ds];row[f"{DISPLAY[ds]} Acc (%)"]=100*z["acc"];row[f"{DISPLAY[ds]} Tok."]=z["tok"]
        row["Avg. Acc (%)"]=sum(100*per_group[ds]["acc"] for ds in DATASETS)/3
        row["Avg. Tok."]=sum(per_group[ds]["tok"] for ds in DATASETS)/3
        row["Pooled Acc (%)"]=100*per_group["overall"]["acc"]
        row["Pooled Tok."]=per_group["overall"]["tok"]
        table.append(row)
    for label,q in (("Always NoThinking (SC-only)",1.),("Always Thinking (LC-only)",0.),
                    ("Random (50/50; expectation)",.5),("Random (25/75; expectation)",.75),("Random (10/90; expectation)",.90)):
        table_row(label,{ds:means(rows,dict.fromkeys((r["sample_id"] for r in rows),q)) for ds,rows in scopes.items()})
    transfers=[]
    for name,_,_ in METHODS:
        for budget in args.budgets:
            p=threshold_for_budget(val,val_scores[name],budget)
            group={ds:apply_threshold(rows,test_scores[name],p["score_cutoff"]) for ds,rows in scopes.items()}
            transfers.append({"method":name,**p, "test_micro_acc":group["overall"]["acc"],
                              "test_micro_tok":group["overall"]["tok"],
                              "test_micro_nt_fraction":group["overall"]["nt_fraction"]})
            if budget==args.main_budget:
                table_row(f"{name} [Val B={int(budget)}]",group)
    write_csv(out/"validation_budget_transfer.csv",transfers,list(transfers[0]))
    cols=list(table[0]);write_csv(out/"test_per_dataset_table.csv",table,cols)
    def fmt(x,token=False):return f"{x:,.0f}" if token else f"{x:.2f}"
    md=["# Held-out test (frozen validation-calibrated operating point)","",
        f"Val target budget = {args.main_budget:,} generated tokens/question. Test observed costs are NOT re-matched.",
        "Avg. is dataset-macro (GSM8K, MATH-500, AIME2024 equally weighted); pooled is question-weighted (485 unique questions).",
        "AIME2024 is averaged over 16 generations per question, then over its 30 questions.","",
        "| Method | GSM8K Acc. | Tok. | MATH-500 Acc. | Tok. | AIME24 Acc. | Tok. | Avg. Acc. | Tok. | Pooled Acc. | Tok. |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in table:
        md.append("| " + " | ".join([r["Method"]]+[fmt(r[f"{DISPLAY[ds]} Acc (%)"]) + "% | " +fmt(r[f"{DISPLAY[ds]} Tok."],True) for ds in DATASETS]+[fmt(r["Avg. Acc (%)"])+"%",fmt(r["Avg. Tok."],True),fmt(r["Pooled Acc (%)"])+"%",fmt(r["Pooled Tok."],True)])+" |")
    md.extend(["","## nAUC-AC (observed-test descriptive curves)","",
               "Raw nAUC-AC = integrated router accuracy-vs-cost minus the Always NT/T random trapezoid (accuracy as fraction).",
               "Mean gain (pp) = 100 * nAUC-AC / (T token endpoint - NT token endpoint).",
               "Macro averages across datasets are NOT equivalent to pooled 485-question AUC.","",
               "| Method | Pooled nAUC-AC | Macro nAUC-AC | Pooled mean gain (pp) | Macro mean gain (pp) |",
               "| :--- | ---: | ---: | ---: | ---: |"])
    for name,_,_ in METHODS:
        x=next(r for r in results if r["method"]==name and r["dataset"]=="overall")
        y=next(r for r in results if r["method"]==name and r["dataset"].startswith("Macro"))
        md.append(f"| {name} | {x['nAUC_AC']:+.2f} | {y['nAUC_AC']:+.2f} | {x['mean_gain_pp']:+.3f} | {y['mean_gain_pp']:+.3f} |")
    (out/"heldout_test_report.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    provenance={"dataset":"canonical held-out 485 unique questions; AIME24 x16 per question",
                "test_labels_sha256":sha,"val_labels_sha256":file_sha(args.val_labels),
                "sources":sources,"budget_target":args.main_budget,"transfer_budgets":list(args.budgets),
                "policy":"validation actual-cost-nearest top-k -> frozen numeric score cutoff -> apply unchanged to test",
                "auc":"descriptive test rank sweep, NOT a selected deployable policy",
                "limitations":"Expected offline replay from cached 2-mode answers; router forward/prefill and encoder latency excluded"}
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    if not args.no_plots:make_figures(out,scopes,test_scores,val,val_scores,args.budgets)
    print("TEST_REPORT_DONE",out)


def self_test():
    rows=[{"sample_id":f"x:{i}","dataset":"x","p_t":a,"p_n":b,"t_t":tt,"t_n":tn}
          for i,(a,b,tt,tn) in enumerate(((1.,0.,20.,5.),(0.,1.,30.,10.),(.5,.5,50.,10.),(.8,.2,50.,10.)))]
    scores={"x:0":0.1,"x:1":0.9,"x:2":0.6,"x:3":0.2}
    c=curve(rows,scores)
    assert abs(c["accuracy"][0]-.425)<1e-10
    assert abs(c["accuracy"][1]-.675)<1e-10
    nt,t=fixed_modes(rows)
    assert abs(c["tokens"][0]-nt["tok"])<1e-8
    assert abs(c["tokens"][-1]-t["tok"])<1e-8
    mid=means(rows,dict.fromkeys(scores,.5))
    assert abs(mid["acc"]-.5*(nt["acc"]+t["acc"]))<1e-10
    assert abs(cost_auc([(5,.3),(7.5,.6),(10,.7)],5,10)-2.75)<1e-8
    assert math.isfinite(auc_row("test",rows,scores)["nAUC_AC"])
    p=threshold_for_budget(rows,scores,18)
    m=apply_threshold(rows,scores,p["score_cutoff"])
    assert abs(m["tok"]-p["validation_mean_tokens"])<1e-8
    print("THREE_ROUTER_TEST_REPORT_SELFTEST_PASS")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--val-labels",type=Path,default=Path("results/val_450_labels.jsonl"))
    p.add_argument("--test-labels",type=Path,default=Path("results/test_485_labels.jsonl"))
    p.add_argument("--baseline-summary",type=Path,default=Path("results/baseline_1p5b/summary.json"))
    p.add_argument("--utility-val",type=Path,default=Path("outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/validation_predictions.jsonl"))
    p.add_argument("--p1-val",type=Path,default=Path("outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json"))
    p.add_argument("--cost-val",type=Path,default=Path("outputs/route_e/token_aware/full/validation_predictions.json"))
    p.add_argument("--utility-test",type=Path,default=Path("outputs/test_485/utility_predictions.jsonl"))
    p.add_argument("--p1-test",type=Path,default=Path("outputs/test_485/p1_predictions.json"))
    p.add_argument("--cost-test",type=Path,default=Path("outputs/test_485/cost_predictions.json"))
    p.add_argument("--main-budget",type=int,default=1300)
    p.add_argument("--budgets",type=int,nargs="+",default=list(DEFAULT_BUDGETS))
    p.add_argument("--out",type=Path,default=Path("results/test_485_three_routers"))
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--strict-val-hash",action="store_true",default=True)
    a=p.parse_args()
    if a.self_test: self_test();return
    if a.main_budget not in a.budgets: p.error("--main-budget must be in --budgets")
    report(a)

if __name__=="__main__":main()
