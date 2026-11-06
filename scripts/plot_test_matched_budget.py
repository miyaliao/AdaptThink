#!/usr/bin/env python3
"""Publication figures/tables for AdaptThink held-out test (NO GPU).

Outputs (separate from the original report):
- matched_test_budget_curve.{png,svg,csv}: exact common *observed test*
  budgets by interpolating adjacent TEST cost-sorted threshold policies.
  This is post-hoc DESCRIPTIVE, not a deployable budget guarantee.
- thinkswitcher_style_val_BXXXX.{png,svg,csv,md,tex}: one validation-
  calibrated cutoff per method, transferred unchanged to all test datasets.
- thinkswitcher_style_testmatched_BXXXX.{png,svg,csv,md,tex}: single global
  TEST matched-cost policy per method, applied to all subdatasets; descriptive.

Random (x/y) follows ThinkSwitcher: x% Thinking, y% NoThinking.
Run from the repository root after the original three-router test finishes:
    python scripts/plot_test_matched_budget.py --self-test
    python scripts/plot_test_matched_budget.py --budget 1300
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import adaptthink_test_report as base

SCOPES = ("gsm8k", "math", "aime")
ALL = ("overall",) + SCOPES
BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)
COLUMN_GROUPS = (("gsm8k", "GSM8K"), ("math", "MATH500"),
                 ("aime", "AIME24"), ("macro", "Avg."))


def split_rows(test):
    return {"overall":test, **{ds:[r for r in test if r["dataset"] == ds]
                                   for ds in SCOPES}}


def policy_curves(test, scores):
    """Sweep one global top-k policy and record every dataset at the SAME k.

    AIME records are question-level means over its 16 repeats. Thresholds are
    global (485 unique questions), never tuned separately for a dataset.
    """
    ordered = sorted(test, key=lambda r:(-scores[r["sample_id"]],r["sample_id"]))[::-1]
    scopes = split_rows(test)
    curves={}
    for ds in ALL:
        group=scopes[ds]
        n=len(group)
        n_acc=sum(r["p_n"] for r in group)/n
        n_tok=sum(r["t_n"] for r in group)/n
        ac_d=np.array([(r["p_t"]-r["p_n"])/n if ds=="overall" or r["dataset"]==ds else 0.
                       for r in ordered], dtype=np.float64)
        tk_d=np.array([(r["t_t"]-r["t_n"])/n if ds=="overall" or r["dataset"]==ds else 0.
                       for r in ordered], dtype=np.float64)
        curves[ds]={"acc":n_acc+np.r_[0.,np.cumsum(ac_d)],
                    "tok":n_tok+np.r_[0.,np.cumsum(tk_d)]}
        _,t=base.fixed_modes(group)
        if abs(curves[ds]["acc"][-1]-t["acc"])>1e-8 or abs(curves[ds]["tok"][-1]-t["tok"])>1e-5:
            raise AssertionError(f"Endpoint mismatch for {ds}")
    return curves


def policies_sorted_by_cost(curves):
    """Group equal actual costs without inspecting any accuracy."""
    xs=curves["overall"]["tok"]
    order=sorted(range(len(xs)), key=lambda k:(float(xs[k]),k))
    result=[]
    for k in order:
        x=float(xs[k])
        if result and x==result[-1]["cost"]:
            result[-1]["ks"].append(k)
        else:
            result.append({"cost":x,"ks":[k]})
    return result


def matched_budget(curves, budget):
    """Interpolate in *test measured cost space*, NO accuracy-based selection.

    Shared stochastic mixture of at most two global policies (or tie groups).
    The very same policy/mixture is evaluated in ALL dataset columns.
    """
    order=policies_sorted_by_cost(curves)
    costs=[p["cost"] for p in order]
    if not costs[0] - 1e-7 <= budget <= costs[-1] + 1e-7:
        raise ValueError(f"Budget {budget} outside observed-cost sweep [{costs[0]:.1f},{costs[-1]:.1f}]")
    import bisect
    hi=bisect.bisect_left(costs,budget)
    if hi < len(costs) and abs(costs[hi]-budget)<1e-9:
        left=right=order[hi];w=0.
    else:
        if hi==0 or hi==len(costs):raise ValueError("Budget bracket missing")
        left=order[hi-1];right=order[hi]
        w=(budget-left["cost"])/(right["cost"]-left["cost"])
    def group_avg(curve, ks):
        return {field:float(np.mean(curve[field][ks])) for field in ("acc","tok")}
    results={}
    for ds in ALL:
        a=group_avg(curves[ds],left["ks"])
        b=group_avg(curves[ds],right["ks"])
        results[ds]={field:(1-w)*a[field]+w*b[field] for field in ("acc","tok")}
    if abs(results["overall"]["tok"]-budget)>5e-6:
        raise AssertionError("Interpolated TEST budget incorrect")
    return results,{"target_test_tokens":float(budget),
                    "low_token_cost":left["cost"],"high_token_cost":right["cost"],
                    "lower_global_k":left["ks"],"upper_global_k":right["ks"],
                    "upper_policy_mixture_weight":float(w),
                    "selection_uses_test_outcomes":False,
                    "selection_uses_observed_test_costs":True}


def from_fixed_probability(scopes, nt_prob):
    return {ds:base.means(group,{r["sample_id"]:float(nt_prob) for r in group})
            for ds,group in scopes.items()}


def summary_row(name, grouped):
    row={"Method":name}
    for ds,label in COLUMN_GROUPS:
        if ds=="macro":
            row[f"{label} Acc."]=sum(100*grouped[d]["acc"] for d in SCOPES)/3
            row[f"{label} Tok."]=sum(grouped[d]["tok"] for d in SCOPES)/3
        else:
            row[f"{label} Acc."]=100*grouped[ds]["acc"]
            row[f"{label} Tok."]=grouped[ds]["tok"]
    row["Pooled Acc."]=100*grouped["overall"]["acc"]
    row["Pooled Tok."]=grouped["overall"]["tok"]
    return row


def paper_table_rows(scopes, methods):
    """ThinkSwitcher Table 1: fixed-ratio random baselines, one policy/router."""
    baselines=[("SC-Only",1.0),("LC-Only",0.0),
               ("Random (50/50)",0.50),("Random (25/75)",0.75),
               ("Random (10/90)",0.90)]
    rows=[summary_row(label,from_fixed_probability(scopes,q)) for label,q in baselines]
    rows += [summary_row(name,methods[name]) for name,_,_ in base.METHODS]
    return rows


def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def make_table_figure(rows, out_base, policy_note):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig=plt.figure(figsize=(15.6,4.65),facecolor="white")
    ax=fig.add_axes([0.035,0.13,0.93,0.77]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis("off")
    method_w=.325
    col_w=(1-method_w)/8
    centers=[method_w+col_w*(i+.5) for i in range(8)]
    ax.text(.008,.86,"Method",ha="left",va="center",weight="semibold",fontsize=11)
    for ix,(_,heading) in enumerate(COLUMN_GROUPS):
        x=(centers[2*ix]+centers[2*ix+1])/2
        ax.text(x,.9,heading,ha="center",va="center",weight="semibold",fontsize=11)
        ax.plot([method_w+ix*2*col_w+.012, method_w+(ix+1)*2*col_w-.012],
                [.83,.83],color="0.45",lw=.65)
        ax.text(centers[2*ix],.765,"Acc.",ha="center",va="center",fontsize=9.5)
        ax.text(centers[2*ix+1],.765,"Tok.",ha="center",va="center",fontsize=9.5)
    ax.plot([0,1],[.965,.965],color="0.13",lw=1.25)
    ax.plot([0,1],[.71,.71],color="0.13",lw=.8)
    heights=np.linspace(.635,.085,len(rows))
    for i,(row,y) in enumerate(zip(rows,heights)):
        if i==5: ax.plot([0,1],[.5*(heights[i-1]+heights[i]),.5*(heights[i-1]+heights[i])],color=".4",lw=.65)
        name=row["Method"]
        fs=9.6 if len(name)>31 else 10.0
        ax.text(.008,y,name,ha="left",va="center",fontsize=fs,
                weight="semibold" if i>=5 else "normal")
        for j,(_,label) in enumerate(COLUMN_GROUPS):
            ax.text(centers[j*2],y,f'{row[f"{label} Acc."]:.1f}',ha="center",va="center",fontsize=10)
            ax.text(centers[j*2+1],y,f'{row[f"{label} Tok."]:.0f}',ha="center",va="center",fontsize=10)
    ax.plot([0,1],[.03,.03],color="0.13",lw=1.3)
    fig.text(.5,.955,"AdaptThink | Held-out Test-485",ha="center",fontsize=15,weight="semibold")
    fig.text(.5,.06,policy_note,ha="center",fontsize=9,color="0.4")
    for ext in ("png","svg"):
        fig.savefig(out_base.with_suffix("."+ext),dpi=250,bbox_inches="tight")
    plt.close(fig)


def write_paper_markdown(rows, path, note):
    head="| Method | GSM8K Acc. | Tok. | MATH500 Acc. | Tok. | AIME24 Acc. | Tok. | Avg. Acc. | Tok. |"
    rule="| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    out=["# AdaptThink — Held-out test benchmark", "",note,"",head,rule]
    for r in rows:
        vals=[r["Method"]]
        for _,label in COLUMN_GROUPS:
            vals.append(f'{r[f"{label} Acc."]:.1f}')
            vals.append(f'{r[f"{label} Tok."]:.0f}')
        out.append("| "+" | ".join(vals)+" |")
    out.extend(["", "Random (x/y): x% Thinking and y% NoThinking, following ThinkSwitcher.",
                "Avg. is the unweighted mean of the three dataset-level metrics.",
                "AIME24 accuracy and tokens are first averaged over 16 generations per question."])
    path.write_text("\n".join(out)+"\n",encoding="utf-8")


def write_tex(rows,path,note):
    # Simple booktabs table; requires \usepackage{booktabs}.
    def esc(t):return t.replace("_",r"\_").replace("%",r"\%")
    out=[r"% requires \usepackage{booktabs}",r"\begin{table*}[t]",r"\centering",r"\small",
         r"\begin{tabular}{l*{8}{r}}",r"\toprule",
         r"Method & \multicolumn{2}{c}{GSM8K} & \multicolumn{2}{c}{MATH500} & \multicolumn{2}{c}{AIME24} & \multicolumn{2}{c}{Avg.} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
         r"& Acc. & Tok. & Acc. & Tok. & Acc. & Tok. & Acc. & Tok. \\",r"\midrule"]
    for i,r in enumerate(rows):
        if i==5:out.append(r"\midrule")
        cols=[esc(r["Method"])]
        for _,label in COLUMN_GROUPS:
            cols.extend([f'{r[f"{label} Acc."]:.1f}', f'{r[f"{label} Tok."]:.0f}'])
        out.append(" & ".join(cols)+r" \\")
    out.extend([r"\bottomrule",r"\end{tabular}",
                r"\caption{"+esc(note)+r" Random (x/y) routes x\% to Thinking and y\% to NoThinking; Avg. is a dataset macro average.}",
                r"\end{table*}"])
    path.write_text("\n".join(out)+"\n",encoding="utf-8")


def draw_matched_budget_figure(scopes, predictions, budgets, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    fig,ax=plt.subplots(figsize=(10,6.1))
    test=scopes["overall"]
    nt,t=base.fixed_modes(test)
    all_stats=[]
    for (name,_,_),color in zip(base.METHODS,base.COLORS):
        curves=policy_curves(test,predictions[name])
        points=[]
        for budget in budgets:
            grouped,detail=matched_budget(curves,budget)
            points.append(grouped["overall"]["acc"]*100)
            all_stats.append({"Method":name,"Test Token Budget":budget,
                              "Test Acc (%)":grouped["overall"]["acc"]*100,
                              "Exact matched test tokens":grouped["overall"]["tok"],
                              "Random expected Acc (%)":None,
                              "lower_global_k":str(detail["lower_global_k"]),
                              "upper_global_k":str(detail["upper_global_k"]),
                              "upper_mix_weight":detail["upper_policy_mixture_weight"]})
        ax.plot(budgets,points,"-o",color=color,linewidth=2,markersize=5,label=name)
    random_ac=[]
    for budget in budgets:
        q=(t["tok"]-budget)/(t["tok"]-nt["tok"])
        if not 0<=q<=1:raise ValueError(f"Random reference budget invalid {budget}")
        random_ac.append(100*(q*nt["acc"]+(1-q)*t["acc"]))
    for r in all_stats:
        r["Random expected Acc (%)"]=random_ac[budgets.index(r["Test Token Budget"])]
    ax.plot(budgets,random_ac,"--",color="#9AA3AF",linewidth=1.7,label="Global Random (expected)")
    ax.scatter([nt["tok"],t["tok"]],[nt["acc"]*100,t["acc"]*100],
               s=140,marker="*",color=["#E05A47","#22996E"],zorder=5)
    ax.annotate("Always NoThinking",(nt["tok"],nt["acc"]*100),xytext=(8,-13),textcoords="offset points",fontsize=8)
    ax.annotate("Always Thinking",(t["tok"],t["acc"]*100),xytext=(-85,10),textcoords="offset points",fontsize=8)
    ax.set_xticks(budgets)
    ax.set_xlim(min(nt["tok"],budgets[0])-90,max(t["tok"],budgets[-1])+110)
    ax.set_xlabel("Matched mean generated tokens per test question",fontsize=11)
    ax.set_ylabel("Strict test accuracy (%)",fontsize=11)
    ax.set_title("Accuracy–Token Trade-off | Matched Test Budgets",fontsize=14,pad=16)
    ax.grid(alpha=.20)
    ax.legend(fontsize=9,loc="best",frameon=False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x:,.0f}"))
    fig.text(.5,.015,"Post-hoc test-cost interpolation (descriptive); no test accuracy used to choose a policy. Router overhead excluded.",
             ha="center",fontsize=8,color=".4")
    fig.tight_layout(rect=(0,.045,1,1))
    for suffix in ("png","svg"):
        fig.savefig(out/f"matched_test_budget_curve.{suffix}",dpi=250,bbox_inches="tight")
    plt.close(fig)
    write_csv(out/"matched_test_budget_curve.csv",all_stats)


def run(args):
    test=base.read_rows(args.test_labels)
    val=base.read_rows(args.val_labels)
    if len(test)!=485 or Counter(r["dataset"] for r in test)!={"gsm8k":330,"math":125,"aime":30}:
        raise ValueError("Canonical Test-485 not found")
    if len(val)!=450 or base.file_sha(args.val_labels)!=base.VAL_SHA:
        raise ValueError("Canonical Validation-450 hash mismatch")
    if any(r["n_generations"] != (16 if r["dataset"]=="aime" else 1) for r in test):
        raise ValueError("AIME must be aggregated over 16 generations per question")
    if {r["sample_id"] for r in val} & {r["sample_id"] for r in test}:
        raise ValueError("Validation/test ID overlap")
    scopes=split_rows(test)
    test_ids={r["sample_id"] for r in test}
    val_ids={r["sample_id"] for r in val}
    sha=base.file_sha(args.test_labels)
    test_scores={};val_scores={}
    for name,key,kind in base.METHODS:
        test_scores[name]=base.read_scores(getattr(args,key+"_test"),kind,test_ids,sha)
        val_scores[name]=base.read_scores(getattr(args,key+"_val"),kind,val_ids,base.VAL_SHA)
    args.out.mkdir(parents=True,exist_ok=True)
    draw_matched_budget_figure(scopes,test_scores,args.budgets,args.out)
    methods_val={};methods_matched={};audit=[]
    for name,_,_ in base.METHODS:
        policy=base.threshold_for_budget(val,val_scores[name],args.budget)
        methods_val[name]={ds:base.apply_threshold(rows,test_scores[name],policy["score_cutoff"])
                           for ds,rows in scopes.items()}
        methods_matched[name],detail=matched_budget(policy_curves(test,test_scores[name]),args.budget)
        audit.append({"method":name,"val_policy":policy,"test_matched_policy":detail,
                      "transfer_test_overall":methods_val[name]["overall"],
                      "descriptive_test_matched_overall":methods_matched[name]["overall"]})
    for tag,policies,note in (
        ("val",methods_val,f"Validation-calibrated score cutoff, val budget {args.budget}; unchanged on test"),
        ("testmatched",methods_matched,f"Descriptive global test-cost matching at {args.budget} tokens; not a deployable threshold")):
        prefix=args.out/f"thinkswitcher_style_{tag}_B{args.budget}"
        rows=paper_table_rows(scopes,policies)
        write_csv(prefix.with_suffix(".csv"),rows)
        write_paper_markdown(rows,prefix.with_suffix(".md"),note)
        write_tex(rows,prefix.with_suffix(".tex"),note)
        make_table_figure(rows,prefix,note)
    (args.out/f"matched_budget_B{args.budget}_policy_audit.json").write_text(
        json.dumps({"fixed_random_definition":"Random (x/y): x% Thinking, y% NoThinking",
                    "test_budget_descriptive":True,"validation_threshold_selected_without_test_labels":True,
                    "methods":audit},indent=2)+"\n",encoding="utf-8")
    print("MATCHED_TEST_BUDGET_DONE",args.out)
    print("TABLE_POLICY_VAL",args.budget,"TABLE_POLICY_TEST_MATCHED",args.budget)
    for x in audit:
        print(x["method"],"val-transfer",f'{x["transfer_test_overall"]["tok"]:.1f} tok',
              "test-matched",f'{x["descriptive_test_matched_overall"]["tok"]:.1f} tok')


def self_test():
    # All three dataset groups must be present; the global policy is identical.
    examples=((0.,1.,5.,20.),(1.,0.,10.,30.),(.5,.5,10.,50.),(.2,.8,10.,50.))
    rows=[{"sample_id":f"{ds}:{i}","dataset":ds,
           "p_n":pn,"p_t":pt,"t_n":tn,"t_t":tt}
          for ds in SCOPES for i,(pn,pt,tn,tt) in enumerate(examples)]
    scores={f"{ds}:{i}":v for ds in SCOPES
            for i,v in enumerate((.1,.9,.6,.2))}
    c=policy_curves(rows,scores)
    for b in [18.8,24.3]:
        points,meta=matched_budget(c,b)
        assert abs(points["overall"]["tok"]-b)<1e-7
        assert 0<=meta["upper_policy_mixture_weight"]<=1
        assert abs(np.mean([points[d]["tok"] for d in SCOPES])-b)<1e-7
    scopes=split_rows(rows)
    table=paper_table_rows(scopes,{k:from_fixed_probability(scopes,.5) for k,_,_ in base.METHODS})
    assert abs(table[3]["Pooled Tok."]-base.means(rows,dict.fromkeys(scores,.75))["tok"])<1e-9
    assert abs(table[4]["Pooled Tok."]-base.means(rows,dict.fromkeys(scores,.9))["tok"])<1e-9
    print("MATCHED_BUDGET_SELFTEST_PASS (global policy, cost interpolation, Random convention)")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--budget",type=int,default=1300,help="Main table operating point")
    p.add_argument("--budgets",type=int,nargs="+",default=list(BUDGETS))
    p.add_argument("--val-labels",type=Path,default=Path("results/val_450_labels.jsonl"))
    p.add_argument("--test-labels",type=Path,default=Path("results/test_485_labels.jsonl"))
    p.add_argument("--out",type=Path,default=Path("results/test_485_three_routers"))
    p.add_argument("--utility-val",type=Path,default=Path("outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/validation_predictions.jsonl"))
    p.add_argument("--p1-val",type=Path,default=Path("outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json"))
    p.add_argument("--cost-val",type=Path,default=Path("outputs/route_e/token_aware/full/validation_predictions.json"))
    p.add_argument("--utility-test",type=Path,default=Path("outputs/test_485/utility_predictions.jsonl"))
    p.add_argument("--p1-test",type=Path,default=Path("outputs/test_485/p1_predictions.json"))
    p.add_argument("--cost-test",type=Path,default=Path("outputs/test_485/cost_predictions.json"))
    args=p.parse_args()
    if args.self_test:self_test();return
    if sorted(set(args.budgets))!=args.budgets or args.budget not in args.budgets:
        p.error("Use strictly increasing unique --budgets including --budget")
    run(args)

if __name__=="__main__":main()