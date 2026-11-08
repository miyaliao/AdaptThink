#!/usr/bin/env python3
"""Compare Utility-Original / Utility-Hard / Utility-Soft by identical lambda.

All nAUC metrics must come from the exact same strict-val-450, 451-policy
cost-sorted integration convention. No new training, no test data.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from compute_nauc_ac import auc_cost_sorted

PATHS={
  "modernbert": ROOT/"results/route_e/modernbert_utility_lambda_sweep/nauc_ac_lambda_sweep.csv",
  "p1": ROOT/"results/route_e/p1_utility_lambda_sweep/nauc_ac_lambda_sweep.csv",
}


def table(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def collect(encoder,hardsoft,original):
    old=table(hardsoft)
    fresh=table(original)
    if not fresh or not old:
        raise ValueError("Empty input")
    store={}
    for r in old:
        if r["objective"] not in ("utility_hard","utility_soft"):
            continue
        key=(r["objective"],float(r["lambda"]))
        if key in store:
            raise ValueError("Duplicate historical objective / lambda")
        store[key]=float(r["nAUC_AC"])
    for r in fresh:
        if r["objective"]!="utility_original":
            continue
        key=("utility_original",float(r["lambda"]))
        if key in store:
            raise ValueError("Duplicate new Original objective / lambda")
        store[key]=float(r["nAUC_AC"])
    lambdas=sorted({lam for obj,lam in store if obj=="utility_original"})
    if not lambdas:
        raise ValueError("No Utility-Original rows")
    rows=[]
    for lam in lambdas:
        needed=("utility_original","utility_hard","utility_soft")
        if any((obj,lam) not in store for obj in needed):
            raise ValueError(f"Missing objective at lambda={lam}")
        r={"lambda":lam,
           "utility_original_nAUC_AC":store["utility_original",lam],
           "utility_hard_nAUC_AC":store["utility_hard",lam],
           "utility_soft_nAUC_AC":store["utility_soft",lam]}
        r["best_validation_objective"]=max(needed,key=lambda x:r[x+"_nAUC_AC"])
        rows.append(r)
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--encoder",choices=("modernbert","p1"),required=True)
    p.add_argument("--hard-soft-csv",type=Path,default=None)
    p.add_argument("--original-csv",type=Path,default=None)
    p.add_argument("--no-plot",action="store_true")
    args=p.parse_args()
    hardsoft=args.hard_soft_csv or PATHS[args.encoder]
    original=args.original_csv or (ROOT/"results/route_a/utility_original"/args.encoder/"nauc_ac_lambda_sweep.csv")
    if not hardsoft.is_file() or not original.is_file():
        raise FileNotFoundError(f"Missing results: old={hardsoft} original={original}")
    rows=collect(args.encoder,hardsoft,original)
    out=original.parent
    dest=out/"three_utility_objectives_nauc.csv"
    with dest.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("lambda | Utility-Original | Utility-Hard | Utility-Soft")
    for r in rows:
        print(f"{r['lambda']:.3f} | {r['utility_original_nAUC_AC']:+.4f} | "
              f"{r['utility_hard_nAUC_AC']:+.4f} | {r['utility_soft_nAUC_AC']:+.4f}")
    print(f"UTILITY_ORIGINAL_COMPARISON_DONE {dest}")
    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("THREE_OBJECTIVE_PLOT_SKIPPED: matplotlib unavailable")
        return
    fig,ax=plt.subplots(figsize=(8,4.8))
    for obj,marker in (("utility_original","o"),("utility_hard","s"),("utility_soft","^")):
        ax.plot([r["lambda"] for r in rows],
                [r[obj+"_nAUC_AC"] for r in rows],
                marker=marker,label=obj)
    ax.set(title=f"{args.encoder}: Utility Learning Objective comparison (validation)",
           xlabel="lambda",ylabel="nAUC-AC")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    for suffix in ("png","svg"):
        fig.savefig(out/f"three_utility_objectives_nauc.{suffix}",dpi=200,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
