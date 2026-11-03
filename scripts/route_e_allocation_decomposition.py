#!/usr/bin/env python3
"""Experiment 1B: descriptive budget allocation vs within-dataset ranking.

Given Experiment 1's immutable strict validation comparison and original
450 strict labels, split each router's global budget-matched advantage as:

  accuracy(router) - accuracy(global random at budget B)
    = [accuracy(stratified random at router's measured per-group costs)
       - accuracy(global random at B)]
    + [accuracy(router)
       - accuracy(stratified random at router's per-group costs)]

The two terms are "cross-dataset allocation" and "within-dataset selection".
These are an EXACT accounting identity on the fixed validation sample, NOT
causal effects. The group-specific random baseline is defined only if each
observed group budget lies between its all-NT and all-T group mean costs.

This is an OFFLINE diagnostic and uses validation observed group costs to
construct counterfactual reference probabilities. Such group budgets must
never be used as online routing inputs or for selecting a test-set policy.

No model training, GPU, feature extraction, or new generation.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import route_e_frozen as base
from train_modernbert_delta import endpoints, random_at_budget, read_labels

ROOT = base.ROOT
SOURCE = ROOT / "results/route_e/cost_only/experiment1/comparison.json"
OUTPUT = ROOT / "results/route_e/cost_only/experiment1b"
KEYS = ("Delta_only", "Cost_only", "Delta_predicted_cost")
TOL = 1e-6


def strict_rows():
    if base.sha256(base.VAL) != base.VAL_SHA256:
        raise ValueError("Strict validation label SHA mismatch")
    rows = read_labels(base.VAL)
    if len(rows) != 450 or sum(r["dataset"]=="gsm8k" for r in rows)!=300 or (
            sum(r["dataset"]=="math" for r in rows)!=150):
        raise ValueError("Wrong strict validation composition")
    return rows


def decompose_one(rows, group_summary, budget, reference_global=None):
    """Return descriptive accounting components or an explicit failure reason."""
    if not math.isfinite(budget):
        raise ValueError("Nonfinite budget")
    all_modes = endpoints(rows)
    gr = random_at_budget(budget, all_modes)
    if gr is None:
        raise ValueError("Global budget outside the observed all-mode means")
    if reference_global is not None and abs(reference_global - gr["accuracy"])>TOL:
        raise ValueError("Global Random from experiment 1 disagrees with strict labels")
    names=sorted({r["dataset"] for r in rows})
    n=len(rows)
    projected=[]
    outside=[]
    for name in names:
        group=[r for r in rows if r["dataset"]==name]
        w=len(group)/n
        gm=group_summary[name]
        actual_cost=float(gm["actual_tokens"])
        actual_accuracy=float(gm["accuracy"])
        if not (math.isfinite(actual_cost) and math.isfinite(actual_accuracy)):
            raise ValueError("Nonfinite group evaluation metric")
        random_at_group_cost=random_at_budget(actual_cost,endpoints(group))
        endpoints_group=endpoints(group)
        item={"dataset":name,"n":len(group),"weight":w,
              "actual_mean_tokens":actual_cost,
              "actual_accuracy":actual_accuracy,
              "global_nt_share":float(gm["nothinking_fraction"]),
              "group_all_t_tokens":endpoints_group["thinking"]["tokens"],
              "group_all_n_tokens":endpoints_group["nothinking"]["tokens"]}
        if random_at_group_cost is None:
            item["group_cost_matched_random_feasible"]=False
            item["note"]="Actual subgroup cost is outside the randomized group endpoint interval; decomposition undefined"
            outside.append(name)
        else:
            a0=random_at_group_cost["accuracy"]
            item.update({
                "group_cost_matched_random_feasible":True,
                "random_nothinking_probability_at_same_group_cost":
                    random_at_group_cost["nt_probability"],
                "group_cost_matched_random_accuracy":a0,
                "within_group_accuracy_gain_pp":100*(actual_accuracy-a0),
                "weighted_within_group_gain_pp":100*w*(actual_accuracy-a0),
            })
        projected.append(item)
    overall=group_summary["overall"]
    recomputed_acc=sum(x["weight"]*x["actual_accuracy"] for x in projected)
    recomputed_budget=sum(x["weight"]*x["actual_mean_tokens"] for x in projected)
    if abs(recomputed_acc-overall["accuracy"])>TOL:
        raise ValueError("Group accuracy aggregation mismatch")
    if abs(recomputed_budget-budget)>TOL or abs(recomputed_budget-overall["actual_tokens"])>TOL:
        raise ValueError("Group cost aggregation mismatch")
    record={
        "feasible":not bool(outside), "budget":float(budget),
        "observed_router_accuracy":float(overall["accuracy"]),
        "global_random_accuracy":float(gr["accuracy"]),
        "overall_uplift_over_global_random_pp":
            100*(float(overall["accuracy"])-float(gr["accuracy"])),
        "groups":projected,
    }
    if outside:
        record["not_feasible_groups"]=outside
        return record
    strat=sum(x["weight"]*x["group_cost_matched_random_accuracy"] for x in projected)
    cross=100*(strat-gr["accuracy"])
    within=100*(overall["accuracy"]-strat)
    record.update({
        "stratified_group_random_accuracy":float(strat),
        "allocation_contribution_pp":float(cross),
        "within_group_selection_contribution_pp":float(within),
        "sum_check_pp":float(cross+within),
    })
    if abs(cross+within-record["overall_uplift_over_global_random_pp"])>TOL:
        raise AssertionError("Decomposition identity failed")
    return record


def decompose_table(rows, experiment_one):
    if (experiment_one["data_lineage"]["strict_val_sha256"] != base.VAL_SHA256 or
            experiment_one["data_lineage"]["strict_train_sha256"]!=base.TRAIN_SHA256 or
            not experiment_one.get("original_p2_b_d_dense_reproduced") or
            not experiment_one.get("dense_cost_matching_uses_observed_validation_lengths") or
            experiment_one.get("ranking_uses_observed_validation_lengths")):
        raise ValueError("Experiment 1 lineage/protocol is not supported")
    table=[]
    for item in experiment_one["budget_comparison"]:
        budget=float(item["budget"])
        row={"budget":budget,"methods":{}}
        reference=item["methods"].get("Global_Random")
        if reference and reference.get("available",True):
            ref_acc=float(reference["groups"]["overall"]["accuracy"])
        else:
            ref_acc=None
        for name in KEYS:
            if name not in item["methods"]:
                raise ValueError(f"Missing {name} from experiment 1")
            row["methods"][name]=decompose_one(
                rows,item["methods"][name]["groups"],budget,ref_acc)
        E=row["methods"]["Cost_only"]
        B=row["methods"]["Delta_only"]
        D=row["methods"]["Delta_predicted_cost"]
        row["E_minus_B_accuracy_pp"] = 100*(E["observed_router_accuracy"] -
                                             B["observed_router_accuracy"])
        row["D_minus_E_accuracy_pp"] = 100*(D["observed_router_accuracy"] -
                                             E["observed_router_accuracy"])
        if E["feasible"] and B["feasible"]:
            delta_cross = E["allocation_contribution_pp"]-B["allocation_contribution_pp"]
            delta_within=E["within_group_selection_contribution_pp"]-B["within_group_selection_contribution_pp"]
            if abs(delta_cross+delta_within-row["E_minus_B_accuracy_pp"])>TOL:
                raise AssertionError("E-B decomposition identity failed")
            row["E_minus_B_allocation_component_pp"]=delta_cross
            row["E_minus_B_within_group_component_pp"]=delta_within
        if D["feasible"] and E["feasible"]:
            delta_cross=D["allocation_contribution_pp"]-E["allocation_contribution_pp"]
            delta_within=D["within_group_selection_contribution_pp"]-E["within_group_selection_contribution_pp"]
            if abs(delta_cross+delta_within-row["D_minus_E_accuracy_pp"])>TOL:
                raise AssertionError("D-E decomposition identity failed")
            row["D_minus_E_allocation_component_pp"]=delta_cross
            row["D_minus_E_within_group_component_pp"]=delta_within
        table.append(row)
    return table


def export_csv(folder, table):
    with (folder/"decomposition.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["budget","method","router_accuracy_pct","global_random_accuracy_pct",
                "stratified_random_accuracy_pct","total_uplift_pp",
                "allocation_contribution_pp","within_group_selection_pp","feasible"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for record in table:
            for method,metric in record["methods"].items():
                w.writerow({
                    "budget":record["budget"],"method":method,
                    "router_accuracy_pct":100*metric["observed_router_accuracy"],
                    "global_random_accuracy_pct":100*metric["global_random_accuracy"],
                    "stratified_random_accuracy_pct":(
                        100*metric["stratified_group_random_accuracy"] if metric["feasible"] else ""),
                    "total_uplift_pp":metric["overall_uplift_over_global_random_pp"],
                    "allocation_contribution_pp":metric.get("allocation_contribution_pp",""),
                    "within_group_selection_pp":metric.get("within_group_selection_contribution_pp",""),
                    "feasible":metric["feasible"]})
    with (folder/"subgroup_random.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["budget","method","dataset","n","mean_tokens","accuracy_pct",
                "nt_fraction","cost_matched_random_accuracy_pct","within_group_gain_pp",
                "group_cost_matched_random_feasible"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for record in table:
            for method,metric in record["methods"].items():
                for group in metric["groups"]:
                    w.writerow({
                        "budget":record["budget"],"method":method,"dataset":group["dataset"],
                        "n":group["n"],"mean_tokens":group["actual_mean_tokens"],
                        "accuracy_pct":100*group["actual_accuracy"],
                        "nt_fraction":group["global_nt_share"],
                        "cost_matched_random_accuracy_pct":(
                            100*group["group_cost_matched_random_accuracy"] if
                            group["group_cost_matched_random_feasible"] else ""),
                        "within_group_gain_pp":group.get("within_group_accuracy_gain_pp",""),
                        "group_cost_matched_random_feasible":
                            group["group_cost_matched_random_feasible"],
                    })


def run(args):
    folder=Path(args.out_dir)
    if (folder/"decomposition.json").exists() and not args.overwrite:
        raise FileExistsError(f"Existing 1B report: {folder}; use --overwrite")
    rows=strict_rows()
    p=Path(args.source)
    e=json.loads(p.read_text(encoding="utf-8"))
    table=decompose_table(rows,e)
    report={
        "title":"Route E P2 Experiment 1B: offline allocation-vs-within decomposition",
        "strict_val_sha256":base.VAL_SHA256,
        "source_experiment1":str(p),
        "interpretation":"Exact algebraic accounting on fixed strict validation sample, "
                         "not a causal estimate, significance test, or online policy.",
        "no_new_training":True,"no_new_generation":True,
        "observed_subgroup_budget_used_only_posthoc":True,
        "method":"Group random at each method's observed group budget, "
                 "weighted by actual group size, compared with Global Random "
                 "at same whole-validation budget.",
        "table":table,
    }
    folder.mkdir(parents=True,exist_ok=True)
    (folder/"decomposition.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    export_csv(folder,table)
    print("COST_ALLOC_DECOMPOSITION_PASS exact uplift accounting",flush=True)
    for row in table:
        b=row["budget"]
        e=row["methods"]["Cost_only"]
        if not e["feasible"]:
            print(f"COST_ALLOC budget={b:.0f} E=NA invalid_groups={e['not_feasible_groups']}",
                  flush=True)
        else:
            print(f"COST_ALLOC budget={b:.0f} "
                  f"E_total={e['overall_uplift_over_global_random_pp']:+.2f}pp "
                  f"E_cross={e['allocation_contribution_pp']:+.2f}pp "
                  f"E_within={e['within_group_selection_contribution_pp']:+.2f}pp",
                  flush=True)
        if "E_minus_B_allocation_component_pp" in row:
            print(f"COST_ALLOC_E_MINUS_B budget={b:.0f} "
                  f"total={row['E_minus_B_accuracy_pp']:+.2f}pp "
                  f"cross={row['E_minus_B_allocation_component_pp']:+.2f}pp "
                  f"within={row['E_minus_B_within_group_component_pp']:+.2f}pp",
                  flush=True)
    print(f"COST_ALLOC_DONE report={folder/'decomposition.json'}",flush=True)


def self_test():
    # Difference in per-group budgets plus positive/negative internal selection
    # must add exactly to the total uplift over overall Global Random.
    rows=[
        {"sample_id":"gsm8k:1","dataset":"gsm8k","p_t":1.,"p_n":.5,"t_t":100.,"t_n":20.},
        {"sample_id":"gsm8k:2","dataset":"gsm8k","p_t":.9,"p_n":.2,"t_t":150.,"t_n":40.},
        {"sample_id":"math:1","dataset":"math","p_t":.8,"p_n":.2,"t_t":300.,"t_n":100.},
        {"sample_id":"math:2","dataset":"math","p_t":.7,"p_n":.3,"t_t":500.,"t_n":150.},
    ]
    probs=[0.3,0.4,0.5,0.6]
    import numpy as np
    g=p2_groups=None
    from route_e_token_aware import group_metrics
    g=group_metrics(rows,np.asarray(probs))
    budget=g["overall"]["actual_tokens"]
    z=decompose_one(rows,g,budget)
    assert z["feasible"]
    assert abs(z["allocation_contribution_pp"]+
               z["within_group_selection_contribution_pp"]-
               z["overall_uplift_over_global_random_pp"])<TOL
    wrong={**g,"overall":{**g["overall"],"actual_tokens":budget+10}}
    try:
        decompose_one(rows,wrong,budget)
    except ValueError as exc:
        assert "aggregation mismatch" in str(exc)
    else:
        raise AssertionError("Inconsistent group aggregation accepted")
    print("COST_ALLOC_SELFTEST_PASS (accounting identity, group budget guard)",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    runp=sub.add_parser("run")
    runp.add_argument("--source",type=Path,default=SOURCE)
    runp.add_argument("--out-dir",type=Path,default=OUTPUT)
    runp.add_argument("--overwrite",action="store_true")
    args=p.parse_args()
    self_test() if args.command=="self-test" else run(args)


if __name__=="__main__":
    main()
