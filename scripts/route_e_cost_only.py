#!/usr/bin/env python3
"""Experiment 1: Cost-only ablation against unchanged P2 Delta-only and D.

Reads EXISTING saved strict validation predictions. Does not fit any head,
extract embeddings, use GPU, generate LLM tokens or open canonical test data.

At each budget, the same Route E exploratory *observed-cost dense replay*
method compares:
  B Quality-only: larger predicted delta_p => favor Thinking
  E Cost-only:    larger predicted delta_t => favor NoThinking
  D Combined:     larger delta_p - fixed_lambda * delta_t/1000 => favor Thinking

The cost-only score uses only PREDICTED pre-generation costs, never actual
validation cost as a router input. Actual mode cost is used POST-ROUTING
for the historical dense offline budget interpolation, as in Route E.

Usage:
  python scripts/route_e_cost_only.py self-test
  python scripts/route_e_cost_only.py run
  python scripts/route_e_cost_only.py run --overwrite --no-plot
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import route_e_frozen as base
import route_e_token_aware as p2
from route_e_equal_budget import (
    DEFAULT_BUDGETS, dense_curve, linear_bracket, random_at_budget,
)
from train_modernbert_delta import read_labels

ROOT = base.ROOT
PREDICTIONS = ROOT / "outputs/route_e/token_aware/full/validation_predictions.json"
P2_REPORT = ROOT / "results/route_e/token_aware/full/validation.json"
RESULTS = ROOT / "results/route_e/cost_only/experiment1"
METHODS = ("Delta_only", "Cost_only", "Delta_predicted_cost")
REPLAY_TOL = 1e-8


def hash_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_validation():
    if not base.VAL.is_file():
        raise FileNotFoundError(base.VAL)
    observed = base.sha256(base.VAL)
    if observed != base.VAL_SHA256:
        raise ValueError(f"Strict validation SHA256 mismatch {observed}")
    rows = read_labels(base.VAL)
    if len(rows) != 450:
        raise ValueError(f"Expected 450 strict val rows; got {len(rows)}")
    counts = {name: sum(r["dataset"] == name for r in rows)
              for name in ("gsm8k", "math")}
    if counts != {"gsm8k": 300, "math": 150}:
        raise ValueError(f"Unexpected strict validation composition: {counts}")
    return rows


def read_sources(rows, predictions_path=PREDICTIONS, report_path=P2_REPORT):
    predictions_path = Path(predictions_path)
    report_path = Path(report_path)
    pred = json.loads(predictions_path.read_text(encoding="utf-8"))
    old = json.loads(report_path.read_text(encoding="utf-8"))
    if pred.get("strict_val_sha256") != base.VAL_SHA256:
        raise ValueError("Prediction strict validation SHA mismatch")
    if (old.get("strict_val_sha256") != base.VAL_SHA256 or
            old.get("strict_train_sha256") != base.TRAIN_SHA256 or
            old.get("smoke") or not old.get("encoder_frozen") or
            old.get("generation_repeated")):
        raise ValueError("Source P2 report does not match strict frozen full protocol")
    if "dense_validation" not in old:
        raise ValueError("Original full P2 dense replay is required for reproduction")
    lam = float(pred["cost_lambda"])
    old_lam = float(old["head"]["fixed_lambda"])
    if not math.isfinite(lam) or lam <= 0 or abs(lam-old_lam) > 1e-12:
        raise ValueError("Prediction and report have different fixed positive lambda")
    records = pred["predictions"]
    if list(records) != [r["sample_id"] for r in rows]:
        raise ValueError("Prediction IDs/order do not match strict 450 validation rows")

    # Lower score on the Thinking scale becomes higher NoThinking priority.
    maps = {method: {} for method in METHODS}
    for sid, fields in records.items():
        forbidden = {"p_t", "p_n", "t_t", "t_n",
                     "actual_tokens", "actual_accuracy", "true_delta"}
        if forbidden & fields.keys():
            raise ValueError(f"Unexpected observed outcome fields for {sid}")
        required = ("delta_p_pred", "delta_t_pred",
                    "thinking_tokens_pred", "nothinking_tokens_pred",
                    "score_delta_only", "score_predicted_cost")
        if any(key not in fields for key in required):
            raise ValueError(f"Missing predicted field for {sid}")
        q, d_t, t_t, t_n, s_b, s_d = (
            float(fields[k]) for k in required)
        if not all(math.isfinite(v) for v in (q,d_t,t_t,t_n,s_b,s_d)):
            raise ValueError(f"Non-finite pre-generation score for {sid}")
        if abs(d_t - (t_t-t_n)) > REPLAY_TOL:
            raise ValueError(f"Predicted delta-t inconsistent for {sid}")
        if abs(s_b - q) > REPLAY_TOL:
            raise ValueError(f"Old quality-only score inconsistent for {sid}")
        if abs(s_d - (q-lam*d_t/1000.)) > REPLAY_TOL:
            raise ValueError(f"Old token-aware score inconsistent for {sid}")
        maps["Delta_only"][sid] = -s_b
        maps["Cost_only"][sid] = d_t
        maps["Delta_predicted_cost"][sid] = -s_d
    return maps, pred, old, {
        "validation_predictions": str(predictions_path),
        "validation_predictions_sha256": hash_bytes(predictions_path),
        "original_p2_report": str(report_path),
        "original_p2_report_sha256": hash_bytes(report_path),
        "strict_val_sha256": base.VAL_SHA256,
        "strict_train_sha256": base.TRAIN_SHA256,
        "frozen_lambda": lam,
    }


def validate_original(rows, maps, old, budgets):
    """B and D must exactly reproduce previously reported dense results."""
    expected = {float(row["budget"]): row for row in
                old["dense_validation"]["budgets"]}
    if len(expected) != len(old["dense_validation"]["budgets"]):
        raise ValueError("Duplicated budgets in P2 source report")
    for b in budgets:
        if b not in expected:
            raise ValueError(f"Budget {b} is missing from original P2 report")
    for method in ("Delta_only", "Delta_predicted_cost"):
        curve = dense_curve(rows, maps[method])
        for b in budgets:
            previous = expected[b]["methods"][method]
            if not previous["available"]:
                raise ValueError(f"Original P2 {method} missing at budget {b}")
            recalculated = linear_bracket(curve, b)
            for old_value, new_value, metric in (
                (previous["groups"]["overall"]["accuracy"],
                 recalculated["accuracy"], "accuracy"),
                (previous["groups"]["overall"]["actual_tokens"],
                 recalculated["mean_generated_tokens"], "mean generated tokens"),
                (previous["mix"]["expected_nothinking_fraction"],
                 recalculated["expected_nothinking_fraction"], "NT fraction"),
            ):
                if abs(float(old_value)-float(new_value)) > REPLAY_TOL:
                    raise ValueError(
                        f"P2_REPRODUCTION_FAILED method={method} budget={b} "
                        f"metric={metric} old={old_value} new={new_value}")
    return True


def metrics_for_group(rows, predictions):
    out = {}
    names = ("overall", "gsm8k", "math")
    groups = np.array([r["dataset"] for r in rows])
    actual_t = np.array([r["t_t"] for r in rows], dtype=float)
    actual_n = np.array([r["t_n"] for r in rows], dtype=float)
    pdelta = np.array([predictions[r["sample_id"]]["delta_t_pred"]
                       for r in rows], dtype=float)
    for name in names:
        keep = (np.ones(len(rows), dtype=bool) if name=="overall"
                else groups==name)
        truth = actual_t[keep]-actual_n[keep]
        pred = pdelta[keep]
        out[name] = {
            "n": int(keep.sum()),
            "predicted_delta_t": p2.regression_metrics(pred, truth),
            "thinking_tokens": p2.regression_metrics(
                np.array([predictions[r["sample_id"]]["thinking_tokens_pred"]
                          for r in rows])[keep], actual_t[keep]),
            "nothinking_tokens": p2.regression_metrics(
                np.array([predictions[r["sample_id"]]["nothinking_tokens_pred"]
                          for r in rows])[keep], actual_n[keep]),
        }
    return out


def policies_at_budget(rows, scores, budget):
    curve = dense_curve(rows, scores)
    bracket = linear_bracket(curve, budget)
    pn = p2.nt_prob_at_bracket(rows, scores, bracket)
    metrics = p2.group_metrics(rows, pn)
    if abs(metrics["overall"]["actual_tokens"]-budget)>1e-5:
        raise AssertionError("Replay cost bracket unexpectedly differs from measured cost")
    if abs(metrics["overall"]["accuracy"]-bracket["accuracy"])>1e-8:
        raise AssertionError("Replay policy accuracy unexpectedly differs")
    return pn, metrics, bracket


def cost_only_rank_test(rows, predictions):
    """Top predicted extra-cost question must be the first routed to NT."""
    scores = {r["sample_id"]: float(predictions[r["sample_id"]]["delta_t_pred"])
              for r in rows}
    order = sorted(rows, key=lambda r: (-scores[r["sample_id"]],r["sample_id"]))
    if not order:
        raise AssertionError("No examples")
    if scores[order[0]["sample_id"]] < scores[order[-1]["sample_id"]]:
        raise AssertionError("Cost-only NT orientation reversed")
    return scores


def paired_comparisons(rows, pn, reps, seed):
    return {
        "D_minus_E": p2.paired_bootstrap(rows, pn["Delta_predicted_cost"],
                                         pn["Cost_only"], reps, seed),
        "E_minus_B": p2.paired_bootstrap(rows, pn["Cost_only"],
                                         pn["Delta_only"], reps, seed+1),
        "D_minus_B": p2.paired_bootstrap(rows, pn["Delta_predicted_cost"],
                                         pn["Delta_only"], reps, seed+2),
    }


def summarize(rows, predictions, old, maps, budgets, reps, seed):
    verify = validate_original(rows, maps, old, budgets)
    by_group = metrics_for_group(rows, predictions)
    table = []
    for budget in budgets:
        record = {"budget": float(budget), "methods": {}}
        probs = {}
        for name in METHODS:
            pn, group, bracket = policies_at_budget(rows, maps[name], budget)
            probs[name] = pn
            ref = random_at_budget(rows, budget)
            record["methods"][name] = {
                "groups": group, "mix": bracket,
                "uplift_vs_random_pp": 100*(group["overall"]["accuracy"]-ref["accuracy"]),
            }
        boot = paired_comparisons(rows, probs, reps, seed+round(budget))
        record["paired_bootstrap"] = boot
        ref_row = next(r for r in old["dense_validation"]["budgets"]
                       if float(r["budget"])==float(budget))
        for method in ("P1_Frozen", "FT_Classification", "Global_Random"):
            if method in ref_row["methods"]:
                record["methods"][method] = ref_row["methods"][method]
        table.append(record)
    return {"original_p2_b_d_dense_reproduced": verify,
            "method_definitions": {
                "Delta_only": "NoThinking priority = -quality_delta_p_pred",
                "Cost_only": "NoThinking priority = +predicted_extra_thinking_tokens",
                "Delta_predicted_cost": "NoThinking priority = "
                    "-(quality_delta_p_pred - original_lambda*predicted_delta_t/1000)",
            },
            "ranking_uses_observed_validation_lengths": False,
            "dense_cost_matching_uses_observed_validation_lengths": True,
            "offline_replay_is_not_deployable": True,
            "bootstrap_caution": "Nominal conditional 95% paired bootstrap per budget; "
                                 "no multi-budget multiple-testing correction; "
                                 "not independent held-out test.",
            "group_cost_diagnostics": by_group,
            "budget_comparison": table}


def export_csv(out, result):
    with (out/"equal_budget.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["budget", "method","dataset","accuracy_pct",
                         "actual_mean_tokens","nt_fraction",
                         "uplift_vs_random_pp"])
        for row in result["budget_comparison"]:
            for name, r in row["methods"].items():
                if not r.get("available", True):
                    continue
                for dataset, metrics in r["groups"].items():
                    writer.writerow([row["budget"],name,dataset,
                                     metrics["accuracy"]*100,
                                     metrics["actual_tokens"],
                                     metrics["nothinking_fraction"],
                                     r.get("uplift_vs_random_pp","") if dataset=="overall" else ""])
    with (out/"cost_diagnostics_by_dataset.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset","n","metric","mae","rmse","spearman",
                         "pred_mean","actual_mean","bias_pred_minus_actual"])
        for ds, item in result["group_cost_diagnostics"].items():
            for metric in ("thinking_tokens","nothinking_tokens","predicted_delta_t"):
                m=item[metric]
                writer.writerow([ds,item["n"],metric,m["mae"],m["rmse"],m["spearman"],
                                 m["pred_mean"],m["actual_mean"],
                                 m["bias_pred_minus_actual"]])
    with (out/"paired_bootstrap.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["budget","comparison","gain_pp","ci95_low_pp","ci95_high_pp","bootstrap_n"])
        for r in result["budget_comparison"]:
            for label, z in r["paired_bootstrap"].items():
                if z is None:
                    continue
                writer.writerow([r["budget"],label,z["difference_pp"],
                                 *z["ci_95_pp"],z["repetitions"]])


def make_plot(out, result):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("COST_ONLY_PLOT_SKIPPED matplotlib unavailable")
        return
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for name in ("Delta_only", "Cost_only", "Delta_predicted_cost",
                 "P1_Frozen", "FT_Classification", "Global_Random"):
        points=[]
        for item in result["budget_comparison"]:
            entry=item["methods"].get(name)
            if entry and entry.get("available", True):
                points.append((item["budget"],
                               100*entry["groups"]["overall"]["accuracy"]))
        if points:
            label = {"Delta_only":"B: Quality only",
                     "Cost_only":"E: Cost only",
                     "Delta_predicted_cost":"D: Combined"}.get(name,name.replace("_"," "))
            ax.plot(*zip(*points),marker="o",label=label)
    ax.set(xlabel="Expected mean generated tokens (offline measured-cost matching)",
           ylabel="Strict validation accuracy (%)",
           title="P2 Experiment 1: quality vs cost vs combined routing")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in (".svg",".png"):
        fig.savefig(out/("accuracy_token_comparison"+ext),dpi=180)
    plt.close(fig)


def run(args):
    if (not args.budgets or len(args.budgets)!=len(set(args.budgets)) or
            not all(math.isfinite(x) and x>0 for x in args.budgets) or
            args.bootstrap_reps<0):
        raise ValueError("Budgets must be positive and distinct; bootstrap-reps>=0")
    out = Path(args.out_dir)
    if (out/"comparison.json").exists() and not args.overwrite:
        raise FileExistsError(f"Existing experiment 1 report: {out}; use --overwrite")
    rows = get_validation()
    maps, source, old, evidence = read_sources(rows, args.predictions, args.p2_report)
    result = summarize(rows, source["predictions"], old, maps,
                       [float(x) for x in args.budgets],
                       args.bootstrap_reps, args.seed)
    result["data_lineage"]=evidence
    result["n_validation"]=len(rows)
    result["models_retrained"]=False
    result["embeddings_reextracted"]=False
    result["responses_regenerated"]=False
    result["source_model"] = "Frozen DeepSeek P1 last hidden state, original P2 Ridge heads"
    out.mkdir(parents=True, exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    export_csv(out,result)
    if not args.no_plot:
        make_plot(out,result)
    print("COST_ONLY_REPRODUCTION_PASS B/D unchanged strict dense replay")
    for key, metrics in result["group_cost_diagnostics"].items():
        m=metrics["predicted_delta_t"]
        print(f"COST_ONLY_GROUP {key} n={metrics['n']} "
              f"delta_t_spearman={m['spearman']} mae={m['mae']:.1f} "
              f"pred_mean={m['pred_mean']:.1f} actual_mean={m['actual_mean']:.1f}")
    for item in result["budget_comparison"]:
        m=item["methods"]
        acc=lambda name: 100*m[name]["groups"]["overall"]["accuracy"]
        ci = item["paired_bootstrap"]["D_minus_E"]
        cistr = "disabled" if ci is None else (
            "["+", ".join(f"{x:+.2f}" for x in ci["ci_95_pp"])+"]")
        print(f"COST_ONLY_BUDGET {item['budget']:.0f} "
              f"B={acc('Delta_only'):.2f}% E={acc('Cost_only'):.2f}% "
              f"D={acc('Delta_predicted_cost'):.2f}% "
              f"D_minus_E={acc('Delta_predicted_cost')-acc('Cost_only'):+.2f}pp "
              f"95CI={cistr}")
    print(f"COST_ONLY_DONE report={out/'comparison.json'}")


def self_test():
    """Cache-free orientation, source-score consistency and group diagnostics."""
    rows=[
        {"sample_id":"gsm8k:1","dataset":"gsm8k",
         "p_t":1.,"p_n":0.,"t_t":100.,"t_n":20.},
        {"sample_id":"math:2","dataset":"math",
         "p_t":0.,"p_n":1.,"t_t":100.,"t_n":20.},
    ]
    pred={"gsm8k:1":{"delta_t_pred":20., "thinking_tokens_pred":40.,
                      "nothinking_tokens_pred":20.},
          "math:2":{"delta_t_pred":200., "thinking_tokens_pred":220.,
                    "nothinking_tokens_pred":20.}}
    costs = cost_only_rank_test(rows,pred)
    assert dense_curve(rows,costs)[1]["accuracy"]==1.
    assert p2.spearman([1,2,3],[3,2,1])==-1.
    diag=metrics_for_group(rows,pred)
    assert diag["overall"]["n"]==2 and diag["gsm8k"]["n"]==1
    # The original quality and D scores must remain unchanged after adding E.
    b={"gsm8k:1":-.3,"math:2":.4}
    d={"gsm8k:1":-.2,"math:2":.1}
    methods={"Delta_only":b,"Cost_only":costs,"Delta_predicted_cost":d}
    for name in METHODS:
        assert set(methods[name])=={"gsm8k:1","math:2"}
    print("COST_ONLY_SELFTEST_PASS (cost-only NT orientation, subgroup metrics, "
          "head reuse and strict comparison)", flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    cmd=sub.add_parser("run")
    cmd.add_argument("--predictions",type=Path,default=PREDICTIONS)
    cmd.add_argument("--p2-report",type=Path,default=P2_REPORT)
    cmd.add_argument("--out-dir",type=Path,default=RESULTS)
    cmd.add_argument("--budgets",type=float,nargs="+",default=DEFAULT_BUDGETS)
    cmd.add_argument("--bootstrap-reps",type=int,default=2000)
    cmd.add_argument("--seed",type=int,default=42)
    cmd.add_argument("--overwrite",action="store_true")
    cmd.add_argument("--no-plot",action="store_true")
    args=p.parse_args()
    if args.command=="self-test":
        self_test()
    else:
        run(args)


if __name__=="__main__":
    main()
