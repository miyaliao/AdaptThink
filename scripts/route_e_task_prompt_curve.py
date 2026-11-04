#!/usr/bin/env python3
"""Compare task-aware DeepSeek prompt variants over a FULL accuracy-token curve.

CPU-only offline strict replay. Use original 450 validation IDs, cached per-mode
outcomes and per-question router scores. Requires verified DENSE ModernBERT FT
predictions; no six-anchor approximation, no generation, no model weights.

Examples:
  python scripts/route_e_task_prompt_curve.py --self-test
  python scripts/route_e_task_prompt_curve.py
  python scripts/route_e_task_prompt_curve.py --budgets 900 1100 1300 1500 1700 1900 2100

Outputs: JSON, CSV, SVG, PNG in results/route_e/task_prompt/curve/.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import route_e_equal_budget as eq
import route_e_frozen as base
import route_e_task_prompts as prompts

ROOT = base.ROOT
METHOD_PATHS = {
    "P1_frozen_deepseek": (
        ROOT / "outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json"),
    "General_aware": prompts.cache_dir("general", False) / "validation_p_nt.json",
    "Difficulty_aware": prompts.cache_dir("difficulty", False) / "validation_p_nt.json",
    "Routing_aware": prompts.cache_dir("routing", False) / "validation_p_nt.json",
}
ORDER = ("Global_Random", "FT_Classification", *METHOD_PATHS)
LABELS = {
    "Global_Random": "Global Random",
    "FT_Classification": "ModernBERT Full FT",
    "P1_frozen_deepseek": "Frozen DeepSeek P1",
    "General_aware": "General-aware",
    "Difficulty_aware": "Difficulty-aware",
    "Routing_aware": "Routing-aware",
}


def scores_to_curves(rows, args):
    ids = [r["sample_id"] for r in rows]
    for name, path in METHOD_PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} predictions: {path}")
    if not args.classification_scores.is_file():
        raise FileNotFoundError(f"Missing dense FT scores: {args.classification_scores}")
    curves = {name: eq.dense_curve(rows, eq.load_scores(path, ids))
              for name,path in METHOD_PATHS.items()}
    ft = eq.load_scores(args.classification_scores, ids,
                        args.classification_score_direction,
                        args.classification_score_field)
    curves["FT_Classification"] = eq.dense_curve(rows, ft)
    verify = eq.verify_classification_50pct(curves["FT_Classification"])
    return curves, verify


def verify_p1_50pct(curve, tol_acc_pp=0.03, tol_tokens=2.0):
    """Ensure the stored baseline exactly reproduces the historical P1 arm."""
    if len(curve) != 451:
        raise RuntimeError("P1 needs exactly 451 top-k policies")
    point = curve[225]
    acc_err_pp = 100*(point["accuracy"] - 344/450)
    token_err = point["tokens"] - 1427.05
    if abs(acc_err_pp)>tol_acc_pp or abs(token_err)>tol_tokens:
        raise RuntimeError(
            f"P1_50PCT_REPRO_FAIL: accuracy={100*point['accuracy']:.4f}% "
            f"tokens={point['tokens']:.2f}; review IDs and score orientation")
    print(f"P1_50PCT_REPRO_PASS accuracy={100*point['accuracy']:.4f}% "
          f"tokens={point['tokens']:.2f}")
    return {"accuracy":point["accuracy"],"mean_generated_tokens":point["tokens"],
            "error_accuracy_pp":acc_err_pp,"error_tokens":token_err}


def sample_at_budget(rows, curves, budget):
    r = eq.random_at_budget(rows, budget)
    d = {"budget": float(budget), "Global_Random": float(r["accuracy"])}
    for name, curve in curves.items():
        d[name] = float(eq.linear_bracket(curve, budget)["accuracy"])
    return d


def trapezoid_mean_accuracy(sample_points, lo, hi):
    """Mean expected accuracy across a predeclared shared budget interval.

    NOT binary-label ROC AUC. Normalization makes the value an interpretable
    average accuracy over the selected [lo,hi] generated-token interval.
    """
    if hi <= lo:
        raise ValueError("Budget integration interval must be nonempty")
    names = [name for name in ORDER if name in sample_points[0]]
    integrated = {}
    for name in names:
        area = sum((b["budget"]-a["budget"])*
                   (a[name]+b[name])/2 for a,b in zip(sample_points, sample_points[1:]))
        integrated[name] = area/(hi-lo)
    return integrated


def write_plot(samples, path_png, path_svg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    xx = [r["budget"] for r in samples]
    for name in ORDER:
        if name == "Global_Random":
            kw = {"linestyle": "--", "linewidth": 1.7}
        elif name == "P1_frozen_deepseek":
            kw = {"linewidth": 2.6}
        else:
            kw = {"linewidth": 1.7}
        ax.plot(xx, [100*r[name] for r in samples],
                label=LABELS[name], **kw)
    ax.set_xlabel("Mean generated tokens per question (router cost excluded)")
    ax.set_ylabel("Expected strict accuracy (%)")
    ax.set_title("Prompt-aware Representation: Accuracy–Token Cost")
    ax.grid(True, alpha=.25)
    ax.legend(loc="best")
    ax.set_xlim(xx[0], xx[-1])
    fig.tight_layout()
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_svg)
    plt.close(fig)


def run(args):
    rows = eq.check_data()
    curves, ft_verify = scores_to_curves(rows, args)
    p1_verify = verify_p1_50pct(curves["P1_frozen_deepseek"])
    from train_modernbert_delta import endpoints as get_endpoints
    fixed = get_endpoints(rows)
    allowed_lo = fixed["nothinking"]["tokens"]
    allowed_hi = fixed["thinking"]["tokens"]
    budgets = sorted(args.budgets)
    if len(budgets) != len(set(budgets)) or not budgets or not all(
            math.isfinite(b) and allowed_lo-1e-6 <= b <= allowed_hi+1e-6
            for b in budgets):
        raise ValueError(f"Budgets must be unique and within "
                         f"[{allowed_lo:.2f},{allowed_hi:.2f}]")
    lo, hi = args.curve_min, args.curve_max
    if not (math.isfinite(lo) and math.isfinite(hi) and
            allowed_lo <= lo < hi <= allowed_hi):
        raise ValueError(f"Curve interval must lie inside strict endpoints "
                         f"[{allowed_lo:.2f},{allowed_hi:.2f}]")
    if not all(lo <= b <= hi for b in budgets):
        raise ValueError("All reporting budgets must be inside curve interval")
    if args.grid_points < 3:
        raise ValueError("grid-points must be at least 3")

    grid = [lo+(hi-lo)*k/(args.grid_points-1) for k in range(args.grid_points)]
    all_samples = [sample_at_budget(rows, curves, b) for b in grid]
    budget_points = [sample_at_budget(rows, curves, b) for b in budgets]
    summary = trapezoid_mean_accuracy(all_samples, lo, hi)
    baseline_auc = summary["P1_frozen_deepseek"]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "method": "task_prompt_cost_matched_450_validation",
        "strict_validation_sha256": base.VAL_SHA256,
        "sample_count": 450,
        "mode": "offline_descriptive_expected_accuracy",
        "classification_50pct_reproduction": ft_verify,
        "p1_50pct_reproduction": p1_verify,
        "uses_dense_per_question_scores_for_all_nonrandom_methods": True,
        "interpolation": "cost-adjacent randomized mixture among all 451 top-k policies",
        "curve_budget_range": [lo,hi],
        "curve_grid_points": len(grid),
        "report_budgets": budgets,
        "score_paths": {**{name: str(path) for name,path in METHOD_PATHS.items()},
                        "FT_Classification": str(args.classification_scores)},
        "mean_accuracy_over_budget_interval": summary,
        "mean_uplift_vs_P1_pp": {
            name: 100*(acc-baseline_auc) for name,acc in summary.items()},
        "budget_points": budget_points,
        "grid": all_samples,
        "limitations": [
            "Router DeepSeek forward/prefill cost is excluded from generated tokens.",
            "Budget policy interpolation uses measured validation output-token costs.",
            "Fixed 450 validation set previously informed P1 selection; exploratory.",
            "This is not a generation max_new_tokens sweep or prospective online policy.",
        ],
    }
    (out/"accuracy_token_curve.json").write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    fields = ["budget", *ORDER]
    for filename, data in (("accuracy_token_curve.csv",all_samples),
                           ("budget_points.csv",budget_points)):
        with (out/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
    write_plot(all_samples, out/"accuracy_token_curve.png",
               out/"accuracy_token_curve.svg")
    print("BUDGET | " + " | ".join(LABELS[name] for name in ORDER))
    for p in budget_points:
        print(f"{p['budget']:6.0f} | "+" | ".join(
            f"{p[name]*100:.2f}%" for name in ORDER))
    print("MEAN_ACCURACY_OVER_BUDGET_RANGE " +
          " | ".join(f"{LABELS[name]}={summary[name]*100:.3f}%" for name in ORDER))
    print(f"TASK_PROMPT_CURVE_DONE {out}")


def self_test():
    rows = [
        {"sample_id": "a", "p_t": 1., "p_n": 0., "t_t": 100., "t_n": 20.},
        {"sample_id": "b", "p_t": 0., "p_n": 1., "t_t": 100., "t_n": 20.},
    ]
    c = eq.dense_curve(rows, {"a": 0., "b": 1.})
    sampled = sample_at_budget(rows, {"P1_frozen_deepseek":c}, 80)
    assert abs(sampled["P1_frozen_deepseek"]-.75)<1e-9
    assert abs(sampled["Global_Random"]-.5)<1e-9
    print("TASK_PROMPT_CURVE_SELFTEST_PASS")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--budgets",nargs="+",type=float,default=eq.DEFAULT_BUDGETS)
    parser.add_argument("--curve-min",type=float,default=900.)
    parser.add_argument("--curve-max",type=float,default=2100.)
    parser.add_argument("--grid-points",type=int,default=241)
    parser.add_argument("--classification-scores",type=Path,
                        default=ROOT/"outputs/external_modernbert/classification/val_predictions.jsonl")
    parser.add_argument("--classification-score-field",default="p_nothinking_pred")
    parser.add_argument("--classification-score-direction",
                        choices=("higher_nt","lower_nt"),default="higher_nt")
    parser.add_argument("--out-dir",type=Path,
                        default=ROOT/"results/route_e/task_prompt/curve")
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__=="__main__":
    main()
