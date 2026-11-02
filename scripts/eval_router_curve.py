#!/usr/bin/env python3
"""Turn the router's p_nt scores into an Accuracy–Token curve and compare it.

Input
-----
* a label file with the measured evidence per problem
  (``p_thinking``, ``p_nothinking``, ``mean_tokens_thinking``, ``mean_tokens_nothinking``)
* a scores file mapping ``sample_id`` -> ``p_nt`` (the model's routing probability),
  or ``--labels`` containing a ``p_nt`` field directly

Output
------
For every threshold ``t`` (route NoThinking when ``p_nt >= t``):
mean accuracy, mean output tokens, fraction routed NoThinking. Plus the reference
curves:

* always Thinking / always NoThinking (the two fixed modes)
* Random router, budget-matched by linear interpolation between the fixed modes
* Oracle, i.e. routing each problem with its *measured* values. This is the ceiling
  for any router. Two variants:
    - fidelity: sort by ``p_nothinking - p_thinking`` (accuracy only, lambda = 0)
    - cost-aware: sort by ``(p_nothinking - p_thinking) + lam*(t_T - t_NT)/1000``

The gap between the learned curve and the oracle curve is the honest headline
number: how much of the achievable saving this router actually captures.

If the ``p_nt`` values are computed on the *test* split, the thresholds must have
been chosen on validation data; this script does not choose them, it only reports
the whole curve so the choice is visible.

CPU only, no dependencies beyond the standard library.

Usage
-----
    python scripts/eval_router_curve.py \\
        --labels results/val_450_labels.jsonl --scores outputs/lora/p_nt_val.json
    python scripts/eval_router_curve.py --selftest
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95


def load_labels(path, scores_path=None):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError(f"No rows in {path}")
    if scores_path is not None:
        scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
        if isinstance(scores, dict) and "p_nt" in scores and isinstance(scores["p_nt"], dict):
            scores = scores["p_nt"]
        missing = [r["sample_id"] for r in rows if r["sample_id"] not in scores]
        if missing:
            raise RuntimeError(f"{len(missing)} problems have no p_nt, e.g. {missing[:3]}")
        for row in rows:
            row["p_nt"] = float(scores[row["sample_id"]])
    for row in rows:
        for key in ("p_thinking", "p_nothinking", "mean_tokens_thinking",
                    "mean_tokens_nothinking"):
            if key not in row:
                raise RuntimeError(f"Row {row.get('sample_id')} missing {key}")
    return rows


def has_scores(rows):
    return all("p_nt" in row for row in rows)


def route_metrics(rows, chooser):
    """Mean accuracy and tokens when `chooser(row)` decides NoThinking."""
    acc = tok = nt = 0.0
    for row in rows:
        if chooser(row):
            acc += row["p_nothinking"]
            tok += row["mean_tokens_nothinking"]
            nt += 1
        else:
            acc += row["p_thinking"]
            tok += row["mean_tokens_thinking"]
    n = len(rows)
    return {"accuracy": acc / n, "mean_tokens": tok / n, "nothinking_fraction": nt / n}


def threshold_curve(rows, thresholds):
    curve = []
    for t in thresholds:
        point = route_metrics(rows, lambda r, t=t: r["p_nt"] >= t)
        point["threshold"] = t
        curve.append(point)
    return sorted(curve, key=lambda p: p["mean_tokens"])


def fixed_modes(rows):
    return {
        "always_thinking": route_metrics(rows, lambda r: False),
        "always_nothinking": route_metrics(rows, lambda r: True),
    }


def random_curve(rows, fixed, steps=40):
    """Budget-matched random router: linear interpolation between the fixed modes."""
    lo, hi = fixed["always_nothinking"], fixed["always_thinking"]
    out = []
    for i in range(steps + 1):
        p = i / steps
        out.append({
            "nothinking_fraction": p,
            "mean_tokens": lo["mean_tokens"] + (hi["mean_tokens"] - lo["mean_tokens"]) * p,
            "accuracy": lo["accuracy"] + (hi["accuracy"] - lo["accuracy"]) * p,
        })
    return out


def oracle_curve(rows, lam):
    """Upper bound: order problems by measured cost-aware utility."""
    ordered = sorted(rows, key=lambda r: -((r["p_nothinking"] - r["p_thinking"])
                                           + lam * (r["mean_tokens_thinking"]
                                                    - r["mean_tokens_nothinking"]) / 1000.0))
    curve, acc, tok = [], 0.0, 0.0
    total_acc = sum(r["p_thinking"] for r in rows)
    total_tok = sum(r["mean_tokens_thinking"] for r in rows)
    n = len(rows)
    for i, row in enumerate(ordered, start=1):
        total_acc += row["p_nothinking"] - row["p_thinking"]
        total_tok += row["mean_tokens_nothinking"] - row["mean_tokens_thinking"]
        curve.append({"nothinking_fraction": i / n, "accuracy": total_acc / n,
                      "mean_tokens": total_tok / n})
    return sorted(curve, key=lambda p: p["mean_tokens"])


def accuracy_at_budget(curve, budget):
    """Best accuracy with mean tokens <= budget (points near the budget included)."""
    candidates = [p for p in curve if p["mean_tokens"] <= budget * 1.001]
    if not candidates:
        return None
    best = max(candidates, key=lambda p: p["accuracy"])
    return {k: best[k] for k in ("accuracy", "mean_tokens", "nothinking_fraction", "threshold") if k in best}


def summarise(rows, thresholds, budgets, lam):
    fixed = fixed_modes(rows)
    learned = threshold_curve(rows, thresholds) if has_scores(rows) else []
    comparison = {
        "n_problems": len(rows),
        "fixed_modes": fixed,
        "learned_curve": learned,
        "random_curve": random_curve(rows, fixed),
        "oracle_accuracy_only": oracle_curve(rows, 0.0),
        "oracle_cost_aware": oracle_curve(rows, lam),
        "lambda": lam,
    }
    table = []
    for budget in budgets:
        row = {"budget": budget,
               "learned": accuracy_at_budget(learned, budget) if learned else None,
               "random": accuracy_at_budget(random_curve(rows, fixed), budget),
               "oracle_accuracy_only": accuracy_at_budget(comparison["oracle_accuracy_only"], budget),
               "oracle_cost_aware": accuracy_at_budget(comparison["oracle_cost_aware"], budget)}
        if row["learned"] and row["oracle_cost_aware"]:
            row["gap_to_oracle_pp"] = 100 * (row["oracle_cost_aware"]["accuracy"]
                                             - row["learned"]["accuracy"])
        table.append(row)
    comparison["at_budget"] = table
    return comparison


def selftest():
    rows = []
    for i in range(100):
        p_t = (i % 10) / 10.0
        p_nt = max(0.0, p_t - 0.2) if i % 3 else p_t
        rows.append({
            "sample_id": f"s{i}", "p_thinking": p_t, "p_nothinking": p_nt,
            "mean_tokens_thinking": 1000.0, "mean_tokens_nothinking": 200.0,
            "p_nt": 1.0 - p_t,          # perfect ranker in this synthetic case
        })
    result = summarise(rows, DEFAULT_THRESHOLDS, [200, 600, 1000], 0.02)
    assert result["n_problems"] == 100
    fixed = result["fixed_modes"]
    assert fixed["always_nothinking"]["mean_tokens"] == 200.0
    assert fixed["always_thinking"]["mean_tokens"] == 1000.0
    assert len(result["learned_curve"]) == len(DEFAULT_THRESHOLDS)
    # a perfect ranker must dominate the random interpolation at 600 tokens
    at600 = result["at_budget"][1]
    assert at600["learned"]["accuracy"] >= at600["random"]["accuracy"] - 1e-9
    # sorted ascending by tokens: all-NoThinking first; the all-Thinking endpoint is
    # reported separately in fixed_modes, so the curve stops one problem short of it.
    assert result["oracle_cost_aware"][0]["mean_tokens"] == 200.0
    assert result["oracle_cost_aware"][-1]["mean_tokens"] == 992.0
    print("SELFTEST_OK", json.dumps({"learned@600": at600["learned"],
                                     "random@600": at600["random"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None,
                        help="JSON mapping sample_id -> p_nt (optional if labels carry p_nt)")
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[2000, 3000, 4000, 5000, 6000, 8000])
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0
    if args.labels is None:
        parser.error("--labels is required (or use --selftest)")

    rows = load_labels(args.labels, args.scores)
    thresholds = args.thresholds or DEFAULT_THRESHOLDS
    result = summarise(rows, thresholds, args.budgets, args.lam)

    label = "learned" if result["learned_curve"] else "(no scores)"
    print(f"{'budget':>7s} {label:>17s} {'random':>17s} {'oracle(cost)':>17s} {'gap':>7s}")
    for row in result["at_budget"]:
        def fmt(point):
            return f"{point['accuracy']*100:5.1f}% @{point['mean_tokens']:5.0f}" if point else "      -      "
        gap = f"{row['gap_to_oracle_pp']:6.1f}" if "gap_to_oracle_pp" in row else "   -  "
        print(f"{row['budget']:7.0f} {fmt(row['learned']):>17s} {fmt(row['random']):>17s} "
              f"{fmt(row['oracle_cost_aware']):>17s} {gap:>7s}")

    out = args.out or Path(args.labels).with_suffix(".curve.json")
    Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
