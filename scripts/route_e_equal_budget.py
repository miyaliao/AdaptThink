#!/usr/bin/env python3
"""Route E: compare frozen DeepSeek P0/P1/P2 with full-FT Classification
at the SAME expected mean generated-token budgets. CPU only; no generation.

Use all 451 top-k policies for P0/P1/P2, not just the six share points.
Exact budget matching uses a randomized mixture of the two policies
bracketing the target measured mean token cost. This is an *offline
descriptive benchmark*: routing thresholds are not to be tuned on test.

FT Classification: supply its 450 validation routing scores for a dense,
same-protocol comparison. A 50%-NoThinking reproduction check is required
before accepting the dense Classification curve (73.56%, ~1092 tokens).
Otherwise use the six previously reported
Route A aggregate points, **explicitly marked APPROXIMATE**.

Usage:
  python scripts/route_e_equal_budget.py --self-test
  python scripts/route_e_equal_budget.py
  python scripts/route_e_equal_budget.py --classification-scores PATH
  python scripts/route_e_equal_budget.py --classification-scores PATH \
      --classification-score-direction lower_nt --classification-score-field predicted_delta

JSON/CSV outputs are written only under results/route_e/budget_comparison/.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import route_e_frozen as route_e

DEFAULT_BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)
PREDICTIONS = {
    "P0_assistant_think": ROOT / "outputs/route_e/deepseek/full/validation_p_nt.json",
    "P1_assistant_only": ROOT / "outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json",
    "P2_question_only": ROOT / "outputs/route_e/prompt_ablation/user_only/full/validation_p_nt.json",
}
# Source: docs/ROUTE_A_PLAN.md (the classification row of the full sweep).
# These are *rounded* points: a coarse fallback, NOT a dense exact curve.
# Stored as (mean_tokens, accuracy in percentage).
FT_ANCHORS = (
    (567.6, 67.7778),
    (835.0, 70.2222),
    (1092.0, 73.5556),
    (1542.0, 75.5556),
    (1959.0, 77.1111),
    (2209.0, 78.4444),
)


def load_scores(path, ids, direction="higher_nt", field=None):
    """Read {p_nt:{id:score}}, {id:score}, or JSONL with explicit field."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing score file: {p}")
    if p.suffix == ".jsonl":
        if field is None:
            raise ValueError("JSONL requires --classification-score-field to avoid guessing semantics")
        scores = {}
        for line_no, line in enumerate(p.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            sid = str(entry["sample_id"])
            if sid in scores:
                raise ValueError(f"Duplicate sample_id at {line_no}: {sid}")
            scores[sid] = float(entry[field])
    else:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict) and "p_nt" in obj:
            scores = obj["p_nt"]
        elif isinstance(obj, dict) and "scores" in obj:
            scores = obj["scores"]
        elif isinstance(obj, dict):
            scores = obj
        elif isinstance(obj, list) and field is not None:
            scores = {str(item["sample_id"]): float(item[field]) for item in obj}
            if len(scores) != len(obj):
                raise ValueError("Duplicate sample IDs in score list")
        else:
            raise ValueError(f"Cannot interpret scores in {p}")
    if not isinstance(scores, dict):
        raise ValueError(f"Expected ID-to-score mapping in {p}")
    scores = {str(k): float(v) for k, v in scores.items()}
    missing = set(ids) - scores.keys()
    extra = scores.keys() - set(ids)
    if missing or extra:
        raise ValueError(f"{p}: IDs mismatch missing={sorted(missing)[:3]} "
                         f"extra={sorted(extra)[:3]}")
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError(f"Nonfinite score in {p}")
    if direction == "lower_nt":
        scores = {k: -v for k, v in scores.items()}
    return scores


def dense_curve(rows, scores):
    """All 451 prefix top-k policies. Larger score => choose NoThinking first.

    Uses only supplied problem->score to set order; true per-mode outcomes
    and token counts are used *only* for offline validation measurement.
    """
    ordered = sorted(rows, key=lambda r: (-scores[r["sample_id"]], r["sample_id"]))
    n = len(rows)
    at, tt = sum(r["p_t"] for r in rows) / n, sum(r["t_t"] for r in rows) / n
    curve = [{"n_nothinking": 0, "fraction": 0., "tokens": tt, "accuracy": at}]
    for k, r in enumerate(ordered, 1):
        at += (r["p_n"] - r["p_t"]) / n
        tt += (r["t_n"] - r["t_t"]) / n
        curve.append({"n_nothinking": k, "fraction": k/n,
                      "tokens": tt, "accuracy": at})
    return curve


def linear_bracket(curve, budget):
    """Interpolate *by measured token cost*, not by equal NT fraction.

    Adjacent points in sorted cost order define the mixed policy.
    For dense inputs the original top-k policy may be nonmonotonic in
    cost. Sorting by cost and mixing two policies remains well-defined
    (choose either of two scored routing policies at random per request).
    Never optimize accuracy when selecting a bracket.
    """
    curve = sorted(curve, key=lambda x: (x["tokens"], x.get("fraction", 0)))
    if budget < curve[0]["tokens"] - 1e-6 or budget > curve[-1]["tokens"] + 1e-6:
        raise ValueError(
            f"Requested budget {budget} outside observed routing cost range "
            f"[{curve[0]['tokens']:.2f}, {curve[-1]['tokens']:.2f}]")
    if abs(budget - curve[0]["tokens"]) < 1e-9:
        left = right = curve[0]
    elif abs(budget - curve[-1]["tokens"]) < 1e-9:
        left = right = curve[-1]
    else:
        import bisect
        tokens = [r["tokens"] for r in curve]
        k = bisect.bisect_left(tokens, budget)
        if k < len(curve) and abs(tokens[k] - budget) < 1e-9:
            left = right = curve[k]
        else:
            left, right = curve[k-1], curve[k]
    spread = right["tokens"] - left["tokens"]
    weight_right = ((budget - left["tokens"]) / spread
                    if spread > 1e-12 else 0.)
    accuracy = (1-weight_right)*left["accuracy"] + weight_right*right["accuracy"]
    fraction = (1-weight_right)*left.get("fraction", 0) + (
        weight_right*right.get("fraction", 0))
    return {
        "accuracy": accuracy,
        "mean_generated_tokens": budget,
        "expected_nothinking_fraction": fraction,
        "mix_right_probability": weight_right,
        "policy_left_tokens": left["tokens"],
        "policy_right_tokens": right["tokens"],
        "policy_left_fraction": left.get("fraction"),
        "policy_right_fraction": right.get("fraction"),
    }



def verify_classification_50pct(curve, expected_accuracy=331/450,
                                expected_tokens=1092.0,
                                accuracy_tolerance_pp=0.03,
                                token_tolerance=2.0):
    """Reproduce the original Route A 50%-NT result from the 450 raw scores.

    With N=450, top-k index 225 is precisely 50% NoThinking.
    Expected 73.56% (331/450), ~1092 mean generated tokens.
    The original token count was reported rounded to the nearest integer.
    """
    if len(curve) != 451:
        raise ValueError(f"Expected 451 prefix policies, got {len(curve)}")
    point = curve[225]
    measured_acc_pp = 100 * point["accuracy"]
    expected_acc_pp = 100 * expected_accuracy
    acc_error_pp = measured_acc_pp - expected_acc_pp
    token_error = point["tokens"] - expected_tokens
    if abs(acc_error_pp) > accuracy_tolerance_pp or abs(token_error) > token_tolerance:
        raise RuntimeError(
            "FT_CLASSIFICATION_50PCT_FAIL: 225/450 NoThinking policy does not "
            "reproduce historical Classification (73.56%, about 1092 tokens). "
            f"Observed accuracy={measured_acc_pp:.4f}% "
            f"tokens={point['tokens']:.3f}; "
            f"accuracy_error={acc_error_pp:+.4f} pp; "
            f"tokens_error={token_error:+.3f}. "
            "Check the selected score field, score direction, strict "
            "validation ID set, checkpoint, and tie-breaking. "
            "Do not interpret a dense FT comparison until fixed.")
    print("FT_CLASSIFICATION_50PCT_PASS "
          f"nt=225/450 accuracy={measured_acc_pp:.4f}% "
          f"tokens={point['tokens']:.3f} "
          f"accuracy_error_pp={acc_error_pp:+.4f} "
          f"token_error={token_error:+.3f}", flush=True)
    return {
        "verified": True,
        "n_nothinking": 225,
        "accuracy": point["accuracy"],
        "mean_generated_tokens": point["tokens"],
        "reference_accuracy": expected_accuracy,
        "reference_tokens_rounded": expected_tokens,
        "accuracy_error_pp": acc_error_pp,
        "tokens_error": token_error,
    }


def class_anchor_curve(rows):
    """Classification fallback: six Route A rounded aggregate checkpoints."""
    from train_modernbert_delta import endpoints
    fixed = endpoints(rows)
    # Anchor endpoints can be taken from exact current strict label file.
    points = [
        {"tokens": fixed["nothinking"]["tokens"], "accuracy": fixed["nothinking"]["accuracy"],
         "fraction": 1.0},
    ]
    points.extend({"tokens": tok, "accuracy": acc/100, "fraction": frac}
                  for (tok, acc), frac in zip(FT_ANCHORS[1:-1],
                                               (0.75, 0.50, 0.25, 0.10)))
    points.append({"tokens": fixed["thinking"]["tokens"],
                   "accuracy": fixed["thinking"]["accuracy"], "fraction": 0.0})
    return points


def random_at_budget(rows, budget):
    from train_modernbert_delta import endpoints, random_at_budget
    v = random_at_budget(budget, endpoints(rows))
    if v is None:
        raise ValueError(f"Budget outside shared always-T/always-NT range: {budget}")
    return v


def check_data():
    from train_modernbert_delta import read_labels
    path = route_e.VAL
    checksum = route_e.sha256(path)
    if checksum != route_e.VAL_SHA256:
        raise RuntimeError(f"Strict validation SHA mismatch: {checksum}")
    rows = read_labels(path)
    if len(rows) != 450:
        raise ValueError(f"Expected 450 validation questions; received {len(rows)}")
    counts = {ds: sum(r["dataset"] == ds for r in rows)
              for ds in ("gsm8k", "math")}
    if counts != {"gsm8k": 300, "math": 150}:
        raise ValueError(f"Wrong validation groups: {counts}")
    return rows


def run(args):
    rows = check_data()
    ids = [r["sample_id"] for r in rows]
    curves = {}
    evidence = {}
    classification_reproduction = None
    for method, path in PREDICTIONS.items():
        scores = load_scores(path, ids)
        curves[method] = dense_curve(rows, scores)
        evidence[method] = {"type": "dense_prefix_top_k", "source": str(path),
                            "n_policies": len(curves[method]), "exact": True}
    if args.classification_scores is not None:
        scores = load_scores(args.classification_scores, ids,
                             args.classification_score_direction,
                             args.classification_score_field)
        curves["FT_classification"] = dense_curve(rows, scores)
        classification_reproduction = verify_classification_50pct(
            curves["FT_classification"])
        evidence["FT_classification"] = {
            "type": "dense_prefix_top_k", "source": str(args.classification_scores),
            "score_field": args.classification_score_field,
            "score_direction": args.classification_score_direction,
            "n_policies": len(curves["FT_classification"]), "exact": True}
    else:
        curves["FT_classification"] = class_anchor_curve(rows)
        evidence["FT_classification"] = {
            "type": "six_rounded_anchors_linear_interpolation",
            "source": "docs/ROUTE_A_PLAN.md",
            "n_policies": 6, "exact": False,
            "warning": "Classification scores unavailable; this curve is approximate. "
                       "Rerun with --classification-scores to obtain dense exact comparison."}

    from train_modernbert_delta import endpoints
    fixed = endpoints(rows)
    common_min = fixed["nothinking"]["tokens"]
    common_max = fixed["thinking"]["tokens"]
    budgets = list(args.budgets)
    if not budgets or len(budgets) != len(set(budgets)) or not all(
            math.isfinite(b) and common_min-1e-6 <= b <= common_max+1e-6
            for b in budgets):
        raise ValueError(f"Budgets must be distinct and within "
                         f"[{common_min:.1f}, {common_max:.1f}]")
    table = []
    for budget in budgets:
        random = random_at_budget(rows, budget)
        record = {
            "budget": budget, "random_accuracy": random["accuracy"],
            "random_nothinking_fraction": random["nt_probability"],
            "methods": {},
        }
        for name, curve in curves.items():
            p = linear_bracket(curve, budget)
            p["uplift_vs_random_pp"] = 100*(p["accuracy"] - random["accuracy"])
            p["class_anchor_approximation"] = (
                name == "FT_classification" and not evidence[name]["exact"])
            record["methods"][name] = p
        record["p1_minus_ft_accuracy_pp"] = 100*(
            record["methods"]["P1_assistant_only"]["accuracy"] -
            record["methods"]["FT_classification"]["accuracy"])
        table.append(record)

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir/"equal_budget.json"
    filename.write_text(json.dumps({
        "title": "Route E cost-matched validation comparison",
        "scoring": "strict", "val_sha256": route_e.VAL_SHA256,
        "validation_count": len(rows), "dataset_counts": {"gsm8k": 300, "math": 150},
        "metric": "expected accuracy at exact mean generated-token budget",
        "policy_interpolation": "randomized mixture of adjacent policies sorted "
                                "by their measured validation mean generated tokens",
        "selection_warning": "Exploratory validation-only cost-based interpolation. "
                             "Budget policies must not be tuned using sealed test labels.",
        "router_overhead_included": False,
        "classification_dense_scores_present": evidence["FT_classification"]["exact"],
        "classification_50pct_reproduction": classification_reproduction,
        "evidence": evidence, "endpoints": fixed, "table": table,
    }, indent=2) + "\n", encoding="utf-8")

    csv_path = output_dir/"equal_budget.csv"
    fields = ["budget", "random_accuracy_pct"]
    for name in ("FT_classification", "P0_assistant_think",
                 "P1_assistant_only", "P2_question_only"):
        fields += [name+"_accuracy_pct", name+"_uplift_pp", name+"_nt_fraction"]
    fields += ["p1_minus_ft_pp", "classification_approximate"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in table:
            result = {"budget": r["budget"],
                      "random_accuracy_pct": 100*r["random_accuracy"],
                      "p1_minus_ft_pp": r["p1_minus_ft_accuracy_pp"],
                      "classification_approximate":
                          not evidence["FT_classification"]["exact"]}
            for name, p in r["methods"].items():
                result[name+"_accuracy_pct"] = 100*p["accuracy"]
                result[name+"_uplift_pp"] = p["uplift_vs_random_pp"]
                result[name+"_nt_fraction"] = p["expected_nothinking_fraction"]
            writer.writerow(result)

    print(f"VALIDATED validation={len(rows)} sha={route_e.VAL_SHA256}", flush=True)
    print("FT_CLASSIFICATION_SOURCE=" +
          ("DENSE_SCORES" if evidence["FT_classification"]["exact"]
           else "APPROXIMATE_SIX_ANCHORS"), flush=True)
    columns = ["Budget", "Random", "FT Classification", "P0 +think",
               "P1 assistant", "P2 question", "P1 - FT"]
    print(" | ".join(f"{x:>17s}" for x in columns))
    for r in table:
        methods = r["methods"]
        row = [
            f"{r['budget']:,.0f}", f"{r['random_accuracy']*100:.2f}%",
            f"{methods['FT_classification']['accuracy']*100:.2f}%",
            f"{methods['P0_assistant_think']['accuracy']*100:.2f}%",
            f"{methods['P1_assistant_only']['accuracy']*100:.2f}%",
            f"{methods['P2_question_only']['accuracy']*100:.2f}%",
            f"{r['p1_minus_ft_accuracy_pp']:+.2f} pp",
        ]
        print(" | ".join(f"{x:>17s}" for x in row))
    print(f"EQUAL_BUDGET_DONE {filename} {csv_path}", flush=True)


def self_test():
    rows = [
        {"sample_id": "a", "p_t": 1., "p_n": 0., "t_t": 100., "t_n": 20.},
        {"sample_id": "b", "p_t": 0., "p_n": 1., "t_t": 100., "t_n": 20.},
    ]
    ordered = dense_curve(rows, {"a": 0., "b": 1.})
    assert len(ordered) == 3 and ordered[1]["accuracy"] == 1.0
    assert abs(ordered[1]["tokens"]-60.) < 1e-9
    v = linear_bracket(ordered, 80.)
    assert abs(v["accuracy"]-.75) < 1e-9 and abs(v["mean_generated_tokens"]-80) < 1e-9
    assert abs(v["mix_right_probability"]-.5) < 1e-9
    # Unit-test the reproduction invariant with synthetic expected values.
    check = verify_classification_50pct(
        ordered + [ordered[-1]] * 448,
        expected_accuracy=ordered[-1]["accuracy"],
        expected_tokens=ordered[-1]["tokens"])
    assert check["verified"]
    # Verify alternate score orientation yields same sorted policy.
    assert dense_curve(rows, {"a": 0., "b": 1.}) == dense_curve(
        rows, {"a": -0., "b": 1.})
    print("EQUAL_BUDGET_SELFTEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--budgets", type=float, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--classification-scores", type=Path,
                        help="Optional 450-ID classification scores; required for a "
                             "fully dense, exact four-method comparison")
    parser.add_argument("--classification-score-field", type=str,
                        help="Required for JSONL classification predictions")
    parser.add_argument("--classification-score-direction", choices=(
        "higher_nt", "lower_nt"), default="higher_nt",
        help="Direction of external classification scores; higher_nt=larger "
             "score routes NoThinking; lower_nt=smaller score does.")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT/"results/route_e/budget_comparison")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)


if __name__ == "__main__":
    main()
