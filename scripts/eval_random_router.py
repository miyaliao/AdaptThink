#!/usr/bin/env python3
"""Offline Random Router baseline for the frozen AdaptThink 1.5B model.

Reads paired, already-scored Thinking/NoThinking generations. A router decision
is made ONCE PER QUESTION (including all 16 AIME generations). No LLM inference
or scoring is performed. Python stdlib handles evaluation; matplotlib draws PNG.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("gsm8k", "math", "aime")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_grades(path, mode):
    rows = load_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    indexed = {}
    for row in rows:
        task_id = row["id"]
        if task_id in indexed:
            raise ValueError(f"Duplicate ID in {path}: {task_id}")
        if row.get("mode", mode) != mode:
            raise ValueError(f"Mode mismatch for {task_id}: expected {mode}")
        if not 0 <= float(row["accuracy"]) <= 1:
            raise ValueError(f"Invalid accuracy for {task_id}")
        if not isinstance(row["num_output_tokens"], int) or row["num_output_tokens"] < 0:
            raise ValueError(f"Invalid output token count for {task_id}")
        indexed[task_id] = row
    return indexed


def load_question_pairs(input_dir, split_path, summary_path):
    """Verify canonical IDs, then group all repeated generations by question."""
    splits = load_json(split_path)["source_files"]
    expected = {}
    question_pairs = {}
    for dataset in DATASETS:
        cfg = splits[dataset]
        expected[dataset] = {
            f"{dataset}:{index}:{repeat}"
            for index in cfg["selected_indices"]
            for repeat in range(cfg["repetitions"])
        }

    mode_data = {}
    for mode in ("thinking", "nothinking"):
        path = input_dir / f"{mode}_grades.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline grades: {path}")
        mode_data[mode] = load_grades(path, mode)
        allowed = set().union(*expected.values())
        actual = set(mode_data[mode])
        if actual != allowed:
            missing, extra = sorted(allowed - actual), sorted(actual - allowed)
            raise ValueError(f"{mode} IDs differ from canonical split: "
                             f"missing={missing[:5]}, extra={extra[:5]}")

    for dataset in DATASETS:
        cfg = splits[dataset]
        questions = []
        for index in sorted(cfg["selected_indices"]):
            repeats = [f"{dataset}:{index}:{r}" for r in range(cfg["repetitions"])]
            record = {"question_id": f"{dataset}:{index}"}
            for mode in ("thinking", "nothinking"):
                rows = [mode_data[mode][key] for key in repeats]
                if any(row["dataset"] != dataset for row in rows):
                    raise ValueError(f"Dataset mismatch at {dataset}:{index}")
                record[f"{mode}_accuracy"] = statistics.mean(float(x["accuracy"]) for x in rows)
                record[f"{mode}_tokens"] = statistics.mean(x["num_output_tokens"] for x in rows)
                record[f"{mode}_truncation"] = statistics.mean(
                    float(bool(x.get("stopped_at_limit", False))) for x in rows
                )
            questions.append(record)
        question_pairs[dataset] = questions

    if summary_path.is_file():
        published = load_json(summary_path)["results"]
        for ds, questions in question_pairs.items():
            for mode in ("thinking", "nothinking"):
                for expected_key, observed_key in (("accuracy", f"{mode}_accuracy"),
                                                   ("mean_response_tokens", f"{mode}_tokens")):
                    observed = statistics.mean(q[observed_key] for q in questions)
                    stated = published[mode][ds][expected_key]
                    if not math.isclose(observed, stated, abs_tol=1e-8):
                        raise ValueError(f"Published summary disagrees with grades: {ds}/{mode}/{expected_key}")
    return question_pairs


def metrics(questions, choices=None, p_nothinking=None):
    """If choices is None, compute exact mixture EXPECTATION for probability p."""
    if (choices is None) == (p_nothinking is None):
        raise ValueError("Supply either choices or p_nothinking")
    if choices is not None and len(choices) != len(questions):
        raise ValueError("One choice required per question")
    if choices is None:
        weights = [p_nothinking] * len(questions)
    else:
        weights = [float(chosen) for chosen in choices]

    accuracy = statistics.mean(
        w * q["nothinking_accuracy"] + (1 - w) * q["thinking_accuracy"]
        for q, w in zip(questions, weights)
    )
    tokens = statistics.mean(
        w * q["nothinking_tokens"] + (1 - w) * q["thinking_tokens"]
        for q, w in zip(questions, weights)
    )
    trunc = statistics.mean(
        w * q["nothinking_truncation"] + (1 - w) * q["thinking_truncation"]
        for q, w in zip(questions, weights)
    )
    return {"accuracy": accuracy, "mean_output_tokens": tokens,
            "truncation_fraction": trunc,
            "nothinking_fraction": statistics.mean(weights)}


def percentile(sorted_vals, p):
    pos = (len(sorted_vals) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo) if lo != hi else sorted_vals[lo]


def distribution(values):
    vals = sorted(values)
    return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "p025": percentile(vals, 0.025), "p975": percentile(vals, 0.975)}


def evaluate(pairs, probs, seed_start, n_seeds, input_dir, split_path):
    per_seed = []
    summary = {
        "description": "Offline Random Router, one decision per original question (16 AIME generations grouped)",
        "seed_start": seed_start, "n_seeds": n_seeds,
        "p_nothinking_values": probs,
        "random_rule": "independent uniform draw per question and seed; NoThinking if u < p",
        "common_uniforms_across_probabilities": True,
        "input_sha256": {
            "thinking_grades": fingerprint(input_dir / "thinking_grades.json"),
            "nothinking_grades": fingerprint(input_dir / "nothinking_grades.json"),
            "canonical_split": fingerprint(split_path),
        },
        "datasets": {},
    }

    for dataset, questions in pairs.items():
        always_t = metrics(questions, p_nothinking=0.0)
        always_n = metrics(questions, p_nothinking=1.0)
        dataset_summary = {
            "n_questions": len(questions),
            "generations_per_question": 16 if dataset == "aime" else 1,
            "baselines": {"always_thinking": always_t, "always_nothinking": always_n},
            "random": {},
        }
        for seed in range(seed_start, seed_start + n_seeds):
            rng = random.Random(f"adaptthink-random-router:{seed}:{dataset}")
            uniforms = [rng.random() for _ in questions]
            for p in probs:
                chosen = [u < p for u in uniforms]
                result = metrics(questions, choices=chosen)
                per_seed.append({"dataset": dataset, "seed": seed, "p_nothinking": p,
                                 "n_questions": len(questions),
                                 "n_nothinking_questions": sum(chosen),
                                 **result,
                                 "token_savings_vs_thinking":
                                     1 - result["mean_output_tokens"] / always_t["mean_output_tokens"],
                                 "accuracy_drop_pp_vs_thinking":
                                     100 * (always_t["accuracy"] - result["accuracy"])})
        for p in probs:
            selected = [x for x in per_seed if x["dataset"] == dataset and x["p_nothinking"] == p]
            expectation = metrics(questions, p_nothinking=p)
            expectation["token_savings_vs_thinking"] = (
                1 - expectation["mean_output_tokens"] / always_t["mean_output_tokens"]
            )
            expectation["accuracy_drop_pp_vs_thinking"] = 100 * (always_t["accuracy"] - expectation["accuracy"])
            dataset_summary["random"][f"p{int(round(100 * p)):02d}"] = {
                "p_nothinking": p,
                "theoretical_expectation": expectation,
                "monte_carlo": {name: distribution([x[name] for x in selected])
                                for name in ("accuracy", "mean_output_tokens", "truncation_fraction",
                                             "nothinking_fraction", "token_savings_vs_thinking",
                                             "accuracy_drop_pp_vs_thinking")},
            }
        summary["datasets"][dataset] = dataset_summary
    return summary, per_seed


def draw_chart(summary, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Results saved, but PNG needs matplotlib. Install with: "
                         "python -m pip install matplotlib") from exc

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    for ax, (dataset, info) in zip(axes, summary["datasets"].items()):
        baseline = info["baselines"]
        t, n = baseline["always_thinking"], baseline["always_nothinking"]
        ax.plot([n["mean_output_tokens"], t["mean_output_tokens"]],
                [100 * n["accuracy"], 100 * t["accuracy"]],
                linestyle="--", color="0.5", label="Random expectation")
        for name, point, marker in (("Always Thinking", t, "s"),
                                    ("Always NoThinking", n, "X")):
            ax.scatter(point["mean_output_tokens"], 100 * point["accuracy"],
                       marker=marker, s=75, label=name, zorder=4)
        for key, experiment in info["random"].items():
            mc = experiment["monte_carlo"]
            p = experiment["p_nothinking"]
            ax.errorbar(mc["mean_output_tokens"]["mean"], 100 * mc["accuracy"]["mean"],
                        xerr=mc["mean_output_tokens"]["std"],
                        yerr=100 * mc["accuracy"]["std"], fmt="o", capsize=3,
                        label=f"Random {int(p*100)}/{100-int(p*100)}", zorder=5)
        ax.set_title({"gsm8k": "GSM8K (330)", "math": "MATH-500 (125)",
                      "aime": "AIME 2024 (30 × 16)"}[dataset])
        ax.set_xlabel("Mean generated tokens / response")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"AdaptThink 1.5B — Random Router ({summary['n_seeds']} seeds; error bars ±1 SD)", y=1.02)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/baseline_1p5b")
    parser.add_argument("--split", type=Path,
                        default=ROOT / "data/splits/adaptthink_original_baseline_seed42.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/random_router_1p5b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--probs", type=float, nargs="+", default=[0.10, 0.25, 0.50])
    parser.add_argument("--skip-plot", action="store_true", help="No matplotlib required")
    args = parser.parse_args()
    if args.n_seeds < 1:
        parser.error("--n-seeds must be >= 1")
    if len(set(args.probs)) != len(args.probs) or not all(0 < p < 1 for p in args.probs):
        parser.error("--probs must be unique and strictly between 0 and 1")
    if len({int(round(p * 100)) for p in args.probs}) != len(args.probs):
        parser.error("--probs must have distinct whole-percentage labels")

    pairs = load_question_pairs(args.input_dir, args.split, args.input_dir / "summary.json")
    summary, per_seed = evaluate(pairs, sorted(args.probs), args.seed, args.n_seeds,
                                 args.input_dir, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in (("summary.json", summary), ("per_seed.json", per_seed)):
        (args.output_dir / filename).write_text(
            json.dumps(content, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    for dataset, info in summary["datasets"].items():
        print(f"\n{dataset.upper()} — {info['n_questions']} questions")
        for label, point in info["baselines"].items():
            print(f"  {label:20s} accuracy={point['accuracy']:.2%} "
                  f"tokens={point['mean_output_tokens']:.1f}")
        for item in info["random"].values():
            p, mc = item["p_nothinking"], item["monte_carlo"]
            print(f"  Random p={p:.0%}         accuracy={mc['accuracy']['mean']:.2%} "
                  f"tokens={mc['mean_output_tokens']['mean']:.1f} "
                  f"saved={mc['token_savings_vs_thinking']['mean']:.1%}")
    if not args.skip_plot:
        draw_chart(summary, args.output_dir / "accuracy_tokens.png")
    print(f"\nResults: {args.output_dir}")


if __name__ == "__main__":
    main()
