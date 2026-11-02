#!/usr/bin/env python3
"""Build a label file for the canonical test split from the cached baseline grades.

The 485 test questions already have both modes generated and scored
(`results/baseline_1p5b/{thinking,nothinking}_grades.json`), so the measured
evidence needed to draw the Acc–Token curve is already on disk. This script puts it
in the same schema as the router label files so the same evaluator can consume it.

No generation, no scoring, no GPU: it only reshapes committed results.

Output: results/test_485_labels.jsonl

Usage:
    python scripts/build_test_labels.py
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRADES = ROOT / "results/baseline_1p5b"
SPLIT = ROOT / "data/splits/adaptthink_original_baseline_seed42.json"
OUT = ROOT / "results/test_485_labels.jsonl"


def load_grades(path, mode):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for row in rows:
        if row.get("mode", mode) != mode:
            raise RuntimeError(f"Mode mismatch in {path}: {row.get('id')}")
        out[row["id"]] = row
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", type=Path, default=GRADES)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    thinking = load_grades(args.grades / "thinking_grades.json", "thinking")
    nothinking = load_grades(args.grades / "nothinking_grades.json", "nothinking")
    if set(thinking) != set(nothinking):
        raise RuntimeError("The two grade files do not cover the same task ids")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for task_id in thinking:
        grouped[task_id.rsplit(":", 1)[0]].append(task_id)

    rows = []
    for position, question in enumerate(sorted(grouped)):
        ids = sorted(grouped[question])
        dataset = question.split(":")[0]
        rows.append({
            "sample_id": question,
            "dataset": dataset,
            "order_index": position,
            "n_generations": len(ids),
            "p_thinking": statistics.mean(thinking[i]["accuracy"] for i in ids),
            "p_nothinking": statistics.mean(nothinking[i]["accuracy"] for i in ids),
            "mean_tokens_thinking": statistics.mean(thinking[i]["num_output_tokens"] for i in ids),
            "mean_tokens_nothinking": statistics.mean(nothinking[i]["num_output_tokens"] for i in ids),
            "truncation_thinking": statistics.mean(
                float(thinking[i].get("stopped_at_limit", False)) for i in ids),
            "truncation_nothinking": statistics.mean(
                float(nothinking[i].get("stopped_at_limit", False)) for i in ids),
        })

    for row in rows:
        row["gain_thinking"] = row["p_thinking"] - row["p_nothinking"]
        row["label"] = int(row["p_nothinking"] >= row["p_thinking"])

    counts = defaultdict(int)
    for row in rows:
        counts[row["dataset"]] += 1
    summary = {
        "scope": "test_485",
        "n_questions": len(rows),
        "per_dataset": dict(counts),
        "note": ("AIME has 16 generations per question, GSM8K/MATH-500 have 1: their "
                 "accuracy estimates differ in precision"),
        "means": {
            name: statistics.mean(r[name] for r in rows)
            for name in ("p_thinking", "p_nothinking", "mean_tokens_thinking",
                         "mean_tokens_nothinking", "truncation_thinking",
                         "truncation_nothinking")
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out.parent / "test_485_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"TEST_LABELS_READY {args.out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
