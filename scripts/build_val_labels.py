#!/usr/bin/env python3
"""Score the 450 validation answers and write a label file the trainer can use.

Reads outputs/val_450/{thinking,nothinking}.jsonl, scores every answer with the
repository's own scorer, and emits the same schema as
results/router_labels_1p5b/all_3000_labels.jsonl so that
scripts/train_decision_lora.py --val-labels can consume it directly:

    sample_id, problem?, p_thinking, p_nothinking, gain_thinking,
    mean_tokens_thinking, mean_tokens_nothinking, order_index, bucket

Validation rows have no difficulty bucket (they were never stratified), so
``bucket`` is set to "val" and ``order_index`` to the row position; the trainer
only uses order_index for the fallback split, which is not used here.

CPU only. Output: results/val_450_labels.jsonl + results/val_450_summary.json

Usage:
    python scripts/build_val_labels.py
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/val_450"
SPLIT = ROOT / "data/splits/adaptthink_val_seed142.json"
OUT_LABELS = ROOT / "results/val_450_labels.jsonl"
OUT_SUMMARY = ROOT / "results/val_450_summary.json"


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else 0.0


def load(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def main():
    thinking = load(INPUT / "thinking.jsonl")
    nothinking = load(INPUT / "nothinking.jsonl")
    if set(thinking) != set(nothinking):
        only_t = sorted(set(thinking) - set(nothinking))[:5]
        only_n = sorted(set(nothinking) - set(thinking))[:5]
        raise RuntimeError(f"Mode mismatch: only thinking={only_t}, only nothinking={only_n}")

    sys.path.insert(0, str(ROOT))
    from src.adapt_think_rm import adapt_think_rm

    rows = []
    for position, qid in enumerate(sorted(thinking)):
        t_row, n_row = thinking[qid], nothinking[qid]
        if t_row["answer"] != n_row["answer"]:
            raise RuntimeError(f"Answer mismatch for {qid}")
        p_t = float(adapt_think_rm(data_source="", solution_str=t_row["response"],
                                   ground_truth=t_row["answer"])["acc"])
        p_nt = float(adapt_think_rm(data_source="", solution_str=n_row["response"],
                                    ground_truth=n_row["answer"])["acc"])
        rows.append({
            "sample_id": qid,
            "problem": None,                       # kept out of the file on purpose
            "bucket": "val",
            "order_index": position,
            "p_thinking": p_t,
            "p_nothinking": p_nt,
            "gain_thinking": p_t - p_nt,
            "k_thinking": 1,
            "k_nothinking": 1,
            "mean_tokens_thinking": float(t_row["num_output_tokens"]),
            "mean_tokens_nothinking": float(n_row["num_output_tokens"]),
            "truncation_thinking": float(t_row["finish_reason"] == "length"),
            "truncation_nothinking": float(n_row["finish_reason"] == "length"),
            "label": int(p_nt >= p_t),
            "label_margin": int(p_nt + 0.125 >= p_t),
            "label_soft": 0.5,
            "suggested_weight": 1.0,
        })

    per_dataset = defaultdict(list)
    for row in rows:
        per_dataset[row["sample_id"].split(":")[0]].append(row)
    summary = {
        "scope": "val_450",
        "n_questions": len(rows),
        "note": "single generation per mode per question; higher noise than the training labels",
        "means": {
            "p_thinking": mean(r["p_thinking"] for r in rows),
            "p_nothinking": mean(r["p_nothinking"] for r in rows),
            "mean_tokens_thinking": mean(r["mean_tokens_thinking"] for r in rows),
            "mean_tokens_nothinking": mean(r["mean_tokens_nothinking"] for r in rows),
            "truncation_thinking": mean(r["truncation_thinking"] for r in rows),
            "truncation_nothinking": mean(r["truncation_nothinking"] for r in rows),
        },
        "comparison": {
            "thinking_better": sum(r["gain_thinking"] > 0 for r in rows),
            "equal": sum(r["gain_thinking"] == 0 for r in rows),
            "nothinking_better": sum(r["gain_thinking"] < 0 for r in rows),
        },
        "per_dataset": {
            name: {"n": len(items),
                   "p_thinking": mean(r["p_thinking"] for r in items),
                   "p_nothinking": mean(r["p_nothinking"] for r in items),
                   "mean_tokens_thinking": mean(r["mean_tokens_thinking"] for r in items),
                   "mean_tokens_nothinking": mean(r["mean_tokens_nothinking"] for r in items)}
            for name, items in per_dataset.items()
        },
    }

    OUT_LABELS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LABELS.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary["means"], indent=2))
    print(f"VAL_LABELS_READY {OUT_LABELS} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
