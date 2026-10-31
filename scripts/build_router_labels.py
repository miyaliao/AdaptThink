#!/usr/bin/env python3
"""Join released Thinking K>=16 reference labels with locally generated NoThinking K=4.

This is OFFLINE, CPU-only. It does not train the router, change frozen splits,
or generate model outputs. Pilot defaults to the 600 samples marked is_pilot.
"""

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(items):
    return statistics.mean(items) if items else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/splits/sft_train_3000_seed42.json")
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "outputs/router_labels_train_16k")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/router_labels_1p5b")
    parser.add_argument("--all", action="store_true", help="Score full 3000 instead of 600 pilot")
    args = parser.parse_args()

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    manifest = read_json(manifest_path)
    config = read_json(input_dir / "config.json")
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if config.get("manifest_sha256") != digest:
        raise RuntimeError("Sampling manifest hash differs from label-builder manifest")
    if config.get("mode") != "nothinking":
        raise RuntimeError("Sampling mode is not NoThinking")
    if config.get("sampling", {}).get("max_response_tokens") != 16384:
        raise RuntimeError("Expected 16,384-token NoThinking sampling protocol")
    k = int(manifest["sampling"]["k_nothinking"])
    if int(config.get("k_nothinking", -1)) != k:
        raise RuntimeError("NoThinking sampling K mismatch")

    selected = [s for s in manifest["samples"] if args.all or s["is_pilot"]]
    expected = {(s["sample_id"], j) for s in selected for j in range(k)}
    if not selected or len(expected) != len(selected) * k:
        raise RuntimeError("Empty or duplicated training manifest")

    raw_path = input_dir / "nothinking.jsonl"
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    records = {}
    with raw_path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            key = (row["sample_id"], int(row["k"]))
            if key not in expected:
                if args.all:
                    raise RuntimeError(f"Unexpected NoThinking record: {key}")
                continue  # Full train data may be in the same file as the pilot
            if key in records:
                raise RuntimeError(f"Duplicate record {key} at line {line_no}")
            records[key] = row
    missing = sorted(expected - set(records))
    if missing:
        raise RuntimeError(f"Incomplete {len(records)}/{len(expected)} NoThinking outputs; "
                           f"first missing IDs: {missing[:8]}. Resume Slurm pilot first.")

    # Original scorer; keep EXACT same correctness definition as baseline.
    import sys
    sys.path.insert(0, str(ROOT))
    from src.adapt_think_rm import adapt_think_rm

    labels = []
    for sample in selected:
        rows = [records[(sample["sample_id"], j)] for j in range(k)]
        accs = []
        for row in rows:
            if row["order_index"] != sample["order_index"]:
                raise RuntimeError(f"Order mismatch in {sample['sample_id']}")
            if row["seed"] != sample["nothinking_seeds"][int(row["k"])]:
                raise RuntimeError(f"Seed mismatch in {sample['sample_id']}")
            if not isinstance(row["num_output_tokens"], int) or row["num_output_tokens"] <= 0:
                raise RuntimeError(f"Invalid tokens in {sample['sample_id']}")
            outcome = adapt_think_rm(
                data_source="", solution_str=row["response"],
                ground_truth=sample["answer"]
            )
            accs.append(float(outcome["acc"]))
        thinking = float(sample["acc_thinking"])
        nothinking = mean(accs)
        labels.append({
            "sample_id": sample["sample_id"],
            "problem": sample["problem"],
            "bucket": sample["bucket"],
            "p_thinking": thinking,
            "p_nothinking": nothinking,
            "gain_thinking": thinking - nothinking,
            "k_thinking": int(sample["n_thinking_responses"]),
            "k_nothinking": k,
            "mean_tokens_thinking": float(sample["mean_len_thinking"]),
            "mean_tokens_nothinking": mean(r["num_output_tokens"] for r in rows),
            "truncation_thinking": float(sample["clip_ratio_thinking"]),
            "truncation_nothinking": mean(
                r["finish_reason"] == "length" for r in rows
            ),
        })

    bucket_counts = Counter(row["bucket"] for row in labels)
    summary = {
        "scope": "all_3000" if args.all else "pilot_600",
        "label_method": "Original released Thinking K>=16 and new 16K NoThinking K4; original adapt_think_rm",
        "label_noise_warning": "Empirical success rates; K4 NoThinking is much noisier than K>=16 Thinking",
        "selection_warning": "Training manifest stratified by released Thinking accuracy; not population-representative",
        "manifest_sha256": digest,
        "n_questions": len(labels),
        "k_nothinking": k,
        "bucket_counts": dict(bucket_counts),
        "means": {
            name: mean(row[name] for row in labels)
            for name in ("p_thinking", "p_nothinking", "gain_thinking",
                         "mean_tokens_thinking", "mean_tokens_nothinking",
                         "truncation_thinking", "truncation_nothinking")
        },
        "comparison": {
            "thinking_better": sum(x["gain_thinking"] > 0 for x in labels),
            "equal": sum(x["gain_thinking"] == 0 for x in labels),
            "nothinking_better": sum(x["gain_thinking"] < 0 for x in labels),
        },
        "per_bucket": {
            b: {
                "n": bucket_counts[b],
                "p_thinking": mean(x["p_thinking"] for x in labels if x["bucket"] == b),
                "p_nothinking": mean(x["p_nothinking"] for x in labels if x["bucket"] == b),
            } for b in sorted(bucket_counts)
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "all_3000" if args.all else "pilot_600"
    out_labels = output_dir / f"{suffix}_labels.jsonl"
    with out_labels.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_summary = output_dir / f"{suffix}_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"LABELS_READY labels={out_labels} summary={out_summary}")


if __name__ == "__main__":
    main()
