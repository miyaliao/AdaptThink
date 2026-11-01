#!/usr/bin/env python3
"""Join released Thinking K>=16 reference labels with locally generated NoThinking K=4.

This is OFFLINE, CPU-only. It does not train the router, change frozen splits,
or generate model outputs. Pilot defaults to the 600 samples marked is_pilot.

This is the merged version of the two implementations that grew in parallel on
2026-10-31. It keeps the strict input validation from the 16K Slurm pipeline
(config hash, sampling protocol, per-row seed/order checks), the multi-shard
input support needed by the H200 run, and the training-label fields plus the
proxy-calibration report needed by the LoRA router.

Outputs (per scope), in --output-dir:

* ``<scope>_labels.jsonl``: one row per problem
* ``<scope>_summary.json``: per-bucket stats, label balance, proxy calibration

Training fields
---------------
    p_thinking / p_nothinking   empirical success rates (the raw evidence)
    gain_thinking               p_thinking - p_nothinking
    label                       1 if p_nothinking >= p_thinking (ties favour NoThinking)
    label_margin                1 if p_nothinking + 0.125 >= p_thinking
    label_soft                  sigmoid((p_nothinking - p_thinking) / 0.25)
    suggested_weight            1.0 on the boundary buckets, 0.5 elsewhere

Calibration report: the cheap proxy "Thinking accuracy is high, therefore this
problem is easy, therefore NoThinking is fine" is compared with the measured
labels (agreement, precision/recall, per-bucket disagreement). That is what the
pilot is for: it says whether more NoThinking samples are needed on a bucket.

Usage
-----
    python scripts/build_router_labels.py                     # 600-problem pilot
    python scripts/build_router_labels.py --all               # all 3000
    python scripts/build_router_labels.py \
        --input-dir outputs/nothinking_train_seed42 --output-dir results/router_labels_1p5b --all
    python scripts/build_router_labels.py --selftest          # logic check only
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_BUCKETS = {"sometimes", "usually"}
PROXY_THRESHOLD = 0.875     # "Thinking almost always solves it"
EXPECTED_MAX_RESPONSE_TOKENS = 16384


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(items):
    items = list(items)
    return statistics.mean(items) if items else 0.0


def validate_config(input_dir, manifest_path, k):
    """Refuse to score a run that was not produced by the frozen protocol."""
    config_path = input_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json(config_path)
    digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    if config.get("manifest_sha256") != digest:
        raise RuntimeError("Sampling manifest hash differs from label-builder manifest")
    if config.get("mode") != "nothinking":
        raise RuntimeError("Sampling mode is not NoThinking")
    if config.get("sampling", {}).get("max_response_tokens") != EXPECTED_MAX_RESPONSE_TOKENS:
        raise RuntimeError(f"Expected {EXPECTED_MAX_RESPONSE_TOKENS}-token NoThinking "
                           f"protocol, got {config.get('sampling', {}).get('max_response_tokens')}")
    if int(config.get("k_nothinking", -1)) != k:
        raise RuntimeError("NoThinking sampling K mismatch")
    return config


def load_generations(input_dir, selected, k, allow_extra):
    """Read every shard, validate bookkeeping, group answers by problem."""
    files = sorted(input_dir.glob("shard*of*.jsonl")) or sorted(input_dir.glob("nothinking.jsonl"))
    if not files:
        raise FileNotFoundError(f"No generation jsonl under {input_dir}")
    expected = {(s["sample_id"], j) for s in selected for j in range(k)}
    records = defaultdict(dict)
    for path in files:
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["sample_id"], int(row["k"]))
                if key not in expected:
                    if allow_extra:
                        continue      # the same file may already hold non-pilot problems
                    raise RuntimeError(f"{path.name}:{line_no} unexpected record {key}")
                if key[1] in records[key[0]]:
                    raise RuntimeError(f"Duplicate record {key} at {path.name}:{line_no}")
                records[key[0]][key[1]] = row
    missing = [s["sample_id"] for s in selected if len(records[s["sample_id"]]) != k]
    if missing:
        raise RuntimeError(f"Incomplete NoThinking outputs for {len(missing)} problem(s), "
                           f"e.g. {missing[:5]}. Finish the sampling job first.")
    return records, [p.name for p in files]


def score_rows(jobs, workers):
    """Score (response, answer) pairs with the official scorer."""
    sys.path.insert(0, str(ROOT))
    from src.adapt_think_rm import adapt_think_rm

    def one(job):
        response, answer = job
        try:
            out = adapt_think_rm(data_source="", solution_str=response, ground_truth=answer)
            return float(out["acc"]), str(out["pred"])
        except Exception as exc:               # one bad row must not kill the run
            return 0.0, f"ERROR: {type(exc).__name__}"

    if workers > 1 and len(jobs) > 1000:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            return pool.map(one, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
    return [one(job) for job in jobs]


def build_labels(selected, records, scores, k):
    labels = []
    cursor = 0
    for sample in selected:
        rows = [records[sample["sample_id"]][j] for j in range(k)]
        for row in rows:                        # bookkeeping checks from the Slurm pipeline
            if row["order_index"] != sample["order_index"]:
                raise RuntimeError(f"Order mismatch in {sample['sample_id']}")
            if row["seed"] != sample["nothinking_seeds"][int(row["k"])]:
                raise RuntimeError(f"Seed mismatch in {sample['sample_id']}")
            if not isinstance(row["num_output_tokens"], int) or row["num_output_tokens"] <= 0:
                raise RuntimeError(f"Invalid token count in {sample['sample_id']}")
        chunk = scores[cursor:cursor + k]
        cursor += k
        accs = [c[0] for c in chunk]
        preds = [c[1] for c in chunk]
        p_thinking = float(sample["acc_thinking"])
        p_nothinking = mean(accs)
        gain = p_thinking - p_nothinking          # >0 means Thinking was better
        labels.append({
            "sample_id": sample["sample_id"],
            "problem": sample["problem"],
            "bucket": sample["bucket"],
            "order_index": sample["order_index"],
            "is_pilot": sample.get("is_pilot", False),
            "p_thinking": p_thinking,
            "p_nothinking": p_nothinking,
            "gain_thinking": gain,
            "k_thinking": int(sample["n_thinking_responses"]),
            "k_nothinking": k,
            "mean_tokens_thinking": float(sample["mean_len_thinking"]),
            "mean_tokens_nothinking": mean(r["num_output_tokens"] for r in rows),
            "truncation_thinking": float(sample["clip_ratio_thinking"]),
            "truncation_nothinking": mean(r["finish_reason"] == "length" for r in rows),
            "label": int(p_nothinking >= p_thinking),
            "label_margin": int(p_nothinking + 0.125 >= p_thinking),
            "label_soft": 1.0 / (1.0 + math.exp(-(p_nothinking - p_thinking) / 0.25)),
            "suggested_weight": 1.0 if sample["bucket"] in BOUNDARY_BUCKETS else 0.5,
            "predictions": preds,
        })
    return labels


def summarise(labels, scope, manifest_digest, k, files):
    counts = Counter(x["bucket"] for x in labels)
    per_bucket = {}
    for bucket in sorted(counts):
        rows = [x for x in labels if x["bucket"] == bucket]
        per_bucket[bucket] = {
            "n": len(rows),
            "p_thinking": mean(x["p_thinking"] for x in rows),
            "p_nothinking": mean(x["p_nothinking"] for x in rows),
            "frac_nothinking_worse": mean(x["gain_thinking"] > 0 for x in rows),
            "frac_nothinking_better": mean(x["gain_thinking"] < 0 for x in rows),
            "frac_tie": mean(x["gain_thinking"] == 0 for x in rows),
            "mean_tokens_thinking": mean(x["mean_tokens_thinking"] for x in rows),
            "mean_tokens_nothinking": mean(x["mean_tokens_nothinking"] for x in rows),
            "truncation_nothinking": mean(x["truncation_nothinking"] for x in rows),
        }

    # Proxy calibration: "Thinking >= 0.875" as a stand-in for label == 1.
    tp = fp = tn = fn = 0
    for x in labels:
        proxy_positive = x["p_thinking"] >= PROXY_THRESHOLD
        actual_positive = x["label"] == 1
        if proxy_positive and actual_positive:
            tp += 1
        elif proxy_positive:
            fp += 1
        elif actual_positive:
            fn += 1
        else:
            tn += 1
    total = len(labels)
    return {
        "scope": scope,
        "label_method": ("released Thinking K>=16 reference accuracy joined with "
                         "locally generated 16K NoThinking K samples; original adapt_think_rm"),
        "label_noise_warning": "Empirical success rates; K=4 NoThinking is much noisier than K>=16 Thinking",
        "selection_warning": "Training manifest stratified by released Thinking accuracy; not population-representative",
        "manifest_sha256": manifest_digest,
        "input_files": files,
        "n_questions": total,
        "k_nothinking": k,
        "bucket_counts": dict(counts),
        "labels": {
            "nothinking": sum(x["label"] for x in labels),
            "thinking": total - sum(x["label"] for x in labels),
            "nothinking_margin": sum(x["label_margin"] for x in labels),
        },
        "means": {
            name: mean(x[name] for x in labels)
            for name in ("p_thinking", "p_nothinking", "gain_thinking",
                         "mean_tokens_thinking", "mean_tokens_nothinking",
                         "truncation_thinking", "truncation_nothinking")
        },
        "comparison": {
            "thinking_better": sum(x["gain_thinking"] > 0 for x in labels),
            "equal": sum(x["gain_thinking"] == 0 for x in labels),
            "nothinking_better": sum(x["gain_thinking"] < 0 for x in labels),
        },
        "per_bucket": per_bucket,
        "proxy_calibration": {
            "proxy_rule": f"p_thinking >= {PROXY_THRESHOLD}",
            "agreement": (tp + tn) / total if total else 0.0,
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision_nothinking": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall_nothinking": tp / (tp + fn) if (tp + fn) else 0.0,
            "proxy_would_mislabel": fp + fn,
        },
    }


def selftest():
    """Offline logic check: no model, no math_verify, no data files."""
    buckets = ["never", "rare", "sometimes", "sometimes", "usually",
               "always", "always", "always", "never", "sometimes"]
    acc_t = [0.0, 0.25, 0.5, 0.5, 0.75, 0.875, 1.0, 1.0, 0.0, 0.5]
    k = 4
    manifest = [{
        "sample_id": f"s{i}", "problem": "p", "answer": "1", "acc_thinking": acc_t[i],
        "mean_len_thinking": 100.0, "clip_ratio_thinking": 0.0,
        "n_thinking_responses": 16, "bucket": buckets[i], "order_index": i,
        "is_pilot": True, "nothinking_seeds": [1, 2, 3, 4],
    } for i in range(10)]
    records = defaultdict(dict)
    for i in range(10):
        for j in range(k):
            records[f"s{i}"][j] = {"k": j, "order_index": i, "seed": j + 1,
                                   "num_output_tokens": 10, "finish_reason": "stop"}
    scores = []
    for i in range(10):
        accs = [1.0, 0.0, 0.0, 0.0] if i % 2 else [0.0, 0.0, 0.0, 0.0]
        scores += [(a, "1") for a in accs]
    labels = build_labels(manifest, records, scores, k)
    stats = summarise(labels, "selftest", "x" * 64, k, ["fake.jsonl"])
    assert labels[0]["p_nothinking"] == 0.0 and labels[0]["label"] == 1   # tie favours NT
    assert labels[1]["p_nothinking"] == 0.25 and labels[1]["label"] == 1
    assert labels[2]["label"] == 0
    assert stats["labels"]["nothinking"] == 3
    assert stats["per_bucket"]["never"]["n"] == 2
    assert 0.0 <= labels[0]["label_soft"] <= 1.0
    print("SELFTEST_OK", json.dumps(stats["labels"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/splits/sft_train_3000_seed42.json")
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "outputs/router_labels_train_16k")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/router_labels_1p5b")
    parser.add_argument("--all", action="store_true",
                        help="Score the full 3,000 problems instead of the 600-problem pilot")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--selftest", action="store_true", help="Run the offline logic check")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0

    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    input_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    manifest = read_json(manifest_path)
    k = int(manifest["sampling"]["k_nothinking"])
    validate_config(input_dir, manifest_path, k)

    selected = [s for s in manifest["samples"] if args.all or s["is_pilot"]]
    if not selected:
        raise RuntimeError("No samples selected; check --all")
    scope = "all_3000" if args.all else "pilot_600"
    labels_path = output_dir / f"{scope}_labels.jsonl"
    summary_path = output_dir / f"{scope}_summary.json"
    if labels_path.exists() and not args.force:
        raise SystemExit(f"{labels_path} exists; pass --force to overwrite")

    records, files = load_generations(input_dir, selected, k, allow_extra=args.all)
    print(f"loaded {sum(len(v) for v in records.values()):,} answers for "
          f"{len(records):,} problems from {len(files)} file(s)")

    jobs = [(records[s["sample_id"]][j]["response"], s["answer"])
            for s in selected for j in range(k)]
    scores = score_rows(jobs, args.workers)
    labels = build_labels(selected, records, scores, k)
    summary = summarise(labels, scope, hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        k, files)

    output_dir.mkdir(parents=True, exist_ok=True)
    with labels_path.open("w", encoding="utf-8") as f:
        for row in labels:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    print(f"\n{'bucket':10s} {'n':>5s} {'p_T':>6s} {'p_NT':>6s} {'NT worse':>9s} "
          f"{'NT better':>10s} {'tok T':>7s} {'tok NT':>7s}")
    for bucket, b in summary["per_bucket"].items():
        print(f"{bucket:10s} {b['n']:5d} {b['p_thinking']:6.3f} {b['p_nothinking']:6.3f} "
              f"{b['frac_nothinking_worse']:9.1%} {b['frac_nothinking_better']:10.1%} "
              f"{b['mean_tokens_thinking']:7.0f} {b['mean_tokens_nothinking']:7.0f}")
    print(f"\nlabels: {summary['labels']}")
    print(f"proxy calibration: {summary['proxy_calibration']['agreement']:.1%} agreement, "
          f"{summary['proxy_calibration']['proxy_would_mislabel']} problems the cheap proxy "
          f"would mislabel")
    print(f"LABELS_READY labels={labels_path} summary={summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
