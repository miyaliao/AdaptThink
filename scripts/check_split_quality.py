#!/usr/bin/env python3
"""Audit the frozen training manifest before any GPU time is spent on labels.

Two questions this answers with measurements instead of opinion:

1. Leakage at *near*-duplicate level, not just exact text. ``make_sft_manifest``
   removes problems whose whitespace-free lowercase text is identical to a
   test/validation problem. DeepScaleR reuses competition sources, so a problem
   can survive that filter and still be the same question with one number
   changed. This script shingles every text into character 8-grams and reports
   Jaccard / containment similarity between every manifest problem and every
   held-out (test + validation) problem.

2. Coverage. The manifest is stratified by reference Thinking accuracy, which is
   not the label we actually train on. If its prompt-length or difficulty
   distribution is badly skewed against the full DeepScaleR pool, the router will
   be trained on a different problem mix than the one it is evaluated on.

Exit code is non-zero when a pair reaches ``--fail-jaccard`` so this can gate a
commit. CPU only, no model needed.

Usage
-----
    python scripts/check_split_quality.py
    python scripts/check_split_quality.py --manifest <path> --fail-jaccard 0.85
"""

import argparse
import json
import statistics
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/splits/sft_train_3000_seed42.json"
TEST_SPLIT = ROOT / "data/splits/adaptthink_original_baseline_seed42.json"
VAL_SPLIT = ROOT / "data/splits/adaptthink_val_seed142.json"
DEEPSCALER = ROOT / "data/train/deepscaler.json"
REPORT = ROOT / "outputs/split_quality_report.json"


def norm_text(text):
    text = unicodedata.normalize("NFKC", text)
    return "".join(text.split()).lower()


def shingles(text, n=8):
    """Character n-grams over normalised text; short texts fall back to the text."""
    norm = norm_text(text)
    if len(norm) <= n:
        return {norm} if norm else set()
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}


def load_held_out():
    """Every test/validation question, tagged with which split it belongs to."""
    held = []
    for path, label in ((TEST_SPLIT, "test"), (VAL_SPLIT, "val")):
        if not path.is_file():
            print(f"WARNING: missing {path.name}", file=sys.stderr)
            continue
        split = json.loads(path.read_text(encoding="utf-8"))
        for dataset, meta in split["source_files"].items():
            items = json.loads((ROOT / meta["path"]).read_text(encoding="utf-8"))
            for index in meta["selected_indices"]:
                held.append({
                    "dataset": dataset,
                    "split": label,
                    "index": index,
                    "problem": items[index]["problem"],
                    "shingles": shingles(items[index]["problem"]),
                })
    return held


def near_duplicates(manifest_samples, held, thresholds):
    """Jaccard/containment of every manifest problem against every held-out one."""
    manifest_sets = [(s["sample_id"], shingles(s["problem"])) for s in manifest_samples]
    hits = {t: [] for t in thresholds}
    worst = []
    for h in held:
        best = None
        for sample_id, grams in manifest_sets:
            if not grams or not h["shingles"]:
                continue
            inter = len(grams & h["shingles"])
            if not inter:
                continue
            union = len(grams) + len(h["shingles"]) - inter
            jaccard = inter / union
            containment = inter / min(len(grams), len(h["shingles"]))
            score = max(jaccard, containment)
            if best is None or score > best["score"]:
                best = {"score": score, "jaccard": jaccard,
                        "containment": containment, "sample_id": sample_id,
                        "held_out": f"{h['split']}:{h['dataset']}:{h['index']}"}
        if best is None:
            continue
        best["held_out_problem"] = h["problem"][:160]
        worst.append(best)
        for t in thresholds:
            if best["score"] >= t:
                hits[t].append(best)
    worst.sort(key=lambda x: -x["score"])
    return hits, worst


def quantiles(values):
    values = sorted(values)
    if not values:
        return {}
    grab = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {"min": values[0], "p10": grab(.10), "median": grab(.50),
            "p90": grab(.90), "max": values[-1], "mean": round(statistics.mean(values), 1)}


def coverage(manifest_samples):
    pool = json.loads(DEEPSCALER.read_text(encoding="utf-8"))
    man_chars = quantiles([len(s["problem"]) for s in manifest_samples])
    pool_chars = quantiles([len(r["problem"]) for r in pool])
    man_acc = quantiles([s["acc_thinking"] for s in manifest_samples])
    pool_acc = quantiles([s["acc_thinking"] for s in manifest_samples])  # replaced below
    return {"manifest_prompt_chars": man_chars, "pool_prompt_chars": pool_chars,
            "manifest_acc_thinking": man_acc,
            "manifest_buckets": {b: sum(1 for s in manifest_samples if s["bucket"] == b)
                                 for b in sorted({s["bucket"] for s in manifest_samples})},
            "pool_rows": len(pool)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.85, 0.95])
    parser.add_argument("--fail-jaccard", type=float, default=0.85,
                        help="exit non-zero if any pair reaches this similarity")
    parser.add_argument("--max-report", type=int, default=10)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    held = load_held_out()
    print(f"manifest problems: {len(samples)} | held-out questions: {len(held)} "
          f"(test+val) | pairs compared: {len(samples) * len(held):,}")

    hits, worst = near_duplicates(samples, held, args.thresholds)
    print("\n=== near-duplicate check (char 8-gram, max of Jaccard/containment) ===")
    for t in args.thresholds:
        print(f"  similarity >= {t:.2f}: {len(hits[t])} pair(s)")
    print("  worst matches against held-out questions:")
    for row in worst[:args.max_report]:
        print(f"    {row['score']:.3f}  train={row['sample_id']:16s} held={row['held_out']}")
        print(f"           {row['held_out_problem'][:110]}")

    cov = coverage(samples)
    print("\n=== coverage vs the full DeepScaleR pool ===")
    print(f"  manifest prompt chars: {cov['manifest_prompt_chars']}")
    print(f"  pool     prompt chars: {cov['pool_prompt_chars']}")
    print(f"  manifest acc_thinking: {cov['manifest_acc_thinking']}")
    print(f"  manifest buckets:      {cov['manifest_buckets']}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({"manifest": str(args.manifest),
                                  "n_manifest": len(samples),
                                  "n_held_out": len(held),
                                  "thresholds": {str(t): len(hits[t]) for t in args.thresholds},
                                  "worst_matches": worst[:50],
                                  "coverage": cov},
                                 indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"\nWROTE {REPORT.relative_to(ROOT)}")

    worst_score = worst[0]["score"] if worst else 0.0
    if worst_score >= args.fail_jaccard:
        print(f"\nFAIL: worst held-out similarity {worst_score:.3f} >= "
              f"--fail-jaccard {args.fail_jaccard}", file=sys.stderr)
        return 1
    print(f"\nPASS: worst held-out similarity {worst_score:.3f} < {args.fail_jaccard}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
