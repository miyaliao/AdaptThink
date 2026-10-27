#!/usr/bin/env python3
"""Validate canonical GSM8K/MATH500 fixed indices and ALL 30 AIME questions.

Python standard library only; safe on an SSH login node without a GPU.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/splits/adaptthink_original_baseline_seed42.json"


def git_blob_sha1(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canonical_ids(split, dataset):
    cfg = split["source_files"][dataset]
    return [
        f"{dataset}:{index}:{rep}"
        for index in cfg["selected_indices"]
        for rep in range(cfg["repetitions"])
    ]


def verify_manifest_if_present(split):
    """A previous 8-AIME run is historical; don't mistake it for future split."""
    path = ROOT / "outputs/baseline_subset_25pct_seed42/manifest.json"
    if not path.exists():
        return
    old_ids = json.loads(path.read_text(encoding="utf-8")).get("sample_ids", [])
    expected_gsm_math = canonical_ids(split, "gsm8k") + canonical_ids(split, "math")
    actual_gsm_math = [x for x in old_ids if not x.startswith("aime:")]
    if actual_gsm_math != expected_gsm_math:
        raise RuntimeError("Historical manifest GSM8K/MATH500 differs from canonical split")
    canonical_aime = set(canonical_ids(split, "aime"))
    old_aime = [x for x in old_ids if x.startswith("aime:")]
    if not set(old_aime).issubset(canonical_aime):
        raise RuntimeError("Historical manifest contains invalid AIME task IDs")
    print("OK historical run: GSM8K/MATH500 match; AIME canonical remains all 30")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-indices", action="store_true")
    args = parser.parse_args()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if (split.get("schema_version") != 2 or split.get("index_base") != 0
            or split.get("seed_base") != 42):
        raise RuntimeError("Unexpected canonical split schema/seed")

    total_generations = 0
    for ds in split["dataset_order"]:
        cfg = split["source_files"][ds]
        raw = (ROOT / cfg["path"]).read_bytes()
        blob = git_blob_sha1(raw)
        if blob != cfg["git_blob_sha1"]:
            raise RuntimeError(f"{ds}: expected Git blob {cfg['git_blob_sha1']}; found {blob}")
        rows = json.loads(raw)
        if len(rows) != cfg["num_rows"]:
            raise RuntimeError(f"{ds}: dataset length changed")
        indices = cfg["selected_indices"]
        if indices != sorted(set(indices)):
            raise RuntimeError(f"{ds}: indices not sorted/unique")
        if not all(0 <= i < len(rows) for i in indices):
            raise RuntimeError(f"{ds}: invalid row index")
        if ds == "aime":
            if cfg["selection_method"] != "all_rows" or indices != list(range(30)):
                raise RuntimeError("AIME must contain all 30 questions")
            if cfg["repetitions"] != 16:
                raise RuntimeError("AIME must be generated 16 times per question and mode")
        else:
            if cfg["selection_method"] != "seeded_25pct" or cfg["repetitions"] != 1:
                raise RuntimeError(f"{ds}: expected one generation and 25% fixed split")
            n = math.floor(len(rows) * 0.25 + 0.5)
            actual = sorted(
                random.Random(cfg["selection_seed"]).sample(range(len(rows)), n)
            )
            if actual != indices:
                raise RuntimeError(f"{ds}: mismatch from seed-based sample")
        total_generations += len(indices) * cfg["repetitions"]
        print(
            f"OK {ds}: {len(indices)}/{len(rows)} questions; "
            f"{cfg['repetitions']} generation(s) per question; blob={blob}"
        )
        if args.show_indices:
            print(f"  selected_indices={indices}")

    verify_manifest_if_present(split)
    print(
        f"SPLIT_VERIFIED: questions="
        f"{sum(len(split['source_files'][n]['selected_indices']) for n in split['dataset_order'])}"
        f" generations_per_mode={total_generations}"
        f" total_two_modes={2*total_generations} AIME_full_questions=30"
    )


if __name__ == "__main__":
    main()
