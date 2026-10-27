#!/usr/bin/env python3
"""Verify the frozen AdaptThink baseline split against exact dataset bytes.

Pure Python stdlib. Does not import torch, vLLM or run GPU workloads.
"""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/splits/adaptthink_original_baseline_seed42.json"


def git_blob_sha1(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def expected_ids(split):
    result = []
    for ds in split["dataset_order"]:
        cfg = split["source_files"][ds]
        for i in cfg["selected_indices"]:
            for rep in range(cfg["repetitions"]):
                result.append(f"{ds}:{i}:{rep}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-indices", action="store_true")
    args = parser.parse_args()

    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if split["index_base"] != 0 or split["seed_base"] != 42:
        raise RuntimeError("Unexpected split schema/seed")

    for name in split["dataset_order"]:
        cfg = split["source_files"][name]
        data = (ROOT / cfg["path"]).read_bytes()
        digest = git_blob_sha1(data)
        if digest != cfg["git_blob_sha1"]:
            raise RuntimeError(
                f"{name}: dataset bytes changed; expected Git blob "
                f"{cfg['git_blob_sha1']}, got {digest}"
            )
        items = json.loads(data)
        if len(items) != cfg["num_rows"]:
            raise RuntimeError(f"{name}: row count changed")
        chosen = cfg["selected_indices"]
        if chosen != sorted(set(chosen)):
            raise RuntimeError(f"{name}: indices are not unique/ascending")
        count = math.floor(len(items) * .25 + .5)
        expected = sorted(
            random.Random(cfg["selection_seed"]).sample(range(len(items)), count)
        )
        if chosen != expected:
            raise RuntimeError(
                f"{name}: stored indices differ from original seed-based selection"
            )
        print(
            f"OK {name}: {len(chosen)} / {len(items)} questions, "
            f"seed={cfg['selection_seed']}, Git blob={digest}"
        )
        if args.show_indices:
            print(f"  selected_indices={chosen}")

    full = split["aime_full_extension"]
    if full["selected_indices"] != list(range(30)) or full["repetitions"] != 16:
        raise RuntimeError("AIME full extension specification changed")

    manifest = ROOT / "outputs/baseline_subset_25pct_seed42/manifest.json"
    ids = expected_ids(split)
    if manifest.is_file():
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("sample_ids") != ids:
            raise RuntimeError(
                "Original completed baseline manifest sample_ids do not match split"
            )
        print("OK original run manifest: all sample IDs match the frozen split")
    else:
        print("Original run manifest not present; checked tracked files only")

    print(
        f"SPLIT_VERIFIED: questions={sum(len(split['source_files'][n]['selected_indices']) for n in split['dataset_order'])} "
        f"generations_per_mode={len(ids)} total_two_modes={2*len(ids)} "
        "AIME_full_questions=30"
    )


if __name__ == "__main__":
    main()
