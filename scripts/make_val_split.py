#!/usr/bin/env python3
"""Freeze a validation split from the rows NOT used by the canonical test split.

Every router (external, internal, random) must pick its budget operating point on
validation data and be reported on held-out test data. The canonical split in
``data/splits/adaptthink_original_baseline_seed42.json`` is test-only, so this
script carves a disjoint validation set out of the remaining rows.

Design notes
------------
* Selection is decorrelated from the test sampling: the test split used seeds
  42 (GSM8K) / 200042 (MATH-500), this script uses 142 / 200142 over the pool of
  rows the test split did *not* take.
* AIME 2024 has only 30 questions and all of them are in the test split, so no
  AIME validation rows exist. AIME thresholds are therefore never tuned; AIME is
  reported as its own panel with the threshold transferred from GSM8K/MATH.
* The file mirrors the schema of the canonical test split so the existing
  loaders (``scripts/eval_random_router.py``, future routers) can read both.
* CPU only. Refuses to overwrite an existing split unless --force, because the
  frozen file is the single source of truth for every later comparison.

Usage
-----
    python scripts/make_val_split.py            # create
    python scripts/make_val_split.py --force    # regenerate (invalidates results)
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/splits/adaptthink_original_baseline_seed42.json"
OUT = ROOT / "data/splits/adaptthink_val_seed142.json"

# Validation sampling seeds. Deliberately different from the test-split seeds
# (42 / 200042) so the two samples are independent draws from disjoint pools.
SEED_BASE = 142
OFFSETS = {"gsm8k": 0, "math": 200000}
DEFAULT_SIZES = {"gsm8k": 300, "math": 150}


def git_blob_sha1(path):
    """Hash exactly as Git stores a blob, matching the canonical split's field."""
    content = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def load_canonical():
    split = json.loads(CANONICAL.read_text(encoding="utf-8"))
    if split.get("schema_version") != 2:
        raise RuntimeError("Unexpected canonical split schema: "
                           + repr(split.get("schema_version")))
    return split


def verify_source(meta, dataset):
    """Abort if a tracked dataset changed underneath the frozen test split."""
    path = ROOT / meta["path"]
    if not path.is_file():
        raise RuntimeError(f"Missing dataset: {path}")
    actual, expected = git_blob_sha1(path), meta["git_blob_sha1"]
    if actual != expected:
        raise RuntimeError(f"{dataset} changed: git_blob_sha1 {actual} != {expected}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-n", type=int, default=DEFAULT_SIZES["gsm8k"])
    parser.add_argument("--math-n", type=int, default=DEFAULT_SIZES["math"])
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing frozen validation split")
    args = parser.parse_args()
    sizes = {"gsm8k": args.gsm8k_n, "math": args.math_n}
    if any(n < 1 for n in sizes.values()):
        parser.error("validation sizes must be positive")
    if OUT.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite frozen split {OUT}; pass --force "
                         "only if you accept invalidating earlier results.")

    canonical = load_canonical()
    out = {
        "schema_version": 2,
        "split_name": "adaptthink_val_seed142",
        "purpose": (
            "Validation split for selecting router budget operating points. "
            f"GSM8K {sizes['gsm8k']} + MATH-500 {sizes['math']} rows drawn from the "
            "rows NOT used by the canonical test split. No AIME validation rows: "
            "all 30 AIME 2024 questions are in the test split, so AIME thresholds "
            "are transferred and never tuned."
        ),
        "index_base": 0,
        "task_id_format": "{dataset}:{row_index}:{repeat_index}",
        "extraction": (
            "pool = all row indices minus the canonical test selected_indices; "
            "sorted(random.Random(SEED_BASE + OFFSET).sample(pool, n))"
        ),
        "seed_base": SEED_BASE,
        "dataset_order": ["gsm8k", "math", "aime"],
        "notes": [
            "Disjoint from the canonical test split by construction; verified below.",
            "Sampling seeds 142 / 200142 differ from the test seeds 42 / 200042.",
            "Thresholds tuned on this split must be applied unchanged to test.",
            "AIME 2024 (30 questions) is test-only; do not tune anything on AIME.",
        ],
        "source_files": {},
    }

    for dataset in ("gsm8k", "math"):
        meta = canonical["source_files"][dataset]
        path = verify_source(meta, dataset)
        items = json.loads(path.read_text(encoding="utf-8"))
        if len(items) != meta["num_rows"]:
            raise RuntimeError(f"{dataset} row count {len(items)} != {meta['num_rows']}")
        test_idx = set(meta["selected_indices"])
        pool = [i for i in range(len(items)) if i not in test_idx]
        n = sizes[dataset]
        if n > len(pool):
            raise RuntimeError(f"{dataset}: asked {n} validation rows, pool has {len(pool)}")
        seed = SEED_BASE + OFFSETS[dataset]
        selected = sorted(random.Random(seed).sample(pool, n))
        overlap = test_idx.intersection(selected)
        if overlap:
            raise RuntimeError(f"{dataset}: validation overlaps test: {sorted(overlap)[:5]}")
        out["source_files"][dataset] = {
            "path": meta["path"],
            "git_blob_sha1": meta["git_blob_sha1"],
            "num_rows": meta["num_rows"],
            "selection_seed": seed,
            "selected_indices": selected,
            "repetitions": 1,
            "selection_method": "seeded_from_unused_pool",
            "pool_size": len(pool),
        }
        print(f"{dataset}: pool={len(pool)} (test used {len(test_idx)}), "
              f"validation={len(selected)}, seed={seed}")

    # AIME stays test-only; record the fact explicitly instead of an empty bucket.
    aime = canonical["source_files"]["aime"]
    verify_source(aime, "aime")
    out["source_files"]["aime"] = {
        "path": aime["path"],
        "git_blob_sha1": aime["git_blob_sha1"],
        "num_rows": aime["num_rows"],
        "selection_seed": None,
        "selected_indices": [],
        "repetitions": aime["repetitions"],
        "selection_method": "none_all_rows_reserved_for_test",
        "pool_size": 0,
    }
    print("aime: no validation rows (all 30 questions reserved for test)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    total = sum(sizes.values())
    print(f"WROTE {OUT.relative_to(ROOT)} validation_rows={total}")


if __name__ == "__main__":
    main()
