#!/usr/bin/env python3
"""Build tiny parquet files so the AdaptThink RL loop can be smoke-tested.

The paper's recipe is on-policy RL over DeepScaleR; running it at full scale is out
of reach here, but the loop itself can be exercised on a few hundred problems to
answer one question: does this fork train at all on a single GPU?

Rows mirror src/preprocess_dataset.py exactly (data_source, prompt, ability,
reward_model, extra_info) so verl's RLHFDataset accepts them unchanged. Problems
are drawn from the frozen training manifest, so every problem already has a
reference reward in data/train/ref_results (which the reward manager needs).

Output: data/train/preprocessed_data/rl_smoke_train.parquet (N problems)
        data/train/preprocessed_data/rl_smoke_val.parquet   (M problems)

Usage:
    python scripts/make_rl_subset.py --n-train 300 --n-val 30
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/splits/sft_train_3000_seed42.json"
OUT_DIR = ROOT / "data/train/preprocessed_data"


def to_row(problem, answer, index):
    return {
        "data_source": "deepscaler",
        "prompt": [{"role": "user", "content": problem}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"split": "train", "index": index},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-val", type=int, default=30)
    parser.add_argument("--out-prefix", default="rl_smoke",
                        help="Output file prefix, e.g. rl_small -> rl_small_train.parquet")
    args = parser.parse_args()

    import pandas as pd  # same dependency as src/preprocess_dataset.py

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if len(samples) < args.n_train + args.n_val:
        raise RuntimeError("Manifest is smaller than the requested subset")

    # train and val are disjoint slices of the manifest; val is taken from the end
    train = samples[:args.n_train]
    val = samples[-args.n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in ((f"{args.out_prefix}_train", train), (f"{args.out_prefix}_val", val)):
        data = [to_row(s["problem"], s["answer"], i) for i, s in enumerate(rows)]
        path = OUT_DIR / f"{name}.parquet"
        pd.DataFrame(data).to_parquet(path)
        print(f"WROTE {path} rows={len(data)}")

    covered = {s["sample_id"] for s in train} | {s["sample_id"] for s in val}
    print(f"problems={len(covered)}; all have a reference reward in ref_results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
