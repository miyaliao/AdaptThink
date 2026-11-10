#!/usr/bin/env python3
"""Build a balanced, short-thinking RL subset so a small RL run has real signal.

Two things make an RL run learn here:

1. **The response cap must not truncate.** The model needs ~8,800 tokens on
   DeepScaleR overall, so a 4,096 cap truncates everything and every reward is
   zero. Restricting to problems whose reference Thinking answer is short keeps the
   cap non-binding.

2. **The batch must contain both outcomes.** If every problem is easy, "always go
   fast" is optimal and there is nothing to learn; if every problem is hard, "always
   think" is optimal. Mixing problems where the fast mode usually succeeds with
   problems where it usually fails creates advantage variance, which is the only
   thing that moves the policy.

Selection uses the measured routing labels (p_nothinking), so "fast succeeds" is
the model's actual behaviour, not a difficulty proxy.

Output: data/train/preprocessed_data/rl_bal_train.parquet / rl_bal_val.parquet

Usage:
    python scripts/make_rl_balanced.py --n-easy 150 --n-hard 150 --max-think 3500
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/splits/sft_train_3000_seed42.json"
LABELS = ROOT / "results/router_labels_1p5b_compliant/all_3000_labels.jsonl"
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
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--n-easy", type=int, default=150, help="problems where fast mode usually succeeds")
    parser.add_argument("--n-hard", type=int, default=150, help="problems where fast mode usually fails")
    parser.add_argument("--max-think", type=float, default=3500, help="max reference Thinking length")
    parser.add_argument("--easy-threshold", type=float, default=0.5, help="p_nothinking >= this is 'easy'")
    parser.add_argument("--hard-threshold", type=float, default=0.25, help="p_nothinking <= this is 'hard'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-prefix", default="rl_bal")
    args = parser.parse_args()

    import pandas as pd

    if not args.labels.exists():
        raise SystemExit(f"missing label file: {args.labels}\n"
                         f"copy it from the cluster: results/router_labels_1p5b_compliant/all_3000_labels.jsonl")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))["samples"]
    by_id = {s["sample_id"]: s for s in manifest}

    labels = {}
    with args.labels.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                labels[row["sample_id"]] = row

    easy, hard = [], []
    for sample_id, row in labels.items():
        sample = by_id.get(sample_id)
        if sample is None:
            continue
        if float(sample.get("mean_len_thinking", 0)) > args.max_think:
            continue
        p_nt = float(row.get("p_nothinking", 0.0))
        if p_nt >= args.easy_threshold:
            easy.append(sample)
        elif p_nt <= args.hard_threshold:
            hard.append(sample)

    print(f"pool: {len(easy)} easy (p_NT >= {args.easy_threshold}), "
          f"{len(hard)} hard (p_NT <= {args.hard_threshold}), "
          f"thinking <= {args.max_think:.0f}")
    if len(easy) < args.n_easy or len(hard) < args.n_hard:
        raise SystemExit(f"not enough problems: need {args.n_easy} easy / {args.n_hard} hard, "
                         f"have {len(easy)} / {len(hard)}. Lower --n-easy/--n-hard or raise --max-think.")

    import random
    rng = random.Random(args.seed)
    rng.shuffle(easy)
    rng.shuffle(hard)
    chosen = easy[:args.n_easy] + hard[:args.n_hard]
    rng.shuffle(chosen)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / f"{args.out_prefix}_train.parquet"
    val_path = OUT_DIR / f"{args.out_prefix}_val.parquet"
    pd.DataFrame([to_row(s["problem"], s["answer"], i) for i, s in enumerate(chosen)]) \
        .to_parquet(train_path)
    pd.DataFrame([to_row(s["problem"], s["answer"], i) for i, s in enumerate(chosen[:20])]) \
        .to_parquet(val_path)
    print(f"WROTE {train_path} rows={len(chosen)}")
    print(f"WROTE {val_path} rows=20")

    lengths = [float(by_id[s["sample_id"]].get("mean_len_thinking", 0)) for s in chosen]
    lengths.sort()
    print(f"reference thinking length: min={lengths[0]:.0f} "
          f"median={lengths[len(lengths) // 2]:.0f} max={lengths[-1]:.0f}")
    print("next: qsub with SUBSET=" + args.out_prefix + ", STEPS=80, ROLLOUTS=4, RESP_LEN=3584")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
