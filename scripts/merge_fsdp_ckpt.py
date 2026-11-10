#!/usr/bin/env python3
"""Merge a verl FSDP checkpoint into a HuggingFace model directory.

verl saves ``model_world_size_<N>_rank_<i>.pt`` per rank. Each file holds every
parameter name with its *global* shape, wrapped in a ``DTensor`` whose local storage
is one shard. This script concatenates the local shards along the sharding dimension
and writes a normal HF checkpoint, so the trained policy can be loaded with
``AutoModelForCausalLM.from_pretrained`` and evaluated by ordinary generation code.

Why this exists: the vendored verl copy here has no ``verl.model_merger``, and
retraining only to get an exported model is wasteful.

Usage:
    python scripts/merge_fsdp_ckpt.py --inspect          # show placements first
    python scripts/merge_fsdp_ckpt.py --out-dir <dir>    # write the HF model
"""

import argparse
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints/sc4001_rl_bal/global_step_300/actor"
DEFAULT_BASE = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"


def shard_files(ckpt):
    files = sorted(ckpt.glob("model_world_size_*_rank_*.pt"))
    if not files:
        raise SystemExit(f"no model_world_size_*_rank_*.pt under {ckpt}")
    # sort by rank number, not lexicographically (rank_10 must follow rank_9)
    return sorted(files, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


def load_shards(files):
    shards = []
    for path in files:
        print(f"loading {path.name}", flush=True)
        shards.append(torch.load(path, map_location="cpu", weights_only=False))
    return shards


def merge_value(values):
    """Concatenate one parameter's per-rank shards into a full tensor."""
    first = values[0]
    if not hasattr(first, "to_local"):
        return first                              # already a plain tensor

    from torch.distributed.tensor.placement_types import Replicate, Shard

    placements = first.placements
    local = [v.to_local() for v in values]

    if any(isinstance(p, Replicate) for p in placements):
        return local[0]

    tensor = local[0]
    for placement, part in zip(placements, local[1:], strict=False):
        if isinstance(placement, Shard):
            tensor = torch.cat([tensor, part], dim=placement.dim)
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--inspect", action="store_true",
                        help="Only print sharding info for a few parameters")
    args = parser.parse_args()

    files = shard_files(args.ckpt)
    print(f"found {len(files)} shard file(s): {[f.name for f in files]}", flush=True)
    shards = load_shards(files)

    keys = list(shards[0].keys())
    print(f"parameters: {len(keys)}", flush=True)

    if args.inspect:
        for key in keys[:6]:
            values = [s[key] for s in shards]
            first = values[0]
            placements = getattr(first, "placements", None)
            local_shapes = [tuple(v.to_local().shape) if hasattr(v, "to_local") else tuple(v.shape)
                            for v in values]
            print(f"  {key}\n    global={tuple(first.shape)} placements={placements}\n"
                  f"    local={local_shapes}")
        return 0

    if args.out_dir is None:
        raise SystemExit("--out-dir is required unless --inspect is used")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"merging into {args.dtype} ...", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    merged = {}
    for i, key in enumerate(keys):
        full = merge_value([s[key] for s in shards])
        merged[key] = full.to(dtype).contiguous()
        if (i + 1) % 50 == 0:
            print(f"  merged {i + 1}/{len(keys)}", flush=True)

    total = sum(v.numel() for v in merged.values())
    print(f"merged parameters: {total / 1e9:.3f} B", flush=True)

    print("loading base model skeleton ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
    missing, unexpected = model.load_state_dict(merged, strict=False)
    print(f"missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
    if missing[:3]:
        print("  e.g. missing:", missing[:3])
    if unexpected[:3]:
        print("  e.g. unexpected:", unexpected[:3])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True) \
        .save_pretrained(str(args.out_dir))

    meta = {"source_checkpoint": str(args.ckpt), "base_model": str(args.base_model),
            "dtype": args.dtype, "parameters": total,
            "missing_keys": len(missing), "unexpected_keys": len(unexpected)}
    (args.out_dir / "merge_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"WROTE {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
