#!/usr/bin/env python3
"""Generate NoThinking samples for the Internal Router training manifest.

Why this exists: the reference results in ``data/train/ref_results`` already give
the Thinking-side accuracy of DeepScaleR, so the only missing half of the router
label is "does this problem also come out right without thinking". This script
produces exactly that, with the same prompt / forced ``</think>`` token /
decoding / model as the canonical evaluation, and nothing else.

Protocol (kept identical to scripts/eval_baseline_subset.py)
------------------------------------------------------------
* same chat template, then the ``</think>`` token (id 151649) appended
* temperature 0.6, top_p 0.95, top_k -1, natural EOS, fp16, chunked prefill off
* the stored response is prefixed with ``</think>`` so the official scorer sees
  the same string it scored during the baseline run

Deviation, deliberate and documented: ``--max-tokens`` defaults to 8192 for
*training labels* (evaluation still uses 16384). A NoThinking answer that needs
more than 8k tokens is already a routing failure - the point of NoThinking is
short answers - and truncation is the dominant cost driver when sampling
thousands of hard AIME/AMC problems. Recorded in the output header.

Output rows are keyed by ``(sample_id, k)`` and the file is append-only and
resumable, so a Slurm time limit costs at most one batch.

Usage (inside a Slurm job)
--------------------------
    python scripts/gen_nothinking_train.py --pilot            # first 600 problems
    python scripts/gen_nothinking_train.py --limit 2400       # pilot generations
    python scripts/gen_nothinking_train.py --shard 0 --shards 2
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
MANIFEST = ROOT / "data/splits/sft_train_3000_seed42.json"
OUTPUT = ROOT / "outputs/nothinking_train_seed42"
EOS_THINK = 151649  # </think> in the DeepSeek-R1-Distill sequence


def load_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if not samples:
        raise RuntimeError("Manifest contains no samples")
    return manifest, samples


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    sample = tok.apply_chat_template(
        [{"role": "user", "content": "What is 1+1?"}],
        tokenize=False, add_generation_prompt=True,
    )
    if not sample.endswith("<｜Assistant｜><think>\n"):
        raise RuntimeError("Unexpected chat template suffix: " + repr(sample[-90:]))
    if tok.decode([EOS_THINK], skip_special_tokens=False) != "</think>":
        raise RuntimeError("Token 151649 is not </think> in the local tokenizer")
    return tok


def build_config(manifest, args):
    return {
        "name": "internal_router_nothinking_train",
        "manifest": str(args.manifest.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "model": str(MODEL.relative_to(ROOT)),
        "mode": "nothinking",
        "nothinking_forced_first_response_token": EOS_THINK,
        "sampling": {"temperature": .6, "top_p": .95, "top_k": -1,
                     "ignore_eos": False, "max_response_tokens": args.max_tokens},
        "gpu": {"dtype": args.dtype, "max_model_len": 1024 + args.max_tokens,
                "max_num_seqs": args.max_num_seqs,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "chunked_prefill": False, "enforce_eager": True},
        "k_nothinking": manifest["sampling"]["k_nothinking"],
        "max_tokens_note": ("Training-label deviation: 8192 rather than the 16384 used "
                            "for evaluation." if args.max_tokens != 16384 else "same as eval"),
    }


def init_header(config, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "config.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != config:
            raise RuntimeError("Refusing to mix incompatible runs in " + str(output_dir))
    else:
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def flatten(samples, k_total):
    """Flatten (sample, k) pairs in manifest order, then apply sharding."""
    tasks = []
    for sample in samples:
        for k in range(k_total):
            seeds = sample["nothinking_seeds"]
            tasks.append({
                "sample_id": sample["sample_id"],
                "k": k,
                "seed": seeds[k % len(seeds)],
                "problem": sample["problem"],
                "order_index": sample["order_index"],
                "acc_thinking": sample["acc_thinking"],
                "bucket": sample["bucket"],
            })
    tasks.sort(key=lambda t: (t["order_index"], t["k"]))
    return tasks


def completed(path):
    """Resume: salvage a partially written final line, index finished tasks."""
    done = set()
    if not path.exists():
        return done
    with path.open("rb+") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                f.truncate(start)
                print("RECOVERED_PARTIAL_JSONL", path, flush=True)
                break
            done.add((row["sample_id"], int(row["k"])))
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT,
                        help="Separate output directory for each generation protocol; relative to repo root")
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"),
                        help="V100 needs float16; H100/H200 should use bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--pilot", action="store_true",
                        help="Only the stratified pilot prefix of the manifest")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the number of generations (0 = all)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--preflight", action="store_true",
                        help="Tokenizer/scorer/length checks only, no GPU generation")
    args = parser.parse_args()

    if not MODEL.is_dir():
        raise RuntimeError("Missing model directory: " + str(MODEL))
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("--shard must be in [0, --shards)")
    if args.max_tokens <= 1 or args.max_num_seqs < 1 or args.batch < 1 or args.limit < 0:
        parser.error("Require --max-tokens > 1, --max-num-seqs >= 1, --batch >= 1, --limit >= 0")
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    manifest, samples = load_manifest(args.manifest)
    if args.pilot:
        samples = [s for s in samples if s.get("is_pilot")]
        if not samples:
            raise RuntimeError("Manifest has no pilot samples")
    k_total = int(manifest["sampling"]["k_nothinking"])
    tasks = flatten(samples, k_total)
    tasks = tasks[args.shard::args.shards]
    if args.limit:
        tasks = tasks[:args.limit]

    tok = load_tokenizer()
    import sys
    sys.path.insert(0, str(ROOT))
    from src.adapt_think_rm import adapt_think_rm  # noqa: F401  (import check only)

    max_len, too_long = 0, []
    for task in tasks:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": task["problem"]}],
            tokenize=True, add_generation_prompt=True,
        )
        max_len = max(max_len, len(ids))
        if len(ids) + 1 > 1024:
            too_long.append((task["sample_id"], len(ids)))
    if too_long:
        raise RuntimeError(f"{len(too_long)} prompts exceed the 1024-token allowance, "
                           f"e.g. {too_long[:3]}")
    config = build_config(manifest, args)
    init_header(config, output_dir)
    print(f"PREFLIGHT_OK problems={len({t['sample_id'] for t in tasks})} "
          f"generations={len(tasks)} k={k_total} max_prompt_len={max_len} "
          f"max_tokens={args.max_tokens}", flush=True)
    if args.preflight:
        return

    from vllm import LLM, SamplingParams
    path = output_dir / (f"shard{args.shard}of{args.shards}.jsonl" if args.shards > 1
                     else "nothinking.jsonl")
    done = completed(path)
    todo = [t for t in tasks if (t["sample_id"], t["k"]) not in done]
    print(f"GENERATE_START done={len(done)} remaining={len(todo)}", flush=True)
    if not todo:
        print("NOTHINKING_TRAIN_DONE", path, flush=True)
        return

    engine = LLM(model=str(MODEL), dtype=args.dtype,
                 tensor_parallel_size=args.tensor_parallel_size,
                 gpu_memory_utilization=args.gpu_memory_utilization,
                 max_model_len=1024 + args.max_tokens,
                 max_num_seqs=args.max_num_seqs, enable_chunked_prefill=False,
                 enforce_eager=True)
    tkn = engine.get_tokenizer()
    if tkn.chat_template != tok.chat_template:
        raise RuntimeError("Tokenizer mismatch between Transformers and vLLM")

    truncated, total_tokens, written = 0, 0, 0
    with path.open("a", encoding="utf-8") as f:
        for offset in range(0, len(todo), args.batch):
            batch = todo[offset:offset + args.batch]
            prompts, params = [], []
            for task in batch:
                ids = tkn.apply_chat_template(
                    [{"role": "user", "content": task["problem"]}],
                    tokenize=True, add_generation_prompt=True,
                )
                ids.append(EOS_THINK)
                prompts.append({"prompt_token_ids": ids})
                params.append(SamplingParams(
                    temperature=.6, top_p=.95, top_k=-1,
                    max_tokens=args.max_tokens - 1, ignore_eos=False,
                    seed=task["seed"],
                ))
            start = time.perf_counter()
            outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            for task, output in zip(batch, outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected exactly one response per prompt")
                sample = output.outputs[0]
                n_tokens = len(sample.token_ids) + 1
                total_tokens += n_tokens
                is_truncated = sample.finish_reason == "length"
                truncated += int(is_truncated)
                written += 1
                f.write(json.dumps({
                    "sample_id": task["sample_id"],
                    "k": task["k"],
                    "seed": task["seed"],
                    "order_index": task["order_index"],
                    "bucket": task["bucket"],
                    "acc_thinking": task["acc_thinking"],
                    "response": "</think>" + tkn.decode(sample.token_ids,
                                                        skip_special_tokens=True),
                    "num_output_tokens": n_tokens,
                    "finish_reason": sample.finish_reason,
                }, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"BATCH_DONE done={len(done) + written}/{len(tasks)} "
                  f"tokens={total_tokens} seconds={elapsed:.2f} "
                  f"throughput={total_tokens / max(elapsed, 1e-9):.2f} "
                  f"truncated_so_far={truncated}", flush=True)
    print(f"NOTHINKING_TRAIN_DONE {path} generations={written} "
          f"truncated={truncated} tokens={total_tokens}", flush=True)


if __name__ == "__main__":
    main()
