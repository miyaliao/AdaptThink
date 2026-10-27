#!/usr/bin/env python3
"""Fair vLLM concurrency comparison on NTU V100, NOT accuracy evaluation.

Uses one fixed MATH-500 subset across concurrency settings and deliberately
generates an equal number of tokens per request (ignore_eos=True).
Never reuse these responses for baseline accuracy scoring.
"""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_file = root / "data/test/math.json"
    model_path = root / "model/DeepSeek-R1-Distill-Qwen-1.5B"
    if not model_path.exists():
        raise FileNotFoundError(f"Model missing: {model_path}")
    raw = dataset_file.read_bytes()
    records = json.loads(raw)
    if not 1 <= args.num_prompts <= len(records):
        raise ValueError("num-prompts outside dataset size")
    if args.max_num_seqs < 1:
        raise ValueError("max-num-seqs must be positive")

    indices = random.Random(args.seed).sample(
        range(len(records)), args.num_prompts
    )

    from vllm import LLM, SamplingParams

    print(
        f"BENCH_START max_num_seqs={args.max_num_seqs} "
        f"num_prompts={len(indices)} max_tokens={args.max_tokens} "
        f"seed={args.seed}",
        flush=True,
    )
    load_start = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.80,
        max_model_len=17408,
        max_num_seqs=args.max_num_seqs,
        enable_chunked_prefill=False,
        enforce_eager=True,
    )
    load_seconds = time.perf_counter() - load_start
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": records[i]["problem"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in indices
    ]
    # Exact same input order and output-token workload across runs.
    # Each request has its own fixed RNG seed.
    sample_params = [
        SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=args.max_tokens,
            ignore_eos=True,
            seed=args.seed + i,
        )
        for i in indices
    ]

    # Separate small warmup from the measured generation interval.
    llm.generate(
        prompts[:1],
        SamplingParams(temperature=0.0, max_tokens=32),
        use_tqdm=False,
    )
    start = time.perf_counter()
    outputs = llm.generate(prompts, sample_params, use_tqdm=True)
    elapsed = time.perf_counter() - start
    lengths = [len(out.outputs[0].token_ids) for out in outputs]
    total_tokens = sum(lengths)
    metrics = {
        "benchmark": "vllm_concurrency_only_not_accuracy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "dataset": str(dataset_file),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_indices": indices,
        "max_num_seqs": args.max_num_seqs,
        "num_prompts": args.num_prompts,
        "max_tokens_per_request": args.max_tokens,
        "ignore_eos": True,
        "sampling_temperature": 0.6,
        "sampling_top_p": 0.95,
        "seed": args.seed,
        "gpu_memory_utilization": 0.80,
        "max_model_len": 17408,
        "enable_chunked_prefill": False,
        "model_load_seconds": round(load_seconds, 3),
        "generation_seconds": round(elapsed, 3),
        "output_token_lengths": lengths,
        "total_output_tokens": total_tokens,
        "aggregate_output_tokens_per_sec": round(total_tokens / elapsed, 3),
    }
    destination = root / "outputs/benchmarks/vllm_concurrency"
    destination.mkdir(parents=True, exist_ok=True)
    out_file = destination / (
        f"seqs{args.max_num_seqs}_n{args.num_prompts}"
        f"_tok{args.max_tokens}_seed{args.seed}.json"
    )
    out_file.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("BENCH_RESULT " + json.dumps({
        "max_num_seqs": args.max_num_seqs,
        "generation_seconds": metrics["generation_seconds"],
        "total_output_tokens": total_tokens,
        "aggregate_output_tokens_per_sec": metrics["aggregate_output_tokens_per_sec"],
        "result_path": str(out_file),
    }), flush=True)
    if total_tokens != args.num_prompts * args.max_tokens:
        print("WARNING: actual generated tokens differ from requested amount", flush=True)


if __name__ == "__main__":
    main()
