#!/usr/bin/env python3
"""Generate both modes for the frozen validation split (450 GSM8K/MATH questions).

Why this exists: the training labels come from DeepScaleR (hard competition
problems), but the test sets are GSM8K/MATH-500/AIME. A threshold picked on
DeepScaleR would be off-distribution. This script produces Thinking *and*
NoThinking answers for the 450 validation questions that share no problem with the
training set, so the threshold can be picked where it will be used.

Protocol is identical to scripts/eval_baseline_subset.py:
  * same chat template, forced ``</think>`` (151649) for NoThinking
  * temperature 0.6, top_p 0.95, top_k -1, natural EOS
  * 16,384-token cap, bf16 on H200 / fp16 on V100
  * the stored NoThinking response is prefixed with ``</think>`` for the scorer

Output: outputs/val_450/{thinking,nothinking}.jsonl, one row per generation,
        keyed by ``{dataset}:{index}:{repeat}``; append-only and resumable.

Usage (inside a job):
    python scripts/gen_val_outputs.py --mode both
    python scripts/gen_val_outputs.py --mode nothinking
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
SPLIT = ROOT / "data/splits/adaptthink_val_seed142.json"
OUTPUT = ROOT / "outputs/val_450"
EOS_THINK = 151649
OFFSETS = {"gsm8k": 0, "math": 200000, "aime": 400000}


def _fix_vllm_uuid_devices():
    """vLLM 0.8.5 casts CUDA_VISIBLE_DEVICES entries to int; GAAS exports GPU UUIDs.

    With CUDA_VISIBLE_DEVICES="GPU-xxxx", CUDA already maps logical index 0 to that
    device, so returning the logical id unchanged is the correct physical mapping.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-"):
        return
    try:
        import vllm.platforms.cuda as cuda_platform
    except Exception:                                  # pragma: no cover
        return

    def logical_id(device_id):
        if device_id is None:
            return 0
        if isinstance(device_id, str):
            try:
                return int(device_id)
            except ValueError:
                return 0
        return int(device_id)

    cuda_platform.device_id_to_physical_device_id = logical_id
    print(f"PATCHED vllm uuid devices: {visible.split(',')[0][:20]}...", flush=True)

def load_tasks():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    tasks = []
    for dataset in ("gsm8k", "math"):
        meta = split["source_files"][dataset]
        items = json.loads((ROOT / meta["path"]).read_text(encoding="utf-8"))
        for index in meta["selected_indices"]:
            tasks.append({
                "id": f"{dataset}:{index}:0",
                "dataset": dataset,
                "index": index,
                "repeat": 0,
                "problem": items[index]["problem"],
                "answer": items[index]["answer"],
                # same seed convention as the canonical baseline split
                "seed": 42 + OFFSETS[dataset] + 32 * index,
            })
    if not tasks:
        raise RuntimeError("Validation split is empty")
    return tasks


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    if tok.decode([EOS_THINK], skip_special_tokens=False) != "</think>":
        raise RuntimeError("Token 151649 is not </think> in the local tokenizer")
    return tok


def completed(path):
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
            done.add(row["id"])
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--mode", choices=("thinking", "nothinking", "both"), default="both")
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    if not MODEL.is_dir():
        raise RuntimeError(f"Missing model directory: {MODEL}")
    tasks = load_tasks()
    tok = load_tokenizer()

    max_len = 0
    for task in tasks:
        ids = tok.apply_chat_template([{"role": "user", "content": task["problem"]}],
                                      tokenize=True, add_generation_prompt=True)
        max_len = max(max_len, len(ids))
        if len(ids) + 1 > 1024:
            raise RuntimeError(f"Prompt too long for {task['id']}: {len(ids)}")
    print(f"PREFLIGHT_OK tasks={len(tasks)} max_prompt_len={max_len} "
          f"modes={args.mode} max_tokens={args.max_tokens}", flush=True)
    if args.preflight:
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    header = OUTPUT / "config.json"
    config = {
        "name": "val_450_both_modes",
        "split": str(args.split.relative_to(ROOT)),
        "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "model": str(MODEL.relative_to(ROOT)),
        "modes": ["thinking", "nothinking"] if args.mode == "both" else [args.mode],
        "sampling": {"temperature": .6, "top_p": .95, "top_k": -1,
                     "ignore_eos": False, "max_response_tokens": args.max_tokens},
        "nothinking_forced_first_response_token": EOS_THINK,
    }
    if header.exists():
        stored = json.loads(header.read_text(encoding="utf-8"))
        if stored != config:
            print("WARNING: config differs from the stored header; overwriting", flush=True)
    header.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _fix_vllm_uuid_devices()
    from vllm import LLM, SamplingParams
    modes = ["thinking", "nothinking"] if args.mode == "both" else [args.mode]
    engine = LLM(model=str(MODEL), dtype=args.dtype, tensor_parallel_size=1,
                 gpu_memory_utilization=.85, max_model_len=1024 + args.max_tokens,
                 max_num_seqs=args.max_num_seqs, enable_chunked_prefill=False,
                 enforce_eager=True)
    tkn = engine.get_tokenizer()
    if tkn.chat_template != tok.chat_template:
        raise RuntimeError("Tokenizer mismatch between Transformers and vLLM")

    for mode in modes:
        path = OUTPUT / f"{mode}.jsonl"
        done = completed(path)
        todo = [t for t in tasks if t["id"] not in done]
        print(f"MODE_START {mode} done={len(done)} remaining={len(todo)}", flush=True)
        if not todo:
            continue
        total, truncated = 0, 0
        with path.open("a", encoding="utf-8") as f:
            for offset in range(0, len(todo), args.batch):
                batch = todo[offset:offset + args.batch]
                prompts, params = [], []
                for task in batch:
                    ids = tkn.apply_chat_template(
                        [{"role": "user", "content": task["problem"]}],
                        tokenize=True, add_generation_prompt=True)
                    if mode == "nothinking":
                        ids.append(EOS_THINK)
                    prompts.append({"prompt_token_ids": ids})
                    params.append(SamplingParams(
                        temperature=.6, top_p=.95, top_k=-1,
                        max_tokens=args.max_tokens - (1 if mode == "nothinking" else 0),
                        ignore_eos=False, seed=task["seed"]))
                start = time.perf_counter()
                outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
                elapsed = time.perf_counter() - start
                for task, output in zip(batch, outputs):
                    sample = output.outputs[0]
                    n_tokens = len(sample.token_ids) + int(mode == "nothinking")
                    total += n_tokens
                    truncated += int(sample.finish_reason == "length")
                    f.write(json.dumps({
                        "id": task["id"], "dataset": task["dataset"],
                        "index": task["index"], "repeat": task["repeat"],
                        "seed": task["seed"], "answer": task["answer"], "mode": mode,
                        "response": ("</think>" if mode == "nothinking" else "")
                                    + tkn.decode(sample.token_ids, skip_special_tokens=True),
                        "num_output_tokens": n_tokens,
                        "finish_reason": sample.finish_reason,
                    }, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                print(f"BATCH_DONE mode={mode} done={len(done)+offset+len(batch)}/{len(tasks)} "
                      f"tokens={total} seconds={elapsed:.2f} "
                      f"throughput={total/max(elapsed,1e-9):.2f} truncated={truncated}", flush=True)
        print(f"MODE_DONE {mode}", flush=True)
    print("VAL_GEN_DONE", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
