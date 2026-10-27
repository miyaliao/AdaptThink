#!/usr/bin/env python3
"""AdaptThink original 1.5B baseline: reproducible 25% subset, resumable vLLM."""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT = ROOT / "outputs/baseline_subset_25pct_seed42"
DATA = {"gsm8k": 1319, "math": 500, "aime": 30}
OFFSETS = {"gsm8k": 0, "math": 200000, "aime": 400000}
EOS_THINK = 151649  # The original project calls this eot_token_id, '</think>'.


def tasks_and_hashes():
    tasks, hashes = [], {}
    for name, n_expected in DATA.items():
        p = ROOT / "data/test" / (name + ".json")
        content = p.read_bytes()
        hashes[name] = hashlib.sha256(content).hexdigest()
        items = json.loads(content)
        if len(items) != n_expected:
            raise RuntimeError(f"Unexpected {name} dataset size: {len(items)}")
        count = math.floor(len(items) * .25 + .5)
        indices = sorted(random.Random(42 + OFFSETS[name]).sample(range(len(items)), count))
        for index in indices:
            for rep in range(16 if name == "aime" else 1):
                tasks.append({
                    "id": f"{name}:{index}:{rep}", "dataset": name, "index": index,
                    "repeat": rep, "problem": items[index]["problem"],
                    "answer": items[index]["answer"],
                    "seed": 42 + OFFSETS[name] + 32 * index + rep,
                })
    return tasks, hashes


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    sample = tok.apply_chat_template(
        [{"role": "user", "content": "What is 1+1?"}],
        tokenize=False, add_generation_prompt=True,
    )
    expected = "<｜Assistant｜><think>\n"
    if not sample.endswith(expected):
        raise RuntimeError("Unexpected DeepSeek chat template suffix: " + repr(sample[-90:]))
    if tok.decode([EOS_THINK], skip_special_tokens=False) != "</think>":
        raise RuntimeError("Token 151649 is not </think> in the local model tokenizer")
    return tok


def get_scorer():
    sys.path.insert(0, str(ROOT))
    from src.adapt_think_rm import adapt_think_rm
    check = adapt_think_rm(
        data_source="gsm8k", solution_str="</think> \\boxed{18}", ground_truth="18"
    )
    if check["acc"] != 1:
        raise RuntimeError("Official scorer self-check failed: " + str(check))
    return adapt_think_rm


def configuration(tasks, hashes, tok):
    return {
        "name": "original_1p5b_25pct_seed42_v1",
        "dataset_sha256": hashes,
        "tokenizer_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        "sample_ids": [x["id"] for x in tasks],
        "model": str(MODEL), "modes": ["nothinking", "thinking"],
        "sampling": {"temperature": .6, "top_p": .95, "top_k": -1,
                     "ignore_eos": False, "max_response_tokens": 16384},
        "nothinking_forced_first_response_token": EOS_THINK,
        "thinking": "normal tokenizer prefix ending <think>\\n",
        "gpu": {"dtype": "float16", "max_model_len": 17408, "max_num_seqs": 16,
                "gpu_memory_utilization": .8, "chunked_prefill": False},
    }


def init_manifest(config, create):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    p = OUTPUT / "manifest.json"
    if p.exists():
        if json.loads(p.read_text()) != config:
            raise RuntimeError("Refusing to mix incompatible runs in " + str(OUTPUT))
    elif create:
        p.write_text(json.dumps(config, indent=2) + "\n")


def completed(path, allowed_ids):
    """Resume complete JSONL rows; salvage a partial final line after interruption."""
    rows = {}
    if not path.exists():
        return rows
    with path.open("rb+") as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                x = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                f.truncate(start)
                print("RECOVERED_PARTIAL_JSONL", path, flush=True)
                break
            if x.get("id") not in allowed_ids or x["id"] in rows:
                raise RuntimeError(f"Invalid or duplicate task ID in {path}: {x.get('id')}")
            rows[x["id"]] = x
    return rows


def preflight(tok, tasks):
    scorer = get_scorer()
    max_prompt_len = 0
    for t in tasks:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": t["problem"]}],
            tokenize=True, add_generation_prompt=True,
        )
        max_prompt_len = max(max_prompt_len, len(ids))
        if len(ids) + 1 > 1024:
            raise RuntimeError(f"Prompt exceeds 1024-token allowance: {t['id']} {len(ids)}")
    print(f"PREFLIGHT_OK tasks_per_mode={len(tasks)} total={len(tasks)*2} "
          f"max_prompt_len={max_prompt_len} scorer={scorer.__name__}", flush=True)


def generate(mode, tok, tasks):
    from vllm import LLM, SamplingParams
    path = OUTPUT / (mode + ".jsonl")
    done = completed(path, {t["id"] for t in tasks})
    todo = [t for t in tasks if t["id"] not in done]
    print(f"MODE_START {mode} done={len(done)} remaining={len(todo)}", flush=True)
    if not todo:
        return
    engine = LLM(model=str(MODEL), dtype="float16", tensor_parallel_size=1,
                 gpu_memory_utilization=.80, max_model_len=17408,
                 max_num_seqs=16, enable_chunked_prefill=False, enforce_eager=True)
    tkn = engine.get_tokenizer()
    if tkn.chat_template != tok.chat_template:
        raise RuntimeError("Tokenizer mismatch between Transformers and vLLM")
    with path.open("a", encoding="utf-8") as f:
        for offset in range(0, len(todo), 32):
            batch = todo[offset:offset + 32]
            prompts, params = [], []
            for t in batch:
                ids = tkn.apply_chat_template(
                    [{"role": "user", "content": t["problem"]}],
                    tokenize=True, add_generation_prompt=True,
                )
                if mode == "nothinking":
                    ids.append(EOS_THINK)
                prompts.append({"prompt_token_ids": ids})
                params.append(SamplingParams(
                    temperature=.6, top_p=.95, top_k=-1,
                    max_tokens=16384 - (1 if mode == "nothinking" else 0),
                    ignore_eos=False, seed=t["seed"],
                ))
            start = time.perf_counter()
            outputs = engine.generate(prompts, sampling_params=params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            if len(outputs) != len(batch):
                raise RuntimeError("Mismatched number of generation outputs")
            total_tokens = 0
            for t, output in zip(batch, outputs):
                if len(output.outputs) != 1:
                    raise RuntimeError("Expected exactly one response")
                sample = output.outputs[0]
                total_tokens += len(sample.token_ids)
                text = tkn.decode(sample.token_ids, skip_special_tokens=True)
                row = {
                    "id": t["id"], "dataset": t["dataset"], "index": t["index"],
                    "repeat": t["repeat"], "seed": t["seed"], "answer": t["answer"],
                    "mode": mode,
                    "response": ("</think>" if mode == "nothinking" else "") + text,
                    "num_output_tokens": len(sample.token_ids) + int(mode == "nothinking"),
                    "finish_reason": sample.finish_reason,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"BATCH_DONE mode={mode} done={len(done)+offset+len(batch)}/{len(tasks)} "
                  f"tokens={total_tokens} seconds={elapsed:.2f} "
                  f"throughput={total_tokens/elapsed:.2f}", flush=True)
    print("MODE_DONE", mode, flush=True)


def score(tasks):
    grade = get_scorer()
    allowed = {t["id"] for t in tasks}
    summary = {}
    for mode in ("nothinking", "thinking"):
        rows = completed(OUTPUT / (mode + ".jsonl"), allowed)
        if len(rows) != len(tasks):
            raise RuntimeError(f"Incomplete {mode}: {len(rows)} / {len(tasks)}")
        grades = []
        for task in tasks:
            row = rows[task["id"]]
            result = grade(
                data_source=task["dataset"],
                solution_str=row["response"], ground_truth=task["answer"]
            )
            grades.append({
                "id": task["id"], "dataset": task["dataset"],
                "accuracy": float(result["acc"]), "prediction": str(result["pred"]),
                "num_output_tokens": row["num_output_tokens"],
                "stopped_at_limit": row["finish_reason"] == "length",
                "is_nothinking": row["response"].strip().startswith("</think>"),
            })
        (OUTPUT / (mode + "_grades.json")).write_text(
            json.dumps(grades, ensure_ascii=False, indent=2) + "\n"
        )
        summary[mode] = {}
        for ds in DATA:
            sub = [x for x in grades if x["dataset"] == ds]
            metrics = {
                "generations": len(sub),
                "unique_questions": len({x["id"].rsplit(":", 1)[0] for x in sub}),
                "accuracy": sum(x["accuracy"] for x in sub) / len(sub),
                "mean_response_tokens": sum(x["num_output_tokens"] for x in sub) / len(sub),
                "nothinking_ratio": sum(x["is_nothinking"] for x in sub) / len(sub),
                "max_length_fraction": sum(x["stopped_at_limit"] for x in sub) / len(sub),
            }
            summary[mode][ds] = metrics
            print("SCORE", mode, ds, json.dumps(metrics), flush=True)
    (OUTPUT / "summary.json").write_text(
        json.dumps({"scope": "25pct original 1.5B baselines", "results": summary},
                   ensure_ascii=False, indent=2) + "\n"
    )
    print("SCORE_DONE", OUTPUT / "summary.json", flush=True)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--action", choices=("preflight", "generate", "score"), required=True)
    cli.add_argument("--mode", choices=("nothinking", "thinking"))
    args = cli.parse_args()
    if not MODEL.is_dir():
        raise RuntimeError("Missing model directory: " + str(MODEL))
    tasks, hashes = tasks_and_hashes()
    tokenizer = load_tokenizer()
    init_manifest(configuration(tasks, hashes, tokenizer),
                  create=(args.action != "preflight"))
    if args.action == "preflight":
        preflight(tokenizer, tasks)
    elif args.action == "generate":
        if not args.mode:
            cli.error("Specify --mode with generate")
        generate(args.mode, tokenizer, tasks)
    else:
        score(tasks)


if __name__ == "__main__":
    main()
