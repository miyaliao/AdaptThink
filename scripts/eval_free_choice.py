#!/usr/bin/env python3
"""Evaluate a trained checkpoint on the held-out validation split, letting it choose.

The question this answers: after RL, does the model *decide* to skip thinking on
problems where that is safe — or did it just memorise the 250 training problems?

Three conditions are supported:

    free  (default)  the prompt is handed over unchanged; whatever the model emits
                     first decides the mode. If the first generated token is
                     </think> (151649) the model chose the fast path.
    fast             </think> is injected at the end of the prompt, forcing the fast path.
    think            "Alright" (71486) is injected, forcing the thinking path.

For each run it reports how often the fast path was chosen, accuracy (scored with the
same src/adapt_think_rm.py used by the reward), and the mean number of generated
tokens. The base model's numbers are read from the labels file so the comparison is
apples-to-apples on the same 450 problems.

Usage (inside a job):
    python scripts/eval_free_choice.py --model checkpoints/sc4001_rl_bal/global_step_300/huggingface
    python scripts/eval_free_choice.py --mode fast
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
LABELS = ROOT / "results/val_450_labels.jsonl"
EOS_THINK = 151649
NON_EOS_THINK = 71486


def _fix_vllm_uuid_devices():
    """vLLM 0.8.5 casts CUDA_VISIBLE_DEVICES entries to int; GAAS exports GPU UUIDs."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.startswith("GPU-"):
        return
    try:
        import vllm.platforms.cuda as cuda_platform
    except Exception:
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



def _ground_truth(row, answers=None):
    """Resolve the reference answer: row field, then the problem text, then sample_id.

    The problem-text lookup is the reliable one: the labels file and the split file
    number the same questions differently, so matching on ``sample_id`` alone can
    silently pair a question with another question's answer (every score becomes 0).
    """
    # NOTE: never use "label" here - the labels file stores the routing
    # label (0/1) under that name, not the reference answer.
    for key in ("answer", "ground_truth", "gt"):
        if row.get(key) not in (None, ""):
            return row[key]
    problem = (row.get("problem") or "").strip()
    if answers and problem in answers:
        return answers[problem]
    if answers and row.get("sample_id") in answers:
        return answers[row["sample_id"]]
    raise KeyError(f"no ground truth for {row.get('sample_id')}: keys={list(row)[:8]}")


def load_answers(split_path, root):
    """Ground truth keyed by both problem text and sample_id."""
    answers = {}
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    for dataset, meta in split.get("source_files", {}).items():
        items = json.loads((Path(root) / meta["path"]).read_text(encoding="utf-8"))
        for index in meta["selected_indices"]:
            item = items[index]
            answers[item["problem"].strip()] = item["answer"]
            answers[f"{dataset}:{index}"] = item["answer"]
    return answers


def load_rows(path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BASE_MODEL,
                        help="Local model directory (base model, or an exported checkpoint)")
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--mode", choices=("free", "fast", "think"), default="free")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.is_dir():
        raise SystemExit(f"model directory not found: {args.model}")
    if not args.labels.exists():
        raise SystemExit(f"labels not found: {args.labels}")

    _fix_vllm_uuid_devices()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    sys.path.insert(0, str(ROOT / "src"))
    from adapt_think_rm import nothinking_rm

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    if tokenizer.decode([EOS_THINK], skip_special_tokens=False) != "</think>":
        raise SystemExit("token 151649 is not </think> in this tokenizer")

    rows = load_rows(args.labels)
    answers = load_answers(ROOT / "data/splits/adaptthink_val_seed142.json", ROOT)
    prompts, ids = [], []
    for row in rows:
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"]}],
            tokenize=True, add_generation_prompt=True)
        if args.mode == "fast":
            prompt_ids = list(prompt_ids) + [EOS_THINK]
        elif args.mode == "think":
            prompt_ids = list(prompt_ids) + [NON_EOS_THINK]
        prompts.append({"prompt_token_ids": prompt_ids})
        ids.append(row.get("sample_id") or row.get("problem_id") or str(len(ids)))

    print(f"PREFLIGHT ok problems={len(prompts)} mode={args.mode} "
          f"model={args.model} samples={args.samples}", flush=True)

    llm = LLM(model=str(args.model), dtype=args.dtype, trust_remote_code=True,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_tokens + 1024, enforce_eager=True)
    sampling = SamplingParams(n=args.samples, temperature=args.temperature, top_p=0.95,
                              top_k=-1, max_tokens=args.max_tokens)

    outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)

    records = []
    for row, sample_id, output in zip(rows, ids, outputs):
        for candidate in output.outputs:
            tokens = list(candidate.token_ids)
            first = tokens[0] if tokens else None
            chose_fast = first == EOS_THINK
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            raw = nothinking_rm("", text, _ground_truth(row, answers))
            if isinstance(raw, dict):
                score = float(raw.get("score", raw.get("acc", 0.0)))
                ok = bool(raw.get("acc", score >= 1.0))
            else:
                score = float(raw)
                ok = score >= 1.0
            records.append({
                "sample_id": sample_id,
                "chose_fast": bool(chose_fast),
                "first_token": first,
                "tokens": len(tokens),
                "acc": ok,
                "score": float(score),
            })

    fast = [r for r in records if r["chose_fast"]]
    slow = [r for r in records if not r["chose_fast"]]
    summary = {
        "model": str(args.model),
        "mode": args.mode,
        "n_problems": len(rows),
        "n_samples": len(records),
        "fast_rate": len(fast) / max(1, len(records)),
        "accuracy": sum(r["acc"] for r in records) / max(1, len(records)),
        "mean_tokens": sum(r["tokens"] for r in records) / max(1, len(records)),
        "accuracy_when_fast": (sum(r["acc"] for r in fast) / len(fast)) if fast else None,
        "accuracy_when_slow": (sum(r["acc"] for r in slow) / len(slow)) if slow else None,
        "mean_tokens_when_fast": (sum(r["tokens"] for r in fast) / len(fast)) if fast else None,
        "mean_tokens_when_slow": (sum(r["tokens"] for r in slow) / len(slow)) if slow else None,
    }

    # base-model reference on the same problems, from the frozen labels
    if all("p_thinking" in r and "p_nothinking" in r for r in rows):
        summary["base_thinking_acc"] = sum(float(r["p_thinking"]) for r in rows) / len(rows)
        summary["base_nothinking_acc"] = sum(float(r["p_nothinking"]) for r in rows) / len(rows)
        summary["base_thinking_tokens"] = sum(float(r.get("mean_tokens_thinking", 0))
                                              for r in rows) / len(rows)
        summary["base_nothinking_tokens"] = sum(float(r.get("mean_tokens_nothinking", 0))
                                                for r in rows) / len(rows)

    out = args.out or (ROOT / f"results/rl_free_choice_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records},
                              indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== SUMMARY ===")
    for key in ("mode", "n_samples", "fast_rate", "accuracy", "mean_tokens",
                "accuracy_when_fast", "accuracy_when_slow",
                "mean_tokens_when_fast", "mean_tokens_when_slow",
                "base_thinking_acc", "base_nothinking_acc",
                "base_thinking_tokens", "base_nothinking_tokens"):
        if key in summary:
            value = summary[key]
            print(f"  {key:26s} {value:.4f}" if isinstance(value, float) else
                  f"  {key:26s} {value}")
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
