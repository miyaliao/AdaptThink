#!/usr/bin/env python3
"""Extract the model's hidden state at the decision position, for the probe router.

The router only has to answer "should this problem skip thinking?". Training the
language-model head for that failed (the softmax over 150k tokens drowns the
sixnal), so instead read the representation the model already builds for the prompt:

    prompt -> one forward pass -> hidden state at the last prompt token -> classifier

This is decision-only: no answer is generated and none is imitated.

Output: an .npz with sample_id, X (float16, [n, hidden]), and the soft/hard target
carried over from the label file.

Usage
-----
    python scripts/extract_hidden.py \
        --labels results/router_labels_1p5b_compliant/all_3000_labels.jsonl \
        --out outputs/hidden/train.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
LAM = 0.02
TEMPERATURE = 0.25
MAX_LEN = 1024


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    for row in rows:
        if not row.get("problem"):
            raise RuntimeError(f"{row.get('sample_id')} has no problem text")
    return rows


def target_of(row, lam=LAM, temperature=TEMPERATURE):
    raw = (row["p_nothinking"] - row["p_thinking"]
           + lam * (row["mean_tokens_thinking"] - row["mean_tokens_nothinking"]) / 1000.0)
    soft = 1.0 / (1.0 + np.exp(-raw / temperature))
    return float(soft), float(raw >= 0.0), float(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=("float16", "bfloat16"))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(args.labels)
    encoded = []
    for row in rows:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["problem"]}],
            tokenize=True, add_generation_prompt=True)
        if len(ids) > MAX_LEN:
            continue
        encoded.append((row["sample_id"], ids))
    print(f"problems: {len(rows)} usable: {len(encoded)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (torch.bfloat16 if args.dtype == "bfloat16" and torch.cuda.is_bf16_supported()
             else torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model), torch_dtype=dtype, local_files_only=True).to(device).eval()

    pad_id = tokenizer.pad_token_id
    features, kept_ids = [], []
    by_id = {r["sample_id"]: r for r in rows}
    with torch.no_grad():
        for start in range(0, len(encoded), args.batch_size):
            batch = encoded[start:start + args.batch_size]
            width = max(len(ids) for _sid, ids in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention = torch.zeros((len(batch), width), dtype=torch.long)
            lengths = torch.zeros(len(batch), dtype=torch.long)
            for i, (_sid, ids) in enumerate(batch):
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention[i, :len(ids)] = 1
                lengths[i] = len(ids)
            out = model(input_ids=input_ids.to(device),
                        attention_mask=attention.to(device),
                        output_hidden_states=True)
            hidden = out.hidden_states[-1].float()
            picked = hidden[torch.arange(len(batch), device=device),
                            (lengths - 1).to(device)]
            features.append(picked.cpu().numpy().astype(np.float16))
            kept_ids.extend(sid for sid, _ in batch)
            if (start // args.batch_size) % 20 == 0:
                print(f"  {len(kept_ids)}/{len(encoded)}", flush=True)

    X = np.concatenate(features, axis=0)
    soft = np.zeros(len(kept_ids), dtype=np.float32)
    hard = np.zeros(len(kept_ids), dtype=np.float32)
    raw = np.zeros(len(kept_ids), dtype=np.float32)
    for i, sid in enumerate(kept_ids):
        soft[i], hard[i], raw[i] = target_of(by_id[sid])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, sample_id=np.array(kept_ids), X=X,
                        soft=soft, hard=hard, raw=raw)
    print(f"WROTE {args.out} X={X.shape} positive_rate={hard.mean():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
