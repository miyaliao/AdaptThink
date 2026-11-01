#!/usr/bin/env python3
"""Train the Internal Router: a LoRA that decides the first token.

The first generated token *is* the routing decision: `</think>` (id 151649)
starts a NoThinking answer, anything else starts a Thinking answer. This script
trains only that decision:

    p_nt(x) = P(y1 = </think> | x)      under the LoRA model
    loss    = BCE(p_nt, target)         weighted per problem

Nothing else in the model is trained on purpose, so the solver's behaviour stays
put and the ablation "routing gain vs solving-ability change" stays clean.

Targets (--lam selects the cost-aware one)
------------------------------------------
    raw    = (p_nothinking - p_thinking) + lam * (t_thinking - t_nothinking) / 1000
    target = sigmoid(raw / temperature)     with --soft (default)
           = 1 if raw >= 0                  with --hard

With lam = 0 the target only asks "is NoThinking at least as accurate"; with
lam > 0 it asks "is NoThinking worth it at this token price". lam is the shadow
price of a token: the operating point is later chosen by sweeping a threshold on
p_nt, so lam only decides where the model's decision boundary is sharpest.

Split
-----
    order_index % --val-mod != 0 -> train
    order_index % --val-mod == 0 -> held-out, used to report AUC and to pick the
                                    p_nt threshold (never the test set)

CPU-free diagnostics are printed at the end so a failed run is obvious: the p_nt
distribution, AUC, accuracy at 0.5, and how many trainable parameters moved.

Usage
-----
    python scripts/train_decision_lora.py
    python scripts/train_decision_lora.py --lam 0 --output-dir outputs/lora_router_lam0
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "results/router_labels_1p5b/all_3000_labels.jsonl"
MODEL = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
EOS_THINK = 151649  # </think>
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError(f"No label rows in {path}")
    for row in rows:
        for key in ("problem", "p_thinking", "p_nothinking",
                    "mean_tokens_thinking", "mean_tokens_nothinking", "order_index"):
            if key not in row:
                raise RuntimeError(f"Label row {row.get('sample_id')} missing {key}")
    return rows


def make_target(row, lam, temperature, soft):
    raw = (row["p_nothinking"] - row["p_thinking"]
           + lam * (row["mean_tokens_thinking"] - row["mean_tokens_nothinking"]) / 1000.0)
    if not soft:
        return float(raw >= 0.0), raw
    return 1.0 / (1.0 + math.exp(-raw / temperature)), raw


class RouterDataset(torch.utils.data.Dataset):
    """Prompt tokens only: the model is asked about the token right after them."""

    def __init__(self, rows, tokenizer, max_len, lam, temperature, soft):
        self.items = []
        for row in rows:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["problem"]}],
                tokenize=True, add_generation_prompt=True,
            )
            if len(ids) > max_len:
                continue                      # manifest guarantees <=1024; keep the guard
            target, raw = make_target(row, lam, temperature, soft)
            self.items.append({
                "input_ids": ids,
                "target": target,
                "raw": raw,
                "weight": float(row.get("suggested_weight", 1.0)),
                "sample_id": row["sample_id"],
            })
        if not self.items:
            raise RuntimeError("No usable training items")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), max_len), dtype=torch.long)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[i, :len(ids)] = 1
        lengths[i] = len(ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "lengths": lengths,
        "target": torch.tensor([x["target"] for x in batch], dtype=torch.float32),
        "weight": torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
        "sample_id": [x["sample_id"] for x in batch],
    }


def decision_logits(model, batch, device):
    """logit(P=</think>) - logsumexp(logits over every other token)."""
    out = model(input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device))
    logits = out.logits.float()
    last = batch["lengths"].to(device) - 1
    picked = logits[torch.arange(logits.size(0), device=device), last]      # (B, V)
    logit_nt = picked[:, EOS_THINK]
    masked = picked.clone()
    masked[:, EOS_THINK] = -float("inf")
    logit_rest = torch.logsumexp(masked, dim=-1)
    return logit_nt - logit_rest


def auc(scores, labels):
    """Rank-based AUC; ties count as 0.5. No sklearn dependency."""
    pairs = sorted(zip(scores, labels))
    n_pos = sum(1 for _, y in pairs if y > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum, rank, i = 0.0, 1, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0          # 1-based average rank of the tie block
        for k in range(i, j + 1):
            if pairs[k][1] > 0.5:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def p_nt_of(model, dataset, pad_id, device, batch_size=16):
    model.eval()
    scores, ids, targets = [], [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))],
                            pad_id)
            scores.append(torch.sigmoid(decision_logits(model, batch, device)).cpu())
            ids.extend(batch["sample_id"])
            targets.extend(batch["target"].tolist())
    return torch.cat(scores).tolist(), ids, targets


def describe(values):
    if not values:
        return {}
    values = sorted(values)
    grab = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {"min": values[0], "p10": grab(.10), "median": grab(.50),
            "p90": grab(.90), "max": values[-1],
            "mean": sum(values) / len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lora_router_lam002")
    parser.add_argument("--lam", type=float, default=0.02,
                        help="token shadow price in accuracy per 1k tokens (0 = accuracy only)")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--hard", action="store_true", help="Use hard 0/1 targets instead of softened ones")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-mod", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    if not args.model.is_dir():
        raise RuntimeError(f"Missing model directory: {args.model}")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(args.labels)
    train_rows = [r for r in rows if r["order_index"] % args.val_mod != 0]
    val_rows = [r for r in rows if r["order_index"] % args.val_mod == 0]
    if not train_rows or not val_rows:
        raise RuntimeError("Empty train/val split; check --val-mod")

    train_ds = RouterDataset(train_rows, tokenizer, args.max_len,
                             args.lam, args.temperature, not args.hard)
    val_ds = RouterDataset(val_rows, tokenizer, args.max_len,
                           args.lam, args.temperature, not args.hard)
    hard_val = [make_target(r, args.lam, args.temperature, False)[0] for r in val_rows]
    print(f"train={len(train_ds)} val={len(val_ds)} lam={args.lam} "
          f"soft={not args.hard} val_pos_rate={sum(hard_val)/len(hard_val):.3f}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=dtype,
                                                 local_files_only=True).to(device)
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    pad_id = tokenizer.pad_token_id
    order = list(range(len(train_ds)))
    best_auc, best_epoch, best_state = -1.0, None, None

    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        model.train()
        running, seen = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            batch = collate([train_ds[i] for i in order[start:start + args.batch_size]], pad_id)
            logit_diff = decision_logits(model, batch, device)
            raw_loss = loss_fn(logit_diff, batch["target"].to(device))
            weight = batch["weight"].to(device)
            loss = (raw_loss * weight).sum() / weight.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss.detach()) * len(batch["sample_id"])
            seen += len(batch["sample_id"])
        scores, ids, targets = p_nt_of(model, val_ds, pad_id, device)
        score_by_id = dict(zip(ids, scores))
        ordered_scores = [score_by_id[x["sample_id"]] for x in val_ds.items]
        val_auc = auc(ordered_scores, hard_val)
        correct = sum(1 for s, y in zip(ordered_scores, hard_val) if (s >= 0.5) == (y > 0.5))
        print(f"epoch {epoch}: train_loss={running/max(seen,1):.4f} val_auc={val_auc:.4f} "
              f"val_acc@0.5={correct/len(hard_val):.3f} "
              f"p_nt_mean={sum(ordered_scores)/len(ordered_scores):.3f}", flush=True)
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.named_parameters() if v.requires_grad}
    print(f"best val_auc={best_auc:.4f} at epoch {best_epoch}", flush=True)

    # Report on the best epoch, not the last one, then persist the adapter once.
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.save_pretrained(args.output_dir)

    # Final diagnostics: the p_nt distribution decides whether a threshold sweep
    # has any resolution at all. If it collapses, the LoRA barely moved the logit.
    scores, ids, targets = p_nt_of(model, val_ds, pad_id, device)
    score_by_id = dict(zip(ids, scores))
    ordered_scores = [score_by_id[x["sample_id"]] for x in val_ds.items]
    report = {
        "lam": args.lam, "soft": not args.hard, "temperature": args.temperature,
        "epochs": args.epochs, "lora_r": args.lora_r, "lr": args.lr,
        "train_size": len(train_ds), "val_size": len(val_ds),
        "val_pos_rate": sum(hard_val) / len(hard_val),
        "best_val_auc": best_auc, "best_epoch": best_epoch,
        "val_p_nt": describe(ordered_scores),
        "val_auc_final": auc(ordered_scores, hard_val),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.output_dir / "val_val_p_nt.json").write_text(
        json.dumps({"sample_id": ids, "p_nt": scores}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"ADAPTER_SAVED {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
