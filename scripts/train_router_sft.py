#!/usr/bin/env python3
"""Train the Internal Router: decision token + NoThinking answers (route B, full SFT).

Why v2
------
A decision-only version (BCE on P(y1 = </think>), no response supervision) was
measured and failed: train loss stalled at 0.685 against a trivial-value of 0.693,
so the model could not even fit 3,000 training problems. One supervised position per
example is too little signal. This version keeps the decision objective and adds
language-model supervision on the generated NoThinking answers, which gives hundreds
of supervised positions per example:

    loss = BCE(p_nt, target) * w                    # routing decision
         + response_weight * CE(answer tokens) * w  # only where NoThinking works

A response is imitated only when
  * the cost-aware target says "route NoThinking", and
  * one of the generated answers was scored correct after format repair,
so the model is never taught to produce a wrong answer on purpose.

Targets
-------
    raw    = (p_nothinking - p_thinking) + lam * (t_thinking - t_nothinking) / 1000
    target = sigmoid(raw / temperature)   (soft, default)   or  raw >= 0  (--hard)

Data contract
-------------
    --labels         label jsonl (problem, answer, p_*, mean_tokens_*, order_index)
    --nothinking-dir directory with the generated answers; the repaired directory
                     (outputs/nothinking_train_repaired) is preferred because its
                     responses are scorable and carry acc_repaired per generation
    --val-labels     in-distribution validation labels (results/val_450_labels.jsonl)

Usage
-----
    python scripts/train_decision_lora.py
    python scripts/train_decision_lora.py --response-weight 0    # ablation: decision only
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "results/router_labels_1p5b_repaired/all_3000_labels.jsonl"
NOTHINKING_DIR = ROOT / "outputs/nothinking_train_repaired"
MODEL = ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
EOS_THINK = 151649
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        raise RuntimeError(f"No label rows in {path}")
    for row in rows:
        for key in ("problem", "p_thinking", "p_nothinking", "mean_tokens_thinking",
                    "mean_tokens_nothinking", "order_index"):
            if key not in row:
                raise RuntimeError(f"Label row {row.get('sample_id')} missing {key}")
        if not row["problem"]:
            raise RuntimeError(f"Label row {row.get('sample_id')} has no problem text")
    return rows


def load_answers(directory, wanted):
    """sample_id -> list of (k, response, correct_after_repair)."""
    files = sorted(Path(directory).glob("shard*of*.jsonl")) or sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No generation jsonl under {directory}")
    out = defaultdict(list)
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = row["sample_id"]
                if sample_id not in wanted:
                    continue
                correct = float(row.get("acc_repaired", row.get("acc_strict", 0.0)))
                out[sample_id].append((int(row["k"]), row["response"], correct))
    for answers in out.values():
        answers.sort()
    return out


def make_target(row, lam, temperature, soft):
    raw = (row["p_nothinking"] - row["p_thinking"]
           + lam * (row["mean_tokens_thinking"] - row["mean_tokens_nothinking"]) / 1000.0)
    if not soft:
        return float(raw >= 0.0), raw
    return 1.0 / (1.0 + math.exp(-raw / temperature)), raw


class RouterDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_len, lam, temperature, soft,
                 answers=None, max_resp_tokens=1024, answer_scope="correct"):
        self.items = []
        self.skipped_no_answer = 0
        for row in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["problem"]}],
                tokenize=True, add_generation_prompt=True)
            if len(prompt) > max_len:
                continue
            target, raw = make_target(row, lam, temperature, soft)
            response_ids = None
            # Answer supervision is scoped by *correctness*: the model should
            # learn to answer well without thinking wherever that is already
            # possible, not only on the problems the routing target would send to
            # the fast mode. answer_scope="target" restores the stricter scope.
            wants_answer = answers is not None and (answer_scope == "correct"
                                                   or target >= 0.5)
            if wants_answer:
                for _k, text, correct in answers.get(row["sample_id"], []):
                    if correct > 0.5:
                        response_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                        response_ids = response_ids[:max_resp_tokens]
                        break
                if response_ids is None:
                    self.skipped_no_answer += 1
            ids = list(prompt) + (response_ids or [])
            labels = [-100] * len(prompt) + (response_ids or [])
            self.items.append({
                "input_ids": ids, "labels": labels, "prompt_len": len(prompt),
                "has_response": response_ids is not None,
                "target": target, "hard": float(raw >= 0.0),
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
    width = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    prompt_len = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = torch.tensor(item["input_ids"], dtype=torch.long)
        attention[i, :n] = 1
        labels[i, :n] = torch.tensor(item["labels"], dtype=torch.long)
        prompt_len[i] = item["prompt_len"]
    return {
        "input_ids": input_ids, "attention_mask": attention, "labels": labels,
        "prompt_len": prompt_len,
        "target": torch.tensor([x["target"] for x in batch], dtype=torch.float32),
        "weight": torch.tensor([x["weight"] for x in batch], dtype=torch.float32),
        "has_response": torch.tensor([x["has_response"] for x in batch], dtype=torch.float32),
        "sample_id": [x["sample_id"] for x in batch],
    }


def decision_logit_diff(logits, prompt_len):
    rows = torch.arange(logits.size(0), device=logits.device)
    picked = logits[rows, prompt_len - 1]
    logit_nt = picked[:, EOS_THINK]
    masked = picked.clone()
    masked[:, EOS_THINK] = -float("inf")
    return logit_nt - torch.logsumexp(masked, dim=-1)


def forward_losses(model, batch, device, response_weight):
    out = model(input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device))
    logits = out.logits.float()
    diff = decision_logit_diff(logits, batch["prompt_len"].to(device))

    weight = batch["weight"].to(device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        diff, batch["target"].to(device), reduction="none")
    decision_loss = (bce * weight).sum() / weight.sum().clamp_min(1e-6)

    resp_loss = torch.zeros((), device=device)
    if response_weight > 0 and bool(batch["has_response"].any()):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"].to(device)[:, 1:].contiguous()
        token_loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
            ignore_index=-100, reduction="none").view(shift_labels.shape)
        mask = (shift_labels != -100).float()
        per_example = (token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        keep = batch["has_response"].to(device)
        resp_loss = (per_example * keep).sum() / keep.sum().clamp_min(1e-1)

    return decision_loss + response_weight * resp_loss, decision_loss.detach(), resp_loss.detach()


def p_nt_of(model, dataset, pad_id, device, batch_size=16):
    model.eval()
    scores, ids = [], []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            batch = collate([dataset[i] for i in
                             range(start, min(start + batch_size, len(dataset)))], pad_id)
            out = model(input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device))
            diff = decision_logit_diff(out.logits.float(), batch["prompt_len"].to(device))
            scores.append(torch.sigmoid(diff).cpu())
            ids.extend(batch["sample_id"])
    return torch.cat(scores).tolist(), ids


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n_pos = sum(1 for _, y in pairs if y > 0.5)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            if pairs[k][1] > 0.5:
                rank_sum += avg_rank
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def describe(values):
    if not values:
        return {}
    values = sorted(values)
    grab = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {"min": values[0], "p10": grab(.10), "median": grab(.50), "p90": grab(.90),
            "max": values[-1], "mean": sum(values) / len(values)}


def evaluate(model, dataset, pad_id, device, hard, name):
    scores, ids = p_nt_of(model, dataset, pad_id, device)
    by_id = dict(zip(ids, scores))
    ordered = [by_id[item["sample_id"]] for item in dataset.items]
    correct = sum(1 for s, y in zip(ordered, hard) if (s >= 0.5) == (y > 0.5))
    result = {f"{name}_auc": auc(ordered, hard),
              f"{name}_acc_at_0.5": correct / len(hard),
              f"{name}_p_nt": describe(ordered)}
    print(f"{name}: auc={result[f'{name}_auc']:.4f} "
          f"acc@0.5={result[f'{name}_acc_at_0.5']:.3f} "
          f"p_nt_mean={result[f'{name}_p_nt']['mean']:.3f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--val-labels", type=Path, default=None)
    parser.add_argument("--nothinking-dir", type=Path, default=NOTHINKING_DIR)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/lora_router_sft")
    parser.add_argument("--lam", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--answer-scope", choices=("correct", "target"), default="correct",
                        help="Which problems get answer supervision: every problem with a "
                             "compliant correct answer, or only those the router would route "
                             "to the fast mode")
    parser.add_argument("--max-resp-tokens", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--val-mod", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if not args.model.is_dir():
        raise RuntimeError(f"Missing model directory: {args.model}")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(args.labels)
    if args.val_labels is not None:
        train_rows, val_rows = rows, load_rows(args.val_labels)
        overlap = {r["sample_id"] for r in rows} & {r["sample_id"] for r in val_rows}
        if overlap:
            raise RuntimeError(f"{len(overlap)} validation problems also in training")
        print(f"external validation: {args.val_labels} ({len(val_rows)} problems)")
    else:
        train_rows = [r for r in rows if r["order_index"] % args.val_mod != 0]
        val_rows = [r for r in rows if r["order_index"] % args.val_mod == 0]
        print(f"fallback split: held-out every {args.val_mod}th problem")

    answers = load_answers(args.nothinking_dir, {r["sample_id"] for r in train_rows})
    print(f"answers: {len(answers)} problems from {args.nothinking_dir}")

    common = dict(max_len=args.max_len, lam=args.lam, temperature=args.temperature,
                  soft=not args.hard, max_resp_tokens=args.max_resp_tokens)
    train_ds = RouterDataset(train_rows, tokenizer, answers=answers,
                             answer_scope=args.answer_scope, **common)
    val_ds = RouterDataset(val_rows, tokenizer, answers=None, **common)
    imitable = sum(1 for x in train_ds.items if x["has_response"])
    print(f"train={len(train_ds)} (imitable answer: {imitable}) val={len(val_ds)} "
          f"lam={args.lam} response_weight={args.response_weight}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(args.model), torch_dtype=dtype,
                                                 local_files_only=True).to(device)
    model.config.use_cache = False

    base = {}
    base.update(evaluate(model, train_ds, tokenizer.pad_token_id, device,
                         [x["hard"] for x in train_ds.items], "base_train"))
    base.update(evaluate(model, val_ds, tokenizer.pad_token_id, device,
                         [x["hard"] for x in val_ds.items], "base_val"))

    for p in model.parameters():
        p.requires_grad = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.config.use_cache = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05)
    pad_id = tokenizer.pad_token_id
    order = list(range(len(train_ds)))
    best = {"auc": -1.0, "epoch": None, "state": None}

    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        model.train()
        totals, seen = [0.0, 0.0, 0.0], 0
        for start in range(0, len(order), args.batch_size):
            batch = collate([train_ds[i] for i in order[start:start + args.batch_size]], pad_id)
            loss, dec, resp = forward_losses(model, batch, device, args.response_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()
            for i, value in enumerate((loss, dec, resp)):
                totals[i] += float(value) * len(batch["sample_id"])
            seen += len(batch["sample_id"])
        print(f"epoch {epoch}: loss={totals[0]/seen:.4f} decision={totals[1]/seen:.4f} "
              f"response={totals[2]/seen:.4f}", flush=True)
        val = evaluate(model, val_ds, pad_id, device,
                       [x["hard"] for x in val_ds.items], "val")
        if val["val_auc"] > best["auc"]:
            best = {"auc": val["val_auc"], "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in model.named_parameters() if v.requires_grad}}

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    model.save_pretrained(args.output_dir)

    final = {}
    final.update(evaluate(model, train_ds, pad_id, device,
                          [x["hard"] for x in train_ds.items], "train"))
    final.update(evaluate(model, val_ds, pad_id, device,
                          [x["hard"] for x in val_ds.items], "val"))
    report = {
        "recipe": "decision + NoThinking answer SFT" if args.response_weight > 0 else "decision only",
        "lam": args.lam, "soft": not args.hard, "temperature": args.temperature,
        "response_weight": args.response_weight, "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "lora_r": args.lora_r,
        "train_size": len(train_ds), "val_size": len(val_ds),
        "train_with_imitable_answer": imitable,
        "answer_scope": args.answer_scope,
        "best_val_auc": best["auc"], "best_epoch": best["epoch"],
        **base, **final,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)},
                     indent=2, ensure_ascii=False))
    print(f"ADAPTER_SAVED {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
