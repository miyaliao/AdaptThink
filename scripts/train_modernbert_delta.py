#!/usr/bin/env python3
"""ModernBERT Delta Regression: full fine-tuning and offline validation replay.

Input: question text. Target: empirical p_thinking - p_nothinking.
The frozen DeepSeek model is NOT run; the canonical test set is NOT read.

Commands:
  python scripts/train_modernbert_delta.py --check-data
  python scripts/train_modernbert_delta.py --self-test
  python scripts/train_modernbert_delta.py --smoke          # one GPU, small subset
  python scripts/train_modernbert_delta.py                  # one GPU, full train+eval
  python scripts/train_modernbert_delta.py --eval-only      # CPU, saved predictions
"""
import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "results/router_labels_1p5b/all_3000_labels.jsonl"
VAL = ROOT / "results/val_450_labels.jsonl"
OUT = ROOT / "outputs/external_modernbert/delta"
REPORT = ROOT / "results/external_modernbert/delta_val.json"
FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_labels(path):
    rows, ids = [], set()
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            sid = str(raw["sample_id"])
            if sid in ids:
                raise ValueError(f"Duplicate sample_id {sid} in {path}:{number}")
            ids.add(sid)
            problem = raw["problem"]
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"Missing problem text for {sid}")
            p_t = float(raw["p_thinking"])
            # Only the strict math scorer counts; never use repaired answers.
            p_n = float(raw.get("p_nothinking_strict", raw["p_nothinking"]))
            t_t = float(raw["mean_tokens_thinking"])
            t_n = float(raw["mean_tokens_nothinking"])
            if not all(math.isfinite(v) for v in (p_t, p_n, t_t, t_n)):
                raise ValueError(f"Nonfinite input for {sid}")
            if not (0 <= p_t <= 1 and 0 <= p_n <= 1 and t_t >= 0 and t_n >= 0):
                raise ValueError(f"Invalid probability or cost for {sid}")
            delta = p_t - p_n
            if "gain_thinking" in raw and "p_nothinking_strict" not in raw:
                if abs(float(raw["gain_thinking"]) - delta) > 1e-6:
                    raise ValueError(f"gain_thinking mismatch for {sid}")
            rows.append({
                "sample_id": sid, "problem": problem, "delta": delta,
                "p_t": p_t, "p_n": p_n, "t_t": t_t, "t_n": t_n,
                "dataset": sid.split(":")[0],
            })
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def endpoints(rows):
    n = len(rows)
    return {
        "thinking": {"accuracy": sum(r["p_t"] for r in rows)/n,
                     "tokens": sum(r["t_t"] for r in rows)/n},
        "nothinking": {"accuracy": sum(r["p_n"] for r in rows)/n,
                       "tokens": sum(r["t_n"] for r in rows)/n},
    }


def random_at_budget(tokens, modes):
    t, nt = modes["thinking"], modes["nothinking"]
    difference = t["tokens"] - nt["tokens"]
    if abs(difference) < 1e-12:
        return None
    q = (t["tokens"] - tokens) / difference
    if not -1e-9 <= q <= 1 + 1e-9:
        return None  # outside the achievable endpoint budget range
    q = max(0.0, min(1.0, q))
    return {"nt_probability": q,
            "accuracy": (1-q)*t["accuracy"] + q*nt["accuracy"]}


def curve_point(rows, preds, nt_fraction):
    # A score computed from ONLY the problem, never its gold success or length.
    n_nt = math.floor(len(rows)*nt_fraction + 0.5)
    ranked = sorted(rows, key=lambda r: (preds[r["sample_id"]], r["sample_id"]))
    nt_ids = {r["sample_id"] for r in ranked[:n_nt]}
    by_dataset = {"overall": rows}
    for key in sorted(set(r["dataset"] for r in rows)):
        by_dataset[key] = [r for r in rows if r["dataset"] == key]
    groups = {}
    for name, group in by_dataset.items():
        n = len(group)
        actual_nt = sum(r["sample_id"] in nt_ids for r in group)
        acc = sum(r["p_n"] if r["sample_id"] in nt_ids else r["p_t"] for r in group)/n
        tok = sum(r["t_n"] if r["sample_id"] in nt_ids else r["t_t"] for r in group)/n
        reference = random_at_budget(tok, endpoints(group))
        groups[name] = {
            "n": n, "n_nothinking": actual_nt, "nt_fraction": actual_nt/n,
            "accuracy": acc, "mean_tokens": tok, "matched_random": reference,
            "uplift_pp": None if reference is None else 100*(acc-reference["accuracy"]),
        }
    return {"target_nt_fraction": nt_fraction, "groups": groups}


def evaluate(labels, predictions, metadata_path, labels_path, result_path):
    if len(labels) != 450:
        raise ValueError("Full validation must contain 450 problems")
    counts = {k: sum(r["dataset"] == k for r in labels)
              for k in set(r["dataset"] for r in labels)}
    if counts != {"gsm8k": 300, "math": 150}:
        raise ValueError(f"Validation split mismatch: {counts}")
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    if meta["val_sha256"] != sha256(labels_path) or meta.get("smoke"):
        raise ValueError("Validation hash mismatch or smoke predictions")
    if set(predictions) != {r["sample_id"] for r in labels}:
        raise ValueError("Prediction IDs must match all 450 validation IDs")
    if not all(math.isfinite(v) for v in predictions.values()):
        raise ValueError("Nonfinite router predictions")
    curve = [curve_point(labels, predictions, f) for f in FRACTIONS]
    summary = {
        "method": "modernbert_delta_mse", "target": "p_thinking-p_nothinking",
        "decision": "choose NoThinking for smallest predicted delta",
        "validation_sha256": sha256(labels_path),
        "predicted_cost_leakage": False,
        "llm_generation_repeated": False,
        "router_overhead_included": False,
        "scorer": "strict if available; otherwise committed p_nothinking",
        "random_reference": "exact expectation at matched mean output-token budget",
        "endpoints": endpoints(labels), "curve": curve,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = result_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        w = csv.writer(output)
        w.writerow(["dataset", "nt_fraction_global", "nt_fraction_in_dataset",
                    "accuracy", "mean_tokens", "random_accuracy", "uplift_pp"])
        for item in curve:
            for dataset, metric in item["groups"].items():
                ref = metric["matched_random"]
                w.writerow([dataset, item["target_nt_fraction"],
                            metric["nt_fraction"], metric["accuracy"],
                            metric["mean_tokens"],
                            None if ref is None else ref["accuracy"],
                            metric["uplift_pp"]])
    for item in curve:
        v = item["groups"]["overall"]
        upl = "N/A" if v["uplift_pp"] is None else f"{v['uplift_pp']:+.2f} pp"
        print(f"VAL NT={item['target_nt_fraction']:.0%} "
              f"acc={v['accuracy']:.4%} tokens={v['mean_tokens']:.1f} "
              f"uplift_vs_random={upl}", flush=True)
    print(f"DELTA_EVAL_DONE {result_path} {csv_path}", flush=True)


class DeltaDataset:
    def __init__(self, data, tokenizer, max_length):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.data[idx]["problem"], truncation=True, max_length=self.max_length)
        encoded["labels"] = float(self.data[idx]["delta"])
        return encoded


def self_test():
    rows = [
        {"sample_id": "a:0", "dataset": "a", "p_t": 1., "p_n": 0.,
         "t_t": 100., "t_n": 20.},
        {"sample_id": "a:1", "dataset": "a", "p_t": 0., "p_n": 1.,
         "t_t": 100., "t_n": 20.},
    ]
    pred = {"a:0": 0.7, "a:1": -0.7}
    result = curve_point(rows, pred, 0.5)["groups"]["overall"]
    assert result["n_nothinking"] == 1
    assert result["accuracy"] == 1.0
    assert result["mean_tokens"] == 60.0
    ref = random_at_budget(150., {
        "thinking": {"accuracy": 0.8, "tokens": 200.},
        "nothinking": {"accuracy": 0.6, "tokens": 100.},
    })
    assert abs(ref["accuracy"] - 0.7) < 1e-12
    assert random_at_budget(250., {
        "thinking": {"accuracy": 0.8, "tokens": 200.},
        "nothinking": {"accuracy": 0.6, "tokens": 100.},
    }) is None
    print("DELTA_SELF_TEST_PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-labels", type=Path, default=TRAIN)
    ap.add_argument("--val-labels", type=Path, default=VAL)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-data", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    train = read_labels(args.train_labels)
    val = read_labels(args.val_labels)
    if len(train) != 3000 or len(val) != 450:
        raise RuntimeError(f"Frozen split sizes expected 3000/450, got {len(train)}/{len(val)}")
    if {r["sample_id"] for r in train} & {r["sample_id"] for r in val}:
        raise RuntimeError("Train and validation sample IDs overlap")
    print(f"DATA_OK train={len(train)} val={len(val)} "
          f"train_delta_mean={sum(r['delta'] for r in train)/len(train):.4f}",
          flush=True)
    if args.check_data:
        return

    out = args.output_dir
    if args.eval_only:
        pred_path = out / "validation_predictions.jsonl"
        predictions = {}
        with pred_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sid = row["sample_id"]
                if sid in predictions:
                    raise ValueError(f"Duplicate prediction ID: {sid}")
                predictions[sid] = float(row["predicted_delta"])
        evaluate(val, predictions, out / "run_metadata.json",
                 args.val_labels, args.report)
        return

    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("Invalid training arguments")
    # Avoid importing transformers or torch on the login-node preflight.
    import numpy as np
    import torch
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments,
                              set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Use Slurm allocation with 1 GPU; do not train on login node")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.smoke:
        out = out.parent / (out.name + "_smoke")
        train, val = train[:32], val[:24]
        args.epochs = 1.
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, problem_type="regression",
        trust_remote_code=False)
    assert model.config.num_labels == 1
    assert model.config.problem_type == "regression"
    kwargs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        optim="adamw_torch",
        dataloader_num_workers=0,
        report_to="none",
        logging_steps=25,
        seed=args.seed,
        data_seed=args.seed)
    training = DeltaDataset(train, tokenizer, args.max_length)
    validation = DeltaDataset(val, tokenizer, args.max_length)
    trainer = Trainer(
        model=model, args=kwargs, train_dataset=training,
        eval_dataset=validation,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer)
    trainer.train()
    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("No best checkpoint was selected")
    trainer.save_model(str(out / "best_model"))
    tokenizer.save_pretrained(str(out / "best_model"))
    scores = np.asarray(trainer.predict(validation).predictions).reshape(-1)
    if len(scores) != len(val):
        raise RuntimeError("Prediction count mismatch")
    pred_path = out / "validation_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for row, score in zip(val, scores):
            if not math.isfinite(float(score)):
                raise ValueError("Nonfinite predicted delta")
            f.write(json.dumps({"sample_id": row["sample_id"],
                                "predicted_delta": float(score)}) + "\n")
    metadata = {
        "task": "delta_regression", "loss": "mse",
        "train_sha256": sha256(args.train_labels),
        "val_sha256": sha256(args.val_labels),
        "model": args.model, "seed": args.seed,
        "learning_rate": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "train_count": len(train), "val_count": len(val),
        "smoke": args.smoke,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_mse": float(trainer.state.best_metric),
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"BEST_CHECKPOINT {metadata['best_checkpoint']}")
    print(f"BEST_VAL_MSE {metadata['best_val_mse']:.6f}")
    print(f"DELTA_TRAIN_DONE {pred_path}", flush=True)
    if not args.smoke:
        predictions = {row["sample_id"]: float(value)
                       for row, value in zip(val, scores)}
        evaluate(val, predictions, out / "run_metadata.json",
                 args.val_labels, args.report)


if __name__ == "__main__":
    main()
