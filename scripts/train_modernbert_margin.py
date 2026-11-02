#!/usr/bin/env python3
"""Route A: ModernBERT Dual MSE / Dual MSE + Margin, offline strict validation.

Run either objective independently, with identical model/data/training settings:
  python scripts/train_modernbert_margin.py --self-test
  python scripts/train_modernbert_margin.py --check-data
  python scripts/train_modernbert_margin.py --mode dual_mse --smoke
  python scripts/train_modernbert_margin.py --mode dual_margin --smoke
  python scripts/train_modernbert_margin.py --mode dual_mse
  python scripts/train_modernbert_margin.py --mode dual_margin --beta 1.0
  python scripts/train_modernbert_margin.py --mode dual_margin --eval-only

The only router input is the question text. No DeepSeek generation or test set
is used. Targets and offline replay use the strict original scorer. Do NOT
substitute the teammate's repaired / compliant 2777-label set.

Both heads output two *bounded probabilities* via sigmoid(logits).
  L_dual   = mean_over_questions(0.5 * ((pT_hat-pT)^2 + (pN_hat-pN)^2))
  L_margin = mean_over_questions(((pT_hat-pN_hat)-(pT-pN))^2)
  L_total  = L_dual + beta * L_margin       # dual_margin only
"""
import argparse
import csv
import json
import math
import random
from pathlib import Path

from train_modernbert_delta import (
    ROOT, TRAIN, VAL, FRACTIONS, sha256, read_labels, endpoints, curve_point,
)

BASE_OUTPUT = ROOT / "outputs/external_modernbert"
BASE_RESULT = ROOT / "results/external_modernbert"
MODES = ("dual_mse", "dual_margin")


def pure_loss(pred, target, beta):
    """Loss reference without torch, used to check objective algebra."""
    if len(pred) != 2 or len(target) != 2:
        raise ValueError("Expected [p_thinking, p_nothinking] pairs")
    dual = 0.5 * sum((a - b) ** 2 for a, b in zip(pred, target))
    margin = ((pred[0] - pred[1]) - (target[0] - target[1])) ** 2
    return dual, margin, dual + beta * margin


def self_test():
    # The margin is exactly zero when the common probability error cancels.
    d, m, total = pure_loss([0.7, 0.5], [0.6, 0.4], beta=1.0)
    assert abs(d - 0.01) < 1e-12 and abs(m) < 1e-12
    assert abs(total - 0.01) < 1e-12
    d, m, total = pure_loss([0.8, 0.2], [0.4, 0.6], beta=1.0)
    assert abs(d - 0.16) < 1e-12 and abs(m - 0.64) < 1e-12
    assert abs(total - 0.80) < 1e-12
    sample = [
        {"sample_id": "x:1", "dataset": "x", "p_t": 1., "p_n": 0.,
         "t_t": 100., "t_n": 20.},
        {"sample_id": "x:2", "dataset": "x", "p_t": 0., "p_n": 1.,
         "t_t": 100., "t_n": 20.},
    ]
    result = curve_point(sample, {"x:1": 0.9, "x:2": -0.8}, 0.5)
    group = result["groups"]["overall"]
    assert group["n_nothinking"] == 1
    assert group["accuracy"] == 1.0 and group["mean_tokens"] == 60.0
    assert group["uplift_pp"] is not None and group["uplift_pp"] > 0
    print("MARGIN_SELF_TEST_PASS")


class DualDataset:
    def __init__(self, rows, tokenizer, max_length):
        self.rows, self.tokenizer, self.max_length = rows, tokenizer, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = self.tokenizer(row["problem"], truncation=True,
                              max_length=self.max_length)
        item["labels"] = [float(row["p_t"]), float(row["p_n"])]
        return item


def read_predictions(path):
    predictions = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row["sample_id"])
            if sid in predictions:
                raise ValueError(f"Duplicate prediction ID: {sid}")
            p_t = float(row["predicted_p_thinking"])
            p_n = float(row["predicted_p_nothinking"])
            delta = float(row["predicted_delta"])
            if not all(math.isfinite(z) for z in (p_t, p_n, delta)):
                raise ValueError(f"Nonfinite prediction for {sid}")
            if not (0 <= p_t <= 1 and 0 <= p_n <= 1):
                raise ValueError(f"Prediction not a probability for {sid}")
            if abs((p_t - p_n) - delta) > 1e-5:
                raise ValueError(f"Predicted delta mismatch for {sid}")
            predictions[sid] = {"p_t": p_t, "p_n": p_n, "delta": delta}
    return predictions


def evaluate(labels, predictions, metadata_path, train_path, val_path,
             result_path, mode, beta):
    if len(labels) != 450:
        raise ValueError(f"Expected 450 validation questions, got {len(labels)}")
    counts = {name: sum(row["dataset"] == name for row in labels)
              for name in set(row["dataset"] for row in labels)}
    if counts != {"gsm8k": 300, "math": 150}:
        raise ValueError(f"Validation split mismatch: {counts}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["val_sha256"] != sha256(val_path):
        raise RuntimeError("Validation file hash changed since training")
    if metadata["train_sha256"] != sha256(train_path):
        raise RuntimeError("Training label file hash changed since training")
    if metadata.get("smoke"):
        raise RuntimeError("Cannot evaluate smoke predictions as full validation")
    if metadata["mode"] != mode or abs(metadata["beta"] - beta) > 1e-9:
        raise RuntimeError("Checkpoint mode / beta mismatch")
    ids = {row["sample_id"] for row in labels}
    if set(predictions) != ids:
        missing = sorted(ids - set(predictions))[:5]
        extra = sorted(set(predictions) - ids)[:5]
        raise ValueError(f"Prediction ID mismatch missing={missing}, extra={extra}")

    # The common decision score is pT_hat - pN_hat, smallest routed NoThinking.
    scores = {sid: pair["delta"] for sid, pair in predictions.items()}
    curve = [curve_point(labels, scores, fraction) for fraction in FRACTIONS]
    # These error metrics are *descriptive*, not checkpoint-selection metrics.
    mse_t = sum((predictions[r["sample_id"]]["p_t"] - r["p_t"])**2
                for r in labels) / len(labels)
    mse_n = sum((predictions[r["sample_id"]]["p_n"] - r["p_n"])**2
                for r in labels) / len(labels)
    delta_mse = sum((predictions[r["sample_id"]]["delta"] -
                     (r["p_t"] - r["p_n"]))**2 for r in labels) / len(labels)
    summary = {
        "method": "modernbert_" + mode,
        "loss": "dual_mse" if mode == "dual_mse" else "dual_mse_plus_margin",
        "beta": beta,
        "target": ["strict_p_thinking", "strict_p_nothinking"],
        "decision": "choose NoThinking for smallest predicted (pT-pN)",
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(val_path),
        "strict_scoring": True,
        "validation_metrics": {
            "mse_thinking": mse_t, "mse_nothinking": mse_n,
            "dual_mse": (mse_t + mse_n) / 2,
            "delta_mse": delta_mse,
        },
        "predicted_cost_leakage": False,
        "llm_generation_repeated": False,
        "router_overhead_included": False,
        "random_reference": "exact expectation at matched mean output-token budget",
        "endpoints": endpoints(labels),
        "curve": curve,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    csv_path = result_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["dataset", "nt_fraction_global", "nt_fraction_in_dataset",
                         "accuracy", "mean_tokens", "random_accuracy", "uplift_pp"])
        for item in curve:
            for name, metric in item["groups"].items():
                reference = metric["matched_random"]
                writer.writerow([
                    name, item["target_nt_fraction"], metric["nt_fraction"],
                    metric["accuracy"], metric["mean_tokens"],
                    None if reference is None else reference["accuracy"],
                    metric["uplift_pp"],
                ])
    for item in curve:
        point = item["groups"]["overall"]
        uplift = "N/A" if point["uplift_pp"] is None else f"{point['uplift_pp']:+.2f} pp"
        print(f"MARGIN_VAL mode={mode} NT={item['target_nt_fraction']:.0%} "
              f"acc={point['accuracy']:.4%} tokens={point['mean_tokens']:.1f} "
              f"uplift_vs_random={uplift}", flush=True)
    print(f"MARGIN_EVAL_DONE mode={mode} {result_path} {csv_path}", flush=True)


def make_trainer_type(Trainer, torch, mode, beta):
    """Keep all training changes in the loss; predictions remain raw logits."""
    class DualTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            labels = inputs["labels"].float()
            model_inputs = {key: value for key, value in inputs.items()
                            if key != "labels"}
            outputs = model(**model_inputs)
            probabilities = torch.sigmoid(outputs.logits.float())
            if probabilities.shape != labels.shape or labels.shape[-1] != 2:
                raise ValueError(f"Dual targets/logits shape mismatch: "
                                 f"{labels.shape} / {probabilities.shape}")
            errors = probabilities - labels
            dual_loss = errors.square().mean()
            margin_error = (probabilities[:, 0] - probabilities[:, 1]) - (
                labels[:, 0] - labels[:, 1])
            margin_loss = margin_error.square().mean()
            loss = dual_loss + (beta * margin_loss if mode == "dual_margin" else 0.)
            return (loss, outputs) if return_outputs else loss
    return DualTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="dual_margin")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Weight of margin loss; ignored by dual_mse mode")
    parser.add_argument("--train-labels", type=Path, default=TRAIN)
    parser.add_argument("--val-labels", type=Path, default=VAL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.smoke and args.eval_only:
        parser.error("--smoke and --eval-only cannot be combined")
    if not math.isfinite(args.beta) or args.beta < 0:
        parser.error("--beta must be finite and nonnegative")
    if args.mode == "dual_mse":
        args.beta = 0.0
    if args.mode == "dual_margin" and args.beta <= 0:
        parser.error("dual_margin requires --beta > 0; use --mode dual_mse instead")
    if args.epochs <= 0 or args.batch_size <= 0 or args.max_length <= 0:
        parser.error("Invalid training hyperparameters")
    # A nonstandard beta must not silently overwrite the default beta=1 artifacts.
    name = args.mode if (args.mode == "dual_mse" or args.beta == 1.0) else (
        f"dual_margin_beta{args.beta:g}")
    out = args.output_dir or (BASE_OUTPUT / name)
    report = args.report or (BASE_RESULT / f"{name}_val.json")
    if args.smoke:
        out = out.parent / (out.name + "_smoke")
    if not args.train_labels.is_file():
        raise FileNotFoundError(
            f"Missing original strict 3000-label file: {args.train_labels}\n"
            "Check the NTU server-local labels; do NOT substitute "
            "router_labels_1p5b_compliant (2777) or repaired labels.")
    if not args.val_labels.is_file():
        raise FileNotFoundError(f"Missing validation labels: {args.val_labels}")
    train = read_labels(args.train_labels)
    validation = read_labels(args.val_labels)
    if len(train) != 3000 or len(validation) != 450:
        raise RuntimeError(f"Expected original 3000/450 split; got "
                           f"{len(train)}/{len(validation)}. Never use the 2777 set.")
    train_ids = {r["sample_id"] for r in train}
    if train_ids & {r["sample_id"] for r in validation}:
        raise RuntimeError("Train / validation sample IDs overlap")
    print(f"MARGIN_DATA_OK train={len(train)} val={len(validation)} "
          f"mode={args.mode} beta={args.beta:g} "
          f"train_sha256={sha256(args.train_labels)} "
          f"val_sha256={sha256(args.val_labels)}", flush=True)
    if args.check_data:
        return

    if args.eval_only:
        predictions = read_predictions(out / "validation_predictions.jsonl")
        evaluate(validation, predictions, out / "run_metadata.json",
                 args.train_labels, args.val_labels, report, args.mode, args.beta)
        return

    # Avoid transformer / CUDA imports on the login node during preflight.
    import numpy as np
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
        Trainer, TrainingArguments, set_seed,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; submit the Slurm script, not the login node.")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.smoke:
        train, validation = train[:32], validation[:24]
        args.epochs = 1.0
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, problem_type="regression",
        trust_remote_code=False)
    if model.config.num_labels != 2:
        raise RuntimeError("Unexpected classification head size")
    training_args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01, warmup_ratio=0.1, lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, fp16=True, optim="adamw_torch",
        dataloader_num_workers=0, report_to="none", logging_steps=25,
        seed=args.seed, data_seed=args.seed)
    DualTrainer = make_trainer_type(Trainer, torch, args.mode, args.beta)
    trainer = DualTrainer(
        model=model, args=training_args,
        train_dataset=DualDataset(train, tokenizer, args.max_length),
        eval_dataset=DualDataset(validation, tokenizer, args.max_length),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer)
    trainer.train()
    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("Best checkpoint was not saved")
    trainer.save_model(str(out / "best_model"))
    tokenizer.save_pretrained(str(out / "best_model"))
    raw_logits = np.asarray(trainer.predict(
        DualDataset(validation, tokenizer, args.max_length)).predictions)
    if raw_logits.shape != (len(validation), 2):
        raise RuntimeError(f"Expected (N,2) logits, got {raw_logits.shape}")
    # NumPy sigmoid with stable clipping for any extreme model logits.
    bounded_logits = np.clip(raw_logits.astype(np.float64), -60, 60)
    probabilities = 1.0 / (1.0 + np.exp(-bounded_logits))
    pred_path = out / "validation_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as output:
        for row, pair in zip(validation, probabilities):
            p_t, p_n = map(float, pair)
            output.write(json.dumps({
                "sample_id": row["sample_id"], "predicted_p_thinking": p_t,
                "predicted_p_nothinking": p_n,
                "predicted_delta": p_t - p_n,
            }) + "\n")
    metadata = {
        "task": "dual_regression", "mode": args.mode,
        "loss": "dual_mse" if args.mode == "dual_mse" else "dual_mse+margin_mse",
        "beta": args.beta, "probability_link": "sigmoid",
        "train_sha256": sha256(args.train_labels),
        "val_sha256": sha256(args.val_labels),
        "train_label_path": str(args.train_labels),
        "val_label_path": str(args.val_labels),
        "model": args.model, "seed": args.seed, "lr": args.lr,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "max_length": args.max_length, "train_count": len(train),
        "val_count": len(validation), "smoke": args.smoke,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_val_objective": float(trainer.state.best_metric),
        "epoch_history": trainer.state.log_history,
    }
    (out / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"MARGIN_BEST mode={args.mode} checkpoint={metadata['best_checkpoint']} "
          f"eval_loss={metadata['best_val_objective']:.6f}", flush=True)
    print(f"MARGIN_TRAIN_DONE mode={args.mode} {pred_path}", flush=True)
    if not args.smoke:
        predictions = read_predictions(pred_path)
        evaluate(validation, predictions, out / "run_metadata.json",
                 args.train_labels, args.val_labels, report, args.mode, args.beta)


if __name__ == "__main__":
    main()
