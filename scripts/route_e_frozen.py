#!/usr/bin/env python3
"""Route E: frozen question representations + one logistic routing head.

No DeepSeek answers are generated. No canonical test data are read.
The source of truth for labels and accuracy/token replay is Route A's
original strict 3000/450 protocol, not the teammate's compliant labels.

Usage:
  python scripts/route_e_frozen.py self-test
  python scripts/route_e_frozen.py check-data
  python scripts/route_e_frozen.py extract --encoder modernbert --smoke
  python scripts/route_e_frozen.py fit --encoder modernbert --smoke
  python scripts/route_e_frozen.py extract --encoder modernbert
  python scripts/route_e_frozen.py fit --encoder modernbert
  # Repeat with --encoder deepseek; see docs/ROUTE_E_FROZEN.md.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "results/router_labels_1p5b/all_3000_labels.jsonl"
VAL = ROOT / "results/val_450_labels.jsonl"
TRAIN_SHA256 = "761dcbb6338de1a3c71def382089c83e8024e0e55af6334dd974963e8d7e0f2c"
VAL_SHA256 = "4c1e984d3744cc089dcc963a30af0ef3d1ed9808c4bfd245a97e48787d501461"
DEFAULT_MODELS = {
    "modernbert": "answerdotai/ModernBERT-base",
    "deepseek": str(ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"),
}
FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_protocol():
    # Import the existing Route A parser and offline budget-matched evaluator.
    from train_modernbert_delta import read_labels
    for path, expected in ((TRAIN, TRAIN_SHA256), (VAL, VAL_SHA256)):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Restore the ORIGINAL strict file, never "
                "substitute repaired or compliant labels.")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Strict data SHA256 mismatch for {path}: {actual} != {expected}")
    train, val = read_labels(TRAIN), read_labels(VAL)
    if len(train) != 3000 or len(val) != 450:
        raise ValueError(f"Expected strict split sizes 3000/450, got {len(train)}/{len(val)}")
    train_ids = {r["sample_id"] for r in train}
    val_ids = {r["sample_id"] for r in val}
    if len(train_ids) != len(train) or len(val_ids) != len(val) or train_ids & val_ids:
        raise ValueError("Duplicate sample IDs or train/validation ID overlap")
    counts = {ds: sum(r["dataset"] == ds for r in val) for ds in ("gsm8k", "math")}
    if counts != {"gsm8k": 300, "math": 150}:
        raise ValueError(f"Unexpected validation composition: {counts}")
    labels = [int(r["p_n"] >= r["p_t"]) for r in train]
    if sum(labels) != 1236:
        raise ValueError(f"Expected 1236 train NoThinking labels, got {sum(labels)}")
    # A textual overlap is reported (not removed) to preserve original Route A protocol.
    norm = lambda s: " ".join(s.casefold().split())
    overlapping_texts = len({norm(r["problem"]) for r in train} &
                            {norm(r["problem"]) for r in val})
    return train, val, overlapping_texts


def default_output(encoder, smoke):
    p = ROOT / "outputs/route_e" / encoder
    return p / "smoke" if smoke else p / "full"


def encode_inputs(rows, tokenizer, encoder, max_length):
    """Encode all questions without silently dropping or truncating any."""
    encoded, lengths = [], []
    for row in rows:
        if encoder == "deepseek":
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["problem"]}],
                tokenize=True, add_generation_prompt=True)
        else:
            ids = tokenizer(row["problem"], add_special_tokens=True,
                            truncation=False)["input_ids"]
        if not ids:
            raise ValueError(f"No input tokens for {row['sample_id']}")
        if len(ids) > max_length:
            raise ValueError(
                f"Prompt too long: {row['sample_id']} has {len(ids)} tokens "
                f"(max_length={max_length}); increase --max-length; no samples skipped.")
        encoded.append(ids)
        lengths.append(len(ids))
    return encoded, lengths


def extract_split(model, tokenizer, encoder, rows, labels_path, destination,
                  max_length, batch_size, model_id, overwrite, smoke):
    import numpy as np
    import torch
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; use --overwrite to replace it")
    started = time.perf_counter()
    encoded, lengths = encode_inputs(rows, tokenizer, encoder, max_length)
    tokenization_seconds = time.perf_counter() - started
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer must define padding token")
    features = []
    inference_seconds = 0.0
    device = torch.device("cuda")
    with torch.inference_mode():
        for begin in range(0, len(rows), batch_size):
            chunk = encoded[begin:begin + batch_size]
            width = max(map(len, chunk))
            ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for i, seq in enumerate(chunk):
                ids[i, :len(seq)] = torch.tensor(seq)
                mask[i, :len(seq)] = 1
            ids, mask = ids.to(device), mask.to(device)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            kwargs = {"input_ids": ids, "attention_mask": mask, "return_dict": True}
            if encoder == "deepseek":
                kwargs["use_cache"] = False
            output = model(**kwargs)
            h = output.last_hidden_state
            if encoder == "modernbert":
                picked = h[:, 0, :]  # pretrained ModernBERT classification position
            else:
                positions = torch.tensor([len(seq)-1 for seq in chunk], device=device)
                picked = h[torch.arange(len(chunk), device=device), positions]
            picked = picked.float().cpu()
            torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - t0
            if not torch.isfinite(picked).all():
                raise ValueError(f"Non-finite {encoder} features at batch {begin}")
            features.append(picked.numpy().astype(np.float16))
            if begin == 0 or begin + batch_size >= len(rows) or begin % (20*batch_size) == 0:
                print(f"EXTRACT {encoder} {destination.stem}: {min(begin+batch_size, len(rows))}/{len(rows)}",
                      flush=True)
    X = np.concatenate(features)
    if X.shape[0] != len(rows) or not np.isfinite(X).all():
        raise ValueError("Feature extraction lost rows or produced non-finite float16 values")
    metadata = {
        "schema": "route_e_features_v1", "encoder": encoder,
        "model": model_id, "torch_dtype": str(next(model.parameters()).dtype),
        "pooling": "cls" if encoder == "modernbert" else "last_prompt_token",
        "prompt": "raw_problem" if encoder == "modernbert" else "user_chat_template_generation_prefix",
        "max_length": max_length, "batch_size": batch_size,
        "labels_sha256": sha256(labels_path),
        "n_samples": len(rows), "feature_dim": int(X.shape[1]),
        "token_length_max": max(lengths),
        "token_length_mean": sum(lengths) / len(lengths),
        "tokenization_seconds": tokenization_seconds,
        "forward_seconds": inference_seconds,
        "forward_ms_per_question": inference_seconds * 1000 / len(rows),
        "total_wall_seconds": time.perf_counter() - started,
        "truncated": 0, "skipped": 0,
        "smoke": smoke,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, X=X,
                        sample_id=np.asarray([r["sample_id"] for r in rows]),
                        metadata_json=np.asarray(json.dumps(metadata)))
    print(f"FEATURES_SAVED {destination} X={X.shape} "
          f"forward_ms_per_question={metadata['forward_ms_per_question']:.2f}", flush=True)
    return metadata


def extract(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires 1 GPU; submit the Slurm job.")
    torch.manual_seed(args.seed)
    train, val, _ = load_protocol()
    model_id = args.model or DEFAULT_MODELS[args.encoder]
    if args.encoder == "deepseek" and not Path(model_id).is_dir():
        raise FileNotFoundError(f"DeepSeek local checkpoint not found: {model_id}")
    local_only = args.encoder == "deepseek"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only,
                                              trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if args.encoder != "deepseek":
            raise ValueError("ModernBERT tokenizer has no padding token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.float16, local_files_only=local_only,
        trust_remote_code=False).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = (("train", train, TRAIN), ("val", val, VAL))
    for name, rows, path in selected:
        if args.smoke:
            rows = rows[:32] if name == "train" else rows[:24]
        out = default_output(args.encoder, args.smoke) / f"{name}.npz"
        extract_split(model, tokenizer, args.encoder, rows, path, out,
                      args.max_length, args.batch_size, model_id, args.overwrite, args.smoke)


def read_features(path, labels, expected_sha, expected_encoder, smoke):
    import numpy as np
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata_json"].item()))
        sample_ids = f["sample_id"].astype(str).tolist()
        X = f["X"].astype(np.float32)
    if meta["schema"] != "route_e_features_v1" or meta["encoder"] != expected_encoder:
        raise ValueError(f"Wrong feature schema/encoder: {path}")
    if meta["labels_sha256"] != expected_sha or bool(meta["smoke"]) != smoke:
        raise ValueError(f"Labels hash or smoke/full status mismatch: {path}")
    if sample_ids != [r["sample_id"] for r in labels]:
        raise ValueError(f"Feature order/IDs do not match strict labels: {path}")
    if X.ndim != 2 or len(X) != len(labels) or not np.isfinite(X).all():
        raise ValueError(f"Invalid feature matrix: {path}")
    return X, meta


def train_logistic(X, y, l2, max_iter):
    """Convex, full-batch hard-label BCE + weight L2; no validation model selection."""
    import numpy as np
    import torch
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = X.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-4)
    Z = torch.from_numpy(((X - mean) / scale).astype(np.float32))
    target = torch.tensor(y, dtype=torch.float32)
    weight = torch.nn.Parameter(torch.zeros(X.shape[1], dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=max_iter,
                                  tolerance_grad=1e-8, tolerance_change=1e-10,
                                  line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        logits = Z @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss = loss + l2 * weight.square().sum()
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        final_loss = float(closure().detach())
        w = weight.numpy().copy()
        b = float(bias)
    return {"weight": w, "bias": b, "mean": mean, "scale": scale,
            "train_objective": final_loss}


def predict_linear(X, fit):
    import numpy as np
    logits = ((X - fit["mean"]) / fit["scale"]) @ fit["weight"] + fit["bias"]
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))).astype(np.float64)


def auc(scores, targets):
    import numpy as np
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    n1 = int(targets.sum())
    n0 = len(targets) - n1
    if not n1 or not n0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.zeros(len(scores), dtype=float)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i+1+j)/2
        i = j
    return float((ranks[targets == 1].sum() - n1*(n1+1)/2) / (n1*n0))


def fit(args):
    import numpy as np
    from train_modernbert_delta import curve_point, endpoints
    train, val, overlaps = load_protocol()
    if args.smoke:
        train, val = train[:32], val[:24]
    source = default_output(args.encoder, args.smoke)
    Xtr, trmeta = read_features(source/"train.npz", train, TRAIN_SHA256,
                                args.encoder, args.smoke)
    Xva, vameta = read_features(source/"val.npz", val, VAL_SHA256,
                                args.encoder, args.smoke)
    if trmeta["model"] != vameta["model"] or trmeta["pooling"] != vameta["pooling"]:
        raise ValueError("Train/val extraction model or pooling mismatch")
    ytr = np.asarray([int(r["p_n"] >= r["p_t"]) for r in train], dtype=np.float32)
    yva = np.asarray([int(r["p_n"] >= r["p_t"]) for r in val], dtype=np.float32)
    if len(set(ytr)) != 2:
        raise ValueError("Training subset contains only one class")
    start = time.perf_counter()
    trained = train_logistic(Xtr, ytr, args.l2, args.max_iter)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    p_nt = predict_linear(Xva, trained)
    predict_ms = (time.perf_counter() - start) * 1000 / len(val)
    predictions = {row["sample_id"]: float(score) for row, score in zip(val, p_nt)}
    # Route A curve_point takes a *smaller is NoThinking* score. Negate p_nt.
    ranked = {sid: -value for sid, value in predictions.items()}
    curve = [curve_point(val, ranked, f) for f in FRACTIONS]
    scores_path = source / "validation_p_nt.json"
    source.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps({"p_nt": predictions}, indent=2) + "\n",
                           encoding="utf-8")
    np.savez_compressed(source/"linear_probe.npz",
                        weight=trained["weight"], bias=np.asarray(trained["bias"]),
                        mean=trained["mean"], scale=trained["scale"])
    result = {
        "method": f"route_e_frozen_{args.encoder}_logistic",
        "smoke": args.smoke,
        "task": "binary_classification", "target": "1[p_nothinking_strict>=p_thinking]",
        "model": trmeta["model"], "encoder_frozen": True,
        "representation": trmeta["pooling"],
        "head": "linear_logistic_regression", "optimizer": "torch_lbfgs",
        "criterion": "hard_label_bce_plus_l2", "l2": args.l2,
        "max_iter": args.max_iter, "seed": args.seed,
        "train_sha256": TRAIN_SHA256, "validation_sha256": VAL_SHA256,
        "train_count": len(train), "val_count": len(val),
        "train_positive_fraction": float(ytr.mean()),
        "val_positive_fraction": float(yva.mean()),
        "validation_auc_diagnostic_only": auc(p_nt, yva),
        "train_objective": trained["train_objective"],
        "exact_train_validation_text_overlaps": overlaps,
        "training_seconds_cpu": train_seconds,
        "prediction_ms_per_question_cpu_batch": predict_ms,
        "train_feature_extraction": trmeta,
        "val_feature_extraction": vameta,
        "router_overhead_included_in_output_tokens": False,
        "llm_generation_repeated": False,
        "random_reference": "Route A cost-matched Global Random, exact expectation",
        "endpoints": endpoints(val), "curve": curve,
    }
    target_dir = ROOT/"results/route_e"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.encoder}_{'smoke' if args.smoke else 'val'}"
    target = target_dir/f"{filename}.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (target_dir/f"{filename}.csv").open("w", newline="", encoding="utf-8") as fd:
        writer = csv.writer(fd)
        writer.writerow(["dataset", "nt_fraction_global", "accuracy", "mean_tokens",
                         "nt_fraction_dataset", "random_accuracy", "uplift_pp"])
        for point in curve:
            for ds, metrics in point["groups"].items():
                random_ref = metrics["matched_random"]
                writer.writerow([ds, point["target_nt_fraction"], metrics["accuracy"],
                                 metrics["mean_tokens"], metrics["nt_fraction"],
                                 None if random_ref is None else random_ref["accuracy"],
                                 metrics["uplift_pp"]])
    for point in curve:
        m = point["groups"]["overall"]
        print(f"ROUTE_E_VAL encoder={args.encoder} smoke={int(args.smoke)} "
              f"NT={point['target_nt_fraction']:.0%} "
              f"acc={m['accuracy']:.4%} tokens={m['mean_tokens']:.1f} "
              f"uplift={m['uplift_pp']:+.2f}pp", flush=True)
    print(f"{'SMOKE_ONLY' if args.smoke else 'ROUTE_E_FULL_VALIDATION'} "
          f"report={target} auc_diagnostic={result['validation_auc_diagnostic_only']}",
          flush=True)


def self_test():
    import numpy as np
    from train_modernbert_delta import curve_point
    rows = [
        {"sample_id": "m:1", "dataset": "m", "p_t": 1.0, "p_n": 0.0,
         "t_t": 100.0, "t_n": 20.0},
        {"sample_id": "m:2", "dataset": "m", "p_t": 0.0, "p_n": 1.0,
         "t_t": 100.0, "t_n": 20.0},
    ]
    p_nt = {"m:1": .1, "m:2": .9}
    result = curve_point(rows, {k: -v for k,v in p_nt.items()}, .5)
    assert result["groups"]["overall"]["accuracy"] == 1.0
    assert result["groups"]["overall"]["mean_tokens"] == 60.0
    assert auc([.1, .9], [0, 1]) == 1.0
    X = np.asarray([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]], dtype=np.float32)
    y = np.asarray([0., 0., 1., 1.], dtype=np.float32)
    fit_result = train_logistic(X, y, l2=1e-3, max_iter=75)
    scores = predict_linear(X, fit_result)
    assert scores[0] < scores[1] < scores[2] < scores[3]
    assert (scores[:2] < .5).all() and (scores[2:] > .5).all()
    print("ROUTE_E_SELFTEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    sub.add_parser("check-data")
    for cmd in ("extract", "fit"):
        p = sub.add_parser(cmd)
        p.add_argument("--encoder", required=True, choices=tuple(DEFAULT_MODELS))
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--seed", type=int, default=42)
        if cmd == "extract":
            p.add_argument("--model", default=None)
            p.add_argument("--batch-size", type=int, default=8)
            p.add_argument("--max-length", type=int, default=1024)
            p.add_argument("--overwrite", action="store_true")
        else:
            p.add_argument("--l2", type=float, default=1e-3)
            p.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "check-data":
        train, val, overlaps = load_protocol()
        print(f"ROUTE_E_DATA_OK train={len(train)} val={len(val)} "
              f"train_positive={sum(r['p_n']>=r['p_t'] for r in train)} "
              f"overlapping_texts={overlaps}")
    elif args.command == "extract":
        if args.batch_size <= 0 or args.max_length <= 0:
            parser.error("batch-size and max-length must be positive")
        extract(args)
    else:
        if not math.isfinite(args.l2) or args.l2 < 0 or args.max_iter <= 0:
            parser.error("l2 must be finite and nonnegative; max-iter positive")
        fit(args)


if __name__ == "__main__":
    main()
