#!/usr/bin/env python3
"""Route E / P1: frozen DeepSeek hidden-layer and lightweight-head ablation.

Only the original strict 3000/450 Route E protocol is supported. No test set
or LLM answers are read/generated, and no DeepSeek weights are updated.

Workflow:
    python scripts/route_e_layer_head.py self-test
    python scripts/route_e_layer_head.py check-cache [--smoke]
    python scripts/route_e_layer_head.py extract --smoke
    python scripts/route_e_layer_head.py fit-logistic --smoke
    python scripts/route_e_layer_head.py extract
    python scripts/route_e_layer_head.py fit-logistic
    python scripts/route_e_layer_head.py fit-mlp --layer 14
    python scripts/route_e_layer_head.py compare --ft-scores PATH --ft-field FIELD
"""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import route_e_frozen as base
import route_e_prompt_ablation as prefix
import route_e_equal_budget as budget_eval

ROOT = base.ROOT
LAYERS = (7, 14, 21, 28)
BUDGETS = budget_eval.DEFAULT_BUDGETS


def output_dir(smoke):
    return ROOT / "outputs/route_e/layer_head" / ("smoke" if smoke else "full")


def results_dir():
    return ROOT / "results/route_e/layer_head"


def expected_rows(smoke):
    train, val, overlaps = base.load_protocol()
    if smoke:
        train, val = base.choose_smoke_rows(train, val)
    return train, val, overlaps


def check_p1(train, val, smoke, max_length=None, batch_size=None, model_path=None):
    """Require *existing*, original P1 arrays and exact order/label hashes."""
    paths = prefix.cache_dir("assistant", smoke)
    ref = {}
    for split, rows, sha in (("train", train, base.TRAIN_SHA256),
                             ("val", val, base.VAL_SHA256)):
        path = paths / f"{split}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing original P1 cache {path}; do not regenerate over it")
        X, meta = prefix.read_cache(path, rows, sha, "assistant", smoke)
        if meta["layer"] != "last_after_final_norm" or meta["pooling"] != "last_valid_prompt_token":
            raise ValueError(f"Wrong P1 final layer/pooling: {path}")
        for key, value in (("max_length", max_length), ("batch_size", batch_size)):
            if value is not None and int(meta[key]) != value:
                raise ValueError(f"P1 {key}={meta[key]} != requested {value}; keep settings identical")
        if model_path is not None and Path(meta["model"]).resolve() != Path(model_path).resolve():
            raise ValueError(f"Model checkpoint differs from existing P1: {meta['model']}")
        ref[split] = (X, meta)
    for key in ("model", "chat_template_sha256", "max_length", "batch_size", "dtype"):
        if ref["train"][1][key] != ref["val"][1][key]:
            raise ValueError(f"P1 train/val metadata disagreement on {key}")
    return ref


def check_cache(args):
    train, val, overlaps = expected_rows(args.smoke)
    refs = check_p1(train, val, args.smoke)
    print(f"LAYER_HEAD_CACHE_OK smoke={int(args.smoke)} train={len(train)} "
          f"val={len(val)} text_overlap={overlaps}")
    for split, (X, meta) in refs.items():
        print(f"P1_REFERENCE {split} shape={X.shape} "
              f"batch_size={meta['batch_size']} max_length={meta['max_length']} "
              f"template_sha256={meta['chat_template_sha256']}")


def compare_final_reference(X_new, X_ref, split):
    """Regression gate to ensure index 28 agrees with original final P1."""
    import numpy as np
    if X_new.shape != X_ref.shape:
        raise ValueError(f"Final-layer P1 shape mismatch in {split}")
    a, b = X_new.astype(np.float64), X_ref.astype(np.float64)
    d = a-b
    l2 = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-10)
    cos = np.sum(a*b, axis=1) / np.maximum(
        np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1), 1e-10)
    stats = {"mean_abs_diff": float(np.abs(d).mean()),
             "max_abs_diff": float(np.abs(d).max()),
             "mean_relative_l2": float(l2.mean()),
             "p99_relative_l2": float(np.percentile(l2, 99)),
             "mean_cosine": float(cos.mean()), "min_cosine": float(cos.min())}
    print(f"P1_FINAL_REPRO {split} {json.dumps(stats)}", flush=True)
    if stats["min_cosine"] < .995 or stats["p99_relative_l2"] > .10:
        raise RuntimeError(f"Hidden state index 28 does not reproduce original P1 ({split}); "
                           "check model, norm, padding, template and dtype")
    return stats


def atomic_npz(path, arrays):
    """Write an untracked cache without replacing any existing file."""
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}; inspect/remove failed cache explicitly")
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as fp:
            np.savez_compressed(fp, **arrays)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite concurrently created {path}")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def extract(args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("One CUDA GPU is required: submit the NTU Slurm script")
    train, val, _ = expected_rows(args.smoke)
    model_path = Path(args.model).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local DeepSeek checkpoint missing: {model_path}")
    p1 = check_p1(train, val, args.smoke, args.max_length, args.batch_size, model_path)
    folder = output_dir(args.smoke)
    # Check both destinations before loading the model, so retry cannot clobber a split.
    for split in ("train", "val"):
        if (folder/f"{split}.npz").exists():
            raise FileExistsError(f"{folder}/{split}.npz already exists")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path),
        local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    template_sha = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
    if template_sha != p1["train"][1]["chat_template_sha256"]:
        raise ValueError("P1 tokenizer template changed")
    # Fail closed unless this is *exactly* the historical Assistant-only P1.
    prefix.render_prompts(tokenizer, "What is 25 * 16?")
    model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.float16,
        local_files_only=True, trust_remote_code=False).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n = int(model.config.num_hidden_layers)
    if n != 28:
        raise ValueError(f"Expected DeepSeek 1.5B with 28 transformer blocks, got {n}")
    hidden_dim = int(model.config.hidden_size)
    if hidden_dim != 1536:
        raise ValueError(f"Expected 1536-dimensional states, got {hidden_dim}")
    for split, rows, labels_path in (("train", train, base.TRAIN), ("val", val, base.VAL)):
        start = time.perf_counter()
        sequences = prefix.tokenize_ablations(tokenizer, [r["problem"] for r in rows],
                                               "assistant", args.max_length)
        batch_times = []
        features = {k: [] for k in LAYERS}
        with torch.inference_mode():
            for offset in range(0, len(rows), args.batch_size):
                batch = sequences[offset:offset+args.batch_size]
                width = max(map(len, batch))
                ids = torch.full((len(batch), width), tokenizer.pad_token_id,
                                 device="cuda", dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, seq in enumerate(batch):
                    ids[j, :len(seq)] = torch.tensor(seq, device="cuda")
                    mask[j, :len(seq)] = 1
                pos = torch.tensor([len(seq)-1 for seq in batch], device="cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = model(input_ids=ids, attention_mask=mask,
                            use_cache=False, output_hidden_states=True, return_dict=True)
                states = out.hidden_states
                if len(states) != n+1 or not torch.equal(states[-1], out.last_hidden_state):
                    raise RuntimeError("Unexpected Transformers hidden_states indexing or final norm")
                rows_idx = torch.arange(len(batch), device="cuda")
                for layer in LAYERS:
                    picked = states[layer][rows_idx, pos].float().cpu()
                    if not torch.isfinite(picked).all():
                        raise ValueError(f"Nonfinite layer {layer} features in {split} offset {offset}")
                    features[layer].append(picked.numpy().astype(np.float16))
                torch.cuda.synchronize()
                batch_times.append((time.perf_counter()-t0)*1000)
                # Release references to all 29 full-sequence hidden-state tensors
                # BEFORE allocating the next batch, keeping peak GPU memory low.
                del out, states, picked, ids, mask, pos, rows_idx
                if offset == 0 or offset+args.batch_size >= len(rows) or (
                        offset//args.batch_size)%20 == 0:
                    print(f"LAYER_EXTRACT split={split} n={min(offset+args.batch_size,len(rows))}/{len(rows)}",
                          flush=True)
        matrices = {f"X_layer_{k:02d}": np.concatenate(features[k]) for k in LAYERS}
        for k in LAYERS:
            if matrices[f"X_layer_{k:02d}"].shape != (len(rows), hidden_dim):
                raise ValueError(f"Layer {k} lost samples")
        p1_check = compare_final_reference(matrices["X_layer_28"], p1[split][0], split)
        metadata = {
            "schema": "route_e_layer_head_v1", "smoke": args.smoke,
            "prompt_mode": "assistant", "model": str(model_path),
            "model_revision": getattr(model.config, "_commit_hash", None),
            "num_transformer_blocks": n, "hidden_dim": hidden_dim,
            "layer_indices": list(LAYERS),
            "layer_semantics": "hidden_states[k]=output of transformer block k; "
                               "hidden_states[28]=final RMSNorm output; index 0=input embeddings",
            "final_norm_applied": {str(k): (k == n) for k in LAYERS},
            "pooling": "last_valid_prompt_token", "chat_template_sha256": template_sha,
            "dtype": str(next(model.parameters()).dtype),
            "max_length": args.max_length, "batch_size": args.batch_size,
            "labels_sha256": base.sha256(labels_path),
            "sample_count": len(rows), "truncated": 0, "skipped": 0,
            "max_input_tokens": max(map(len, sequences)),
            "mean_input_tokens": sum(map(len, sequences))/len(sequences),
            "forward_ms_including_first_batch": sum(batch_times),
            "post_first_batch_ms_per_question": (
                sum(batch_times[1:])/max(1, len(rows)-min(args.batch_size,len(rows)))),
            "wall_seconds": time.perf_counter()-start,
            "p1_final_layer_reproduction": p1_check,
        }
        arrays = {**matrices, "sample_id": np.asarray([r["sample_id"] for r in rows]),
                  "metadata_json": np.asarray(json.dumps(metadata))}
        path = folder/f"{split}.npz"
        atomic_npz(path, arrays)
        print(f"LAYER_FEATURES_SAVED split={split} path={path} "
              f"rows={len(rows)} layers={LAYERS}", flush=True)


def read_layers(path, rows, label_sha, smoke):
    import numpy as np
    with np.load(path, allow_pickle=False) as fp:
        meta = json.loads(str(fp["metadata_json"].item()))
        ids = fp["sample_id"].astype(str).tolist()
        values = {k: fp[f"X_layer_{k:02d}"].astype(np.float32) for k in LAYERS}
    if meta["schema"] != "route_e_layer_head_v1" or meta["prompt_mode"] != "assistant":
        raise ValueError(f"Wrong Route E layer cache schema/prefix: {path}")
    if meta["layer_indices"] != list(LAYERS) or meta["labels_sha256"] != label_sha:
        raise ValueError(f"Wrong layer indices or original strict label checksum: {path}")
    if bool(meta["smoke"]) != smoke or ids != [r["sample_id"] for r in rows]:
        raise ValueError(f"Wrong smoke/full split, IDs or order: {path}")
    for k, X in values.items():
        if X.shape != (len(rows), 1536) or not np.isfinite(X).all():
            raise ValueError(f"Invalid layer {k} features: {path}")
    return values, meta


def load_splits(smoke):
    train, val, overlaps = expected_rows(smoke)
    refs = check_p1(train, val, smoke)
    Xtr, mt = read_layers(output_dir(smoke)/"train.npz", train, base.TRAIN_SHA256, smoke)
    Xva, mv = read_layers(output_dir(smoke)/"val.npz", val, base.VAL_SHA256, smoke)
    for key in ("model", "chat_template_sha256", "pooling", "layer_indices", "dtype",
                "max_length", "batch_size"):
        if mt[key] != mv[key]:
            raise ValueError(f"Train/validation layer caches mismatch: {key}")
    for key in ("model", "chat_template_sha256", "max_length", "batch_size", "dtype"):
        if mt[key] != refs["train"][1][key]:
            raise ValueError(f"Layer cache differs from original P1 reference in {key}")
    for split, X, ref in (("train", Xtr, refs["train"][0]),
                          ("val", Xva, refs["val"][0])):
        compare_final_reference(X[28], ref, split)
    return train, val, Xtr, Xva, {"train": mt, "val": mv, "text_overlap": overlaps}


def targets(rows):
    import numpy as np
    return np.asarray([int(r["p_n"] >= r["p_t"]) for r in rows], dtype=np.float32)


def report_scores(scores, train_scores, ytr, val, yva, method, layer,
                  smoke, training_info, metas):
    import numpy as np
    from train_modernbert_delta import curve_point
    p = {r["sample_id"]: float(v) for r, v in zip(val, scores)}
    curve = [curve_point(val, {sid: -s for sid,s in p.items()}, f)
             for f in base.FRACTIONS]
    report = {
        "method": method, "layer": layer, "smoke": smoke,
        "encoder": "DeepSeek-R1-Distill-Qwen-1.5B", "encoder_frozen": True,
        "prompt_mode": "assistant", "target": "1[p_nothinking_strict>=p_thinking]",
        "train_sha256": base.TRAIN_SHA256, "val_sha256": base.VAL_SHA256,
        "train_count": len(ytr), "val_count": len(val),
        "train_auc": base.auc(train_scores, ytr), "val_auc": base.auc(scores, yva),
        "train_accuracy_at_0_5": float(((train_scores>=.5) == ytr).mean()),
        "val_accuracy_at_0_5": float(((scores>=.5) == yva).mean()),
        "training": training_info, "feature_meta": metas,
        "curve": curve,
        "mean_generation_tokens_excludes_router_overhead": True,
        "validation_only_exploratory": True,
    }
    return report, p


def save_predictions(folder, report_path, preds, report, checkpoint=None):
    folder.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (folder/"validation_p_nt.json", report_path):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite existing result {p}")
    (folder/"validation_p_nt.json").write_text(json.dumps({"p_nt": preds}, indent=2)+"\n")
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    if checkpoint is not None:
        import numpy as np
        np.savez_compressed(folder/"head.npz", **checkpoint)


def fit_logistic(args):
    import numpy as np
    import torch
    train, val, Xtr, Xva, metas = load_splits(args.smoke)
    yt, yv = targets(train), targets(val)
    summary = {}
    for layer in LAYERS:
        torch.manual_seed(args.seed)
        started = time.perf_counter()
        fit = base.train_logistic(Xtr[layer], yt, args.l2, args.max_iter)
        train_seconds = time.perf_counter()-started
        ptr = base.predict_linear(Xtr[layer], fit)
        started = time.perf_counter()
        pva = base.predict_linear(Xva[layer], fit)
        latency = 1000*(time.perf_counter()-started)/len(val)
        info = {"head": "logistic", "l2": args.l2, "max_iter": args.max_iter,
                "seed": args.seed, "seconds_cpu": train_seconds,
                "pred_ms_per_question_cpu_batch": latency,
                "train_objective": fit["train_objective"]}
        report, preds = report_scores(pva, ptr, yt, val, yv, "logistic", layer,
                                      args.smoke, info, metas)
        if layer == 28:
            # This is the same final normalized hidden state and linear head
            # as the historical frozen P1 run. Fail if its top-k policy changes.
            reference_path = prefix.cache_dir("assistant", args.smoke)/"validation_p_nt.json"
            if reference_path.is_file():
                reference = budget_eval.load_scores(reference_path,
                    [r["sample_id"] for r in val])
                ref_curve = budget_eval.dense_curve(val, reference)
                new_curve = budget_eval.dense_curve(val, preds)
                half = len(val)//2
                ref, new = ref_curve[half], new_curve[half]
                max_pred_diff = max(abs(preds[sid]-reference[sid]) for sid in preds)
                print(f"P1_LOGISTIC_REPRO nt50_old={ref['accuracy']:.5f}/"
                      f"{ref['tokens']:.2f} nt50_new={new['accuracy']:.5f}/"
                      f"{new['tokens']:.2f} max_pred_diff={max_pred_diff:.6g}",flush=True)
                if (abs(ref["accuracy"]-new["accuracy"]) > 1e-8 or
                    abs(ref["tokens"]-new["tokens"]) > 1e-3):
                    raise RuntimeError("Layer 28 logistic fails historical P1 50%-NT reproduction")
            else:
                print(f"P1_SCORE_REFERENCE_MISSING {reference_path}; "
                      "feature-level reproduction passed but score-level check skipped")
        folder = output_dir(args.smoke)/"heads"/f"layer_{layer:02d}_logistic"
        target = results_dir()/f"logistic_L{layer:02d}_{'smoke' if args.smoke else 'val'}.json"
        save_predictions(folder, target, preds, report, checkpoint={
            k: v for k, v in fit.items() if k in ("weight", "bias", "mean", "scale")})
        top = report["curve"][3]["groups"]["overall"]
        summary[str(layer)] = {"auc": report["val_auc"], "nt50_accuracy": top["accuracy"],
                               "nt50_tokens": top["mean_tokens"]}
        print(f"LAYER_LOGISTIC layer={layer} auc={report['val_auc']:.4f} "
              f"nt50_accuracy={top['accuracy']:.4%} nt50_tokens={top['mean_tokens']:.1f}",
              flush=True)
    print("LAYER_LOGISTIC_DONE " + json.dumps(summary), flush=True)


def stratified_train_split(y, seed, frac=.20):
    """Inner train-only holdout for MLP early stopping; never use 450 val labels."""
    import numpy as np
    rng = np.random.default_rng(seed)
    tr, hold = [], []
    for cls in (0,1):
        ids = np.flatnonzero(y == cls)
        if len(ids) < 2:
            raise ValueError("Not enough data for stratified MLP split")
        rng.shuffle(ids)
        n_hold = min(len(ids)-1, max(1, round(len(ids)*frac)))
        hold.extend(ids[:n_hold].tolist())
        tr.extend(ids[n_hold:].tolist())
    return np.asarray(sorted(tr), dtype=int), np.asarray(sorted(hold), dtype=int)


def train_mlp(X, y, seed, hidden_dim, dropout, epochs, patience, lr,
              weight_decay, batch_size):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    tr, hold = stratified_train_split(y, seed)
    mean = X[tr].mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(X[tr].std(axis=0, dtype=np.float64).astype(np.float32), 1e-4)
    Z = torch.from_numpy(((X-mean)/scale).astype(np.float32))
    target = torch.as_tensor(y, dtype=torch.float32)
    def network():
        return torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden_dim),
                                   torch.nn.GELU(), torch.nn.Dropout(dropout),
                                   torch.nn.Linear(hidden_dim, 1))
    net = network()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    generator = torch.Generator().manual_seed(seed)
    for ep in range(1, epochs+1):
        net.train()
        indices = tr[torch.randperm(len(tr),generator=generator).numpy()]
        for part in range(0,len(indices),batch_size):
            idx = indices[part:part+batch_size]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(Z[idx]).squeeze(-1),target[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            loss_val = float(F.binary_cross_entropy_with_logits(
                net(Z[hold]).squeeze(-1), target[hold]))
        if loss_val < best_loss-1e-5:
            best_loss, best_epoch, stale = loss_val, ep, 0
        else:
            stale += 1
            if stale >= patience:
                break
    # Fix best epoch on the train-only inner split, then refit using all 3000.
    mean_full = X.mean(axis=0,dtype=np.float64).astype(np.float32)
    scale_full = np.maximum(X.std(axis=0,dtype=np.float64).astype(np.float32),1e-4)
    Zfull = torch.from_numpy(((X-mean_full)/scale_full).astype(np.float32))
    torch.manual_seed(seed)
    final = network()
    opt = torch.optim.AdamW(final.parameters(),lr=lr,weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    for ep in range(best_epoch):
        final.train()
        indices = torch.randperm(len(y),generator=generator)
        for part in range(0,len(y),batch_size):
            ids = indices[part:part+batch_size]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(final(Zfull[ids]).squeeze(-1),
                                                      target[ids])
            loss.backward()
            opt.step()
    final.eval()
    return final, mean_full, scale_full, {
        "selection": "stratified 80/20 train-only inner holdout; no 450-val early stopping",
        "selection_best_epoch": best_epoch, "selection_val_bce": best_loss,
        "selection_holdout_n": int(len(hold)), "full_train_refit_epochs": best_epoch,
        "hidden_dim": hidden_dim, "dropout": dropout, "lr": lr,
        "weight_decay": weight_decay, "batch_size": batch_size,
        "max_epochs": epochs, "patience": patience, "seed": seed,
    }


def predict_mlp(model, X, mean, scale):
    import numpy as np
    import torch
    model.eval()
    with torch.no_grad():
        z = torch.from_numpy(((X-mean)/scale).astype(np.float32))
        return torch.sigmoid(model(z).squeeze(-1)).cpu().numpy().astype(np.float64)


def fit_mlp(args):
    import numpy as np
    import torch
    train, val, Xtr, Xva, metas = load_splits(args.smoke)
    yt, yv = targets(train), targets(val)
    started = time.perf_counter()
    net, mean, scale, info = train_mlp(Xtr[args.layer], yt, args.seed,
        args.hidden_dim, args.dropout, args.epochs, args.patience, args.lr,
        args.weight_decay, args.batch_size)
    info["training_seconds_cpu"] = time.perf_counter()-started
    ptr = predict_mlp(net, Xtr[args.layer], mean, scale)
    start = time.perf_counter()
    pva = predict_mlp(net, Xva[args.layer], mean, scale)
    info["pred_ms_per_question_cpu_batch"] = 1000*(time.perf_counter()-start)/len(val)
    info["head"] = "Linear-GELU-Dropout-Linear"
    report, preds = report_scores(pva, ptr, yt, val, yv, "mlp", args.layer,
                                  args.smoke, info, metas)
    folder = output_dir(args.smoke)/"heads"/f"layer_{args.layer:02d}_mlp"
    target = results_dir()/f"mlp_L{args.layer:02d}_{'smoke' if args.smoke else 'val'}.json"
    save_predictions(folder, target, preds, report)
    torch.save({"state_dict": net.state_dict(), "mean": mean, "scale": scale,
                "configuration": info}, folder/"head.pt")
    m = report["curve"][3]["groups"]["overall"]
    print(f"LAYER_MLP_DONE layer={args.layer} auc={report['val_auc']:.4f} "
          f"nt50_accuracy={m['accuracy']:.4%} nt50_tokens={m['mean_tokens']:.1f}",
          flush=True)


def group_nt50(rows, scores, dataset):
    from train_modernbert_delta import curve_point
    point = curve_point(rows, {k: -v for k,v in scores.items()}, 0.5)
    return point["groups"].get(dataset)


def bootstrap_cost_matched(rows, scores, budgets, n_boot, seed):
    """Stratified paired resampling; rank+cost bracket recomputed per draw."""
    import numpy as np
    if n_boot <= 0:
        return None
    names = sorted(scores)
    p_t = np.asarray([r["p_t"] for r in rows],dtype=np.float64)
    p_n = np.asarray([r["p_n"] for r in rows],dtype=np.float64)
    t_t = np.asarray([r["t_t"] for r in rows],dtype=np.float64)
    t_n = np.asarray([r["t_n"] for r in rows],dtype=np.float64)
    arrays = {name: np.asarray([scores[name][r["sample_id"]] for r in rows])
              for name in names}
    groups = [np.asarray([i for i,r in enumerate(rows) if r["dataset"]==name])
              for name in ("gsm8k","math")]
    rng = np.random.default_rng(seed)
    observations = {name: {float(b): [] for b in budgets} for name in names}
    observations["random"] = {float(b): [] for b in budgets}
    for iteration in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        a, b, c, d = p_t[idx],p_n[idx],t_t[idx],t_n[idx]
        n=len(idx)
        lo,hi=d.mean(),c.mean()
        for token_budget in budgets:
            if not(lo<=token_budget<=hi):
                continue
            q=(hi-token_budget)/(hi-lo)
            observations["random"][float(token_budget)].append(float((1-q)*a.mean()+q*b.mean()))
        for name in names:
            ranking = np.argsort(-arrays[name][idx],kind="stable")
            accuracies = a.mean() + np.r_[0,np.cumsum((b-a)[ranking])]/n
            costs = c.mean() + np.r_[0,np.cumsum((d-c)[ranking])]/n
            perm = np.argsort(costs,kind="stable")
            sorted_cost, sorted_acc=costs[perm],accuracies[perm]
            for token_budget in budgets:
                if not(sorted_cost[0] <= token_budget <= sorted_cost[-1]):
                    continue
                k=np.searchsorted(sorted_cost,token_budget,side="left")
                if k==0:
                    value=sorted_acc[0]
                elif k==len(sorted_cost):
                    value=sorted_acc[-1]
                elif sorted_cost[k] == token_budget:
                    value=sorted_acc[k]
                else:
                    weight=(token_budget-sorted_cost[k-1])/(sorted_cost[k]-sorted_cost[k-1])
                    value=(1-weight)*sorted_acc[k-1]+weight*sorted_acc[k]
                observations[name][float(token_budget)].append(float(value))
    # To preserve pairing, intervals for differences use only resamples where both
    # methods have a valid bracket. Above validation range should contain budgets.
    result={}
    for name in names:
        result[name]={}
        for token_budget in budgets:
            candidate=observations[name][float(token_budget)]
            if len(candidate)!=n_boot:
                result[name][str(token_budget)]={"warning":"Some bootstrap budgets out of range",
                                                  "valid":len(candidate)}
                continue
            result[name][str(token_budget)]={}
            for baseline in ("P1", "FT", "random"):
                if baseline==name or baseline not in observations:
                    continue
                control=observations[baseline][float(token_budget)]
                if len(control)!=n_boot:
                    result[name][str(token_budget)][baseline]={"warning":"incomplete bootstrap pairing"}
                    continue
                diff=100*(np.asarray(candidate)-np.asarray(control))
                result[name][str(token_budget)][baseline]={
                    "mean_diff_pp":float(diff.mean()),
                    "ci95_pp":[float(v) for v in np.percentile(diff,[2.5,97.5])],
                    "paired_resamples":n_boot}
    return result


def compare(args):
    import numpy as np
    train, val, _ = expected_rows(False)
    check_p1(train, val, False)
    ids = [r["sample_id"] for r in val]
    scores = {"P1": budget_eval.load_scores(
        prefix.cache_dir("assistant",False)/"validation_p_nt.json",ids)}
    evidence={"P1":str(prefix.cache_dir("assistant",False)/"validation_p_nt.json")}
    for layer in LAYERS:
        name=f"L{layer:02d}_logistic"
        path=output_dir(False)/"heads"/f"layer_{layer:02d}_logistic"/"validation_p_nt.json"
        scores[name]=budget_eval.load_scores(path,ids)
        evidence[name]=str(path)
    for layer in LAYERS:
        path=output_dir(False)/"heads"/f"layer_{layer:02d}_mlp"/"validation_p_nt.json"
        if path.exists():
            name=f"L{layer:02d}_mlp"
            scores[name]=budget_eval.load_scores(path,ids)
            evidence[name]=str(path)
    if args.ft_scores is not None:
        scores["FT"]=budget_eval.load_scores(args.ft_scores,ids,
            direction=args.ft_direction,field=args.ft_field)
        evidence["FT"]=str(args.ft_scores)
        budget_eval.verify_classification_50pct(budget_eval.dense_curve(val,scores["FT"]))
    else:
        print("FT_SCORES_MISSING: omitting FT rather than using approximate anchors")
    curves={name:budget_eval.dense_curve(val,vec) for name,vec in scores.items()}
    table=[]
    for b in args.budgets:
        random=budget_eval.random_at_budget(val,b)
        record={"budget":b,"random_accuracy":random["accuracy"],"methods":{}}
        for name,curve in curves.items():
            estimate=budget_eval.linear_bracket(curve,b)
            estimate["uplift_vs_random_pp"]=100*(estimate["accuracy"]-random["accuracy"])
            estimate["delta_vs_P1_pp"]=100*(estimate["accuracy"]-
                budget_eval.linear_bracket(curves["P1"],b)["accuracy"])
            if "FT" in curves:
                estimate["delta_vs_FT_pp"]=100*(estimate["accuracy"]-
                    budget_eval.linear_bracket(curves["FT"],b)["accuracy"])
            record["methods"][name]=estimate
        table.append(record)
    bootstrap=bootstrap_cost_matched(val,scores,args.budgets,args.bootstrap,args.seed)
    diagnostics={name:{ds:group_nt50(val,vec,ds) for ds in ("overall","gsm8k","math")}
                 for name,vec in scores.items()}
    output={
        "title":"Frozen DeepSeek P1 hidden-layer / head ablation",
        "validation_sha256":base.VAL_SHA256, "train_sha256":base.TRAIN_SHA256,
        "validation_count":len(val),"metric":"expected strict accuracy at mean generated-token budget",
        "evaluation":"all 451 top-k NoThinking policies; adjacent measured-cost interpolation",
        "frozen_model":True,"prompt_mode":"assistant",
        "selection_warning":"Exploratory validation reuse for layer/head selection; "
                            "bootstrap CIs are descriptive, not post-selection confirmatory evidence",
        "router_overhead_included":False,
        "source_scores":evidence,"table":table,"diagnostics_50pct_nt":diagnostics,
        "bootstrap":{"resamples":args.bootstrap,"seed":args.seed,
                     "sampling":"stratified paired bootstrap within GSM8K/MATH", "differences":bootstrap},
    }
    dest=results_dir()
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/"equal_budget.json").exists() and not args.overwrite_report:
        raise FileExistsError("Existing comparison report. Pass --overwrite-report for intentional rerun")
    (dest/"equal_budget.json").write_text(json.dumps(output,indent=2)+"\n")
    names=sorted(scores)
    with (dest/"equal_budget.csv").open("w",newline="") as f:
        wr=csv.writer(f)
        wr.writerow(["budget","Random_accuracy_pct"]+[f"{n}_accuracy_pct" for n in names])
        for row in table:
            wr.writerow([row["budget"],100*row["random_accuracy"]]+[
                100*row["methods"][n]["accuracy"] for n in names])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9,5.5))
        for name, pts in curves.items():
            x=[p["tokens"] for p in pts]
            y=[100*p["accuracy"] for p in pts]
            ax.plot(x,y,lw=1.4 if name=="P1" else .9,
                    alpha=1.0 if name=="P1" else .65,label=name)
        ax.plot([p["budget"] for p in table],
                [100*p["random_accuracy"] for p in table],
                linestyle="--",label="Global Random")
        ax.set(xlabel="Mean generated tokens / question (router overhead excluded)",
               ylabel="Strict validation accuracy (%)",
               title="P1 frozen DeepSeek: hidden layer and head ablation (450 val)")
        ax.legend(fontsize=8,ncol=2)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(dest/"accuracy_token_curve.svg")
        fig.savefig(dest/"accuracy_token_curve.png",dpi=180)
        plt.close(fig)
        print("LAYER_HEAD_PLOT_SAVED accuracy_token_curve.svg/png")
    except ImportError:
        print("PLOT_SKIPPED: matplotlib unavailable; tabular results saved")
    print("BUDGET | Random | "+" | ".join(names))
    for row in table:
        print(f"{row['budget']:.0f} | {100*row['random_accuracy']:.2f}% | "+" | ".join(
            f"{100*row['methods'][n]['accuracy']:.2f}%" for n in names))
    print(f"LAYER_HEAD_COMPARE_DONE {dest/'equal_budget.json'}",flush=True)


def self_test():
    import numpy as np
    import torch
    import route_e_equal_budget as eb
    x=np.asarray([[-2.,0.],[-1.,.1],[1.,0.],[2.,.1]]*4,dtype=np.float32)
    y=np.asarray([0,0,1,1]*4,dtype=np.float32)
    t, h = stratified_train_split(y,42)
    assert len(t)+len(h)==len(y) and not (set(t)&set(h))
    assert set(y[t])==set(y[h])=={0,1}
    net, mu, sigma, info=train_mlp(x,y,42,8,.1,5,2,1e-3,1e-3,4)
    pp=predict_mlp(net,x,mu,sigma)
    assert pp.shape==(len(y),) and np.isfinite(pp).all()
    rows=[{"sample_id":"a","dataset":"gsm8k","p_t":1.,"p_n":0.,"t_t":100.,"t_n":20.},
          {"sample_id":"b","dataset":"math","p_t":0.,"p_n":1.,"t_t":100.,"t_n":20.}]
    c=eb.dense_curve(rows,{"a":.1,"b":.9})
    assert abs(eb.linear_bracket(c,80)["accuracy"]-.75)<1e-9
    data=bootstrap_cost_matched(rows, {"P1":{"a":.1,"b":.9},
                                           "L7_logistic":{"a":.1,"b":.9}},
                                [80],10,42)
    assert "L7_logistic" in data
    fake=np.asarray([[.1,.2],[.3,.4]],dtype=np.float32)
    assert compare_final_reference(fake,fake,"self-test")["mean_abs_diff"]==0
    print("LAYER_HEAD_SELFTEST_PASS")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest="command",required=True)
    commands.add_parser("self-test")
    cp=commands.add_parser("check-cache")
    cp.add_argument("--smoke",action="store_true")
    ex=commands.add_parser("extract")
    ex.add_argument("--smoke",action="store_true")
    ex.add_argument("--model",default=prefix.MODEL)
    ex.add_argument("--batch-size",type=int,default=8)
    ex.add_argument("--max-length",type=int,default=1024)
    ex.add_argument("--seed",type=int,default=42)
    lg=commands.add_parser("fit-logistic")
    lg.add_argument("--smoke",action="store_true")
    lg.add_argument("--l2",type=float,default=1e-3)
    lg.add_argument("--max-iter",type=int,default=200)
    lg.add_argument("--seed",type=int,default=42)
    ml=commands.add_parser("fit-mlp")
    ml.add_argument("--smoke",action="store_true")
    ml.add_argument("--layer",type=int,required=True,choices=LAYERS)
    ml.add_argument("--seed",type=int,default=42)
    ml.add_argument("--hidden-dim",type=int,default=128)
    ml.add_argument("--dropout",type=float,default=.20)
    ml.add_argument("--epochs",type=int,default=80)
    ml.add_argument("--patience",type=int,default=8)
    ml.add_argument("--lr",type=float,default=1e-3)
    ml.add_argument("--weight-decay",type=float,default=1e-3)
    ml.add_argument("--batch-size",type=int,default=128)
    co=commands.add_parser("compare")
    co.add_argument("--ft-scores",type=Path)
    co.add_argument("--ft-field",type=str)
    co.add_argument("--ft-direction",choices=("higher_nt","lower_nt"),default="higher_nt")
    co.add_argument("--budgets",type=float,nargs="+",default=BUDGETS)
    co.add_argument("--bootstrap",type=int,default=500)
    co.add_argument("--seed",type=int,default=42)
    co.add_argument("--overwrite-report",action="store_true")
    args=p.parse_args()
    if args.command=="self-test":return self_test()
    if args.command=="check-cache":return check_cache(args)
    if args.command=="extract":
        if args.batch_size<1 or args.max_length<1:p.error("positive batch-size/max-length required")
        return extract(args)
    if args.command=="fit-logistic":
        if not math.isfinite(args.l2) or args.l2<0 or args.max_iter<1:
            p.error("invalid logistic hyperparameters")
        return fit_logistic(args)
    if args.command=="fit-mlp":
        if (args.hidden_dim<1 or not 0<=args.dropout<1 or args.epochs<1 or
            args.patience<1 or args.batch_size<1 or args.lr<=0 or args.weight_decay<0):
            p.error("invalid MLP hyperparameters")
        return fit_mlp(args)
    if args.ft_scores is not None and args.ft_scores.suffix==".jsonl" and not args.ft_field:
        p.error("--ft-field required for FT JSONL")
    if args.bootstrap<0 or not args.budgets or len(set(args.budgets))!=len(args.budgets):
        p.error("invalid bootstrap/budgets")
    return compare(args)


if __name__=="__main__":
    main()
