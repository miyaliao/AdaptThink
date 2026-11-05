#!/usr/bin/env python3
"""Controlled readout experiment: original LM Head vs trained linear head.

Both heads read the SAME final normalized hidden state of the SAME
self-assessment ORIGINAL-option prompt. The LM head is frozen/pretrained;
the logistic probe uses only the original 3,000 train labels.

CPU:
  python scripts/route_self_assessment_head_control.py self-test
GPU (Slurm; smoke first):
  python scripts/route_self_assessment_head_control.py extract --split train --smoke
CPU smoke (TRAIN only, in-sample checks, not a validation result):
  python scripts/route_self_assessment_head_control.py fit --smoke
GPU full (separate Slurm jobs):
  python scripts/route_self_assessment_head_control.py extract --split train
  python scripts/route_self_assessment_head_control.py extract --split val
CPU:
  python scripts/route_self_assessment_head_control.py fit

Do not tune prompts, probe hyperparameters, or option ordering on 450-val.
No answer generation and no LRM fine-tuning.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import route_e_frozen as base
import route_e_equal_budget as budgets
import route_self_assessment as sa

ROOT = base.ROOT
SCHEMA = "self_assessment_same_state_heads_v1"
PROMPT_VARIANT = "original"
BUDGETS = budgets.DEFAULT_BUDGETS


def cache_path(split, smoke=False):
    return (ROOT / "outputs/route_e/self_assessment/head_control" /
            ("smoke" if smoke else "full") / (split + ".npz"))


def result_dir(smoke=False):
    return ROOT / "results/route_e/self_assessment/head_control" / (
        "smoke" if smoke else "full")


def self_test():
    import numpy as np
    X = np.asarray([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]], dtype=np.float32)
    y = np.asarray([0., 0., 1., 1.], dtype=np.float32)
    fitted = base.train_logistic(X, y, 1e-3, 200)
    p = base.predict_linear(X, fitted)
    assert p.shape == (4,) and np.isfinite(p).all()
    assert p[0] < p[1] < p[2] < p[3]
    assert p[1] < .5 < p[2]
    assert abs(base.auc(p, y) - 1.) < 1e-12
    assert sa.score_margins(2., 3., "original") == 1.
    print("HEAD_CONTROL_SELFTEST_PASS")


def source_self_val_scores(rows):
    """Read the already executed original LM Head run, never recompute labels."""
    path = sa.output_dir(False) / "val_predictions.json"
    obj = json.loads(path.read_text(encoding="utf-8"))
    meta = obj["metadata"]
    if (meta["schema"] != sa.SCORE_SCHEMA or meta["smoke"] or
            meta["split"] != "val" or
            meta["labels_sha256"] != base.VAL_SHA256 or
            meta["prompt_sha256"] != sa.prompt_hash()):
        raise ValueError("Original self-assessment validation source mismatch")
    preds = obj["predictions"]
    if [r["sample_id"] for r in preds] != [r["sample_id"] for r in rows]:
        raise ValueError("Original self-assessment validation IDs mismatch")
    return {r["sample_id"]: float(r["original"]["nt_minus_t_logit"])
            for r in preds}, meta, str(path)


def extract(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required; use Slurm, not the login node")
    if args.smoke and args.split != "train":
        raise ValueError("Smoke is TRAIN-only; never choose val for smoke")
    train, val, _ = base.load_protocol()
    if args.smoke:
        train = base.choose_smoke_rows(train, val)[0]
    rows = train if args.split == "train" else val
    labels_file = base.TRAIN if args.split == "train" else base.VAL
    destination = cache_path(args.split, args.smoke)
    if destination.exists():
        raise FileExistsError("Refusing to overwrite feature cache: " + str(destination))
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    tokenizer = sa.load_local_tokenizer(args.model)
    audit = sa.audit_all(tokenizer, args.max_length)
    ids_ab = audit["candidate_token_ids"]
    sequences = [sa.encode(tokenizer, sa.render_prompt(
        tokenizer, row["problem"], PROMPT_VARIANT)) for row in rows]
    if not sequences or any(not s or len(s) > args.max_length for s in sequences):
        raise ValueError("Empty/overlength sequences")

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16, local_files_only=True,
        trust_remote_code=False).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    features, margins = [], []
    seconds, worst_reconstruction_error = 0., 0.
    start_time = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            chunk = sequences[start:start + args.batch_size]
            width = max(map(len, chunk))
            tokens = torch.full((len(chunk), width), tokenizer.pad_token_id,
                                dtype=torch.long, device="cuda")
            mask = torch.zeros_like(tokens)
            for i, seq in enumerate(chunk):
                tokens[i, :len(seq)] = torch.tensor(seq, device="cuda")
                mask[i, :len(seq)] = 1
            positions = torch.tensor([len(seq)-1 for seq in chunk],
                                     dtype=torch.long, device="cuda")
            index = torch.arange(len(chunk), device="cuda")
            torch.cuda.synchronize()
            begin = time.perf_counter()
            out = model(input_ids=tokens, attention_mask=mask,
                        use_cache=False, output_hidden_states=True,
                        return_dict=True)
            # Qwen final layer after RMSNorm: the exact vector sent to lm_head.
            hidden = out.hidden_states[-1][index, positions]
            lm_logits = out.logits[index, positions]
            rebuilt = model.get_output_embeddings()(hidden)
            ab = [ids_ab["A"], ids_ab["B"]]
            diff = float((rebuilt[:, ab].float()-lm_logits[:, ab].float())
                         .abs().max().item())
            worst_reconstruction_error = max(worst_reconstruction_error, diff)
            if diff > 0.03:
                raise RuntimeError(
                    f"Same-hidden-state LM-head reconstruction failed: {diff:.5f}")
            score = (lm_logits[:, ids_ab["B"]] -
                     lm_logits[:, ids_ab["A"]]).float()
            if not torch.isfinite(hidden).all() or not torch.isfinite(score).all():
                raise RuntimeError(f"Non-finite features/logits in batch {start}")
            features.append(hidden.float().cpu().numpy().astype(np.float16))
            margins.extend(score.cpu().numpy().astype(np.float32).tolist())
            torch.cuda.synchronize()
            seconds += time.perf_counter()-begin
            if start == 0 or start+len(chunk) == len(rows) or (
                    start//args.batch_size) % 20 == 0:
                print(f"HEAD_EXTRACT split={args.split} smoke={int(args.smoke)} "
                      f"n={start+len(chunk)}/{len(rows)}", flush=True)

    X = np.concatenate(features)
    expected_dim = int(model.config.hidden_size)
    if X.shape != (len(rows), expected_dim) or not np.isfinite(X).all():
        raise RuntimeError(f"Wrong extracted features {X.shape}")
    if not np.isfinite(margins).all():
        raise RuntimeError("Non-finite original LM logits")
    original_delta = None
    source_path = None
    if args.split == "val":
        prev, prior_meta, source_path = source_self_val_scores(rows)
        if str(model_path) != prior_meta["model_path"]:
            raise ValueError("Model path differs from original self-assessment")
        if audit["chat_template_sha256"] != prior_meta["chat_template_sha256"]:
            raise ValueError("Chat template hash mismatch with original scores")
        original_delta = float(max(abs(m-prev[r["sample_id"]])
                                   for m, r in zip(margins, rows)))
        print(f"HEAD_ORIGINAL_LOGIT_REPLAY max_abs_delta={original_delta:.6f}",
              flush=True)
        # Inference batching/dtype can introduce small numeric differences.
        if original_delta > 0.10:
            raise RuntimeError("Original LM score replay mismatch >0.10; inspect "
                               "checkpoint/batching/forward before saving")
    metadata = {
        "schema": SCHEMA, "split": args.split, "smoke": args.smoke,
        "labels_sha256": base.sha256(labels_file), "sample_count": len(rows),
        "model_path": str(model_path),
        "model_revision": getattr(model.config, "_commit_hash", None),
        "dtype": str(next(model.parameters()).dtype),
        "prompt_variant": PROMPT_VARIANT, "prompt_sha256": sa.prompt_hash(),
        "chat_template_sha256": audit["chat_template_sha256"],
        "candidate_token_ids": ids_ab,
        "pooling": "final_normalized_hidden_state_at_last_prompt_token",
        "feature_dim": int(X.shape[1]), "max_length": args.max_length,
        "batch_size": args.batch_size, "seed": args.seed,
        "max_input_tokens": max(map(len, sequences)),
        "mean_input_tokens": sum(map(len, sequences))/len(sequences),
        "lm_head_reconstruction_max_abs_logit_diff": worst_reconstruction_error,
        "original_self_assessment_replay_max_abs_margin_diff": original_delta,
        "original_self_assessment_source": source_path,
        "forward_seconds": seconds,
        "forward_ms_per_question": 1000*seconds/len(rows),
        "wall_seconds": time.perf_counter()-start_time,
        "trainable_lrm_parameters": 0, "answers_generated": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, X=X,
        sample_id=np.asarray([r["sample_id"] for r in rows]),
        original_lm_margin=np.asarray(margins, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)))
    print(f"HEAD_EXTRACT_DONE split={args.split} smoke={int(args.smoke)} "
          f"shape={X.shape} max_reconstruction_error="
          f"{worst_reconstruction_error:.6f} path={destination}", flush=True)


def read_cache(split, smoke, rows, expected_hash):
    import numpy as np
    path = cache_path(split, smoke)
    with np.load(path, allow_pickle=False) as obj:
        X = obj["X"].astype(np.float32)
        margins = obj["original_lm_margin"].astype(np.float64)
        ids = obj["sample_id"].astype(str).tolist()
        meta = json.loads(str(obj["metadata_json"].item()))
    if (meta["schema"] != SCHEMA or meta["split"] != split or
            bool(meta["smoke"]) != smoke or
            meta["labels_sha256"] != expected_hash or
            meta["prompt_sha256"] != sa.prompt_hash() or
            meta["prompt_variant"] != PROMPT_VARIANT or
            meta["pooling"] != "final_normalized_hidden_state_at_last_prompt_token"):
        raise ValueError(f"Feature metadata mismatch: {path}")
    if ids != [r["sample_id"] for r in rows]:
        raise ValueError(f"Sample ID/order mismatch: {path}")
    if (X.shape != (len(rows), 1536) or
            margins.shape != (len(rows),) or
            not np.isfinite(X).all() or not np.isfinite(margins).all()):
        raise ValueError(f"Feature/margin shape or finite check failed: {path}")
    return X, margins, meta


def fit(args):
    import numpy as np

    train, val, overlaps = base.load_protocol()
    if args.smoke:
        train = base.choose_smoke_rows(train, val)[0]
    Xtr, tr_margin, trmeta = read_cache(
        "train", args.smoke, train, base.TRAIN_SHA256)
    if args.smoke:
        # Training-only smoke: explicitly in-sample, never a val measurement.
        val = train
        Xva, val_margin, vameta = Xtr, tr_margin, trmeta
    else:
        Xva, val_margin, vameta = read_cache(
            "val", False, val, base.VAL_SHA256)
    for name in ("model_path", "model_revision", "dtype", "prompt_variant",
                 "prompt_sha256", "chat_template_sha256", "candidate_token_ids",
                 "pooling", "feature_dim", "max_length"):
        if trmeta[name] != vameta[name]:
            raise ValueError(f"Train/val extraction differs on {name}")
    target = result_dir(args.smoke)
    if (target/"report.json").exists():
        raise FileExistsError(f"Refusing to overwrite old evaluation: {target}")
    ytr = np.asarray([int(r["p_n"] >= r["p_t"]) for r in train],
                     dtype=np.float32)
    if len(set(ytr.tolist())) != 2:
        raise ValueError("Single-class training data")
    fitted = base.train_logistic(Xtr, ytr, args.l2, args.max_iter)
    probabilities = base.predict_linear(Xva, fitted)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Non-finite supervised probe scores")

    train_only = bool(args.smoke)
    prediction = {r["sample_id"]: float(p) for r, p in zip(val, probabilities)}
    scores = {
        "Original_LM_Head": {r["sample_id"]: float(v)
                             for r, v in zip(val, val_margin)},
        "Same_State_Linear_Head": {
            sid: math.log(max(min(p,1.-1e-9),1e-9) /
                          (1.-max(min(p,1.-1e-9),1e-9)))
            for sid, p in prediction.items()},
    }
    metrics = {
        name: sa.classification_stats(val, mapping)
        for name, mapping in scores.items()}
    report = {
        "experiment": "same_prompt_same_hidden_state_readout_control",
        "status": "TRAIN_ONLY_IN_SAMPLE_SMOKE" if train_only else "EXPLORATORY_VAL",
        "prompt_mode": PROMPT_VARIANT,
        "target": "1[p_NoThinking>=p_Thinking]; ties => NoThinking",
        "encoder_frozen": True,
        "reference_head": "original_checkpoint_lm_head_A_B_logits",
        "trained_head": "linear_logistic_regression_hard_BCE_plus_L2",
        "l2": args.l2, "max_iter": args.max_iter, "seed": args.seed,
        "train_count": len(train), "evaluation_count": len(val),
        "train_sha256": base.TRAIN_SHA256,
        "evaluation_sha256": base.TRAIN_SHA256 if train_only else base.VAL_SHA256,
        "train_val_normalized_text_overlap": overlaps,
        "classification_metrics": metrics,
        "train_feature_meta": trmeta, "eval_feature_meta": vameta,
        "same_forward_same_hidden_state": True,
        "router_prefill_cost_in_generated_token_curve": False,
        "method_selection_on_validation": False,
    }
    if not train_only:
        original_scores, source_meta, path = source_self_val_scores(val)
        replay_diff = max(abs(scores["Original_LM_Head"][sid]-v)
                          for sid,v in original_scores.items())
        if replay_diff > 0.10:
            raise ValueError("Validation LM-head result not reproducible")
        report["original_self_assessment_source"] = path
        report["original_self_assessment_score_max_abs_delta"] = replay_diff
        p1path = ROOT / (
            "outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json")
        p1 = budgets.load_scores(p1path, [r["sample_id"] for r in val])
        p1curve = budgets.dense_curve(val, p1)
        anchor = p1curve[225]
        if (abs(anchor["accuracy"] - 344/450) > .0003 or
                abs(anchor["tokens"] - 1427.05) > 2):
            raise RuntimeError("P1 50pct reproduction gate failed")
        scores["P1_Logistic_Different_Prompt"] = {
            sid: math.log(max(min(p,1.-1e-9),1e-9) /
                          (1.-max(min(p,1.-1e-9),1e-9)))
            for sid,p in p1.items()}
        report["classification_metrics"]["P1_Logistic_Different_Prompt"] = (
            sa.classification_stats(val, scores["P1_Logistic_Different_Prompt"]))
        all_curves = {key: budgets.dense_curve(val, s)
                      for key,s in scores.items()}
        budget_points = []
        for b in BUDGETS:
            point = {"budget": b,
                     "Global_Random": budgets.random_at_budget(val, b)["accuracy"]}
            for name, c in all_curves.items():
                point[name] = budgets.linear_bracket(c, b)["accuracy"]
            budget_points.append(point)
        report["budgets"] = list(BUDGETS)
        report["budget_points"] = budget_points
        report["cost_warning"] = (
            "450-val offline replay; only generated answer tokens, "
            "not added self-assessment prefill cost")
        report["p1_control_warning"] = (
            "P1 uses a different prompt and hidden state; not part of "
            "the matched-head causal control")
    target.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target/"linear_head.npz",
                        weight=fitted["weight"], bias=np.asarray(fitted["bias"]),
                        mean=fitted["mean"], scale=fitted["scale"],
                        prompt_sha256=np.asarray(sa.prompt_hash()),
                        l2=np.asarray(args.l2))
    (target/"validation_p_nt.json" if not train_only else
     target/"training_smoke_p_nt.json").write_text(
        json.dumps({"p_nt": prediction}, indent=2)+"\n", encoding="utf-8")
    (target/"report.json").write_text(
        json.dumps(report, indent=2)+"\n", encoding="utf-8")
    for name, d in report["classification_metrics"].items():
        print(f"HEAD_EVAL {name} auc={d['auc']:.4f} "
              f"acc={100*d['accuracy']:.2f}% "
              f"bacc={100*d['balanced_accuracy']:.2f}% "
              f"nt_pred_frac={d['predicted_nt_fraction']:.3f}")
    if not train_only:
        csv_path = target/"budget_points.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(budget_points[0]))
            writer.writeheader()
            writer.writerows(budget_points)
        for pt in budget_points:
            print(f"HEAD_BUDGET {int(pt['budget'])} " +
                  " ".join(f"{k}={100*v:.2f}%"
                           for k,v in pt.items() if k != "budget"))
    print(f"HEAD_CONTROL_DONE smoke={int(train_only)} "
          f"report={target/'report.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test")
    ex = sub.add_parser("extract")
    ex.add_argument("--split", choices=("train","val"), required=True)
    ex.add_argument("--smoke", action="store_true")
    ex.add_argument("--model", default=sa.DEFAULT_MODEL)
    ex.add_argument("--batch-size", type=int, default=4)
    ex.add_argument("--max-length", type=int, default=1024)
    ex.add_argument("--seed", type=int, default=42)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--smoke", action="store_true")
    fit_parser.add_argument("--l2", type=float, default=1e-3)
    fit_parser.add_argument("--max-iter", type=int, default=200)
    fit_parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "extract":
        if args.batch_size < 1 or args.max_length < 1:
            parser.error("batch-size/max-length must be >= 1")
        if args.smoke and args.split != "train":
            parser.error("smoke extraction is train-only")
        extract(args)
    else:
        if args.l2 < 0 or not math.isfinite(args.l2) or args.max_iter < 1:
            parser.error("invalid probe hyperparameters")
        fit(args)


if __name__ == "__main__":
    main()
