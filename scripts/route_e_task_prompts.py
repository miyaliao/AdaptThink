#!/usr/bin/env python3
"""Task-aware assistant-prefix frozen DeepSeek representations for Route E.

CPU: self-test, inspect-template, audit. One GPU: extract. CPU: fit.
No generation, no held-out test reads, no modification of existing P1 caches.

Run from the repository root:
  python scripts/route_e_task_prompts.py self-test
  python scripts/route_e_task_prompts.py inspect-template
  python scripts/route_e_task_prompts.py audit
  python scripts/route_e_task_prompts.py extract --mode general --smoke
  python scripts/route_e_task_prompts.py fit --mode general --smoke
  python scripts/route_e_task_prompts.py extract --mode general
  python scripts/route_e_task_prompts.py fit --mode general

Repeat for difficulty and routing only after examining CPU audit outputs.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import route_e_frozen as base
import route_e_prompt_ablation as prefix

ROOT = base.ROOT
MODEL = base.DEFAULT_MODELS["deepseek"]
# Freeze variants BEFORE looking at validation results. General is a semantic
# task control, not a meaningless-string or pure position/length control.
SUFFIXES = {
    "general": 'Problem summary in one word: "',
    "difficulty": 'Problem difficulty in one word: "',
    "routing": 'Reasoning mode in one word: "',
}
MODES = tuple(SUFFIXES)
SCHEMA = "route_e_task_prompt_features_v1"


def cache_dir(mode, smoke):
    return ROOT / "outputs/route_e/task_prompt" / mode / ("smoke" if smoke else "full")


def render_prompt(tokenizer, question, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}")
    p1 = prefix.render_prompts(tokenizer, question)["assistant"]
    return p1 + SUFFIXES[mode]


def encode_question(tokenizer, question, mode):
    text = render_prompt(tokenizer, question, mode)
    # The DeepSeek chat template already emits BOS; never insert it twice.
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    if not ids:
        raise ValueError("Empty encoded prompt")
    return ids


def model_tokenizer(model_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def inspect_template(model_path):
    tokenizer = model_tokenizer(model_path)
    question = "What is 25 * 16?"
    p1 = prefix.render_prompts(tokenizer, question)["assistant"]
    p1_ids = tokenizer(p1, add_special_tokens=False, truncation=False)["input_ids"]
    print(f"P1 len={len(p1_ids)} last={p1_ids[-1]} suffix={p1[-70:]!r}")
    for mode in MODES:
        txt = render_prompt(tokenizer, question, mode)
        ids = encode_question(tokenizer, question, mode)
        print(f"PROMPT mode={mode} len={len(ids)} delta={len(ids)-len(p1_ids)} "
              f"last={ids[-1]} repr={txt[-110:]!r}")
    print("PROMPT_INSPECTION_PASS (token audit across all data still required)")


def audit_rows(tokenizer, split_rows, max_length, max_spread):
    """Full-split audit with no labels inspected for prompt construction.

    N/D/R must share their last token ID for every sample and differ by at
    most max_spread tokens within each sample (P1 intentionally excluded).
    """
    from collections import Counter
    stats = {name: {"lengths": [], "deltas_p1": [], "last_ids": Counter()}
             for name in MODES}
    too_long = []
    last_mismatches = []
    spread_mismatches = []
    for split, rows in split_rows:
        for r in rows:
            text = r["problem"]
            sid = r["sample_id"]
            p1 = prefix.render_prompts(tokenizer, text)["assistant"]
            base_ids = tokenizer(p1, add_special_tokens=False,
                                 truncation=False)["input_ids"]
            lengths, last = {}, {}
            for mode in MODES:
                ids = encode_question(tokenizer, text, mode)
                lengths[mode] = len(ids)
                last[mode] = ids[-1]
                stats[mode]["lengths"].append(len(ids))
                stats[mode]["deltas_p1"].append(len(ids)-len(base_ids))
                stats[mode]["last_ids"][ids[-1]] += 1
                if len(ids) > max_length:
                    too_long.append([split, sid, mode, len(ids)])
            if len(set(last.values())) > 1:
                last_mismatches.append([split, sid, last])
            if max(lengths.values())-min(lengths.values()) > max_spread:
                spread_mismatches.append([split, sid, lengths])
    count = sum(len(rows) for _, rows in split_rows)
    return {
        "schema": "route_e_task_prompt_token_audit_v1",
        "model": str(getattr(tokenizer, "name_or_path", "unknown")),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode("utf-8")).hexdigest(),
        "prompt_suffixes": SUFFIXES,
        "prompt_suffixes_sha256": hashlib.sha256(json.dumps(
            SUFFIXES, sort_keys=True).encode("utf-8")).hexdigest(),
        "sample_count": count,
        "max_length": max_length,
        "allowed_variant_length_spread": max_spread,
        "variants": {
            mode: {
                "mean_input_tokens": sum(s["lengths"])/count,
                "min_input_tokens": min(s["lengths"]),
                "max_input_tokens": max(s["lengths"]),
                "mean_added_tokens_vs_p1": sum(s["deltas_p1"])/count,
                "min_added_tokens_vs_p1": min(s["deltas_p1"]),
                "max_added_tokens_vs_p1": max(s["deltas_p1"]),
                "last_token_id_counts": dict(s["last_ids"]),
            } for mode, s in stats.items()
        },
        "overlength_count": len(too_long),
        "last_token_mismatch_count": len(last_mismatches),
        "length_spread_mismatch_count": len(spread_mismatches),
        "overlength_examples": too_long[:10],
        "last_token_mismatch_examples": last_mismatches[:10],
        "length_spread_mismatch_examples": spread_mismatches[:10],
        "passed": not (too_long or last_mismatches or spread_mismatches),
    }


def validate_audit(audit):
    if not audit["passed"]:
        raise RuntimeError(
            "PROMPT_AUDIT_FAIL: review examples and revise all prompt suffixes "
            "before GPU use. overlength={overlength_count} "
            "last_token_mismatch={last_token_mismatch_count} "
            "length_spread_mismatch={length_spread_mismatch_count}".format(**audit))


def audit_command(args):
    train, val, _ = base.load_protocol()
    tokenizer = model_tokenizer(args.model)
    report = audit_rows(tokenizer, [("train", train), ("val", val)],
                        args.max_length, args.max_spread)
    out = ROOT / "results/route_e/task_prompt" / (
        "prompt_token_audit_" + report["prompt_suffixes_sha256"][:12] + ".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if existing != report:
            raise FileExistsError(f"Audit report already exists but content changed: {out}")
    else:
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                       encoding="utf-8")
    print(f"PROMPT_AUDIT_SAVED path={out} count={report['sample_count']}")
    for name, s in report["variants"].items():
        print(f"  {name}: mean_input={s['mean_input_tokens']:.2f} "
              f"mean_added={s['mean_added_tokens_vs_p1']:.2f} "
              f"last_token_id_counts={s['last_token_id_counts']}")
    validate_audit(report)
    print("PROMPT_AUDIT_PASS")


def extract(args):
    import numpy as np
    import torch
    from transformers import AutoModel
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for feature extraction")
    train, val, _ = base.load_protocol()
    tokenizer = model_tokenizer(args.model)
    # Fail closed before loading model weights. Inspect *all* 3450 questions
    # even when the requested extraction itself is a smoke run.
    full_audit = audit_rows(tokenizer, [("train", train), ("val", val)],
                            args.max_length, args.max_spread)
    validate_audit(full_audit)
    if args.smoke:
        train, val = base.choose_smoke_rows(train, val)
    torch.manual_seed(args.seed)
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model = AutoModel.from_pretrained(
        str(model_path), torch_dtype=torch.float16,
        local_files_only=True, trust_remote_code=False).to("cuda").eval()
    for par in model.parameters():
        par.requires_grad_(False)
    pad_id = tokenizer.pad_token_id
    for split, rows, source in (("train", train, base.TRAIN),
                                ("val", val, base.VAL)):
        dst = cache_dir(args.mode, args.smoke) / f"{split}.npz"
        if dst.exists():
            raise FileExistsError(f"Refusing to overwrite existing features: {dst}")
        started = time.perf_counter()
        encoded = [encode_question(tokenizer, r["problem"], args.mode)
                   for r in rows]
        if max(map(len, encoded)) > args.max_length:
            raise RuntimeError("Overlength sample unexpectedly passed audit")
        matrices, elapsed, counts = [], [], []
        with torch.inference_mode():
            for start in range(0, len(encoded), args.batch_size):
                chunk = encoded[start:start+args.batch_size]
                width = max(map(len, chunk))
                ids = torch.full((len(chunk), width), pad_id,
                                 dtype=torch.long, device="cuda")
                mask = torch.zeros_like(ids)
                for j, one in enumerate(chunk):
                    ids[j, :len(one)] = torch.tensor(one, device="cuda")
                    mask[j, :len(one)] = 1
                torch.cuda.synchronize()
                stamp = time.perf_counter()
                out = model(input_ids=ids, attention_mask=mask,
                            return_dict=True, use_cache=False)
                pos = torch.tensor([len(s)-1 for s in chunk], device="cuda")
                h = out.last_hidden_state[
                    torch.arange(len(chunk), device="cuda"), pos].float().cpu()
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter()-stamp)*1000.)
                counts.append(len(chunk))
                if not torch.isfinite(h).all():
                    raise RuntimeError(f"Nonfinite feature in {split} at {start}")
                matrices.append(h.numpy().astype(np.float16))
                if start == 0 or start+args.batch_size >= len(encoded) or (
                        start//args.batch_size) % 20 == 0:
                    print(f"TASK_PROMPT_EXTRACT {args.mode} {split} "
                          f"{min(start+args.batch_size,len(rows))}/{len(rows)}",
                          flush=True)
        X = np.concatenate(matrices)
        if X.shape != (len(rows), 1536) or not np.isfinite(X).all():
            raise RuntimeError(f"Invalid feature matrix: {X.shape}")
        meta = {
            "schema": SCHEMA, "encoder": "deepseek",
            "model": str(model_path),
            "revision": getattr(model.config, "_commit_hash", None),
            "prompt_mode": args.mode, "prompt_suffix": SUFFIXES[args.mode],
            "prompt_suffixes_sha256": full_audit["prompt_suffixes_sha256"],
            "chat_template_sha256": full_audit["chat_template_sha256"],
            "pooling": "last_valid_prompt_token", "layer": "last_after_final_norm",
            "dtype": str(next(model.parameters()).dtype),
            "sample_count": len(rows), "feature_dim": X.shape[1],
            "labels_sha256": base.sha256(source),
            "max_length": args.max_length, "max_spread": args.max_spread,
            "batch_size": args.batch_size, "seed": args.seed,
            "mean_input_tokens": sum(map(len, encoded))/len(encoded),
            "max_input_tokens": max(map(len, encoded)),
            "first_batch_forward_ms": elapsed[0],
            "post_first_batch_forward_ms_per_question": (
                sum(elapsed[1:])/sum(counts[1:]) if len(counts)>1 else None),
            "full_forward_seconds": sum(elapsed)/1000.,
            "wall_seconds": time.perf_counter()-started,
            "smoke": args.smoke, "truncated": 0, "skipped": 0,
        }
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, X=X,
                            sample_id=np.asarray([r["sample_id"] for r in rows]),
                            metadata_json=np.asarray(json.dumps(meta)))
        print(f"TASK_PROMPT_FEATURES_SAVED mode={args.mode} "
              f"split={split} X={X.shape} path={dst}", flush=True)


def read_cache(path, rows, expected_sha, mode, smoke):
    import numpy as np
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        X = data["X"].astype(np.float32)
        ids = data["sample_id"].astype(str).tolist()
        meta = json.loads(data["metadata_json"].item())
    if meta["schema"] != SCHEMA or meta["prompt_mode"] != mode:
        raise ValueError(f"Invalid task prompt schema/mode in {path}")
    if bool(meta["smoke"]) != smoke or meta["labels_sha256"] != expected_sha:
        raise ValueError(f"Split or strict SHA mismatch: {path}")
    if meta["prompt_suffix"] != SUFFIXES[mode] or meta["pooling"] != "last_valid_prompt_token":
        raise ValueError(f"Prompt/pooling mismatch: {path}")
    if ids != [r["sample_id"] for r in rows]:
        raise ValueError(f"Sample order/IDs mismatch: {path}")
    if X.shape != (len(rows), 1536) or not np.isfinite(X).all():
        raise ValueError(f"Invalid cached features: {path} {X.shape}")
    return X, meta


def fit(args):
    import numpy as np
    from train_modernbert_delta import curve_point, endpoints
    train, val, overlaps = base.load_protocol()
    if args.smoke:
        train, val = base.choose_smoke_rows(train, val)
    folder = cache_dir(args.mode, args.smoke)
    Xtr, trmeta = read_cache(folder/"train.npz", train, base.TRAIN_SHA256,
                             args.mode, args.smoke)
    Xva, vameta = read_cache(folder/"val.npz", val, base.VAL_SHA256,
                             args.mode, args.smoke)
    for key in ("model", "chat_template_sha256", "prompt_suffixes_sha256",
                "prompt_mode", "layer", "max_length", "max_spread", "dtype"):
        if trmeta[key] != vameta[key]:
            raise RuntimeError(f"Train/val metadata differs: {key}")
    ytr = np.asarray([int(r["p_n"] >= r["p_t"]) for r in train], dtype=np.float32)
    yva = np.asarray([int(r["p_n"] >= r["p_t"]) for r in val], dtype=np.float32)
    fitted = base.train_logistic(Xtr, ytr, args.l2, args.max_iter)
    scores = base.predict_linear(Xva, fitted)
    predictions = {r["sample_id"]: float(score) for r, score in zip(val, scores)}
    curve = [curve_point(val, {sid: -sc for sid,sc in predictions.items()}, frac)
             for frac in base.FRACTIONS]
    pred_path = folder / "validation_p_nt.json"
    model_path = folder / "linear_probe.npz"
    report_dir = ROOT / "results/route_e/task_prompt"
    report_path = report_dir/f"{args.mode}_{'smoke' if args.smoke else 'val'}.json"
    if pred_path.exists() or model_path.exists() or report_path.exists():
        raise FileExistsError("Refusing to replace existing probe, predictions, or report")
    report = {
        "method": "route_e_task_prompt_"+args.mode,
        "smoke": args.smoke, "encoder_frozen": True,
        "strict_supervision": "1[p_nothinking_strict>=p_thinking]",
        "head": "linear_logistic_regression", "objective": "hard_bce_l2",
        "l2": args.l2, "max_iter": args.max_iter,
        "validation_auc_diagnostic_only": base.auc(scores, yva),
        "train_sha256": base.TRAIN_SHA256, "val_sha256": base.VAL_SHA256,
        "train_count": len(train), "validation_count": len(val),
        "train_val_normalized_text_overlaps": overlaps,
        "train_features": trmeta, "val_features": vameta,
        "router_overhead_included_in_output_tokens": False,
        "llm_generation_repeated": False,
        "endpoints": endpoints(val), "curve": curve,
    }
    folder.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(json.dumps({"p_nt": predictions}, indent=2)+"\n",
                         encoding="utf-8")
    np.savez_compressed(model_path, weight=fitted["weight"],
                        bias=np.asarray(fitted["bias"]), mean=fitted["mean"],
                        scale=fitted["scale"])
    report_path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"TASK_PROMPT_FIT_PASS mode={args.mode} val_auc={report['validation_auc_diagnostic_only']:.4f} "
          f"predictions={pred_path} report={report_path}", flush=True)


def self_test():
    prefix.self_test()
    class FakeTokenizer:
        name_or_path = "fake"
        chat_template = "fake"
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            prompt = "<BOS><｜User｜>"+messages[0]["content"]
            if add_generation_prompt:
                prompt += prefix.ASSISTANT_START+prefix.THINK_START
            return [ord(x) for x in prompt] if tokenize else prompt
        def __call__(self, content, add_special_tokens, truncation=False):
            assert add_special_tokens is False and truncation is False
            return {"input_ids": [ord(x) for x in content]}
    t = FakeTokenizer()
    for mode in MODES:
        s = render_prompt(t, "1+1?", mode)
        assert s.endswith(SUFFIXES[mode])
        assert s.startswith(prefix.render_prompts(t,"1+1?")["assistant"])
        assert encode_question(t, "1+1?", mode)[-1] == ord('"')
    print("TASK_PROMPT_SELFTEST_PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    for action in ("inspect-template", "audit", "extract", "fit"):
        a = commands.add_parser(action)
        if action in ("inspect-template", "audit", "extract"):
            a.add_argument("--model", default=MODEL)
        if action in ("audit", "extract"):
            a.add_argument("--max-length", type=int, default=1024)
            a.add_argument("--max-spread", type=int, default=2)
        if action in ("extract", "fit"):
            a.add_argument("--mode", choices=MODES, required=True)
            a.add_argument("--smoke", action="store_true")
            a.add_argument("--seed", type=int, default=42)
        if action == "extract":
            a.add_argument("--batch-size", type=int, default=8)
        if action == "fit":
            a.add_argument("--l2", type=float, default=1e-3)
            a.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "inspect-template":
        inspect_template(args.model)
    elif args.command == "audit":
        if args.max_length < 1 or args.max_spread < 0:
            parser.error("Invalid audit limits")
        audit_command(args)
    elif args.command == "extract":
        if args.batch_size < 1 or args.max_length < 1 or args.max_spread < 0:
            parser.error("Invalid extraction configuration")
        extract(args)
    elif args.command == "fit":
        if not math.isfinite(args.l2) or args.l2 < 0 or args.max_iter < 1:
            parser.error("Invalid fit configuration")
        fit(args)


if __name__ == "__main__":
    main()
