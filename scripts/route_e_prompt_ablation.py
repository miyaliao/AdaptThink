#!/usr/bin/env python3
"""Route E prompt-prefix ablation: frozen DeepSeek P1 / P2, no test reads.

P0 is the existing route_e_frozen.py DeepSeek arm (Assistant + <think>).
P1 = assistant: same user chat prompt plus <｜Assistant｜>, without <think>.
P2 = user_only: same chat prompt with add_generation_prompt=False.

Train/validation data, strict labels, final-layer last-real-token pooling,
logistic probe, and cost-matched Random evaluation are identical to P0.
Features live in *separate* output directories; nothing overwrites P0.

Examples:
 python scripts/route_e_prompt_ablation.py self-test
 python scripts/route_e_prompt_ablation.py inspect-template
 python scripts/route_e_prompt_ablation.py extract --prompt-mode assistant --smoke
 python scripts/route_e_prompt_ablation.py fit --prompt-mode assistant --smoke
 python scripts/route_e_prompt_ablation.py extract --prompt-mode assistant
 python scripts/route_e_prompt_ablation.py fit --prompt-mode assistant
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import route_e_frozen as base

ROOT = base.ROOT
MODEL = base.DEFAULT_MODELS["deepseek"]
ASSISTANT_START = "<｜Assistant｜>"
THINK_START = "<think>\n"
MODES = ("assistant", "user_only")
FRACTIONS = base.FRACTIONS


def cache_dir(mode, smoke):
    return ROOT / "outputs/route_e/prompt_ablation" / mode / ("smoke" if smoke else "full")


def render_prompts(tokenizer, question):
    """Construct exact nested prefixes with the *loaded* tokenizer template.

    Fail closed if this particular checkpoint does not use the official
    DeepSeek-R1-Distill-Qwen-1.5B assistant+think generation template.
    """
    messages = [{"role": "user", "content": question}]
    user_only = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    p0 = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    expected = user_only + ASSISTANT_START + THINK_START
    if p0 != expected:
        raise ValueError(
            "Local tokenizer chat template differs from expected DeepSeek-R1 "
            "assistant+think prefix. Abort before extracting features. "
            f"expected suffix={repr(expected[-100:])}; "
            f"actual suffix={repr(p0[-100:])}")
    p1 = user_only + ASSISTANT_START
    return {"user_only": user_only, "assistant": p1,
            "assistant_think": p0}


def tokenize_ablations(tokenizer, questions, mode, max_length):
    sequences = []
    for i, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid question at position {i}")
        prompts = render_prompts(tokenizer, question)
        # The chat template itself emits the BOS token. Do not append another.
        ids = tokenizer(prompts[mode], add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Empty tokenization at position {i}")
        # Verify end-to-end that the P0/P2 text paths agree with HF's official
        # tokenize=True paths, including their special tokens.
        if i == 0:
            messages = [{"role": "user", "content": question}]
            for value, use_generation_prompt in (
                    ("user_only", False), ("assistant_think", True)):
                baseline_ids = tokenizer.apply_chat_template(
                    messages, tokenize=True,
                    add_generation_prompt=use_generation_prompt)
                ours = tokenizer(prompts[value], add_special_tokens=False)["input_ids"]
                if ours != baseline_ids:
                    raise RuntimeError(
                        f"Tokenization mismatch for {value}: "
                        "render+tokenize differs from apply_chat_template")
        if len(ids) > max_length:
            raise ValueError(
                f"Overlength question at index={i} mode={mode}: "
                f"{len(ids)} > max_length={max_length}. Nothing was truncated.")
        sequences.append(ids)
    return sequences


def metadata_path(mode, smoke, split):
    return cache_dir(mode, smoke) / f"{split}.npz"


def extract(args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for feature extraction (use Slurm).")
    train, val, _ = base.load_protocol()
    if args.smoke:
        train, val = base.choose_smoke_rows(train, val)
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Validate local template before paying for model loading.
    inspection = render_prompts(tokenizer, "What is 25 * 16?")
    print("TEMPLATE_OK mode=" + args.prompt_mode +
          " p0_suffix=" + repr(inspection["assistant_think"][-50:]) +
          " p1_suffix=" + repr(inspection["assistant"][-50:]), flush=True)
    template_hash = hashlib.sha256(
        str(tokenizer.chat_template).encode("utf-8")).hexdigest()
    model = AutoModel.from_pretrained(
        str(model_path), torch_dtype=torch.float16,
        local_files_only=True, trust_remote_code=False).to("cuda").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pad_id = tokenizer.pad_token_id

    for split, rows, labels_path in (("train", train, base.TRAIN),
                                    ("val", val, base.VAL)):
        dst = metadata_path(args.prompt_mode, args.smoke, split)
        if dst.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing cache: {dst}. "
                "Delete the failed cache explicitly if re-extraction is needed.")
        t_start = time.perf_counter()
        encoded = tokenize_ablations(
            tokenizer, [r["problem"] for r in rows],
            args.prompt_mode, args.max_length)
        tokenization_seconds = time.perf_counter() - t_start
        embeddings = []
        measured_ms = []
        measured_batch_counts = []
        with torch.inference_mode():
            for start in range(0, len(encoded), args.batch_size):
                batch = encoded[start:start + args.batch_size]
                length = max(map(len, batch))
                ids = torch.full((len(batch), length), pad_id,
                                 dtype=torch.long, device="cuda")
                mask = torch.zeros_like(ids)
                for j, seq in enumerate(batch):
                    ids[j, :len(seq)] = torch.tensor(seq, dtype=torch.long,
                                                     device="cuda")
                    mask[j, :len(seq)] = 1
                torch.cuda.synchronize()
                begin = time.perf_counter()
                out = model(input_ids=ids, attention_mask=mask,
                            return_dict=True, use_cache=False)
                position = torch.tensor([len(seq)-1 for seq in batch],
                                        device="cuda")
                picked = out.last_hidden_state[
                    torch.arange(len(batch), device="cuda"), position]
                picked = picked.float().cpu()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - begin)*1000
                if not torch.isfinite(picked).all():
                    raise ValueError(f"Nonfinite hidden states in {split} batch {start}")
                embeddings.append(picked.numpy().astype(np.float16))
                measured_ms.append(elapsed_ms)
                measured_batch_counts.append(len(batch))
                if start == 0 or start+args.batch_size >= len(encoded) or (
                        start // args.batch_size) % 20 == 0:
                    print(f"ABLAT_EXTRACT mode={args.prompt_mode} "
                          f"split={split} n={min(start+args.batch_size, len(rows))}/{len(rows)}",
                          flush=True)
        X = np.concatenate(embeddings)
        if X.shape != (len(rows), 1536) or not np.isfinite(X).all():
            raise ValueError(f"Unexpected feature matrix {X.shape}; expected (N,1536)")
        # The first forward batch includes cold-start effects. Log separately;
        # this is *not* end-to-end single-query router latency.
        postwarm_ms = sum(measured_ms[1:])
        postwarm_questions = sum(measured_batch_counts[1:])
        metadata = {
            "schema": "route_e_prefix_features_v1",
            "encoder": "deepseek",
            "model": str(model_path),
            "revision": getattr(model.config, "_commit_hash", None),
            "prompt_mode": args.prompt_mode,
            "chat_template_sha256": template_hash,
            "pooling": "last_valid_prompt_token",
            "layer": "last_after_final_norm",
            "dtype": str(next(model.parameters()).dtype),
            "sample_count": len(rows), "feature_dim": X.shape[1],
            "labels_sha256": base.sha256(labels_path),
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_input_tokens": max(map(len, encoded)),
            "mean_input_tokens": sum(map(len, encoded))/len(encoded),
            "tokenization_seconds": tokenization_seconds,
            "first_batch_forward_ms": measured_ms[0],
            "forward_seconds_including_cold": sum(measured_ms)/1000,
            "post_first_batch_forward_ms_per_question": (
                postwarm_ms/postwarm_questions if postwarm_questions else None),
            "wall_seconds": time.perf_counter()-t_start,
            "smoke": args.smoke,
            "truncated": 0, "skipped": 0,
        }
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, X=X,
                            sample_id=np.asarray([r["sample_id"] for r in rows]),
                            metadata_json=np.asarray(json.dumps(metadata)))
        print(f"ABLAT_FEATURES_SAVED mode={args.prompt_mode} "
              f"split={split} X={X.shape} path={dst} "
              f"postwarm_ms={metadata['post_first_batch_forward_ms_per_question']}",
              flush=True)


def read_cache(path, rows, label_hash, mode, smoke):
    import numpy as np
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata_json"].item()))
        ids = f["sample_id"].astype(str).tolist()
        X = f["X"].astype(np.float32)
    if meta["schema"] != "route_e_prefix_features_v1":
        raise ValueError(f"Unexpected metadata schema in {path}")
    if meta["prompt_mode"] != mode or bool(meta["smoke"]) != smoke:
        raise ValueError(f"Wrong prefix mode or split type in {path}")
    if meta["labels_sha256"] != label_hash:
        raise ValueError(f"Strict labels SHA mismatch in {path}")
    if ids != [r["sample_id"] for r in rows]:
        raise ValueError(f"Sample IDs/order mismatch in {path}")
    if X.shape != (len(rows), 1536) or not np.isfinite(X).all():
        raise ValueError(f"Invalid features {X.shape} in {path}")
    return X, meta


def fit(args):
    import numpy as np
    from train_modernbert_delta import curve_point, endpoints
    train, val, overlaps = base.load_protocol()
    if args.smoke:
        train, val = base.choose_smoke_rows(train, val)
    folder = cache_dir(args.prompt_mode, args.smoke)
    Xtr, train_meta = read_cache(folder/"train.npz", train, base.TRAIN_SHA256,
                                args.prompt_mode, args.smoke)
    Xva, val_meta = read_cache(folder/"val.npz", val, base.VAL_SHA256,
                              args.prompt_mode, args.smoke)
    for key in ("model", "prompt_mode", "chat_template_sha256", "pooling",
                "layer", "dtype", "max_length"):
        if train_meta[key] != val_meta[key]:
            raise ValueError(f"Inconsistent train/val feature metadata: {key}")
    ytr = np.asarray([int(r["p_n"] >= r["p_t"]) for r in train], dtype=np.float32)
    yva = np.asarray([int(r["p_n"] >= r["p_t"]) for r in val], dtype=np.float32)
    if len(set(ytr)) != 2:
        raise ValueError("Training split contains only one label")
    started = time.perf_counter()
    trained = base.train_logistic(Xtr, ytr, args.l2, args.max_iter)
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    p_nt = base.predict_linear(Xva, trained)
    prediction_ms = (time.perf_counter()-started)*1000/len(val)
    preds = {r["sample_id"]: float(p) for r,p in zip(val, p_nt)}
    # Route A scorer ranks by ascending score for NoThinking.
    curve = [curve_point(val, {sid: -p for sid,p in preds.items()}, f)
             for f in FRACTIONS]
    report = {
        "method": "route_e_deepseek_prompt_" + args.prompt_mode,
        "smoke": args.smoke, "encoder": "deepseek", "encoder_frozen": True,
        "strict_supervision": "1[p_nothinking_strict>=p_thinking]",
        "head": "linear_logistic_regression", "objective": "hard_bce_l2",
        "l2": args.l2, "max_iter": args.max_iter, "seed": args.seed,
        "train_sha256": base.TRAIN_SHA256, "val_sha256": base.VAL_SHA256,
        "train_count": len(train), "validation_count": len(val),
        "train_class_1": int(ytr.sum()), "validation_class_1": int(yva.sum()),
        "train_val_normalized_text_overlaps": overlaps,
        "validation_auc_diagnostic_only": base.auc(p_nt, yva),
        "train_objective": trained["train_objective"],
        "train_time_seconds": train_seconds,
        "prediction_ms_per_question_cpu_batch": prediction_ms,
        "train_features": train_meta, "val_features": val_meta,
        "router_overhead_included_in_output_tokens": False,
        "llm_generation_repeated": False,
        "random_reference": "Route A exact cost-matched Global Random",
        "endpoints": endpoints(val), "curve": curve,
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"validation_p_nt.json").write_text(
        json.dumps({"p_nt": preds}, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(folder/"linear_probe.npz",
                        weight=trained["weight"], bias=np.asarray(trained["bias"]),
                        mean=trained["mean"], scale=trained["scale"])
    result_dir = ROOT/"results/route_e/prompt_ablation"
    result_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.prompt_mode}_{'smoke' if args.smoke else 'val'}"
    target = result_dir/f"{filename}.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite report: {target}")
    target.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (result_dir/f"{filename}.csv").open("w", newline="", encoding="utf-8") as fd:
        writer = csv.writer(fd)
        writer.writerow(["dataset", "nt_fraction_global", "accuracy", "mean_tokens",
                         "nt_fraction_dataset", "random_accuracy", "uplift_pp"])
        for point in curve:
            for ds, m in point["groups"].items():
                ref = m["matched_random"]
                writer.writerow([ds, point["target_nt_fraction"], m["accuracy"],
                                 m["mean_tokens"], m["nt_fraction"],
                                 None if ref is None else ref["accuracy"], m["uplift_pp"]])
    for point in curve:
        m = point["groups"]["overall"]
        uplift = "N/A" if m["uplift_pp"] is None else f"{m['uplift_pp']:+.2f}pp"
        print(f"ABLAT_VAL mode={args.prompt_mode} smoke={int(args.smoke)} "
              f"NT={point['target_nt_fraction']:.0%} "
              f"acc={m['accuracy']:.4%} tokens={m['mean_tokens']:.1f} "
              f"uplift={uplift}", flush=True)
    print(f"ABLAT_DONE mode={args.prompt_mode} smoke={int(args.smoke)} "
          f"report={target}", flush=True)


def inspect_template(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False)
    examples = render_prompts(tokenizer, "What is 25 * 16?")
    for mode, value in examples.items():
        print(f"TEMPLATE mode={mode} repr={value!r}")
    for mode in ("user_only", "assistant", "assistant_think"):
        sequences = tokenize_ablations(tokenizer, ["What is 25 * 16?"], mode, 1024)
        print(f"TEMPLATE_TOKENS mode={mode} length={len(sequences[0])}")
    print("TEMPLATE_INSPECTION_PASS")


def self_test():
    base.self_test()
    class Fake:
        chat_template = "synthetic"
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            s = "<BOS><｜User｜>" + messages[0]["content"]
            if add_generation_prompt:
                s += ASSISTANT_START + THINK_START
            return [ord(char) for char in s] if tokenize else s
        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(char) for char in text]}
    fake = Fake()
    prompts = render_prompts(fake, "1+1?")
    assert prompts["assistant_think"].endswith(ASSISTANT_START+THINK_START)
    assert prompts["assistant"].endswith(ASSISTANT_START)
    assert not prompts["user_only"].endswith(ASSISTANT_START)
    for mode in MODES:
        assert tokenize_ablations(fake, ["1+1?"], mode, 1024)
    print("ABLAT_SELFTEST_PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    inspect = commands.add_parser("inspect-template")
    inspect.add_argument("--model", default=MODEL)
    for action in ("extract", "fit"):
        q = commands.add_parser(action)
        q.add_argument("--prompt-mode", choices=MODES, required=True)
        q.add_argument("--seed", type=int, default=42)
        q.add_argument("--smoke", action="store_true")
        if action == "extract":
            q.add_argument("--model", default=MODEL)
            q.add_argument("--batch-size", type=int, default=8)
            q.add_argument("--max-length", type=int, default=1024)
        else:
            q.add_argument("--l2", type=float, default=1e-3)
            q.add_argument("--max-iter", type=int, default=200)
    args = p.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "inspect-template":
        inspect_template(args)
    elif args.command == "extract":
        if args.batch_size < 1 or args.max_length < 1:
            p.error("batch-size and max-length must be positive")
        extract(args)
    else:
        if not math.isfinite(args.l2) or args.l2 < 0 or args.max_iter < 1:
            p.error("invalid logistic hyperparameters")
        fit(args)


if __name__ == "__main__":
    main()
