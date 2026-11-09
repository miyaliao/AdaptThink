#!/usr/bin/env python3
"""Frozen P1 layer ablation for the historical Cost-only Ridge router.

GPU: extract all layers in ONE forward pass per batch.
CPU: fit the SAME two log1p token-cost Ridge heads independently per layer.
No test access, no generation, no backbone updates, no validation-driven hyperparameter search.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_e_frozen as base
import route_e_prompt_ablation as prefix
from route_e_equal_budget import dense_curve, linear_bracket

ROOT = base.ROOT
LAYERS = (7, 14, 16, 21, 24, 26, 27, 28)
L2 = 0.02  # historical cost heads
MAX_OUTPUT_TOKENS = 16384
BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)


def folder(smoke):
    return ROOT / "outputs/route_e/cost_layer_ablation" / ("smoke" if smoke else "full")


def extract(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen prefill extraction requires GPU")
    tr, va, _ = base.load_protocol()
    if args.smoke:
        tr, va = base.choose_smoke_rows(tr, va)
    dest = folder(args.smoke)
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    tok = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    tok.padding_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    prefix.render_prompts(tok, "What is 2 + 2?")  # fail if official template differs
    model = AutoModel.from_pretrained(
        str(model_path), torch_dtype=torch.float16, local_files_only=True
    ).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if model.config.num_hidden_layers != 28:
        raise ValueError("Expected exactly 28 transformer layers")
    for split, rows, label_file in (("train", tr, base.TRAIN), ("val", va, base.VAL)):
        path = dest / (split + ".npz")
        if path.exists():
            raise FileExistsError(f"Refusing overwrite: {path}")
        seqs = prefix.tokenize_ablations(
            tok, [r["problem"] for r in rows], "assistant", args.max_length
        )
        features = {layer: [] for layer in LAYERS}
        with torch.inference_mode():
            for start in range(0, len(seqs), args.batch_size):
                batch = seqs[start:start + args.batch_size]
                width = max(map(len, batch))
                ids = torch.full((len(batch), width), tok.pad_token_id,
                                 dtype=torch.long, device="cuda")
                mask = torch.zeros_like(ids)
                for i, seq in enumerate(batch):
                    ids[i, :len(seq)] = torch.tensor(seq, device="cuda")
                    mask[i, :len(seq)] = 1
                out = model(input_ids=ids, attention_mask=mask, use_cache=False,
                            output_hidden_states=True, return_dict=True)
                if len(out.hidden_states) != 29:
                    raise ValueError(f"Expected embeddings + 28 layers; got {len(out.hidden_states)}")
                positions = torch.tensor([len(s)-1 for s in batch], device="cuda")
                idx = torch.arange(len(batch), device="cuda")
                for layer in LAYERS:
                    # HF hidden_states[L] is the representation after block L.
                    h = out.last_hidden_state if layer == 28 else out.hidden_states[layer]
                    picked = h[idx, positions].float()
                    if not torch.isfinite(picked).all():
                        raise ValueError(f"Nonfinite layer {layer} at batch {start}")
                    features[layer].append(picked.cpu().numpy().astype(np.float16))
                del out
                if start == 0 or start+args.batch_size >= len(seqs) or start//args.batch_size % 30 == 0:
                    print(f"LAYER_EXTRACT {split} {min(start+args.batch_size,len(seqs))}/{len(seqs)}", flush=True)
        arrays = {f"L{layer}": np.concatenate(features[layer]) for layer in LAYERS}
        metadata = {
            "layers": LAYERS, "pooling": "last_valid_prompt_token",
            "prompt": "P1 assistant", "last_layer": "after_final_norm",
            "intermediate": "transformer_block_output",
            "model": str(model_path), "dataset_sha256": base.sha256(label_file),
            "smoke": args.smoke, "max_length": args.max_length,
            "sample_count": len(rows)
        }
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays,
                            sample_id=np.array([r["sample_id"] for r in rows]),
                            metadata_json=np.array(json.dumps(metadata)))
        print(f"LAYER_CACHE_SAVED {path}", flush=True)


def load(split, rows, smoke):
    path = folder(smoke) / (split + ".npz")
    with np.load(path, allow_pickle=False) as obj:
        meta = json.loads(obj["metadata_json"].item())
        if meta["dataset_sha256"] != base.sha256(base.TRAIN if split == "train" else base.VAL):
            raise ValueError("Strict label hash mismatch")
        if meta["sample_count"] != len(rows) or meta["smoke"] != smoke:
            raise ValueError("Split mismatch")
        if obj["sample_id"].astype(str).tolist() != [r["sample_id"] for r in rows]:
            raise ValueError("Sample ID order mismatch")
        return {L: obj[f"L{L}"].astype(np.float64) for L in LAYERS}


def train_cost_heads(X, rows):
    """Historical Ridge specification, adapted to the two cost targets only."""
    y = np.log1p(np.array([[r["t_t"], r["t_n"]] for r in rows], dtype=np.float64))
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-4)
    Z = (X - mean) / std
    ym = y.mean(axis=0)
    n, d = Z.shape
    gram = Z.T @ Z / n
    gram.flat[::d+1] += L2
    weights = np.linalg.solve(gram, (Z.T @ (y-ym)) / n)
    intercept = ym - Z.mean(axis=0) @ weights
    residual = y - (Z @ weights + intercept)
    smear = np.exp(np.clip(residual, -12., 12.)).mean(axis=0)
    return mean, std, weights, intercept, smear


def predict_cost(head, X):
    mean, std, weights, intercept, smear = head
    raw = ((X-mean)/std) @ weights + intercept
    clipped = np.clip(raw, 0., np.log1p(MAX_OUTPUT_TOKENS))
    return np.clip(np.exp(clipped)*smear - 1., 0., MAX_OUTPUT_TOKENS)


def ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)


def correlation(x, y):
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def nt_probs(rows, scores, bracket):
    order = sorted(range(len(rows)), key=lambda i: (-scores[i], rows[i]["sample_id"]))
    left = round(bracket["policy_left_fraction"] * len(rows))
    right = round(bracket["policy_right_fraction"] * len(rows))
    w = bracket["mix_right_probability"]
    p = np.zeros(len(rows), dtype=np.float64)
    p[order[:left]] += 1.-w
    p[order[:right]] += w
    return p


def eval_layer(rows, scores):
    scores_map = {r["sample_id"]: float(score) for r, score in zip(rows, scores)}
    curve = dense_curve(rows, scores_map)
    output = {}
    for budget in BUDGETS:
        try:
            bracket = linear_bracket(curve, budget)
            pn = nt_probs(rows, scores, bracket)
            acc = np.mean([
                (1-p)*r["p_t"] + p*r["p_n"]
                for p, r in zip(pn, rows)
            ])
            actual_tokens = np.mean([
                (1-p)*r["t_t"] + p*r["t_n"]
                for p, r in zip(pn, rows)
            ])
            if abs(actual_tokens-budget) > 1e-4:
                raise AssertionError(f"Budget mismatch: {actual_tokens} vs {budget}")
            output[str(budget)] = {"accuracy": float(acc), "nothinking_fraction": float(pn.mean())}
        except ValueError as exc:
            output[str(budget)] = {"available": False, "reason": str(exc)}
    return output


def evaluate(args):
    tr, va, _ = base.load_protocol()
    if args.smoke:
        tr, va = base.choose_smoke_rows(tr, va)
    train = load("train", tr, args.smoke)
    val = load("val", va, args.smoke)
    true_delta = np.array([r["t_t"]-r["t_n"] for r in va])
    report = {"protocol": "frozen_P1_cost_only_layer_ablation", "l2": L2,
              "budgets": BUDGETS, "smoke": args.smoke, "layers": {}}
    for L in LAYERS:
        head = train_cost_heads(train[L], tr)
        pred = predict_cost(head, val[L])
        extra = pred[:, 0]-pred[:, 1]
        ds = np.array([r["dataset"] for r in va])
        group_corr = {}
        for name in sorted(set(ds)):
            subset = ds == name
            group_corr[name] = correlation(extra[subset], true_delta[subset])
        record = {
            "spearman_extra_tokens": correlation(extra, true_delta),
            "spearman_by_dataset": group_corr,
            "mae_extra_tokens": float(np.mean(np.abs(extra-true_delta))),
            "cost_only": eval_layer(va, extra),
        }
        report["layers"][str(L)] = record
        print(f"LAYER_COST L{L} spearman={record['spearman_extra_tokens']:.4f} "
              + " ".join(f"B{b}={record['cost_only'][str(b)].get('accuracy',float('nan')):.4%}"
                         for b in BUDGETS), flush=True)
    dest = ROOT / "results/route_e/cost_layer_ablation"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / ("smoke.json" if args.smoke else "validation.json")
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"LAYER_COST_DONE {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("check", "extract", "evaluate"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--model", default=base.DEFAULT_MODELS["deepseek"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    args = p.parse_args()
    if args.command == "check":
        tr, va, _ = base.load_protocol()
        print(f"LAYER_DATA_OK train={len(tr)} val={len(va)} layers={LAYERS}")
    elif args.command == "extract":
        extract(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
