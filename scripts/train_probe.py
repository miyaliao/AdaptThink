#!/usr/bin/env python3
"""Fit the routing probe on the model's hidden states and report honest AUCs.

The probe is a logistic regression on the last-layer hidden state at the decision
position. It answers one question before we invest more in training:

    is "should this problem skip thinking" linearly decodable from what the model
    already represents?

If the probe AUC is near 0.5, the label is not predictable from the prompt with this
model, and no amount of LoRA will fix that. If it is high, the probe itself is a
deployable router: one forward pass, one threshold, no generation.

Outputs
-------
    results/probe_report.json          AUCs, distributions, chosen threshold
    results/probe_<split>_p_nt.json    sample_id -> p_nt for the curve evaluator

Usage
-----
    python scripts/train_probe.py \
        --train outputs/hidden/train.npz --val outputs/hidden/val.npz \
        --test outputs/hidden/test.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def auc(scores, labels):
    order = np.argsort(scores, kind="mergesort")
    scores, labels = np.asarray(scores)[order], np.asarray(labels)[order]
    n_pos = float((labels > 0.5).sum())
    n_neg = float(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[j + 1] == scores[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = ranks[labels > 0.5].sum()
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def describe(values):
    values = np.asarray(values)
    return {"min": float(values.min()), "p10": float(np.quantile(values, .10)),
            "median": float(np.quantile(values, .50)), "p90": float(np.quantile(values, .90)),
            "max": float(values.max()), "mean": float(values.mean())}


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return {"sample_id": data["sample_id"].astype(str), "X": data["X"].astype(np.float32),
            "soft": data["soft"].astype(np.float32), "hard": data["hard"].astype(np.float32)}


def fit_logistic(X_train, y_train, X_val, y_val, epochs=400, lr=0.05, l2=1e-4, seed=0):
    """Full-batch logistic regression in torch, early-stopped on validation AUC."""
    import torch
    torch.manual_seed(seed)
    mean, std = X_train.mean(0, keepdims=True), X_train.std(0, keepdims=True) + 1e-6
    Xtr = torch.tensor((X_train - mean) / std, dtype=torch.float32)
    Xva = torch.tensor((X_val - mean) / std, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)
    head = torch.nn.Linear(X_train.shape[1], 1)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best = {"auc": -1.0, "state": None, "epoch": None}
    for epoch in range(1, epochs + 1):
        head.train()
        optimizer.zero_grad()
        logits = head(Xtr).squeeze(-1)
        loss = loss_fn(logits, ytr) + l2 * (head.weight ** 2).sum()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs:
            head.eval()
            with torch.no_grad():
                val_scores = torch.sigmoid(head(Xva).squeeze(-1)).numpy()
            score = auc(val_scores, y_val)
            if score > best["auc"]:
                best = {"auc": score, "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    head.load_state_dict(best["state"])
    head.eval()
    return head, mean, std, best["auc"], best["epoch"]


def predict(head, mean, std, X):
    import torch
    with torch.no_grad():
        tensor = torch.tensor((X - mean) / std, dtype=torch.float32)
        return torch.sigmoid(head(tensor).squeeze(-1)).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    train, val = load_npz(args.train), load_npz(args.val)
    print(f"train={len(train['sample_id'])} val={len(val['sample_id'])} "
          f"dim={train['X'].shape[1]} train_pos_rate={train['hard'].mean():.3f} "
          f"val_pos_rate={val['hard'].mean():.3f}", flush=True)

    head, mean, std, best_auc, best_epoch = fit_logistic(
        train["X"], train["soft"], val["X"], val["hard"],
        epochs=args.epochs, lr=args.lr)
    print(f"probe: best val AUC={best_auc:.4f} at epoch {best_epoch}", flush=True)

    report = {"probe": "logistic regression on last-layer hidden state",
              "n_train": len(train["sample_id"]), "n_val": len(val["sample_id"]),
              "dim": int(train["X"].shape[1]),
              "best_val_auc": best_auc, "best_epoch": best_epoch}
    for name, split in (("train", train), ("val", val)):
        scores = predict(head, mean, std, split["X"])
        acc = float(((scores >= 0.5) == (split["hard"] > 0.5)).mean())
        report[f"{name}_auc"] = auc(scores, split["hard"])
        report[f"{name}_acc_at_0.5"] = acc
        report[f"{name}_p_nt"] = describe(scores)
        (args.out_dir / f"probe_{name}_p_nt.json").write_text(
            json.dumps({"p_nt": dict(zip(split["sample_id"].tolist(), scores.tolist()))}),
            encoding="utf-8")
        print(f"{name}: auc={report[f'{name}_auc']:.4f} acc@0.5={acc:.3f} "
              f"p_nt median={report[f'{name}_p_nt']['median']:.3f}", flush=True)

    if args.test is not None:
        test = load_npz(args.test)
        scores = predict(head, mean, std, test["X"])
        report["test_p_nt"] = describe(scores)
        (args.out_dir / "probe_test_p_nt.json").write_text(
            json.dumps({"p_nt": dict(zip(test["sample_id"].tolist(), scores.tolist()))}),
            encoding="utf-8")
        print(f"test: n={len(test['sample_id'])} "
              f"p_nt median={report['test_p_nt']['median']:.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "probe_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"PROBE_READY {args.out_dir / 'probe_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
