#!/usr/bin/env python3
"""Cross-validated probe AUC inside the training distribution (numpy only).

Motivation: the probe and the joint SFT both fit the training set moderately
(train AUC ~0.67-0.69) but scored *below* chance on the GSM8K/MATH validation set
(AUC 0.47-0.53). An AUC under 0.5 means the score is anti-correlated with the
labels, which is what a systematically different problem distribution produces:
DeepScaleR is 59.5% "route fast", the validation set is 82.2%.

This script answers one question before any more training: is the label learnable
*within* the training distribution? Stratified k-fold on the cached features, no
model, no GPU.

Usage:
    python scripts/probe_cv.py --features outputs/hidden_means/train.npz --folds 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_probe import auc, fit_logistic, predict  # noqa: E402


def stratified_folds(hard, folds, seed=0):
    rng = np.random.default_rng(seed)
    idx_pos = np.where(hard > 0.5)[0]
    idx_neg = np.where(hard <= 0.5)[0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)
    assignment = np.zeros(len(hard), dtype=int)
    for indices in (idx_pos, idx_neg):
        for position, index in enumerate(indices):
            assignment[index] = position % folds
    return assignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=ROOT / "outputs/hidden_means/train.npz")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=400)
    parser.add_argument("--out", type=Path, default=ROOT / "results/probe_cv.json")
    args = parser.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X, soft, hard = data["X"].astype(np.float32), data["soft"], data["hard"]
    print(f"n={len(hard)} dim={X.shape[1]} positive_rate={hard.mean():.3f} "
          f"folds={args.folds}", flush=True)

    assignment = stratified_folds(hard, args.folds)
    scores = []
    for fold in range(args.folds):
        test_mask = assignment == fold
        train_mask = ~test_mask
        model = fit_logistic(X[train_mask], soft[train_mask], X[test_mask], hard[test_mask],
                             iters=args.iters)
        fold_auc = auc(predict(model, X[test_mask]), hard[test_mask])
        scores.append(fold_auc)
        print(f"  fold {fold + 1}: n_train={train_mask.sum()} n_test={test_mask.sum()} "
              f"auc={fold_auc:.4f}", flush=True)

    scores = np.array(scores)
    report = {"features": str(args.features), "folds": args.folds,
              "fold_aucs": scores.tolist(),
              "mean_auc": float(scores.mean()), "std_auc": float(scores.std()),
              "positive_rate": float(hard.mean())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nCV AUC = {scores.mean():.4f} ± {scores.std():.4f}")
    print("VERDICT:", "learnable in-distribution" if scores.mean() > 0.62
          else "no usable signal even in-distribution")
    print(f"WROTE {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
