#!/usr/bin/env python3
"""Route E / Experiment F: direct and margin-aware incremental token-cost learning.

Three frozen-P1, CPU-head experiments on the ORIGINAL strict 3000/450 split:
 A0: identical historical P2 Ridge separate log-cost heads (no retraining).
 A1: Ridge regression predicting raw (t_thinking - t_nothinking)/scale.
 A2: P2 separate log-cost Ridge heads + differentiable token-space delta MSE.

A2 beta selection is TRAIN-ONLY K-fold OOF Spearman; validation/test are never
used to tune beta. All routing ranks use PRE-GENERATION predictions only.
Observed validation lengths appear only in DESCRIPTIVE cost-matched replay.

Usage (from the AdaptThink repository root):
  python scripts/route_e_cost_difference.py self-test
  python scripts/route_e_cost_difference.py run
  python scripts/route_e_cost_difference.py run --no-plot --bootstrap-reps 0

Dependencies: numpy, torch and existing AdaptThink scripts/cached P1 features.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'results/route_e/cost_difference/experiment_f'
DEFAULT_BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)
DEFAULT_BETAS = (0., 0.1, 1.)
MAX_TOKENS = 16384.


def finite_array(a, name):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError(f'{name}: nonfinite values')
    return a


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def delta_target(rows, scale):
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError('Positive finite scale required')
    return np.asarray([(r['t_t'] - r['t_n'])/scale for r in rows], dtype=np.float64)


def direct_ridge_fit(X, rows, l2, scale):
    """Same feature normalization/regularization definition as historical P2."""
    if l2 <= 0:
        raise ValueError('l2 must be strictly positive')
    X = finite_array(X, 'X')
    y = delta_target(rows, scale)
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-4)
    Z = (X - mean)/std
    n, d = Z.shape
    gram = (Z.T @ Z)/n
    gram.flat[::d+1] += l2
    weights = np.linalg.solve(gram, (Z.T @ (y - y.mean()))/n)
    bias = float(y.mean() - (Z.mean(axis=0) @ weights))
    return {'mean': mean, 'scale': std, 'weight': weights,
            'intercept': bias, 'l2': l2, 'delta_scale': scale}


def direct_ridge_predict(model, X):
    X = finite_array(X, 'X')
    p = (((X-model['mean'])/model['scale']) @ model['weight'] + model['intercept']) * model['delta_scale']
    return np.clip(p, -MAX_TOKENS, MAX_TOKENS)


def margin_predict(model, X):
    """Retransform the same way as P2; smearing is fixed from fold-train A0."""
    X = finite_array(X, 'X')
    Z = (X-model['mean'])/model['scale']
    logits = Z @ model['weight'] + model['intercept']
    logits = np.clip(logits, 0., np.log1p(MAX_TOKENS))
    token_pred = np.clip(np.exp(logits)*model['smear'][None, :] - 1., 0., MAX_TOKENS)
    return token_pred[:, 0]-token_pred[:, 1], token_pred


def fit_margin(X, rows, l2, delta_scale, beta, max_iter, *, base_fit=None):
    """Train-only nonlinear margin loss; beta=0 EXACTLY returns historical P2.

    Loss uses log-token MSE + ridge + beta * MSE(pred_delta/scale, true_delta/scale).
    The predicted tokens use P2's log clipping and fixed train-derived smearing.
    P2's 3-column closed-form fit (delta_p, log_t, log_n) initializes log cost
    heads; quality head is never trained/used by A2.
    """
    import torch
    if beta < 0 or not math.isfinite(beta):
        raise ValueError('Invalid beta')
    if base_fit is None:
        import route_e_token_aware as p2
        base_fit = p2.ridge_fit(X, p2.target_matrix(rows), l2)
    model = {'mean': base_fit['mean'].copy(),
             'scale': base_fit['scale'].copy(),
             'weight': base_fit['weight'][:, 1:3].copy(),
             'intercept': base_fit['intercept'][1:3].copy(),
             'smear': base_fit['smear'].copy(),
             'l2': l2, 'beta': float(beta), 'delta_scale': delta_scale,
             'smearing_rule': 'fixed train-only P2 baseline correction'}
    if beta == 0:
        return model, {'optimized': False, 'reason': 'exact original ridge fit'}

    torch.set_num_threads(min(4, torch.get_num_threads()))
    dtype = torch.float64
    X = finite_array(X, 'features')
    Z = torch.tensor((X - model['mean'])/model['scale'], dtype=dtype)
    yt = torch.tensor([[np.log1p(r['t_t']), np.log1p(r['t_n'])] for r in rows], dtype=dtype)
    target_dt = torch.tensor(delta_target(rows, delta_scale), dtype=dtype)
    smear = torch.tensor(model['smear'], dtype=dtype)
    w = torch.nn.Parameter(torch.tensor(model['weight'].copy(), dtype=dtype))
    b = torch.nn.Parameter(torch.tensor(model['intercept'].copy(), dtype=dtype))
    optim = torch.optim.LBFGS([w,b], lr=1., max_iter=max_iter, history_size=25,
                               tolerance_grad=1e-10, tolerance_change=1e-12,
                               line_search_fn='strong_wolfe')
    state = {'closure_calls': 0}

    def closure():
        optim.zero_grad()
        pred_log = Z @ w + b
        # 1/2 on both data and weight regularizer matches P2's independent
        # mean-MSE + l2 ridge normal equations when beta=0.
        log_mse = (pred_log-yt).square().mean()
        reg = 0.5*l2*w.square().sum()
        log_clipped = torch.clamp(pred_log, 0., math.log1p(MAX_TOKENS))
        tokens = torch.clamp(torch.exp(log_clipped)*smear-1., 0., MAX_TOKENS)
        dt_scaled = (tokens[:,0]-tokens[:,1])/delta_scale
        diff_mse = (dt_scaled-target_dt).square().mean()
        objective = log_mse + reg + beta*diff_mse
        if not torch.isfinite(objective):
            raise FloatingPointError('Nonfinite A2 objective')
        objective.backward()
        state['closure_calls'] += 1
        return objective

    initial = float(closure().detach())
    optim.step(closure)
    final = float(closure().detach())
    new_w, new_b = w.detach().numpy().copy(), b.detach().numpy().copy()
    if not np.isfinite(new_w).all() or not np.isfinite(new_b).all():
        raise FloatingPointError('Nonfinite A2 parameters')
    model['weight'], model['intercept'] = new_w, new_b
    if final > initial + 1e-7:
        raise RuntimeError(f'Optimizer made loss worse: {initial} -> {final}')
    return model, {'optimized': True, 'objective_initial': initial,
                   'objective_final': final, 'closure_calls': state['closure_calls'],
                   'max_iter': max_iter}


def fold_assignments(rows, nfolds, seed):
    """Stable train-only folds stratified by dataset and train-only cost rank."""
    n = len(rows)
    if not 2 <= nfolds <= n:
        raise ValueError('Invalid CV fold count')
    rng = np.random.default_rng(seed)
    fold = np.empty(n, dtype=np.int64)
    ds = np.asarray([r['dataset'] for r in rows])
    cost = np.asarray([r['t_t'] - r['t_n'] for r in rows])
    for group in np.unique(ds):
        ind = np.flatnonzero(ds==group)
        # Split ranked train cost into roughly equally populated local strata.
        sorted_inds = ind[np.argsort(cost[ind], kind='mergesort')]
        for block in np.array_split(sorted_inds, min(5, len(sorted_inds))):
            block = rng.permutation(block)
            for j, sample in enumerate(block):
                fold[sample] = j % nfolds
    if any(np.sum(fold==k)==0 for k in range(nfolds)):
        raise RuntimeError('Empty CV fold')
    return fold


def select_beta_cv(X, rows, betas, folds, seed, l2, scale, max_iter):
    """ONLY TRAIN data and train labels may flow into this function."""
    import route_e_token_aware as p2
    X = finite_array(X, 'CV train X')
    assign = fold_assignments(rows, folds, seed)
    truth = scale*delta_target(rows, scale)
    oof = {beta: np.full(len(rows), np.nan) for beta in betas}
    per_fold=[]
    for k in range(folds):
        tr_idx = np.flatnonzero(assign != k)
        va_idx = np.flatnonzero(assign == k)
        tr_rows = [rows[i] for i in tr_idx]
        base = p2.ridge_fit(X[tr_idx], p2.target_matrix(tr_rows), l2)
