#!/usr/bin/env python3
"""Route E / P2: token-aware routing on cached frozen DeepSeek P1 representations.

Only train the light CPU regression heads. NO GPU prefill, decoding, model
updates, feature extraction, or sealed-test access is performed by this script.

  python scripts/route_e_token_aware.py self-test
  python scripts/route_e_token_aware.py check
  python scripts/route_e_token_aware.py run --smoke
  python scripts/route_e_token_aware.py run

The dense actual-token curves are exploratory replay, NOT a deployable
budget selector. Separately reported prospective policies choose only from
pre-generation predictions; their realized costs are measured afterwards.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

import route_e_frozen as base
import route_e_prompt_ablation as prefix
from route_e_equal_budget import (FT_ANCHORS, DEFAULT_BUDGETS, class_anchor_curve,
                                  dense_curve, linear_bracket, load_scores,
                                  random_at_budget, verify_classification_50pct)
from train_modernbert_delta import endpoints

ROOT = base.ROOT
DEFAULT_L2 = 0.02
DEFAULT_LAMBDA = 0.05
LAMBDA_GRID = (0., 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0)
MAX_OUTPUT_TOKENS = 16384
P1_SCORES = ROOT / "outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json"
FT_SCORES = ROOT / "outputs/external_modernbert/classification/val_predictions.jsonl"


def paths(smoke):
    suffix = "smoke" if smoke else "full"
    return (ROOT / "outputs/route_e/token_aware" / suffix,
            ROOT / "results/route_e/token_aware" / suffix)


def read_protocol_and_cache(smoke):
    """Fail closed on original strict labels, feature schema, ID order, and hashes."""
    train, val, overlaps = base.load_protocol()
    if smoke:
        train, val = base.choose_smoke_rows(train, val)
    cache = prefix.cache_dir("assistant", smoke)
    Xtr, tm = prefix.read_cache(cache / "train.npz", train, base.TRAIN_SHA256,
                                "assistant", smoke)
    Xva, vm = prefix.read_cache(cache / "val.npz", val, base.VAL_SHA256,
                                "assistant", smoke)
    for key in ("schema", "encoder", "model", "prompt_mode",
                "chat_template_sha256", "pooling", "layer", "dtype",
                "max_length", "feature_dim"):
        if tm[key] != vm[key]:
            raise ValueError(f"Train/validation feature metadata differ: {key}")
    if tm["layer"] != "last_after_final_norm" or tm["pooling"] != "last_valid_prompt_token":
        raise ValueError("Unexpected P1 layer/pooling; P2 must use Frozen DeepSeek P1")
    if Xtr.shape != (len(train), 1536) or Xva.shape != (len(val), 1536):
        raise ValueError("Expected last-layer DeepSeek representations of width 1536")
    return train, val, Xtr, Xva, tm, vm, overlaps


def target_matrix(rows):
    """Train labels ONLY. Column 0: empirical delta p; 1/2: log1p output lengths."""
    return np.stack((
        np.array([r["delta"] for r in rows], dtype=np.float64),
        np.log1p(np.array([r["t_t"] for r in rows], dtype=np.float64)),
        np.log1p(np.array([r["t_n"] for r in rows], dtype=np.float64)),
    ), axis=1)


def ridge_fit(X, y, l2):
    """One shared *standardized design* and three independently supervised heads.

    Minimize mean squared residual + l2 * ||weight||_2^2, with no bias penalty.
    The same matrix is solved for three targets, but quality and length labels
    are never mixed into one target or used as inference-time inputs.
    """
    if not np.isfinite(l2) or l2 <= 0:
        raise ValueError("Use strictly positive ridge regularization")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if X.ndim != 2 or y.shape != (len(X), 3):
        raise ValueError("Expected X: (N,D), y: (N,3)")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite regression data")
    mean = X.mean(axis=0)
    scale = np.maximum(X.std(axis=0), 1e-4)
    Z = (X - mean) / scale
    target_mean = y.mean(axis=0)
    yc = y - target_mean
    n, d = Z.shape
    gram = (Z.T @ Z) / n
    gram.flat[::d+1] += l2
    rhs = (Z.T @ yc) / n
    weight = np.linalg.solve(gram, rhs)
    intercept = target_mean - ((Z.mean(axis=0)) @ weight)
    fitted_log = Z @ weight + intercept
    # Train-only Duan-style retransformation corrects Jensen bias for E[t].
    # It does not guarantee calibration on the differently sampled validation set.
    residual = y[:, 1:3] - fitted_log[:, 1:3]
    smear = np.exp(np.clip(residual, -12., 12.)).mean(axis=0)
    model = {"mean": mean, "scale": scale, "weight": weight,
             "intercept": intercept, "smear": smear, "l2": float(l2)}
    if not all(np.isfinite(v).all() for v in model.values()
               if isinstance(v, np.ndarray)):
        raise ValueError("Nonfinite trained head")
    return model


def predict_heads(model, X):
    """Prediction reads only P1 hidden states and fitted training parameters."""
    X = np.asarray(X, dtype=np.float64)
    Z = (X - model["mean"]) / model["scale"]
    raw = Z @ model["weight"] + model["intercept"]
    delta = np.clip(raw[:, 0], -1., 1.)
    # Both output lengths are generated BEFORE seeing actual mode outcomes.
    log_t = np.clip(raw[:, 1:3], 0., np.log1p(MAX_OUTPUT_TOKENS))
    length = np.clip(np.exp(log_t) * model["smear"][None, :] - 1.,
                     0., MAX_OUTPUT_TOKENS)
    result = {"delta": delta, "t_t": length[:, 0], "t_n": length[:, 1]}
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite prediction")
    return result


def scores_for(pred, lam, constant_delta=None):
    """Higher score means Thinking is preferred; lower => NoThinking priority.

    This score cannot access ground-truth validation lengths or success rates.
    """
    cost_delta = (pred["t_t"] - pred["t_n"] if constant_delta is None
                  else np.full(len(pred["delta"]), constant_delta))
    benefit = pred["delta"] - lam * cost_delta / 1000.
    return np.asarray(benefit, dtype=np.float64)


def nt_priority(scores, rows):
    """Existing dense_curve convention: higher score => route NoThinking first."""
    return {r["sample_id"]: float(-v) for r, v in zip(rows, scores)}


def nt_prob_at_bracket(rows, scores, bracket):
    """Per-question NoThinking probability for an existing mixed top-k policy."""
    n = len(rows)
    ranking = sorted(range(n), key=lambda i: (-scores[rows[i]["sample_id"]],
                                                 rows[i]["sample_id"]))
    k_left = round(float(bracket["policy_left_fraction"]) * n)
    k_right = round(float(bracket["policy_right_fraction"]) * n)
    w = float(bracket["mix_right_probability"])
    pn = np.zeros(n, dtype=np.float64)
    pn[ranking[:k_left]] += 1. - w
    pn[ranking[:k_right]] += w
    if pn.min() < -1e-10 or pn.max() > 1+1e-10:
        raise ValueError("Invalid mixed policy probabilities")
    return np.clip(pn, 0., 1.)


def measured_policy(rows, pn):
    """Uses true strict per-mode labels ONLY after routing probabilities are fixed."""
    p_t = np.array([r["p_t"] for r in rows])
    p_n = np.array([r["p_n"] for r in rows])
    t_t = np.array([r["t_t"] for r in rows])
    t_n = np.array([r["t_n"] for r in rows])
    return (p_t*(1-pn) + p_n*pn, t_t*(1-pn) + t_n*pn)


def group_metrics(rows, pn):
    successes, tokens = measured_policy(rows, pn)
    groups = {}
    ds = np.array([r["dataset"] for r in rows])
    for group in ["overall"] + sorted(set(ds)):
        mask = np.ones(len(rows), dtype=bool) if group == "overall" else ds == group
        groups[group] = {
            "n": int(mask.sum()), "accuracy": float(successes[mask].mean()),
            "actual_tokens": float(tokens[mask].mean()),
            "nothinking_fraction": float(pn[mask].mean()),
        }
    return groups


def paired_bootstrap(rows, a, b, repeats, seed):
    """Conditional paired CI: frozen policies and lambda; no re-selection in resamples."""
    if repeats < 1:
        return None
    va, _ = measured_policy(rows, a)
    vb, _ = measured_policy(rows, b)
    delta = va - vb
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for j in range(repeats):
        draw_ids = rng.integers(0, len(rows), size=len(rows))
        draws[j] = 100. * delta[draw_ids].mean()
    return {"difference_pp": float(100.*delta.mean()),
            "ci_95_pp": [float(q) for q in np.percentile(draws, (2.5, 97.5))],
            "repetitions": repeats,
            "warning": "Paired per-question CI conditional on the fitted heads, "
                       "selected score ordering and mixed policies; does not "
                       "cover train/validation selection uncertainty."}


def average_rank(values):
    """Average rank for ties, without scipy dependency."""
    v = np.asarray(values, dtype=np.float64)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    i = 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[order[j]] == v[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.
        i = j
    return ranks


def spearman(a, b):
    x, y = average_rank(a), average_rank(b)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def regression_metrics(pred, truth):
    p, y = np.asarray(pred), np.asarray(truth)
    diff = p - y
    return {"mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "spearman": spearman(p, y),
            "pred_mean": float(p.mean()), "actual_mean": float(y.mean()),
            "bias_pred_minus_actual": float(diff.mean())}


def cost_diagnostics(rows, pred):
    actual_t = np.array([r["t_t"] for r in rows])
    actual_n = np.array([r["t_n"] for r in rows])
    return {
        "thinking_tokens": regression_metrics(pred["t_t"], actual_t),
        "nothinking_tokens": regression_metrics(pred["t_n"], actual_n),
        "extra_thinking_tokens": regression_metrics(
            pred["t_t"] - pred["t_n"], actual_t - actual_n),
        "delta_success": regression_metrics(pred["delta"],
            np.array([r["delta"] for r in rows])),
        "clipped_at_max_output": {
            "thinking": int((pred["t_t"] >= MAX_OUTPUT_TOKENS).sum()),
            "nothinking": int((pred["t_n"] >= MAX_OUTPUT_TOKENS).sum()),
        },
    }


def prospective_budget_policy(rows, pred, score, budget):
    """Calibration based on predicted lengths ONLY, not actual validation costs.

    Batch-level (transductive) score-threshold calibration on UNLABELED val
    inputs is illustrative: for online deployment this requires an independent
    representative unlabeled calibration workload. Observed validation lengths
    are used after policies are fixed, solely to assess budget error.
    """
    n = len(score)
    # score is THINKING preference: select the LOWEST score for NoThinking first.
    ranked = sorted(range(n), key=lambda i: (float(score[i]), rows[i]["sample_id"]))
    base_cost = float(np.mean(pred["t_t"]))
    predicted = []
    cost = base_cost
    for k in range(n+1):
        if k:
            i = ranked[k-1]
            cost += (pred["t_n"][i] - pred["t_t"][i]) / n
        predicted.append({"fraction": k/n, "tokens": cost,
                          "accuracy": 0.0})  # dummy; interpolation uses predicted costs only
    low, high = min(r["tokens"] for r in predicted), max(r["tokens"] for r in predicted)
    if not low-1e-8 <= budget <= high+1e-8:
        return {"feasible": False, "target_budget": budget,
                "predicted_reachable_range": [float(low), float(high)],
                "reason": "Target is unreachable under the length head; "
                          "no actual-length-based calibration is permitted"}
    budget = min(high, max(low, budget))
    bracket = linear_bracket(predicted, budget)
    ids = [r["sample_id"] for r in rows]
    # Use ID-based scores to preserve the same tie ordering as dense replay.
    pn = nt_prob_at_bracket(rows, {sid: float(-v) for sid, v in zip(ids, score)}, bracket)
    groups = group_metrics(rows, pn)
    return {"feasible": True, "target_budget": float(budget),
            "predicted_budget": float(
                np.mean((1-pn)*pred["t_t"] + pn*pred["t_n"])),
            "actual_budget": groups["overall"]["actual_tokens"],
            "actual_accuracy": groups["overall"]["accuracy"],
            "budget_error_actual_minus_predicted": (
                groups["overall"]["actual_tokens"] - budget),
            "expected_nothinking_fraction": groups["overall"]["nothinking_fraction"],
            "groups": groups, "mix": bracket,
            "policy_sees_actual_lengths": False}


def evaluate_dense(rows, score_maps, budgets, bootstrap_reps, seed):
    curves = {k: dense_curve(rows, v) for k, v in score_maps.items()}
    results, per_question = [], {}
    for budget in budgets:
        rec = {"budget": float(budget), "methods": {}}
        for method, curve in curves.items():
            try:
                q = linear_bracket(curve, budget)
            except ValueError as exc:
                rec["methods"][method] = {"available": False, "reason": str(exc)}
                continue
            pn = nt_prob_at_bracket(rows, score_maps[method], q)
            groups = group_metrics(rows, pn)
            if abs(groups["overall"]["actual_tokens"] - budget) > 1e-4:
                raise AssertionError(f"Mismatch in actual-token dense interpolation: {method}")
            rec["methods"][method] = {"available": True, "mix": q, "groups": groups}
            per_question[(method, budget)] = pn
        random = random_at_budget(rows, budget)
        pnrand = np.full(len(rows), random["nt_probability"])
        rec["methods"]["Global_Random"] = {
            "available": True, "groups": group_metrics(rows, pnrand),
            "random_nothinking_probability": random["nt_probability"],
        }
        per_question[("Global_Random", budget)] = pnrand
        for name in ("Delta_only", "Delta_constant_cost",
                     "Delta_predicted_cost", "P1_Frozen", "FT_Classification"):
            if rec["methods"].get(name, {}).get("available"):
                acc = rec["methods"][name]["groups"]["overall"]["accuracy"]
                rec["methods"][name]["uplift_vs_random_pp"] = float(
                    100*(acc-random["accuracy"]))
        target = "Delta_predicted_cost"
        if rec["methods"].get(target, {}).get("available"):
            rec["paired_bootstrap"] = {}
            for other in ("Delta_only", "P1_Frozen", "FT_Classification",
                          "Global_Random"):
                if rec["methods"].get(other, {}).get("available"):
                    rec["paired_bootstrap"]["D_minus_"+other] = paired_bootstrap(
                        rows, per_question[(target,budget)],
                        per_question[(other,budget)], bootstrap_reps, seed+int(budget))
        results.append(rec)
    return results


def get_baseline_scores(val, allow_missing=True):
    ids = [r["sample_id"] for r in val]
    sources, scores = {}, {}
    if P1_SCORES.is_file():
        scores["P1_Frozen"] = load_scores(P1_SCORES, ids)
        curve = dense_curve(val, scores["P1_Frozen"])
        half = curve[225]
        if abs(half["accuracy"] - 344/450) > 1e-8 or abs(
                half["tokens"] - 1427.05) > 2.:
            raise RuntimeError("P1 strict 50% NT reproduction mismatch")
        sources["P1_Frozen"] = {"source": str(P1_SCORES),
                                "reproduced_50pct": half}
    elif not allow_missing:
        raise FileNotFoundError(P1_SCORES)
    else:
        print("WARNING P1 scores unavailable; P1 comparison omitted", flush=True)

    if FT_SCORES.is_file():
        scores["FT_Classification"] = load_scores(
            FT_SCORES, ids, direction="higher_nt", field="p_nothinking_pred")
        ft_curve = dense_curve(val, scores["FT_Classification"])
        reproduction = verify_classification_50pct(ft_curve)
        sources["FT_Classification"] = {"source": str(FT_SCORES),
                                          "field": "p_nothinking_pred",
                                          "reproduction": reproduction}
    else:
        print("WARNING FT dense score file unavailable; FT classification "
              "will be shown as historical approximate anchors only", flush=True)
    return scores, sources


def val_stats(rows):
    return {"n": len(rows),
            "mean_t_t": float(np.mean([r["t_t"] for r in rows])),
            "mean_t_n": float(np.mean([r["t_n"] for r in rows])),
            "mean_delta_p": float(np.mean([r["delta"] for r in rows]))}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n",
                    encoding="utf-8")


def save_csv(folder, budget_results, lam_rows):
    with (folder/"budget_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["budget", "method", "dataset", "accuracy_pct",
                         "actual_tokens", "nt_fraction", "uplift_vs_random_pp"])
        for b in budget_results:
            for method, entry in b["methods"].items():
                if not entry["available"]:
                    continue
                for ds, group in entry["groups"].items():
                    writer.writerow([b["budget"], method, ds,
                                     100*group["accuracy"], group["actual_tokens"],
                                     group["nothinking_fraction"],
                                     entry.get("uplift_vs_random_pp") if ds=="overall" else ""])
    with (folder/"lambda_grid.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "lambda", "predicted_tokens", "actual_tokens", "accuracy",
            "nt_fraction", "pred_minus_actual_tokens"))
        writer.writeheader()
        writer.writerows(lam_rows)


def try_plot(folder, budgets):
    """Optional figure; absence of matplotlib never invalidates numeric report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("PLOT_SKIPPED matplotlib unavailable", flush=True)
        return
    order = ("Global_Random", "FT_Classification", "P1_Frozen",
             "Delta_only", "Delta_predicted_cost")
    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    for method in order:
        points = [(b["budget"], 100*b["methods"][method]["groups"]["overall"]["accuracy"])
                  for b in budgets if b["methods"].get(method, {}).get("available")]
        if points:
            ax.plot(*zip(*points), marker="o", label=method.replace("_", " "))
    ax.set(xlabel="Mean generated tokens per question (observed, offline)",
           ylabel="Strict validation accuracy (%)",
           title="P2 — Frozen P1 Quality vs Token-aware Routing")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("svg", "png"):
        fig.savefig(folder/f"accuracy_token_curve.{ext}", dpi=170)
    plt.close(fig)


def run(args):
    start = time.perf_counter()
    train, val, Xtr, Xva, tm, vm, overlaps = read_protocol_and_cache(args.smoke)
    out, report_dir = paths(args.smoke)
    if not args.overwrite and (out/"ridge_heads.npz").exists():
        raise FileExistsError("P2 output exists: use --overwrite to replace P2 files only")
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    print(f"P2_CACHE_OK train={Xtr.shape} val={Xva.shape} "
          f"train_sha={base.TRAIN_SHA256} val_sha={base.VAL_SHA256}", flush=True)
    t0 = time.perf_counter()
    trained = ridge_fit(Xtr, target_matrix(train), args.l2)
    seconds_train = time.perf_counter()-t0
    t0 = time.perf_counter()
    preds_train = predict_heads(trained, Xtr)
    preds = predict_heads(trained, Xva)
    head_ms = (time.perf_counter()-t0)*1000/(len(train)+len(val))
    mean_cost = float(np.mean([r["t_t"]-r["t_n"] for r in train]))
    quality = scores_for(preds, 0.)
    const = scores_for(preds, args.cost_lambda, mean_cost)
    aware = scores_for(preds, args.cost_lambda)
    ids = [r["sample_id"] for r in val]
    maps = {
        "Delta_only": nt_priority(quality, val),
        "Delta_constant_cost": nt_priority(const, val),
        "Delta_predicted_cost": nt_priority(aware, val),
    }
    # Constant cost shifts every score by the same value: NO ordering benefit.
    a = sorted(ids, key=lambda sid: (-maps["Delta_only"][sid], sid))
    b = sorted(ids, key=lambda sid: (-maps["Delta_constant_cost"][sid], sid))
    if a != b:
        raise AssertionError("Constant cost changed the question ranking")
    out_scores = {
        sid: {"delta_p_pred": float(preds["delta"][i]),
              "thinking_tokens_pred": float(preds["t_t"][i]),
              "nothinking_tokens_pred": float(preds["t_n"][i]),
              "delta_t_pred": float(preds["t_t"][i]-preds["t_n"][i]),
              "score_delta_only": float(quality[i]),
              "score_constant_cost": float(const[i]),
              "score_predicted_cost": float(aware[i])}
        for i, sid in enumerate(ids)
    }
    np.savez_compressed(out/"ridge_heads.npz", **trained)
    save_json(out/"validation_predictions.json", {
        "strict_val_sha256": base.VAL_SHA256, "cost_lambda": args.cost_lambda,
        "note": "Model features only; no observed success or token fields included",
        "predictions": out_scores,
    })
    report = {
        "method": "route_e_p2_frozen_deepseek_p1_token_aware",
        "smoke": args.smoke, "strict_train_sha256": base.TRAIN_SHA256,
        "strict_val_sha256": base.VAL_SHA256, "overlap_normalized_text": overlaps,
        "encoder_frozen": True, "generation_repeated": False,
        "feature_source": str(prefix.cache_dir("assistant", args.smoke)),
        "train_feature_metadata": tm, "val_feature_metadata": vm,
        "train_distribution": val_stats(train), "validation_distribution": val_stats(val),
        "head": {"type": "ridge_multi_output_shared_design",
                 "target_columns": ["p_thinking_minus_p_nothinking_strict",
                                    "log1p(thinking_tokens)",
                                    "log1p(nothinking_tokens)"],
                 "ridge_penalty": args.l2,
                 "smearing_train_only": [float(v) for v in trained["smear"]],
                 "fixed_lambda": args.cost_lambda,
                 "constant_cost_train_mean_delta_tokens": mean_cost,
                 "train_seconds_cpu": seconds_train,
                 "head_ms_per_question_cpu_batch": head_ms,
                 "shared_features_three_heads": True},
        "diagnostics": {"train": cost_diagnostics(train, preds_train),
                        "validation": cost_diagnostics(val, preds)},
        "cost_prediction_model_input": "pre-generation P1 X only",
        "router_overhead_included_in_generated_tokens": False,
        "dense_replay_is_deployable": False,
        "notes": [
            "B/C/D use EXACT same fitted quality head; C differs only by a constant.",
            "Validation actual outcome and token fields never enter score construction.",
            "Validation distribution differs from success-bucket-stratified train.",
            "Do not choose lambda, layer, regularization or classification threshold "
            "on the sealed test set.",
        ],
    }

    if not args.smoke:
        baselines, sources = get_baseline_scores(val)
        maps.update(baselines)
        report["baseline_sources"] = sources
        budgets = [float(v) for v in args.budgets]
        for b in budgets:
            if not math.isfinite(b) or b <= 0:
                raise ValueError("Budgets must be finite, positive values")
        dense = evaluate_dense(val, maps, budgets, args.bootstrap_reps, args.seed)
        report["dense_validation"] = {
            "type": "451 actual-token policy points per router, then "
                    "observed-token bracket interpolation",
            "warning": "Exploratory, cost-matched replay: validation true token "
                       "counts choose interpolation brackets; not deployable",
            "budgets": dense,
        }
        if "FT_Classification" not in maps:
            reference = class_anchor_curve(val)
            report["ft_approximate_only"] = [
                {"budget": b, "accuracy": linear_bracket(reference, b)["accuracy"]}
                for b in budgets]
        report["prospective_predicted_budget"] = {
            "protocol": "Predicted-only threshold calibration using unlabeled "
                        "validation P1 features (transductive batch example)",
            "not_for_test_tuning": True,
            "policies": [prospective_budget_policy(val, preds, aware, b)
                         for b in budgets],
        }
        lam_rows = []
        for lam in LAMBDA_GRID:
            score = scores_for(preds, lam)
            pn = np.asarray(score <= 0., dtype=np.float64)
            groups = group_metrics(val, pn)
            forecast = float(np.mean((1-pn)*preds["t_t"]+pn*preds["t_n"]))
            lam_rows.append({
                "lambda": lam, "predicted_tokens": forecast,
                "actual_tokens": groups["overall"]["actual_tokens"],
                "accuracy": groups["overall"]["accuracy"],
                "nt_fraction": groups["overall"]["nothinking_fraction"],
                "pred_minus_actual_tokens": (
                    forecast-groups["overall"]["actual_tokens"]),
            })
        report["lambda_sweep"] = {
            "rule": "Thinking iff predicted_delta_p - lambda * predicted_delta_t/1000 > 0",
            "selection": "predeclared grid (diagnostic only, not selected by val accuracy)",
            "rows": lam_rows}
        save_csv(report_dir, dense, lam_rows)
        if not args.no_plot:
            try_plot(report_dir, dense)
        for b in dense:
            chunks = []
            for method in ("Global_Random", "Delta_only", "Delta_predicted_cost",
                           "P1_Frozen", "FT_Classification"):
                entry = b["methods"].get(method, {})
                if entry.get("available"):
                    chunks.append(f"{method}={entry['groups']['overall']['accuracy']:.2%}")
            print(f"P2_BUDGET {b['budget']:.0f} | "+" | ".join(chunks), flush=True)
    else:
        report["smoke_note"] = ("32 strict train / 24 validation; no full "
                                "450-question budget comparisons or model selection")

    save_json(report_dir/"validation.json", report)
    save_json(report_dir/"config.json", {
        "l2": args.l2, "cost_lambda": args.cost_lambda, "seed": args.seed,
        "budgets": args.budgets, "smoke": args.smoke,
        "bootstrap_reps": args.bootstrap_reps,
        "train_sha256": base.TRAIN_SHA256, "val_sha256": base.VAL_SHA256})
    print(f"P2_DONE smoke={int(args.smoke)} train_seconds={seconds_train:.2f} "
          f"total_seconds={time.perf_counter()-start:.2f} "
          f"report={report_dir/'validation.json'}", flush=True)


def check(args):
    train, val, Xtr, Xva, tm, vm, _ = read_protocol_and_cache(args.smoke)
    print(f"P2_DATA_CACHE_PASS train={Xtr.shape} val={Xva.shape} "
          f"mode={tm['prompt_mode']} layer={tm['layer']} "
          f"train_mean_tokens={val_stats(train)['mean_t_t']:.1f}/"
          f"{val_stats(train)['mean_t_n']:.1f} "
          f"val_mean_tokens={val_stats(val)['mean_t_t']:.1f}/"
          f"{val_stats(val)['mean_t_n']:.1f}", flush=True)


def self_test():
    """No models, original labels, GPU, or downloads required."""
    import tempfile
    from route_e_prompt_ablation import read_cache
    with tempfile.TemporaryDirectory() as d:
        path = Path(d)/"test.npz"
        rows = [{"sample_id": "a"}, {"sample_id": "b"}]
        meta = {"schema": "route_e_prefix_features_v1", "prompt_mode": "assistant",
                "smoke": True, "labels_sha256": "synthetic"}
        np.savez_compressed(path, X=np.ones((2, 1536), dtype=np.float16),
                            sample_id=np.array(["a","b"]),
                            metadata_json=np.array(json.dumps(meta)))
        X, _ = read_cache(path, rows, "synthetic", "assistant", True)
        assert X.shape == (2,1536)
        try:
            read_cache(path, rows[::-1], "synthetic", "assistant", True)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-order cache accepted")
    rng = np.random.default_rng(123)
    Xtr = rng.normal(size=(30, 9))
    Ytr = np.stack((np.tanh(Xtr[:, 0]) / 2.,
                     np.log1p(np.exp(3+Xtr[:,1])),
                     np.log1p(np.exp(2+Xtr[:,2]))), axis=1)
    model = ridge_fit(Xtr, Ytr, 0.05)
    pred = predict_heads(model, Xtr[:6])
    assert all(np.isfinite(v).all() for v in pred.values())
    assert np.all(pred["t_t"] >= 0) and np.all(pred["t_n"] <= MAX_OUTPUT_TOKENS)
    s1 = scores_for(pred, 0.)
    s2 = scores_for(pred, .05, constant_delta=1000.)
    assert np.array_equal(np.argsort(s1), np.argsort(s2))
    # More efficient T problem should be prioritized when delta p is equal.
    artificial = {"delta": np.array([.1,.1]), "t_t": np.array([400.,4100.]),
                  "t_n": np.array([100.,100.])}
    assert scores_for(artificial, .02)[0] > scores_for(artificial, .02)[1]
    rows = [
        {"sample_id": "a", "dataset": "gsm8k", "p_t": 1., "p_n": 0.,
         "t_t": 100., "t_n": 20.},
        {"sample_id": "b", "dataset": "math", "p_t": 0., "p_n": 1.,
         "t_t": 100., "t_n": 20.},
    ]
    ranking = {"a":0., "b":1.}
    curve = dense_curve(rows, ranking)
    bracket = linear_bracket(curve, 80.)
    pn = nt_prob_at_bracket(rows, ranking, bracket)
    assert np.isclose(group_metrics(rows, pn)["overall"]["actual_tokens"],80.)
    assert np.isclose(group_metrics(rows, pn)["overall"]["accuracy"],.75)
    # Prospective calibration is solely a function of predicted costs/scores;
    # perturbing ground-truth lengths must NOT change the selected probabilities.
    ex = {"delta": np.array([.3,-.3]), "t_t": np.array([100.,100.]),
          "t_n": np.array([20.,20.])}
    score = np.array([.3,-.3])
    p1 = prospective_budget_policy(rows, ex, score, 60.)
    changed = [dict(r, t_t=9e5, t_n=1e4) for r in rows]
    p2 = prospective_budget_policy(changed, ex, score, 60.)
    assert p1["feasible"] and p2["feasible"]
    assert p1["mix"] == p2["mix"]
    assert np.isclose(p1["predicted_budget"],60.)
    # The lower-Thinking-benefit problem b must be routed to NoThinking.
    assert np.isclose(p1["actual_accuracy"], 1.0)
    assert not np.isclose(p1["actual_budget"],p2["actual_budget"])
    assert not prospective_budget_policy(rows, ex, score, 500.)["feasible"]
    assert spearman([1,2,3],[3,2,1]) == -1.
    print("P2_SELFTEST_PASS (cache alignment, ridge, score, no cost leakage, "
          "budget mixture, metrics)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    cmd.add_parser("self-test")
    for name in ("check", "run"):
        p = cmd.add_parser(name)
        p.add_argument("--smoke", action="store_true")
        if name == "run":
            p.add_argument("--l2", type=float, default=DEFAULT_L2)
            p.add_argument("--cost-lambda", type=float, default=DEFAULT_LAMBDA)
            p.add_argument("--budgets", nargs="+", type=float,
                           default=list(DEFAULT_BUDGETS))
            p.add_argument("--bootstrap-reps", type=int, default=1000)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--overwrite", action="store_true")
            p.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    elif args.command == "check":
        check(args)
    else:
        if not math.isfinite(args.cost_lambda) or args.cost_lambda < 0 or (
            not math.isfinite(args.l2) or args.l2 <= 0 or args.bootstrap_reps < 0):
            parser.error("Require lambda>=0, l2>0 and bootstrap-reps>=0")
        run(args)


if __name__ == "__main__":
    main()
