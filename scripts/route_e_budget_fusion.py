#!/usr/bin/env python3
"""Budget-conditioned fusion of frozen P1 Classification and Cost heads.

CPU-only. Does not train DeepSeek or regenerate answers. Relies on existing
server-local P1 feature/probe caches and P2 Ridge cost-head predictions.

The dense measured-token evaluation is DESCRIPTIVE. Its routing cutoffs use
observed validation lengths after ranking; they are not deployable policies.
The predicted-token evaluation selects cutoffs without observed lengths.
No canonical held-out test data are read by this script.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import route_e_frozen as base
import route_e_prompt_ablation as prefix
from route_e_equal_budget import dense_curve, linear_bracket, random_at_budget

BUDGETS = (900, 1100, 1300, 1500, 1700, 1900, 2100)
ALPHAS = (0., .25, .50, .75, 1.)
FOLDER = ROOT / "outputs/route_e/prompt_ablation/assistant/full"
COST_MODEL = ROOT / "outputs/route_e/token_aware/full/ridge_heads.npz"
COST_PRED = ROOT / "outputs/route_e/token_aware/full/validation_predictions.json"
OUT = ROOT / "results/route_e/budget_fusion"
MAX_TOKENS = 16384


def _assert(condition, message):
    if not condition:
        raise ValueError(message)


def load_npz(path, keys):
    with np.load(path, allow_pickle=False) as src:
        _assert(set(src.files) == set(keys), f"Unexpected keys in {path}: {src.files}")
        d = {k: np.asarray(src[k], dtype=np.float64) for k in keys}
    _assert(all(np.isfinite(x).all() for x in d.values()), f"Nonfinite weights: {path}")
    return d


def logistic_logits(features, probe):
    return ((features - probe["mean"]) / probe["scale"]) @ probe["weight"] + float(probe["bias"])


def predict_cost(features, model):
    z = (features - model["mean"]) / model["scale"]
    raw = z @ model["weight"] + model["intercept"]
    log_t = np.clip(raw[:, 1:3], 0., np.log1p(MAX_TOKENS))
    return np.clip(np.exp(log_t) * model["smear"][None, :] - 1., 0., MAX_TOKENS)


def normalizer(train):
    train = np.asarray(train, dtype=np.float64)
    std = float(np.std(train))
    _assert(np.isfinite(std) and std > 1e-10, "Degenerate train-only normalization")
    return float(np.mean(train)), std


def standardize(data, stats):
    return (np.asarray(data, dtype=np.float64) - stats[0]) / stats[1]


def fusion_scores(q, c, alpha):
    _assert(0. <= alpha <= 1., "alpha must be within [0,1]")
    # HIGHER => NoThinking first. Both signals use the same orientation.
    return alpha * q + (1. - alpha) * c


def data_inputs():
    train, val, _ = base.load_protocol()  # strict 3000/450 IDs and SHA checks
    Xtr, mt = prefix.read_cache(FOLDER / "train.npz", train, base.TRAIN_SHA256, "assistant", False)
    Xva, mv = prefix.read_cache(FOLDER / "val.npz", val, base.VAL_SHA256, "assistant", False)
    for field in ("model", "chat_template_sha256", "layer", "pooling", "feature_dim"):
        _assert(mt[field] == mv[field], f"P1 cache mismatch: {field}")
    _assert(mt["layer"] == "last_after_final_norm" and Xtr.shape == (3000, 1536)
            and Xva.shape == (450, 1536), "Expected full P1 L28 embeddings")
    probe = load_npz(FOLDER / "linear_probe.npz", ("weight", "bias", "mean", "scale"))
    cost = load_npz(COST_MODEL, ("weight", "intercept", "mean", "scale", "smear", "l2"))
    _assert(probe["weight"].shape == (1536,) and probe["mean"].shape == (1536,)
            and probe["scale"].shape == (1536,) and np.all(probe["scale"] > 0), "Bad P1 Logistic head")
    _assert(cost["weight"].shape == (1536, 3) and cost["intercept"].shape == (3,)
            and cost["mean"].shape == (1536,) and cost["scale"].shape == (1536,)
            and cost["smear"].shape == (2,) and np.all(cost["scale"] > 0), "Bad P2 Cost heads")
    logits_train, logits_val = logistic_logits(Xtr, probe), logistic_logits(Xva, probe)
    ct, cv = predict_cost(Xtr, cost), predict_cost(Xva, cost)
    def predict_delta(x):
        raw = ((x - cost["mean"]) / cost["scale"]) @ cost["weight"] + cost["intercept"]
        return np.clip(raw[:,0], -1., 1.)
    source = json.loads(COST_PRED.read_text(encoding="utf-8"))
    _assert(source.get("strict_val_sha256") == base.VAL_SHA256, "P2 validation hash mismatch")
    _assert(set(source["predictions"]) == {r["sample_id"] for r in val}, "Missing/extra cost prediction IDs")
    for i, row in enumerate(val):
        v = source["predictions"][row["sample_id"]]
        _assert(abs(cv[i, 0] - float(v["thinking_tokens_pred"])) < 1e-4
                and abs(cv[i, 1] - float(v["nothinking_tokens_pred"])) < 1e-4,
                f"P2 Cost Head reproduction mismatch: {row['sample_id']}")
    from route_e_equal_budget import load_scores
    old_p1 = load_scores(FOLDER / "validation_p_nt.json", [r["sample_id"] for r in val])
    p_val = 1. / (1. + np.exp(-np.clip(logits_val, -60., 60.)))
    _assert(max(abs(p_val[i] - old_p1[r["sample_id"]]) for i, r in enumerate(val)) < 1e-5,
            "P1 Logistic reproduction mismatch")
    old_half = dense_curve(val, old_p1)[225]
    _assert(abs(old_half["accuracy"] - 344/450) < 3e-4
            and abs(old_half["tokens"] - 1427.05) < 3., "P1 50% baseline mismatch")
    qstats = normalizer(logits_train)
    cstats = normalizer((ct[:, 0] - ct[:, 1]) / 1000.)
    q = standardize(logits_val, qstats)
    c = standardize((cv[:, 0] - cv[:, 1]) / 1000., cstats)
    _assert(np.isfinite(q).all() and np.isfinite(c).all(), "Nonfinite fusion features")
    legacy = legacy_score_controls(val, logits_train, logits_val, ct, cv,
                                   predict_delta(Xtr), predict_delta(Xva))
    return val, q, c, cv, legacy, {"classification_logit": qstats, "cost_delta_thousand": cstats,
                           "train_sha256": base.TRAIN_SHA256, "val_sha256": base.VAL_SHA256,
                           "p1_half": old_half, "p2_prediction_sha256": base.sha256(COST_PRED)}


def ranking(scores, rows):
    _assert(len(scores) == len(rows), "Wrong score count")
    _assert(np.isfinite(scores).all(), "Invalid routing scores")
    return sorted(range(len(rows)), key=lambda i: (-float(scores[i]), rows[i]["sample_id"]))


def probabilities(rank, left, right, mix):
    n = len(rank)
    k0 = int(round(left * n))
    k1 = int(round(right * n))
    _assert(0 <= k0 <= n and 0 <= k1 <= n, "Invalid bracket fraction")
    nt = np.zeros(n)
    nt[rank[:k0]] += 1. - mix
    nt[rank[:k1]] += mix
    return nt


def summarize(rows, nt, predicted_lengths=None):
    nt = np.asarray(nt, dtype=float)
    _assert(len(nt) == len(rows) and np.all((nt >= -1e-9) & (nt <= 1+1e-9)), "Invalid NT probabilities")
    t = np.array([[r["t_t"], r["t_n"]] for r in rows], dtype=float)
    p = np.array([[r["p_t"], r["p_n"]] for r in rows], dtype=float)
    ac = (1-nt)*p[:, 0] + nt*p[:, 1]
    tc = (1-nt)*t[:, 0] + nt*t[:, 1]
    out = {"accuracy": float(np.mean(ac)), "actual_tokens": float(np.mean(tc)),
           "nt_fraction": float(np.mean(nt))}
    if predicted_lengths is not None:
        expected = (1-nt)*predicted_lengths[:, 0] + nt*predicted_lengths[:, 1]
        out["predicted_tokens"] = float(np.mean(expected))
        out["budget_error"] = out["actual_tokens"] - out["predicted_tokens"]
    out["by_dataset"] = {}
    for key in sorted({r["dataset"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["dataset"] == key]
        out["by_dataset"][key] = {"accuracy": float(np.mean(ac[idx])),
                                  "actual_tokens": float(np.mean(tc[idx])),
                                  "nt_fraction": float(np.mean(nt[idx])), "n": len(idx)}
    return out


def policy_at_budget(rows, scores, budget, mode="observed", predicted_lengths=None):
    """Observed: research-only realized-cost bracketing. Predicted: no realized-cost selection."""
    n = len(rows)
    order = ranking(np.asarray(scores), rows)
    if mode == "observed":
        score_map = {r["sample_id"]: float(scores[i]) for i, r in enumerate(rows)}
        curve = dense_curve(rows, score_map)
    elif mode == "predicted":
        _assert(predicted_lengths is not None and predicted_lengths.shape == (n, 2),
                "Predicted lengths required for prospective cutoff")
        base_tokens = float(np.mean(predicted_lengths[:, 0]))
        diff = (predicted_lengths[:, 1] - predicted_lengths[:, 0]) / n
        costs = base_tokens + np.r_[0, np.cumsum(diff[order])]
        curve = [{"tokens": float(v), "fraction": k/n} for k, v in enumerate(costs)]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    try:
        mix = linear_bracket(curve, budget)
    except ValueError:
        return None  # predicted budget may be unreachable
    nt = probabilities(order, mix["policy_left_fraction"],
                       mix["policy_right_fraction"], mix["mix_right_probability"])
    m = summarize(rows, nt, predicted_lengths)
    if mode == "observed":
        _assert(abs(m["actual_tokens"]-budget) < 1e-5, "Observed-cost bracket not matched")
    else:
        _assert(abs(m["predicted_tokens"]-budget) < 1e-5, "Predicted-cost bracket not matched")
    return m, nt


def make_folds(rows, k, seed):
    """Deterministic stratification by dataset, without reading any labels/outcomes."""
    _assert(2 <= k <= 10 and len(rows) >= 2*k, "Invalid folds")
    rng = np.random.default_rng(seed)
    groups = [[] for _ in range(k)]
    for ds in sorted({r["dataset"] for r in rows}):
        ids = np.array([i for i, r in enumerate(rows) if r["dataset"] == ds])
        rng.shuffle(ids)
        for j, index in enumerate(ids):
            groups[j % k].append(int(index))
    _assert(sorted(i for g in groups for i in g) == list(range(len(rows)))
            and all(groups), "Fold coverage/overlap failure")
    return [np.array(sorted(group), dtype=int) for group in groups]


def bucket(budget):
    if budget <= 1100:
        return "low"
    if budget <= 1700:
        return "medium"
    return "high"


def choose_weights(rows, q, c, budgets, alphas, penalty):
    """Choose alpha using DEVELOPMENT rows only; never see held-out fold labels."""
    scores = {}
    for a in alphas:
        for b in budgets:
            trial = policy_at_budget(rows, fusion_scores(q, c, a), b)
            _assert(trial is not None, "Development budget unreachable")
            scores[(a, b)] = trial[0]["accuracy"]
    fixed = max(alphas, key=lambda a: (np.mean([scores[a, b] for b in budgets]), -abs(a-.5)))
    best = None
    # All 5^3 combinations, with a predeclared small smoothness penalty.
    for lo, mid, hi in itertools.product(alphas, repeat=3):
        coeff = {"low":lo, "medium":mid, "high":hi}
        objective = float(np.mean([scores[coeff[bucket(b)], b] for b in budgets]))
        objective -= penalty * (abs(lo-mid) + abs(mid-hi))
        rank = (objective, -abs(lo-mid)-abs(mid-hi))
        if best is None or rank > best[0]:
            best = (rank, coeff)
    hard = None
    for lo, mid, hi in itertools.product((0.,1.), repeat=3):
        coeff = {"low":lo, "medium":mid, "high":hi}
        value = float(np.mean([scores[coeff[bucket(b)], b] for b in budgets]))
        value -= penalty * (abs(lo-mid) + abs(mid-hi))
        rank = (value, -abs(lo-mid)-abs(mid-hi))
        if hard is None or rank > hard[0]:
            hard = (rank, coeff)
    return {"best_fixed": fixed, "piecewise": best[1], "hard_switch": hard[1],
            "dev_accuracy": {str(a): {str(b): scores[a,b] for b in budgets} for a in alphas}}


def conditional_ci(a, b, reps, rng):
    """Paired CI on *frozen* mixed policies (not training/selection uncertainty)."""
    delta = np.asarray(a)-np.asarray(b)
    n = len(delta)
    if reps == 0:
        return None
    indices = rng.integers(0, n, size=(reps, n))
    draws = 100 * np.mean(delta[indices], axis=1)
    return [float(np.quantile(draws,.025)),float(np.quantile(draws,.975))]


def legacy_score_controls(rows, logits_train, logits_val, t_train, t_val,
                          ridge_quality_train, ridge_quality_val):
    """Historical fixed lambda=0.05 score, with train-only logit scale matching."""
    # Old logistic+cost used NEGATIVE class logits as Thinking preference.
    source = -np.asarray(logits_train, float)
    target = np.asarray(ridge_quality_train, float)
    scale = float(np.std(target) / np.std(source))
    offset = float(np.mean(target) - scale * np.mean(source))
    scaled_thinking = offset + scale * (-np.asarray(logits_val, float))
    dcost = (t_val[:, 0] - t_val[:, 1]) / 1000.
    return {"legacy_logistic_plus_cost": -scaled_thinking + .05 * dcost,
            "legacy_ridge_plus_cost": -np.asarray(ridge_quality_val, float) + .05 * dcost}


def run(rows, q, c, lengths, budgets, alphas, folds, seed, penalty, bootstrap,
        legacy=None):
    """Select head weights on 4/5 folds; rank all held-out scores once per budget.

    This avoids in-fold weight selection AND tiny-fold 2,100-token coverage issues.
    The dense expected-token cutoff remains descriptive (observed-cost based).
    """
    all_sweep = []
    for a in alphas:
        for b in budgets:
            result = policy_at_budget(rows, fusion_scores(q, c, a), b)
            all_sweep.append({"alpha": a, "budget": b, **result[0]})
    n = len(rows)
    oof = {b: {name: np.full(n, np.nan) for name in
               ("fixed", "adaptive", "hard_switch")} for b in budgets}
    selection_records = []
    for f, heldout in enumerate(make_folds(rows, folds, seed)):
        dev = np.setdiff1d(np.arange(n), heldout)
        devrows = [rows[i] for i in dev]
        selected = choose_weights(devrows, q[dev], c[dev], budgets, alphas, penalty)
        selection_records.append({"fold":f,"heldout_n":len(heldout),
                                  "fixed":selected["best_fixed"],
                                  "piecewise":selected["piecewise"],
                                  "hard_switch":selected["hard_switch"]})
        for b in budgets:
            choices = {"fixed":selected["best_fixed"],
                       "adaptive":selected["piecewise"][bucket(b)],
                       "hard_switch":selected["hard_switch"][bucket(b)]}
            for name, alpha in choices.items():
                oof[b][name][heldout] = fusion_scores(q[heldout],c[heldout],alpha)
    _assert(all(np.isfinite(s).all() for scores in oof.values() for s in scores.values()),
            "OOF score coverage failed")
    rng = np.random.default_rng(seed + 7919)
    report = []
    for b in budgets:
        scores = {"classification_only": fusion_scores(q,c,1.),
                  "cost_only": fusion_scores(q,c,0.), **oof[b]}
        if legacy:
            scores.update(legacy)
        methods, accuracy_per_question = {}, {}
        for name, score in scores.items():
            obs = policy_at_budget(rows, score, b, "observed", lengths)
            fut = policy_at_budget(rows, score, b, "predicted", lengths)
            methods[name] = {"observed":obs[0],"predicted":None if fut is None else fut[0]}
            nt = obs[1]
            accuracy_per_question[name] = np.array([
                (1-nt[i])*r["p_t"] + nt[i]*r["p_n"] for i,r in enumerate(rows)])
        diff = 100*(methods["adaptive"]["observed"]["accuracy"]-
                    methods["fixed"]["observed"]["accuracy"])
        ci = conditional_ci(accuracy_per_question["adaptive"],
                            accuracy_per_question["fixed"],bootstrap,rng)
        report.append({"budget":b,"methods":methods,"adaptive_minus_fixed_pp":diff,
                       "paired_bootstrap_95ci_pp":ci,
                       "random_expected_accuracy":random_at_budget(rows,b)["accuracy"]})
    return all_sweep, selection_records, report


def render_plots(folder, sweep, cv, budgets, alphas):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("BCHF_PLOT_SKIPPED: matplotlib unavailable")
        return
    arr = np.array([[next(r["accuracy"] for r in sweep if r["alpha"]==a and r["budget"]==b)
                     for b in budgets] for a in alphas]) * 100
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set(xticks=range(len(budgets)),xticklabels=budgets,
           yticks=range(len(alphas)),yticklabels=[f"{a:.2f}" for a in alphas],
           xlabel="Mean generated tokens (observed-cost matched)",
           ylabel="Classification weight α",title="Fixed fusion weight sweep (exploratory)")
    for i in range(len(alphas)):
        for j in range(len(budgets)):
            ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=8,
                    color="white" if arr[i,j] < (arr.min()+arr.max())/2 else "black")
    fig.colorbar(im,ax=ax,label="Accuracy (%)")
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(folder/f"alpha_budget_heatmap.{ext}",dpi=170)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for name in ("classification_only","cost_only","fixed","adaptive","hard_switch"):
        ax.plot(budgets,[100*next(c["methods"][name]["observed"]["accuracy"]
                                  for c in cv if c["budget"]==b) for b in budgets],
                marker="o",label=name.replace("_"," "))
    ax.plot(budgets,[100*next(c["random_expected_accuracy"] for c in cv if c["budget"]==b)
                       for b in budgets],ls="--",label="Global random")
    ax.set(xlabel="Mean generated tokens (descriptive equal-cost replay)",
           ylabel="Strict expected accuracy (%)",title="5-fold held-out fusion comparison")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png","svg"):
        fig.savefig(folder/f"crossfit_accuracy_token.{ext}",dpi=170)
    plt.close(fig)


def save_report(folder, sweep, records, cv, metadata, config, budgets, alphas, make_plots=True):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"report.json").write_text(json.dumps({
        "status":"exploratory validation only; canonical test never loaded",
        "metadata":metadata,"configuration":config,"fixed_sweep":sweep,
        "oof_selection_by_fold":records,"crossfit_summary":cv,
        "limitations":["Observed-cost matching uses validation actual lengths AFTER ranking",
          "Predicted-only policies may miss target budget due to length-calibration bias",
          "The 450 validation problems were investigated in earlier Route E experiments",
          "Conditional bootstrap ignores α selection / retraining and multiple testing",
          "No router prefill/inference overhead in generated-token costs"]
        },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    with (folder/"fixed_alpha_sweep.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h);w.writerow(["budget","alpha","accuracy_pct","tokens","nt_fraction"])
        for r in sweep:w.writerow([r["budget"],r["alpha"],100*r["accuracy"],r["actual_tokens"],r["nt_fraction"]])
    with (folder/"crossfit_summary.csv").open("w",newline="",encoding="utf-8") as h:
        w=csv.writer(h);w.writerow(["budget","method","observed_accuracy_pct","observed_tokens",
                      "predicted_accuracy_pct","actual_tokens_predicted_policy","predicted_tokens","nt_fraction_predicted_policy"])
        for r in cv:
            for name,m in r["methods"].items():
                p=m["predicted"]
                w.writerow([r["budget"],name,100*m["observed"]["accuracy"],m["observed"]["actual_tokens"],
                            None if p is None else 100*p["accuracy"],
                            None if p is None else p["actual_tokens"],
                            None if p is None else p["predicted_tokens"],
                            None if p is None else p["nt_fraction"]])
    if make_plots:
        render_plots(folder,sweep,cv,budgets,alphas)


def self_test():
    rows = [{"sample_id": f"gsm8k:{i}","dataset":"gsm8k",
             "p_t":.8,"p_n":.5,"t_t":1500.+25*i,"t_n":200.+10*i} for i in range(16)]
    q=np.linspace(-2,2,len(rows));c=np.linspace(2,-2,len(rows))
    assert np.allclose(fusion_scores(q,c,0),c)
    assert np.allclose(fusion_scores(q,c,1),q)
    assert ranking(c,rows) == list(reversed(ranking(q,rows)))
    for mode in ("observed","predicted"):
        p=policy_at_budget(rows,q,800,mode,np.array([[r["t_t"],r["t_n"]] for r in rows]))
        assert p is not None and 0 <= p[0]["accuracy"] <= 1
        k=p[0]["actual_tokens" if mode=="observed" else "predicted_tokens"]
        assert abs(k-800)<1e-5
    folds=make_folds(rows,4,42)
    assert sorted(i for fold in folds for i in fold)==list(range(16))
    w=choose_weights(rows,q,c,(800,900),ALPHAS,.0025)
    assert set(w["piecewise"])=={"low","medium","high"}
    assert conditional_ci([.7,.5],[.5,.5],100,np.random.default_rng(0)) is not None
    # No score uses observed outcomes or actual token lengths.
    print("BCHF_SELFTEST_PASS score orientation, endpoints, fold coverage, budget interpolation")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("self-test","run"))
    parser.add_argument("--out",type=Path,default=OUT)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--penalty",type=float,default=.0025,
                        help="Smoothness penalty in fractional accuracy per unit alpha jump")
    parser.add_argument("--bootstrap",type=int,default=1000)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--no-plot",action="store_true")
    args=parser.parse_args()
    if args.command=="self-test":
        self_test();return
    _assert(0 <= args.penalty <= .1 and args.bootstrap >= 0,"Invalid penalty/bootstrap")
    _assert(not (args.out/"report.json").exists() or args.overwrite,
            "Results already exist; use --overwrite to replace only new BCHF outputs")
    val,q,c,cost,legacy,meta=data_inputs()
    sweep,records,cv=run(val,q,c,cost,BUDGETS,ALPHAS,args.folds,args.seed,
                         args.penalty,args.bootstrap,legacy)
    config={"budgets":BUDGETS,"alphas":ALPHAS,"folds":args.folds,
            "seed":args.seed,"smoothness_penalty":args.penalty,
            "selection":"5-fold held-out alpha score selection; held-out scores ranked across entire 450",
            "cutoff":"observed=descriptive validation actual-length; predicted=predicted-length transductive",
            "classification_head":"P1 frozen Logistic logit (positive=NT)",
            "cost_head":"P2 Ridge predicted (Thinking−NoThinking) generated tokens / 1000",
            "score":"alpha(B)*zscore_train(logit_NT)+(1-alpha(B))*zscore_train(delta_t/1000)",
            "prediction_budget_mode":"uncalibrated original cost heads; diagnostic only"}
    save_report(args.out,sweep,records,cv,meta,config,BUDGETS,ALPHAS,
                make_plots=not args.no_plot)
    for r in cv:
        print(f"BCHF_BUDGET {r['budget']} fixed={100*r['methods']['fixed']['observed']['accuracy']:.2f}% "
              f"adaptive={100*r['methods']['adaptive']['observed']['accuracy']:.2f}% "
              f"delta={r['adaptive_minus_fixed_pp']:+.2f}pp")
    print(f"BCHF_DONE {args.out}")


if __name__=="__main__":
    main()