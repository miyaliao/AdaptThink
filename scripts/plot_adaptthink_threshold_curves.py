#!/usr/bin/env python3
"""Real per-question threshold curves for three AdaptThink routers (strict val-450).

No generation or GPU. Requires server-local per-question validation scores.
Each router is swept through 451 exact top-k policies, from all NoThinking
to all Thinking. Random Router is simulated for 21 fractions x 1000 seeds.
Do NOT interpolate the previously reported seven budget-summary points.

From repository root:
  python scripts/plot_adaptthink_threshold_curves.py self-test
  python scripts/plot_adaptthink_threshold_curves.py run
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "ModernBERT + Utility-Hard": ROOT / "outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/validation_predictions.jsonl",
    "DeepSeek P1 Classification": ROOT / "outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json",
    "DeepSeek P1 Cost-only": ROOT / "outputs/route_e/token_aware/full/validation_predictions.json",
}
OUT = ROOT / "results/route_e/threshold_curves"


def load_scores(path, kind, rows, strict_sha):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing per-question prediction file: {path}")
    ids = [r["sample_id"] for r in rows]
    if kind == "modernbert":
        entries = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        scores = {str(x["sample_id"]): float(x["predicted_p_nothinking"]) for x in entries}
        if len(scores) != len(entries):
            raise ValueError("Duplicated ModernBERT sample ID")
    elif kind == "p1":
        obj = json.loads(path.read_text())
        scores = {str(k): float(v) for k, v in obj["p_nt"].items()}
    elif kind == "cost":
        obj = json.loads(path.read_text())
        if obj.get("strict_val_sha256") != strict_sha:
            raise ValueError("Cost-only prediction strict-validation SHA mismatch")
        scores = {str(k): float(v["delta_t_pred"]) for k, v in obj["predictions"].items()}
    else:
        raise ValueError(kind)
    if len(scores) != len(rows) or set(scores) != set(ids):
        raise ValueError(f"{kind}: missing/extra sample IDs")
    if not all(math.isfinite(v) for v in scores.values()):
        raise ValueError(f"{kind}: non-finite prediction scores")
    return scores


def threshold_curve(rows, scores):
    """k=0: 100% NT; k=N: 100% Thinking.

    High score means route to NoThinking first. Reverse the NoThinking
    priority ordering so increasing k moves the least NT-preferred items
    back to Thinking. Same deterministic sample-ID tie breaking as Route E.
    """
    n = len(rows)
    nt_order = sorted(range(n), key=lambda i: (-scores[rows[i]["sample_id"]],
                                                 rows[i]["sample_id"]))
    order = nt_order[::-1]
    pn = np.array([r["p_n"] for r in rows], dtype=float)
    pt = np.array([r["p_t"] for r in rows], dtype=float)
    tn = np.array([r["t_n"] for r in rows], dtype=float)
    tt = np.array([r["t_t"] for r in rows], dtype=float)
    a = pn.mean() + np.r_[0., np.cumsum((pt - pn)[order])] / n
    t = tn.mean() + np.r_[0., np.cumsum((tt - tn)[order])] / n
    assert abs(a[-1] - pt.mean()) < 1e-9
    assert abs(t[-1] - tt.mean()) < 1e-7
    return {"accuracy": a, "tokens": t, "thinking_fraction": np.arange(n+1)/n}


def random_router(rows, n_shares, n_seeds, seed):
    """Actual replay with exact fixed-count random assignments per fraction."""
    n = len(rows)
    pn = np.array([r["p_n"] for r in rows], dtype=float)
    pt = np.array([r["p_t"] for r in rows], dtype=float)
    tn = np.array([r["t_n"] for r in rows], dtype=float)
    tt = np.array([r["t_t"] for r in rows], dtype=float)
    counts = np.rint(np.linspace(0, n, n_shares)).astype(int)
    rng = np.random.default_rng(seed)
    acc = np.empty((n_seeds, len(counts)))
    tok = np.empty_like(acc)
    for i in range(n_seeds):
        perm = rng.permutation(n)
        acc[i] = pt.mean() + np.r_[0., np.cumsum((pn-pt)[perm])][counts]/n
        tok[i] = tt.mean() + np.r_[0., np.cumsum((tn-tt)[perm])][counts]/n
    q = counts/n
    return {
        "n_nothinking": counts, "nt_fraction": q,
        "accuracy": acc.mean(axis=0), "tokens": tok.mean(axis=0),
        "sd_accuracy": acc.std(axis=0), "sd_tokens": tok.std(axis=0),
        "expected_accuracy": (1-q)*pt.mean() + q*pn.mean(),
        "expected_tokens": (1-q)*tt.mean() + q*tn.mean(),
    }


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(curves, random, out, n_markers, n_reps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(next(iter(curves.values()))["accuracy"]) - 1
    marked = np.unique(np.rint(np.linspace(0, n, n_markers)).astype(int))
    colors = ["#2251d1", "#168578", "#7951aa"]

    def decorate(ax):
        ax.plot(random["tokens"], random["accuracy"]*100, "--x",
                color="#dc9a43", linewidth=1.25, markersize=4,
                label=f"Random Router ({n_reps} seeds)")
        ax.plot(random["expected_tokens"], random["expected_accuracy"]*100,
                linestyle=":", color="#8591a3", linewidth=1.1,
                label="Random expectation")
        ax.scatter(random["expected_tokens"][0], random["expected_accuracy"][0]*100,
                   marker="*", s=115, color="#2ba344", zorder=6, label="Always Thinking")
        ax.scatter(random["expected_tokens"][-1], random["expected_accuracy"][-1]*100,
                   marker="*", s=115, color="#d84850", zorder=6, label="Always NoThinking")
        ax.set_xlabel("Mean generated tokens per question")
        ax.set_ylabel("Strict validation accuracy (%)")
        ax.grid(alpha=.25)
        ax.legend(loc="lower right", fontsize=7.5)

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.1), sharex=True, sharey=True)
    for ax, (title, c), color in zip(axes, curves.items(), colors):
        ax.plot(c["tokens"], c["accuracy"]*100, "-", color=color,
                linewidth=1.2, alpha=.8, label="All 451 threshold policies")
        ax.plot(c["tokens"][marked], c["accuracy"][marked]*100, "o",
                color=color, markersize=3)
        decorate(ax)
        ax.set_title(title, fontsize=10.5, weight="bold")
    fig.suptitle("AdaptThink | Threshold-swept Accuracy–Token Curves", fontsize=15)
    fig.text(.5, .018,
             f"450 strict validation questions; 451 policies per method; "
             f"{len(marked)} marked thresholds; 21 random fractions; "
             f"{n_reps} random seeds. Router overhead excluded.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, .93))
    for fmt in ("png", "svg"):
        fig.savefig(out/f"three_router_threshold_curves.{fmt}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for (title, c), color in zip(curves.items(), colors):
        ax.plot(c["tokens"], c["accuracy"]*100, color=color,
                linewidth=1.5, label=title)
        ax.scatter(c["tokens"][marked], c["accuracy"][marked]*100,
                   color=color, s=12)
    decorate(ax)
    ax.set_title("AdaptThink | Three Router Threshold Sweeps")
    fig.tight_layout()
    for fmt in ("png", "svg"):
        fig.savefig(out/f"overlay_threshold_curves.{fmt}",
                    dpi=240, bbox_inches="tight")
    plt.close(fig)


def self_test():
    rows = [
        {"sample_id":"x","p_n":0.,"p_t":1.,"t_n":5.,"t_t":20.},
        {"sample_id":"y","p_n":1.,"p_t":0.,"t_n":10.,"t_t":50.},
        {"sample_id":"z","p_n":.5,"p_t":.5,"t_n":10.,"t_t":50.},
        {"sample_id":"w","p_n":.2,"p_t":.8,"t_n":10.,"t_t":50.},
    ]
    c = threshold_curve(rows, {"x":.1,"y":.9,"z":.6,"w":.2})
    assert abs(c["accuracy"][0]-.425) < 1e-9
    assert abs(c["accuracy"][1]-.675) < 1e-9
    assert abs(c["accuracy"][-1]-.575) < 1e-9
    r = random_router(rows, 5, 13, 42)
    assert abs(r["accuracy"][0]-c["accuracy"][-1]) < 1e-9
    assert abs(r["accuracy"][-1]-c["accuracy"][0]) < 1e-9
    print("THRESHOLD_CURVES_SELFTEST_PASS")


def run(args):
    import route_e_frozen as base
    train, val, _ = base.load_protocol()
    if len(train) != 3000 or len(val) != 450:
        raise ValueError("Unexpected strict split")
    sources = [
        ("ModernBERT + Utility-Hard", args.modernbert_scores, "modernbert"),
        ("DeepSeek P1 Classification", args.p1_scores, "p1"),
        ("DeepSeek P1 Cost-only", args.cost_scores, "cost"),
    ]
    curves = {}
    provenance = {}
    for label, path, kind in sources:
        score = load_scores(path, kind, val, base.VAL_SHA256)
        curves[label] = threshold_curve(val, score)
        provenance[label] = str(path)
        print(f"LOADED {label}: {len(score)} strict per-question scores", flush=True)

    p1 = curves["DeepSeek P1 Classification"]
    # Historical reproduction gate, not an input to plotted values.
    if abs(100*p1["accuracy"][225] - 76.4444) > .05 or \
       abs(p1["tokens"][225] - 1427.05) > 2:
        raise RuntimeError("P1 50% NoThinking historical reproduction failed")

    random = random_router(val, args.random_shares, args.random_seeds, args.seed)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for label, curve in curves.items():
        for k in range(451):
            records.append({"method":label, "n_thinking":k,
                            "thinking_fraction":float(curve["thinking_fraction"][k]),
                            "accuracy":float(curve["accuracy"][k]),
                            "mean_output_tokens":float(curve["tokens"][k])})
    write_csv(out/"all_451_threshold_policies.csv",
              ["method","n_thinking","thinking_fraction","accuracy","mean_output_tokens"],
              records)
    rr = []
    for i in range(len(random["nt_fraction"])):
        rr.append({key:float(random[key][i]) for key in
                   ("nt_fraction", "accuracy", "tokens", "sd_accuracy",
                    "sd_tokens", "expected_accuracy", "expected_tokens")})
    write_csv(out/"random_router_fraction_sweep.csv",
              list(rr[0]), rr)
    make_plots(curves, random, out, args.plot_points, args.random_seeds)
    (out/"provenance.json").write_text(
        json.dumps({"data":"strict val-450 offline replay",
                    "sources":provenance,
                    "train_sha256":base.TRAIN_SHA256,
                    "val_sha256":base.VAL_SHA256,
                    "random_seed":args.seed,
                    "random_repetitions":args.random_seeds,
                    "limitations":"Observed validation per-mode outcomes, output tokens only; no sealed test or router overhead"},
                   indent=2)+"\n")
    print(f"THRESHOLD_CURVES_DONE {out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    run_p = commands.add_parser("run")
    run_p.add_argument("--modernbert-scores", type=Path, default=DEFAULTS["ModernBERT + Utility-Hard"])
    run_p.add_argument("--p1-scores", type=Path, default=DEFAULTS["DeepSeek P1 Classification"])
    run_p.add_argument("--cost-scores", type=Path, default=DEFAULTS["DeepSeek P1 Cost-only"])
    run_p.add_argument("--out", type=Path, default=OUT)
    run_p.add_argument("--random-seeds", type=int, default=1000)
    run_p.add_argument("--random-shares", type=int, default=21)
    run_p.add_argument("--plot-points", type=int, default=31)
    run_p.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        if min(args.random_seeds, args.random_shares-1, args.plot_points-1) < 1:
            parser.error("invalid random-seeds/random-shares/plot-points")
        run(args)

if __name__ == "__main__":
    main()
