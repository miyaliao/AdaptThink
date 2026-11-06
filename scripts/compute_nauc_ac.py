#!/usr/bin/env python3
"""Compute ThinkSwitcher-style nAUC-AC from *real* 451-policy threshold sweeps.

Input: results/route_e/threshold_curves/all_451_threshold_policies.csv,
produced by plot_adaptthink_threshold_curves.py, NOT seven budget summaries.

The paper defines nAUC-AC = AUC-AC(router) - trapezoid(Always NT, Always T).
Because some validation questions have fewer Thinking than NoThinking tokens,
observed costs along a threshold sweep may locally move backwards.
Convention: sort all 451 empirical operating points by *observed mean cost*,
average exact duplicate costs, linearly interpolate in token-cost space,
clip to the common Always-NoThinking/Always-Thinking endpoint interval,
and trapezoid-integrate the resulting sampled accuracy(cost) curve.

No GPU, no validation-based selection of thresholds, no synthetic results.
"""
import argparse
import bisect
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results/route_e/threshold_curves/all_451_threshold_policies.csv"


def read_curves(path):
    data = defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            method = row["method"]
            k = int(row["n_thinking"])
            if k in data[method]:
                raise ValueError(f"Duplicate threshold for {method} k={k}")
            acc = float(row["accuracy"])
            cost = float(row["mean_output_tokens"])
            if not math.isfinite(acc) or not math.isfinite(cost) or not (0 <= acc <= 1):
                raise ValueError(f"Invalid accuracy or cost: {row}")
            data[method][k] = (cost, acc)
    if not data:
        raise ValueError("Empty input")
    for name, points in data.items():
        if sorted(points) != list(range(451)):
            raise ValueError(f"{name}: expected exactly 451 policies, k=0..450")
    return dict(data)


def auc_cost_sorted(points, low, high):
    if not low < high:
        raise ValueError("Invalid integration interval")
    by_x = sorted(points)
    # Group identical token costs to avoid ambiguity from vertical segments.
    unique = []
    for x, y in by_x:
        if unique and x == unique[-1][0]:
            unique[-1][1] += y
            unique[-1][2] += 1
        else:
            unique.append([x, y, 1])
    xs = [v[0] for v in unique]
    ys = [v[1]/v[2] for v in unique]
    if xs[0] > low + 1e-7 or xs[-1] < high - 1e-7:
        raise ValueError("Curve does not cover the shared baseline token interval")

    def at(x):
        idx = bisect.bisect_left(xs, x)
        if idx < len(xs) and abs(xs[idx]-x) < 1e-9:
            return ys[idx]
        if idx == 0 or idx >= len(xs):
            raise ValueError("Endpoint interpolation out of range")
        a, b = xs[idx-1], xs[idx]
        t = (x-a)/(b-a)
        return ys[idx-1]*(1-t)+ys[idx]*t

    grid = [low] + [x for x in xs if low < x < high] + [high]
    acc = [at(x) for x in grid]
    return sum((acc[i]+acc[i+1])*.5*(grid[i+1]-grid[i])
               for i in range(len(grid)-1))


def calculate(curves):
    name0 = next(iter(curves))
    low, acc_low = curves[name0][0]  # 100% NoThinking
    high, acc_high = curves[name0][450]  # 100% Thinking
    if low >= high:
        raise ValueError("Expected Always NoThinking to use fewer tokens")
    baseline = (acc_low+acc_high)*.5*(high-low)
    results = []
    for name, policies in curves.items():
        x0, a0 = policies[0]
        x1, a1 = policies[450]
        if max(abs(x0-low),abs(x1-high)) > 1e-5 or \
           max(abs(a0-acc_low),abs(a1-acc_high)) > 1e-8:
            raise ValueError(f"Different baseline endpoints for {name}")
        ordered = [policies[k] for k in range(451)]
        backtracks = sum(ordered[k+1][0] < ordered[k][0] - 1e-9
                         for k in range(450))
        auc = auc_cost_sorted(ordered,low,high)
        difference = auc-baseline
        results.append({
            "method":name,
            "AUC_AC":auc,
            "nAUC_AC":difference,
            "mean_gain_pp":100*difference/(high-low),
            "cost_backtracks":backtracks,
            "min_cost":low,
            "max_cost":high,
            "random_reference_AUC":baseline,
        })
    return results


def self_test():
    policy = {
        0:(5.,.3), 450:(10.,.7),
        **{k:(5.+5*k/450, .3+.4*k/450) for k in range(1,450)}
    }
    result = auc_cost_sorted(list(policy.values()), 5., 10.)
    assert abs(result-2.5) < 1e-9, result
    assert abs(auc_cost_sorted([(5.,.3),(7.5,.6),(10.,.7)],5.,10.)-2.75) < 1e-10
    print("NAUC_AC_SELFTEST_PASS")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",type=Path,default=DEFAULT_CSV)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
        return
    curves=read_curves(a.csv)
    results=calculate(curves)
    print("METHOD | AUC-AC | nAUC-AC | average gain (pp) | backwards token steps")
    for r in results:
        print(f"{r['method']} | {r['AUC_AC']:.4f} | {r['nAUC_AC']:+.4f}"
              f" | {r['mean_gain_pp']:+.3f} | {r['cost_backtracks']}")
    output=a.csv.parent/"nauc_ac_summary.csv"
    with output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"BASELINE AUC={results[0]['random_reference_AUC']:.5f}, "
          f"token interval=[{results[0]['min_cost']:.3f},"
          f"{results[0]['max_cost']:.3f}]")
    print("NAUC_AC_DONE",output)


if __name__=="__main__":
    main()
