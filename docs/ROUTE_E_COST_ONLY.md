# Route E / P2 — Experiment 1: Cost-only ablation

**Status: code and synthetic CI passing; real strict 450-question server replay pending.**
**Date: 2026-11-03**

Parent protocol: [ROUTE_E_TOKEN_AWARE.md](ROUTE_E_TOKEN_AWARE.md).

## 1. Why this experiment?

Existing frozen P1 Ridge heads predict:
- Thinking benefit \(\widehat{\Delta p}(x)\), validation Spearman approximately **0.096**.
- Extra Thinking output tokens \(\widehat{\Delta t}(x)\), validation Spearman approximately **0.623**.

The combined P2 score with frozen \(\lambda=0.05\) improved observed-budget accuracy over quality-only B on all seven budgets, but why remains unclear. The Quality Head might contribute little. This experiment isolates whether the Cost Head **alone** can explain those gains.

Hypotheses: (H1a) D (quality+cost) > E (cost only) would support complementary quality information; (H1b) D ≈ E > B (quality only) would suggest cost dominates. A third possibility is E < B or E < D on some budgets. No result is predetermined.

## 2. Exactly what changes?

**Nothing is trained or regenerated.** Only the score used to rank NoThinking candidates is changed. All three methods read the exact **same 450 saved pre-generation predictions** from the existing P2 Ridge heads:

| Method | Thinking-preference score (higher = allocate Thinking) | NoThinking priority (higher = NT sooner) |
| --- | --- | --- |
| B — Quality only (unchanged) | \(\hat{\Delta p}\) | \(-\hat{\Delta p}\) |
| **E — Cost only (new)** | \(-\hat{\Delta t}\) | \(+\hat{\Delta t}\) |
| D — Quality + Cost (unchanged) | \(\hat{\Delta p}-0.05\hat{\Delta t}/1000\) | \(-(\hat{\Delta p}-0.05\hat{\Delta t}/1000)\) |

The cost-only policy sends **highest predicted incremental Thinking-cost** questions to NoThinking first and saves Thinking for those predicted to cost relatively little.

- Identical frozen DeepSeek Assistant-only P1 feature representation.
- Identical original Ridge heads, train/validation data, fixed \(\lambda=0.05\), strict scoring and canonical 450 validation IDs.
- Identical budgets: **900/1100/1300/1500/1700/1900/2100 generated tokens per question**.
- Identical dense top-k and observed-token interpolation from Route E/P2; the resulting cost-matched curves are **exploratory, not deployable**.
- Validation true token lengths are **never** inputs to B, E or D scores. They are used only *after* ranking to measure offline cost and interpolate to reporting budgets.
- For each budget also report GSM8K and MATH accuracy, realized cost and fraction NoThinking, which may differ across datasets.

The new script hard-fails unless its recomputed B and D curves match the **previous P2 validation.json** at all seven budgets (accuracy, token cost, NT fraction) within tolerance. It checks original strict validation SHA, source P2 train/val hashes, frozen full status, lambda, exact ID order, and saved score consistency. Old outputs are **never overwritten**.

## 3. Questions and statistics

1. **Does E outperform B?** Compare E minus B at matched cost. If yes, input-dependent cost contains useful routing information, regardless of quality head accuracy.
2. **Does D outperform E?** Compare D minus E at matched cost. If yes, adding quality head provides incremental value beyond cost.
3. **Is cost prediction question-level?** Report Spearman/MAE/bias for predicted vs true extra Thinking tokens **within GSM8K and within MATH**. Global Spearman can partly reflect average cost differences between datasets. Inspect per-dataset NT allocation and accuracy.
4. **How robust?** Conditional **paired per-question bootstrap** (default 2,000 replications) for D−E, E−B, D−B. Intervals are **nominal unadjusted 95%**, with seven tested budgets and no independent test: do not infer population-wide significance automatically.

Read the 1,300 budget closely because previous D−B uplift was largest there (+2.44 pp, nominal 95% CI [0.22, 4.89]). **This is a post-hoc exploratory focus**, not an independent preregistered test.

## 4. Reproduce on NTU TC1 (CPU only)

Run from the already-configured server repository, on the existing P2 feature branch:

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git switch feature/route-e-token-aware-20261103
git pull --ff-only origin feature/route-e-token-aware-20261103
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

python scripts/route_e_cost_only.py self-test
python scripts/route_e_cost_only.py run
~~~

This reuses existing P2 files:
- outputs/route_e/token_aware/full/validation_predictions.json
- results/route_e/token_aware/full/validation.json

It does not rerun the P2 Ridge training script or touch DeepSeek weights. Use the --overwrite flag only if re-running **new Experiment 1** results, not historical P2.

New output directory: results/route_e/cost_only/experiment1/
- comparison.json: full source hashes, seven budgets, strict B/D reproduction, three paired bootstrap contrasts and per-dataset diagnostics
- equal_budget.csv: B, E, D and historical baselines by budget and dataset
- paired_bootstrap.csv: D−E, E−B, D−B and confidence intervals
- cost_diagnostics_by_dataset.csv: cost prediction rank/error/bias within each dataset
- accuracy_token_comparison.svg/png: optional plot if matplotlib is installed

Successful log markers:
- COST_ONLY_SELFTEST_PASS
- COST_ONLY_REPRODUCTION_PASS
- COST_ONLY_GROUP overall/gsm8k/math
- COST_ONLY_BUDGET 900...2100
- COST_ONLY_DONE

## 5. Interpretation guardrails

- Within-dataset cost rank correlations tell us whether the cost signal distinguishes questions inside groups. A substantial within-GSM8K and within-MATH Spearman supports fine-grained cost prediction, but does not causally prove what the model encodes.
- The existing P2's prospective budget-control error (e.g. target 2,100 vs realized ~1,692) is a separate problem. **Experiment 1 intentionally does not calibrate lengths** or make deployment claims.
- A negligible D−E difference is not proof the quality head is useless generally; it is evidence under **this frozen representation, fitted heads, lambda and offline replay protocol**.
- Report all seven budgets and negative results; do not tune a score using the sealed test set.
