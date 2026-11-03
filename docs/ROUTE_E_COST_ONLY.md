# Route E / P2 — Experiment 1: Cost-only ablation

**Status: strict 450-question validation replay completed; exploratory interpretation, further uncertainty and type-controlled analysis pending.**
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

## 6. Strict 450-question run: reported results (2026-11-03)

The NTU TC1 operator reported `COST_ONLY_SELFTEST_PASS` and `COST_ONLY_REPRODUCTION_PASS B/D unchanged strict dense replay`; canonical 450 rows and frozen saved-head predictions were used. The following are **strict validation exploratory dense actual-token-matched** values, not online deployment or held-out test numbers.

| Budget | B quality | E cost-only | D combined | D−E (pp) | Nominal paired-bootstrap 95% CI for D−E (pp) |
| ---: | ---: | ---: | ---: | ---: | --- |
| 900 | 70.00% | **74.44%** | 72.00% | −2.44 | [−5.55, +0.45] |
| 1,100 | 72.23% | **75.11%** | 73.35% | −1.76 | [−4.43, +0.71] |
| 1,300 | 73.33% | 75.56% | **75.78%** | +0.22 | [−2.22, +2.67] |
| 1,500 | 74.89% | 75.56% | **75.78%** | +0.22 | [−2.00, +2.22] |
| 1,700 | 76.89% | 75.97% | **77.33%** | +1.36 | [−0.19, +3.11] |
| 1,900 | 77.49% | 77.56% | **77.78%** | +0.22 | [−1.11, +1.56] |
| 2,100 | 78.00% | 78.00% | **78.05%** | +0.05 | [−0.53, +0.67] |

**Cost rank diagnostics:**

| Group | N | Predicted extra-cost Spearman | Δt MAE | Predicted Δt mean | Actual Δt mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overall | 450 | 0.623 | 1,531.8 | 2,165.3 | 1,641.4 |
| GSM8K | 300 | 0.351 | 861.3 | 972.2 | 616.0 |
| MATH | 150 | 0.474 | 2,872.9 | 4,551.5 | 3,692.1 |

### What these findings support and do not support

- **Strong descriptive low-budget signal:** at 900 and 1,100, cost-only E exceeds quality-only B by +4.44 and +2.88 percentage points, and exceeds D by +2.44 and +1.76 points, respectively. This is consistent with weak Δp prediction contaminating cost-first ranking at tight budgets, but is not proof of that mechanism.
- **Possible higher-budget complementary quality signal:** D exceeds E in the remaining five budgets, most notably +1.36 pp at 1,700. **Every D−E nominal paired 95% interval includes zero**, so the apparent superiority of D over E is uncertain.
- **Not just type classification:** within-group extra-cost Spearman is positive (0.351 GSM8K, 0.474 MATH), although substantially below the overall 0.623. Group-level cost differences clearly contribute; ranking information also exists within groups. To quantify how much *routing accuracy* comes from cross-group allocation versus within-group ranking, perform a type-controlled ablation on fixed evaluation-only strata and compare per-dataset routing fractions and outcomes.
- **Cost calibration is still inaccurate:** predicted Δt exceeds observed mean in both groups (GSM8K +356.2; MATH +859.4 tokens). This does not mean either group is properly calibrated for prospective target budgets.
- **Required next readout:** the original `comparison.json` holds E−B and D−B paired CIs and GSM8K/MATH budget-level accuracy and NoThinking fractions, which are **not in the printed log**. Inspect these before asserting statistical robustness or assigning the improvements to cross-dataset allocation.

These outcomes document the operator-provided console log; the full server-side comparison.json has not yet been independently checked here. The seven reported budget points are all from one strict validation set with no multiple-comparison adjustment. No sealed test access or model retraining should follow from these observations without a separate protocol.
