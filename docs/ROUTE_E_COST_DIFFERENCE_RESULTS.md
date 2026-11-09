# Experiment F — Cost-Difference Supervision: Final Results and Closeout

**Status: CLOSED (2026-11-09).** Original **A0 Cost-only** is retained as AdaptThink's **preferred cost-aware external-router baseline**. A1 Direct Delta-Cost and A2 Margin-aware Cost did **not demonstrate a consistent improvement**; avoid further beta/threshold searches on the already-used validation/test splits.

> Evidence provenance: numeric results below are transcribed from the NTU TC1 operator logs shared on 2026-11-09. The complete server-local `summary.json`, `test_summary.json`, `test_cost_diagnostics.csv`, and `test_bootstrap.csv` were **not independently inspected or archived here**; claims needing them remain explicitly unverified.

## 1. Research question and exact controls

Inspired by ThinkSwitcher (which supervises **pass-rate margin**, not token-cost margin), we tested whether directly supervising the **incremental Thinking output-token cost** `Δt = t_T - t_N` improves prediction and downstream routing relative to independently estimating mode-specific token lengths.

| Arm | Train objective | Router score / NoThinking priority | Status |
| --- | --- | --- | --- |
| **A0 — Original Cost-only** | Frozen DeepSeek P1 (1,536-D) + separate Ridge on `log1p(t_T)`, `log1p(t_N)`, training-only smearing correction; L2=0.02 | `t̂_T - t̂_N` (largest first) | **Retained baseline** |
| **A1 — Direct Delta-Cost** | One standardized Ridge on `(t_T-t_N)/1000`; L2=0.02 | `Δt̂` (largest first) | No consistent net gain |
| **A2 — Margin-aware Cost** | A0 log-cost loss plus `β·MSE((Δt̂-Δt)/1000)`; β chosen by train-only CV | `t̂_T - t̂_N` | CV chose β=0, thus exactly A0 |

All arms use the original **3,000 DeepScaleR strict train**, frozen **DeepSeek-R1-Distill-Qwen-1.5B P1 Assistant-only last-final-norm hidden state**, and identical inference-time ranking. No new answer generations and no LRM fine-tuning. Strict validation: **450 questions** (GSM8K 300 + MATH 150); independent canonical test: **485 distinct questions** (GSM8K 330 + MATH-500 125 + AIME 2024 30). For AIME, 16 samples **per mode per question** are aggregated before macro-averaging across 485 questions.

## 2. Training and validation (450 examples)

- Synthetic self-test: `COST_DIFF_SELFTEST_PASS`.
- Original A0 source predictions reproduced: `COST_DIFF_A0_REPRO_PASS max_abs_diff=4.96e-10`.
- A2 train-only 3-fold CV, cost-difference OOF Spearman: **β=0: 0.4145; β=0.1: 0.3651; β=1: 0.3656**. The fixed selection procedure chose **β=0**. A2 equals A0, not an independently improved model.
- Seven budgets below are **offline observed-cost-matched replay**, not deployable token-budget guarantees.

| Validation mean output-token budget | A0 Cost-only accuracy | A1 Direct Delta accuracy | A2 selected accuracy |
| ---: | ---: | ---: | ---: |
| 900 | 74.44% | 74.47% | 74.44% |
| 1,100 | 75.11% | 76.00% | 75.11% |
| 1,300 | 75.56% | 75.33% | 75.56% |
| 1,500 | 75.56% | 75.56% | 75.56% |
| 1,700 | 75.97% | 77.07% | 75.97% |
| 1,900 | 77.56% | 77.11% | 77.56% |
| 2,100 | 78.00% | 77.78% | 78.00% |

| Validation metric | A0 | A1 |
| --- | ---: | ---: |
| Δt MAE (tokens) | **1,531.83** | 2,597.27 |
| Δt RMSE (tokens) | **2,686.58** | 3,416.35 |
| Δt Spearman, overall | **0.62294** | 0.52099 |
| Δt Spearman, GSM8K | **0.35128** | 0.27115 |
| Δt Spearman, MATH | **0.47423** | 0.43040 |
| Cost-sorted nAUC-AC | 38.3885 | **38.9911** |

A1's validation nAUC increase of **+0.6026** was small and accompanied by **worse cost prediction** across the reported diagnostics. Seven per-budget conditional paired-bootstrap 95% intervals for A1−A0 crossed zero **except** the 1,700-token point (+1.099 pp, nominal CI [+0.210, +2.210] pp); this single point is not multiplicity-adjusted evidence of general dominance.

## 3. Independent canonical test (485 unique questions)

### 3.1 Descriptive test-cost-matched sweep (NOT a deployable policy)

The Router uses **frozen pre-generation scores**, but **actual test output tokens** are used *after routing* to bracket and interpolate to each displayed mean cost. The cost-sorted nAUC-AC is computed from all **486 top-k policies** on test. It must be labeled **Descriptive Test-Sweep Analysis**.

| Mean generated tokens on test (offline) | A0 accuracy | A1 accuracy |
| ---: | ---: | ---: |
| 900 | 66.34% | 67.27% |
| 1,100 | **71.08%** | 68.61% |
| 1,300 | **71.97%** | 71.41% |
| 1,500 | 73.05% | 73.53% |
| 1,700 | 73.81% | 74.91% |
| 1,900 | 74.41% | 74.67% |
| 2,100 | 75.23% | 75.59% |
| **Test descriptive nAUC-AC** | **49.3104** | **49.3580** |

A1−A0 **test nAUC-AC = +0.0476**, essentially a tie on the full test cost–accuracy curve. nAUC values from validation and test have **different cost integration intervals** and are not directly comparable as absolute scores.

### 3.2 Validation-frozen numerical thresholds transferred to test

Below, the number before '/' is **test accuracy**, and the number after '/' is **realized mean test output tokens**. Thresholds and mixture weights were selected on validation and **not reselected on test**.

| Target budget selected on validation | A0 test acc / realized tokens | A1 test acc / realized tokens |
| ---: | ---: | ---: |
| 900 | **71.49% / 1,195** | 70.30% / 1,231 |
| 1,100 | **72.63% / 1,354** | 71.46% / 1,357 |
| 1,300 | 72.92% / 1,515 | **73.70% / 1,511** |
| 1,500 | 73.49% / 1,637 | **75.08% / 1,617** |
| 1,700 | 73.85% / 1,780 | 74.57% / 1,860 |
| 1,900 | 74.69% / 1,928 | 75.52% / 2,036 |
| 2,100 | 75.98% / 2,322 | 76.01% / 2,320 |

These are **not equal-cost comparisons**. A1 offers more accuracy for fewer tokens at validation targets 1,300 and 1,500; A0 offers more accuracy for fewer tokens at targets 900 and 1,100. Budget drift is substantial (e.g. validation target 900 yields 1,195 tokens for A0 on test). The reported results do **not** establish A1 as a uniformly better or reliably deployable cost controller.

## 4. Final decision

1. **Keep A0 — original Frozen P1 Cost-only** as the **preferred and frozen baseline for subsequent AdaptThink external-router comparisons**. This is a pragmatic research decision supported by stronger cost-difference prediction and no consistent downstream improvement from A1/A2, **not** a claim that A0 has the highest test accuracy at every budget.
2. **Close Experiment F as a mostly negative/conditional ablation**. Direct Δt regression slightly changed the ranking and occasionally helped, but did not meaningfully improve test nAUC-AC. ThinkSwitcher-inspired additional Cost Margin Loss was not selected by train-only CV (β=0).
3. **No further model/β/threshold tuning against these test outcomes**. Test-set paired-bootstrap CIs and test Δt diagnostics were generated by the runner, but **their numeric contents were not supplied** for this closeout: do not assert significance or test-prediction MAE/Spearman without inspecting those files.
4. All costs are **generated-output tokens**, excluding additional DeepSeek P1 prefill, Router/head compute and possible KV-cache overhead. Neither offline cost matching nor validation-frozen thresholds guarantee a production budget.

## 5. Reproducibility and data lineage

- [Original Cost-only ablation](https://github.com/miyaliao/AdaptThink/blob/feature/route-e-token-aware-20261103/docs/ROUTE_E_COST_ONLY.md)
- [Experiment F training/validation protocol](https://github.com/miyaliao/AdaptThink/blob/feature/route-e-cost-diff-test-20261109/docs/ROUTE_E_COST_DIFFERENCE.md) · [training/eval script](https://github.com/miyaliao/AdaptThink/blob/feature/route-e-cost-diff-test-20261109/scripts/route_e_cost_difference.py)
- [Independent test protocol](https://github.com/miyaliao/AdaptThink/blob/feature/route-e-cost-diff-test-20261109/docs/ROUTE_E_COST_DIFFERENCE_TEST.md) · [test evaluator](https://github.com/miyaliao/AdaptThink/blob/feature/route-e-cost-diff-test-20261109/scripts/route_e_cost_difference_test.py)
- Server-local, **not independently uploaded to GitHub**: `results/route_e/cost_difference/experiment_f/{summary.json,cost_diagnostics.csv,paired_bootstrap.csv}` and `results/route_e/cost_difference/test/{test_summary.json,test_cost_diagnostics.csv,test_bootstrap.csv}`.

**Closeout rule:** Preserve these recorded outcomes, do not choose post-hoc winning test budgets, and use A0 unchanged for further experiments.
