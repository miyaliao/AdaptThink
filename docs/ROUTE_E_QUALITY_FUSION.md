# Route E / P2 — Experiment 3: Frozen P1 Logistic Quality + Predicted Cost

**Status:** first full strict 450-row validation replay completed; research conclusions exploratory. 2026-11-03.
Parent protocol: [ROUTE_E_TOKEN_AWARE.md](ROUTE_E_TOKEN_AWARE.md).

## Scientific question

The P2 Ridge delta-success Head had validation Spearman 0.096, while the P1 Frozen Logistic classification had stronger mode-preference ranking. Does changing ONLY the quality signal improve token-aware routing beyond Cost-only and the original Ridge+Cost combination?

## Controlled intervention

| Method | Thinking-preference score | Trained components |
| --- | --- | --- |
| E Cost-only | negative predicted extra Thinking tokens | Existing P2 Cost Heads |
| D Ridge + Cost | Ridge predicted Delta p − 0.05 × predicted Delta t / 1000 | Original P2 Ridge heads |
| **F Logistic + Cost** | train-scaled negative P1 Logistic logit − 0.05 × predicted Delta t / 1000 | Original P1 Logistic + original P2 Cost Heads |
| Frozen P1 Logistic-only | negative P1 class logit | Original P1 Logistic |

**Critical point:** P1 Logistic class probability is NOT a per-mode answer-success probability. We use a preference score, not an estimated Delta p. To align its contribution with Ridge Delta p, compute the mean/std of the NEGATIVE P1 Logistic raw logit on TRAIN only, then affine-transform it to match the already-fitted Ridge quality predictions' TRAIN mean/std. No validation success or output-token labels enter scale fitting. This is score-scale normalization, NOT probabilistic calibration. Lambda stays fixed at 0.05 and is operationally a score mixing coefficient, not an interpretable accuracy price.

## Fixed experimental protocol

- Frozen DeepSeek 1.5B Assistant-only P1 representations and existing train/validation caches, strict SHA256 and 3,000/450 IDs.
- Original P1 Linear Logistic model, unchanged Ridge Delta-quality Head, unchanged Thinking and NoThinking length Heads.
- Exact reproduction gates: old P2 B/D Dense equal-budget accuracy/cost, old P1 50% NT 76.44% and ~1,427 tokens, P1 per-question validation class predictions, P2 per-question predicted mode costs and quality scores.
- Seven equal observed-generation-token budgets: 900, 1100, 1300, 1500, 1700, 1900, 2100.
- Strict accuracy vs cost, subgroup GSM8K/MATH, conditional paired-bootstrap intervals for F−D, F−E, F−P1.
- Additional, distinct predicted-cost-only threshold diagnostic on unlabeled validation inputs; reported actual costs and accuracy are measured afterwards.
- Offline Dense matching uses validation true lengths after routing for interpolation and is NOT an online deployable controller.
- No new training, GPU inference, generation, test labels, validation-based coefficient selection, or overwrites of previous results.

## NTU commands

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git switch feature/route-e-token-aware-20261103
git pull --ff-only origin feature/route-e-token-aware-20261103
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
python scripts/route_e_quality_fusion.py self-test
python scripts/route_e_quality_fusion.py run
~~~

## Outputs

New isolated directory: results/route_e/quality_head/experiment3/

- comparison.json: full lineage and seven-budget comparison
- equal_budget.csv: E/D/F/P1 overall and per-dataset performance
- paired_bootstrap.csv: uncertainty for F−D, F−E, F−P1
- predicted_budget.csv: predicted-only threshold cost/accuracy
- validation_pre_generation_scores.json: strictly pre-generation scores
- equal_budget.svg/png: optional Matplotlib curve

Key log markers: QUALITY_FUSION_SELFTEST_PASS, QUALITY_FUSION_SOURCE_PASS, QUALITY_FUSION_TRAIN_SCALE, QUALITY_FUSION_BUDGET, QUALITY_FUSION_PREDICTED_BUDGET, QUALITY_FUSION_DONE.

Interpretation: F consistently beating D and E would support complementary quality/cost ranking information for these fixed heads. F matching P1 or E indicates weaker incremental cost or quality contribution respectively. All inferences remain strict validation-only and must acknowledge multiple comparisons and distribution shift.

## 6. First strict validation results (2026-11-03)

Operator log checks: `QUALITY_FUSION_SELFTEST_PASS`, `QUALITY_FUSION_SOURCE_PASS strict_val=450 lambda=0.0500 max_p1_score_error=7.73e-08`; train-only scale **0.074770**, offset **0.132026**, training Ridge predicted quality SD **0.194972**.

### Exploratory actual-token equal-budget accuracy

| Mean generated tokens | E Cost-only | D Ridge + Cost | F Logistic + Cost | Frozen P1 Logistic | F−D (pp) | Nominal 95% CI for F−D (pp) |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 900 | 74.44% | 72.00% | 74.44% | 70.67% | +2.44 | [+0.22, +4.67] |
| 1,100 | 75.11% | 73.35% | 74.89% | 74.22% | +1.54 | [−0.24, +3.53] |
| 1,300 | 75.56% | 75.78% | 75.11% | 75.33% | −0.67 | [−2.44, +1.11] |
| 1,500 | 75.56% | 75.78% | 75.36% | 76.44% | −0.42 | [−2.00, +1.17] |
| 1,700 | 75.97% | 77.33% | 76.67% | 76.89% | −0.67 | [−2.22, +0.89] |
| 1,900 | 77.56% | 77.78% | 77.11% | 76.67% | −0.67 | [−1.78, +0.22] |
| 2,100 | 78.00% | 78.05% | 77.99% | 77.78% | −0.05 | [−0.89, +0.78] |

F reproduces E's accuracy at 900, is slightly behind E at 1,100, and trails D from 1,300 onward. At only the 900-token point is F−D's nominal 95% paired interval wholly positive; this is exploratory and **not multiple-comparison adjusted**. Thus replacing the quality signal improves neither the full curve nor high-budget accuracy consistently.

### Predicted-only budget diagnostic: actual tokens / accuracy

| Requested tokens | E Cost-only | D Ridge + Cost | F Logistic + Cost |
| ---: | --- | --- | --- |
| 900 | 764.3 / 73.78% | 811.3 / 71.55% | 778.3 / 73.02% |
| 1,100 | 872.3 / 74.00% | 955.6 / 72.00% | 931.4 / 74.28% |
| 1,300 | 1,037.7 / 74.89% | 1,104.6 / 73.47% | 1,103.4 / 74.89% |
| 1,500 | 1,245.6 / 75.33% | 1,265.8 / 75.08% | 1,234.8 / 75.11% |
| 1,700 | 1,387.4 / 75.56% | 1,421.8 / 76.00% | 1,415.0 / 75.71% |
| 1,900 | 1,483.4 / 75.66% | 1,546.9 / 76.67% | 1,545.1 / 75.33% |
| 2,100 | 1,677.9 / 75.78% | 1,691.8 / 77.33% | 1,669.0 / 76.67% |

All methods miss the requested budget due to systematic predicted-cost calibration bias. Accuracy comparisons at identical *requested* budgets are **not** exactly cost-matched if realized tokens differ. Changing the quality score alone does not fix this limitation.

**Conclusion:** A previously strong classification preference signal is not automatically a stronger combined quality-cost signal. The train-only affine normalization is not success-rate calibration and a fixed mixing coefficient may be suboptimal. Preserve these observations as negative/conditional results; do not retune mixing weights on the same validation set and present the resulting improvement as held-out evidence. The server-side comparison.json with subgroup details is the canonical artifact; this summary transcribes operator logs.
