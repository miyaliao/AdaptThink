# Route E / P2 — Experiment 3: Frozen P1 Logistic Quality + Predicted Cost

**Status:** implementation and synthetic 450-row CI tests passed; real strict validation pending. 2026-11-03.
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
