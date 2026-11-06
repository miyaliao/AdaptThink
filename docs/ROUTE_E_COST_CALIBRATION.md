# Route E / P2 — Experiment 2: Cost Calibration

**Status:** implementation and CPU synthetic tests; NTU strict full-validation execution pending.
**Date:** 2026-11-03. Parent protocol: [ROUTE_E_TOKEN_AWARE.md](ROUTE_E_TOKEN_AWARE.md).

## 1. Scientific question

Original P2 uses frozen DeepSeek-P1 Ridge-head predictions for the router score and pre-generation budgeting. The **predicted budget** of 2,100 tokens led to only **1,691.8 actual tokens**, showing systematic budget underutilization. Is this corrected by a lightweight, **train-only** calibration of the lengths, without changing the routing score or using validation outcomes to choose a correction?

## 2. Exactly what stays fixed

- Original 3,000/450 strict SHA256-gated label protocol and P1 1,536-dimensional cached representations.
- Original saved **full-trained** P2 Ridge heads from outputs/route_e/token_aware/full/ridge_heads.npz, reproduced on 450 validation features; lambda and original D predicted-score values are verified for every sample.
- Original combined D score: S = predicted_delta_p − 0.05 * predicted_delta_tokens / 1000.
- **Do not change that score or its question ordering** when calibrating length estimates: this experiment isolates **budget control**, not routing ranking quality.
- Identical 900/1100/1300/1500/1700/1900/2100 *requested* average generated-token budgets.
- No LLM parameter updates, GPU extraction, new responses, or sealed test reads.

## 3. What changes

Five-fold out-of-fold (OOF) prediction on **only the original strict TRAIN rows**:

1. Deterministically shuffle the 3,000 train examples with seed 42 into 5 folds.
2. In each fold, fit the same Ridge heads on other 2,400 train examples, then predict Thinking and NoThinking token costs for the held-out 600. Every training row is predicted by a model that **did not train on it**.
3. Fit two positive multiplicative calibration constants from all 3,000 OOF predicted vs original train actual means:

   c_T = mean(train actual Thinking tokens) / mean(train OOF predicted Thinking tokens)

   c_N = mean(train actual NoThinking tokens) / mean(train OOF predicted NoThinking tokens).

4. Apply c_T / c_N to the **original existing full-trained heads'** per-mode validation predicted lengths and clip to [0, 16384].
5. Keep the original P2 router scores fixed. Compare original vs calibrated length forecasts as inputs **only to the predicted-budget threshold/mixture selector**. After fixing these policies, use observed validation outcomes only to measure realized cost, budget error, accuracy and dataset subgroups.

All calibration coefficients are therefore learned without a single validation answer or output length. Auxiliary five-fold Ridge models are trained **only to obtain OOF calibration predictions**; the saved full-trained P2 head used for routing is never replaced.

**Important limitation:** success-bucket-stratified TRAIN mean lengths (~8,847 Thinking / 2,865 NoThinking) differ drastically from validation (~2,209/568). Even perfectly held-out TRAIN OOF mean calibration may **fail or worsen** the out-of-distribution validation cost bias. This is an informative negative result, not grounds to use the same 450 validation labels to tune c_T, c_N.

## 4. Measures and interpretation

Primary metric: absolute error between the requested and **actual** average generated Token budgets. Secondary: real mean generated tokens, accuracy at requested budget, reachable/unreachable status under predicted costs, per-mode length MAE/RMSE/Spearman on TRAIN OOF and validation. Report all budgets, including failures, rather than choosing a favorable budget afterward.

**Two necessary caveats:**

- Accuracy at the same *requested* predicted budget is not automatically accuracy at the same *realized* generated-token cost if one arm uses more tokens. Increased accuracy driven by using more tokens must be described as such.
- The predicted-budget threshold is calibrated on unlabeled validation features in a batch/transductive manner, matching P2's earlier prospective diagnostic. It is not yet an online one-question-at-a-time deployment policy. Independent representative calibration of a fixed threshold would be required for real deployment.

## 5. Server execution (CPU, no GPU)

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git switch feature/route-e-token-aware-20261103
git pull --ff-only origin feature/route-e-token-aware-20261103
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
python scripts/route_e_cost_calibration.py self-test
python scripts/route_e_cost_calibration.py run
~~~

Reads the existing full P2 heads/predictions/strict caches. New isolated results:

- results/route_e/cost_calibration/experiment2/comparison.json
- results/route_e/cost_calibration/experiment2/budget_comparison.csv
- results/route_e/cost_calibration/experiment2/length_metrics.csv
- results/route_e/cost_calibration/experiment2/budget_calibration.svg/png (if Matplotlib available)
- outputs/route_e/cost_calibration/experiment2/train_oof_calibrator.npz (ignored by Git)

Log markers: CALIB_SELFTEST_PASS, CALIB_SOURCE_PASS, CALIB_OOF_FOLD, CALIB_FACTORS, CALIB_METRICS, CALIB_BUDGET and CALIB_DONE. All existing Route A/P1/P2 and Experiment 1 artifacts stay unchanged.

Do not merge final calibration claims into main until the actual full strict-validation budget and error results are reviewed.
