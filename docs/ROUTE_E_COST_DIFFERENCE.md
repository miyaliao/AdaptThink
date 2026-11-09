# Route E / Experiment F — Direct Cost-Difference Supervision

**Status:** implementation + cache-free synthetic tests; no TC1 training / strict validation run claimed.

## Research question

Can directly supervising the extra output-token cost of Thinking improve pre-generation routing, compared with separately regressing the token lengths of Thinking and NoThinking? This adapts ThinkSwitcher's **difference-supervision principle** to token cost; ThinkSwitcher itself supervised the **pass-rate difference**, not token cost.

## Controlled experimental arms

All arms use the **same original frozen DeepSeek P1 assistant-only last-valid-prompt-token, final-normalized 1,536-d representations** and the **same original strict 3,000 train / 450 validation** split. No backbone gradient, new decoding, or sealed test access.

| Arm | Fitting target | Model | NoThinking priority |
| --- | --- | --- | --- |
| A0 original Cost-only | Separate log1p(t_T), log1p(t_N) | **Exact existing P2 Ridge** at L2=.02, train-only smearing | t̂_T − t̂_N |
| A1 Direct Delta | (t_T − t_N)/1000 | One standardized Ridge, L2=.02 | Predicted Δt |
| A2 Cost Difference Margin | Two separate log1p costs **plus** token-space Δt penalty | Two-output linear head initialized from P2 Ridge, torch float64 LBFGS | t̂_T − t̂_N |

A2 objective:

```text
L_A2 = 0.5*[mean((yT_hat-yT)^2) + mean((yN_hat-yN)^2)]
     + 0.5*l2*(||wT||^2+||wN||^2)
     + beta*mean(((tT_hat-tN_hat)-(tT-tN))^2 / scale^2)
```

The length inverse transform **exactly matches** P2 `exp(clip(log_prediction))*smear - 1`, clipped to 0–16,384. The smearing factors are calculated **from the fold's A0 ridge training residuals and held fixed during A2 optimization** so A2's difference penalty and inference-time score use the same transform. No validation correction factors are fitted.

`beta=0` is the exact original ridge solution, **not a separate approximate optimization**. Other beta candidates: `.1, 1.0`. A2's beta selection uses only **3-fold out-of-fold training-data Spearman on the cost difference** (seed 42, stratified by train-only dataset/cost bins). Ties choose the smaller beta. Validation accuracy is never used to choose beta. Keep the full beta CV table, not only the winner.

## Data/provenance

Existing files required on TC1 (do not substitute repaired/compliant labels):

```text
results/router_labels_1p5b/all_3000_labels.jsonl
results/val_450_labels.jsonl
outputs/route_e/prompt_ablation/assistant/full/train.npz
outputs/route_e/prompt_ablation/assistant/full/val.npz
outputs/route_e/token_aware/full/validation_predictions.json
results/route_e/token_aware/full/validation.json
```

The script reuses `route_e_token_aware.read_protocol_and_cache` (SHA256, train/val IDs and feature checks) and `route_e_cost_only` (source score and B/D historical dense replay checks). It **hard-fails** if refitting A0 doesn't match the saved P2 cost scores. The source P2 artifacts are read-only.

## Run (from existing AdaptThink checkout)

These original dependencies live in `feature/route-e-token-aware-20261103`; if you operate on a newer branch, retain/merge the source scripts and caches before running.

```bash
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# Copy the Experiment F script from the supplied bundle to scripts/
python scripts/route_e_cost_difference.py self-test
python scripts/route_e_token_aware.py check
python scripts/route_e_cost_difference.py run
```

For an initial pipeline check without plots or bootstrap (still **uses the full strict 3000/450 data**):

```bash
python scripts/route_e_cost_difference.py run --no-plot --bootstrap-reps 0
```

`run` is intentionally fail-closed if Experiment F's output folder already exists. Use `--overwrite` **only** when intentionally rerunning this experiment's own files.

## Evaluation

The script reproduces the existing `dense_curve` and `linear_bracket` convention with 451 top-k NoThinking policies per score and the same seven budgets: 900, 1100, 1300, 1500, 1700, 1900, 2100 mean generated tokens/question. Matching uses **observed validation token lengths only after score ordering**, making this descriptive, **not** a deployable budget controller. True answer accuracy and cost never enter the score.

Primary: ThinkSwitcher-style **nAUC-AC** integrated over all 451 policy points after sorting by observed mean-token cost, subtracting the shared always-NT/always-T linear baseline. Secondary: 7-budget accuracy, uplift over budget-matched global random, NT share, per-dataset Δt MAE/RMSE/bias/Spearman, and conditional paired bootstrap 95% intervals (2,000 reps, seed 42) comparing A1−A0, A2−A0, A2−A1. A0 historical reported nAUC is about +38.3885 **on the already-reused validation split**; do not assume this result before source reproduction succeeds.

A separate prospective diagnostic picks budgets based on **shared A0 pre-generation predicted lengths** for all arms, isolating ordering changes rather than changes in length calibration; output includes actual costs measured afterwards. It calibrates on **unlabeled validation-batch features** (transductive demonstration), and is **not** evidence of deployable online budget control. A1's direct Δt model has no absolute per-mode length predictions, so this is explicitly a common-cost-head diagnostic, not an A1-native budget forecast.

Evaluation excludes backbone-prefill and router overhead. Results on the repeatedly inspected 450-question validation split are exploratory; preregister/train-only select policy and evaluate a sealed test only once subsequently. The original train/validation output-length distribution is shifted and may limit transfer.

## Outputs

```text
results/route_e/cost_difference/experiment_f/
    config.json
    cv_beta_selection.json
    trained_heads.npz
    validation_pre_generation_scores.json
    cost_diagnostics.csv
    equal_budget.csv
    paired_bootstrap.csv
    prospective_shared_a0_lengths.json
    summary.json
    accuracy_token_curve.svg / .png   # optional
```

## Integrity checks

- Only original strict hashes, frozen P1 feature caches and saved P2 prediction files are read.
- A0 source scores must match the original P2 Cost-only **per question** within absolute error 1e-5.
- A2 `beta=0` returns *identical* historical A0 log-cost Ridge parameters and scores.
- CV sees only train rows; validation cost/accuracy is used after the fixed score rankings for reporting.
- No held-out test access, generation, GPU model updates, or old output file overwrites.

## Interpretation rules

A1 > A0 supports a better single-difference prediction target; A2 > A0 supports an incremental nonlinear token-margin penalty. Stronger Δt Spearman without higher nAUC means better cost prediction did not yield a better route. Even if A2 improves, do **not** claim it improved quality prediction: A0/A1/A2 use **no quality head at routing time**.
