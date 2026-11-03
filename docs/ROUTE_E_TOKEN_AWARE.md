# Route E / P2 — Token-aware routing on Frozen DeepSeek P1

**Status: first full strict NTU validation run complete; statistical and deployability checks pending.**  
**Date: 2026-11-03** · Reference: [ROUTE_E_RESEARCH.md](ROUTE_E_RESEARCH.md) and [ROUTE_A_PLAN.md](ROUTE_A_PLAN.md).

## 1. Research question and fixed protocol

**Can predicting future generated-token costs from a frozen, pre-generation DeepSeek representation improve strict accuracy at equal generated-token cost?**

This phase uses the **existing frozen DeepSeek P1 (Assistant-only) final hidden state, 1,536 dimensions**, extracted from the raw strict training data. No DeepSeek weight updates, feature recomputation, new generations, or test-set access.

| Source | Path | Required SHA256 |
| --- | --- | --- |
| Strict train, 3,000 DeepScaleR | results/router_labels_1p5b/all_3000_labels.jsonl | 761dcbb6338de1a3c71def382089c83e8024e0e55af6334dd974963e8d7e0f2c |
| Strict validation, 450 GSM8K/MATH | results/val_450_labels.jsonl | 4c1e984d3744cc089dcc963a30af0ef3d1ed9808c4bfd245a97e48787d501461 |
| Cached train / val P1 | outputs/route_e/prompt_ablation/assistant/full/{train,val}.npz | Embedded strict label hashes plus exact sample ID/order |
| Frozen P1 classification scores | outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json | Verify 50% NT reproduction |
| Full FT ModernBERT scores | outputs/external_modernbert/classification/val_predictions.jsonl | p_nothinking_pred (higher = NT), verify 50% NT reproduction |

The original 3,000 training labels and cached feature matrices are **not tracked in Git**. Restore them on NTU TC1 first. Validation strict scoring uses p_nothinking_strict if present; otherwise the original strict p_nothinking. Repaired/compliant labels must not replace the original train file. The frozen 485-example test set is sealed throughout development.

The train examples were success-rate stratified (approximately 8,847 Thinking tokens and 2,865 NoThinking), while validation averages are roughly 2,209/568. This is a major **covariate/label-shift risk** for token regression. Training outcomes may also be noisy, especially when NoThinking has few samples.

## 2. Heads and controlled comparison

A **single shared standardized P1 feature matrix** is used to fit three lightweight Ridge regressions, with **training-only feature statistics** and fixed regularization.

- Quality head: empirical \(\Delta p = p_T - p_N\) (strict success rates), predicted/clipped to [-1,1].
- Cost heads: \(\log(1+t_T)\) and \(\log(1+t_N)\), transformed back to token scale with a **training-only smearing factor** and clipped to [0,16384].
- The cost difference is \(\widehat{\Delta t}=\hat t_T-\hat t_N\). Do **not** interpret the existing Logistic classification score as either mode's success probability.

Default \(\ell_2=0.02\) is a preregistered simplicity choice, not chosen by validation accuracy. Default \(\lambda=0.05\) is similarly a diagnostic starting point; there is **no claim it is optimal**. Full quality/cost labels are used for training and post-hoc metrics only, never as validation-time router inputs.

| Arm | Routing score (higher = Thinking preference) | What differs? |
| --- | --- | --- |
| **B Delta-only** | \(\widehat{\Delta p}\) | Quality head only |
| **C Constant cost** | \(\widehat{\Delta p}-\lambda\bar{\Delta t}_{train}/1000\) | Same quality head, fixed train-only scalar |
| **D Predicted cost** | \(\widehat{\Delta p}-\lambda\widehat{\Delta t}(x)/1000\) | **Same** quality head + input-dependent predicted cost |
| Reference | P1 Frozen Logistic, FT ModernBERT Classification, Global Random | Existing controlled baselines |

**Important:** C produces precisely the same **question ranking** as B. Under top-k / dense ranking at a given mean budget they must yield identical results. C may change binary decisions if an unadjusted zero threshold is used; that is not evidence of a more informative ranking.

The \(\lambda\) grid is predeclared: 0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0. **Do not choose the best \(\lambda\) by looking at validation accuracy** or any sealed test result. To select a production \(\lambda\), use an independent development/calibration set and report the frozen policy separately.

## 3. Two separate, clearly labeled evaluation modes

### A. Exploratory dense actual-cost replay (comparability with Route A/E)

For each frozen routing score, order the **same 450 questions** and evaluate top-k NT prefixes k=0..450. Observed strict mode outputs and token lengths are used **after** ordering to calculate the measured cost/accuracy. At reporting budgets **900, 1100, 1300, 1500, 1700, 1900, 2100**, the evaluator mixes neighboring policies in **sorted observed-token cost space**. It does **not** optimize validation accuracy when choosing neighbors.

This deliberately reproduces Route E's *descriptive* cost-matched convention. The interpolation **uses validation observed costs** and therefore **is not a deployable budget policy**. Never claim that an online router knows its future output length. All plotted costs include output generation only, not encoder prefill/head overhead.

The report additionally provides GSM8K and MATH subgroup outcome/cost breakdowns, and paired bootstrap 95% intervals for D minus B, P1, FT, and Global Random. Intervals **condition on the fitted models and their selected mixed policies**; they do not account for retraining or hyperparameter-selection uncertainty. A subgroup can have a different realized budget from the global budget.

The script fails if the provided P1 or FT score file does not reproduce the historical 50%-NT result. If the FT per-question file is absent, a clearly marked **six-anchor approximate FT reference** is included; it is not silently described as dense.

### B. Predicted-cost-only prospective policy (deployability diagnostic)

A **separate** batch-level policy uses only:
1. P1 question features and the fitted Delta / token-cost heads,
2. the fixed token-aware score,
3. predicted per-mode output lengths.

It chooses the threshold/adjacent top-k policy mixture based on **predicted cost only**. Only afterwards does it compute actual strict accuracy and actual tokens, and report **actual minus predicted budget error**.

The current example calibrates the batch threshold on **unlabeled validation inputs**; it is *transductive* and should not be presented as a production guarantee. A deployed threshold must be calibrated on a separate representative unlabeled workload or using only pre-established model parameters. If the length head predicts a reachable cost range that excludes the requested budget, the report marks the budget **unreachable** rather than inspecting observed validation lengths to force a match. This possibility is particularly relevant given the train/val length shift.

A second diagnostic applies exactly \(S(x)>0\) (Thinking) vs \(S(x)\le0\) (NoThinking) for each preregistered \(\lambda\), then reports predicted cost, realized cost, and NT share. This requires no target-budget selection.

## 4. Run on NTU TC1

All steps **use cached features and CPU linear algebra**; P2 does not need another DeepSeek GPU forward pass. Activate the already configured Python environment.

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git fetch origin
git switch feature/route-e-token-aware-20261103
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

# Independent no-cache self-test (may run on a login node)
python scripts/route_e_token_aware.py self-test

# Verify original 3000/450 strict labels and both 1536-D P1 caches
python scripts/route_e_token_aware.py check

# Small CPU smoke (requires existing assistant/smoke feature caches)
python scripts/route_e_token_aware.py check --smoke
python scripts/route_e_token_aware.py run --smoke --no-plot

# Full 3000+450 experiment: no DeepSeek GPU inference required
python scripts/route_e_token_aware.py run

# Alternate Slurm entrypoint (no GPU requested)
sbatch --export=ALL,P2_STAGE=smoke scripts/ntu_route_e_token_aware.sbatch
sbatch --export=ALL,P2_STAGE=full scripts/ntu_route_e_token_aware.sbatch
~~~

If only the full cache exists, the *smoke* command cannot invent a 32/24 cache and will fail closed; run the independent self-test, full-cache check and full fit instead. If rerunning **this P2 stage only**, add --overwrite for the head output; it never overwrites Frozen P1 or Route A outputs.

### Produced files

| Path | Contents |
| --- | --- |
| outputs/route_e/token_aware/full/ridge_heads.npz | Train-standardized Ridge weights/intercepts + smearing factors |
| outputs/route_e/token_aware/full/validation_predictions.json | Predictions for every strict validation ID, **without observed outcome/length fields** |
| results/route_e/token_aware/full/validation.json | SHA, metadata, MAE/RMSE/Spearman, dense replay, bootstrap and prospective budgets |
| results/route_e/token_aware/full/budget_comparison.csv | Overall, GSM8K, MATH strict accuracy / actual tokens / NT share |
| results/route_e/token_aware/full/lambda_grid.csv | Zero-threshold lambda sweep, predicted vs realized budget |
| results/route_e/token_aware/full/accuracy_token_curve.svg/png | Optional Matplotlib figure |

Save the validation artifacts to Git **only after checking the actual outputs and verifying they are not stale**. P2 outputs are isolated from historical Route E/Route A artifacts. Head and prefill compute overhead are recorded separately from generated tokens.

## 5. Interpretation / go-no-go

1. **D beats B under paired cost matching, without severe predicted-budget bias:** evidence for input-dependent token-cost information in the frozen representation.
2. **B improves, D does not:** evaluate token regression MAE, bias, Spearman and GSM8K/MATH shift before making the model more complex.
3. **B and D underperform P1 Logistic:** a useful negative result. Check label noise, ranking structure and distribution mismatch before adding additional losses or finetuning the backbone.

Do **not** select a conclusion until the NTU full-validation report has been executed and inspected. No code in this stage modifies DeepSeek, reruns generation, or accesses the sealed test set.

## 6. First full NTU validation run (2026-11-03; preliminary)

The operator reported successful no-GPU P2 execution using **3,000 × 1,536 train** and **450 × 1,536 validation** frozen P1 feature caches with expected strict SHA256 checks. Full-FT ModernBERT reproduction passed at **73.5556% accuracy, 1,091.849 mean generated tokens**. P2 CPU head training took **0.24 s**, total script run **1.87 s** (excluding original DeepSeek feature extraction).

The numbers below are **exploratory measured-token cost-matched dense replay**, not predicted-budget deployment policies:

| Budget | Global Random | Delta-only B | Token-aware D | Frozen P1 | Full FT ModernBERT | D−B (pp) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 | 69.94% | 70.00% | 72.00% | 70.67% | 72.22% | +2.00 |
| 1,100 | 71.24% | 72.23% | 73.35% | 74.22% | 73.52% | +1.12 |
| 1,300 | 72.54% | 73.33% | 75.78% | 75.33% | 74.22% | +2.45 |
| 1,500 | 73.84% | 74.89% | 75.78% | 76.44% | 75.56% | +0.89 |
| 1,700 | 75.14% | 76.89% | 77.33% | 76.89% | 75.78% | +0.44 |
| 1,900 | 76.44% | 77.49% | 77.78% | 76.67% | 76.63% | +0.29 |
| 2,100 | 77.74% | 78.00% | 78.05% | 77.78% | 77.56% | +0.05 |

**Preliminary interpretation:** D improves on its identical-quality-head control B at all seven budgets, most strongly at 1,300 (+2.45 pp) and 900 (+2.00 pp). D leads the displayed methods at 1,300, 1,700, 1,900 and 2,100, but not at 900, 1,100 or 1,500. These are descriptive observations only, not a significance test.

**Outstanding checks:**
1. Review validation length MAE, RMSE, Spearman, difference-cost predictions, and predicted vs observed length bias. Train length means ~8,847/2,865 vs validation ~2,209/568 imply substantial distribution shift.
2. Review paired-bootstrap 95% intervals for D minus Delta-only; avoid claiming significance without supported uncertainty.
3. **Re-run the prospective predicted-only budget policy** after commit `6a6669d`: an orientation bug in this *separate* policy previously routed the highest predicted Thinking-preference questions to NoThinking. The logged dense P2_BUDGET results were **unaffected**, because the dense score map correctly negates Thinking preference.
4. For deployable thresholds, reserve a representative held-out calibration workload, evaluate realized token-budget error without using target outcomes, and account for encoder-prefill overhead.

The precise validation.json with bootstrap and prospective-budget diagnostics has **not yet been independently reviewed**. Draft PR #4 remains open pending review.
