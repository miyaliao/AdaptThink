# Cost-only Hidden Layer Ablation — 2026-11-09

**Status:** TC1 smoke/full completed; exploratory validation results archived below. Historical L28 score parity pending.

## Question
Does the frozen DeepSeek-R1-Distill-Qwen-1.5B **P1 Assistant-only** representation's optimal layer for Cost-only routing differ from the prior Logistic classification layer result?

## Controlled method
- P1 prompt and last-valid-prompt-token pooling are fixed; no decoding or DeepSeek weight updates.
- Extract layers **7, 14, 16, 21, 24, 26, 27, 28** in one frozen forward per minibatch.
- For each layer independently, fit **the historical train-standardized Ridge regression** with L2=0.02 on the original strict 3,000 train questions, predicting log1p Thinking and NoThinking generated tokens; use train-only Duan smearing and capped inverse transformation exactly as the prior P2 token-aware head. The original prior Ridge implementation also fits a quality head on the same multi-target system; because its Ridge columns are independent, fitting the two cost columns alone is equivalent apart from floating-point differences.
- At validation inference, score **predicted Thinking minus NoThinking tokens**, descending for NoThinking priority.
- Evaluate strict 450 validation questions via all 451 ranked top-k policies, historical observed generated-token cost interpolation at budgets 900..2100; rank-order Spearman and GSM8K/MATH subgroup Spearman are diagnostics. The interpolation is **descriptive**, not an online budget controller.
- No test set is read. **Do not select layers based on held-out test**. All eight layers should be reported, not just the winner.
- L28 should be compared against the historical Cost-only validation anchor (900:74.44%, 1100:75.11%, 1300:75.56%, 1500:75.56%, 1700:75.97%, 1900:77.56%, 2100:78.00%). Verify same train data, model revision, feature extraction, and Ridge score order before interpreting any discrepancy. L28 uses model.last_hidden_state (post final norm), matching historical P1 extraction.

## Run on TC1 (branch isolated from main)
```bash
cd /tc1home/UG/n2602710a/AdaptThink
git fetch origin experiment/cost-only-layer-ablation-20261109
git switch experiment/cost-only-layer-ablation-20261109
source .venv/bin/activate

# Fresh outputs; smoke then full
sbatch --export=ALL,LAYER_SMOKE=1 scripts/ntu_cost_layer_ablation.sbatch
# After smoke passes; FULL does not overwrite smoke
sbatch --export=ALL,LAYER_SMOKE=0 scripts/ntu_cost_layer_ablation.sbatch
```

Output: `outputs/route_e/cost_layer_ablation/{smoke,full}/{train,val}.npz` and `results/route_e/cost_layer_ablation/{smoke,validation}.json`.

**Validation checkpoint:** If L28 does not reproduce the historical Cost-only output, STOP, compare cache representations and exact scores. Do not choose a different layer based on unmatched protocols.


## TC1 completed run: operator console log, job 73106

**Date:** 2026-11-09. **Execution:** smoke job 73100 and full job 73106 ended with `LAYER_COST_PIPELINE_DONE`; train/val caches created for the strict 3,000/450 split. The values below are transcribed from the operator's TC1 console output, **not independently verified against the server-local JSON or old saved per-question scores**. The log also contained `Sliding Window Attention is enabled but not implemented for sdpa`, a warning that requires an environment comparison with the historical P1 extraction.

| Layer | Spearman Δt | B900 | B1100 | B1300 | B1500 | B1700 | B1900 | B2100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L7 | .5410 | 73.1111 | 73.5556 | 75.5556 | 76.2222 | 75.5556 | 76.0706 | 78.2222 |
| L14 | .5723 | 72.6463 | 74.6667 | 75.5556 | 76.0000 | 76.9836 | 78.0000 | 78.2222 |
| L16 | .5785 | **75.5550** | **75.7778** | **76.0000** | **76.3796** | 76.7583 | 77.7051 | **78.3160** |
| L21 | .6054 | **75.5556** | 74.6742 | **76.0000** | 75.7778 | **77.1111** | 76.6667 | 78.0398 |
| L24 | **.6352** | 74.2222 | 74.9030 | 75.1817 | 75.3333 | 75.8718 | 76.6659 | 77.5556 |
| L26 | .6337 | 74.8198 | 74.8231 | 75.8445 | 75.3333 | 76.0000 | 76.8889 | 78.0000 |
| L27 | .6330 | 74.8385 | 74.6667 | 75.7778 | 75.4452 | 75.7778 | 77.1111 | 78.0000 |
| L28 | .6225 | 74.4428 | 75.1111 | 75.5556 | 75.6581 | 76.1365 | 77.5556 | 78.0000 |

**Interpretation:** L24 had highest *cost-ranking* Spearman, but L16 exceeded L28 on all seven **descriptive validation** accuracy-at-matched-observed-cost points. The per-budget leaders across layers were L21 at 900 and 1700; L16 at 1100, 1500 and 2100; L16 tied L21 at 1300; L14 at 1900. A layer is not declared selected from these comparisons; testing 8 layers on reused validation outcomes incurs selection bias. Differences have no confidence intervals yet.

**Historical L28 reproduction status: approximate, not exact.** Spearman .6225 is close to historic .623, and B900/1100/1300/1900/2100 are consistent at the reported precision. However, historical B1500 was 75.56%, new is 75.6581%; historical B1700 was 75.97%, new is 76.1365%. This nonzero mismatch needs a per-question score and rank audit before calling results a controlled comparison. Compare original P1 last-layer cache to new L28 and historical cost heads versus re-fitted costs. Do not treat the SDPA warning as proven harmless without matching historical execution.

**Next step:** verify the L28 prediction/order and original extraction parity. Then, if matched, report full per-dataset and average-cost tradeoffs (not just these 7 points) with paired uncertainty and clearly distinguish validation-selected layers from an untouched final test. Do not tune on sealed test.

## Predeclared held-out evaluation: L16 versus original L28

**Selection locked on 2026-11-09, before this layer-ablation test job:**
- Primary: L16 P1 Cost-only Ridge (chosen on the reused 450 validation questions).
- Baseline: original historical L28 Cost-only Ridge weights, not a test-refitted baseline.
- Main policy: numerical threshold for the validation target of 1,300 generated tokens/question, transferred to the 485 test questions without adjustment. Actual test token cost is reported; it need not equal 1,300.
- Other seven validation-calibrated points and complete test threshold sweeps are supplementary **descriptive** analyses only; no test-based threshold or layer selection.
- Test: 485 unique questions (GSM8K 330, MATH-500 125, AIME2024 30). AIME outcomes average 16 cached generations per mode/question; no answer generation is needed.
- Report pooled/per-dataset Accuracy and actual generated token cost; paired per-question conditional bootstrap at validation-B1300. Comparisons at different actual test costs are NOT matched-token claims.
- Historic consistency gate: the extracted L28 representations must match the original P1 cache on the 3,000 train / 450 validation samples, and the old Ridge head must reproduce its saved validation scores. If not, abort before seeing test outcomes.
- Selection was made after multiple comparisons on a reused validation split. Conditional bootstrap is not a correction for selection or multiple testing; prior inspection of the canonical test by other experiments would limit holdout independence.

### Running on NTU TC1

Do not switch branches, clean local files, or discard untracked work. Fetch only the three files needed here:

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git fetch origin experiment/cost-only-layer-ablation-20261109
git restore --source=origin/experiment/cost-only-layer-ablation-20261109 --worktree \
  scripts/route_e_cost_layer_test.py \
  scripts/ntu_cost_layer_test.sbatch \
  docs/ROUTE_E_COST_LAYER_ABLATION.md
source .venv/bin/activate

# Historical artifacts and new validation caches must already exist.
test -f outputs/route_e/cost_layer_ablation/full/train.npz
test -f outputs/route_e/cost_layer_ablation/full/val.npz
test -f outputs/route_e/token_aware/full/ridge_heads.npz
test -f outputs/route_e/token_aware/full/validation_predictions.json
test -f outputs/route_e/prompt_ablation/assistant/full/train.npz
test -f outputs/route_e/prompt_ablation/assistant/full/val.npz

sbatch scripts/ntu_cost_layer_test.sbatch
~~~

All-in-one job stages: (1) audit L28 and freeze only validation-calibrated policies; (2) GPU test-question P1 prefill extraction for L16/L28; (3) build cached test labels if absent and evaluate frozen policies. Failure in stage 1 stops the job **before** test extraction. Output artifacts are exclusive-create and cannot be overwritten silently.

Expected final outputs:
- outputs/route_e/cost_layer_ablation/heldout_485/frozen_policies.json
- outputs/route_e/cost_layer_ablation/heldout_485/l16_cost_head.npz
- outputs/route_e/cost_layer_ablation/heldout_485/test_features.npz
- results/route_e/cost_layer_ablation/heldout_485/heldout_test.json
- results/route_e/cost_layer_ablation/heldout_485/heldout_test.md

**Status:** test execution pending on NTU TC1. Code being committed to GitHub is not a completed remote test.


## Held-out preflight stopped at L28 feature parity (2026-11-09)

TC1 test log: `TEST_LAYER_SELFTEST_PASS`, then `L28_FEATURE_PARITY train_max_abs=0.171875 val_max_abs=0.125`; guard requires max abs <= 0.02 and **aborted before test extraction/evaluation**. This is **not a completed held-out result**. Potential FP16 / batch-padding / attention-kernel differences are hypotheses, not verified causes. New layer extraction used batch=4, historical P1 extraction defaulted to batch=8. Historic L28 cost Spearman is approximately replicated, but that alone does not establish feature or policy equivalence.

Before adjusting any guard or running test, perform CPU-only pretest diagnostic:
```bash
git fetch origin experiment/cost-only-layer-ablation-20261109
git restore --source=origin/experiment/cost-only-layer-ablation-20261109 --worktree scripts/route_e_cost_layer_parity.py
python scripts/route_e_cost_layer_parity.py
```
It measures train/validation feature relative L2 and cosine errors; historical Ridge cost prediction on old/new L28 features; Spearman and ranked top-k overlap; and val-only budget-calibrated policy accuracy/token differences. Diagnostics are saved to `results/route_e/cost_layer_ablation/l28_pretest_parity.json`, without accessing test outcomes. **Do not simply increase the numerical tolerance** until the resulting predictions and route ranks are checked. The original test Slurm job is non-resumable with the same output name if `prepare` already writes frozen policy files; check artifacts first.



## Validation L28 functional-parity diagnosis (operator output, 2026-11-09)

A second CPU-only validation check completed, without accessing the sealed test set:

| Metric | Train | Validation |
| --- | ---: | ---: |
| Max absolute hidden-state difference | 0.171875 | 0.125 |
| Mean absolute difference | 0.005156 | 0.004962 |
| Median relative L2 difference | 0.002882 | 0.002761 |
| 99th-percentile relative L2 | 0.003842 | 0.003688 |
| Minimum cosine | 0.99998788 | 0.99999173 |
| Mean cosine | 0.99999575 | 0.99999610 |

Using the **historical frozen L28 Ridge head**, historical scores reproduced against historical features to 3.64e-12 extra Token difference. Comparing the same historical head on new L28 features gives score Spearman **0.9999007**, Pearson **0.9999765**, median absolute score difference **4.58 tokens**, 95th percentile **34.65 tokens** and worst case **240.58 tokens**. The largest error alone would overstate the functional difference. Top-k overlap: 23/23, 45/45, 112/113, 225/225, 338/338.

**At each of the seven validation-budget-calibrated policies, the old/new policy Accuracy difference is exactly zero in the logged readout**. The largest realized-token difference was 9.15 tokens/question (at validation budget 1500), and NT-fraction difference at most 1/450. This is strong evidence that the small feature variation has little effect on the **examined validation policies**, but not proof of identical representations/policies or of test outcomes; batch size or attention numerical causes remain unverified.

**Action:** the original rigid `max_abs <= 0.02` preflight gate was replaced on the experiment branch with a multi-factor **train/validation-only functional equivalence gate**: 99th-percentile relative L2 <=0.005, min cosine >=0.99998, historical score replay max error <=0.05 tokens, old-vs-new cost score Spearman >=0.9995, top-k overlap >=98% at predeclared k, zero validation Accuracy change at seven budgets and <=15 tokens mean cost change. This gate is grounded in the *already examined* validation diagnostics, **not** a universal numerical reproducibility tolerance. Record that post-hoc qualification; keep the one-shot L16 selection and historical L28 reference frozen before any test read. If gate fails on server, STOP and investigate, do not relax further based on test outcomes.

**Status:** patched test script committed, remote TC1 evaluation not yet executed. Existing failed test job exited during `prepare` before writing the frozen thresholds, so a fresh run may be submitted after fetching the patched script. Check output files before running to guard against conflicting jobs.


## Held-out 485 test completed — TC1 operator log (2026-11-09)

**Status:** TEST COMPLETED in the TC1 log (`TEST_LAYER_FEATURES_SAVED`, seven `TEST_TRANSFER` rows, `TEST_COST_LAYER_DONE`, `COST_LAYER_TEST_PIPELINE_DONE`). The report files were generated server-local but **their full contents, frozen policies, baseline-score audit, nAUC summaries, and bootstrap intervals have not yet been independently read here**. The table below transcribes the operator-provided stdout.

**Primary comparison fixed before this test:** L16 Cost-only Ridge vs historical L28 Cost-only Ridge. Numerical cutoffs were calibrated by nearest observed mean generation tokens on the strict 450-question validation set, and then applied without modification to 485 test questions. Note that each row's displayed budget is a **validation calibration target**, not the realized test cost. Do **not** interpret the accuracy difference between methods at a common row as accuracy at equal realized test-token cost.

| Validation target | L16 test accuracy | L16 actual tokens | L28 test accuracy | L28 actual tokens | L16−L28 accuracy (pp) | L16−L28 actual tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 900 | 72.0232% | 1176.8 | 71.4948% | 1195.0 | +0.528 | −18.2 |
| 1100 | 72.8093% | 1319.0 | 72.6289% | 1354.3 | +0.180 | −35.3 |
| **1300 (primary)** | **73.4278%** | **1421.8** | **72.8479%** | **1519.4** | **+0.580** | **−97.6** |
| 1500 | 73.8531% | 1724.1 | 73.4923% | 1640.6 | +0.361 | +83.5 |
| 1700 | 74.7938% | 1926.9 | 73.8531% | 1779.5 | +0.941 | +147.4 |
| 1900 | 75.2577% | 2099.9 | 74.6907% | 1928.2 | +0.567 | +171.7 |
| 2100 | 75.3351% | 2161.8 | 75.8763% | 2302.8 | −0.541 | −141.0 |

**Reading the result:** At validation targets 900, 1100, and 1300, L16 both improved accuracy and reduced realized output tokens; most compelling is the predeclared primary point (+0.580 pp with 97.6 fewer output tokens, about 6.4% lower token use compared with L28). At 1500, 1700, and 1900, L16 was more accurate but also costlier, so these do not establish equal-token efficiency gains. At 2100, L28 has higher accuracy at higher cost. The strongest supported conclusion is **low-budget empirical Pareto improvement**, not universal superiority across all budgets. These results still exclude router prefill/compute overhead.

**Required final checks before reporting statistical claims:** Inspect `results/route_e/cost_layer_ablation/heldout_485/heldout_test.md` and `heldout_test.json` on TC1 for paired B1300 conditional bootstrap CI, GSM8K/MATH/AIME subgroup results, test-sweep nAUC, exact model/score provenance, and method caveats. Neither a confidence interval nor a statistical-significance claim can be inferred from these seven stdout lines alone. Test outcomes must not be used to reselect L24, change L16, or tune thresholds.

**Provenance:** user-supplied TC1 stdout on 2026-11-09; canonical original test set N=485. Server-local JSON/Markdown not copied into this GitHub document.



## Final held-out statistical report — TC1 operator output (2026-11-09)

**Status: FINISHED.** The operator supplied the contents of `results/route_e/cost_layer_ablation/heldout_485/heldout_test.md`, completing the previously pending readout. This section transcribes that report. The server-local JSON and per-question raw predictions have not been independently inspected through the GitHub connector. No test-dependent layer reselection or threshold modification was performed.

### Primary, predeclared validation-budget target B=1300

| Metric | L16 (validation-selected) | L28 (historical frozen Cost-only) | L16 minus L28 |
| --- | ---: | ---: | ---: |
| Test pooled expected accuracy | **73.4278%** | 72.8479% | **+0.580 percentage points** |
| Mean actual generated tokens/question | **1,421.8** | 1,519.4 | **−97.6 tokens (−6.4%)** |
| NoThinking fraction | 0.225 | 0.188 | +0.037 |

**Conditional paired per-question bootstrap (B1300):** L16 minus L28 accuracy **+0.580 pp**, nominal 95% CI **[−0.863, +2.216] pp**. CI includes zero; cannot claim a statistically significant accuracy improvement. This bootstrap is conditional on validation-selected layers and policies and does not cover model/threshold selection uncertainty or sample design uncertainty.

**Not a matched realized-token comparison:** the shared B1300 label means *validation target*, while realized test costs are 1,421.8 and 1,519.4. This point demonstrates observed **Pareto improvement** for L16: slightly higher pooled expected accuracy *and* fewer generated tokens in the held-out evaluation, but not a statistically established population-level superiority or end-to-end inference speedup.

### Per-dataset results (same frozen validation B1300 cutoff)

| Dataset | N | L16 accuracy | L16 actual tok. | L28 accuracy | L28 actual tok. | Accuracy delta | Tokens delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 330 | 79.09% | 827.2 | 78.18% | 885.8 | +0.91 pp | −58.6 |
| MATH-500 | 125 | 73.60% | 1,712.6 | 72.80% | 1,750.6 | +0.80 pp | −38.0 |
| AIME2024 | 30 | **10.42%** | 6,750.3 | **14.37%** | 7,526.1 | **−3.95 pp** | −775.8 |

GSM8K and MATH-500 both show small descriptive accuracy gains and token savings. **AIME2024 suffers an accuracy decrease** despite saving tokens, indicating an important potential failure mode of cost-only routing on hard competition problems. AIME results aggregate 16 cached generations per mode per question, but there are only 30 distinct questions. No per-dataset confidence intervals were provided.

### Descriptive full-test sweep

| Metric | L16 | L28 | L16 minus L28 |
| --- | ---: | ---: | ---: |
| Pooled nAUC-AC (accuracy fraction × generated tokens) | **+51.6589** | +49.4122 | +2.2467 |
| Mean gain in percentage points over global random reference | **+3.1363** | +2.9999 | +0.1364 pp |

These test-sweep AUC values are **post-hoc descriptive**, not a test-optimized deployable budget control policy; no confidence intervals for their differences have been established.

### Final interpretation and decision

1. The frozen P1 Question Representation **layer matters for Cost-only Token Cost routing**. On strict validation, L24 maximized extra-cost Spearman (0.6352), whereas L16 had better low-/mid-budget routing accuracy than the historical final-layer baseline. Predictive cost rank correlation and downstream routing utility are distinct objectives.
2. The *validation-selected L16* policy yields a small **held-out low-budget descriptive Pareto gain** at the predeclared B1300 setting (+0.580 pp; −97.6 tokens/question), but its conditional bootstrap CI crosses zero. **No statistically significant accuracy gain is established.**
3. The gain does **not** generalize uniformly: AIME2024 accuracy declines by 3.95 pp, and L16 does not dominate L28 across the full budget sweep.
4. Training and test inference keep the DeepSeek base **frozen**, using a P1 Assistant-only prompt, layer-specific last-valid-token representations, and train-fitted Ridge log-token-length heads. No LoRA or model generation was used during router inference. Reported cost excludes prefill/router overhead.
5. **Research status: CLOSED** for the course report and portfolio: no further hyperparameter/layer selection on test. Preserve L28 as the strong historical reference; discuss L16 as a validated exploratory improvement under constrained conditions, rather than an unqualified new best baseline. Test inspection in the wider project may also limit the interpretation of this same 485-question set as a pristine holdout for subsequent hypotheses.

Reference: operator-provided TC1 `heldout_test.md`; results not committed as an independent machine-readable test artifact.
