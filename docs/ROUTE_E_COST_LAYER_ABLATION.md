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
