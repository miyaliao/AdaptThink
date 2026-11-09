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
