# Cost-only Hidden Layer Ablation — 2026-11-09

**Status:** code prepared; TC1 GPU extraction, validation results, and historical L28 reproduction NOT YET RUN.

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
