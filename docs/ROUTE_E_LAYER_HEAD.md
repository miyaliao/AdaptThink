# Route E / P1 — Frozen DeepSeek Hidden-Layer & Router-Head Ablation

**Protocol:** 2026-11-03. **Status:** implementation prepared; full TC1 results not yet run/verified. This document extends the completed [Route E study](ROUTE_E_RESEARCH.md) and **does not replace any historical results**.

## Research question and controlled design

Which transformer hidden layer and lightweight head produce the best *accuracy at the same expected mean generated-token budget*? Hold fixed the DeepSeek-R1-Distill-Qwen-1.5B checkpoint, strict train/validation labels, and P1 Assistant-only prefix. No DeepSeek parameter updates, generation, or held-out test access.

- **Layers:** `hidden_states[7,14,21,28]` from a **single forward pass** using `AutoModel(output_hidden_states=True, use_cache=False)`. The checkpoint must contain 28 transformer blocks. Indices 7/14/21 represent respective block outputs; index 28 is the **final RMSNorm** output (`last_hidden_state`). Intermediate states are *not* normalized by the final RMSNorm. The same per-feature train-fitted StandardScaler is used for every head.
- **Pooling:** final non-padding prompt token; right padding with `attention_mask`. Inputs built through the existing `route_e_prompt_ablation.tokenize_ablations(..., "assistant", ...)`, which checks the tokenizer's official template.
- **Feature consistency:** verify original P1 `{train,val}.npz`, sample IDs/order, strict hashes, model path, template hash, precision, max length, and batch size. Numerically compare new Layer 28 to original P1 final-layer arrays, then re-run its logistic head and require the original 50%-NoThinking policy accuracy/tokens to match.
- **Head 1:** existing frozen Route E Logistic Regression, BCE + L2 (`l2=1e-3`, CPU LBFGS), identical for all four layers.
- **Head 2:** small `Linear(1536,128) → GELU → Dropout(0.2) → Linear(128,1)`, BCE + AdamW. **Only** test one prespecified layer after reviewing the linear probe; do not sweep heads arbitrarily. The MLP chooses epoch using a stratified 80/20 *train-only* split, then refits for the chosen number of epochs on all original 3000 training questions. The fixed 450 validation labels do not control early stopping.

**Strict data invariants:** `results/router_labels_1p5b/all_3000_labels.jsonl` (original strict 3000, SHA256 `761dcbb6338de1a3c71def382089c83e8024e0e55af6334dd974963e8d7e0f2c`); `results/val_450_labels.jsonl` (GSM8K 300 + MATH 150, SHA256 `4c1e984d3744cc089dcc963a30af0ef3d1ed9808c4bfd245a97e48787d501461`). Class `1 = p_nothinking_strict >= p_thinking`, ties to NoThinking. The original train file is ignored in GitHub; *do not* substitute repaired/compliant labels.

## TC1 reproduction (one V100 GPU at a time)

Do not delete untracked original strict labels or old `outputs/route_e/prompt_ablation/assistant/full` caches. The new extraction *requires* the original P1 cache and never overwrites it. From the repository root:

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git status --short
git fetch origin
git switch feature/route-e-layer-head-20261103
source .venv/bin/activate
ls -lh outputs/route_e/prompt_ablation/assistant/{smoke,full}/{train,val}.npz
python scripts/route_e_frozen.py check-data
python scripts/route_e_layer_head.py self-test
python scripts/route_e_layer_head.py check-cache --smoke
python scripts/route_e_layer_head.py check-cache

# Smoke: 32 train / 24 validation; after completion check Slurm state/log.
sbatch --export=ALL,E_SMOKE=1 scripts/ntu_route_e_layer_head.sbatch
# Only after smoke passes; full: 3000 train / 450 validation.
sbatch --export=ALL,E_SMOKE=0 scripts/ntu_route_e_layer_head.sbatch

# Review existing P1 and four layer reports before choosing an MLP layer.
ls results/route_e/layer_head/logistic_L*_val.json
# Example ONLY: replace 14 with the selected L7 / L14 / L21 / L28.
sbatch --export=ALL,E_STAGE=mlp,E_SMOKE=0,E_LAYER=14 scripts/ntu_route_e_layer_head.sbatch
```

If the smoke/full job aborts halfway through, inspect new `outputs/route_e/layer_head/{smoke,full}/{train,val}.npz` before **explicitly removing only the failed new files**; extraction intentionally refuses overwriting existing artifacts. If CUDA OOM occurs, use `E_BATCH_SIZE=4` **only if the existing P1 cache was also made with batch size 4**, or separate a new well-documented representation configuration instead of claiming P1 identity. On full cache, Layer-28 accuracy/tokens at 50% NT must reproduce original P1. Log only cache checks / Slurm job completion as verified, not full-results claims before their actual run.

## Cost-matched validation and uncertainty

After all four logistic probes exist, evaluate all 451 top-k strategies, fixed budgets 900/1100/1300/1500/1700/1900/2100, and exact cost-matched Global Random. For **exact** Full ModernBERT Classification, pass its true 450-ID score file and the actual score-field/direction; *do not* substitute six aggregate anchors. Inspect JSONL schema before using a field:

```bash
head -n 1 outputs/external_modernbert/classification/val_predictions.jsonl
# Substitute FIELD / DIRECTION after inspecting the real file.
python scripts/route_e_layer_head.py compare \
  --ft-scores outputs/external_modernbert/classification/val_predictions.jsonl \
  --ft-field FIELD --ft-direction DIRECTION
# After completing MLP, repeat explicitly to refresh the comparison:
python scripts/route_e_layer_head.py compare \
  --ft-scores outputs/external_modernbert/classification/val_predictions.jsonl \
  --ft-field FIELD --ft-direction DIRECTION --overwrite-report
```

The code asserts the established Classification 50%-NT reproduction (331/450, ~1092 generated tokens) before including FT. If FT per-question scores are unavailable, omit `--ft-scores`, and clearly mark FT **not evaluated** rather than use approximate anchors. The compare step records the seven budgets, 50%-NT GSM8K/MATH breakdown, and **stratified paired bootstrap** intervals (default 500 draws) for layer/head minus P1, FT, and Random, plus a unified accuracy–token SVG/PNG if matplotlib is installed.

The bootstrap is **exploratory** because these 450 validation examples are also inspected when selecting a layer/head. Interpolation is based on the observed validation mean generation cost of adjacent policies, not on deployed per-question predictions of output length. All token budgets exclude encoder prefill and router scoring overhead. Further gain claims require fresh held-out evidence; **never choose hyperparameters on GSM8K 330 + MATH-500 125 + AIME-2024 30**.

### New artifact paths

- New feature matrices and head checkpoints: `outputs/route_e/layer_head/{smoke,full}/...` (**ignored**, do not commit)
- Four Logistic reports: `results/route_e/layer_head/logistic_L{07,14,21,28}_val.json`
- Optional selected MLP report: `results/route_e/layer_head/mlp_LXX_val.json`
- Seven-budget table + 451-policy curves and descriptive uncertainty: `results/route_e/layer_head/equal_budget.{json,csv}` and `accuracy_token_curve.{svg,png}`
- GPU/Slurm logs: `route_e_layer_head_JOBID.{out,err}` (ignore; inspect with `sacct -j JOBID --format=JobID,State,Elapsed,ExitCode`)

**Implementation commit is not an experimental result.** Only commit final reports/figures after TC1 checks, self-test, smoke and full runs pass; preserve prior Route E output and conclusions unchanged.
