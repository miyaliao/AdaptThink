# AdaptThink — Held-out Test-485 Evaluation (3 Representative Routers)

This package is designed for the frozen `miyaliao/AdaptThink` repo and the existing NTU TC1 **single V100 GPU** environment. It was synthetically self-tested, but **has not run against the sealed test or the remote GPU**. No benchmark results are invented or bundled.

## Protocol

- Frozen test: `data/splits/adaptthink_original_baseline_seed42.json`, **485 unique questions**: 330 GSM8K + 125 MATH-500 + 30 AIME2024. AIME has 16 cached generations per mode/question. The cached strict grades already exist at `results/baseline_1p5b/`.
- Validation: original strict 450 questions (GSM8K 300 + MATH 150), SHA256 `4c1e984d3744cc089dcc963a30af0ef3d1ed9808c4bfd245a97e48787d501461`.
- Three **frozen** candidate routers: full FT ModernBERT Utility-Hard (lambda 0.02); frozen DeepSeek P1 logistic classification; frozen DeepSeek P1 Ridge cost-only (rank by predicted T minus NT output tokens).
- Never train, adjust labels, choose method, tune a head, or optimize a *test* threshold using test outcomes. The proposed main table **predeclares** a validation-calibrated target of **1,300 output tokens/question**; the test's actual cost is reported, even if it differs.
- Each router's descriptive **full test** sweep includes 486 top-k policies pooled, and separate 331/126/31 policy sweeps on GSM8K/MATH/AIME. These test-wide descriptive curves must **not** be described as deployable test-budget-selected policies.
- The three output figures are `test_threshold_overlay.png/svg`, `test_per_dataset_threshold_curves.png/svg`, `test_validation_budget_transfer.png/svg`.
- The table `test_per_dataset_table.csv` has Always NT, Always T, exact expected Random(50/50), Random(25/75), Random(10/90), and three routers at the fixed **validation** target. `Avg.` means **macro across the 3 datasets** (to match the ThinkSwitcher-style reference table); `Pooled` means **485-question weighted**.
- `nauc_ac_test.csv` gives dataset-level raw nAUC-AC, mean gain (percentage points) and pooled/macro summaries. Raw units are fraction × output tokens; `mean_gain_pp = 100 * nAUC_AC / (Always T tokens − Always NT tokens)`. Raw nAUC-AC is **not directly comparable across datasets** with different endpoint token ranges. The implementation follows the existing AdaptThink `compute_nauc_ac.py` observed-cost sorting, duplicate-cost averaging, and trapezoid convention.
- Only output-generation token costs are included. Router prefill/encoder overhead and inference latency are excluded and must be reported separately for end-to-end claims.

## Prepare

Pull the committed test-evaluation scripts from `main` on NTU TC1:

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git status --short
git pull --ff-only origin main
source .venv/bin/activate

python scripts/verify_baseline_split.py
python scripts/build_test_labels.py
python scripts/adaptthink_test_predict.py --stage self-test
python scripts/adaptthink_test_report.py --self-test
python scripts/adaptthink_test_predict.py --stage verify
```

**Required server-local artifacts that are not in GitHub `main`:**

| Artifact | Default location |
| --- | --- |
| Actual fine-tuned **Utility-Hard best checkpoint** (HF `AutoModelForSequenceClassification`) | `outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/best_model/` (verify actual location) |
| Its 450 validation predictions | `outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/validation_predictions.jsonl` |
| Frozen DeepSeek P1 logistic weights | `outputs/route_e/prompt_ablation/assistant/full/linear_probe.npz` |
| Frozen DeepSeek P1 450 features | `outputs/route_e/prompt_ablation/assistant/full/val.npz` |
| Frozen DeepSeek P1 450 validation scores | `outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json` |
| Ridge cost-only weights | `outputs/route_e/token_aware/full/ridge_heads.npz` |
| Ridge cost-only 450 validation scores | `outputs/route_e/token_aware/full/validation_predictions.json` |
| DeepSeek checkpoint | `model/DeepSeek-R1-Distill-Qwen-1.5B/` |

The **Cost-only training script** originally lived on the `feature/route-e-token-aware-20261103` branch; this package implements the exact frozen-head prediction algebra separately and **requires the already-trained `.npz`**. Do not retrain on the test set.

## Run (one GPU)

Check the actual Utility-Hard checkpoint directory before submission:

```bash
ls outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/
# If best_model/ exists and contains config.json + weights:
sbatch --export=ALL,UTILITY_MODEL=outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/best_model scripts/ntu_three_router_test.sbatch
squeue -u "$USER"
```

If the checkpoint is in a differently named directory, change `UTILITY_MODEL` to that directory. The code first **reproduces the historical validation predictions** (P1/COST on CPU, Utility-Hard on GPU) before accepting the corresponding test scores. Utility-Hard supports one-logit binary sigmoid and two-logit softmax; the validation comparison gates incorrect choices.

**Resume cautiously**: GPU stages refuse to overwrite prior test prediction files unless `--overwrite`; DeepSeek can reuse a matching cached test-feature file. If a stage fails, check the logs, then run the necessary stage alone. Avoid `--overwrite` until checking the existing cache provenance.

## Output files

```text
outputs/test_485/
    deepseek_p1_features.npz
    p1_predictions.json
    cost_predictions.json
    utility_predictions.jsonl
results/test_485_three_routers/
    test_threshold_overlay.{png,svg}
    test_per_dataset_threshold_curves.{png,svg}
    test_validation_budget_transfer.{png,svg}
    test_per_dataset_table.csv
    heldout_test_report.md
    nauc_ac_test.csv
    validation_budget_transfer.csv
    provenance.json
```

The final Markdown report contains a paper-style table for per-dataset Acc. and Tok. as well as pooled/macro nAUC-AC. If `ModernBERT` validation score reproduction fails, first inspect the original training code/checkpoint semantics; **do not bypass the gate and claim those scores are Utility-Hard**.

## Caveat about operational budgets

The headline 1,300-token policy chooses the **closest validation mean-token point** along a router's ordered top-k validation sweep, converts it into a frozen numeric prediction cutoff, and applies that cutoff unchanged to test. This is reproducible and avoids test-set tuning, but does not guarantee 1,300 realized tokens on test. We provide seven validation-calibrated policies at 900–2,100 alongside their actual test costs. The separate full test threshold curves show descriptive rankings, not threshold calibration.
