# Route A — External Router via ModernBERT

Status: **five Router objectives evaluated on validation as of 2026-11-02**. Dual MSE and Dual MSE + Margin (beta=1) completed in NTU Slurm job **71767** ([experiment details](EXTERNAL_ROUTER_MARGIN.md)); **Classification remains the strongest observed validation Router**. SC4001 deadline: **2026-11-16**.

## 1. Research question

Can a lightweight **external router**, seeing only the problem text *before* generation, improve the accuracy–output-token trade-off by selecting **Thinking** or **NoThinking** for a frozen `DeepSeek-R1-Distill-Qwen-1.5B`?

- Objective: maximize answer accuracy under a mean output-token budget.
- No DeepSeek weight updates; no generated answers, ground-truth answers, or actual future token counts used as router input.
- Compare routes at **the same measured mean output-token cost**, not merely the same NoThinking share.
- Router compute and latency overhead are **not included** in the tables below; report them separately before making an end-to-end efficiency claim.

## 2. Shared data and evaluation

| Item | Protocol |
| --- | --- |
| Router encoder | `answerdotai/ModernBERT-base`, full fine-tuning |
| Training | 3,000 DeepScaleR problems, `results/router_labels_1p5b/all_3000_labels.jsonl` |
| Training targets | Empirical success rates: Thinking `K >= 16`, NoThinking `K = 4` |
| Validation | 450 problems: GSM8K 300 + MATH-500 150, `results/val_450_labels.jsonl` |
| Unused held-out test | GSM8K 330 + MATH-500 125 + AIME 2024 30 (16 generations/problem) |
| Scoring | Strict `src/adapt_think_rm.py`; no repaired-label substitution |
| Offline evaluation | Replay cached paired Thinking/NoThinking generations; do not regenerate |
| Baselines | Always Thinking; Always NoThinking; budget-matched Global Random |

Always Thinking: **78.44%**, **2,209 tokens/problem**.  
Always NoThinking: **67.78%**, **568 tokens/problem**.

The observed uplift in the tables means **accuracy minus the expected accuracy of Global Random at the same measured mean output-token budget**, in percentage points (pp). It is **not** improvement over Always Thinking.

## 3. Five completed methods

| Method | Prediction / supervision | Training loss | Best checkpoint/epoch | Best validation loss | Finding |
| --- | --- | --- | --- | ---: | --- |
| **Classification** | Binary label `1[p_N >= p_T]` (ties → NoThinking) | Cross-Entropy | Epoch 2 | 0.7146 | **Strongest observed cost-matched uplift** |
| Dual Regression (BCE) | `[p_T, p_N]` | Soft-label BCE | Epoch 4 | 0.6727 | Limited uplift |
| Delta Regression | `delta = p_T - p_N` | MSE | Epoch 1 | 0.21987 | Below matched Random at all intermediate points measured |
| **Dual MSE** | `[p_T, p_N]`, sigmoid bounded | Dual MSE | `checkpoint-752` | 0.247712 | Positive uplift, particularly at 75% NoThinking |
| **Dual MSE + Margin** | `[p_T, p_N]` and their difference | Dual MSE + 1.0 × Margin MSE | `checkpoint-564` | 0.492873 | Improved some budget regions, not uniformly better than Dual MSE |

Dual MSE and Margin were trained with the original strict 3,000-question file, not the repaired 3,000 or compliant 2,777 versions. Both use seed 42, ModernBERT-base full fine-tuning, batch 16, LR `2e-5`, up to 5 epochs and the same 450-question strict validation replay. Their job `71767` finished **COMPLETED (0:0)** in **00:07:45**. The best checkpoint was selected on each model's own validation objective.

**Do not compare validation loss magnitudes across the rows:** the target and loss functions differ.

Delta run configuration: LR `2e-5`, batch `16`, max `5` epochs, seed `42`, FP16; best checkpoint `checkpoint-188`. Earlier Classification/Dual run settings should be checked against their server-side run metadata rather than assumed identical.

### Main comparison: routing 50% of validation questions to NoThinking

| Router | Accuracy | Mean output tokens | Uplift vs cost-matched Random |
| --- | ---: | ---: | ---: |
| **Classification** | **73.56%** | **1,092** | **+2.37 pp** |
| Dual Regression | 73.11% | 1,363 | +0.16 pp |
| Delta Regression | 72.00% | 1,610 | -2.55 pp |
| Dual MSE | 72.89% | 1,295 | +0.39 pp |
| Dual MSE + Margin (beta=1) | 72.89% | 1,180 | +1.13 pp |

**Important:** all five choose 50% of questions for NoThinking, but they select different questions and therefore **use different token budgets**. The uplift column uses each method's *own* budget-matched Random reference. Do not compare raw accuracy at 50% share as if the compute were equal.

### Full validation sweep

| Router | NoThinking share | Accuracy | Mean output tokens | Uplift vs budget-matched Random |
| --- | ---: | ---: | ---: | ---: |
| Classification | 10% | 77.11% | 1,959 | +0.29 pp |
| Classification | 25% | 75.56% | 1,542 | +1.45 pp |
| **Classification** | **50%** | **73.56%** | **1,092** | **+2.37 pp** |
| Classification | 75% | 70.22% | 835 | +0.70 pp |
| Dual Regression | 10% | 77.33% | 2,034 | +0.03 pp |
| Dual Regression | 25% | 75.33% | 1,747 | -0.11 pp |
| Dual Regression | 50% | 73.11% | 1,363 | +0.16 pp |
| Dual Regression | 75% | 70.44% | 934 | +0.29 pp |
| Delta Regression | 10% | 77.33% | 2,122 | -0.55 pp |
| Delta Regression | 25% | 76.22% | 1,977 | -0.72 pp |
| Delta Regression | 50% | 72.00% | 1,610 | -2.55 pp |
| Delta Regression | 75% | 70.44% | 1,088 | -0.72 pp |
| Dual MSE | 10% | 77.33% | 2,005 | +0.22 pp |
| Dual MSE | 25% | 75.11% | 1,677 | +0.13 pp |
| Dual MSE | 50% | 72.89% | 1,295 | +0.39 pp |
| **Dual MSE** | **75%** | **71.33%** | **874** | **+1.57 pp** |
| Dual MSE + Margin | 10% | 77.33% | 2,016 | +0.15 pp |
| Dual MSE + Margin | 25% | 75.33% | 1,620 | +0.72 pp |
| **Dual MSE + Margin** | **50%** | **72.89%** | **1,180** | **+1.13 pp** |
| Dual MSE + Margin | 75% | 70.22% | 832 | +0.73 pp |

NoThinking shares 0% and 100% recover the fixed-mode endpoints above. The tables round accuracy/tokens for readability; full-precision source values are in the NTU job logs and server-local `results/external_modernbert/{dual_mse,dual_margin}_val.json` files.

## 4. Interpretation and limits

1. **Classification is the current validation leader**, with a maximum observed uplift of **+2.37 pp** over budget-matched Global Random. This is a validation observation, **not a held-out test result**.
2. The old Dual BCE has little observed uplift; Delta MSE is below matched Random at every intermediate point measured. **Dual MSE and Dual MSE + Margin both improve over matched Random at the measured intermediate points**, but neither clearly dominates the existing Classification curve.
3. **Margin effect is budget-dependent.** At 50% NoThinking, Dual MSE and Margin have the **same 72.89% accuracy**, but Margin uses **1,180 vs 1,295 tokens** (about 8.9% fewer); matched-Random uplift rises from **+0.39 to +1.13 pp**. At 75% NoThinking, however, **Dual MSE is better** (+1.57 vs +0.73 pp), so the margin term does not establish a uniform improvement.
4. Classification preferentially routes many MATH questions to NoThinking. MATH Thinking responses are much longer, so some overall benefit may come from **dataset-level cost allocation**, not necessarily finer problem-level prediction. This has not been isolated.
5. The 3,000 training examples are stratified by Thinking success rate and differ from validation difficulty. NoThinking training targets use only four samples and are noisy; empirical ties are common. The strict grader counts format failures.
6. Validation has only one cached generation per mode/problem. Keep the test set untouched while choosing objectives, checkpoints, or thresholds.

## 5. Completed ablation and research decision

The margin experiment was completed **2026-11-02** under Slurm job **71767**:

```text
L_dual   = 0.5 * [(pT_hat - pT)^2 + (pN_hat - pN)^2]
L_margin = [(pT_hat - pN_hat) - (pT - pN)]^2
L_total  = L_dual + beta * L_margin; beta = 1.0
```

The **Dual MSE-only control** isolates the BCE → MSE change from the added margin term. At a common *50% routed NoThinking share*, Margin saves about 115 average output tokens per question while maintaining the same measured accuracy. But this does **not** establish dominance at a common mean-token budget, and at 75% NoThinking the control has a larger uplift. **The existing Classification model remains the most promising Route A approach observed on validation**, with a recorded +2.37 pp cost-matched Random uplift at 1,092 tokens.

**Decision:** keep all five approaches as baselines. Do not launch a large margin-β sweep now. Next investigate **frozen DeepSeek-R1-Distill-Qwen-1.5B prompt hidden-state embeddings with a lightweight classifier/probe**, and compare against ModernBERT on the **same original strict training/validation protocol**, avoiding the teammate's repaired/compliant data during this ablation. Report Router overhead separately. Neither the canonical test split nor any test labels should be used for tuning.

See [experiment details and job results](EXTERNAL_ROUTER_MARGIN.md), [training/evaluation script](../scripts/train_modernbert_margin.py), and [single-V100 Slurm script](../scripts/ntu_modernbert_margin.sbatch).

## 6. Code, artifacts, and reproducibility status

| Artifact | Path / status |
| --- | --- |
| Classification & Dual training | `scripts/train_modernbert_router.py` — **not found on GitHub main as of this review**; check NTU server local files |
| Classification & Dual evaluation | `scripts/eval_modernbert_router.py` — **not found on GitHub main as of this review** |
| Classification / Dual local outputs | `outputs/external_modernbert/{classification,dual}/` — described in experiment handoff, not verified in repository |
| Classification / Dual comparison | `results/external_modernbert/comparison_val.json` — **not found on GitHub main as of this review** |
| Delta training | `scripts/train_modernbert_delta.py` — committed |
| Delta Slurm job | `scripts/ntu_modernbert_delta.sbatch` — committed |
| Delta documentation | `docs/EXTERNAL_ROUTER_DELTA.md` — committed |
| Delta best model | `outputs/external_modernbert/delta/best_model/` — server-local artifact |
| Delta predictions | `outputs/external_modernbert/delta/validation_predictions.jsonl` — server-local artifact |
| Delta accuracy–token evaluation | `results/external_modernbert/delta_val.json`, `delta_val.csv` — reported server-local; not found on GitHub main as of this review |
| Delta implementation commit | [`b812deda`](https://github.com/miyaliao/AdaptThink/commit/b812deda186eac132b57b9fe3ebce9603135ad45) |
| Dual MSE + Margin training | `scripts/train_modernbert_margin.py`; `scripts/ntu_modernbert_margin.sbatch` — committed |
| Dual MSE saved outputs | `outputs/external_modernbert/dual_mse/` — server-local; best `checkpoint-752` |
| Dual MSE validation results | `results/external_modernbert/dual_mse_val.json`, `.csv` — server-local, Job 71767 |
| Dual Margin saved outputs | `outputs/external_modernbert/dual_margin/` — server-local; best `checkpoint-564` |
| Dual Margin validation results | `results/external_modernbert/dual_margin_val.json`, `.csv` — server-local, Job 71767 |

**Repo:** [miyaliao/AdaptThink](https://github.com/miyaliao/AdaptThink). **Server:** `/tc1home/UG/n2602710a/AdaptThink/`.

Next reproducibility action: check `git status` on the server and commit the original Classification/Dual source files plus compact validation metrics/metadata (avoid pushing large checkpoints). Ensure all five method baselines are fully reconstructible from documented inputs and scripts; compact Dual MSE/Margin result JSON/CSV artifacts are still server-local and should be committed separately if desired.

For the other project track, see [`ROUTE_B_PLAN.md`](ROUTE_B_PLAN.md).
