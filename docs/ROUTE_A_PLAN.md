# Route A — External Router via ModernBERT

Status: three router objectives evaluated on validation as of **2026-11-02**. **Dual MSE and Dual MSE + Margin are implemented, awaiting GPU runs** ([run guide](EXTERNAL_ROUTER_MARGIN.md)). SC4001 deadline: **2026-11-16**.

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

## 3. Three completed methods

| Method | Prediction / supervision | Training loss | Best epoch | Best validation loss | Status |
| --- | --- | --- | ---: | ---: | --- |
| **Classification** | Binary label `1[p_N >= p_T]` (ties → NoThinking) | Cross-Entropy | 2 | 0.7146 | **Best observed cost-matched uplift** |
| **Dual Regression** | `[p_T, p_N]` | Soft-label BCE | 4 | 0.6727 | Limited uplift |
| **Delta Regression** | `delta = p_T - p_N` | MSE | 1 | 0.21987 | Below cost-matched Random at all measured intermediate points |

**Do not compare validation loss magnitudes across the rows:** the target and loss functions differ.

Delta run configuration: LR `2e-5`, batch `16`, max `5` epochs, seed `42`, FP16; best checkpoint `checkpoint-188`. Earlier Classification/Dual run settings should be checked against their server-side run metadata rather than assumed identical.

### Main comparison: routing 50% of validation questions to NoThinking

| Router | Accuracy | Mean output tokens | Uplift vs cost-matched Random |
| --- | ---: | ---: | ---: |
| **Classification** | **73.56%** | **1,092** | **+2.37 pp** |
| Dual Regression | 73.11% | 1,363 | +0.16 pp |
| Delta Regression | 72.00% | 1,610 | -2.55 pp |

**Important:** all three choose 50% of questions for NoThinking, but they select different questions and therefore **use different token budgets**. The uplift column uses each method's *own* budget-matched Random reference. Do not compare raw accuracy at 50% share as if the compute were equal.

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

NoThinking shares 0% and 100% recover the fixed-mode endpoints above.

## 4. Interpretation and limits

1. **Classification is the current validation leader**, with a maximum observed uplift of **+2.37 pp** over budget-matched Global Random. This is a validation observation, **not a held-out test result**.
2. Dual Regression has little observed uplift. Delta Regression underperforms budget-matched Random at every measured intermediate budget. Keep both as negative/weak baselines; do not claim regression objectives are generally ineffective.
3. Classification preferentially routes many MATH questions to NoThinking. MATH Thinking responses are much longer, so some overall benefit may come from **dataset-level cost allocation**, not necessarily finer problem-level prediction. This has not been isolated.
4. The 3,000 training examples are stratified by Thinking success rate and differ from validation difficulty. NoThinking training targets use only four samples and are noisy; empirical ties are common. The strict grader counts format failures.
5. Validation has only one cached generation per mode/problem. Keep the test set untouched while choosing objectives, checkpoints, or thresholds.

## 5. Next experiment: Dual MSE + Margin Loss

Keep ModernBERT-base and the same train/validation data. Predict two success rates, but explicitly supervise their **difference**:

```text
L_dual   = 0.5 * [(pT_hat - pT)^2 + (pN_hat - pN)^2]
L_margin = [(pT_hat - pN_hat) - (pT - pN)]^2
L_total  = L_dual + beta * L_margin
```

- Start with a fixed, documented `beta` (e.g., `1.0`) to avoid a large hyperparameter search.
- **Include a Dual MSE-only ablation** if possible: without it, any improvement over the old Dual BCE cannot be attributed specifically to margin supervision.
- Select checkpoints on validation only; compare accuracy–token curves at **matched budgets** against the three completed methods and Random.
- If the new objective does not improve over Classification, prioritize **frozen DeepSeek embeddings + a lightweight probe/router** rather than expanding the training set or using test data for tuning.

**Implemented, GPU results pending:** [training/evaluation script](../scripts/train_modernbert_margin.py), [single-V100 Slurm script](../scripts/ntu_modernbert_margin.sbatch), and [step-by-step run guide](EXTERNAL_ROUTER_MARGIN.md). Run both Dual MSE and Dual MSE + Margin; do not enter numbers into the result table until evaluated.

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

**Repo:** [miyaliao/AdaptThink](https://github.com/miyaliao/AdaptThink). **Server:** `/tc1home/UG/n2602710a/AdaptThink/`.

Next reproducibility action: check `git status` on the server and commit the original Classification/Dual source files plus compact validation metrics/metadata (avoid pushing large checkpoints). Ensure all three baselines are fully reconstructible from documented inputs and scripts.

For the other project track, see [`ROUTE_B_PLAN.md`](ROUTE_B_PLAN.md).
