# Route E — Budget-Conditioned Head Fusion (BCHF)

**Status:** **COMPLETED AND CLOSED (2026-11-04)** — strict 450-question offline validation run and 8/8 unit tests passed; no stable improvement over existing routing baselines. No new model training, DeepSeek generation, or test-set access.

## 1. Research question and hypothesis

Does choosing the weighting of **NoThinking classification preference** and **predicted extra Thinking tokens** based on the **target average generated-token budget** improve the accuracy–token trade-off compared with an equally tuned, budget-independent fixed weighting?

- **H1 — complementarity:** An interior fixed fusion weight can outperform either isolated signal at some budgets.
- **H2 — budget dependence:** A budget-conditioned `α(B)` can outperform a **development-selected best fixed α** on evaluation questions whose outcomes were not used to select their weights.
- **H3 — low overhead, real budget awareness:** Predicted-token thresholding can approximate requested budgets without looking at actual validation/test token lengths. Existing length bias may invalidate H3.

We must not claim that cost always dominates at low budgets or quality always dominates at high budgets. Existing validation curves are nonmonotonic. The results below are **not yet measured**; run the script to produce them.

## 2. Frozen input model and scores

No weights or cached features are changed. Load and cross-check:

| Resource | Required local path |
|---|---|
| Original 3,000 strict labels | `results/router_labels_1p5b/all_3000_labels.jsonl` |
| 450 strict validation labels | `results/val_450_labels.jsonl` |
| P1 Full Hidden States | `outputs/route_e/prompt_ablation/assistant/full/{train,val}.npz` |
| P1 Logistic Head | `outputs/route_e/prompt_ablation/assistant/full/linear_probe.npz` |
| P1 Logistic validation scores | `outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json` |
| P2 Ridge Heads (originally trained on P1) | `outputs/route_e/token_aware/full/ridge_heads.npz` |
| P2 saved validation predictions | `outputs/route_e/token_aware/full/validation_predictions.json` |

The P2 Heads/prediction files were produced by the earlier Token-aware experiment (`feature/route-e-token-aware-20261103`); the **script itself was not on `main` when this feature was authored**. The new BCHF script needs these existing **server-local outputs**, but **does not import the P2 training script**. If they are missing, first reproduce the P2 experiment using its documented branch, *without overwriting* any verified result.

All 3,000/450 strict SHA256 checks, P1 feature cache IDs/order and metadata, P1 validation probabilities, and stored P2 token predictions must match. The script fails on any discrepancy. It reproduces P1's 50% NoThinking result (331/450 ≈ 73.56% belongs to **Full FT ModernBERT**, not P1; **P1 is 76.44%, ≈1427 output tokens**).

### Score orientation and normalization

Let `z_N(x)` be the **raw NoThinking class logit** from Frozen P1 Logistic (not the probability of a correct NoThinking answer). Let `d_t(x) = (t̂_T(x)−t̂_N(x))/1000`, from the existing P2 Ridge Cost Heads. Fit **means and standard deviations on the original 3,000 training questions only**:

```
q(x) = (z_N(x) − mean_train(z_N)) / std_train(z_N)
c(x) = (d_t(x) − mean_train(d_t)) / std_train(d_t)
S_NT(x, B) = α(B) q(x) + (1−α(B)) c(x)
```

**Higher** `S_NT` = route to **NoThinking first**. `α=0` exactly recovers Cost-only ranking, `α=1` exactly recovers Frozen P1 Logistic-only ranking. Scores are preferences, **not calibrated expected quality gains**.

## 3. Controlled experiments

1. **Fixed-α sweep:** `α ∈ {0,.25,.5,.75,1}` across budgets `[900,1100,1300,1500,1700,1900,2100]` output tokens. Produce an exploratory heatmap and all five cost-matched curves. No weight selection claims from this full-450 sweep.
2. **5-fold development/held-out α selection:** deterministically split the existing validation IDs into five dataset-stratified folds. For each fold, choose α using **only the other four folds**. Compare:
   - **Best Fixed α:** one α selected across all seven budgets.
   - **Piecewise α(B):** one α for low (`≤1100`), medium (`1100<B≤1700`), and high (`>1700`) budgets. Selection maximizes mean development accuracy over the seven listed budgets with predetermined `0.0025` accuracy penalty for adjacent α jumps.
   - **Hard Switch:** same three budget groups, α restricted to `{0,1}`.
3. **OOF ranking:** each held-out question gets the α selected without its own outcome; combine scores from all five held-out folds and rank the full 450 questions once at each budget. No α selection uses its own fold's labels, although the overall system was designed after examining this validation dataset in earlier experiments.
4. **Reference routers:** Classification-only, Cost-only, **historical frozen Logistic+Cost and Ridge+Cost** fixed-lambda=0.05 priorities, plus dense Full FT ModernBERT Classification when its original score file is available (reproduction required; no approximate fallback). The new method's key comparison is against development-selected **Best Fixed α**, not a hand-picked weight.
5. **Diagnostics:** the score order is evaluated under two *separate* cutoff rules:
   - `observed`: historical descriptive dense replay using observed validation output lengths **after sorting** to exactly match expected mean output-token budgets. **Not deployable.**
   - `predicted`: choose cutoff using **only** saved predicted per-mode token lengths. Then compute realized accuracy, mean actual tokens, and NT share. If a budget is predicted unreachable, report missing; do not secretly use actual lengths to force it.

Paired bootstrap uncertainty is **conditional on the fitted, selected policies and cutoffs**, unadjusted for selecting multiple budgets/weights. Router prefill and inference overhead are not included. The final 485-question test remains untouched.

### Budget reachability and predicted-policy safeguards

A development fold does not necessarily span all seven full-validation token
budgets. The selector now marks any budget that is not reachable for **every**
candidate alpha as unavailable for that fold, and compares all candidates on
the *same* feasible subset. If a low/mid/high bucket has no reachable
development budget, its weight inherits the fold's best fixed alpha rather
than extrapolating. Per-fold `dev_available_budgets` and
`dev_unreachable_budgets` are archived with the result. A fold with no
jointly reachable budgets fails clearly; it is not assigned an invented score.

The predicted-token policy supplies a zero-valued placeholder accuracy to
the shared cost-interpolation helper **solely for threshold selection**.
Reported accuracy is recomputed from actual outcomes only *after* the
predicted-cost routing policy is frozen; actual output lengths are never
used to choose that threshold.

## 4. Run on NTU TC1 (CPU-only)

On a TC1 login or appropriate CPU Slurm allocation, use the existing repo environment:

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git status --short
git pull --ff-only origin main
source .venv/bin/activate

python scripts/route_e_budget_fusion.py self-test
python scripts/test_route_e_budget_fusion.py -v
python scripts/route_e_budget_fusion.py run
```

Use `--no-plot` if Matplotlib is absent. Use `--overwrite` only to replace **this experiment's** own results; original cached features, heads, and P2 reports are not modified. No GPU allocation or new generation is needed. Results are saved to `results/route_e/budget_fusion/`:

- `report.json` — strict provenance, full fixed sweep, OOF α selection, historical fixed-lambda baselines, per-budget and per-dataset metrics, and limitations.
- `fixed_alpha_sweep.csv` and `alpha_budget_heatmap.svg/png` — five α settings at seven budgets.
- `crossfit_summary.csv` and `crossfit_accuracy_token.svg/png` — OOF Best Fixed vs Piecewise vs Hard Switch and baselines, with both predicted-budget and descriptive observed-budget views.
- `frozen_policy.json` — a development-selected α(B) configuration, standardized by **train-only** statistics, reserved for a **future** independent evaluation. It is not a tested production policy.

Expected log: `BCHF_SELFTEST_PASS`, then `BCHF_BUDGET ...`, ending with `BCHF_DONE`.

## 5. What constitutes a meaningful result?

- **Evidence for H1:** several interior α settings beat both endpoints on meaningful budget regions; then confirm on independent data.
- **Evidence for H2:** held-out OOF budget-conditioned α exceeds OOF best-fixed α across a predeclared budget summary, with paired uncertainty. The 450-set is exploratory due to previous inspection; **a frozen independent test is necessary** for confirmatory claims.
- **Evidence against H2:** weights vary but held-out gains disappear, or Best Fixed α is consistently as good.
- **Deployment caveat:** prospective predicted-budget curves can miss their budgets even when offline observed-budget curves look good. Separate length calibration on representative data is needed before claiming reliable budget control.

### Next phase, not implemented here

A separate independent budget-calibration workload, frozen test-set scoring for the 485 examples, end-to-end P1 Prefill + Head latency, and group-controlled allocation analysis. Do **not** use test outcomes to select α(B), budget thresholds, or normalization statistics.

## 6. Completed strict-validation results and final decision (2026-11-04)

**Status: COMPLETED — negative/inconclusive ablation; BCHF is closed.** This section transcribes the operator-provided NTU TC1 console output and follow-up `report.json` readout. The full server-local JSON/CSV/plots were not uploaded or independently inspected here. No DeepSeek parameters, P1/P2 Heads, stored generations, or sealed test examples were changed or used.

**Verification:** `BCHF_SELFTEST_PASS`; **8/8 unit tests passed**; `FT_CLASSIFICATION_50PCT_PASS nt=225/450 accuracy=73.5556% tokens=1091.849`; the full CPU experiment ended with `BCHF_DONE`. Dataset: original strict **450 validation questions** (GSM8K 300, MATH-500 150), using the unchanged frozen P1 Logistic and P2 Cost Heads.

### 6.1 Budget-conditioned vs cross-fitted best fixed alpha

The **Fixed** column selects one alpha per four-fold *development subset* and applies it to that fold's held-out questions across budgets; it is **not** one globally constant alpha across the five held-out folds. **Adaptive** similarly selects low/medium/high alpha on each development subset. Their held-out routing scores are assembled across five folds, then evaluated by the historical **observed-cost matched** dense replay. Positive differences favor Adaptive.

| Mean output tokens / problem | Fixed (OOF) | Adaptive (OOF) | Adaptive − Fixed (pp) | Nominal paired 95% CI (pp) |
| ---: | ---: | ---: | ---: | --- |
| 900 | 71.33% | **73.11%** | +1.78 | [0.00, 3.78] |
| 1,100 | 72.22% | **73.81%** | +1.59 | [-0.44, 3.40] |
| 1,300 | 73.11% | **73.83%** | +0.72 | [-1.06, 2.67] |
| 1,500 | 74.67% | **75.11%** | +0.44 | [-1.12, 1.99] |
| 1,700 | 76.00% | 76.00% | +0.00 | [-1.33, 1.33] |
| 1,900 | **77.11%** | 76.00% | -1.11 | [-2.67, 0.44] |
| 2,100 | **78.00%** | 77.16% | -0.84 | [-1.68, -0.18] |

The 900-token interval touches zero; five other intervals through 1,900 include zero; at 2,100 the **nominal**, unadjusted interval is below zero. **These are descriptive, conditional intervals**, not multiple-testing-adjusted or full cross-fitting/retraining confidence intervals. The 450-question validation set had already been examined extensively while developing related Route E experiments, so even the held-out folds do **not** establish a new independent test result.

### 6.2 Selected alpha varies substantially across folds

`alpha=0` means **Cost-only**; `alpha=1` means **Classification-only**.

| Fold | Fixed alpha | Adaptive low | Adaptive medium | Adaptive high | Unreachable dev budget |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.75 | 0.50 | 1.00 | 1.00 | None |
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | None |
| 2 | 0.00 | 0.00 | 0.00 | 0.00 | None |
| 3 | 0.75 | 0.25 | 0.75 | 0.50 | None |
| 4 | 0.00 | 0.25 | 0.75 | 0.75 | 2,100 |

Folds 1–2 select Cost-only across all budgets while the other folds use nonzero Classification weights. Thus there is **no stable, reproducible budget-to-alpha relationship** in these data. Fold 4's unreachable 2,100-token *development* budget was excluded from the development selection as specified by the reachability safeguard; the full 450-question evaluation still reports 2,100 tokens.

### 6.3 Context against previously observed routing baselines

BCHF Adaptive did **not** improve on the strongest previously reported per-budget validation reference among the recorded Route A/E comparisons:

| Budget | BCHF Adaptive | Previously observed reference | Reference accuracy |
| ---: | ---: | --- | ---: |
| 900 | 73.11% | Cost-only | 74.44% |
| 1,100 | 73.81% | Cost-only | 75.11% |
| 1,300 | 73.83% | Ridge + Cost | 75.78% |
| 1,500 | 75.11% | Frozen P1 Logistic | 76.44% |
| 1,700 | 76.00% | Ridge + Cost | 77.33% |
| 1,900 | 76.00% | Ridge + Cost | 77.78% |
| 2,100 | 77.16% | Frozen DeepSeek P0 Logistic | 78.44% |

These reference values come from earlier **exploratory validation** and should not be interpreted as a statistically established best-of-many ranking. BCHF itself does not compute a prospective online token-cost guarantee. Predicted-budget diagnostics were produced by the script but their actual numbers were **not included in the supplied console outputs**, so no calibrated-cost or end-to-end efficiency claim is made.

### 6.4 Decision

- **H1 (fusion complementarity):** not established by this readout; the full fixed-alpha heatmap would be needed to isolate interior-alpha benefits, but no further tuning is planned.
- **H2 (budget conditioning helps):** **not supported**. Limited low-budget descriptive gains are offset by high-budget declines, unstable fold-selected weights, and failure to beat existing references.
- **H3 (practical budget control):** **not established**. No verified prospective budget-calibration results or Router Prefill/latency accounting are available.
- **Close BCHF.** Retain the code, tests, research protocol, negative/conditional findings, and original server-local reports for reproducibility. **Do not run a wider alpha/architecture/penalty sweep on the same 450-question validation set**, promote BCHF as the selected Router, or spend the sealed 485-example test set to rescue this exploratory hypothesis. Future work should focus on separate Representation, quality-signal, or cost-calibration questions.

### Audit artifacts

- Code: [`scripts/route_e_budget_fusion.py`](../scripts/route_e_budget_fusion.py); tests: [`scripts/test_route_e_budget_fusion.py`](../scripts/test_route_e_budget_fusion.py).
- NTU TC1 output (server-local; not tracked): `results/route_e/budget_fusion/report.json`, `crossfit_summary.csv`, `fixed_alpha_sweep.csv`, `frozen_policy.json`, `crossfit_accuracy_token.{png,svg}`, and `alpha_budget_heatmap.{png,svg}`.
- No new model training, GPU inference, answer generation, or test-set evaluation was required.

## References

- [Route A objectives and baseline](ROUTE_A_PLAN.md)
- [Route E representations and prompt-prefix ablations](ROUTE_E_RESEARCH.md)
- [Prior Cost/Quality fusion and fixed λ=0.05](https://github.com/miyaliao/AdaptThink/blob/feature/route-e-token-aware-20261103/docs/ROUTE_E_QUALITY_FUSION.md)