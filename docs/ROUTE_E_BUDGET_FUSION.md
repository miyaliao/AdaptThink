# Route E — Budget-Conditioned Head Fusion (BCHF)

**Status:** Implemented, **not yet run against the real 3,000/450 server caches**. Experimental design and code as of 2026-11-04. No new model training, DeepSeek generation, or test-set access.

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

## References

- [Route A objectives and baseline](ROUTE_A_PLAN.md)
- [Route E representations and prompt-prefix ablations](ROUTE_E_RESEARCH.md)
- [Prior Cost/Quality fusion and fixed λ=0.05](https://github.com/miyaliao/AdaptThink/blob/feature/route-e-token-aware-20261103/docs/ROUTE_E_QUALITY_FUSION.md)