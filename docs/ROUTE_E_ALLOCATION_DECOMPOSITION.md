# Route E / P2 — Experiment 1B: Between-Dataset Allocation vs Within-Dataset Selection

**Status:** Descriptive first-pass estimates from user-reported grouped metrics completed; exact strict-val accounting script and synthetic CI passing, pending NTU run on unrounded `comparison.json`.

Parent experiment: [ROUTE_E_COST_ONLY.md](ROUTE_E_COST_ONLY.md).

## 1. Motivation

Experiment 1 showed E (cost-only) exceeds B (quality-only) by +4.44 pp at 900 tokens, but below 1,700 tokens E starts losing. E routes most MATH questions to NoThinking and reserves Thinking for GSM8K; the split alone does not tell whether its gains are *between-group resource allocation* or *within-group question selection*.

On original strict validation, GSM8K has Thinking/NoThinking endpoints **79.00% / 68.67%**, generated-token costs **906.51 / 290.49**; MATH has **77.33% / 66.00%**, costs **4,813.95 / 1,121.84**. Naive dataset-level marginal gain per 1,000 generated tokens (from an expectation over the *entire* dataset, not per-example) is around **16.8 pp/1k GSM8K** versus **3.1 pp/1k MATH**. This gives a plausible reason to allocate scarce Thinking tokens to GSM8K, but is not yet proof of the mechanism.

## 2. Exact descriptive identity

At each global budget B, take a router m's *observed* mean generated-token budget B_g for each dataset g. Use only the dataset's all-Thinking and all-NoThinking mean endpoints to construct a hypothetical **dataset-stratified randomized policy** that consumes that same per-group mean budget in expectation. Compute its expected accuracy R_strat,m, with group weights n_g/450. Let R_global(B) be globally randomized mode allocation at mean budget B, and A_m the router's strict accuracy.

- **Between-dataset allocation component:** R_strat,m − R_global(B).
- **Within-dataset selection component:** A_m − R_strat,m.
- **Identity:** A_m − R_global(B) = allocation component + within-group component.

For E − B, subtract their respective components, yielding an exact **descriptive accounting decomposition** (conditional on the fixed data) with the same total E−B accuracy difference. This decomposition is not causal: observed group budgets were computed using outcomes of routing, and the stratified random benchmark uses actual validation-group endpoint means. It is **not a deployable router**.

If a router's actual dataset cost falls outside a group's all-T to all-NT *randomized* endpoint range, the cost-matched random baseline cannot be represented and the script explicitly marks the decomposition unavailable instead of extrapolating.

## 3. First-pass E−B decomposition (rounded printed inputs)

The following approximate numbers use the user's printed four-decimal group accuracies, one-decimal group generated tokens, and the strict endpoint means. The exact server script recomputes from full-precision `comparison.json`.

| Global budget | E − B total (pp) | Allocation difference (pp) | Within-group selection difference (pp) |
| ---: | ---: | ---: | ---: |
| 900 | +4.44 | +2.94 | +1.50 |
| 1,100 | +2.88 | +3.28 | −0.40 |
| 1,300 | +2.23 | +4.00 | −1.77 |
| 1,700 | −0.92 | +3.15 | −4.07 |

Interpretation: Cost-only shows consistently positive **allocation** advantage relative to B over these four budgets, but can lose via **within-dataset selection**, especially at 1,700. The 900-budget improvement appears to contain both components. These estimates are diagnostic, not statistical hypothesis tests. The complete server report also decomposes D−E.

## 4. Exact run on NTU server

Use the **existing** Experiment 1 output; no model fits, GPU, feature extraction, or new generations:

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git switch feature/route-e-token-aware-20261103
git pull --ff-only origin feature/route-e-token-aware-20261103
source .venv/bin/activate

python scripts/route_e_allocation_decomposition.py self-test
python scripts/route_e_allocation_decomposition.py run
~~~

Expected outputs: `results/route_e/cost_only/experiment1b/decomposition.json`, `decomposition.csv`, and `subgroup_random.csv`.

Expected log lines: `COST_ALLOC_DECOMPOSITION_PASS`, `COST_ALLOC budget=...`, `COST_ALLOC_E_MINUS_B budget=...`, `COST_ALLOC_DONE`.

Verify the complete seven-budget result before scientific interpretation. Keep Experiment 1's original results untouched. This follow-up neither improves nor validates an online budget selector; that remains a separate Cost Calibration task.
