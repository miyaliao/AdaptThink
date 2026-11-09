# Experiment F — Independent Canonical Test Evaluation

**Status: CLOSED — canonical 485-question TC1 test evaluation completed on 2026-11-09.** See [Experiment F Final Results](ROUTE_E_COST_DIFFERENCE_RESULTS.md) for the recorded outcomes and the retained original A0 Cost-only baseline. Re-running or retuning on the test set is not advised.

## Locked methods

The original strict 3,000-train/450-validation Experiment F produces all fitted heads, the selected beta, and validation policies. Train-only CV selected **beta=0**.

- **A0:** Existing P2 Cost-only: two independent log-token Ridge heads, L2=0.02.
- **A1:** Direct Delta-Cost: Ridge regression on (t_T-t_N)/1000.
- **A2:** Margin-aware with train-selected beta=0: exactly equals A0.

No model retraining, tuning or method selection may use the test set. The frozen P1 1.5B representation uses the same tokenizer, chat template, last-valid-prompt-token pooling and final hidden layer as the original train/validation caches.

## Test split

The canonical split is `data/splits/adaptthink_original_baseline_seed42.json`: 485 **unique** questions (GSM8K 330, MATH500 125, AIME24 30). Existing paired raw grade files contain 935 generations per mode: one each for GSM8K/MATH and 16 per AIME problem. Average the 16 AIME repeats *within each question*, then average over the 485 questions, not 935 raw generations.

The test script hard-checks fixed split hash, Git blob hashes of source questions, IDs, modes, repetition completeness and uniqueness. It reads test grades only at the final evaluate stage.

## Run on NTU TC1

First pull the new branch (preserving your server-local caches and trained heads):

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git fetch origin
git switch feature/route-e-cost-diff-test-20261109
git pull --ff-only origin feature/route-e-cost-diff-test-20261109
source .venv/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# CPU: local tests + freeze validation-selected thresholds BEFORE test evaluation
python scripts/route_e_cost_difference_test.py self-test
python scripts/route_e_cost_difference_test.py freeze
python scripts/route_e_cost_difference_test.py check

# GPU: allocate one GPU on TC1 for one-time frozen P1 test prefill
python scripts/route_e_cost_difference_test.py extract --batch-size 4 --max-length 2048

# CPU only: apply frozen heads, no test grades used
python scripts/route_e_cost_difference_test.py score

# CPU only: FIRST independent test evaluation, reads paired test grades
python scripts/route_e_cost_difference_test.py evaluate
```

If a question exceeds the specified max-length, rerun extraction with a higher guard; the script does not silently truncate. If a test-feature cache exists, extraction refuses to overwrite it (verify the provenance first). Never overwrite the historical P2 or Experiment F validation artifacts.

## Outputs

```text
outputs/route_e/cost_difference/test/test_p1_features.npz
results/route_e/cost_difference/test/
    frozen_validation_policies.json
    test_pre_generation_scores.json
    test_frozen_threshold.csv
    test_descriptive_equal_budget.csv
    test_cost_diagnostics.csv
    test_bootstrap.csv
    test_summary.json
```

## Interpret TWO test analyses separately

**Primary practical check: Frozen validation thresholds.** `test_frozen_threshold.csv` applies the exact preselected numeric routing score thresholds and mixture weights from VALIDATION to test questions. Report *actual test accuracy, realized mean tokens, NoThinking fraction and budget miss*. A requested validation budget (900–2100) need not be met at test time. Compare A0/A1 at their **realized** costs, never as if these costs were equal.

**Secondary descriptive test-cost-matched sweep.** `test_descriptive_equal_budget.csv` uses the frozen scores to rank 485 questions; only after ranking, it uses their observed TEST output lengths to interpolate nearest test-cost policy points at budgets 900–2100. The 486-policy curve gives a descriptive test nAUC-AC. This is *not deployable*, not real budget control and never a tuning signal. Label as **Descriptive Test-Sweep**.

`test_bootstrap.csv` has conditional nominal 95% paired CIs, treating each of the 485 questions as one unit (not 935 repeated generations); seven-budget multiple comparisons are not adjusted. Results use mean generated output tokens only, excluding router/prefill overhead.

## Precommitted comparison

Validation results: A0 Δt Spearman 0.623, MAE 1,531.8, nAUC +38.3885; A1 Spearman 0.521, MAE 2,597.3, nAUC +38.9911. A2 beta=0 identical to A0. Only the 1,700-token A1−A0 nominal paired interval excluded zero. The independent test found descriptive test nAUC-AC A0 **+49.3104** versus A1 **+49.3580** (Δ=+0.0476), with no consistent frozen-threshold dominance. See the final-results report; preserve the original A0 baseline.

Stop after the test and write down all findings; do not retune on test.

## One-job Slurm alternative

For a single-GPU TC1 submission, run:

```bash
sbatch scripts/ntu_route_e_cost_difference_test.sbatch
```

This sequentially freezes validation thresholds, checks inputs, extracts test P1 features once, scores the frozen heads and evaluates the fixed test set. Existing stage outputs are reused and fingerprint-checked; no stage automatically overwrites old artifacts. Inspect `cost_diff_test_<jobid>.out` and `.err` before interpreting results. GPU is needed only for one-time P1 feature extraction.

## Completed test readout (operator logs, 2026-11-09)

- Frozen-validation thresholds and descriptive test-cost-matched sweep ran successfully; test AIME repeats were aggregated by question. The runner printed `COST_DIFF_TEST_DONE`.
- Descriptive nAUC-AC: A0 **+49.3104**, A1 **+49.3580**, A2 **+49.3104** (A2 β=0).
- For detailed accuracy/token-budget tables and interpretation, see [Experiment F Final Results](ROUTE_E_COST_DIFFERENCE_RESULTS.md).
- Numeric test-bootstrap and test cost-diagnostic CSV contents were not supplied at closeout; do not claim test statistical significance.
