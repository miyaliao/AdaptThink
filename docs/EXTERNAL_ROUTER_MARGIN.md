# Route A — ModernBERT Dual MSE and Dual MSE + Margin

**Status (2026-11-02): implementation committed; GPU experiments not yet run.**

This is the next controlled comparison after Classification, Dual BCE, and Delta MSE.
The objectives are trained and evaluated independently using the **same** frozen
DeepSeek answer evidence; the DeepSeek model is not modified or regenerated.

## 1. Experiment definition

For every problem `x`, the full-fine-tuned `answerdotai/ModernBERT-base`
produces two logits. A sigmoid transforms them into bounded predicted success
rates `pT_hat` and `pN_hat`.

```text
L_dual   = ((pT_hat-pT)^2 + (pN_hat-pN)^2) / 2
L_margin = ((pT_hat-pN_hat)-(pT-pN))^2
dual_mse   : L = L_dual
dual_margin: L = L_dual + 1.0 * L_margin
```

The **Dual MSE-only control is essential**: the previous Dual Regression used
soft-label BCE, so a gain from the combined objective could otherwise be due
to switching BCE → MSE rather than to the margin term.

| Setting | Value |
| --- | --- |
| Encoder | `answerdotai/ModernBERT-base`, full fine-tuning |
| Train | **original strict** DeepScaleR 3,000 labels |
| Validation | GSM8K 300 + MATH-500 150 (strict field is preferred) |
| Test | **never accessed**; GSM8K 330 + MATH 125 + AIME 2024 30 remain sealed |
| Seed / lr / batch / epochs | `42` / `2e-5` / `16` / `5` |
| Sequence length | 512; FP16 single V100 |
| Weight decay / warmup / schedule | `0.01` / `0.1` / linear |
| Best checkpoint | smallest validation **objective loss** (not Classification BCE loss) |
| Router action | NoThinking for the **smallest predicted** `pT_hat - pN_hat` |
| Evaluation | offline strict cached answer replay, budget-matched Random reference |
| Cost accounting | measured mean **LLM output** tokens only; router overhead separate |

The best validation loss is **not comparable across objectives**; compare
their Accuracy–Token curves at the same *mean tokens*, not merely the same
NoThinking percentage. These are validation results, not test claims.

### IMPORTANT: keep Route A's original strict training data

Expected input paths:

- `results/router_labels_1p5b/all_3000_labels.jsonl` (3000 strict labels)
- `results/val_450_labels.jsonl` (450; `p_nothinking_strict` used when present)

The original strict **3,000-row JSONL currently is not on GitHub main**,
although its summary is committed. Confirm that it still exists **on the NTU
server**, where the previous Delta run succeeded. The new code refuses a
missing file or non-3000 size. **Do not substitute**:

- `results/router_labels_1p5b_compliant/all_3000_labels.jsonl` (2777 rows)
- `results/router_labels_1p5b_repaired/all_3000_labels.jsonl` (3000 repaired)

These would change the scientific meaning of the supervision. The new scripts
read the inputs, never rewrite them. The current validation file may contain
both repaired and strict probabilities: the reader **always prefers strict**.

## 2. Run on NTU (TC1)

First inspect local changes so teammates' untracked scripts/results are not
lost during a pull:

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git status --short
git pull --ff-only origin main
source .venv/bin/activate
test -s results/router_labels_1p5b/all_3000_labels.jsonl
python scripts/train_modernbert_margin.py --self-test
python scripts/train_modernbert_margin.py --check-data
```

**1 GPU smoke** runs both arms on 32 train / 24 val samples, one epoch each.
Smoke writes to separate `*_smoke` paths and **never** generates a 450-question
comparison table:

```bash
sbatch --export=ALL,MARGIN_SMOKE=1 scripts/ntu_modernbert_margin.sbatch
```

After smoke succeeds, run both independent full experiments sequentially
within **one single-GPU Slurm job** (no double GPU usage):

```bash
sbatch scripts/ntu_modernbert_margin.sbatch
```

If queue/time constraints require individual arms:

```bash
sbatch --export=ALL,MARGIN_MODE=dual_mse scripts/ntu_modernbert_margin.sbatch
sbatch --export=ALL,MARGIN_MODE=dual_margin scripts/ntu_modernbert_margin.sbatch
```

**Do not submit both single-arm jobs concurrently**: TC1 coursework users
have a one-GPU limit. The default job already runs both sequentially.

To inspect status (substitute the job ID):

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
grep -E 'MARGIN_|DATA_|Traceback' modernbert_margin_JOBID.out
tail -n 50 modernbert_margin_JOBID.err
```

CPU-only replay of previously saved predictions, if required:

```bash
python scripts/train_modernbert_margin.py --mode dual_mse --eval-only
python scripts/train_modernbert_margin.py --mode dual_margin --eval-only
```

## 3. Results and file locations

| Experiment | Best model / metadata / validation predictions | Accuracy–Token summary |
| --- | --- | --- |
| Dual MSE | `outputs/external_modernbert/dual_mse/` | `results/external_modernbert/dual_mse_val.json` + `.csv` |
| Dual MSE + Margin (`beta=1`) | `outputs/external_modernbert/dual_margin/` | `results/external_modernbert/dual_margin_val.json` + `.csv` |
| Smoke | `outputs/external_modernbert/{dual_mse,dual_margin}_smoke/` | No curve; never treated as full evaluation |

Each output directory contains `best_model/`, `validation_predictions.jsonl`,
and `run_metadata.json` with source hashes, training config, best checkpoint,
and evaluation history. JSON/CSV results include per-dataset curves, matching
Random references, and observed uplift. Large model checkpoints are intentionally
**not pushed** to GitHub.

After running, use the same matched-budget comparison as the three methods
in [`ROUTE_A_PLAN.md`](ROUTE_A_PLAN.md). The historical best is **Classification
+2.37 pp** observed uplift vs Global Random at 1,092 average tokens (validation),
not an independent test result.

### Pending results (leave blank until GPU training)

| Method | Best validation objective | Accuracy at ~1,100 tokens | Uplift vs matched Random |
| --- | ---: | ---: | ---: |
| Dual MSE | pending | pending | pending |
| Dual MSE + Margin, beta=1 | pending | pending | pending |

## 4. Files

- [`scripts/train_modernbert_margin.py`](../scripts/train_modernbert_margin.py)
  - `--self-test`: math and routing direction checks without GPU
  - `--check-data`: strict split count/hash audit on CPU
  - `--smoke`: isolated small GPU training check
  - `--mode dual_mse|dual_margin`: controlled objective
  - `--eval-only`: recompute the offline curves from saved predictions
- [`scripts/ntu_modernbert_margin.sbatch`](../scripts/ntu_modernbert_margin.sbatch)
  - single V100; `MARGIN_MODE` selects one/both arms
- [`scripts/train_modernbert_delta.py`](../scripts/train_modernbert_delta.py)
  - reuses its tested strict label loader and matched-budget curve calculations

**Implementation has been committed, but cannot be claimed GPU-validated until
the NTU smoke/full Slurm jobs finish.**
