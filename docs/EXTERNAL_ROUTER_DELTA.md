# ModernBERT Delta Regression (SC4001)

This new code does not overwrite previous uncommitted server-local ModernBERT
Classification / Dual Regression scripts or results.

## What changes

Train the existing ModernBERT-base encoder with a **single regression output**
and **MSE loss**. For question x, the label is:

    delta(x) = p_thinking(x) - p_nothinking(x)

Source files remain unchanged:
- Training: results/router_labels_1p5b/all_3000_labels.jsonl (3,000 DeepScaleR)
- Validation: results/val_450_labels.jsonl (300 GSM8K + 150 MATH-500)
- Test: NEVER used (330 GSM8K + 125 MATH-500 + 30 AIME 2024).

The model sees question text only. Labels are the measured strict scoring rates.
If a future validation file has a separate "p_nothinking_strict" field, that
field is preferred to repaired scoring. The training/validation labels are
sample-based and have different amounts of noise.

Model: answerdotai/ModernBERT-base, full fine-tuning, 2e-5 LR, 5 epochs,
batch 16, length 512, FP16, seed 42. Best checkpoint by validation MSE.
Previous server-local training may have different batch/warmup/weight decay;
document such differences rather than claim perfect hyperparameter matching.

## NTU run steps

Check local changes before pulling, particularly untracked existing scripts:

    cd /tc1home/UG/n2602710a/AdaptThink
    git status --short
    git pull --ff-only origin main
    source .venv/bin/activate
    python scripts/train_modernbert_delta.py --self-test
    python scripts/train_modernbert_delta.py --check-data

First use a tiny 32-train/24-val GPU smoke (stored separately):

    sbatch --export=ALL,DELTA_SMOKE=1 scripts/ntu_modernbert_delta.sbatch

After successful smoke, submit the full 3,000/450 experiment:

    sbatch scripts/ntu_modernbert_delta.sbatch

Inspect the job (replace JOBID):

    sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
    grep -E 'DATA_OK|DELTA_TRAIN_DONE|DELTA_EVAL_DONE|BEST_VAL_MSE|DELTA_PIPELINE_DONE' modernbert_delta_JOBID.out
    tail -n 30 modernbert_delta_JOBID.err

Re-evaluate existing saved predictions without a GPU:

    python scripts/train_modernbert_delta.py --eval-only

## Files

- outputs/external_modernbert/delta/best_model/ (model + tokenizer)
- outputs/external_modernbert/delta/validation_predictions.jsonl
- outputs/external_modernbert/delta/run_metadata.json (label hashes / config)
- results/external_modernbert/delta_val.json and delta_val.csv

All output Token counts come from the already generated validation answers.
For a target NoThinking share, the smallest predicted Delta goes to NT.
Random reference is the exact mixture EXPECTATION at the SAME measured mean
LLM output-token cost, not simply at the same NT share. Per-GSM8K/MATH
metrics are included. Router overhead is excluded and should be measured later.

This script does NOT regenerate DeepSeek outputs, touch test labels,
or tune on test. No claim of improved accuracy should be made until the
GPU training and offline evaluation finish.
