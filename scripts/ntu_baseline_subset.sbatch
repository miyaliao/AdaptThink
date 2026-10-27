#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:45:00
#SBATCH --job-name=baseline_1p5b
#SBATCH --output=baseline_1p5b_%j.out
#SBATCH --error=baseline_1p5b_%j.err

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false

echo "Slurm job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Fail fast before loading the model if the prompt, forced token, or scorer is wrong.
python -u scripts/eval_baseline_subset.py --action preflight

# Each mode writes a JSONL row for each response and fsyncs every 32 outputs.
# Re-submit this file after timeout/interruption: completed IDs are skipped.
python -u scripts/eval_baseline_subset.py --action generate --mode nothinking
python -u scripts/eval_baseline_subset.py --action generate --mode thinking

# Official mathematical scorer; writes summary.json after both modes complete.
python -u scripts/eval_baseline_subset.py --action score
