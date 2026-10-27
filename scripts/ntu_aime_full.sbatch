#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:45:00
#SBATCH --job-name=aime30_extend
#SBATCH --output=aime30_extend_%j.out
#SBATCH --error=aime30_extend_%j.err

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false

echo "JOB_ID=$SLURM_JOB_ID NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -u scripts/extend_aime_full.py --action preflight
python -u scripts/extend_aime_full.py --action generate --mode nothinking
python -u scripts/extend_aime_full.py --action generate --mode thinking
python -u scripts/extend_aime_full.py --action score
