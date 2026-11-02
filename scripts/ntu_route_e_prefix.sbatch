#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=route_e_prefix
#SBATCH --output=route_e_prefix_%j.out
#SBATCH --error=route_e_prefix_%j.err

# P1: E_PREFIX=assistant; P2: E_PREFIX=user_only
# E_SMOKE=1 (default) runs 32 train + 24 val; E_SMOKE=0 runs full.
# Run P0 full BEFORE launching these ablations, not concurrently.
# Outputs: outputs/route_e/prompt_ablation/ and results/route_e/prompt_ablation/
# Examples:
# sbatch --export=ALL,E_PREFIX=assistant,E_SMOKE=1 scripts/ntu_route_e_prefix.sbatch
# sbatch --export=ALL,E_PREFIX=user_only,E_SMOKE=1 scripts/ntu_route_e_prefix.sbatch
# sbatch --export=ALL,E_PREFIX=assistant,E_SMOKE=0 scripts/ntu_route_e_prefix.sbatch
# sbatch --export=ALL,E_PREFIX=user_only,E_SMOKE=0 scripts/ntu_route_e_prefix.sbatch

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

E_PREFIX="${E_PREFIX:-assistant}"
E_SMOKE="${E_SMOKE:-1}"
E_BATCH_SIZE="${E_BATCH_SIZE:-8}"
E_MAX_LENGTH="${E_MAX_LENGTH:-1024}"
case "$E_PREFIX" in
  assistant|user_only) ;;
  *) echo "Invalid E_PREFIX=$E_PREFIX" >&2; exit 2 ;;
esac
case "$E_SMOKE" in
  1) SMOKE_ARGS=(--smoke) ;;
  0) SMOKE_ARGS=() ;;
  *) echo "Invalid E_SMOKE=$E_SMOKE" >&2; exit 2 ;;
esac

echo "ABLAT_START mode=$E_PREFIX smoke=$E_SMOKE job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/route_e_prompt_ablation.py self-test
python -u scripts/route_e_frozen.py check-data
python -u scripts/route_e_prompt_ablation.py inspect-template
python -u scripts/route_e_prompt_ablation.py extract \
  --prompt-mode "$E_PREFIX" \
  --batch-size "$E_BATCH_SIZE" --max-length "$E_MAX_LENGTH" \
  "${SMOKE_ARGS[@]}"
python -u scripts/route_e_prompt_ablation.py fit \
  --prompt-mode "$E_PREFIX" "${SMOKE_ARGS[@]}"
echo "ABLAT_PIPELINE_DONE mode=$E_PREFIX smoke=$E_SMOKE"
