#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=route_e_frozen
#SBATCH --output=route_e_%j.out
#SBATCH --error=route_e_%j.err

# Usage:
#   sbatch --export=ALL,E_ENCODER=modernbert,E_SMOKE=1 scripts/ntu_route_e.sbatch
#   sbatch --export=ALL,E_ENCODER=modernbert,E_SMOKE=0 scripts/ntu_route_e.sbatch
#   sbatch --export=ALL,E_ENCODER=deepseek,E_SMOKE=1 scripts/ntu_route_e.sbatch
#   sbatch --export=ALL,E_ENCODER=deepseek,E_SMOKE=0 scripts/ntu_route_e.sbatch
# Only one 1-GPU job at a time under NTU TC1 quota. No test set is accessed.

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

E_ENCODER="${E_ENCODER:-modernbert}"
E_SMOKE="${E_SMOKE:-1}"
E_BATCH_SIZE="${E_BATCH_SIZE:-8}"
E_MAX_LENGTH="${E_MAX_LENGTH:-1024}"
case "$E_ENCODER" in
    modernbert|deepseek) ;;
    *) echo "Invalid E_ENCODER=$E_ENCODER" >&2; exit 2 ;;
esac
case "$E_SMOKE" in
    0) SMOKE_ARGS=() ;;
    1) SMOKE_ARGS=(--smoke) ;;
    *) echo "Invalid E_SMOKE=$E_SMOKE" >&2; exit 2 ;;
esac

echo "ROUTE_E_START encoder=$E_ENCODER smoke=$E_SMOKE job=${SLURM_JOB_ID:-local}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u scripts/route_e_frozen.py self-test
python -u scripts/route_e_frozen.py check-data
python -u scripts/route_e_frozen.py extract --encoder "$E_ENCODER" \
    --batch-size "$E_BATCH_SIZE" --max-length "$E_MAX_LENGTH" "${SMOKE_ARGS[@]}"
python -u scripts/route_e_frozen.py fit --encoder "$E_ENCODER" "${SMOKE_ARGS[@]}"
echo "ROUTE_E_DONE encoder=$E_ENCODER smoke=$E_SMOKE"
