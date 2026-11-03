#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=route_e_layer_head
#SBATCH --output=route_e_layer_head_%j.out
#SBATCH --error=route_e_layer_head_%j.err

# E_STAGE=extract_fit (default) extracts 4 layers and fits 4 linear probes.
# E_STAGE=mlp fits ONE selected layer on cached features (no extraction).
# E_SMOKE=1: train32/val24; E_SMOKE=0: original train3000/val450.
# Never run full and smoke jobs concurrently under NTU's one-GPU quota.
# Example: sbatch --export=ALL,E_SMOKE=1 scripts/ntu_route_e_layer_head.sbatch
# Example: sbatch --export=ALL,E_SMOKE=0 scripts/ntu_route_e_layer_head.sbatch
# Example: sbatch --export=ALL,E_STAGE=mlp,E_SMOKE=0,E_LAYER=14 scripts/ntu_route_e_layer_head.sbatch

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

E_STAGE="${E_STAGE:-extract_fit}"
E_SMOKE="${E_SMOKE:-1}"
E_LAYER="${E_LAYER:-28}"
E_BATCH_SIZE="${E_BATCH_SIZE:-8}"
E_MAX_LENGTH="${E_MAX_LENGTH:-1024}"
case "$E_SMOKE" in 1) S=(--smoke);; 0) S=();; *) echo 'Invalid E_SMOKE' >&2; exit 2;; esac
case "$E_STAGE" in extract_fit|mlp) ;; *) echo 'Invalid E_STAGE' >&2; exit 2;; esac

echo "LAYER_HEAD_START job=${SLURM_JOB_ID:-local} stage=$E_STAGE smoke=$E_SMOKE layer=$E_LAYER"
python -u scripts/route_e_layer_head.py self-test
python -u scripts/route_e_frozen.py check-data
python -u scripts/route_e_layer_head.py check-cache "${S[@]}"
python -u scripts/route_e_prompt_ablation.py inspect-template
if [[ "$E_STAGE" == extract_fit ]]; then
  python -u scripts/route_e_layer_head.py extract "${S[@]}" \
    --batch-size "$E_BATCH_SIZE" --max-length "$E_MAX_LENGTH"
  python -u scripts/route_e_layer_head.py fit-logistic "${S[@]}"
else
  python -u scripts/route_e_layer_head.py fit-mlp "${S[@]}" --layer "$E_LAYER"
fi
echo "LAYER_HEAD_JOB_DONE job=${SLURM_JOB_ID:-local} stage=$E_STAGE smoke=$E_SMOKE"
