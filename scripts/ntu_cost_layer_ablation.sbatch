#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=cost_layer
#SBATCH --output=cost_layer_%j.out
#SBATCH --error=cost_layer_%j.err
set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
case "${LAYER_SMOKE:-1}" in
  1) SMOKE=(--smoke) ;;
  0) SMOKE=() ;;
  *) echo "LAYER_SMOKE must be 0 or 1" >&2; exit 2 ;;
esac
python -u scripts/route_e_cost_layer_ablation.py check
python -u scripts/route_e_cost_layer_ablation.py extract --batch-size "${LAYER_BATCH_SIZE:-4}" "${SMOKE[@]}"
python -u scripts/route_e_cost_layer_ablation.py evaluate "${SMOKE[@]}"
echo "LAYER_COST_PIPELINE_DONE"
