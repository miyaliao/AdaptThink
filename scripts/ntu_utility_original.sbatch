#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=utility_orig
#SBATCH --output=utility_orig_%j.out
#SBATCH --error=utility_orig_%j.err

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# Smoke one lambda first:
# sbatch --export=ALL,ORIGINAL_SMOKE=1 scripts/ntu_utility_original.sbatch
# Full ModernBERT: six lambda configurations, sequentially, only one GPU.
# sbatch scripts/ntu_utility_original.sbatch

python -u scripts/utility_original_ablation.py self-test
if [[ "${ORIGINAL_SMOKE:-0}" == "1" ]]; then
  python -u scripts/utility_original_ablation.py train --encoder modernbert --lambdas 0.02 --smoke
  echo UTILITY_ORIGINAL_SMOKE_DONE
  exit 0
fi
python -u scripts/utility_original_ablation.py plan --encoder modernbert
python -u scripts/utility_original_ablation.py train --encoder modernbert
python -u scripts/utility_original_ablation.py evaluate --encoder modernbert
echo UTILITY_ORIGINAL_MODERNBERT_DONE
