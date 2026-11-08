#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=modernbert_lambda
#SBATCH --output=modernbert_lambda_%j.out
#SBATCH --error=modernbert_lambda_%j.err

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

# SMOKE=1 tests one new λ arm without changing full-run artifacts.
# sbatch --export=ALL,SMOKE=1 scripts/ntu_modernbert_utility_lambda_sweep.sbatch
# Full sweep: sequential missing runs on one GPU.
# sbatch scripts/ntu_modernbert_utility_lambda_sweep.sbatch

python scripts/modernbert_utility_lambda_sweep.py self-test
python scripts/modernbert_utility_lambda_sweep.py plan
if [[ "${SMOKE:-0}" == "1" ]]; then
  python scripts/train_modernbert_utility.py --objective utility_hard \
    --lambda-cost 0.005 --tau 0.25 --smoke
  echo MODERNBERT_LAMBDA_SMOKE_DONE
  exit 0
fi

python scripts/modernbert_utility_lambda_sweep.py run
echo MODERNBERT_LAMBDA_PIPELINE_DONE
