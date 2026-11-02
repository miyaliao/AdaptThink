#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=modernbert_delta
#SBATCH --output=modernbert_delta_%j.out
#SBATCH --error=modernbert_delta_%j.err

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
DELTA_SMOKE="$(printenv DELTA_SMOKE || echo 0)"

python scripts/train_modernbert_delta.py --self-test
python scripts/train_modernbert_delta.py --check-data
if [[ "$DELTA_SMOKE" == "1" ]]; then
    python scripts/train_modernbert_delta.py --smoke
    echo "DELTA_SMOKE_DONE"
else
    python scripts/train_modernbert_delta.py
    echo "DELTA_PIPELINE_DONE"
fi
