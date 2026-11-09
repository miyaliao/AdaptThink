#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=cost_layer_test
#SBATCH --output=cost_layer_test_%j.out
#SBATCH --error=cost_layer_test_%j.err
set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -u scripts/route_e_cost_layer_test.py self-test
# Historical L28 parity, train L16 cost head, freeze VAL-only numerical cutoffs.
python -u scripts/route_e_cost_layer_test.py prepare
# GPU frozen prefill on canonical 485 TEST QUESTIONS; no outcome read.
python -u scripts/route_e_cost_layer_test.py extract --batch-size 4
# Only after thresholds fixed and features extracted, materialize cached test outcomes.
if [ ! -f results/test_485_labels.jsonl ]; then
  python -u scripts/build_test_labels.py
fi
python -u scripts/route_e_cost_layer_test.py report
echo COST_LAYER_TEST_PIPELINE_DONE
