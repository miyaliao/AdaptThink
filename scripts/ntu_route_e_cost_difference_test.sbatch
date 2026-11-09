#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=cost_diff_test
#SBATCH --output=cost_diff_test_%j.out
#SBATCH --error=cost_diff_test_%j.err

# One-time independent test execution. Never retrain / tune on test.
# Reuses original P2 and Experiment F model heads + paired test grade records.
set -euo pipefail

cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

SCRIPT=scripts/route_e_cost_difference_test.py
FROZEN=results/route_e/cost_difference/test/frozen_validation_policies.json
CACHE=outputs/route_e/cost_difference/test/test_p1_features.npz
SCORES=results/route_e/cost_difference/test/test_pre_generation_scores.json
FINAL=results/route_e/cost_difference/test/test_summary.json

python -u "$SCRIPT" self-test

# Lock policies before opening test grades.
if [[ ! -f "$FROZEN" ]]; then
    python -u "$SCRIPT" freeze
else
    echo "COST_DIFF_TEST_FROZEN_EXISTS: reuse; evaluator checks hashes"
fi
python -u "$SCRIPT" check

# GPU is used ONLY for frozen DeepSeek P1 test prefill.
if [[ ! -f "$CACHE" ]]; then
    python -u "$SCRIPT" extract --batch-size 4 --max-length 2048
else
    echo "COST_DIFF_TEST_FEATURES_EXISTS: reuse; scorer checks hashes"
fi

# CPU prediction only, no test grades read.
if [[ ! -f "$SCORES" ]]; then
    python -u "$SCRIPT" score
else
    echo "COST_DIFF_TEST_SCORES_EXISTS: reuse; evaluator checks hashes"
fi

# Read fixed paired baseline grade files ONLY during final evaluation.
if [[ -f "$FINAL" ]]; then
    echo "COST_DIFF_TEST_ALREADY_EVALUATED: no reruns / test-based tuning"
else
    python -u "$SCRIPT" evaluate
fi
echo "COST_DIFF_TEST_PIPELINE_DONE"
