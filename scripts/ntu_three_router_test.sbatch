#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:45:00
#SBATCH --job-name=route3_test485
#SBATCH --output=route3_test485_%j.out
#SBATCH --error=route3_test485_%j.err

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# Set the actual trained Utility-Hard best checkpoint before submission:
#   sbatch --export=ALL,UTILITY_MODEL=outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/best_model \
#       scripts/ntu_three_router_test.sbatch
UTILITY_MODEL="${UTILITY_MODEL:-outputs/external_modernbert/utility_ablation/utility_hard_lam0.02/best_model}"

printf 'NODE=%s GPU=%s UTILITY_MODEL=%s\n' "$(hostname)" "${CUDA_VISIBLE_DEVICES:-unset}" "$UTILITY_MODEL"

# Zero-GPU checks and dataset construction (do not generate model answers):
python -u scripts/verify_baseline_split.py
python -u scripts/build_test_labels.py
python -u scripts/adaptthink_test_predict.py --stage self-test
python -u scripts/adaptthink_test_report.py --self-test
python -u scripts/adaptthink_test_predict.py --stage verify

# Sequential model forwards under one GPU; NO LLM answer decoding:
python -u scripts/adaptthink_test_predict.py --stage deepseek
python -u scripts/adaptthink_test_predict.py --stage modernbert --utility-model "$UTILITY_MODEL"

# CPU offline replay / figures / macro-vs-pooled summaries:
python -u scripts/adaptthink_test_report.py --main-budget 1300

echo "THREE_ROUTER_TEST485_DONE"
