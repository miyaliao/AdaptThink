#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:45:00
#SBATCH --job-name=router_nt16k
#SBATCH --output=router_nt16k_%j.out
#SBATCH --error=router_nt16k_%j.err

# Isolated NoThinking labels for External Router + Internal Router.
# Original DeepScaleR Thinking statistics are reused from the released K>=16 file.
#
# Run SEQUENTIALLY, never submit overlapping jobs writing the same JSONL:
#   sbatch scripts/ntu_router_labels_16k.sbatch --smoke  # 2 questions x K4 (8 outputs)
#   sbatch scripts/ntu_router_labels_16k.sbatch          # pilot 600 x K4
#   sbatch scripts/ntu_router_labels_16k.sbatch          # resume if Slurm time limit
#   sbatch scripts/ntu_router_labels_16k.sbatch --all    # later, expand to all 3000
#
# 16K protocol. Keep this output separate from the historical 8K label runs.

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false

MODE=(--pilot)
LIMIT=()
for arg in "$@"; do
  case "$arg" in
    --smoke) LIMIT=(--limit 8 --batch 8) ;;
    --all)   MODE=() ;;
    *) echo "Unknown option: $arg. Allowed: --smoke, --all"; exit 2 ;;
  esac
done

if [ ${#MODE[@]} -eq 0 ] && [ ${#LIMIT[@]} -ne 0 ]; then
  echo "Do not combine --all with --smoke"; exit 2
fi

COMMON=(--max-tokens 16384 --max-num-seqs 8
        --output-dir outputs/router_labels_train_16k)

echo "Slurm job: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -u scripts/gen_nothinking_train.py "${COMMON[@]}" "${MODE[@]}" "${LIMIT[@]}" --preflight
python -u scripts/gen_nothinking_train.py "${COMMON[@]}" "${MODE[@]}" "${LIMIT[@]}"
