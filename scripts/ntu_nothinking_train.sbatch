#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=05:45:00
#SBATCH --job-name=nt_train
#SBATCH --output=nt_train_%j.out
#SBATCH --error=nt_train_%j.err
#
# NoThinking generation for the Internal Router training labels (route B).
#
# Usage:
#   sbatch scripts/ntu_nothinking_train.sbatch                   # pilot: 600 problems
#   sbatch scripts/ntu_nothinking_train.sbatch --all             # all 3000 problems
#   sbatch scripts/ntu_nothinking_train.sbatch --all --shard=2/4 # shard 2 of 4
#
# The run is append-only and resumable: re-submitting the same command skips
# finished (sample_id, k) pairs, so the 5h45m time limit costs one batch at most.

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false

# --pilot is the default; --all switches to the full manifest. Sharding is
# optional and parsed as --shard=I/N (1-based shard index).
MODE=(--pilot)
SHARD=()
for arg in "$@"; do
  case "$arg" in
    --all)     MODE=() ;;
    --shard=*)
      IFS='/' read -r idx count <<< "${arg#--shard=}"
      if [ -z "${idx:-}" ] || [ -z "${count:-}" ]; then
        echo "Bad shard spec (expected --shard=I/N): $arg"; exit 2
      fi
      SHARD=(--shard "$((idx - 1))" --shards "$count")
      ;;
    *) echo "Unknown argument: $arg"; exit 2 ;;
  esac
done

echo "Slurm job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ ${#SHARD[@]} -eq 0 ]; then
  python -u scripts/gen_nothinking_train.py --preflight "${MODE[@]}"
  python -u scripts/gen_nothinking_train.py "${MODE[@]}"
else
  python -u scripts/gen_nothinking_train.py --preflight "${MODE[@]}" "${SHARD[@]}"
  python -u scripts/gen_nothinking_train.py "${MODE[@]}" "${SHARD[@]}"
fi
