#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --job-name=vllm_concurrency
#SBATCH --output=vllm_concurrency_%j.out
#SBATCH --error=vllm_concurrency_%j.err

set -uo pipefail
cd /tc1home/UG/n2602710a/AdaptThink || exit 1
source .venv/bin/activate || exit 1
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false

echo "Compute node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "Python: $(which python)"

# One SLURM allocation; sequential processes free GPU memory between runs.
# The same 32 questions and 2048 output tokens per request for 4 / 8 / 16.
fail=0
for max_seqs in 4 8 16; do
  echo "========== max_num_seqs=$max_seqs =========="
  if python -u scripts/benchmark_vllm_concurrency.py \
      --max-num-seqs "$max_seqs" \
      --num-prompts 32 \
      --max-tokens 2048 \
      --seed 42; then
    echo "CONFIG_STATUS $max_seqs SUCCESS"
  else
    code=$?
    echo "CONFIG_STATUS $max_seqs FAILED exit=$code"
    fail=1
  fi
done

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/benchmarks/vllm_concurrency")
print("\n=== SUMMARY (equal-token load, not accuracy) ===")
for n in (4, 8, 16):
    f = root / f"seqs{n}_n32_tok2048_seed42.json"
    if not f.exists():
        print(f"max_num_seqs={n}: NO SUCCESSFUL RESULT")
        continue
    row = json.loads(f.read_text())
    print(
        f"max_num_seqs={n}: "
        f"{row['aggregate_output_tokens_per_sec']:.2f} tokens/s, "
        f"{row['total_output_tokens']} output tokens, "
        f"{row['generation_seconds']:.2f} s"
    )
PY

exit "$fail"
