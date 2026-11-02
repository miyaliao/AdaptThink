#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=modernbert_margin
#SBATCH --output=modernbert_margin_%j.out
#SBATCH --error=modernbert_margin_%j.err

# Two controlled objectives with the same encoder, original 3,000 strict labels,
# and unchanged 450-problem validation set. Does NOT use the canonical test set.
# Default MARGIN_MODE=both runs dual_mse, then dual_margin (beta=1).
# Example smoke: sbatch --export=ALL,MARGIN_SMOKE=1 scripts/ntu_modernbert_margin.sbatch
# Example full : sbatch scripts/ntu_modernbert_margin.sbatch
# Run one arm: sbatch --export=ALL,MARGIN_MODE=dual_margin scripts/ntu_modernbert_margin.sbatch

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

MARGIN_SMOKE="${MARGIN_SMOKE:-0}"
MARGIN_MODE="${MARGIN_MODE:-both}"
case "$MARGIN_MODE" in
    both) MODES=(dual_mse dual_margin) ;;
    dual_mse|dual_margin) MODES=("$MARGIN_MODE") ;;
    *) echo "Invalid MARGIN_MODE=$MARGIN_MODE" >&2; exit 2 ;;
esac
if [[ "$MARGIN_SMOKE" != "0" && "$MARGIN_SMOKE" != "1" ]]; then
    echo "Invalid MARGIN_SMOKE=$MARGIN_SMOKE" >&2
    exit 2
fi

python scripts/train_modernbert_margin.py --self-test
python scripts/train_modernbert_margin.py --check-data

for mode in "${MODES[@]}"; do
    echo "MARGIN_START mode=$mode smoke=$MARGIN_SMOKE"
    if [[ "$MARGIN_SMOKE" == "1" ]]; then
        python -u scripts/train_modernbert_margin.py --mode "$mode" --smoke
        echo "MARGIN_SMOKE_DONE mode=$mode"
    else
        python -u scripts/train_modernbert_margin.py --mode "$mode" --beta 1.0
        echo "MARGIN_EXPERIMENT_DONE mode=$mode"
    fi
done

echo "MARGIN_PIPELINE_DONE mode=$MARGIN_MODE smoke=$MARGIN_SMOKE"
