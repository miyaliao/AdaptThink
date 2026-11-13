#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:50:00
#SBATCH --job-name=route_e_p2
#SBATCH --output=route_e_p2_%j.out
#SBATCH --error=route_e_p2_%j.err

# P2 trains CPU-only heads on cached P1 features. No GPU is requested:
# do not waste the course's one-GPU quota on a non-GPU experiment.
# If this GPU partition disallows CPU-only jobs, run in an allowed CPU
# partition or interactively in the existing .venv.
#
# sbatch --export=ALL,P2_STAGE=smoke scripts/ntu_route_e_token_aware.sbatch
# sbatch --export=ALL,P2_STAGE=full scripts/ntu_route_e_token_aware.sbatch

set -euo pipefail
cd /tc1home/UG/n2602710a/AdaptThink
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

P2_STAGE="$(printenv P2_STAGE || echo smoke)"
P2_OVERWRITE="$(printenv P2_OVERWRITE || echo 0)"
case "$P2_STAGE" in
  smoke|full) ;;
  *) echo "Invalid P2_STAGE=$P2_STAGE (smoke/full only)" >&2; exit 2 ;;
esac
case "$P2_OVERWRITE" in
  0) EXTRA="" ;;
  1) EXTRA="--overwrite" ;;
  *) echo "Invalid P2_OVERWRITE=$P2_OVERWRITE (0/1 only)" >&2; exit 2 ;;
esac

echo "P2_START stage=$P2_STAGE job=$(printenv SLURM_JOB_ID || echo local)"
python -u scripts/route_e_token_aware.py self-test
if [[ "$P2_STAGE" == "smoke" ]]; then
  python -u scripts/route_e_token_aware.py check --smoke
  # EXTRA is a validated optional CLI flag, deliberately word-split.
  python -u scripts/route_e_token_aware.py run --smoke --no-plot $EXTRA
else
  python -u scripts/route_e_token_aware.py check
  python -u scripts/route_e_token_aware.py run $EXTRA
fi
echo "P2_PIPELINE_DONE stage=$P2_STAGE"
