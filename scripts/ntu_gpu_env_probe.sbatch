#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=adaptthink_env_probe
#SBATCH --output=gpu_probe_%j.out
#SBATCH --error=gpu_probe_%j.err

set -e
echo '=== Slurm ==='
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo '=== NVIDIA ==='
nvidia-smi
echo '=== Conda ==='
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list
conda activate "$HOME/.conda/envs/adaptthink"
which python
python --version
echo '=== Installed dependencies and PyTorch CUDA ==='
python - <<'PY'
import importlib.metadata as m
for pkg in ('torch','vllm','transformers','flash-attn','sympy','math-verify','pylatexenc','datasets','pyarrow','pandas'):
    try:
        print(f'{pkg}: {m.version(pkg)}')
    except m.PackageNotFoundError:
        print(f'{pkg}: NOT INSTALLED')
try:
    import torch
    print('PyTorch CUDA:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print('GPU:', p.name)
        print('Total VRAM (GiB):', round(p.total_memory/2**30, 2))
        print('Compute capability:', f'{p.major}.{p.minor}')
except Exception as e:
    print('PyTorch failed:', repr(e))
PY
