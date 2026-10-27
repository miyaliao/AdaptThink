# AdaptThink — NTU HPC Project Environment

Updated: 2026-10-27. This file documents the **project-relevant** environment, not the entire cluster. "Observed" refers to terminal output; "To verify" means not yet confirmed on a GPU compute node.

## 1. Repository, environment and model

| Item | Value | Evidence |
| --- | --- | --- |
| Account | n2602710a | Observed |
| Login host | CCDS-TC1 | Observed |
| Project root | /tc1home/UG/n2602710a/AdaptThink | User-reported |
| GitHub fork (origin) | https://github.com/miyaliao/AdaptThink | Verified |
| Research upstream | https://github.com/THU-KEG/AdaptThink | Verified |
| Conda base | /tc1apps/anaconda3 | Observed |
| Conda version | 26.5.3 | Observed |
| Project Conda environment | adaptthink (expected path: $HOME/.conda/envs/adaptthink) | Created during setup; confirm path |
| Project Python | 3.10 target | Upstream recommendation; check installed version |
| Base model | model/DeepSeek-R1-Distill-Qwen-1.5B | Downloaded per project setup; verify files |
| Slurm partition | UGGPU-TC1 | Observed from submitted job |
| Slurm QoS | normal | Observed from submitted job |

Keep downloaded weights in **model/** and outside Git. The fork's .gitignore already excludes /model/.

### Check the actual Python environment

~~~bash
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list
conda activate "$HOME/.conda/envs/adaptthink"
which python
python --version
python -m pip show torch vllm transformers flash-attn 2>/dev/null
~~~

The initial SSH environment used /usr/bin/python, and after loading Anaconda used /tc1apps/anaconda3/bin/python. The initial environment list showed only base; the dedicated environment was created later. Do not confuse the base environment with the AdaptThink environment.

## 2. Reusable CUDA/GPU software

The NTU module system showed **available CUDA toolkit modules**:

- cuda/11.8, cuda/12.2, cuda/12.3, cuda/12.4, cuda/12.5, cuda/12.6, cuda/12.8, cuda/12.9 and cuda/13.0.
- anaconda, miniconda, python/3.10, python/3.11 and other Python versions.
- slurm/slurm/23.02.8.

These are **available modules**, not proof of which CUDA toolkit has been loaded, which NVIDIA driver is installed, or which GPU has been allocated. The TC1 User Guide identifies its cards as **NVIDIA Tesla V100 32GB (Volta/SM70)**; verify the actual allocated GPU with nvidia-smi in a Slurm job. The previous MyTCinfo output gave the assigned normal QoS limits: **1 GPU, 20 CPUs, 64 GB host RAM, 2 submitted jobs, and 6 hours maximum wall time per job**. These limits may change by semester, so recheck MyTCinfo before submission. On V100 use float16 (not bfloat16), and do not assume FlashAttention-2 works; vLLM 0.8.2 supports SM70 with a suitable backend.

Check loaded modules with:

~~~bash
module list
module avail cuda
# Do not choose a toolkit version until PyTorch/driver compatibility is checked.
# module load cuda/12.4
# nvcc --version
~~~

Upstream AdaptThink uses vLLM **0.8.2**, transformers **4.51.2** (in requirements.txt), FlashAttention, and Python **3.10**. These are upstream requirements, **not confirmation that those packages are installed** locally. Avoid an indiscriminate dependency install before the CUDA/driver/PyTorch compatibility check.

## 3. GPU allocation through Slurm

An earlier test job was submitted as job **69774**, requesting:

- partition UGGPU-TC1; QoS normal; 1 node; 1 GPU
- memory 8 GB; time 10 minutes

The submission message did not confirm successful GPU allocation. Check status and logs:

~~~bash
sacct -j 69774 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
cat ~/SC4001-Efficient-Reasoning/gpu_check_69774.out
cat ~/SC4001-Efficient-Reasoning/gpu_check_69774.err
~~~

A fresh project-specific probe can be created in the project root:

~~~bash
cat > gpu_probe.sh <<'EOF'
#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=adaptthink_gpu_probe
#SBATCH --output=gpu_probe_%j.out
#SBATCH --error=gpu_probe_%j.err

echo "Compute host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/adaptthink"
python --version
python - <<'PY'
try:
    import torch
    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print("GPU:", gpu.name)
        print("VRAM (GiB):", round(gpu.total_memory / 2**30, 2))
except ImportError:
    print("PyTorch not installed yet")
PY
EOF

sbatch gpu_probe.sh
squeue -u n2602710a
# Then inspect gpu_probe_<JOBID>.out and gpu_probe_<JOBID>.err
~~~

The guide says the cluster uses V100 32GB; confirm the allocated GPU and driver with the job output. A login-node test is not sufficient.

## 4. Base model and AdaptThink-specific compatibility

Base model location:

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
du -sh model/DeepSeek-R1-Distill-Qwen-1.5B
ls -lh model/DeepSeek-R1-Distill-Qwen-1.5B/
~~~

The upstream README requires checking the downloaded model's tokenizer_config.json chat_template for the expected assistant/thinking prefix before training. Upstream also publishes precomputed reference sampling results under data/train/ref_results, potentially avoiding expensive repeated presampling if the data/model configuration matches.

The upstream **1.5B full RL training** was reported using a node with **8 H800 GPUs**. This is *not* evidence that NTU provides H800s. Determine NTU GPU allocation and scale the experiment before reproducing the original large training script.

## 5. Git workflow

The intended remotes are:

- origin: git@github.com:miyaliao/AdaptThink.git
- upstream: https://github.com/THU-KEG/AdaptThink.git

Check and synchronize:

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git remote -v
git branch --show-current
git status -sb
git fetch origin
# If on main and local changes won't conflict:
git pull --ff-only origin main
~~~

Submit code/doc changes from HPC with git add, git commit, and git push to origin. Never commit model weights, secrets, Conda environments, checkpoints or logs.

## 6. Information still to verify

1. Output of the GPU probe: actual GPU type, VRAM, driver, CUDA visibility and job status.
2. Exact Conda environment prefix and Python version.
3. Installed PyTorch, vLLM, transformers and FlashAttention versions.
4. Loaded CUDA toolkit and compatibility with the PyTorch CUDA runtime.
5. Model directory integrity and tokenizer template.

## 7. Evaluation baseline: scope and runtime planning

- Model: local DeepSeek-R1-Distill-Qwen-1.5B (before AdaptThink RL).
- Original Thinking: ordinary model thinking prefix/template.
- Original NoThinking: force the first generated token to be </think> (empty thinking segment), as defined in the paper.
- Evaluation data already present in this repo: GSM8K 1,319; MATH500 500; AIME2024 30 questions each sampled 16 times (480 generations).
- Thus 2,299 outputs per mode; 4,598 outputs for both modes. Use 16K context/output budget and temperature 0.6 per paper, with exact tokenizer/sampling checks before claiming numerical replication.
- Use the original src/adapt_think_rm.py checker; its imports include math_verify, sympy, and pylatexenc, which must be verified in the active environment.
- Do not run the upstream scripts/run_eval_hf.sh unchanged: it evaluates 7B and MMLU too, assumes 8 GPUs, and has absolute paths.
- Expected generated tokens estimated from paper mean lengths: Thinking 9.53M; NoThinking 1.75M; combined ~11.28M (input tokens/runtime setup excluded).
- At hypothetical sustained 100/200/400 generated tokens/s, generation takes about 31.3/15.7/7.8 h respectively, before overhead. Actual throughput must be measured on the V100.
- A six-hour per-job limit requires resumable batches unless measured aggregate throughput is unusually high.
- Recommended first experiment: GPU/Conda probe; 3–5 GSM8K cases per mode; a few MATH500; one longer AIME; save outputs and time by dataset/mode.
