# NTU TC1 — AdaptThink 1.5B Inference Environment

Validated on **2026-10-27**. Scope: DeepSeek-R1-Distill-Qwen-1.5B baseline evaluation on NTU TC1; this is **not** the RL training environment.

## Canonical test split for all future experiments

Use `data/splits/adaptthink_original_baseline_seed42.json` as the **single source of truth**: GSM8K **330/1319**, MATH-500 **125/500**, and AIME 2024 **all 30/30**, with **16 generations per AIME question per mode**. There are **935 generations per mode, 1,870 across Thinking + NoThinking**. Verify locally with `python scripts/verify_baseline_split.py` (CPU only). GSM8K/MATH indices are seed-42 fixed; **AIME is not sampled**. The earlier eight-question AIME subset exists only as a historical run, not a future split.

**Do not use `scripts/eval_baseline_subset.py` alone to define future AIME selection**: that script was intentionally preserved unchanged because `scripts/extend_aime_full.py` depends on its earlier run and saved manifest. Future evaluations should read the canonical split file or use the completed full-AIME combined results from `outputs/baseline_aime_remaining_seed42/summary_combined.json` when available.

## Paths and activation

- Repository: `/tc1home/UG/n2602710a/AdaptThink`
- Model: `model/DeepSeek-R1-Distill-Qwen-1.5B` (excluded from Git)
- Python environment: **`.venv`**, created using NTU's `module load python/3.10`. Do **not** activate the failed `$HOME/.conda/envs/adaptthink` path.
- Project GitHub: https://github.com/miyaliao/AdaptThink

```bash
cd /tc1home/UG/n2602710a/AdaptThink
module load python/3.10
source .venv/bin/activate
export PYTHONNOUSERSITE=1
which python
python --version
```

Validated installed versions:

| Component | Version |
| --- | --- |
| Python | 3.10 (`.venv`) |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.51.2 |
| vLLM | 0.8.5 |
| XFormers | 0.0.29.post2 |
| XGrammar | 0.1.18 |
| SymPy | 1.13.1 |
| pylatexenc | 2.11 |

The official score code `src/adapt_think_rm.py` also imports `math_verify`; its installation/compatibility is **not yet validated**. Do not run `pip install -r requirements.txt` unnecessarily just to reproduce inference.

## GPU / SLURM

- Verified compute node: `TC1N03`.
- GPU: **NVIDIA Tesla V100-PCIE-32GB**, ~31.73 GiB usable.
- NVIDIA driver **580.173.02**; `nvidia-smi` shows CUDA driver support **13.0**.
- Python/PyTorch uses bundled CUDA runtime **12.4**. The driver-reported 13.0 is *not* the installed PyTorch CUDA runtime.
- Slurm partition: `UGGPU-TC1`, QoS: `normal`; observed working request: **one GPU**, 4 CPUs, 16 GB RAM, 15–30 minutes.
- The user's previous normal QoS record indicated one GPU and a 6-hour job ceiling; recheck `MyTCinfo` when submitting larger jobs.
- **Run GPU code only in Slurm jobs, not on login host `CCDS-TC1`.**

Recommended Slurm header for a short inference test:

```bash
#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err
```

Inside the job (after `cd` to repository root):

```bash
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
```

## Verified vLLM long-context configuration

V100 supports FP16 but **not native BF16** and does **not** support FlashAttention-2. These settings succeeded for 17,408 max context:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="model/DeepSeek-R1-Distill-Qwen-1.5B",
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.80,
    max_model_len=17408,  # 1024 prompt + 16384 output budget
    max_num_seqs=4,
    enable_chunked_prefill=False,
    enforce_eager=True,
)

# Speed benchmark only; choose exact evaluation sampling separately.
params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)
```

**Critical:** The same 17,408-token setup with `enable_chunked_prefill=True` and `max_num_batched_tokens=4096` failed at generation time with `LLVM ERROR: Failed to compute parent layout for slice layout` (job 69983). Disabling chunked prefill fixed that failure (job 69987). Keep it disabled unless separately tested.

`max_num_seqs` is the primary concurrency parameter to benchmark (e.g., 4 / 8 / 16), with **enough submitted requests** to saturate it. `--cpus-per-task` is not the equivalent of vLLM concurrency.

## Benchmarks on one V100

| Job | Workload | Output tokens | Generation time | Aggregate output throughput | Result |
| --- | --- | ---: | ---: | ---: | --- |
| 69968 | Transformers, 1 short question, greedy | 256 | 7.91 s | 32.38 tok/s | Pass |
| 69980 | vLLM, 8 short questions, 4K context | 1,001 | 3.22 s | 310.80 tok/s | Pass |
| 69983 | vLLM, 17,408 context, chunked prefill ON | — | — | — | LLVM crash |
| 69987 | vLLM, 17,408 context, chunked prefill OFF | 1,001 | 4.09 s | 244.84 tok/s | Pass |
| 69992 | vLLM, 17,408 context, 4 MATH-500 prompts, max 4,096 output, temp 0.6 | 9,053 | 53.93 s | **167.88 tok/s** | Pass |

Model load time was ~25–32 s. These are **different workloads**; do not interpret the speed ratios as controlled comparisons. Long outputs up to 16K have **not** yet been benchmarked.

Existing local scripts/logs (may be untracked): `smoke_gpu.sbatch`, `smoke_69968.*`, `vllm_smoke.sbatch`, `vllm_smoke_69980.*`, `vllm_long_context.sbatch`, `vllm_long_69983.*`, `vllm_long_fix.sbatch`, `vllm_fix_69987.*`, `vllm_benchmark.sbatch`, `vllm_benchmark_69992.*`. GitHub does **not** automatically contain local Slurm logs.

## Original project datasets and historical subset experiment

Reuse repository datasets `data/test/gsm8k.json` (**1,319**), `data/test/math.json` (**500**), `data/test/aime.json` (**30**). Full evaluation: AIME repeated 16 times -> **2,299 generations/mode**, **4,598 total** across Thinking / NoThinking.

Historical initial quick experiment (superseded by the canonical split above): deterministic seed **42**, per-dataset 25% sampling: **330 GSM8K + 125 MATH-500 + 8 AIME** independent questions (round half-up for AIME). Keep the same sampled question IDs in both modes and retain 16 generations/AIME question (128 AIME outputs per mode) -> **1,166 outputs total**. Document exact sample IDs for reproducibility. This is a **subset study**, not a full published baseline.

Use the official scoring logic in `src/adapt_think_rm.py`, including the Thinking/NoThinking output convention and boxed-answer extraction. Beware: original `scripts/run_eval_hf.sh` also evaluates 7B and MMLU, assumes 8 GPUs, and uses author-specific paths; adapt it rather than running as-is. Persist generations incrementally, with resumable Slurm shards below the 6-hour limit.

## Fast rerun checklist

1. `git pull --ff-only origin main` and `source .venv/bin/activate`.
2. Submit an inference job with the SLURM header and three vLLM environment variables above.
3. Start from the **working, chunked-prefill-OFF** long-context configuration.
4. Verify local model path and official scorer imports before evaluation.
5. Record generated output tokens, elapsed seconds, throughput, mode and seed for each run.

## Reproducible concurrency sweep (added 2026-10-27)

The **only** purpose of this benchmark is selecting `max_num_seqs` before the 25% baseline evaluation.

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git pull --ff-only origin main
sbatch scripts/ntu_vllm_concurrency.sbatch
# After job ends, replace JOBID:
grep -E 'BENCH_RESULT|CONFIG_STATUS|SUMMARY|max_num_seqs=' vllm_concurrency_JOBID.out
```

The single Slurm job tests **4, 8 and 16** sequentially, rebuilding the engine between settings. All three use the **same 32 MATH-500 questions**, selected with seed 42, 17,408 max context, FP16, XFormers, 0.80 GPU utilization, **chunked prefill OFF**. For an apples-to-apples *speed-only* comparison, each question generates exactly **2,048 tokens** using `ignore_eos=True` (32 × 2,048 = 65,536 output tokens per configuration); this is **not** how baseline evaluation will generate/stop or grade responses. It is intentionally synthetic fixed-work benchmarking. See the vLLM 0.8.5 SamplingParams documentation for `ignore_eos` and per-prompt `seed`.

Results are written locally to `outputs/benchmarks/vllm_concurrency/seqs{4,8,16}_n32_tok2048_seed42.json`; Slurm logs remain locally as `vllm_concurrency_<JOBID>.out/.err`. These outputs are ignored by Git. Inspect the summary to select the fastest **successful** configuration. Do not claim an optimal full-16K setting until a representative 16K generation benchmark has also run.

### Results: job 70005 (fixed workload, 2026-10-27)

- Slurm: `COMPLETED`, exit `0:0`, 9 min 31 sec total including three engine initializations.
- 32 seeded MATH-500 prompts, 2,048 tokens per prompt, `ignore_eos=True`, **65,536 generated tokens per configuration**.

| `max_num_seqs` | Output tokens | Generation seconds | Aggregate tokens/s | Status |
| ---: | ---: | ---: | ---: | --- |
| 4 | 65,536 | 235.353 | 278.458 | Success |
| 8 | 65,536 | 143.512 | 456.659 | Success |
| **16** | **65,536** | **66.777** | **981.411** | **Success** |

**Recommended among tested settings:** `max_num_seqs=16`. This is a controlled fixed-length throughput benchmark, **not** proof that 16 simultaneous 16K sequences will fit or have the same speed. For true evaluation disable `ignore_eos`, keep paper decoding/stopping behavior, save outputs/seed/lengths, and benchmark at least one representative long-thinking workload before assigning a full-run runtime.

## Historical original 1.5B baselines, 25% subset (not the future AIME split)

**Protocol:** pretrained DeepSeek-R1-Distill-Qwen-1.5B, 330 GSM8K + 125 MATH-500 + 8 AIME questions (16 draws per AIME question), seed 42; **1,166 outputs total**. Thinking uses default model chat template; NoThinking appends the original project EOT token ID 151649 (`</think>`) to the prompt. Sampling uses temperature 0.6, top-p 0.95, top-k -1, 16,384 max response tokens (including forced prefix) and natural EOS. The original `src/adapt_think_rm.py` mathematical scorer is reused.

First run on SSH login node (CPU-only):

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install 'math-verify==0.9.0'
python -m pip check
python scripts/eval_baseline_subset.py --action preflight
```

Only after `PREFLIGHT_OK`, submit the GPU task:

```bash
sbatch scripts/ntu_baseline_subset.sbatch
```

After submission, closing SSH **does not stop Slurm**. The job ends on normal completion, time limit (5:45), failure or cancellation. A fresh submission of the same SBATCH skips saved completed task IDs. Every 32 outputs it fsyncs into `outputs/baseline_subset_25pct_seed42/{nothinking,thinking}.jsonl`. The manifest checks the dataset, sample IDs and configuration to avoid mixing different experiments.

```bash
squeue -u n2602710a
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
grep -E 'PREFLIGHT_OK|MODE_START|BATCH_DONE|MODE_DONE|SCORE' baseline_1p5b_JOBID.out
tail -n 30 baseline_1p5b_JOBID.err
cat outputs/baseline_subset_25pct_seed42/summary.json
```

Final results are `summary.json` and per-mode grade JSON files. Note: 981 tok/s was measured at fixed 2,048-token outputs, not full 16K; an actual 16K run could slow down, fail or require re-submission. This is a **25% subset adaptation**, not exact full-paper reproduction.

## AIME 2024 full-30 extension (after subset Job 70028)

Keep the original 330 GSM8K and 125 MATH-500 questions/mode unchanged. The original subset already generated 8 AIME questions x 16 draws x 2 modes (256 responses). The extension generates only the remaining 22 questions x 16 draws x 2 modes = **704 new responses**, then reuses all 256 original AIME responses for scoring across the 30-question AIME set.

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git pull --ff-only origin main
source .venv/bin/activate
python scripts/extend_aime_full.py --action preflight
# Only if AIME_EXPANSION_PREFLIGHT_OK appears:
sbatch scripts/ntu_aime_full.sbatch
```

The new generator imports existing `eval_baseline_subset.py`, preserving the same model, seed per question/repeat, chat template, forced NoThinking token, scoring and vLLM settings. The preflight checks old manifest and saved outputs; it refuses missing/incompatible sources.

New JSONL and summaries are saved in `outputs/baseline_aime_remaining_seed42/`. Each mode saves progress every 32 generations and can be resumed by resubmitting the same SBATCH after interruption/time limit. The original `outputs/baseline_subset_25pct_seed42/` is not modified. Final `summary_combined.json` contains GSM8K/MATH subset scores plus 30x16 full-AIME scores. This is a mixed-scope subset/full-AIME experiment, **not** an all-three-datasets full run.

```bash
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
grep -E 'AIME_EXPANSION_PREFLIGHT_OK|MODE_START|BATCH_DONE|MODE_DONE|FULL_AIME_SCORE|FINAL_SUMMARY' aime30_extend_JOBID.out
tail -n 25 aime30_extend_JOBID.err
cat outputs/baseline_aime_remaining_seed42/summary_combined.json
```

Closing SSH does not stop an accepted SLURM job. Re-submit after job time limit if needed. Full AIME may still require error analysis for large discrepancies, especially response truncation and scoring mismatches.
