# AdaptThink — SC4001 Reproduction & Extension

This is a course-project fork of **[AdaptThink: LLM Can Learn When to Think](https://arxiv.org/abs/2505.13417)** (EMNLP 2025), originally developed by [THU-KEG](https://github.com/THU-KEG/AdaptThink).

## Project Goal

For **NTU SC4001 Project G**, we study whether selecting **Thinking** or **NoThinking** adaptively improves mathematical reasoning accuracy under a fixed *average* output-token budget:

**Maximize accuracy, subject to mean LLM output tokens ≤ B.**

Our planned comparison uses **DeepSeek-R1-Distill-Qwen-1.5B** with the same evaluation protocol:

- **Baselines:** Always Fast, Always Slow, and budget-matched Random Router.
- **External Router:** A lightweight predictor chooses the mode; Qwen remains frozen.
- **Internal Router:** LoRA SFT teaches Qwen to choose its mode (RL is optional).

These router experiments are **planned**, not implemented in this fork yet. Router overhead will be reported separately from generated LLM tokens; budget thresholds will be chosen on validation data and evaluated on held-out test data.

## Current Work: AdaptThink Baseline Evaluation

We evaluate **DeepSeek-R1-Distill-Qwen-1.5B** in forced NoThinking and regular Thinking modes using the original mathematical scorer. This is a **subset/full-AIME inference baseline**, not a reproduction of AdaptThink's RL training.

| Dataset | Fixed questions | Generations per mode |
| --- | ---: | ---: |
| GSM8K | 330 / 1,319 (25%) | 330 |
| MATH-500 | 125 / 500 (25%) | 125 |
| AIME 2024 | **30 / 30 (all)** | 480 (16 per question) |
| **Total** | **485** | **935** |

Both modes together require **1,870 generations**. The canonical question indices, dataset hashes, and sampling seeds are recorded in [`data/splits/adaptthink_original_baseline_seed42.json`](data/splits/adaptthink_original_baseline_seed42.json) (base seed 42). Both modes use the same questions; mathematical scoring uses [`src/adapt_think_rm.py`](src/adapt_think_rm.py).

**Important:** `eval_baseline_subset.py` contains the historical **8-question AIME** run. `extend_aime_full.py` adds the other 22 questions and combines all **30**; the canonical split above, not the historical 8-question manifest, defines future evaluation.

## Baseline Results

| Dataset | NoThinking Accuracy | Thinking Accuracy | Mean Output Tokens (NoThinking / Thinking) |
| --- | ---: | ---: | ---: |
| GSM8K | 66.06% | 77.88% | 269 / 924 |
| MATH-500 | 67.20% | 84.80% | 856 / 3,964 |
| AIME 2024 | 10.83% | 25.00% | 6,013 / 12,489 |

Results use the original **1.5B model weights** (no SFT/RL). AIME accuracy is averaged over **30 questions × 16 generations per mode**, **not pass@16**. At the 16,384-token limit, AIME truncation rates were **17.1% (NoThinking)** and **47.7% (Thinking)**.

## Quick Start (NTU TC1)

See [`docs/NTU_HPC_ENV.md`](docs/NTU_HPC_ENV.md) for Python/vLLM setup, GPU settings, Slurm requirements, and detailed commands. The model weights must be available locally at `model/DeepSeek-R1-Distill-Qwen-1.5B` (not stored in Git). Slurm scripts contain environment-specific paths.

```bash
source .venv/bin/activate
python scripts/verify_baseline_split.py
python scripts/eval_baseline_subset.py --action preflight
sbatch scripts/ntu_baseline_subset.sbatch

# After the historical subset run finishes:
python scripts/extend_aime_full.py --action preflight
sbatch scripts/ntu_aime_full.sbatch
```

Results are saved locally (Git-ignored) under `outputs/baseline_subset_25pct_seed42/` and `outputs/baseline_aime_remaining_seed42/`. The combined AIME-30 summary is `outputs/baseline_aime_remaining_seed42/summary_combined.json`.

## Original Work & Attribution

- **Upstream code and full training instructions:** [THU-KEG/AdaptThink](https://github.com/THU-KEG/AdaptThink)
- **Paper:** [AdaptThink: LLM Can Learn When to Think](https://arxiv.org/abs/2505.13417)
- **Released models:** [THU-KEG Hugging Face collection](https://huggingface.co/collections/THU-KEG/adaptthink-682a1059aa9f5102c4fa0470)
- **License:** [MIT](LICENSE). Original copyright and license notices are retained.
