# AdaptThink — SC4001 Reproduction & Extension

This is a course-project fork of **[AdaptThink: LLM Can Learn When to Think](https://arxiv.org/abs/2505.13417)** (EMNLP 2025), originally developed by [THU-KEG](https://github.com/THU-KEG/AdaptThink).

## Project Goal

For **NTU SC4001 Project G**, we study whether selecting **Thinking** or **NoThinking** adaptively improves mathematical reasoning accuracy under a fixed *average* output-token budget:

**Maximize accuracy, subject to mean LLM output tokens ≤ B.**

Our planned comparison uses **DeepSeek-R1-Distill-Qwen-1.5B** with the same evaluation protocol:

- **Baselines:** Always Fast, Always Slow, and offline Random Router (implemented; budget-matched comparison planned).
- **External Router:** A lightweight predictor chooses the mode; Qwen remains frozen.
- **Internal Router:** LoRA SFT teaches Qwen to choose its mode (RL is optional).

The **Random Router baseline is implemented** as an offline replay of the paired baseline grades. External and Internal Routers remain planned. Router overhead will be reported separately from generated LLM tokens; budget thresholds will be chosen on validation data and evaluated on held-out test data.

## Shared splits (frozen — do not resample)

Both routes (external router, internal router) must train and tune on the *same*
data, otherwise the two Accuracy–Token curves are not comparable. Three frozen
files, disjoint by construction:

| File | Rows | Role | sha256 (first 12) |
| --- | ---: | --- | --- |
| `data/splits/sft_train_3000_seed42.json` | 3,000 | **training** (DeepScaleR, stratified) | `9c71b21f3272` |
| `data/splits/adaptthink_val_seed142.json` | 450 | **validation** (GSM8K 300 + MATH-500 150) | `36883adfe2d4` |
| `data/splits/adaptthink_original_baseline_seed42.json` | 485 | **test** (GSM8K 330 + MATH-500 125 + AIME 2024 30x16) | `831f35187368` |

Rules:

1. Never resample and never hand-edit these files. Regenerating changes the hash
   and silently invalidates every result produced against the old one.
2. Training data is never used to pick thresholds or checkpoints; that is the
   validation split's only job. Test numbers are reported once, at the end.
3. Both routes consume the same `sample_id`s and the same labels. Only the model
   and the feature representation differ.
4. Verify a copy with `shasum -a 256 data/splits/sft_train_3000_seed42.json`.

The training manifest removes exact duplicates (3) and *near*-duplicates (27 at
character 8-gram similarity >= 0.80) against all 935 held-out questions. The
near-duplicate layer is not cosmetic: it recovered a genuine duplicate that exact
text matching missed — `deepscaler:6417` and `math:489` are the same problem with
the same answer 15, differing only in whether a comma sits inside `$...$`.

```bash
python scripts/make_val_split.py --force                 # regenerate validation (8 KB)
python scripts/make_sft_manifest.py --n-total 3000 --pilot 600   # regenerate manifest (~76 s)
python scripts/check_split_quality.py                    # must print PASS (worst 0.667)
```

`data/splits/sft_train_3000_seed42.json` holds, per problem: `sample_id`
(traceable back to a `deepscaler.json` row), `problem`, `answer`, `acc_thinking`
(the released K>=16 reference accuracy), `bucket`, `order_index` and the fixed
`nothinking_seeds` used to sample it. Labels (`acc_nothinking`, `label`, `gain`)
are joined on later by `scripts/build_router_labels.py` and stored separately in
`data/sft/labels_3000.jsonl`; labels never change which problems are in the split.

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


## Offline Random Router baseline (CPU only)

The original model is **not** rerun. This script reads the paired grades already
tracked under `results/baseline_1p5b/` and checks their IDs against the canonical
split. It evaluates NoThinking probabilities **0.10, 0.25, 0.50** plus both
always-on endpoints. AIME has **one route per question**, not one route per
generation: all 16 draws of each question use the same selected mode.

```bash
cd ~/AdaptThink
git pull --ff-only origin main
source .venv/bin/activate
# Only if matplotlib is not already installed (required for PNG only):
python -m pip install matplotlib
python scripts/eval_random_router.py
```

By default, each probability is replayed across 100 deterministic router seeds
(starting at 42). The script reports per-dataset mean Accuracy, mean output
tokens, fraction routed NoThinking, fraction truncated, and relative savings,
and writes:

- `results/random_router_1p5b/summary.json` — analytical random-mixture expectations,
  100-seed simulation means, standard deviations, and seed percentiles;
- `results/random_router_1p5b/per_seed.json` — all 900 dataset × probability × seed outcomes;
- `results/random_router_1p5b/accuracy_tokens.png` — separate per-dataset
  Accuracy–Token panels with the analytical random baseline line.

To compute JSON outputs without matplotlib, use
`python scripts/eval_random_router.py --skip-plot`.

**Interpretation:** the 2.5%/97.5% quantiles and error bars represent variation
across **router random seeds**, not confidence intervals for test-set sampling
or AIME question difficulty. Use the analytical expectation line when comparing
future learned routers at an equal token budget. The Random Router never uses
gold answers or the graded outputs to decide its route; those cached values are
used only to evaluate the chosen mode.

## Original Work & Attribution

- **Upstream code and full training instructions:** [THU-KEG/AdaptThink](https://github.com/THU-KEG/AdaptThink)
- **Paper:** [AdaptThink: LLM Can Learn When to Think](https://arxiv.org/abs/2505.13417)
- **Released models:** [THU-KEG Hugging Face collection](https://huggingface.co/collections/THU-KEG/adaptthink-682a1059aa9f5102c4fa0470)
- **License:** [MIT](LICENSE). Original copyright and license notices are retained.
