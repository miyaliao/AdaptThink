# Route E — Task-aware Prompt Representation Ablation (proposed; no GPU results yet)

**Date:** 2026-11-04 · **Status:** CPU tokenizer audit required before any full feature extraction.

## Research question

Can a fixed task cue appended to DeepSeek-R1-Distill-Qwen-1.5B's **P1 Assistant-only prefix** yield a more linearly recoverable Thinking-vs-NoThinking routing signal, without changing the frozen encoder or generating an answer?

The results from P0 / P1 / P2 motivate this as a *hypothesis*, not an established mechanism. We treat every Assistant-side cue as **partial assistant output**, not a system/user instruction.

## Predeclared prompts

All prompts begin with the *actual local tokenizer's* user message + P1 (`<｜Assistant｜>`). Suffixes:

| Mode | Added Assistant-side suffix | Purpose |
| --- | --- | --- |
| Existing P1 (reference) | *(none)* | Historical baseline; reuse existing cached predictions/features |
| `general` | `Problem summary in one word: "` | Generic task cue / semantic control |
| `difficulty` | `Problem difficulty in one word: "` | Difficulty-oriented cue |
| `routing` | `Reasoning mode in one word: "` | Routing-related cue |

These are *candidate* suffixes. **No actual DeepSeek tokenizer length claims are made before the CPU audit passes.** The three added cues use a common final phrase to reduce changes in last-token identity. The general cue is a *semantic control*, not a pure no-op control.

Before GPU: inspect **every** original 3,000 training question and 450 validation questions with the server-local tokenizer, fail if any variant exceeds the configured `--max-length` (default 1024), if their last token IDs differ, or if any per-question span between added-prompt token lengths exceeds 2. If this fails, revise suffixes **without reading validation accuracy** and re-audit. P1 is excluded from the variant length-match constraint because it contains no appended task cue.

## Fixed modeling protocol

- Encoder: **frozen** DeepSeek-R1-Distill-Qwen-1.5B, `AutoModel` in FP16, final RMSNorm last-real-prompt-token hidden vector (1,536 dimensions). One forward pass, no decoding.
- Data: original strict 3,000 train / 450 validation, original SHA256 and sample IDs/order checked; no held-out test access.
- Label: `1[p_NoThinking_strict >= p_Thinking_strict]`, tie to NoThinking.
- Router: independent Logistic Regression for each suffix, train-only standardization, BCE + `1e-3 * ||w||^2`, full-batch LBFGS, at most 200 steps.
- Do not add sample-specific answers, difficulty, success probabilities, mode outcomes, or labels to the input prompt.
- Existing P1 / ModernBERT Classification / Global Random are references; new features and scores are written to **separate** folders.

## Main evaluation: full Accuracy–Token Curve

The user-facing *budget* is **expected average generated tokens per question** across all validation questions, *not* each decode's `max_new_tokens` cap. Keep the existing inference protocol and cached per-question scores/outcomes; no new answer generation for this evaluation.

**Report points:** 900, 1100, 1300, 1500, 1700, 1900, 2100 tokens/question.

**Full curve:** all **451** top-k NoThinking decisions, ranked using only the router score. For fair common-budget comparisons, follow the existing `route_e_equal_budget.py` protocol: sort the 451 policies by their measured validation output-token cost; interpolate cost-adjacent policies using a randomized policy mixture. Display 241 grid points across [900,2100] by default, retaining dense source policy curves. **This is offline descriptive evaluation:** interpolation uses observed validation output lengths and is not a deployable online budget controller.

Compare **six lines** on the same PNG/SVG figure: Global Random, full-fine-tuned ModernBERT Classification (must provide dense 450-question scores; six-point approximate fallback disallowed), Frozen DeepSeek P1, and General/Difficulty/Routing-aware. Verify FT 50% NT = 73.56% at ~1092 tokens and P1 50% NT = 76.44% at ~1427 tokens before accepting any graph. Recompute all accuracies from original strict per-mode cached outcomes rather than hardcoding chart values.

Report per-budget accuracies and **normalized area under the Accuracy–Token Curve** over the *predeclared* [900,2100] interval, i.e., mean expected accuracy over that interval. It is **not** the same quantity as validation binary-routing ROC AUC, which remains diagnostic. Interpret narrow differences cautiously; consider paired uncertainty intervals before superiority claims.

**Limitations:** the curve excludes DeepSeek prompt-feature extraction/prefill, CPU router computation and communication costs; model generation cap remains fixed. The P1 baseline was previously selected based on the same validation split, making comparisons exploratory. A genuine `max_new_tokens` cap sweep would require new generation and scoring, not this offline replay.

## CPU-only preparation (run from repository root)

```bash
source .venv/bin/activate
python scripts/route_e_task_prompts.py self-test
python scripts/route_e_task_prompts.py inspect-template
python scripts/route_e_task_prompts.py audit
python scripts/route_e_task_prompt_curve.py --self-test
```

If `PROMPT_AUDIT_FAIL`, **do not submit GPU jobs**. Inspect the SHA-named JSON under `results/route_e/task_prompt/` and modify the three candidate suffixes before repeating the audit.

## GPU extraction (only after CPU audit and approval)

Extract each mode sequentially on a single TC1 GPU. The script exposes these two stages per mode:

```bash
python scripts/route_e_task_prompts.py extract --mode general --smoke
python scripts/route_e_task_prompts.py fit --mode general --smoke
# After reviewing smoke outputs and scheduling appropriate sbatch jobs:
python scripts/route_e_task_prompts.py extract --mode general
python scripts/route_e_task_prompts.py fit --mode general
```

Replace `general` by `difficulty` and `routing` to process other arms. **Do not run GPU extraction directly on a login node**; these commands must be wrapped in a TC1 Slurm job and obey the single-GPU quota.

When all three full runs finish, evaluate on CPU:

```bash
python scripts/route_e_task_prompt_curve.py
```

Outputs (paths relative to repository root):

- `outputs/route_e/task_prompt/{general,difficulty,routing}/full/{train,val}.npz`: frozen embeddings and verified sample IDs.
- `outputs/route_e/task_prompt/{general,difficulty,routing}/full/validation_p_nt.json`: independent logistic scores.
- `results/route_e/task_prompt/curve/{accuracy_token_curve.png,accuracy_token_curve.svg,accuracy_token_curve.json,accuracy_token_curve.csv,budget_points.csv}`: graph, all grid points and table.

Do not commit large hidden-state caches or linear-probe weights. Keep compact experiment metadata, results and figures after full runs have been checked.
