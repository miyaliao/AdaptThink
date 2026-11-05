# LRM Zero-shot Self-Assessment vs Supervised P1 Router

**Status: original 450-validation zero-shot assessment completed (2026-11-05); same-state head control is the next experiment.**
**Research question:** For the *same frozen* DeepSeek-R1-Distill-Qwen-1.5B, can a training-free language-model readout rank which math problems should use NoThinking as well as the previously supervised P1 hidden-state Logistic Regression router?

## 1. First-phase scope

Implement **only the existing P1 binary target**:

\[
y_N(x) = \mathbf{1}[p_N(x) \ge p_T(x)]
\]

Ties count as NoThinking. This is *accuracy preference*, **not** token-budget-optimal routing. The full study will later separate (a) per-mode success probability, (b) per-mode expected generated tokens and (c) the budget-constrained routing decision. Do not interpret a sigmoid over letter logits as actual correctness probability.

- **Existing head:** frozen DeepSeek final 1,536-d P1 assistant-only representation + trained Logistic Regression. Reuse its **saved per-question** validation scores: `outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json`.
- **New self-assessment:** construct a different *user-side* question asking which strategy has higher final-answer correctness. Append **P1 assistant-only prefix** and the partial completion `Choice:`. Use the checkpoint's own pretrained `AutoModelForCausalLM` LM Head to score **one-token continuations** `" A"` and `" B"`.
- Original option order (primary): A = Thinking; B = NoThinking. Score \(s_N=z_B-z_A\), with larger meaning higher NoThinking preference. Swap order (diagnostic): A = NoThinking; B = Thinking; score \(s_N=z_A-z_B\). The swapped prompt requires a second forward pass; the primary comparison uses **one** prompt and **one** forward pass. Their average is diagnostic only.
- Check **actual concatenated-token boundaries** with the *local* tokenizer on all strict 3,000 training / 450 validation questions; fail before GPU if `prompt + " A"` or `prompt + " B"` does not append exactly one token. **Never guess** the A/B IDs. Verify max input length, template signature, SHA256, sample IDs. No automatic truncation.
- Record raw A/B logits, binary normalized probability, **full-vocabulary probability mass on A+B** and input tokens. If A+B has negligible full-vocab mass, the score is a *forced-choice probe*, not evidence that the LRM naturally outputs the letter without a constrained decoder.
- **No answer generation, no model fine-tuning, no new labels**, and no held-out test reads.

**Important confound:** the old P1 Logistic and the new LM-head self-assessment use **different prompts** and therefore different hidden states. This is a comparison of two complete systems, *not* a controlled test of the readout head alone.

## 2. Target wording

The script prompts the LRM to assess **which mode is more likely to produce a correct final answer**, explicitly ignoring speed and token costs, and instructing ties to NoThinking. It does not supply temperature, top-p or numerical success rates in the prompt: the cached outcome labels already correspond to the fixed external generation protocol.

The script scores the next-token positions using `AutoModelForCausalLM` and the right-padded batch's **last valid token**. It **does not** extract the hidden state as the self-assessment's final output; the old supervised head is the hidden-state method.

## 3. Run on NTU TC1

First synchronize the new file without overwriting any experimental artifacts. The server `main` branch has previously diverged from `origin/main`, so prefer a narrow restore instead of a failing `git pull --ff-only`:

```bash
cd ~/AdaptThink
git fetch origin main
git restore --source=origin/main --worktree -- \
  scripts/route_self_assessment.py \
  docs/SELF_ASSESSMENT_RESEARCH.md

source .venv/bin/activate
python scripts/route_self_assessment.py self-test
python scripts/route_self_assessment.py audit
```

**Stop if the audit fails.** Inspect the actual tokenizer's candidate boundaries and full prompt before altering anything. If the new instruction makes some inputs exceed `1024` tokens, inspect the lengths before deciding whether to use a higher consistent `--max-length`; never silently drop samples.

Then run a **32-question training-only GPU smoke** from Slurm (do not run directly on a login node). Only one GPU job at a time; there is also a 2-submitted-job QoS cap:

```bash
sbatch --partition=UGGPU-TC1 --qos=normal --nodes=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=24G \
  --time=01:00:00 --job-name=self_mode_smoke \
  --output=self_mode_smoke_%j.out --error=self_mode_smoke_%j.err \
  --wrap='set -eu; cd /tc1home/UG/n2602710a/AdaptThink; . .venv/bin/activate; \
  export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false; \
  python -u scripts/route_self_assessment.py extract --split train --smoke'
```

Look for `SELF_EXTRACT_DONE split=train smoke=1 records=32`. Smoke is **not** a validation performance estimate.

Only after confirming smoke and fixing no post-hoc prompt choices based on validation accuracy, run the full 450 validation forward passes using the same Slurm wrapper, changing the job name/output paths, `--time`, and final command to:

```bash
python -u scripts/route_self_assessment.py extract --split val
```

CPU-only evaluation after full validation extraction:

```bash
python scripts/route_self_assessment.py evaluate
```

This prints ROC AUC, Accuracy, Balanced Accuracy, selected NoThinking fraction, letter-option mass, and seven cost-matched budget comparisons (900 / 1100 / 1300 / 1500 / 1700 / 1900 / 2100 generated tokens per question).

The budget replay uses the existing 450 strict outcomes, all 451 top-k policies and the established **cost-adjacent interpolation** procedure; it does not redo math generation. The seven budgets compare **generation-only** cost; separately report the extra LRM forward-pass time and input tokens. It is *not* a fixed max_new_tokens cap sweep.

## 4. Output artifacts

- `outputs/route_e/self_assessment/mode/smoke/train_predictions.json`
- `outputs/route_e/self_assessment/mode/full/val_predictions.json`
- `results/route_e/self_assessment/mode_token_audit_<prompt_hash>.json`
- `results/route_e/self_assessment/mode_val.json`
- `results/route_e/self_assessment/mode_budget_points.csv`

All validation results are **exploratory** because the original 450-question split has been used repeatedly to develop previous routing methods. Do not tune the self-assessment wording or switch between original/swapped based on its final Accuracy–Token curve. Keep any future calibration or task-prompt search within training-only data, and use a separate untouched test set for confirmatory results.

**Next, only after this baseline is valid:** add distinct success-rate prediction and token-length prediction tasks. They require new target wording, separate checks for numerical probability interpretation, and matched success/cost regression head baselines—not merely reinterpreting the mode-choice logits.


## 5. Recorded zero-shot result (450-val; exploratory)

The original A/B-mode zero-shot LM Head returned:
- Self original: AUC 0.5300; classification accuracy 16.89%; balanced accuracy 50.13%; NoThinking predictions 0.2%.
- Self swapped (diagnostic): AUC 0.4906; accuracy 20.89%; balanced accuracy 51.47%; NoThinking predictions 5.1%.
- Self average (diagnostic only): AUC 0.5525; accuracy 16.67%; NoThinking predictions 0%.
- P1 Logistic (different-prompt reference): AUC 0.6110; accuracy 54.67%; balanced accuracy 59.47%; NoThinking predictions 49.1%.

In the fixed-budget offline replay, Self original ranked questions competitively at some budgets despite weak direct classification; this is not autonomous reasoning-budget allocation. The option-order sensitivity and exclusion of extra self-assessment prefill costs prevent strong end-to-end claims.

**Next:** Same-state LM Head vs supervised linear readout, strictly matching the original self-assessment prompt and final hidden state. See [SELF_ASSESSMENT_HEAD_CONTROL.md](SELF_ASSESSMENT_HEAD_CONTROL.md). Do not choose the swapped condition or tune wording on these validation results.
