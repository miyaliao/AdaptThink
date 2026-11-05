# Same-Prompt, Same-Hidden-State Head Control

**Status: code committed; TC1 smoke is still pending.**

## Question and controlled setup

Can a trained linear readout extract more information about the empirical Thinking-vs-NoThinking label from the very same frozen representation that the original LM Head uses for zero-shot A/B mode choice?

Use the **unchanged original** self-assessment prompt from scripts/route_self_assessment.py: A=Thinking; B=NoThinking. Append the existing P1 assistant start and Choice: completion. Do not select wording or option order using validation outcomes.

One frozen DeepSeek-R1-Distill-Qwen-1.5B forward per problem yields:
- The **final post-RMSNorm 1536-dimensional last-real-prompt-token hidden state**.
- The original checkpoint **LM Head** A/B logits. Score logit(B) - logit(A).
- A **trained linear Logistic Regression Head** on the same state, trained on 3,000 strict training labels y=1[p_N >= p_T] (ties => NoThinking).

The probe uses the same Route E hard-label BCE + L2=1e-3 and LBFGS max_iter=200; scaling is fit only on train. No LRM weight updates, no math answer generation, no new labels, no hyperparameter search on the 450 validation set.

Verify that calling model.get_output_embeddings() on the extracted vector reproduces the same forward's candidate LM logits. On validation, compare the extracted LM margin against the previously saved original self-assessment margins and abort if the maximum discrepancy exceeds 0.10 (allowing small batch/FP16 variation).

Compare AUC, classification accuracy, balanced accuracy and predicted NoThinking fraction, plus the existing 451-policy equal-generated-token budget replay. The old P1 Logistic results may be displayed as **a different-prompt external reference**, not a matched-head control.

All 450-val measurements remain exploratory: the split has been used repeatedly in earlier method development. Better supervised decoding does not prove calibrated self-knowledge or autonomous reasoning-budget control.

## Stage 1 — CPU self-test and training-only smoke

Update only the new paths on TC1:

~~~bash
cd /tc1home/UG/n2602710a/AdaptThink
git fetch origin main
git restore --source=origin/main --worktree -- \
  scripts/route_self_assessment_head_control.py \
  docs/SELF_ASSESSMENT_HEAD_CONTROL.md
. .venv/bin/activate
python scripts/route_self_assessment_head_control.py self-test
~~~

Run one Slurm job (32 train samples only):

~~~bash
sbatch --partition=UGGPU-TC1 --qos=normal --nodes=1 \
  --gres=gpu:1 --cpus-per-task=4 --mem=24G \
  --time=01:00:00 --job-name=self_head_smoke \
  --output=self_head_smoke_%j.out --error=self_head_smoke_%j.err \
  --wrap='set -eu; cd /tc1home/UG/n2602710a/AdaptThink; . .venv/bin/activate; \
  export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false; \
  python -u scripts/route_self_assessment_head_control.py extract --split train --smoke'
~~~

Gate: HEAD_EXTRACT_DONE split=train smoke=1 shape=(32, 1536), no NaNs/OOM and a very small candidate-logit reconstruction discrepancy.

Then on CPU, check that the linear head can fit these train-only features:

~~~bash
python scripts/route_self_assessment_head_control.py fit --smoke
~~~

This only reports **in-sample training diagnostics**, not validation performance.

## Stage 2 — full extraction and CPU evaluation (after smoke)

Run the two full extraction commands in **separate** Slurm GPU jobs, sequentially, with one GPU each. Start with batch-size 4. Choose each Slurm time limit using observed smoke throughput rather than a guessed runtime. Do not run the full inference on a login node.

~~~bash
python -u scripts/route_self_assessment_head_control.py extract --split train
python -u scripts/route_self_assessment_head_control.py extract --split val
~~~

After the full 3000-train and 450-val features are present:

~~~bash
python scripts/route_self_assessment_head_control.py fit
~~~

The CPU fit reports two matched-head methods (Original_LM_Head and Same_State_Linear_Head) and the separate P1 reference for budgets 900/1100/1300/1500/1700/1900/2100.

## Outputs

- outputs/route_e/self_assessment/head_control/smoke/train.npz
- outputs/route_e/self_assessment/head_control/full/train.npz
- outputs/route_e/self_assessment/head_control/full/val.npz
- results/route_e/self_assessment/head_control/smoke/report.json
- results/route_e/self_assessment/head_control/full/linear_head.npz
- results/route_e/self_assessment/head_control/full/validation_p_nt.json
- results/route_e/self_assessment/head_control/full/report.json
- results/route_e/self_assessment/head_control/full/budget_points.csv

Prefill overhead is recorded separately in metadata. The equal-budget comparison includes **only answer generation tokens**, not this additional self-assessment forward.

## Existing zero-shot baseline

For 450 validation questions: Original LM Head AUC=0.5300, classification accuracy=16.89%, predicted NoThinking fraction=0.2%; swapped ordering AUC=0.4906, predicted NoThinking fraction=5.1%. The P1 Logistic different-prompt reference scored AUC=0.6110. The original LM score sometimes ranked problems effectively for a budget-routing replay, despite its poor direct classification; it is not evidence that it autonomously chose the budget correctly.
