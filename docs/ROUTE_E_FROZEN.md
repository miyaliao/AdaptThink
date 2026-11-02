# Route E — Frozen representations for adaptive reasoning routing

Status (2026-11-02): **code prepared; not yet run on NTU GPU**. This is a
strictly controlled representation ablation, linked to
[Route A's five-objective comparison](ROUTE_A_PLAN.md).

## Research question and fixed decisions

**Can pre-generation representations from the frozen
DeepSeek-R1-Distill-Qwen-1.5B model improve cost-matched routing over frozen
ModernBERT-base representations?** The first pass varies *only the encoder*:

| Arm | Input / feature | Parameters updated | Prediction |
| --- | --- | --- | --- |
| A (historical reference) | ModernBERT-base fine-tuned with classification | Full encoder + head | NoThinking score |
| B (new) | Frozen ModernBERT-base, first-token/CLS hidden vector | Logistic head only | P(NoThinking) |
| C (new) | Frozen DeepSeek 1.5B, last prompt-token hidden vector | Logistic head only | P(NoThinking) |

Arms B and C use the identical **hard binary supervision**,
**full-batch logistic BCE + L2**, and the existing Route A cost-matched
evaluation code. A is a system-level reference, not a strict frozen-only
ablation. Optional TF-IDF baseline and layer/pooling sweeps are deferred.

Classification target (NoThinking = 1, ties go to NoThinking):
`y = 1[p_nothinking_strict >= p_thinking]`.
No regression labels, lambda, oracle feature, solution, future output length,
generated answer, or NoThinking/Thinking result is used as router input.
The DeepSeek weights are not updated, nor are generated answers repeated.

## Mandatory protocol and leakage guards

The script hard-fails unless the original strict source SHA256 matches:

| Split | File | Count | Expected SHA256 |
| --- | --- | ---: | --- |
| Train | results/router_labels_1p5b/all_3000_labels.jsonl | 3000 | 761dcbb6338de1a3c71def382089c83e8024e0e55af6334dd974963e8d7e0f2c |
| Validation | results/val_450_labels.jsonl | 450 | 4c1e984d3744cc089dcc963a30af0ef3d1ed9808c4bfd245a97e48787d501461 |

The original strict train file is **not tracked on current main**; see
[Route A restoration instructions](ROUTE_A_PLAN.md#6-reproducing-route-a-experiments-on-ntu-tc1).
Do not substitute the 2,777 compliant examples or repaired results.
The 450 validation examples consist of GSM8K 300 and MATH-500 150.
The script shares `read_labels`, `curve_point`, `endpoints`, and
budget-matched Global Random logic with `scripts/train_modernbert_delta.py`.
For validation, that loader uses `p_nothinking_strict` if present and
otherwise the stored strict `p_nothinking`, consistent with Route A.
The canonical 485-example test set is not read or accessed by these scripts;
do not use it for layer, classifier, or hyperparameter decisions.

The script reports sample-ID overlap and normalized exact text overlap
between train/validation (the latter is disclosed rather than silently
deleting examples, to preserve Route A's fixed protocol). Train and
validation feature caches carry their original strict file SHA256.
No silent dropping, no hidden truncation: an overlength prompt raises an
error with the offending example ID. Increase `--max-length` if necessary,
using the same maximum for B and C.

## Implementation choices

- **ModernBERT**: use original `answerdotai/ModernBERT-base` weights
  and raw question text, encode with `AutoModel`, take first/CLS token vector.
- **DeepSeek**: use the frozen server-local
  `model/DeepSeek-R1-Distill-Qwen-1.5B` backbone (`AutoModel`),
  user chat template + generation prompt, take the last real prompt token.
  This performs prefill **only** and never decodes a token.
- Both are evaluated with right padding and attention mask, FP16 on the
  single V100, `max_length=1024` (error if exceeded),
  batch size 8 (adjust via Slurm env vars). Both embeddings are cached as
  Float16 without labels or responses; records contain sample IDs and metadata.
- **Head**: standardize features using training-only means/stds; one
  affine score + sigmoid, full-batch hard-label BCE + coefficient `1e-3`
  times squared L2 norm of weights, trained with CPU LBFGS for at most
  200 iterations, with zero initialization. No selection on validation AUC.
- **Evaluation**: rank by predicted P(NoThinking), send the top fraction
  to NoThinking, replay per-question frozen-mode empirical success rates
  and token costs using Route A's `curve_point`. Shares:
  0%, 10%, 25%, 50%, 75%, 100%; reports overall, GSM8K, and MATH.
  Compare **uplift against Random at the same observed mean generated tokens**,
  not at the same NoThinking share. AUC is diagnostic only.
- **Overhead**: cache records tokenization, per-question GPU forward
  time, complete extraction time, and batch CPU head latency separately.
  Generated-token curve **does not include router overhead**. Decoder
  backbone prefill is extra work unless a compatible inference stack
  explicitly proves safe reuse. Do not claim zero overhead or automatic
  KV-cache reuse.
- No gradient updates to either encoder; no alternative loss,
  data split, label repair, or test-driven tuning in this phase.

## Run on NTU TC1 (one GPU at a time)

First inspect the working tree and preserve the server's restored untracked
strict labels before pulling. Switch to the experiment branch without deleting
or overwriting Route B artifacts.

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git status --short
git fetch origin
git switch feature/route-e-frozen-probes-20261102
git pull --ff-only origin feature/route-e-frozen-probes-20261102
source .venv/bin/activate
python scripts/route_e_frozen.py self-test
python scripts/route_e_frozen.py check-data

# First: two separate small smoke tests (32 train + 24 validation)
sbatch --export=ALL,E_ENCODER=modernbert,E_SMOKE=1 scripts/ntu_route_e.sbatch
# After the first job finishes, separately:
sbatch --export=ALL,E_ENCODER=deepseek,E_SMOKE=1 scripts/ntu_route_e.sbatch

# Once both smoke jobs pass, launch full extraction + probe, sequentially:
sbatch --export=ALL,E_ENCODER=modernbert,E_SMOKE=0 scripts/ntu_route_e.sbatch
# After completion:
sbatch --export=ALL,E_ENCODER=deepseek,E_SMOKE=0 scripts/ntu_route_e.sbatch
```

To inspect results:
```bash
squeue -u "$USER"
# Substitute actual job ID
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
grep -E "ROUTE_E_VAL|ROUTE_E_FULL_VALIDATION|FEATURES_SAVED|ROUTE_E_DONE" route_e_JOBID.out
```

Jobs are independent, use one GPU, and have a default four-hour walltime.
If there is a CUDA OOM, lower `E_BATCH_SIZE` (e.g. 4); if max token length
is rejected, raise `E_MAX_LENGTH` consistently for B and C. This must
create a separate feature cache/configuration, not silently reuse an older
incompatible one. To replace failed partial outputs, use the extractor
`--overwrite` option deliberately after inspecting the files.

## Artifacts and result interpretation

| Artifact | Path |
| --- | --- |
| Frozen train/validation features | outputs/route_e/{modernbert,deepseek}/{smoke,full}/{train,val}.npz |
| Logistic probe weights and scaler | outputs/route_e/{encoder}/{smoke,full}/linear_probe.npz |
| Validation predicted NoThinking scores | outputs/route_e/{encoder}/{smoke,full}/validation_p_nt.json |
| Accuracy–Token full report | results/route_e/{encoder}_val.json, .csv |
| Smoke-only report | results/route_e/{encoder}_smoke.json, .csv |

Large embeddings and weights live under ignored `outputs/`; do not commit
them. The compact `results/route_e/` files may be committed **after**
completed full runs. The smoke report is not a final validation result.

At the common *50% NoThinking share*, historical full-fine-tuned
Classification had **73.56% accuracy, 1,092 output tokens, +2.37 pp vs
cost-matched Global Random**; compare both new arms at their individually
measured token budgets. Do not assert a clear encoder winner solely from
different 50%-share accuracies. Prefer the whole curve and common-budget
comparison, and keep extraction/forward latency visible. Repeated runs or
bootstrap confidence intervals are advisable before claiming small gains.

## Caveats and follow-ups

1. A strict encoder B/C comparison uses identical labeled questions,
   head, penalty, routing policy and evaluation, but their tokenization
   and native representation position necessarily differ.
2. The small labeled sample and ties in empirical success rates motivate
   frozen probes but do **not** establish full fine-tuning overfit.
3. Prefill representation generation cost and communication overhead may
   dominate a small-model serving deployment; measure end-to-end latency
   before translating output-token savings into speedups.
4. Record actual GPU/software environment, ModernBERT checkpoint
   revision (cached Hugging Face revision), and DeepSeek checkpoint
   provenance in the completed experiment log.
5. If both frozen probes lose, first inspect calibration, representation
   quality, label noise, and linear separability; MLPs, layer sweeps, and
   altered objectives belong to **later** experiments, not this control.
