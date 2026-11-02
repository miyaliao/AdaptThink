# Route E — DeepSeek prompt-prefix ablation (P1/P2)

Status (2026-11-02): **implementation prepared, GPU experiments not run**.
This adds two prompt variants to [the Route E frozen probe study](ROUTE_E_FROZEN.md).
The P0 baseline code and its files are not modified.

## Hypothesis and experiment arms

The official **DeepSeek-R1-Distill-Qwen-1.5B** tokenizer template
inserts `<｜Assistant｜><think>\n` when
`add_generation_prompt=True`. False leaves only the rendered user message.

| Arm | Encoding / final prompt content | Status |
| --- | --- | --- |
| P0 | official user chat prompt + `<｜Assistant｜><think>\n` | Existing DeepSeek Full, no new extraction |
| P1 (`assistant`) | official user chat prompt + `<｜Assistant｜>` | New: isolate `<think>\n` effect |
| P2 (`user_only`) | official user chat prompt; `add_generation_prompt=False` | New: remove full assistant prefix |

The new script derives P1 from the *loaded tokenizer's* P2 and checks
that P0 exactly equals `P2 + <｜Assistant｜><think>\n`.
If the local model's template differs from the expected official form,
**execution fails before loading the transformer model**. It also verifies
that tokenize=False followed by tokenization without extra special tokens
matches Hugging Face's direct `tokenize=True` pathway.

These are three different *input-position representations*, not claims
about what reasoning the model actually performed. All are **prefill only**,
28-layer backbone final hidden state / last real prompt token (1536 dims),
without generating answers.

## Controlled variables

- Same original strict 3,000-question training set and 450-question validation set
  with hard labels `1[p_nothinking_strict >= p_thinking]`.
- Same local DeepSeek checkpoint, `AutoModel`, FP16, right padding,
  maximum 1024 input tokens (no silent truncation), and batch size 8.
- Same P0 linear logistic regression implementation from
  `scripts/route_e_frozen.py`, with train-only standardization,
  L2 coefficient `1e-3` and LBFGS at most 200 iterations.
- Same P0 ranking protocol and strict Route A cost-matched Global Random
  evaluation from `scripts/train_modernbert_delta.py`.
- Same 0%, 10%, 25%, 50%, 75%, and 100% NoThinking fractions.
- No use of canonical test data; no label repair, cost-aware training,
  generation, or encoder parameter updates.

Note: because the final token itself changes between P0/P1/P2,
differences cannot automatically be attributed to intrinsic model
reasoning ability or model-internal preference. The study measures
representation sensitivity to prompt prefixes.

## Execution order on NTU TC1

**First finish P0 ModernBERT Full, then P0 DeepSeek Full**, on the existing
`feature/route-e-frozen-probes-20261102` branch. Never run the two jobs
simultaneously under the single-GPU quota. Confirm both logs and
saved P0 result files.

After both have finished, switch branches:

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git status --short
git fetch origin
git switch feature/route-e-prefix-ablation-20261102
git pull --ff-only origin feature/route-e-prefix-ablation-20261102
source .venv/bin/activate

# Check exact local tokenizer template BEFORE GPU feature extraction
python scripts/route_e_prompt_ablation.py self-test
python scripts/route_e_frozen.py check-data
python scripts/route_e_prompt_ablation.py inspect-template

# Then submit one GPU job at a time
sbatch --export=ALL,E_PREFIX=assistant,E_SMOKE=1 scripts/ntu_route_e_prefix.sbatch
# after first smoke finishes:
sbatch --export=ALL,E_PREFIX=user_only,E_SMOKE=1 scripts/ntu_route_e_prefix.sbatch

# Once both smoke jobs pass (sequentially):
sbatch --export=ALL,E_PREFIX=assistant,E_SMOKE=0 scripts/ntu_route_e_prefix.sbatch
# after full P1 job finishes:
sbatch --export=ALL,E_PREFIX=user_only,E_SMOKE=0 scripts/ntu_route_e_prefix.sbatch
```

Outputs do **not** overwrite P0:
- `outputs/route_e/deepseek/full/{train,val}.npz` (P0, unchanged)
- `outputs/route_e/prompt_ablation/assistant/{smoke,full}/` (P1)
- `outputs/route_e/prompt_ablation/user_only/{smoke,full}/` (P2)
- `results/route_e/prompt_ablation/{assistant,user_only}_{smoke,val}.json/.csv`

The extractor refuses to overwrite a cache. For failed partial jobs,
inspect or remove the specific output files explicitly before re-running.
Input file SHA256, model path, tokenizer template SHA256, actual token
lengths and timing data are saved in each cache's metadata. The first
GPU forward batch is recorded separately from post-first-batch throughput.
Neither timing metric includes model load time or establishes a real-world
single-query serving latency.

### Expected evaluation

Report cost-matched Random uplift at *each actual output-token budget*,
full Accuracy–Token curves for overall, GSM8K and MATH-500,
and timing/latency caveats. Keep validation AUC only as a diagnostic.
P0 vs P1 isolates the Thinking prefix; P1 vs P2 isolates the Assistant
prefix; P0 vs P2 is their combined effect. No canonical test-based tuning.
