# Route E — Compare at equal generated-token budgets

Status: **CPU analysis implementation prepared; server-local score files
not accessible through GitHub and analysis has not been executed on TC1**.

Prior context:
- [Route A full-fine-tuned Classification](ROUTE_A_PLAN.md) has full
  validation points but its **per-question score file is only server-local**.
- [Route E frozen P0](ROUTE_E_FROZEN.md) and
  [Route E P1/P2 prefix ablations](ROUTE_E_PREFIX_ABLATION.md)
  produce per-question NoThinking scores on the strict 450 validation set.

## Metric definition

To compare P0/P1/P2 and Full FT Classification under budget B, rank
questions by router score (higher means prefer NoThinking). Evaluate
**each prefix k in 0..450**: route precisely those top k questions to
NoThinking, the others to Thinking. Record measured mean strict
answer success and generated tokens. This is a dense 451-point curve,
not merely six fractions.

For an exact expected mean budget B between the observed mean-token
costs of two policies, interpolate via a **randomized mixture of those
two policies**. The evaluation script sorts policies by measured
generated tokens and interpolates adjacent cost points. It does not
select an accuracy-maximizing envelope (which would be overly optimistic
on validation). Report the mixture probabilities along with the expected
accuracy. The interpolation is between **complete router policies**,
not between ground-truth answers or an oracle per-question selector.

**Important limitations**
- This is an offline replay on the validation set. Choice of a routing
  threshold or mixture from observed validation *token costs* is
  descriptive and must not be presented as a test-time deployment
  policy with prior knowledge of future output lengths.
- No router compute/prefill overhead is included.
- Accuracy is an empirical average of cached per-mode success rates.
  It is not necessarily a realized accuracy on a single sampled decode.
- The canonical held-out test set must not be used for tuning.
- Curves can be nonmonotonic in accuracy and occasionally mean token cost;
  interpolation between cost-adjacent policies is a mathematically valid
  randomized policy but not necessarily a single deterministic top-k rule.
- Tiny differences require paired uncertainty estimates before
  significance claims.

## Run (on the login node; **no GPU necessary**)

```bash
cd /tc1home/UG/n2602710a/AdaptThink
git status --short
git fetch origin
git switch feature/route-e-equal-budget-20261102
source .venv/bin/activate

python scripts/route_e_equal_budget.py --self-test
python scripts/route_e_equal_budget.py
```

The default command reads only:
- `results/val_450_labels.jsonl` (validated strict SHA256);
- `outputs/route_e/deepseek/full/validation_p_nt.json` (P0);
- `outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json` (P1);
- `outputs/route_e/prompt_ablation/user_only/full/validation_p_nt.json` (P2).

It creates:
- `results/route_e/budget_comparison/equal_budget.json` (metadata,
  exact budget-point mixtures, classification-source status);
- `results/route_e/budget_comparison/equal_budget.csv` (summary table).

Default common budgets: 900, 1100, 1300, 1500, 1700, 1900, 2100
mean generated tokens per question.

## Classification: prefer 450 raw scores, if available

The previously recorded classification aggregate shares have budgets:
0%=2209; 10%=1959; 25%=1542; 50%=1092; 75%=835;
100%=567.6. These are sufficient to form only an **approximate,
six-anchor interpolated Classification line**. They are not the
451-point dense exact Classification curve.

If the server has raw Classification per-question predictions, rerun
with their real path and explicit field/score direction, for example:

```bash
# First inspect available files; do not guess a prediction schema.
find outputs/external_modernbert results/external_modernbert -type f \
  \( -iname '*prediction*' -o -iname '*score*' \) | sort

# Hypothetical case (only if JSONL actually contains this field):
python scripts/route_e_equal_budget.py \
  --classification-scores outputs/external_modernbert/classification/validation_predictions.jsonl \
  --classification-score-field predicted_p_nothinking \
  --classification-score-direction higher_nt

# If classification score is a Thinking-benefit margin instead:
python scripts/route_e_equal_budget.py \
  --classification-scores path/to/classification_validation_scores.jsonl \
  --classification-score-field predicted_delta \
  --classification-score-direction lower_nt
```

Supported input forms:
1. JSON: `{"p_nt":{"sample_id":score,...}}`, or bare
   `{"sample_id":score,...}`;
2. JSONL: one record per sample with `sample_id` and the specified
   `--classification-score-field`.

The script asserts exact coverage of all 450 strict validation IDs,
rejects duplicates and nonfinite scores, and labels which methods
used dense versus coarse evidence. **Do not treat a 0/1 hard class
prediction as the intended ranked score** if continuous logits or
probabilities are available.

If Classification scores are missing, do **not** claim the four-model
table is uniformly exact. The report prints:

`FT_CLASSIFICATION_SOURCE=APPROXIMATE_SIX_ANCHORS`.

Once true scores are supplied, it prints:

`FT_CLASSIFICATION_SOURCE=DENSE_SCORES`.

## Interpreting preliminary six-point approximation

A coarse linear interpolation of the public validation points gives
at budget 1500: P1 76.57%, Full FT Classification 75.37%,
P0 75.32%, P2 74.48%. This is **not** the new dense 451-point
result, and it excludes routing overhead.

Use `equal_budget.json` produced by the script for the detailed
same-budget analysis before presenting a final research conclusion.
