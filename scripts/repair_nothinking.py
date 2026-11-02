#!/usr/bin/env python3
"""Repair NoThinking responses that are correct but badly formatted, then rescore.

Why
---
23.4% of the 12,000 generated NoThinking answers are rejected by
`src/adapt_think_rm.py` for formatting, not for being wrong:

* 1,500 ``Multiple <think>``  - the model opened a second thinking block anyway
*   757 ``Num of </think> == 2`` - it closed thinking twice
*   554 ``Answer Extraction Failed`` - it answered without a ``\\boxed{}``

The scorer requires **exactly one** ``</think>`` and a boxed answer, so a correct
answer inside a badly formatted response is counted as wrong and depresses
``p_nothinking`` for every affected problem. This script keeps the response text
but repairs the *scorable* part, and writes a directory that
``scripts/build_router_labels.py`` can consume unchanged.

Repair rules (deterministic, applied only to NoThinking responses)
-----------------------------------------------------------------
1. Keep everything after the **last** ``</think>``; if there is no ``</think>``,
   keep the text as is.
2. Prefix the result with a single ``</think>`` so the scorer's format check passes.
3. If the kept segment has no ``\\boxed{}``, fall back to the last ``\\boxed{}``
   anywhere in the original response (it is the model's own final answer).
4. If there is still no boxed answer, leave it alone: no answer is invented.

Both the strict and the repaired outcome are recorded per generation, plus a
``format_issue`` flag, so the report can quantify how much of the "NoThinking is
worse" gap was formatting rather than reasoning.

Output: --out-dir/nothinking.jsonl (same schema as the original rows) and, if the
source directory has one, a copy of config.json so the label builder's protocol
checks still pass.

Usage
-----
    python scripts/repair_nothinking.py
    python scripts/repair_nothinking.py --src outputs/nothinking_train_seed42 \
        --out-dir outputs/nothinking_train_repaired
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/nothinking_train_seed42"
OUT = ROOT / "outputs/nothinking_train_repaired"


def last_boxed(text):
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    depth, start = 0, None
    for i in range(idx, len(text)):
        if text[i] == "{":
            if start is None:
                start = i
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[idx:i + 1]
    return None


def is_compliant(response):
    """Did the model actually skip thinking? Judged on the ORIGINAL response.

    Compliant means: no second thinking block was opened (no "<think>") and
    thinking was closed exactly once. Anything else is a protocol violation and
    its answer is not evidence about NoThinking ability, however the format is
    repaired for scoring.
    """
    return "<think>" not in response and response.count("</think>") == 1


def repair(response):
    """Return (repaired_response, format_issue, note)."""
    issues = []
    if response.count("</think>") != 1:
        issues.append("multiple </think>")
    if "<think>" in response:
        issues.append("<think> present")
    body = response.split("</think>")[-1] if "</think>" in response else response
    if "\\boxed" not in body:
        boxed = last_boxed(response)
        if boxed is not None:
            issues.append("boxed outside final segment")
            body = (body + "\n" if body.strip() else "") + "\\boxed{" + boxed[len("\\boxed{"):-1] + "}"
        else:
            issues.append("no boxed answer")
    if not issues:
        return response, False, ""
    return "</think>" + body, True, "; ".join(issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=SRC)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/splits/sft_train_3000_seed42.json",
                        help="Generation rows carry no 'answer'; it is looked up here")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    answers = {s["sample_id"]: s["answer"] for s in manifest["samples"]}
    problems = {s["sample_id"]: s["problem"] for s in manifest["samples"]}

    files = sorted(args.src.glob("shard*of*.jsonl")) or sorted(args.src.glob("nothinking.jsonl"))
    if not files:
        raise FileNotFoundError(f"No generation jsonl under {args.src}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ROOT))
    from src.adapt_think_rm import adapt_think_rm

    stats = Counter()
    out_path = args.out_dir / "nothinking.jsonl"
    with out_path.open("w", encoding="utf-8") as out:
        for path in files:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = row["sample_id"]
                    if sample_id not in answers:
                        raise RuntimeError(f"{sample_id} is not in the manifest {args.manifest}")
                    row["answer"] = answers[sample_id]
                    row["problem"] = problems[sample_id]
                    # compliance is judged on the ORIGINAL text, before any repair
                    original = row["response"]
                    compliant = is_compliant(original)
                    strict = adapt_think_rm(data_source="", solution_str=original,
                                            ground_truth=row["answer"])
                    repaired, issue, note = repair(original)
                    if issue:
                        fixed = adapt_think_rm(data_source="", solution_str=repaired,
                                               ground_truth=row["answer"])
                    else:
                        fixed = strict
                    stats["generations"] += 1
                    stats["strict_correct"] += int(strict["acc"])
                    stats["repaired_correct"] += int(fixed["acc"])
                    if issue:
                        stats["format_issue"] += 1
                        stats[f"issue:{note}"] += 1
                        if fixed["acc"] > strict["acc"]:
                            stats["rescued"] += 1
                    row["response"] = repaired
                    row["acc_strict"] = float(strict["acc"])
                    row["acc_repaired"] = float(fixed["acc"])
                    row["format_issue"] = issue
                    row["compliant"] = compliant
                    stats["compliant"] += int(compliant)
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")

    config = args.src / "config.json"
    if config.is_file():
        shutil.copy(config, args.out_dir / "config.json")

    summary = {
        "generations": stats["generations"],
        "strict_accuracy": stats["strict_correct"] / max(stats["generations"], 1),
        "repaired_accuracy": stats["repaired_correct"] / max(stats["generations"], 1),
        "format_issue_generations": stats["format_issue"],
        "format_issue_rate": stats["format_issue"] / max(stats["generations"], 1),
        "rescued_by_repair": stats["rescued"],
        "compliant_generations": stats["compliant"],
        "compliance_rate": stats["compliant"] / max(stats["generations"], 1),
        "issue_breakdown": {k.split(":", 1)[1]: v for k, v in stats.items()
                            if k.startswith("issue:")},
    }
    (args.out_dir / "repair_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"REPAIRED {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
