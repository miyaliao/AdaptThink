#!/usr/bin/env python3
"""Complete AIME 2024 (30x16/mode), reusing original 8-question subset results.

Generates ONLY the previously untested AIME questions. The 25% GSM8K/MATH500
results and the original AIME JSONL files are never changed.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_baseline_subset as base

OLD = base.OUTPUT
NEW = base.ROOT / "outputs/baseline_aime_remaining_seed42"


def get_tasks():
    old_all, hashes = base.tasks_and_hashes()
    old_aime = [t for t in old_all if t["dataset"] == "aime"]
    old_ids = {t["id"] for t in old_aime}
    raw = (base.ROOT / "data/test/aime.json").read_bytes()
    data = json.loads(raw)
    if len(data) != 30 or len(old_aime) != 128:
        raise RuntimeError("Unexpected AIME dataset/subset size")
    full = [
        {
            "id": f"aime:{i}:{rep}",
            "dataset": "aime", "index": i, "repeat": rep,
            "problem": row["problem"], "answer": row["answer"],
            "seed": 42 + base.OFFSETS["aime"] + 32 * i + rep,
        }
        for i, row in enumerate(data) for rep in range(16)
    ]
    missing = [t for t in full if t["id"] not in old_ids]
    if len(missing) != 352:
        raise RuntimeError(f"Expected 352 additional AIME generations/mode, got {len(missing)}")
    return old_all, full, missing, hashes, hashlib.sha256(raw).hexdigest()


def verify_old(tok, old_all, hashes):
    path = OLD / "manifest.json"
    if not path.exists():
        raise RuntimeError(f"Missing original manifest {path}")
    previous = json.loads(path.read_text())
    expected = base.configuration(old_all, hashes, tok)
    if previous != expected:
        raise RuntimeError("Original subset manifest does not match the existing code/datasets/template")
    allowed = {x["id"] for x in old_all}
    original = {}
    for mode in ("nothinking", "thinking"):
        rows = base.completed(OLD / (mode + ".jsonl"), allowed)
        original[mode] = {k: v for k, v in rows.items() if k.startswith("aime:")}
        if len(original[mode]) != 128:
            raise RuntimeError(f"Original AIME {mode} incomplete: {len(original[mode])}/128")
        for t in old_all:
            if t["dataset"] != "aime":
                continue
            r = original[mode][t["id"]]
            if (r.get("id"), r.get("index"), r.get("repeat"), r.get("seed"), r.get("answer"), r.get("mode")) != (
                t["id"], t["index"], t["repeat"], t["seed"], t["answer"], mode
            ):
                raise RuntimeError(f"Original AIME data mismatch: {t['id']} {mode}")
    summary_path = OLD / "summary.json"
    if not summary_path.exists():
        raise RuntimeError("Original summary.json missing")
    old_summary = json.loads(summary_path.read_text())
    for mode in ("nothinking", "thinking"):
        if old_summary["results"][mode]["gsm8k"]["generations"] != 330:
            raise RuntimeError("GSM8K result count unexpected")
        if old_summary["results"][mode]["math"]["generations"] != 125:
            raise RuntimeError("MATH500 result count unexpected")
    return original, previous, old_summary


def init_manifest(previous, missing, digest):
    manifest = {
        "run": "extend_aime2024_30x16_from_25pct_seed42",
        "original_subset_manifest": previous,
        "aime_sha256": digest,
        "new_task_ids": [t["id"] for t in missing],
        "existing_aime_per_mode": 128,
        "new_aime_per_mode": 352,
        "no_changes_to_original_subset": True,
    }
    NEW.mkdir(parents=True, exist_ok=True)
    path = NEW / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f"Run configuration changed, refusing to append to {path}")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def grade_all(full, missing, original, old_summary):
    scorer = base.get_scorer()
    all_summary = {}
    for mode in ("nothinking", "thinking"):
        added = base.completed(NEW / (mode + ".jsonl"), {t["id"] for t in missing})
        if len(added) != len(missing):
            raise RuntimeError(f"New AIME {mode} is incomplete: {len(added)}/{len(missing)}")
        if set(original[mode]) & set(added):
            raise RuntimeError("Overlapping old/new AIME sample IDs")
        rows = {**original[mode], **added}
        if len(rows) != 480:
            raise RuntimeError(f"Expected 480 results for {mode}, got {len(rows)}")
        grades = []
        for task in full:
            row = rows[task["id"]]
            if (row["seed"], row["answer"], row["mode"]) != (
                task["seed"], task["answer"], mode
            ):
                raise RuntimeError(f"Response metadata mismatch: {task['id']}")
            result = scorer(
                data_source="aime", solution_str=row["response"],
                ground_truth=task["answer"]
            )
            grades.append({
                "id": task["id"], "index": task["index"], "repeat": task["repeat"],
                "correct": float(result["acc"]), "prediction": str(result["pred"]),
                "tokens": row["num_output_tokens"],
                "truncated": row["finish_reason"] == "length",
                "is_nothinking": row["response"].strip().startswith("</think>"),
                "source": "original_25pct" if task["id"] in original[mode] else "new_22_questions",
            })
        (NEW / (mode + "_aime30_grades.json")).write_text(
            json.dumps(grades, ensure_ascii=False, indent=2) + "\n"
        )
        n = len(grades)
        result = {
            "generations": n,
            "unique_questions": 30,
            "accuracy": sum(x["correct"] for x in grades) / n,
            "mean_response_tokens": sum(x["tokens"] for x in grades) / n,
            "nothinking_ratio": sum(x["is_nothinking"] for x in grades) / n,
            "max_length_fraction": sum(x["truncated"] for x in grades) / n,
        }
        print("FULL_AIME_SCORE", mode, json.dumps(result), flush=True)
        all_summary[mode] = {
            "gsm8k": old_summary["results"][mode]["gsm8k"],
            "math": old_summary["results"][mode]["math"],
            "aime": result,
        }
    final = {
        "description": "GSM8K 330 and MATH500 125 per mode from original subset; AIME2024 30x16/mode",
        "original_run": str(OLD),
        "new_aime_responses": str(NEW),
        "results": all_summary,
    }
    target = NEW / "summary_combined.json"
    target.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
    print(f"FINAL_SUMMARY {target}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=("preflight", "generate", "score"))
    parser.add_argument("--mode", choices=("nothinking", "thinking"))
    args = parser.parse_args()
    old_all, full, missing, hashes, digest = get_tasks()
    tok = base.load_tokenizer()
    original, previous, old_summary = verify_old(tok, old_all, hashes)

    if args.action == "preflight":
        base.preflight(tok, [t for t in missing if t["repeat"] == 0])
        print("AIME_EXPANSION_PREFLIGHT_OK new_questions=22 new_generations=704 "
              "reuse_existing=256", flush=True)
        return

    init_manifest(previous, missing, digest)
    if args.action == "generate":
        if args.mode is None:
            parser.error("--mode required for generate")
        previous_output = base.OUTPUT
        try:
            base.OUTPUT = NEW
            base.generate(args.mode, tok, missing)
        finally:
            base.OUTPUT = previous_output
    else:
        grade_all(full, missing, original, old_summary)


if __name__ == "__main__":
    main()
