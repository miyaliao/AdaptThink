#!/usr/bin/env python3
"""Shared text helpers for split construction and leakage auditing.

Both the training-manifest builder and the audit script must normalise and
shingle text in exactly the same way, otherwise "the manifest was checked" means
nothing. Keep every such rule here, in one place.
"""

import json
import unicodedata
from pathlib import Path

SHINGLE = 8


def norm_text(text):
    """NFKC + drop all whitespace + lowercase (the exact-duplicate key)."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(text.split()).lower()


def shingles(text, n=SHINGLE):
    """Character n-grams over normalised text; short texts fall back to the text.

    Character n-grams are used instead of word n-grams because math problems are
    dominated by LaTeX tokens, where a single number or symbol change should not
    hide a near-duplicate.
    """
    norm = norm_text(text)
    if len(norm) <= n:
        return {norm} if norm else set()
    return {norm[i:i + n] for i in range(len(norm) - n + 1)}


def similarity(a, b):
    """Max of Jaccard and containment; containment catches a subset/truncation."""
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    if not inter:
        return 0.0, 0.0
    union = len(a) + len(b) - inter
    return inter / union, inter / min(len(a), len(b))


def load_held_out(root, splits):
    """Every question of the given split files, tagged test/val and dataset."""
    held = []
    for path, label in splits:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        split = json.loads(path.read_text(encoding="utf-8"))
        for dataset, meta in split["source_files"].items():
            items = json.loads((Path(root) / meta["path"]).read_text(encoding="utf-8"))
            for index in meta["selected_indices"]:
                problem = items[index]["problem"]
                held.append({
                    "dataset": dataset,
                    "split": label,
                    "index": index,
                    "problem": problem,
                    "grams": shingles(problem),
                })
    return held
