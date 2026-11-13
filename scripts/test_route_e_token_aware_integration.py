#!/usr/bin/env python3
"""Cache-free synthetic integration smoke for Route E/P2.

Exercises the real 1536-D cache loader, train/validation order checks,
Ridge fit, P2-only output isolation, and head artifact serialization.
It monkeypatches only temporary test paths/protocol; no real data are read.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_e_frozen as base
import route_e_prompt_ablation as prefix
import route_e_token_aware as p2


def make_rows(n, split):
    rows = []
    for i in range(n):
        sid = f"{'gsm8k' if i % 2 == 0 else 'math'}:{split}:{i}"
        better_n = i % 2 == 0
        pt, pn = (.4, .8) if better_n else (.8, .4)
        rows.append({
            "sample_id": sid, "problem": f"Synthetic arithmetic problem {i}",
            "dataset": "gsm8k" if i % 2 == 0 else "math",
            "delta": pt-pn, "p_t": pt, "p_n": pn,
            "t_t": float(850+i*11), "t_n": float(300+i*3),
        })
    return rows


def save_cache(path, ids, X, data_hash, smoke=True):
    meta = {
        "schema": "route_e_prefix_features_v1",
        "encoder": "deepseek", "model": "synthetic_no_model_forward",
        "prompt_mode": "assistant", "chat_template_sha256": "synthetic",
        "pooling": "last_valid_prompt_token", "layer": "last_after_final_norm",
        "dtype": "torch.float16", "max_length": 1024, "feature_dim": 1536,
        "smoke": smoke, "labels_sha256": data_hash,
        "post_first_batch_forward_ms_per_question": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float16), sample_id=np.asarray(ids),
                        metadata_json=np.asarray(json.dumps(meta)))


def main():
    rng = np.random.default_rng(20261103)
    train, val = make_rows(32, "tr"), make_rows(24, "va")
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        p2.ROOT = temp
        prefix.ROOT = temp
        base.load_protocol = lambda: (train, val, 0)

        cache = prefix.cache_dir("assistant", True)
        Xtr = rng.normal(size=(len(train),1536))
        Xva = rng.normal(size=(len(val),1536))
        save_cache(cache/"train.npz", [r["sample_id"] for r in train],
                   Xtr, base.TRAIN_SHA256)
        save_cache(cache/"val.npz", [r["sample_id"] for r in val],
                   Xva, base.VAL_SHA256)
        args = SimpleNamespace(smoke=True, overwrite=False, l2=0.02,
                               cost_lambda=0.05, seed=42, budgets=[900],
                               bootstrap_reps=0, no_plot=True)
        p2.check(args)
        p2.run(args)
        out, reports = p2.paths(True)
        assert (out/"ridge_heads.npz").exists()
        artifact = json.loads((out/"validation_predictions.json").read_text())
        assert len(artifact["predictions"]) == 24
        assert not any("actual" in key or key in ("p_t","p_n","t_t","t_n")
                       for value in artifact["predictions"].values() for key in value)
        report = json.loads((reports/"validation.json").read_text())
        assert report["smoke"] is True
        assert report["generation_repeated"] is False
        assert len(report["train_feature_metadata"]) > 0
        assert report["diagnostics"]["validation"]["thinking_tokens"]["mae"] >= 0
        # Also execute the entire 450-question dense offline evaluation path.
        # Use low-size synthetic training (32) to keep the CI test cheap; only
        # this fake protocol bypasses strict 3000/450 production assertions.
        val_full = make_rows(450, "full")
        for i, row in enumerate(val_full):
            row["t_t"] = float(2100+20*(i%11))
            row["t_n"] = float(520+9*(i%11))
        base.load_protocol = lambda: (train, val_full, 0)
        full_cache = prefix.cache_dir("assistant", False)
        save_cache(full_cache/"train.npz", [r["sample_id"] for r in train],
                   Xtr, base.TRAIN_SHA256, smoke=False)
        save_cache(full_cache/"val.npz", [r["sample_id"] for r in val_full],
                   rng.normal(size=(450,1536)), base.VAL_SHA256, smoke=False)
        full_args = SimpleNamespace(smoke=False, overwrite=False, l2=0.02,
                                    cost_lambda=0.05, seed=42,
                                    budgets=[900,1100,1300,1500,1700,1900,2100],
                                    bootstrap_reps=25, no_plot=True)
        p2.check(full_args)
        p2.run(full_args)
        _, full_reports = p2.paths(False)
        full_result = json.loads((full_reports/"validation.json").read_text())
        assert len(full_result["dense_validation"]["budgets"]) == 7
        assert len(full_result["lambda_sweep"]["rows"]) == 8
        first = full_result["dense_validation"]["budgets"][0]["methods"]
        assert abs(first["Delta_only"]["groups"]["overall"]["accuracy"] -
                   first["Delta_constant_cost"]["groups"]["overall"]["accuracy"]) < 1e-12
        assert (full_reports/"budget_comparison.csv").is_file()
        assert (full_reports/"lambda_grid.csv").is_file()
        # Restore the fake smoke protocol, then corrupt its cache ID order.
        base.load_protocol = lambda: (train, val, 0)
        # Corrupt ID order. Loader must fail closed, not silently align by index.
        save_cache(cache/"val.npz", [r["sample_id"] for r in reversed(val)],
                   Xva, base.VAL_SHA256)
        try:
            p2.read_protocol_and_cache(True)
        except ValueError as exc:
            assert "order" in str(exc).lower()
        else:
            raise AssertionError("Corrupted cache ID order was accepted")
    print("P2_SYNTHETIC_INTEGRATION_PASS (32/24 + mocked 450 dense budgets, "
          "artifacts, bootstrap, strict ID failure)", flush=True)


if __name__ == "__main__":
    main()
