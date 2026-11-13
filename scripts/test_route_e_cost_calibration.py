#!/usr/bin/env python3
"""Synthetic integration test for P2 Experiment 2; does not open real labels."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import route_e_cost_calibration as exp
import route_e_token_aware as p2
import route_e_frozen as base


def synthetic_rows(n,kind,X):
    rows=[]
    for i in range(n):
        sid=f"{'gsm8k' if i%2==0 else 'math'}:{kind}:{i}"
        t_t=320.+140.*float(np.exp(0.4*X[i,0]))
        t_n=100.+40.*float(np.exp(0.2*X[i,1]))
        p_t=.85 if i%4 else .45
        p_n=.65 if i%3 else .35
        rows.append({"sample_id":sid,"dataset":sid.split(":")[0],
                     "problem":f"synthetic {kind} {i}",
                     "t_t":t_t,"t_n":t_n,
                     "p_t":p_t,"p_n":p_n,"delta":p_t-p_n})
    return rows


def main():
    rng=np.random.default_rng(20261103)
    Xtr=rng.normal(size=(84,8))
    Xva=rng.normal(size=(36,8))
    train=synthetic_rows(84,"train",Xtr)
    val=synthetic_rows(36,"val",Xva)
    original=p2.ridge_fit(Xtr,p2.target_matrix(train),.02)
    predicted=p2.predict_heads(original,Xva)
    lam=.05
    scored=p2.scores_for(predicted,lam)
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        headpath=folder/"old_model.npz"
        predpath=folder/"pred.json"
        reportpath=folder/"old_report.json"
        np.savez_compressed(headpath,**original)
        source={}
        for i,r in enumerate(val):
            source[r["sample_id"]]={
                "delta_p_pred":float(predicted["delta"][i]),
                "thinking_tokens_pred":float(predicted["t_t"][i]),
                "nothinking_tokens_pred":float(predicted["t_n"][i]),
                "score_predicted_cost":float(scored[i]),
            }
        predpath.write_text(json.dumps({
            "strict_val_sha256":base.VAL_SHA256,
            "cost_lambda":lam,"predictions":source}))
        reportpath.write_text(json.dumps({
            "strict_val_sha256":base.VAL_SHA256,
            "strict_train_sha256":base.TRAIN_SHA256,
            "smoke":False,"encoder_frozen":True,"generation_repeated":False,
            "head":{"ridge_penalty":.02,"fixed_lambda":lam},
        }))
        # This local synthetic integration overrides ONLY the cache loader
        # and 1536-wide npz dimension gate; real production gates remain intact.
        original_loader=p2.read_protocol_and_cache
        original_head_loader=exp.load_model
        p2.read_protocol_and_cache=lambda smoke: (
            train,val,Xtr,Xva,{}, {},0)
        exp.load_model=lambda path: original
        try:
            tmin=float(min(predicted["t_n"].mean(),predicted["t_t"].mean()))
            tmax=float(max(predicted["t_n"].mean(),predicted["t_t"].mean()))
            budgets=[round(tmin+(tmax-tmin)*w,3) for w in (.25,.5,.75)]
            args=SimpleNamespace(
                folds=4,seed=42,budgets=budgets,
                p2_model=headpath,p2_predictions=predpath,
                p2_report=reportpath,
                out_dir=folder/"results",model_out=folder/"outputs",
                no_plot=True,overwrite=False)
            exp.run(args)
            report=json.loads((args.out_dir/"comparison.json").read_text())
            assert report["validation_targets_used_for_fit"] is False
            assert report["routing_score_changed"] is False
            assert report["original_frozen_p2_head_reproduced"] is True
            assert len(report["budgets"])==3
            assert report["calibrator"]["folds"]==4
            assert (args.out_dir/"budget_comparison.csv").is_file()
            assert (args.out_dir/"length_metrics.csv").is_file()
            with np.load(args.model_out/"train_oof_calibrator.npz") as f:
                assert f["multipliers"].shape==(2,)
                assert f["fold_id"].shape==(84,)
                assert set(f["fold_id"])==set(range(4))
            # Reject unexpected overwrite of new experiment artifacts.
            try:
                exp.run(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Experiment 2 overwritten without explicit --overwrite")
            # Reject modifications to original P2 pre-generation scores.
            broken=json.loads(predpath.read_text())
            sid=next(iter(broken["predictions"]))
            broken["predictions"][sid]["score_predicted_cost"]+=.02
            predpath.write_text(json.dumps(broken))
            try:
                exp.read_original_scores(val,predpath,reportpath,original,Xva)
            except ValueError as err:
                assert "score changed" in str(err)
            else:
                raise AssertionError("Changed original router scores accepted")
        finally:
            p2.read_protocol_and_cache=original_loader
            exp.load_model=original_head_loader
    print("CALIB_INTEGRATION_PASS (head reproduction, four-fold held-out fit, "
          "unchanged ranking, three target budgets, strict fail gates)",flush=True)


if __name__=="__main__":
    main()
