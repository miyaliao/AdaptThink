#!/usr/bin/env python3
"""450-row synthetic integration for Experiment 3; all inputs are temporary.

Includes exact P1 50% NT reproduction, original P2 B/D dense replay gates,
frozen P1 and Ridge prediction checks, and the seven-budget comparison.
The synthetic outcome accuracy is flat on purpose; it cannot demonstrate
an empirical gain, only prove that the evaluation pipeline is functional.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import route_e_quality_fusion as new
import route_e_cost_only as old
import route_e_token_aware as p2
import route_e_frozen as base
from route_e_equal_budget import dense_curve


def main():
    rng=np.random.default_rng(20261103)
    budgets=[900.,1100.,1300.,1500.,1700.,1900.,2100.]
    ntrain=64
    Xtr=rng.normal(size=(ntrain,1536)).astype(np.float32)
    Xva=np.zeros((450,1536),dtype=np.float32)
    Xva[:,0]=np.linspace(-2.,2.,450)
    Xva[:,1]=rng.normal(size=450)
    train=[{"sample_id":f"train:{i}","dataset":"train",
            "p_t":.75,"p_n":.65,"delta":.1,"t_t":2100.,"t_n":500.}
           for i in range(ntrain)]
    val=[]
    for i in range(450):
        ds="gsm8k" if i<300 else "math"
        sid=f"{ds}:{i}:0"
        val.append({"sample_id":sid,"dataset":ds,
                    "p_t":344/450,"p_n":344/450,"delta":0.,
                    "t_t":2209.,"t_n":645.1 if i>=225 else 490.1})
    head={"mean":np.zeros(1536,dtype=np.float64),
          "scale":np.ones(1536,dtype=np.float64),
          "weight":np.zeros((1536,3),dtype=np.float64),
          "intercept":np.array([.1,np.log1p(2100.),np.log1p(550.)]),
          "smear":np.ones(2,dtype=np.float64),
          "l2":np.array(.02)}
    head["weight"][0,:]=[.04,.21,.1]
    head["weight"][1,:]=[-.03,.06,.03]
    pred=p2.predict_heads(head,Xva)
    lambda_fixed=.05
    probe={"mean":np.zeros(1536,dtype=np.float32),
           "scale":np.ones(1536,dtype=np.float32),
           "weight":np.zeros(1536,dtype=np.float32),
           "bias":np.asarray(0.,dtype=np.float32)}
    probe["weight"][0]=1.
    logits=new.frozen_logits(Xva,probe)
    probs=1./(1.+np.exp(-logits))
    with tempfile.TemporaryDirectory() as temp:
        temp=Path(temp)
        p1=temp/"p1_scores.json"
        probe_path=temp/"linear_probe.npz"
        head_path=temp/"ridge_heads.npz"
        saved_path=temp/"validation_predictions.json"
        old_report_path=temp/"validation.json"
        output_dir=temp/"experiment3"
        np.savez_compressed(probe_path,**probe)
        np.savez_compressed(head_path,**head)
        p1.write_text(json.dumps({"p_nt":{
            r["sample_id"]:float(p) for r,p in zip(val,probs)}}))
        records={}
        for i,r in enumerate(val):
            sid=r["sample_id"]
            delta_t=float(pred["t_t"][i]-pred["t_n"][i])
            delta_p=float(pred["delta"][i])
            records[sid]={
                "delta_p_pred":delta_p,
                "thinking_tokens_pred":float(pred["t_t"][i]),
                "nothinking_tokens_pred":float(pred["t_n"][i]),
                "delta_t_pred":delta_t,
                "score_delta_only":delta_p,
                "score_constant_cost":delta_p-.05*1600/1000,
                "score_predicted_cost":delta_p-lambda_fixed*delta_t/1000,
            }
        saved_path.write_text(json.dumps({
            "strict_val_sha256":"synthetic_val_hash",
            "cost_lambda":lambda_fixed,"predictions":records}))
        maps={
            "Delta_only":{sid:-v["score_delta_only"] for sid,v in records.items()},
            "Delta_predicted_cost":{
                sid:-v["score_predicted_cost"] for sid,v in records.items()},
        }
        historical=[]
        for budget in budgets:
            methods={}
            for method,mapping in maps.items():
                pn,g,bracket=old.policies_at_budget(val,mapping,budget)
                methods[method]={"available":True,"groups":g,"mix":bracket}
            historical.append({"budget":budget,"methods":methods})
        old_report_path.write_text(json.dumps({
            "strict_val_sha256":"synthetic_val_hash",
            "strict_train_sha256":"synthetic_train_hash",
            "smoke":False,"encoder_frozen":True,"generation_repeated":False,
            "head":{"fixed_lambda":lambda_fixed},
            "dense_validation":{"budgets":historical},
        }))
        orig_loader=p2.read_protocol_and_cache
        old_train_sha,old_val_sha=base.TRAIN_SHA256,base.VAL_SHA256
        try:
            base.TRAIN_SHA256="synthetic_train_hash"
            base.VAL_SHA256="synthetic_val_hash"
            p2.read_protocol_and_cache=lambda smoke: (
                train,val,Xtr,Xva,{"schema":"synthetic"},
                {"schema":"synthetic"},0)
            args=SimpleNamespace(
                probe=probe_path,p1_scores=p1,
                p2_head=head_path,p2_predictions=saved_path,
                p2_report=old_report_path,out_dir=output_dir,
                budgets=budgets,bootstrap_reps=25,seed=42,
                no_plot=True,overwrite=False)
            new.evaluate(args)
            report=json.loads((output_dir/"comparison.json").read_text())
            assert report["lineage"]["p2_control_reproduced"] is True
            assert report["no_model_retraining"] is True
            assert len(report["dense_equal_budget"])==7
            assert len(report["predicted_only_budget"])==7
            assert (output_dir/"equal_budget.csv").exists()
            assert (output_dir/"paired_bootstrap.csv").exists()
            assert (output_dir/"predicted_budget.csv").exists()
            frozen_p1=dense_curve(val,{r["sample_id"]:float(s)
                                      for r,s in zip(val,probs)})
            assert abs(frozen_p1[225]["accuracy"]-344/450)<1e-8
            assert abs(frozen_p1[225]["tokens"]-1427.05)<1e-8
            # Changing the frozen original model must be detected, not silently
            # accepted as a valid historical control.
            broken=np.array(head["weight"],copy=True)
            broken[0,0]+=.05
            altered={**head,"weight":broken}
            np.savez_compressed(head_path,**altered)
            try:
                new.strict_data_and_head(args)
            except ValueError as err:
                assert "predictions changed" in str(err)
            else:
                raise AssertionError("Corrupted old P2 head unexpectedly accepted")
        finally:
            p2.read_protocol_and_cache=orig_loader
            base.TRAIN_SHA256,base.VAL_SHA256=old_train_sha,old_val_sha
    print("QUALITY_FUSION_INTEGRATION_PASS (450 synthetic strict cases, P1 50% "
          "reproduction, seven budgets, frozen-control failure gate)",flush=True)


if __name__=="__main__":
    main()
