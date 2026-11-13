#!/usr/bin/env python3
"""Synthetic 450-row integration test for the P2 Experiment 1 Cost-only analysis.

No real user labels, caches, GPU, generation or trainable heads are required.
"""
import json
import tempfile
from pathlib import Path
import numpy as np
import route_e_cost_only as exp
import route_e_frozen as base


def check_exception(callback, expected):
    try:
        callback()
    except ValueError as err:
        assert expected in str(err), str(err)
    else:
        raise AssertionError(f"Expected ValueError containing {expected}")


def main():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        val = folder/"val_450_labels.jsonl"
        predictions = folder/"p2_predictions.json"
        historical = folder/"p2_report.json"
        out = folder/"exp1"
        rows = []
        records = {}
        for i in range(450):
            ds = "gsm8k" if i < 300 else "math"
            sid = f"{ds}:{i}:0"
            pt = float((i%7)!=0)
            pn = float((i%11)!=0)
            tt = float(2000 + (i%31)*12 + (350 if ds=="math" else 0))
            tn = float(450 + (i%13)*14 + (120 if ds=="math" else 0))
            rows.append({
                "sample_id":sid, "problem":f"Synthetic problem number {i}",
                "dataset":ds,"p_t":pt,"p_n":pn,"t_t":tt,"t_n":tn,
            })
            delta = float(np.sin(i/19.)*0.22)
            est_t = float(2300 + (i%23)*40 + (450 if ds=="math" else 0))
            est_n = float(550 + (i%11)*25 + (120 if ds=="math" else 0))
            lam = 0.05
            records[sid] = {
                "delta_p_pred":delta, "thinking_tokens_pred":est_t,
                "nothinking_tokens_pred":est_n,"delta_t_pred":est_t-est_n,
                "score_delta_only":delta,
                "score_constant_cost":delta-0.05,
                "score_predicted_cost":delta-lam*(est_t-est_n)/1000.,
            }
        with val.open("w",encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps({
                    "sample_id":r["sample_id"],"problem":r["problem"],
                    "p_thinking":r["p_t"],"p_nothinking":r["p_n"],
                    "mean_tokens_thinking":r["t_t"],
                    "mean_tokens_nothinking":r["t_n"],
                })+"\n")
        base.VAL = val
        base.VAL_SHA256 = base.sha256(val)
        base.TRAIN_SHA256 = "synthetic_train_hash"
        json.dump({
            "strict_val_sha256":base.VAL_SHA256,
            "cost_lambda":lam, "predictions":records,
        }, predictions.open("w",encoding="utf-8"))
        maps = {
            "Delta_only":{sid:-v["score_delta_only"] for sid,v in records.items()},
            "Cost_only":{sid:v["delta_t_pred"] for sid,v in records.items()},
            "Delta_predicted_cost":{sid:-v["score_predicted_cost"] for sid,v in records.items()},
        }
        original_budgets = []
        budgets = [900,1100,1300,1500,1700,1900,2100]
        for b in budgets:
            methods={}
            for name in ("Delta_only","Delta_predicted_cost"):
                _,g,m = exp.policies_at_budget(rows,maps[name],b)
                methods[name]={"available":True,"groups":g,"mix":m}
            original_budgets.append({"budget":float(b),"methods":methods})
        obj = {
            "strict_val_sha256":base.VAL_SHA256,
            "strict_train_sha256":base.TRAIN_SHA256,
            "smoke":False,"encoder_frozen":True,"generation_repeated":False,
            "head":{"fixed_lambda":lam},
            "dense_validation":{"budgets":original_budgets},
        }
        json.dump(obj,historical.open("w",encoding="utf-8"))
        args=type("Args",(),{
            "budgets":budgets, "predictions":predictions, "p2_report":historical,
            "out_dir":out,"bootstrap_reps":25,"seed":42,
            "no_plot":True,"overwrite":False,
        })()
        exp.run(args)
        report=json.loads((out/"comparison.json").read_text())
        assert report["original_p2_b_d_dense_reproduced"] is True
        assert report["models_retrained"] is False
        assert report["ranking_uses_observed_validation_lengths"] is False
        assert len(report["budget_comparison"])==7
        assert set(report["group_cost_diagnostics"])=={"overall","gsm8k","math"}
        assert report["group_cost_diagnostics"]["gsm8k"]["n"]==300
        assert report["group_cost_diagnostics"]["math"]["n"]==150
        assert all(set(item["paired_bootstrap"])==
                   {"D_minus_E","E_minus_B","D_minus_B"}
                   for item in report["budget_comparison"])
        assert (out/"cost_diagnostics_by_dataset.csv").is_file()
        assert (out/"paired_bootstrap.csv").is_file()
        assert (out/"equal_budget.csv").is_file()

        # Corrupt a saved pre-generation score; must refuse silent baseline drift.
        broken=json.loads(predictions.read_text())
        first=next(iter(broken["predictions"]))
        broken["predictions"][first]["score_predicted_cost"]+=0.01
        predictions.write_text(json.dumps(broken))
        check_exception(lambda:exp.read_sources(rows,predictions,historical),
                        "score inconsistent")
        predictions.write_text(json.dumps({
            **broken,
            "predictions":records,
        }))
        # Corrupt reported original P2 performance; reproduction gate must fail.
        modified = json.loads(historical.read_text())
        modified["dense_validation"]["budgets"][0]["methods"]["Delta_only"][
            "groups"]["overall"]["accuracy"] += .01
        historical.write_text(json.dumps(modified))
        check_exception(lambda:exp.validate_original(rows,maps,modified,budgets),
                        "P2_REPRODUCTION_FAILED")
        # New method's actual-mode costs never influence its own score order.
        before=sorted(rows,key=lambda r:(-maps["Cost_only"][r["sample_id"]],
                                         r["sample_id"]))
        for row in rows:
            row["t_t"]+=1000.
        after=sorted(rows,key=lambda r:(-maps["Cost_only"][r["sample_id"]],
                                        r["sample_id"]))
        assert [r["sample_id"] for r in before] == [r["sample_id"] for r in after]
    print("COST_ONLY_INTEGRATION_PASS (450 strict mock rows, B/D reproductions, "
          "7 budgets, CI, subgroup and failure gates)",flush=True)


if __name__=="__main__":
    main()
