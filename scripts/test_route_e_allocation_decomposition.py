#!/usr/bin/env python3
"""Cache-free synthetic 450-row integration for allocation decomposition."""
import json
import tempfile
from pathlib import Path

import numpy as np

import route_e_frozen as base
import route_e_allocation_decomposition as dec
from route_e_token_aware import group_metrics


def main():
    with tempfile.TemporaryDirectory() as rootdir:
        root=Path(rootdir)
        labels=root/"val.jsonl"
        rows=[]
        with labels.open("w",encoding="utf-8") as fp:
            for i in range(450):
                ds="gsm8k" if i<300 else "math"
                sid=f"{ds}:{i}:0"
                tt=float(850+i%17*18) if ds=="gsm8k" else float(4300+i%25*55)
                tn=float(250+i%9*9) if ds=="gsm8k" else float(990+i%11*45)
                pt=.8 if i%5!=0 else .2
                pn=.7 if i%4!=0 else .1
                raw={
                    "sample_id":sid,"problem":f"synthetic question {i}",
                    "p_thinking":pt,"p_nothinking":pn,
                    "mean_tokens_thinking":tt,
                    "mean_tokens_nothinking":tn,
                }
                fp.write(json.dumps(raw)+"\n")
                rows.append({"sample_id":sid,"dataset":ds,
                             "p_t":pt,"p_n":pn,"t_t":tt,"t_n":tn})
        base.VAL=labels
        base.VAL_SHA256=base.sha256(labels)
        base.TRAIN_SHA256="synthetic_train_sha"
        rng=np.random.default_rng(13)
        names=("Delta_only","Cost_only","Delta_predicted_cost")
        report={
            "data_lineage":{"strict_val_sha256":base.VAL_SHA256,
                            "strict_train_sha256":base.TRAIN_SHA256},
            "original_p2_b_d_dense_reproduced":True,
            "ranking_uses_observed_validation_lengths":False,
            "dense_cost_matching_uses_observed_validation_lengths":True,
            "budget_comparison":[],
        }
        for pct in [0.15,0.25,0.45]:
            methods={}
            for name in names:
                # Dataset-specific routing probabilities with modest
                # within-dataset noise; all budgets lie inside endpoints.
                if name=="Cost_only":
                    pn=np.where(np.arange(450)<300,pct/2,
                                min(1.,pct*1.65))
                elif name=="Delta_only":
                    pn=np.where(np.arange(450)<300,pct*1.1,pct*0.9)
                else:
                    pn=np.where(np.arange(450)<300,pct*0.8,pct*1.3)
                pn=np.clip(pn+rng.uniform(-.02,.02,450),0.,1.)
                metrics=group_metrics(rows,pn)
                methods[name]={"groups":metrics}
            # All methods may have different global mean costs, so this
            # synthetic test uses the SAME per-method budget from a given
            # base policy (measured-cost equality is essential).
            # Create a common target by generating policy mixtures with
            # group-constant allocation and solving for one group's rate.
            target=methods["Delta_only"]["groups"]["overall"]["actual_tokens"]
            for name in ("Cost_only","Delta_predicted_cost"):
                pn=np.where(np.arange(450)<300,
                            (pct*.2 if name=="Cost_only" else pct*.7),0.)
                # choose math NT rate from total expected cost equation
                g=rows
                tt=np.array([r["t_t"] for r in g])
                tn=np.array([r["t_n"] for r in g])
                group_gsm=np.arange(450)<300
                cost_gsm=np.sum(tt[group_gsm]*(1-pn[group_gsm])
                                + tn[group_gsm]*pn[group_gsm])
                cost_math_t=np.sum(tt[~group_gsm])
                denominator=np.sum(tt[~group_gsm]-tn[~group_gsm])
                required=(cost_gsm+cost_math_t-450*target)/denominator
                if required<0 or required>1:
                    # Try a midrange GSM allocation if the attempted
                    # type-first cost match is infeasible.
                    pn[group_gsm]=pct
                    cost_gsm=np.sum(tt[group_gsm]*(1-pn[group_gsm])
                                    +tn[group_gsm]*pn[group_gsm])
                    required=(cost_gsm+cost_math_t-450*target)/denominator
                assert 0<=required<=1
                pn[~group_gsm]=required
                methods[name]={"groups":group_metrics(rows,pn)}
            for method in names:
                assert abs(methods[method]["groups"]["overall"]["actual_tokens"]-
                           target)<1e-5
            report["budget_comparison"].append({"budget":target,"methods":methods})
        source=root/"experiment1.json"
        source.write_text(json.dumps(report))
        args=type("Args",(),{"out_dir":root/"result","source":source,
                             "overwrite":False})()
        dec.run(args)
        results=json.loads((args.out_dir/"decomposition.json").read_text())
        assert len(results["table"])==3
        for b in results["table"]:
            assert "E_minus_B_allocation_component_pp" in b
            assert abs(b["E_minus_B_allocation_component_pp"]+
                       b["E_minus_B_within_group_component_pp"]-
                       b["E_minus_B_accuracy_pp"])<1e-6
        assert (args.out_dir/"decomposition.csv").is_file()
        assert (args.out_dir/"subgroup_random.csv").is_file()
    print("COST_ALLOC_INTEGRATION_PASS (450 rows, three budgets, exact identity, CSV)",flush=True)


if __name__=="__main__":
    main()
