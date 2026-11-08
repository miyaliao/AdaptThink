#!/usr/bin/env python3
"""AdaptThink Route A: Utility-Original = direct regression of raw Delta U_N.

Delta U_N = (p_N-p_T) + lambda*(t_T-t_N)/1000.
- Full-FT ModernBERT: 1-output regression head, MSE, same training settings
  as original ModernBERT Utility Hard/Soft run (5 epochs, lr=2e-5, bs=16,
  maxlen=512, fp16, seed=42); independent checkpoint per lambda (one GPU).
- Frozen DeepSeek P1: train-only standardized final-token 1536-d hidden state,
  linear Ridge MSE + L2=1e-3, LBFGS (CPU); independent head per lambda.
- Evaluate: 451 top-k threshold policies, original cost-sorted nAUC-AC;
  seven equal-budget points are supplementary, never used for the AUC.

Examples:
 python scripts/utility_original_ablation.py self-test
 python scripts/utility_original_ablation.py plan --encoder modernbert
 python scripts/utility_original_ablation.py train --encoder p1
 python scripts/utility_original_ablation.py train --encoder modernbert  # GPU!
 python scripts/utility_original_ablation.py evaluate --encoder p1
 python scripts/utility_original_ablation.py evaluate --encoder modernbert

Original strict 3000/450 only, no sealed test, no LLM generation.
λ tuning on the reused 450 validation examples is exploratory.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
LAMBDAS = (0., .005, .01, .02, .05, .10)
BUDGETS = (900., 1100., 1300., 1500., 1700., 1900., 2100.)
P1_DIR = ROOT / "outputs/route_e/prompt_ablation/assistant/full"
MODERN_DIR = ROOT / "outputs/external_modernbert/utility_original"
P1_HEADS = ROOT / "outputs/route_e/p1_utility_original"
REPORT_ROOT = ROOT / "results/route_a/utility_original"


def target_one(row, lam):
    return (float(row["p_n"]) - float(row["p_t"])
            + lam*(float(row["t_t"])-float(row["t_n"]))/1000.)


def make_targets(rows, lam):
    y = np.asarray([target_one(r, lam) for r in rows], dtype=np.float32)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Invalid continuous Utility target")
    return y


def label(lam):
    return f"utility_original_lam{lam:g}"


def output_dir(encoder, lam, smoke=False):
    base = MODERN_DIR if encoder == "modernbert" else P1_HEADS
    return base / (label(lam) + ("_smoke" if smoke else ""))


def protocol():
    import route_e_frozen as base
    train, val, overlaps = base.load_protocol()
    if len(train) != 3000 or len(val) != 450:
        raise RuntimeError("Strict protocol must be 3000 train + 450 val")
    return base, train, val, overlaps


def read_scores(folder, rows, expected_train_sha, expected_val_sha, lam, encoder, smoke=False):
    """Completed artifact gate: refuse mixing splits, lambdas, or smoke runs."""
    p = folder / "validation_utility_scores.json"
    m = folder / "run_metadata.json"
    if not p.exists() or not m.exists():
        return None
    meta = json.loads(m.read_text(encoding="utf-8"))
    for key, expected in (
        ("train_sha256", expected_train_sha),
        ("val_sha256", expected_val_sha),
        ("encoder", encoder), ("smoke", smoke), ("objective", "utility_original"),
    ):
        if meta.get(key) != expected:
            raise RuntimeError(f"{folder}: metadata {key} mismatch")
    if abs(float(meta["lambda_cost"])-lam) > 1e-12:
        raise RuntimeError(f"{folder}: lambda mismatch")
    if not smoke and (meta.get("n_train") != 3000 or meta.get("n_val") != 450):
        raise RuntimeError(f"{folder}: incorrect sample counts")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if obj.get("score_definition") != "predicted_delta_utility_n":
        raise RuntimeError(f"{folder}: score direction unclear")
    scores = {str(k):float(v) for k,v in obj["scores"].items()}
    if len(scores) != len(rows) or set(scores) != {r["sample_id"] for r in rows}:
        raise RuntimeError(f"{folder}: validation IDs mismatch")
    if not all(math.isfinite(v) for v in scores.values()):
        raise RuntimeError(f"{folder}: nonfinite prediction")
    return scores


def write_completed(folder, scores, base, lam, encoder, ntrain, nval, smoke, extras):
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"validation_utility_scores.json").write_text(
        json.dumps({"score_definition":"predicted_delta_utility_n",
                    "scores":scores}, indent=2, ensure_ascii=False)+"\n",
        encoding="utf-8")
    meta = {
        "objective":"utility_original", "loss":"MSE",
        "target":"(p_N-p_T)+lambda*(t_T-t_N)/1000",
        "score_direction":"higher = prefer NoThinking",
        "lambda_cost":lam, "encoder":encoder,
        "train_sha256":base.TRAIN_SHA256, "val_sha256":base.VAL_SHA256,
        "n_train":ntrain, "n_val":nval, "smoke":smoke,
        "token_cost_units":"output tokens per 1000",
        "router_overhead_included":False, "heldout_test_used":False,
        **extras,
    }
    (folder/"run_metadata.json").write_text(
        json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def fit_linear_mse(X, y, l2=1e-3, max_iter=200):
    """Ridge regression on frozen P1; train-only scaling and MSE+L2."""
    import torch
    if X.ndim != 2 or X.shape[0] != len(y) or len(y) < 2:
        raise ValueError("Invalid P1 regression shapes")
    mean = X.mean(axis=0,dtype=np.float64).astype(np.float32)
    scale = np.maximum(X.std(axis=0,dtype=np.float64).astype(np.float32),1e-4)
    Z = torch.from_numpy(((X-mean)/scale).astype(np.float32))
    target = torch.from_numpy(np.asarray(y,dtype=np.float32))
    w = torch.nn.Parameter(torch.zeros(X.shape[1],dtype=torch.float32))
    bias = torch.nn.Parameter(torch.tensor(float(target.mean())))
    opt = torch.optim.LBFGS([w,bias],lr=1.,max_iter=max_iter,
                            tolerance_grad=1e-8,tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        pred = Z@w+bias
        loss = torch.nn.functional.mse_loss(pred,target) + l2*w.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        final = float(torch.nn.functional.mse_loss(Z@w+bias,target)
                      +l2*w.square().sum())
    return {"weight":w.detach().numpy().copy(),"bias":float(bias.detach()),
            "mean":mean,"scale":scale,"train_loss":final}


def predict_linear(X, fit):
    return ((X-fit["mean"])/fit["scale"]) @ fit["weight"] + fit["bias"]


def train_p1(args, base, train, val):
    import route_e_prompt_ablation as p1
    Xtr, mt = p1.read_cache(P1_DIR/"train.npz",train,base.TRAIN_SHA256,"assistant",False)
    Xva, mv = p1.read_cache(P1_DIR/"val.npz",val,base.VAL_SHA256,"assistant",False)
    for k in ("model","prompt_mode","chat_template_sha256","pooling",
              "layer","dtype","max_length"):
        if mt[k] != mv[k]:
            raise RuntimeError(f"P1 feature metadata mismatch: {k}")
    if mt["layer"] != "last_after_final_norm" or mt["pooling"] != "last_valid_prompt_token":
        raise RuntimeError("Wrong P1 representation")
    if Xtr.shape != (3000,1536) or Xva.shape != (450,1536):
        raise RuntimeError("Wrong P1 feature shape")
    for lam in args.lambdas:
        folder = output_dir("p1",lam)
        old=read_scores(folder,val,base.TRAIN_SHA256,base.VAL_SHA256,lam,"p1")
        if old is not None:
            print(f"ORIGINAL_REUSED encoder=p1 lambda={lam:g}",flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Incomplete output, inspect manually: {folder}")
        fit=fit_linear_mse(Xtr,make_targets(train,lam),l2=args.l2,max_iter=args.max_iter)
        pred=predict_linear(Xva,fit)
        if not np.isfinite(pred).all():
            raise RuntimeError("Nonfinite P1 regression predictions")
        scores={r["sample_id"]:float(x) for r,x in zip(val,pred)}
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/"ridge_probe.npz",
                            weight=fit["weight"],bias=np.asarray(fit["bias"]),
                            mean=fit["mean"],scale=fit["scale"])
        write_completed(folder,scores,base,lam,"p1",len(train),len(val),False,{
            "loss":"MSE + L2*sum(weight^2)","l2":args.l2,
            "max_iter":args.max_iter,"solver":"full-batch torch LBFGS CPU",
            "encoder_frozen":True, "train_objective":fit["train_loss"],
            "representation":"P1 assistant-only; last normalized final prompt token",
            "feature_dim":1536,
        })
        print(f"ORIGINAL_TRAINED encoder=p1 lambda={lam:g} "
              f"loss={fit['train_loss']:.6f}",flush=True)


class ModernDataset:
    def __init__(self, rows, tokenizer, max_length, lam):
        self.rows, self.tok, self.max_length, self.lam = rows,tokenizer,max_length,lam
    def __len__(self):
        return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        enc=self.tok(row["problem"],truncation=True,max_length=self.max_length)
        enc["labels"]=target_one(row,self.lam)
        return enc


def train_modern(args, base, train, val):
    import torch
    from transformers import (AutoModelForSequenceClassification,AutoTokenizer,
        DataCollatorWithPadding,Trainer,TrainingArguments,set_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("ModernBERT Full Fine-Tuning requires one GPU Slurm allocation")
    for lam in args.lambdas:
        folder=output_dir("modernbert",lam,args.smoke)
        used_train,used_val=(train[:32],val[:24]) if args.smoke else (train,val)
        old=read_scores(folder,used_val,base.TRAIN_SHA256,base.VAL_SHA256,
                        lam,"modernbert",args.smoke)
        if old is not None:
            print(f"ORIGINAL_REUSED encoder=modernbert lambda={lam:g}",flush=True)
            continue
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Incomplete output, inspect manually: {folder}")
        set_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=False)
        model=AutoModelForSequenceClassification.from_pretrained(
            args.model,num_labels=1,problem_type="regression",trust_remote_code=False)
        trainset=ModernDataset(used_train,tokenizer,args.max_length,lam)
        valset=ModernDataset(used_val,tokenizer,args.max_length,lam)
        folder.mkdir(parents=True,exist_ok=True)
        config=TrainingArguments(
            output_dir=str(folder/"checkpoints"),
            num_train_epochs=1. if args.smoke else args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            weight_decay=.01,warmup_ratio=.1,lr_scheduler_type="linear",
            eval_strategy="epoch",save_strategy="epoch",save_total_limit=2,
            load_best_model_at_end=True,metric_for_best_model="eval_loss",
            greater_is_better=False,fp16=True,optim="adamw_torch",
            dataloader_num_workers=0,report_to="none",logging_steps=25,
            seed=args.seed,data_seed=args.seed)
        trainer=Trainer(model=model,args=config,train_dataset=trainset,
                        eval_dataset=valset,
                        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                        processing_class=tokenizer)
        trainer.train()
        if not trainer.state.best_model_checkpoint:
            raise RuntimeError("No selected validation-loss checkpoint")
        trainer.save_model(str(folder/"best_model"))
        tokenizer.save_pretrained(str(folder/"best_model"))
        pred=np.asarray(trainer.predict(valset).predictions).reshape(-1)
        if len(pred)!=len(used_val) or not np.isfinite(pred).all():
            raise RuntimeError("Invalid ModernBERT predictions")
        scores={r["sample_id"]:float(v) for r,v in zip(used_val,pred)}
        write_completed(folder,scores,base,lam,"modernbert",len(used_train),
                        len(used_val),args.smoke,{
            "encoder_frozen":False,"model":args.model,"architecture":"1-logit MSE",
            "learning_rate":args.lr,"epochs":1. if args.smoke else args.epochs,
            "batch_size":args.batch_size,"max_length":args.max_length,
            "seed":args.seed,"best_checkpoint":trainer.state.best_model_checkpoint,
            "best_eval_loss":float(trainer.state.best_metric),
        })
        print(f"ORIGINAL_TRAINED encoder=modernbert lambda={lam:g} "
              f"best_eval_mse={trainer.state.best_metric:.6f}",flush=True)
        del trainer,model
        torch.cuda.empty_cache()


def write_csv(path,records):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def old_reference(encoder,val,base):
    """Historical router references remain separately labelled, not controls."""
    from route_e_equal_budget import load_scores
    refs={}
    if encoder == "p1":
        p=P1_DIR/"validation_p_nt.json"
        if p.exists():
            refs["P1 Classification"]=load_scores(p,[r["sample_id"] for r in val])
        cost=ROOT/"outputs/route_e/token_aware/full/validation_predictions.json"
        if cost.exists():
            data=json.loads(cost.read_text(encoding="utf-8"))
            if data.get("strict_val_sha256")!=base.VAL_SHA256:
                raise RuntimeError("Cost-only reference validation SHA mismatch")
            refs["P1 Cost-only"]={
                k:float(v["delta_t_pred"]) for k,v in data["predictions"].items()}
    else:
        p=ROOT/"outputs/external_modernbert/utility_ablation/classification/validation_predictions.jsonl"
        if p.exists():
            rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            refs["ModernBERT Classification"]={
                str(x["sample_id"]):float(x["predicted_p_nothinking"]) for x in rows}
    for name,scores in refs.items():
        if set(scores)!={r["sample_id"] for r in val}:
            raise RuntimeError(f"{name}: old reference IDs mismatch")
    return refs


def evaluate(args,base,val):
    from route_e_equal_budget import dense_curve,linear_bracket
    from compute_nauc_ac import calculate
    curves={}
    scores_by_name={}
    results=[]
    dense_by_name={}
    for lam in args.lambdas:
        folder=output_dir(args.encoder,lam)
        scores=read_scores(folder,val,base.TRAIN_SHA256,base.VAL_SHA256,
                           lam,args.encoder)
        if scores is None:
            raise FileNotFoundError(f"Missing trained original λ={lam:g}: {folder}")
        name=f"Utility-Original λ={lam:g}"
        scores_by_name[name]=scores
    scores_by_name.update(old_reference(args.encoder,val,base))
    for name,scores in scores_by_name.items():
        dense=dense_curve(val,scores)
        dense_by_name[name]=dense
        curves[name]={k:(float(dense[450-k]["tokens"]),
                          float(dense[450-k]["accuracy"])) for k in range(451)}
    metrics={x["method"]:x for x in calculate(curves)}
    for name in scores_by_name:
        m=metrics[name]
        lam=(float(name.split("λ=")[-1]) if name.startswith("Utility-Original") else None)
        results.append({
            "method":name,"objective":"utility_original" if lam is not None else "reference",
            "lambda":lam,"AUC_AC":m["AUC_AC"],"nAUC_AC":m["nAUC_AC"],
            "mean_gain_pp":m["mean_gain_pp"],"cost_backtracks":m["cost_backtracks"],
        })
        print(f"ORIGINAL_NAUC encoder={args.encoder} {name} "
              f"nAUC={m['nAUC_AC']:+.4f}",flush=True)
    if "P1 Classification" in metrics and abs(metrics["P1 Classification"]["nAUC_AC"]-23.4831)>.025:
        raise RuntimeError("P1 historical nAUC reproduction failed")
    if "P1 Cost-only" in metrics and abs(metrics["P1 Cost-only"]["nAUC_AC"]-38.3885)>.025:
        raise RuntimeError("P1 Cost-only historical nAUC reproduction failed")
    if "ModernBERT Classification" in metrics and abs(metrics["ModernBERT Classification"]["nAUC_AC"]+6.92318)>.025:
        raise RuntimeError("ModernBERT original Classification control reproduction failed")
    best=min((x for x in results if x["objective"]=="utility_original"),
             key=lambda x:(-x["nAUC_AC"],x["lambda"]))
    print(f"ORIGINAL_DEV_BEST encoder={args.encoder} lambda={best['lambda']:g} "
          f"nAUC={best['nAUC_AC']:+.4f} VALIDATION_ONLY",flush=True)
    out=REPORT_ROOT/args.encoder
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Results exist: {out}; pass --overwrite to replace ONLY summaries")
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"nauc_ac_lambda_sweep.csv",results)
    policies=[]
    budgets=[]
    for name,dense in dense_by_name.items():
        for k in range(451):
            policies.append({"method":name,"n_thinking":k,
                "thinking_fraction":k/450,"accuracy":curves[name][k][1],
                "mean_output_tokens":curves[name][k][0]})
        for budget in BUDGETS:
            p=linear_bracket(dense,budget)
            budgets.append({"method":name,"budget":int(budget),
                            "accuracy_pct":100*p["accuracy"],
                            "expected_nothinking_fraction":p["expected_nothinking_fraction"]})
    write_csv(out/"all_451_threshold_policies.csv",policies)
    write_csv(out/"fixed_budget_accuracy.csv",budgets)
    (out/"protocol_and_selection.json").write_text(json.dumps({
        "status":"Utility-Original new experiment, exploratory validation only",
        "encoder":args.encoder,"objective":"continuous DeltaU_N MSE",
        "lambda_grid":args.lambdas,
        "strict_train_sha256":base.TRAIN_SHA256,
        "strict_val_sha256":base.VAL_SHA256,
        "nAUC_metric":"same cost-sorted 451-policy integration as scripts/compute_nauc_ac.py",
        "best_validation_lambda":best["lambda"],
        "best_validation_nAUC_AC":best["nAUC_AC"],
        "random_reference_AUC":metrics[best["method"]]["random_reference_AUC"],
        "never_tuned_using_test":True,
        "encoder_or_router_overhead_included":False,
        "historical_hard_soft_selection_unchanged":True,
    },indent=2)+"\n",encoding="utf-8")
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            seq=sorted((r for r in results if r["objective"]=="utility_original"),
                       key=lambda r:r["lambda"])
            fig,ax=plt.subplots(figsize=(7.5,4.4))
            ax.plot([r["lambda"] for r in seq],[r["nAUC_AC"] for r in seq],
                    "-o",label="Utility-Original (MSE)")
            for r in results:
                if r["objective"]=="reference":
                    ax.axhline(r["nAUC_AC"],ls="--",lw=1,label=r["method"])
            ax.set(xlabel="lambda",ylabel="nAUC-AC",
                   title=f"{args.encoder}: raw Utility regression")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            fig.tight_layout()
            for ext in ("png","svg"):
                fig.savefig(out/f"nauc_vs_lambda.{ext}",dpi=200,bbox_inches="tight")
            plt.close(fig)
        except ImportError:
            print("ORIGINAL_PLOT_SKIPPED: matplotlib unavailable",flush=True)
    print(f"UTILITY_ORIGINAL_DONE encoder={args.encoder} path={out}",flush=True)


def self_test():
    row={"p_n":.6,"p_t":.7,"t_n":500.,"t_t":2500.}
    assert abs(target_one(row,0.) + .1)<1e-8
    assert abs(target_one(row,.05))<1e-8
    assert abs(target_one(row,.10)-.1)<1e-8
    assert make_targets([row,row],.1).shape==(2,)
    assert np.isfinite(make_targets([row],.1)).all()
    assert label(.005)=="utility_original_lam0.005"
    assert len(LAMBDAS)==6
    print("UTILITY_ORIGINAL_SELFTEST_PASS")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    for cmd in ("plan","train","evaluate"):
        q=sub.add_parser(cmd)
        q.add_argument("--encoder",required=True,choices=("p1","modernbert"))
        q.add_argument("--lambdas",nargs="+",type=float,default=list(LAMBDAS))
        q.add_argument("--overwrite",action="store_true",
                       help="Only for re-evaluating result summaries, never overwrite models")
        q.add_argument("--no-plot",action="store_true")
        q.add_argument("--smoke",action="store_true")
        q.add_argument("--l2",type=float,default=1e-3)
        q.add_argument("--max-iter",type=int,default=200)
        q.add_argument("--model",default="answerdotai/ModernBERT-base")
        q.add_argument("--lr",type=float,default=2e-5)
        q.add_argument("--epochs",type=float,default=5.)
        q.add_argument("--batch-size",type=int,default=16)
        q.add_argument("--max-length",type=int,default=512)
        q.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    if args.command=="self-test":
        self_test()
        return
    if not args.lambdas or len(set(args.lambdas))!=len(args.lambdas) or \
       any(not math.isfinite(x) or x<0 for x in args.lambdas):
        p.error("lambda grid must be unique, finite and nonnegative")
    if args.encoder=="p1" and (abs(args.l2-1e-3)>1e-12 or args.max_iter!=200):
        p.error("Frozen P1 uses fixed L2=1e-3 and max_iter=200")
    if args.encoder=="modernbert" and (
        args.model!="answerdotai/ModernBERT-base" or abs(args.lr-2e-5)>1e-12
        or args.epochs!=5. or args.batch_size!=16 or args.max_length!=512 or args.seed!=42
    ):
        p.error("Keep original ModernBERT hyperparameters to isolate Utility-Original")
    if args.smoke and (args.encoder!="modernbert" or args.command!="train" or len(args.lambdas)!=1):
        p.error("--smoke only allowed with ModernBERT train and one lambda")
    if args.overwrite and args.command!="evaluate":
        p.error("--overwrite only applies to evaluate result summaries")
    base,train,val,_=protocol()
    if args.command=="plan":
        for lam in args.lambdas:
            folder=output_dir(args.encoder,lam)
            scores=read_scores(folder,val,base.TRAIN_SHA256,base.VAL_SHA256,lam,args.encoder)
            status="REUSE_COMPLETE" if scores is not None else "NEEDS_TRAINING"
            print(f"ORIGINAL_PLAN encoder={args.encoder} lambda={lam:g} "
                  f"status={status} folder={folder}",flush=True)
    elif args.command=="train":
        if args.encoder=="p1":
            train_p1(args,base,train,val)
        else:
            train_modern(args,base,train,val)
    else:
        if args.smoke:
            p.error("Smoke results may not be evaluated as full validation")
        evaluate(args,base,val)


if __name__=="__main__":
    main()
