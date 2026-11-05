#!/usr/bin/env python3
"""AdaptThink: training-free LRM self-assessment of Thinking vs NoThinking.

Only the MODE-PREFERENCE task is implemented here. The frozen LRM is asked
which mode has higher expected answer correctness (ties => NoThinking),
not which mode optimizes cost. For each question, read the ORIGINAL language
model head's logits on single-token ' A'/' B' continuations; no text is
sampled, no LRM parameters or probe weights are updated.

This is a system-level comparison with existing P1 Logistic Regression,
NOT a controlled LM-head-vs-linear-head comparison on identical hidden states.

CPU (TC1 login):
    python scripts/route_self_assessment.py self-test
    python scripts/route_self_assessment.py audit
GPU (Slurm allocation; NEVER login):
    python scripts/route_self_assessment.py extract --split train --smoke
    python scripts/route_self_assessment.py extract --split val
CPU (after full val):
    python scripts/route_self_assessment.py evaluate

A/B-swapped prompts are a predeclared sensitivity test; original ordering is
primary. The current script does not perform success-rate/token-cost self
predictions; these are separate follow-up tasks requiring separate prompts.
"""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import route_e_frozen as base
import route_e_prompt_ablation as prefix
import route_e_equal_budget as budget_eval

ROOT = base.ROOT
DEFAULT_MODEL = base.DEFAULT_MODELS["deepseek"]
TASK = "mode"
VARIANTS = ("original", "swapped")
CANDIDATES = {"A": " A", "B": " B"}
DEFAULT_BUDGETS = budget_eval.DEFAULT_BUDGETS
SCORE_SCHEMA = "adaptthink_self_assessment_mode_v1"

INSTRUCTION = """You are assessing how reliably you would solve a mathematics problem.

Thinking: solve using extended step-by-step reasoning.
NoThinking: answer directly without extended reasoning.

Choose the mode that has a higher probability of producing a correct final
answer. Compare correctness only: ignore time, length, and token costs.
If both modes have equal probability of being correct, choose NoThinking.
Do not solve the problem. Select just one letter.

Problem:
{question}

Options:
{options}
"""
OPTIONS = {
    "original": "A: Thinking\nB: NoThinking",
    "swapped": "A: NoThinking\nB: Thinking",
}


def output_dir(smoke):
    return ROOT / "outputs/route_e/self_assessment/mode" / ("smoke" if smoke else "full")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_hash():
    return sha256_text(json.dumps({"instruction": INSTRUCTION, "options": OPTIONS,
                                   "candidate_texts": CANDIDATES}, sort_keys=True))


def render_prompt(tokenizer, question, variant):
    """Exact tokenizer chat template + P1 assistant prefix + partial answer."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Empty problem")
    prompt_user = INSTRUCTION.format(question=question, options=OPTIONS[variant])
    p1 = prefix.render_prompts(tokenizer, prompt_user)["assistant"]
    return p1 + "Choice:"


def encode(tokenizer, text):
    return tokenizer(text, add_special_tokens=False,
                     truncation=False)["input_ids"]


def candidate_token_ids(tokenizer, rendered, prefix_ids=None):
    """Fail if appending an answer does not extend prompt by ONE token.

    Tokenizing the candidate alone is insufficient for BPE tokenizers:
    concatenation with the context can change boundaries. Verify full
    prefix invariance instead. This must pass for each actual sample.
    """
    if prefix_ids is None:
        prefix_ids = encode(tokenizer, rendered)
    result = {}
    for label, suffix in CANDIDATES.items():
        total = encode(tokenizer, rendered + suffix)
        if len(total) != len(prefix_ids) + 1 or total[:-1] != prefix_ids:
            raise ValueError(
                f"TOKEN_BOUNDARY_FAIL for {label}: appending {suffix!r} to "
                "the prompt does not produce exactly one new token. "
                "Inspect your real tokenizer before any GPU extraction.")
        result[label] = int(total[-1])
    if result["A"] == result["B"]:
        raise ValueError("A and B candidate tokens collide")
    return result


def load_local_tokenizer(model_path):
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(model_path),
                                      local_files_only=True, trust_remote_code=False)
    t.padding_side = "right"
    if t.pad_token_id is None:
        t.pad_token = t.eos_token
    return t


def audit_all(tokenizer, max_length):
    train, val, overlaps = base.load_protocol()
    all_data = (("train", train), ("val", val))
    lengths = {variant: [] for variant in VARIANTS}
    candidate_ids = None
    examples = {}
    checked = 0
    for split, rows in all_data:
        for row in rows:
            for variant in VARIANTS:
                text = render_prompt(tokenizer, row["problem"], variant)
                ids = encode(tokenizer, text)
                if not ids or len(ids) > max_length:
                    raise ValueError(
                        f"Prompt overlength/empty: {split} {row['sample_id']} "
                        f"variant={variant} length={len(ids)} max={max_length}. "
                        "No silent truncation permitted.")
                found = candidate_token_ids(tokenizer, text, ids)
                if candidate_ids is None:
                    candidate_ids = found
                if found != candidate_ids:
                    raise RuntimeError(f"Different A/B token IDs at {split} "
                                       f"{row['sample_id']}: {found} != {candidate_ids}")
                lengths[variant].append(len(ids))
                if not examples.get(variant):
                    examples[variant] = text[-550:]
            checked += 1
    report = {
        "schema": "adaptthink_self_assessment_token_audit_v1",
        "model": str(getattr(tokenizer, "name_or_path", "unknown")),
        "train_sha256": base.TRAIN_SHA256,
        "validation_sha256": base.VAL_SHA256,
        "count": checked,
        "train_val_text_overlaps": overlaps,
        "prompt_sha256": prompt_hash(),
        "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        "max_length": max_length,
        "candidate_suffixes": CANDIDATES,
        "candidate_token_ids": candidate_ids,
        "input_tokens": {
            mode: {"min": min(v), "max": max(v), "mean": sum(v)/len(v)}
            for mode, v in lengths.items()},
        "example_prompt_suffixes": examples,
        "passed": True,
    }
    return report


def audit_command(args):
    tok = load_local_tokenizer(args.model)
    report = audit_all(tok, args.max_length)
    output = ROOT / "results/route_e/self_assessment" / (
        f"mode_token_audit_{prompt_hash()[:12]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior != report:
            raise FileExistsError(f"Refusing to overwrite differing audit: {output}")
    else:
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                          encoding="utf-8")
    print(f"SELF_AUDIT_PASS count={report['count']} "
          f"candidate_ids={report['candidate_token_ids']} path={output}")
    for variant in VARIANTS:
        d = report["input_tokens"][variant]
        print(f"SELF_AUDIT variant={variant} input_tokens_mean={d['mean']:.1f} "
              f"min={d['min']} max={d['max']}")
        print(f"SELF_PROMPT_EXAMPLE variant={variant} "
              f"repr={report['example_prompt_suffixes'][variant]!r}")


def pick_rows(smoke, split):
    train, val, _ = base.load_protocol()
    if smoke:
        if split != "train":
            raise ValueError("Smoke must use TRAIN split only, not validation")
        return base.choose_smoke_rows(train, val)[0], base.TRAIN, "train"
    return (val, base.VAL, "val") if split == "val" else (
        train, base.TRAIN, "train")


def score_margins(logit_a, logit_b, variant):
    """Larger margin always means a greater preference for NoThinking."""
    if variant == "original":
        return logit_b - logit_a  # A=T, B=N
    if variant == "swapped":
        return logit_a - logit_b  # A=N, B=T
    raise ValueError(variant)


def extract(args):
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required: launch under a Slurm GPU allocation")
    if args.smoke and args.split != "train":
        raise ValueError("Smoke uses TRAIN only; pass --split train")
    tokenizer = load_local_tokenizer(args.model)
    # Verify ALL train/val inputs and answer-token boundaries before GPU load.
    audit = audit_all(tokenizer, args.max_length)
    rows, labels_path, effective_split = pick_rows(args.smoke, args.split)
    dst = output_dir(args.smoke) / f"{effective_split}_predictions.json"
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {dst}")
    model_path = Path(args.model)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16,
        local_files_only=True, trust_remote_code=False).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    record = {row["sample_id"]: {"sample_id": row["sample_id"]}
              for row in rows}
    timing = {}
    for variant in VARIANTS:
        sequences = [encode(tokenizer, render_prompt(tokenizer,r["problem"],variant))
                     for r in rows]
        ids_expected = audit["candidate_token_ids"]
        timings, counts = [], []
        start = time.perf_counter()
        with torch.inference_mode():
            for first in range(0, len(rows), args.batch_size):
                chunk = sequences[first:first+args.batch_size]
                width = max(map(len,chunk))
                tokens = torch.full((len(chunk),width),tokenizer.pad_token_id,
                                    device="cuda",dtype=torch.long)
                mask = torch.zeros_like(tokens)
                for i,seq in enumerate(chunk):
                    tokens[i,:len(seq)] = torch.tensor(seq,device="cuda")
                    mask[i,:len(seq)] = 1
                torch.cuda.synchronize()
                tick = time.perf_counter()
                outputs = model(input_ids=tokens, attention_mask=mask,
                                return_dict=True,use_cache=False)
                positions = torch.tensor([len(seq)-1 for seq in chunk],
                                         dtype=torch.long,device="cuda")
                final = outputs.logits[
                    torch.arange(len(chunk),device="cuda"),positions].float()
                za = final[:,ids_expected["A"]]
                zb = final[:,ids_expected["B"]]
                margin = score_margins(za,zb,variant)
                binary_probability = torch.sigmoid(margin)
                # Probability mass assigned to A or B among all vocabulary tokens.
                log_ab = torch.logsumexp(torch.stack((za,zb),dim=-1),dim=-1)
                log_vocab = torch.logsumexp(final,dim=-1)
                mass_ab = torch.exp(log_ab-log_vocab)
                torch.cuda.synchronize()
                elapsed = 1000*(time.perf_counter()-tick)
                if not (torch.isfinite(margin).all() and
                        torch.isfinite(mass_ab).all()):
                    raise RuntimeError(f"Nonfinite logits: {variant} batch={first}")
                z_a = za.cpu().numpy()
                z_b = zb.cpu().numpy()
                s = margin.cpu().numpy()
                p = binary_probability.cpu().numpy()
                mass = mass_ab.cpu().numpy()
                for j,r in enumerate(rows[first:first+len(chunk)]):
                    record[r["sample_id"]][variant] = {
                        "logit_A": float(z_a[j]), "logit_B": float(z_b[j]),
                        "nt_minus_t_logit": float(s[j]),
                        "p_nt_conditional_AB": float(p[j]),
                        "full_vocab_mass_AB": float(mass[j]),
                        "input_tokens": len(chunk[j]),
                    }
                timings.append(elapsed)
                counts.append(len(chunk))
                if first == 0 or first+len(chunk) == len(rows) or (
                        first // args.batch_size) % 20 == 0:
                    print(f"SELF_EXTRACT variant={variant} "
                          f"split={effective_split} "
                          f"n={first+len(chunk)}/{len(rows)}",flush=True)
        timing[variant] = {
            "first_batch_ms": timings[0],
            "post_first_batch_ms_per_question": (
                sum(timings[1:])/sum(counts[1:]) if len(counts)>1 else None),
            "forward_total_seconds": sum(timings)/1000,
            "variant_wall_seconds": time.perf_counter()-start,
        }
        masses = [record[r["sample_id"]][variant]["full_vocab_mass_AB"]
                  for r in rows]
        margins = [record[r["sample_id"]][variant]["nt_minus_t_logit"]
                   for r in rows]
        print(f"SELF_SCORE_DIAG variant={variant} split={effective_split} "
              f"mean_full_vocab_AB_mass={sum(masses)/len(masses):.6f} "
              f"frac_mass_below_1pct={sum(v<.01 for v in masses)/len(masses):.3f} "
              f"frac_prefers_NoThinking={sum(s>=0 for s in margins)/len(margins):.3f} "
              f"margin_min={min(margins):.3f} margin_max={max(margins):.3f}",
              flush=True)

    # Original and swapped are paired by sample ID; no oracle label is used.
    for r in rows:
        d=record[r["sample_id"]]
        d["avg_nt_minus_t_logit"] = 0.5 * (
            d["original"]["nt_minus_t_logit"] +
            d["swapped"]["nt_minus_t_logit"])

    meta = {
        "schema": SCORE_SCHEMA, "task": TASK, "smoke": bool(args.smoke),
        "split": effective_split, "sample_count": len(rows),
        "labels_sha256": base.sha256(labels_path),
        "model_path": str(model_path),
        "model_revision": getattr(model.config,"_commit_hash",None),
        "prompt_sha256": prompt_hash(),
        "chat_template_sha256": audit["chat_template_sha256"],
        "candidate_token_ids": audit["candidate_token_ids"],
        "score_direction": "higher = more NoThinking preference",
        "primary_score": "original.nt_minus_t_logit",
        "swapped_score": "swapped.nt_minus_t_logit (diagnostic only)",
        "dtype": str(next(model.parameters()).dtype),
        "max_length": args.max_length, "batch_size": args.batch_size,
        "seed": args.seed,
        "input_generation": "forward only; no answers sampled",
        "trainable_parameters": 0,
        "timing": timing,
        "router_overhead_included_in_generated_token_curve": False,
    }
    dst.parent.mkdir(parents=True,exist_ok=True)
    dst.write_text(json.dumps({"metadata":meta,"predictions":list(record.values())},
                              indent=2)+"\n",encoding="utf-8")
    print(f"SELF_EXTRACT_DONE split={effective_split} smoke={int(args.smoke)} "
          f"records={len(rows)} path={dst}",flush=True)


def read_predictions(path, rows):
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    meta=obj["metadata"]
    if meta["schema"] != SCORE_SCHEMA or meta["task"] != TASK or meta["smoke"]:
        raise ValueError(f"Wrong schema/task or smoke predictions: {path}")
    if meta["split"] != "val" or meta["labels_sha256"] != base.VAL_SHA256:
        raise ValueError(f"Validation SHA/split mismatch: {path}")
    if meta["prompt_sha256"] != prompt_hash():
        raise ValueError("Prompt changed since predictions were extracted")
    preds=obj["predictions"]
    ids=[r["sample_id"] for r in rows]
    if [d["sample_id"] for d in preds] != ids:
        raise ValueError("Validation sample IDs/order mismatch")
    for d in preds:
        for variant in VARIANTS:
            stats=d[variant]
            if not all(math.isfinite(stats[k]) for k in (
                    "logit_A","logit_B","nt_minus_t_logit",
                    "p_nt_conditional_AB","full_vocab_mass_AB")):
                raise ValueError("Nonfinite prediction record")
    return preds,meta


def classification_stats(rows,score_dict):
    # Ground truth is EXACTLY the original P1 label; equal=>NoThinking.
    y = [int(r["p_n"] >= r["p_t"]) for r in rows]
    scores = [score_dict[r["sample_id"]] for r in rows]
    p = [int(v >= 0.) for v in scores]  # logit NT-vs-T threshold 0
    tp=sum(int(yp==1 and yt==1) for yp,yt in zip(p,y))
    tn=sum(int(yp==0 and yt==0) for yp,yt in zip(p,y))
    fp=sum(int(yp==1 and yt==0) for yp,yt in zip(p,y))
    fn=sum(int(yp==0 and yt==1) for yp,yt in zip(p,y))
    sensitivity=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    bacc=0.5*(sensitivity+specificity)
    return {"auc": base.auc(scores,y), "accuracy":(tp+tn)/len(y),
            "balanced_accuracy":bacc,"tp":tp,"tn":tn,"fp":fp,"fn":fn,
            "predicted_nt_fraction":sum(p)/len(y)}


def evaluate(args):
    rows=budget_eval.check_data()  # 450-ID strict val SHA, counts checked
    path=output_dir(False)/"val_predictions.json"
    preds,meta=read_predictions(path,rows)
    ids=[r["sample_id"] for r in rows]
    scores={
        "Self_original": {r["sample_id"]:r["original"]["nt_minus_t_logit"]
                          for r in preds},
        "Self_swapped": {r["sample_id"]:r["swapped"]["nt_minus_t_logit"]
                          for r in preds},
        "Self_average_diagnostic": {r["sample_id"]:r["avg_nt_minus_t_logit"]
                                    for r in preds},
    }
    reference_path=ROOT/"outputs/route_e/prompt_ablation/assistant/full/validation_p_nt.json"
    ref=budget_eval.load_scores(reference_path,ids)
    scores["P1_Logistic"]={k: math.log(max(min(v,1-1e-9),1e-9)/
                                         (1-max(min(v,1-1e-9),1e-9)))
                           for k,v in ref.items()}
    p1curve=budget_eval.dense_curve(rows,ref)
    p1point=p1curve[225]
    if abs(p1point["accuracy"]-344/450)>0.0003 or abs(p1point["tokens"]-1427.05)>2:
        raise RuntimeError("P1 reproduction gate failed; wrong score source/orientation")
    diagnostics={name:classification_stats(rows,one) for name,one in scores.items()}
    weights={variant:[r[variant]["full_vocab_mass_AB"] for r in preds]
             for variant in VARIANTS}
    for variant,vals in weights.items():
        diagnostics["Self_"+variant]["candidate_full_vocab_mass_mean"] = sum(vals)/len(vals)
        diagnostics["Self_"+variant]["candidate_mass_below_1pct_fraction"] = (
            sum(v<.01 for v in vals)/len(vals))

    curve={name:budget_eval.dense_curve(rows,one) for name,one in scores.items()}
    from train_modernbert_delta import endpoints
    endpoints_data=endpoints(rows)
    lo=endpoints_data["nothinking"]["tokens"]
    hi=endpoints_data["thinking"]["tokens"]
    if not all(math.isfinite(b) and lo<=b<=hi for b in args.budgets):
        raise ValueError(f"Budget outside observable range [{lo:.1f}, {hi:.1f}]")
    if len(args.budgets)!=len(set(args.budgets)):
        raise ValueError("Repeated budget")
    tab=[]
    for b in sorted(args.budgets):
        point={"budget":b,"Global_Random":budget_eval.random_at_budget(rows,b)["accuracy"]}
        for name,c in curve.items():
            point[name]=budget_eval.linear_bracket(c,b)["accuracy"]
        tab.append(point)
    out=ROOT/"results/route_e/self_assessment"
    out.mkdir(parents=True,exist_ok=True)
    dest=out/"mode_val.json"
    if dest.exists() and not args.overwrite_report:
        raise FileExistsError(f"Refusing to overwrite report: {dest}. "
                              "Pass --overwrite-report after inspecting it.")
    report={
        "experiment":"zero_shot_mode_self_assessment_vs_frozen_P1",
        "validation_sha256":base.VAL_SHA256,
        "validation_count":len(rows),
        "input_predictions":str(path),
        "p1_reference":str(reference_path),
        "primary_self_variant":"Self_original",
        "swapped_diagnostic":"Self_swapped",
        "model_no_finetuning":True,
        "prompt_differs_from_P1":True,
        "metric_explanation":"ROC AUC for empirical binary NT>=T; conditional AB score is NOT correctness probability",
        "equal_budget_explanation":"offline; 451 top-k policies; interpolated using realized validation generated-token costs",
        "extra_self_assessment_prefill_included_in_cost":False,
        "p1_50pct_reproduction":p1point,
        "model_logit_meta":meta,
        "classification_metrics":diagnostics,
        "budget_points":tab,
    }
    dest.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    csv_path=out/"mode_budget_points.csv"
    fields=["budget","Global_Random","P1_Logistic","Self_original",
            "Self_swapped","Self_average_diagnostic"]
    with csv_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(tab)
    print("SELF_EVAL_VALIDATED val=450 P1_50pct="
          f"{100*p1point['accuracy']:.4f}% tokens={p1point['tokens']:.2f}")
    for name in ("Self_original","Self_swapped","Self_average_diagnostic","P1_Logistic"):
        d=diagnostics[name]
        print(f"SELF_EVAL {name} auc={d['auc']:.4f} "
              f"acc={100*d['accuracy']:.2f}% bacc={100*d['balanced_accuracy']:.2f}% "
              f"nt_pred_frac={d['predicted_nt_fraction']:.3f}")
    print("BUDGET | Random | P1 | Self original | Self swapped | Self avg(diag)")
    for p in tab:
        print(f"{p['budget']:.0f} | "+" | ".join(
            f"{100*p[name]:.2f}%" for name in fields[1:]))
    print(f"SELF_EVAL_DONE report={dest} csv={csv_path}")


def self_test():
    # No real tokenizer/model needed for a CPU smoke of string/score plumbing.
    class FakeTokenizer:
        chat_template="fake"
        name_or_path="fake"
        def apply_chat_template(self,messages,tokenize,add_generation_prompt):
            content="<BOS><｜User｜>"+messages[0]["content"]
            if add_generation_prompt:
                content += prefix.ASSISTANT_START+prefix.THINK_START
            return [ord(x) for x in content] if tokenize else content
        def __call__(self,text,add_special_tokens,truncation=False):
            assert not add_special_tokens and not truncation
            if text.endswith(" A") or text.endswith(" B"):
                tail=501 if text.endswith(" A") else 502
                return {"input_ids":[ord(x) for x in text[:-2]]+[tail]}
            return {"input_ids":[ord(x) for x in text]}
    t=FakeTokenizer()
    for variant in VARIANTS:
        rendered=render_prompt(t,"1 + 1 = ?",variant)
        assert rendered.endswith("Choice:")
        assert candidate_token_ids(t,rendered)=={"A":501,"B":502}
    assert score_margins(2.,3.,"original")==1.
    assert score_margins(3.,2.,"swapped")==1.
    rows=[{"sample_id":"x","p_n":.5,"p_t":.5},
          {"sample_id":"y","p_n":0.,"p_t":1.}]
    d=classification_stats(rows,{"x":2.,"y":-1.})
    assert d["accuracy"]==1. and d["auc"]==1. and d["balanced_accuracy"]==1.
    print("SELF_ASSESSMENT_SELFTEST_PASS")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("self-test")
    au=sub.add_parser("audit")
    au.add_argument("--model",default=DEFAULT_MODEL)
    au.add_argument("--max-length",type=int,default=1024)
    ex=sub.add_parser("extract")
    ex.add_argument("--model",default=DEFAULT_MODEL)
    ex.add_argument("--split",choices=("train","val"),default="train")
    ex.add_argument("--smoke",action="store_true")
    ex.add_argument("--batch-size",type=int,default=8)
    ex.add_argument("--max-length",type=int,default=1024)
    ex.add_argument("--seed",type=int,default=42)
    ev=sub.add_parser("evaluate")
    ev.add_argument("--budgets",nargs="+",type=float,default=DEFAULT_BUDGETS)
    ev.add_argument("--overwrite-report",action="store_true")
    args=p.parse_args()
    if args.command=="self-test":
        self_test()
    elif args.command=="audit":
        if args.max_length<=0:p.error("max-length must be positive")
        audit_command(args)
    elif args.command=="extract":
        if args.batch_size<=0 or args.max_length<=0:p.error("batch-size/max-length must be positive")
        extract(args)
    else:
        evaluate(args)


if __name__=="__main__":
    main()
