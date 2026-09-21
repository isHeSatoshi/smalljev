"""Evaluate the semantic-v1 system on the nimble13 13-subset battery.

Choice: batched single-pass span-pooled semantic scoring (shared scorer).
Noul/Score: readout + bit/ordinal heads (same decision shape as crown_scores).

Usage: python evals/eval_semantic.py [--only massive-en-US] [--out evals/arms/semantic-v1]
Out: evals/semantic13.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from smalljev.heads import OptionScorerHead, BinaryNoulHead, OrdinalScoreHead
from smalljev.model import render_prompt
from smalljev.semantic import build_semantic_ids
from evals.train_heads import readout_hidden
from evals.nimble13 import BUILDERS, SUBSET_QUESTIONS, summarize


def load_system(prefix, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B-Base",
                                        trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        "openbmb/MiniCPM5-2B-Base", torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, attn_implementation="eager").eval()
    model = PeftModel.from_pretrained(base, f"{prefix}-lora").eval()
    scorer = OptionScorerHead.load(f"{prefix}-scorer.pt").to(device).eval()
    blob = torch.load(f"{prefix}-noulscore.pt", map_location=device,
                       weights_only=True)
    noul = BinaryNoulHead(blob["hidden_size"]).to(device).eval()
    noul.load_state_dict(blob["noul"])
    score = OrdinalScoreHead(blob["hidden_size"], 8).to(device).eval()
    score.load_state_dict(blob["score"])
    return tok, model, scorer, noul, score, device


def semantic_choice_probs(tok, model, scorer, items, device, batch=8):
    """items: list of (state, question, options). Returns list of prob arrays."""
    out = []
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        ids_list, spans_list = [], []
        for state, q, opts in chunk:
            ids, spans = build_semantic_ids(tok, state, q, list(opts))
            ids_list.append(ids)
            spans_list.append(spans)
        pad = tok.pad_token_id
        mx = max(len(x) for x in ids_list)
        input_ids = torch.tensor([[*x, *[pad] * (mx - len(x))] for x in ids_list],
                                 device=device)
        with torch.no_grad():
            H_all = model(input_ids=input_ids, use_cache=False,
                          output_hidden_states=True).hidden_states[-1].float()
            for r, spans in enumerate(spans_list):
                reps = torch.stack([H_all[r, a:b, :].mean(0) for a, b in spans])
                p = scorer.probs(reps.unsqueeze(0).to(
                    next(scorer.parameters()).device)).float().cpu().numpy()[0]
                out.append(p / p.sum())
    return out


def system_scores(name, recs, C, batch=8, noul_mode="bit"):
    tok, model, scorer, noul, score_head, device = C
    q = SUBSET_QUESTIONS[name]
    out = []
    if recs[0]["kind"] == "choice":
        opts = recs[0]["options"]
        # battery choice recs share one option list per subset (massive/mnli/...)
        # except vitaminc/pubmedqa which also share. Batch freely.
        triplets = [(r["state"], q, r["options"]) for r in recs]
        for r, p in zip(recs, semantic_choice_probs(tok, model, scorer, triplets,
                                                    device, batch)):
            out.append((r["id"], int(p.argmax()), p))
    elif recs[0]["kind"] == "noul" and noul_mode == "twin":
        # twin-choice (v6 unification experiment): statement vs its negation.
        # Falls back to the Noul bit only if no twin rule matches.
        from evals.train_semantic import noul_twin
        triplets, bit_idx = [], []
        for r in recs:
            stmt = r.get("stmt") or q
            twin = noul_twin(stmt)
            if twin is None:
                bit_idx.append(len(bit_idx))
                continue
            triplets.append((r["state"], "Which statement is correct?", [stmt, twin]))
        P = semantic_choice_probs(tok, model, scorer, triplets, device, batch)
        out = []
        ti = 0
        for r in recs:
            stmt = r.get("stmt") or q
            if noul_twin(stmt) is None:
                prompts = [render_prompt(r["state"], stmt, ["yes", "no"])]
                H = readout_hidden(model, tok, prompts, device).to(device)
                with torch.no_grad():
                    pv = float(noul.prob(H).cpu().numpy())
                out.append((r["id"], int(pv >= 0.5), np.array([1 - pv, pv])))
            else:
                p = P[ti]
                ti += 1
                pv = float(p[0] / p.sum())
                out.append((r["id"], int(pv >= 0.5), np.array([1 - pv, pv])))
        return out
    elif recs[0]["kind"] == "noul":
        # Noul bit head (v5 hybrid default): judge the statement as given.
        outs = []
        for i in range(0, len(recs), batch):
            chunk = recs[i:i + batch]
            stmts = [r.get("stmt") or q for r in chunk]
            prompts = [render_prompt(r["state"], s, ["yes", "no"])
                       for r, s in zip(chunk, stmts)]
            H = readout_hidden(model, tok, prompts, device).to(device)
            with torch.no_grad():
                outs.extend(noul.prob(H).float().cpu().numpy().tolist())
        for r, pv in zip(recs, outs):
            pv = float(pv)
            out.append((r["id"], int(pv >= 0.5), np.array([1 - pv, pv])))
    else:
        # ordinal levels as semantic options (v5+; ordinality lives in the
        # level text, not in head positions). Expected value -> round.
        triplets = [(r["state"], q, r["levels"]) for r in recs]
        for r, p in zip(recs, semantic_choice_probs(tok, model, scorer, triplets,
                                                    device, batch)):
            exp = float((p * np.arange(len(p))).sum())
            out.append((r["id"], int(round(exp)), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="evals/arms/semantic-v1")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--noul-mode", default="bit", choices=("bit", "twin"),
                    help="bit = Noul head (v5 hybrid, default); twin = statement-vs-"
                         "negation scorer (v6 unification experiment)")
    args = ap.parse_args()
    rng = np.random.RandomState(11)
    C = load_system(args.out)
    print("loaded semantic system", flush=True)
    try:
        with open("evals/semantic13.json") as f:
            R = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        R = {"subsets": {}}
    names = [k for k in BUILDERS if args.only is None or k == args.only]
    for name in names:
        try:
            recs = BUILDERS[name](rng)
            co = system_scores(name, recs, C, args.batch, args.noul_mode)
            sc = summarize(recs, co)
            sc["items"] = [[o[0], int(o[1]),
                            int(next(r["y"] for r in recs if r["id"] == o[0]))]
                           for o in co]
            if name not in R["subsets"]:
                R["subsets"][name] = {}
            R["subsets"][name]["semantic"] = sc
            R["subsets"][name]["n"] = len(recs)
            with open("evals/semantic13.json", "w") as f:
                json.dump(R, f, indent=2)
            print(f"{name}: n={len(recs)} semantic={sc['accuracy']:.3f}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
    print("wrote evals/semantic13.json")


if __name__ == "__main__":
    main()
