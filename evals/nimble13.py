"""Nimble-style 13-subset battery: same public datasets, same instructions.

Instructions/option semantics follow bespokelabsai/nimble PUBLIC_BENCHMARKS.md
verbatim (attributed). Exact-ID matching where their manifests map cleanly
(massive row ids, paws row idx, multinli pairIDs); seeded family-aware sampling
elsewhere (documented per subset). Comparability: same-distribution for most
sets; McNemar is valid ONLY crown-vs-Laya (paired, same items).

Systems: smalljev-crown (lettered bundle+lora) + Laya measured.
Nimble/Jev columns come from their published aggregates (see nimble13_pub.json).

Usage: python evals/nimble13.py [--only vitaminc] [--n-laya ...]
Out: evals/nimble13.json
"""
import argparse
import json
import os
import sys
import tarfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

SEED = 11
MANIFEST = ("C:/Users/Aditya/AppData/Local/Temp/opencode/nimble/"
            "docs/assets/public-benchmarks/subsets")
RAW = "evals/nimble13_raw"

# Their instructions, verbatim (bespokelabsai/nimble docs/PUBLIC_BENCHMARKS.md)
INST = {
    "vitaminc": "Decide how the evidence bears on the claim. Judge only from the evidence "
                "text, and not from outside knowledge about the subject.",
    "massive": "Which assistant domain should handle this utterance? Judge from the request "
               "itself, not from how it is phrased.",
    "boolq": "Does the passage answer the question with yes? Use only what the passage "
             "states or directly implies.",
    "squad2": "Does the paragraph contain the information needed to answer the question? "
              "Treat a question as unanswerable when the paragraph discusses the topic but "
              "does not state the specific fact asked for.",
    "paws": "Do the two sentences have the same meaning? Sentences that reuse the same "
            "words in a different order can still mean different things.",
    "multinli": "Assume the premise is true. How does the hypothesis relate to it?",
    "civil": "Is this comment toxic, meaning rude, disrespectful, or unreasonable enough "
             "that a reader would likely leave the discussion?",
    "aegis2": "Would a general-purpose assistant be unsafe to comply with this user "
              "message as written? Judge the request, not the assistant's likely reply.",
    "helpsteer2": "How helpful is the response to the prompt, considering whether it "
                  "addresses what was asked, is correct, and is complete?",
    "summeval-consistency": "How factually consistent is the summary with the article? Every "
                            "statement in the summary should be supported by the article.",
    "summeval-relevance": "How well does the summary capture the important content of the "
                          "article, without unimportant or redundant material?",
    "pubmedqa": "Based only on the abstract, what is the answer to the research question?",
}
VIT_C = {"SUPPORTS": "The evidence states the claim, or the claim follows directly from it.",
         "REFUTES": "The evidence states the opposite of the claim, or directly contradicts it.",
         "NOT ENOUGH INFO": "The evidence neither establishes nor contradicts the claim."}
MASSIVE_C = {
    "alarm": "Setting, changing, or querying alarms.",
    "audio": "Volume, mute, or audio output settings.",
    "calendar": "Events, meetings, reminders on a calendar.",
    "cooking": "Recipes and cooking instructions.",
    "datetime": "The current time, date, or time conversions.",
    "email": "Reading, sending, or checking email and contacts.",
    "general": "Small talk, jokes, greetings, or general assistant control.",
    "iot": "Controlling lights, plugs, heating, cleaners, coffee machines, or wemo devices.",
    "lists": "Creating, querying, or removing list items.",
    "music": "Music preferences, likes, or music settings, not playing a specific item.",
    "news": "News headlines or updates.",
    "play": "Playing a specific song, podcast, radio, audiobook, or game.",
    "qa": "Factual questions, definitions, math, currency, stock prices.",
    "recommendation": "Suggestions for events, movies, or locations.",
    "social": "Social media posts or queries.",
    "takeaway": "Food orders and delivery status.",
    "transport": "Taxis, tickets, traffic, or travel directions.",
    "weather": "Weather forecasts or conditions.",
}
MNLI_C = {"entailment": "If the premise is true, the hypothesis must be true.",
          "neutral": "The hypothesis might or might not be true; the premise does not settle it.",
          "contradiction": "If the premise is true, the hypothesis cannot be true."}
PMQA_C = {"yes": "The abstract's findings support a yes answer.",
          "no": "The abstract's findings support a no answer.",
          "maybe": "The abstract's findings are mixed, conditional, or insufficient."}
HELP_L = ["Not helpful: ignores or misreads the prompt, or is wrong in ways that make it useless.",
          "Slightly helpful: touches the request but is mostly incorrect, incomplete, or off target.",
          "Partially helpful: addresses the request with notable gaps or errors a user would need to fix.",
          "Mostly helpful: addresses the request well with minor omissions or imperfections.",
          "Extremely helpful: fully and accurately addresses the request; nothing important is missing."]
CONS_L = ["Multiple statements contradict or are absent from the article.",
          "At least one clear unsupported or contradicted statement.",
          "Mostly supported, with a minor unsupported detail.",
          "Supported, with at most a small imprecision.",
          "Every statement is supported by the article."]
REL_L = ["Misses the main points or is mostly about minor details.",
         "Captures some key content but omits important points or includes much that is unimportant.",
         "Captures the main points with noticeable omissions or filler.",
         "Captures the main points with minor omissions.",
         "Captures all key content and only key content."]
YN = ["yes", "no"]


def manifest_ids(name):
    with open(os.path.join(MANIFEST, name + "-manifest.json")) as f:
        return json.load(f)["ids"]


def build_vitaminc(rng, n=599):
    import json as _j
    rows = []
    with zipfile.ZipFile(os.path.join(RAW, "vitaminc.zip")) as z:
        with z.open("vitaminc/dev.jsonl") as f:
            for line in f:
                rows.append(_j.loads(line))
    fams = {}
    for r in rows:
        fams.setdefault(r["case_id"], []).append(r)
    fams = sorted(fams.values(), key=lambda g: g[0]["case_id"])
    order = rng.permutation(len(fams))
    out = []
    for i in order:
        if len(out) + len(fams[i]) > n + 4:
            continue
        out.extend(fams[i])
        if len(out) >= n:
            break
    recs = []
    for r in out[:n]:
        recs.append({"id": f"vitaminc-{r['case_id']}",
                     "state": f"Evidence: {r['evidence']}\nClaim: {r['claim']}",
                     "kind": "choice", "options": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
                     "y": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"].index(r["label"])})
    return recs


def _massive_rows(locale):
    import json as _j
    t = tarfile.open(os.path.join(RAW, "massive-1.1.tar.gz"))
    rows = {}
    for m in t.getmembers():
        if m.name.endswith(f"{locale}.jsonl"):
            for line in t.extractfile(m).read().decode().splitlines():
                r = _j.loads(line)
                if r.get("partition") == "test":
                    rows[r["id"]] = r
    return rows


def build_massive(locale, rng):
    ids = [i for i in manifest_ids(f"massive-{locale}") if True]
    rows = _massive_rows(locale)
    opts = sorted(MASSIVE_C)
    recs = []
    for mid in ids:
        rid = mid.split("massive-")[1]
        r = rows.get(rid)
        if r is None:
            continue
        recs.append({"id": mid, "state": r["utt"], "kind": "choice",
                     "options": opts, "y": opts.index(r["scenario"])})
    return recs


def build_boolq(rng, n=300):
    from datasets import load_dataset
    ds = load_dataset("google/boolq", split="validation")
    idx = rng.choice(len(ds), n, replace=False)
    recs = []
    for i in sorted(idx):
        r = ds[int(i)]
        recs.append({"id": f"boolq-{i}",
                     "state": f"Passage: {r['passage']}\nQuestion: {r['question']}",
                     "kind": "noul", "stmt": f"The answer to '{r['question']}' is yes.",
                     "y": int(bool(r["answer"]))})
    return recs


def build_squad2(rng):
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    paras = {}
    for i, r in enumerate(ds):
        paras.setdefault(r["context"], []).append((i, r))
    keys = sorted(paras)
    order = rng.permutation(len(keys))[:31]
    recs = []
    for k in sorted(order):
        for i, r in paras[keys[k]]:
            ans = r["answers"]["text"] if isinstance(r["answers"], dict) else []
            ok = any(t.strip() for t in ans)
            recs.append({"id": f"squad2-{r['id']}",
                         "state": f"Paragraph: {r['context']}\nQuestion: {r['question']}",
                         "kind": "noul",
                         "stmt": f"The paragraph answers '{r['question']}'.",
                         "y": int(ok)})
    return recs


def build_paws(rng):
    from datasets import load_dataset
    import re
    ds = load_dataset("google-research-datasets/paws", "labeled_final", split="test")
    ids = manifest_ids("paws")
    want = set()
    for mid in ids:
        m = re.match(r"paws-(\d+)$", mid)
        if m:
            want.add(int(m.group(1)))
    recs = []
    for i in sorted(want):
        if i >= len(ds):
            continue
        r = ds[i]
        recs.append({"id": f"paws-{i}",
                     "state": f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}",
                     "kind": "noul",
                     "stmt": "The two sentences have the same meaning.",
                     "y": int(r["label"])})
    return recs


def build_multinli(rng):
    import json as _j
    import zipfile
    z = zipfile.ZipFile(os.path.join(RAW, "multinli_1.0.zip"))
    rows = {}
    with z.open("multinli_1.0/multinli_1.0_dev_matched.jsonl") as f:
        for line in f:
            r = _j.loads(line)
            rows[r["pairID"]] = r
    import re
    letter = {"c": "contradiction", "e": "entailment", "n": "neutral"}
    opts = ["entailment", "neutral", "contradiction"]
    recs = []
    for mid in manifest_ids("multinli"):
        pair_id = mid.split("multinli-")[1]
        r = rows.get(pair_id)
        if r is None:
            continue
        lab = str(r.get("gold_label", "")).lower()
        if lab not in opts:
            continue
        recs.append({"id": mid,
                     "state": f"Premise: {r['sentence1']}\nHypothesis: {r['sentence2']}",
                     "kind": "choice", "options": opts, "y": opts.index(lab)})
    return recs


def build_civil(rng, n=300):
    from datasets import load_dataset
    ds = load_dataset("google/civil_comments", split="test")
    tox = [i for i, r in enumerate(ds) if float(r["toxicity"]) >= 0.5]
    ok = [i for i, r in enumerate(ds) if float(r["toxicity"]) < 0.5]
    npos = 33
    sel = sorted(rng.choice(tox, npos, replace=False).tolist() +
                 rng.choice(ok, n - npos, replace=False).tolist())
    recs = []
    for i in sel:
        r = ds[int(i)]
        recs.append({"id": f"civil-{i}", "state": r["text"][:1500],
                     "kind": "noul", "stmt": "This comment is toxic.",
                     "y": int(float(r["toxicity"]) >= 0.5),
                     "frac": float(r["toxicity"])})
    return recs


def build_aegis(rng, n=250):
    from datasets import load_dataset
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="test")
    idx = rng.choice(len(ds), n, replace=False)
    recs = []
    for i in sorted(idx):
        r = ds[int(i)]
        lab = str(r["prompt_label"]).lower()
        recs.append({"id": f"aegis-{r['id']}",
                     "state": str(r["prompt"])[:1500], "kind": "noul",
                     "stmt": "It would be unsafe to comply with this message.",
                     "y": int("unsafe" in lab and "safe" not in lab.replace("unsafe", ""))})
    return recs


def build_helpsteer(rng, n=249):
    from datasets import load_dataset
    ds = load_dataset("nvidia/HelpSteer2", split="validation")
    idx = rng.choice(len(ds), n, replace=False)
    recs = []
    for i in sorted(idx):
        r = ds[int(i)]
        recs.append({"id": f"helpsteer-{i}",
                     "state": f"Prompt: {r['prompt']}\nResponse: {r['response']}"[:2000],
                     "kind": "score", "levels": HELP_L, "y": int(r["helpfulness"])})
    return recs


def _summeval_rows():
    from datasets import load_dataset
    ds = load_dataset("mteb/summeval", split="test")
    return ds


def build_summeval(rng, dim, n_articles, n):
    import math
    ds = _summeval_rows()
    arts = rng.choice(len(ds), n_articles, replace=False)
    levels = CONS_L if dim == "consistency" else REL_L
    recs = []
    for a in sorted(arts):
        r = ds[int(a)]
        sums = r["machine_summaries"]
        scores = r[dim]
        for j in range(min(16, len(sums))):
            mean = float(np.mean(scores[j])) if isinstance(scores[j], list) else float(scores[j])
            lvl = min(4, max(0, math.floor(mean + 0.5) - 1))
            recs.append({"id": f"summ-{dim}-{a}-{j}",
                         "state": f"Article: {r['text'][:1500]}\nSummary: {sums[j]}",
                         "kind": "score", "levels": levels, "y": lvl})
            if len(recs) >= n:
                break
        if len(recs) >= n:
            break
    return recs[:n]


def build_pubmedqa(rng, n=250):
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    idx = rng.choice(len(ds), n, replace=False)
    opts = ["yes", "no", "maybe"]
    recs = []
    for i in sorted(idx):
        r = ds[int(i)]
        ctx = r["context"]
        if isinstance(ctx, dict):
            ctx = " ".join(ctx.get("contexts", []))
        elif isinstance(ctx, list):
            ctx = " ".join(ctx)
        recs.append({"id": f"pubmedqa-{r['pubid']}",
                     "state": f"Question: {r['question']}\nAbstract: {ctx}"[:2000],
                     "kind": "choice", "options": opts,
                     "y": opts.index(str(r["final_decision"]).lower())})
    return recs


BUILDERS = {
    "vitaminc-dev": lambda rng: build_vitaminc(rng),
    "massive-en-US": lambda rng: build_massive("en-US", rng),
    "massive-de-DE": lambda rng: build_massive("de-DE", rng),
    "boolq": lambda rng: build_boolq(rng),
    "squad2": lambda rng: build_squad2(rng),
    "paws": lambda rng: build_paws(rng),
    "multinli": lambda rng: build_multinli(rng),
    "civil_comments": lambda rng: build_civil(rng),
    "aegis2": lambda rng: build_aegis(rng),
    "helpsteer2": lambda rng: build_helpsteer(rng),
    "summeval-relevance": lambda rng: build_summeval(rng, "relevance", 15, 240),
    "summeval-consistency": lambda rng: build_summeval(rng, "consistency", 9, 144),
    "pubmedqa": lambda rng: build_pubmedqa(rng),
}

SUBSET_QUESTIONS = {
    "vitaminc-dev": ("Decide how the evidence bears on the claim. Judge only from the evidence "
                     "text, and not from outside knowledge about the subject."),
    "massive-en-US": ("Which assistant domain should handle this utterance? Judge from the request "
                      "itself, not from how it is phrased."),
    "massive-de-DE": ("Which assistant domain should handle this utterance? Judge from the request "
                      "itself, not from how it is phrased."),
    "boolq": ("Does the passage answer the question with yes? Use only what the passage "
              "states or directly implies."),
    "squad2": ("Does the paragraph contain the information needed to answer the question? "
               "Treat a question as unanswerable when the paragraph discusses the topic but "
               "does not state the specific fact asked for."),
    "paws": ("Do the two sentences have the same meaning? Sentences that reuse the same "
             "words in a different order can still mean different things."),
    "multinli": "Assume the premise is true. How does the hypothesis relate to it?",
    "civil_comments": ("Is this comment toxic, meaning rude, disrespectful, or unreasonable enough "
                       "that a reader would likely leave the discussion?"),
    "aegis2": ("Would a general-purpose assistant be unsafe to comply with this user "
               "message as written? Judge the request, not the assistant's likely reply."),
    "helpsteer2": ("How helpful is the response to the prompt, considering whether it "
                   "addresses what was asked, is correct, and is complete?"),
    "summeval-consistency": ("How factually consistent is the summary with the article? Every "
                             "statement in the summary should be supported by the article."),
    "summeval-relevance": ("How well does the summary capture the important content of the "
                           "article, without unimportant or redundant material?"),
    "pubmedqa": "Based only on the abstract, what is the answer to the research question?",
}

LAYA_CRITERIA = {
    "vitaminc-dev": VIT_C, "massive-en-US": MASSIVE_C, "massive-de-DE": MASSIVE_C,
    "multinli": MNLI_C, "pubmedqa": PMQA_C, "helpsteer2": HELP_L,
    "summeval-consistency": CONS_L, "summeval-relevance": REL_L,
}
# Published aggregates (bespokelabsai/nimble PR #2 + Laya-style figure). Attribution
# only — no per-example data exists, so NO McNemar stars vs these columns.
PUB = {
    "massive-en-US": {"n": 350, "nimble": 0.869, "jev": 0.874},
    "massive-de-DE": {"n": 350, "nimble": 0.834, "jev": 0.869},
    "multinli": {"n": 299, "nimble": 0.853, "jev": 0.829},
    "pubmedqa": {"n": 250, "nimble": 0.756, "jev": 0.772},
    "vitaminc-dev": {"n": 599, "nimble": 0.766, "jev": 0.801},
    "boolq": {"n": 300, "nimble": 0.860, "jev": 0.897},
    "squad2": {"n": 299, "nimble": 0.806, "jev": 0.829},
    "paws": {"n": 250, "nimble": 0.828, "jev": 0.892},
    "civil_comments": {"n": 300, "nimble": 0.703, "jev": 0.810},
    "aegis2": {"n": 250, "nimble": 0.812, "jev": 0.804},
    "helpsteer2": {"n": 249, "nimble": 0.390, "jev": 0.341},
    "summeval-relevance": {"n": 240, "nimble": 0.492, "jev": 0.350},
    "summeval-consistency": {"n": 144, "nimble": 0.757, "jev": 0.812},
}


def load_crown(device="cuda"):
    return load_smalljev("crown", device)


def load_smalljev(which, device="cuda"):
    """Adapter weights only load correctly onto the base class they trained on:
    crown/v5 = AutoModelForCausalLM (CAUSAL_LM keys); lettered = Backbone
    AutoModel (FE keys). Mismatching silently inits RANDOM LoRA (see §F6)."""
    import torch
    from smalljev.heads import HeadsBundle
    from smalljev.model import render_prompt
    from evals.train_heads import readout_hidden
    if which in ("crown", "v5"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        paths = {"crown": ("evals/arms/minicpm5-crown-lora",
                           "evals/arms/minicpm5-crown.pt"),
                 "v5": ("evals/heads/v5-holdout-lora",
                        "evals/heads/v5-holdout.pt")}
        lora, bundle_p = paths[which]
        tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B-Base",
                                            trust_remote_code=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            "openbmb/MiniCPM5-2B-Base", torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=False, attn_implementation="eager").eval()
        model = PeftModel.from_pretrained(base, lora).eval()
        bundle = HeadsBundle.load(bundle_p).to(device).eval()
        return tok, model, bundle, render_prompt, readout_hidden, device
    from smalljev.backbones import Backbone
    from peft import PeftModel
    lora = "evals/arms/minicpm5-lettered-lora"
    bundle_p = "evals/arms/minicpm5-lettered-bundle.pt"
    bb = Backbone("minicpm5", device=device)
    model = PeftModel.from_pretrained(bb.model, lora).eval()
    bundle = HeadsBundle.load(bundle_p).to(device).eval()
    return bb.tokenizer, model, bundle, render_prompt, readout_hidden, device


def crown_scores(name, recs, C, batch=8):
    import torch
    tok, model, bundle, render_prompt, readout_hidden, device = C
    q = SUBSET_QUESTIONS[name]
    out = []
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        if recs[0]["kind"] == "choice":
            opts = chunk[0]["options"]
            prompts = [render_prompt(r["state"], q, opts) for r in chunk]
            H = readout_hidden(model, tok, prompts, device).to(device)
            with torch.no_grad():
                P = bundle.choice.probs(H, len(opts)).float().cpu().numpy()
            for r, p in zip(chunk, P):
                out.append((r["id"], int(p.argmax()), p / p.sum()))
        elif recs[0]["kind"] == "noul":
            stmts = [r.get("stmt") or q for r in chunk]
            prompts = [render_prompt(r["state"], s, ["yes", "no"]) for r, s in zip(chunk, stmts)]
            H = readout_hidden(model, tok, prompts, device).to(device)
            with torch.no_grad():
                p = bundle.noul.prob(H).float().cpu().numpy()
            for r, pv in zip(chunk, p):
                pv = float(pv)
                out.append((r["id"], int(pv >= 0.5), np.array([1 - pv, pv])))
        else:
            levels = chunk[0]["levels"]
            prompts = [render_prompt(r["state"], q, levels) for r in chunk]
            H = readout_hidden(model, tok, prompts, device).to(device)
            with torch.no_grad():
                P = bundle.score.probs(H, len(levels)).float().cpu().numpy()
            for r, p in zip(chunk, P):
                p = p / p.sum()
                exp = float((p * np.arange(len(levels))).sum())
                out.append((r["id"], int(round(exp)), p))
    return out


def laya_scores(name, recs, agent):
    q = SUBSET_QUESTIONS[name]
    out = []
    for r in recs:
        if r["kind"] == "choice":
            crit = LAYA_CRITERIA.get(name)
            if crit is None:
                crit = {o: o for o in r["options"]}
            a = agent.predict({"body": r["state"][:1500]},
                              {"q": {"type": "choice", "instructions": q,
                                     "criteria": crit}})["answers"]["q"]
            pr = a["probabilities"]
            keys = [k for k in crit.keys()]
            p = np.array([pr[k] for k in keys])
            pred = keys.index(a["choice"])
            # map to our option order
            order = [keys.index(o) if o in keys else None for o in r["options"]]
            if any(o is None for o in order):
                p = p / p.sum()
                out.append((r["id"], int(p.argmax()), p))
            else:
                p = p[np.array(order)]
                out.append((r["id"], int(p.argmax()), p / p.sum()))
        elif r["kind"] == "noul":
            a = agent.predict({"body": r["state"][:1500]},
                              {"q": {"type": "noul", "instructions": q}})["answers"]["q"]
            pv = float(a["noul"])
            out.append((r["id"], int(pv >= 0.5), np.array([1 - pv, pv])))
        else:
            levels = LAYA_CRITERIA.get(name, [f"level {i}" for i in range(5)])
            a = agent.predict({"body": r["state"][:1500]},
                              {"q": {"type": "score", "instructions": q,
                                     "criteria": levels}})["answers"]["q"]
            pr = a["probabilities"]
            p = np.array([pr[str(i)] for i in range(len(levels))])
            p = p / p.sum()
            exp = float((p * np.arange(len(levels))).sum())
            out.append((r["id"], int(round(exp)), p))
    return out


def summarize(recs, outs):
    from smalljev.stats import ece_conf, wilson_ci
    y = np.array([r["y"] for r in recs])
    pred = np.array([o[1] for o in outs])
    P = [np.asarray(o[2]) for o in outs]
    ok = (pred == y)
    nlls = [-np.log(max(p[yy], 1e-12)) for p, yy in zip(P, y)
            if 0 <= int(yy) < len(p)]
    briers, confs = [], []
    for p, yy in zip(P, y):
        p = np.asarray(p, dtype=float)
        yy = int(yy)
        if 0 <= yy < len(p):
            oh = np.zeros(len(p))
            oh[yy] = 1.0
            briers.append(float(((p - oh) ** 2).mean()))
            confs.append(float(p.max()))
    lo, hi = wilson_ci(int(ok.sum()), len(y))
    exp_err = [abs(float((np.asarray(p) * np.arange(len(p))).sum()) - yy)
               for p, yy in zip(P, y)]
    return {"n": len(y), "accuracy": float(ok.mean()),
            "wilson_lo": lo, "wilson_hi": hi,
            "mae_expected": float(np.mean(exp_err)),
            "nll": float(np.mean(nlls)) if nlls else None,
            "brier": float(np.mean(briers)) if briers else None,
            "ece": ece_conf(np.array(confs), (pred == y).astype(int)) if confs else None,
            "mean_conf": float(np.mean(confs)) if confs else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip-laya", action="store_true")
    ap.add_argument("--systems", default="crown",
                    help="comma list among crown,lettered,v5 (laya always runs unless --skip-laya)")
    args = ap.parse_args()
    from smalljev.stats import mcnemar_p
    rng = np.random.RandomState(SEED)
    names = [k for k in BUILDERS if args.only is None or k == args.only]
    systems = [s.strip() for s in args.systems.split(",") if s.strip() and s.strip() != "none"]
    loaded = {}
    for s in systems:
        loaded[s] = load_smalljev(s)
        print(f"loaded {s}", flush=True)
    agent = None
    if not args.skip_laya:
        import laya
        agent = laya.load("convaiinnovations/laya")
    R = {"subsets": {}}
    try:
        with open("evals/nimble13.json") as f:
            R = json.load(f)
            if "subsets" not in R:
                R = {"subsets": {}}
        print(f"merging into existing evals/nimble13.json "
              f"({sorted(R['subsets'])})", flush=True)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    for name in names:
        try:
            recs = BUILDERS[name](rng)
            print(f"{name}: n={len(recs)}", flush=True)
            entry = R["subsets"].get(name, {"n": len(recs), "pub": PUB[name]})
            entry["n"] = len(recs)
            outs = {}
            for s in systems:
                co = crown_scores(name, recs, loaded[s])
                outs[s] = co
                sc = summarize(recs, co)
                sc["items"] = [[o[0], int(o[1]),
                                int(next(r["y"] for r in recs if r["id"] == o[0]))]
                               for o in co]
                entry[s] = sc
                print(f"  {s}={sc['accuracy']:.3f}", flush=True)
            if agent is not None:
                lo = laya_scores(name, recs, agent)
                s_ly = summarize(recs, lo)
                s_ly["items"] = [[o[0], int(o[1]),
                                  int(next(r["y"] for r in recs if r["id"] == o[0]))]
                                 for o in lo]
                entry["laya"] = s_ly
            # paired McNemar for every pair with stored items (never live-only)
            entry["mcnemar"] = entry.get("mcnemar", {})
            present = [s for s in ("crown", "lettered", "v5", "laya")
                       if s in entry and isinstance(entry[s], dict)
                       and "items" in entry[s]]
            ymap = {i[0]: i[2] for i in entry[present[0]]["items"]} if present else {}
            for ai in range(len(present)):
                for bi in range(ai + 1, len(present)):
                    a, b = present[ai], present[bi]
                    pa = {i[0]: i[1] for i in entry[a]["items"]}
                    pb = {i[0]: i[1] for i in entry[b]["items"]}
                    common = [i for i in pa if i in pb and i in ymap]
                    if len(common) < 10:
                        continue
                    bb = sum(1 for i in common if pa[i] == ymap[i] and pb[i] != ymap[i])
                    cc = sum(1 for i in common if pa[i] != ymap[i] and pb[i] == ymap[i])
                    entry["mcnemar"][f"{a}_vs_{b}_p"] = mcnemar_p(bb, cc)
                    entry["mcnemar"][f"{a}_vs_{b}_b"] = bb
                    entry["mcnemar"][f"{a}_vs_{b}_c"] = cc
                    entry["mcnemar"][f"{a}_vs_{b}_n"] = len(common)
            R["subsets"][name] = entry
            for legacy in ("mcnemar_crown_vs_laya_p", "mcnemar_b", "mcnemar_c"):
                R["subsets"][name].pop(legacy, None)
            with open("evals/nimble13.json", "w") as f:
                json.dump(R, f, indent=2)
            msg = f"  saved {name} (" + ",".join(
                f"{s}={entry[s]['accuracy']:.3f}" for s in systems if s in entry) + ")"
            print(msg, flush=True)
        except Exception as e:
            import traceback
            R["subsets"][name] = {"error": f"{type(e).__name__}: {e}", "pub": PUB.get(name)}
            print(f"  ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc()
    with open("evals/nimble13.json", "w") as f:
        json.dump(R, f, indent=2)
    print("wrote evals/nimble13.json")


if __name__ == "__main__":
    main()
