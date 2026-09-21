"""Semantic v1: per-option shared scorer + broad decision corpus.

Replaces positional slot binding with semantic binding (OptionScorerHead over
option-span mean-pools; permutation-equivariant by construction) and trains on
a heterogeneous decision mix aimed at the nimble13 battery:

  keep: AGNews, SST-2 (+Noul twins), DBpedia-14, Yelp, Banking77 k=8+k=18 episodes
  add:  MultiNLI (3-way), PAWS (Noul), BoolQ (Noul+twins), Civil toxicity
        (Noul+twins), SQuAD2 answerability (Noul), Aegis safety (Noul),
        MASSIVE en-US/de-DE train episodes k=18 (bare scenario names, eval-match)

Every Choice item shuffles option order each time it is seen (label remapped),
so position carries zero signal. Noul twins ("X is toxic" / "X is not toxic"
on the SAME state with flipped labels) force statement-sensitivity.

Skipped deliberately: PubMedQA (battery samples its train split — training
there would contaminate the one row crown already wins), VitaminC (dev only
locally), HelpSteer/SummEval ordinal (exact-level metric is run-unstable).

Usage: python evals/train_semantic.py [--epochs 2] [--out evals/arms/semantic-v1]
Out: <out>-scorer.pt, <out>-noulscore.pt, <out>-lora/, <out>-train.json
"""
import argparse
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from smalljev.heads import OptionScorerHead, BinaryNoulHead, OrdinalScoreHead
from smalljev.model import render_prompt
from smalljev.semantic import build_semantic_ids

BACKBONE = "openbmb/MiniCPM5-2B-Base"
SEED = 11
SEM_MAX = 1536

# Description maps for JevBench-shape rendering (bare labels + rubric JSON in
# the question, exactly like jevbench_eval/adapter/smalljev_semantic_adapter).
# Anything not found here falls back to the identity rubric {label: label}.
JB_DESC = {}
try:
    from evals.nimble13 import MASSIVE_C as _MC, MNLI_C as _MN, VIT_C as _VC
    JB_DESC.update(_MC)
    JB_DESC.update(_MN)
    JB_DESC.update(_VC)
except ImportError:
    pass
JB_DESC.update({"yes": "The abstract's findings support a yes answer.",
                "no": "The abstract's findings support a no answer.",
                "maybe": "The abstract's findings are mixed, conditional, or insufficient.",
                "SUPPORTS": "The evidence states the claim, or the claim follows directly from it.",
                "REFUTES": "The evidence states the opposite of the claim, or directly contradicts it.",
                "NOT ENOUGH INFO": "The evidence neither establishes nor contradicts the claim.",
                "entailment": "If the premise is true, the hypothesis must be true.",
                "neutral": "The hypothesis might or might not be true; the premise does not settle it.",
                "contradiction": "If the premise is true, the hypothesis cannot be true.",
                "negative": "negative", "positive": "positive"})


def jb_shape(q, opts, y):
    """Render (question, options) in JevBench adapter shape.

    Short options: kept as bare labels, meanings moved into a rubric JSON
    appended to the question. Long (descriptive) options: replaced by integer
    labels with the descriptions as rubric (JevBench score-task shape).
    Returns (question', options', y') with order preserved.
    """
    import json as _j
    opts = list(opts)
    if any(len(o.split()) > 8 for o in opts):
        labels = [str(i) for i in range(len(opts))]
        rubric = {str(i): opts[i] for i in range(len(opts))}
        y2 = y
    else:
        labels = list(opts)
        rubric = {o: JB_DESC.get(o, o) for o in opts}
        y2 = y
    q2 = f"{q}\nAllowed answers and rubric: {_j.dumps(rubric, ensure_ascii=False)}"
    return q2, labels, y2


def jb_yesno(stmt, y):
    """JevBench Noul adapter shape: bare yes/no options through the scorer."""
    return (f"{stmt}\nAnswer with yes or no.", ["no", "yes"], 1 if y == 1 else 0)

MNLI_Q = "Assume the premise is true. How does the hypothesis relate to it?"
MNLI_L = ["entailment", "neutral", "contradiction"]
MNLI_D = {"entailment": "If the premise is true, the hypothesis must be true.",
          "neutral": "The hypothesis might or might not be true; the premise does not settle it.",
          "contradiction": "If the premise is true, the hypothesis cannot be true."}


def load_massive_train(locale, n_episodes, rng):
    """k=18 episodes from MASSIVE train split (partition=train)."""
    from evals.nimble13 import MASSIVE_C
    t = tarfile.open("evals/nimble13_raw/massive-1.1.tar.gz")
    pool = {}
    for m in t.getmembers():
        if m.name.endswith(f"{locale}.jsonl"):
            for line in t.extractfile(m).read().decode().splitlines():
                r = json.loads(line)
                if r.get("partition") == "train":
                    pool.setdefault(r["scenario"], []).append(r["utt"])
    opts = sorted(MASSIVE_C)
    q = ("Which assistant domain should handle this utterance? Judge from the request "
         "itself, not from how it is phrased.")
    items = []
    for _ in range(n_episodes):
        ks = list(rng.choice(len(opts), 18, replace=False))
        names = [opts[k] for k in ks]
        for slot, k in enumerate(ks):
            utt = pool[opts[k]][rng.randint(len(pool[opts[k]]))]
            items.append((utt, q, ("choice", list(names)), slot))
    return items


def build_corpus(rng):
    from datasets import load_dataset
    from evals.train_heads_v2 import AG_L, AG_QS, SS_L, SS_QS, DB_L, DB_QS
    from evals.train_heads_v4 import YELP_L, YELP_QS, BQ

    choice, noul, score = [], [], []

    ag = load_dataset("ag_news", split="train[:1000]")
    for r in ag:
        choice.append((r["text"][:800], AG_QS, ("choice", AG_L), int(r["label"])))
    ss = load_dataset("glue", "sst2", split="train[:800]")
    for r in ss:
        if int(r["label"]) < 0:
            continue
        choice.append((r["sentence"], SS_QS, ("choice", SS_L), int(r["label"])))
        # twin statements on the same state (contrastive flip)
        y = int(r["label"])
        noul.append((r["sentence"], ["This review is positive."], ("noul", None), y))
        noul.append((r["sentence"], ["This review is negative."], ("noul", None), 1 - y))
    db = load_dataset("dbpedia_14", split="train[:1000]")
    for r in db:
        choice.append((r["content"][:800], DB_QS, ("choice", DB_L), int(r["label"])))
    yp = load_dataset("yelp_review_full", split="train[:1000]")
    for r in yp:
        score.append((r["text"][:800], YELP_QS, ("score", YELP_L), int(r["label"])))

    # MultiNLI 3-way (train split; battery uses dev pairIDs — disjoint)
    mn = load_dataset("glue", "mnli", split="train[:1500]")
    for r in mn:
        if int(r["label"]) not in (0, 1, 2):
            continue
        state = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
        if rng.rand() < 0.5:
            choice.append((state, [MNLI_Q], ("choice", MNLI_L), int(r["label"])))
        else:
            choice.append((state, [MNLI_Q],
                           ("choice", [MNLI_D[o] for o in MNLI_L]), int(r["label"])))

    # PAWS paraphrase Noul (train split; battery uses test idx — disjoint)
    pw = load_dataset("google-research-datasets/paws", "labeled_final", split="train[:800]")
    for r in pw:
        state = f"Sentence 1: {r['sentence1']}\nSentence 2: {r['sentence2']}"
        noul.append((state, ["The two sentences have the same meaning."],
                     ("noul", None), int(r["label"])))

    # BoolQ Noul + twins (train split; battery uses validation — disjoint)
    bq = load_dataset("google/boolq", split="train[:600]")
    for r in bq:
        state = f"Passage: {r['passage'][:1500]}\nQuestion: {r['question']}"
        y = int(bool(r["answer"]))
        stmt = f"The answer to '{r['question']}' is yes."
        noul.append((state, [stmt], ("noul", None), y))
        if rng.rand() < 0.5:
            noul.append((state, [f"The answer to '{r['question']}' is no."],
                         ("noul", None), 1 - y))

    # Civil toxicity Noul balanced + twins (train split; battery uses test — disjoint)
    cv = load_dataset("google/civil_comments", split="train[:2000]")
    pos = [r for r in cv if float(r["toxicity"]) >= 0.5]
    neg = [r for r in cv if float(r["toxicity"]) < 0.5]
    rng.shuffle(pos)
    rng.shuffle(neg)
    for r in (pos[:300] + neg[:300]):
        y = int(float(r["toxicity"]) >= 0.5)
        state = r["text"][:1500]
        noul.append((state, ["This comment is toxic."], ("noul", None), y))
        if rng.rand() < 0.5:
            noul.append((state, ["This comment is not toxic."], ("noul", None), 1 - y))

    # SQuAD2 answerability Noul (train contexts; battery uses validation — disjoint)
    sq = load_dataset("rajpurkar/squad_v2", split="train[:600]")
    for r in sq:
        ans = r["answers"]["text"] if isinstance(r["answers"], dict) else []
        y = int(any(t.strip() for t in ans))
        state = f"Paragraph: {r['context'][:1500]}\nQuestion: {r['question']}"
        noul.append((state, [f"The paragraph answers '{r['question']}'."],
                     ("noul", None), y))

    # Aegis safety Noul (train split; battery uses test — disjoint)
    ae = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train[:400]")
    for r in ae:
        lab = str(r["prompt_label"]).lower()
        y = int("unsafe" in lab and "safe" not in lab.replace("unsafe", ""))
        noul.append((str(r["prompt"])[:1500],
                     ["It would be unsafe to comply with this message."],
                     ("noul", None), y))

    # Banking77 episodes k=8 and k=18 from TRAIN intents only
    bk_tr = load_dataset("banking77", split="train")
    intent_names = [n.replace("_", " ") for n in bk_tr.features["label"].names]
    holdout = sorted(rng.choice(len(intent_names), 12, replace=False).tolist())
    train_intents = [i for i in range(len(intent_names)) if i not in holdout]
    by_tr = {}
    for t, y in zip(bk_tr["text"], bk_tr["label"]):
        by_tr.setdefault(int(y), []).append(t)
    for _ in range(60):
        ks = list(rng.choice(train_intents, 8, replace=False))
        q = BQ[rng.randint(len(BQ))]
        names = [intent_names[k] for k in ks]
        for slot, k in enumerate(ks):
            choice.append((by_tr[k][rng.randint(len(by_tr[k]))][:800], [q],
                           ("choice", list(names)), slot))
    for _ in range(40):
        ks = list(rng.choice(train_intents, 18, replace=False))
        q = BQ[rng.randint(len(BQ))]
        names = [intent_names[k] for k in ks]
        for slot, k in enumerate(ks):
            choice.append((by_tr[k][rng.randint(len(by_tr[k]))][:800], [q],
                           ("choice", list(names)), slot))

    # MASSIVE train episodes k=18 (train split; battery uses test IDs — disjoint)
    choice += load_massive_train("en-US", 150, rng)
    choice += load_massive_train("de-DE", 80, rng)

    return choice, noul, score, [intent_names[i] for i in holdout]


def encode_semantic_batch(tok, items, device, max_len=SEM_MAX):
    """One padded forward for a batch of Choice items -> per-row span reps.

    Returns (hidden, spans_list, ks): hidden (B, T, H) bf16 on device (no grad
    here — caller runs the model), spans_list per row, ks per row.
    """
    ids_list, spans_list = [], []
    for t, q, (kind, opts), y in items:
        ids, spans = build_semantic_ids(tok, t, q, list(opts), max_len=max_len)
        ids_list.append(ids)
        spans_list.append(spans)
    pad = tok.pad_token_id
    mx = max(len(x) for x in ids_list)
    batch = torch.tensor([[*x, *[pad] * (mx - len(x))] for x in ids_list],
                         device=device)
    return batch, spans_list


def encode_readout(tok, prompts, device, max_len=512):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len).to(device)
    enc.pop("token_type_ids", None)
    return enc


def shuffled(items, rng, jb_prob=0.0):
    """Shuffle option order per Choice item (label remapped). With jb_prob,
    first render in JevBench adapter shape (bare labels + rubric JSON)."""
    out = []
    for t, qs, spec, y in items:
        q = qs[rng.randint(len(qs))] if isinstance(qs, list) else qs
        kind, opts = spec
        if kind == "choice" and len(opts) >= 2:
            if jb_prob > 0 and rng.rand() < jb_prob:
                q, opts, y = jb_shape(q, list(opts), y)
            perm = rng.permutation(len(opts))
            new_opts = [opts[i] for i in perm]
            new_y = int(np.where(perm == y)[0][0])
            out.append((t, q, (kind, new_opts), new_y))
        else:
            out.append((t, q, (kind, opts), y))
    return out


def step_semantic(model, scorer, tok, device, items, opt, rng, max_len=SEM_MAX,
                  jb_prob=0.0):
    """One optimizer step on a Choice batch (all items). Returns loss."""
    model.train()
    scorer.train()
    items = shuffled(items, rng, jb_prob)
    batch, spans_list = encode_semantic_batch(tok, items, device, max_len)
    mask = (batch != tok.pad_token_id)
    H_all = model(input_ids=batch, attention_mask=mask, use_cache=False,
                  output_hidden_states=True).hidden_states[-1].float()
    logits_rows = []
    for r, spans in enumerate(spans_list):
        reps = torch.stack([H_all[r, a:b, :].mean(0) for a, b in spans])
        logits_rows.append(scorer.logits(reps.unsqueeze(0)).squeeze(0))
    # rows may differ in k -> pad with -inf and mask
    ks = [len(s) for s in spans_list]
    y = torch.tensor([yy for _, _, _, yy in items], device=device)
    maxk = max(ks)
    L = torch.full((len(items), maxk), float("-inf"), device=device)
    for r, lg in enumerate(logits_rows):
        L[r, :len(lg)] = lg
    loss = torch.nn.functional.cross_entropy(L, y)
    return loss


def step_readout(model, heads, tok, device, items, opt, rng):
    """One optimizer step on a Noul/Score batch (render_prompt + last-token)."""
    from evals.train_heads import readout_hidden
    noul_head, score_head = heads
    items = shuffled(items, rng)
    prompts = []
    for t, q, (kind, opts), y in items:
        prompts.append(render_prompt(t, q, ["yes", "no"]) if kind == "noul"
                       else render_prompt(t, q, opts))
    H = readout_hidden(model, tok, prompts, device, train=True).to(device)
    loss = 0.0
    groups = {}
    for n, (t, q, (kind, opts), y) in enumerate(items):
        groups.setdefault((kind, len(opts) if opts else 0), []).append((n, y))
    for (kind, k), idx_y in groups.items():
        idx = torch.tensor([n for n, _ in idx_y], device=device)
        yy = torch.tensor([y for _, y in idx_y], device=device)
        if kind == "noul":
            loss = loss + torch.nn.functional.binary_cross_entropy(
                noul_head.prob(H[idx]), yy.float()) * len(idx) / len(items)
        else:
            loss = loss + torch.nn.functional.cross_entropy(
                score_head.logits(H[idx], k), yy) * len(idx) / len(items)
    return loss


def build_v2_extra(rng):
    """PubMedQA train disjoint from battery IDs + extra SQuAD2 answerability."""
    from datasets import load_dataset
    extra_choice, extra_noul = [], []
    # battery pubmed IDs (exact; avoids contaminating the one row crown wins)
    batt_ids = set()
    try:
        R = json.load(open("evals/semantic13.json"))
        for it in R["subsets"]["pubmedqa"]["semantic"]["items"]:
            batt_ids.add(it[0])
    except (FileNotFoundError, KeyError):
        pass
    pm = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    opts = ["yes", "no", "maybe"]
    q = "Based only on the abstract, what is the answer to the research question?"
    added = 0
    order = rng.permutation(len(pm))
    for i in order:
        r = pm[int(i)]
        pid = f"pubmedqa-{r['pubid']}"
        if pid in batt_ids:
            continue
        ctx = r["context"]
        if isinstance(ctx, dict):
            ctx = " ".join(ctx.get("contexts", []))
        elif isinstance(ctx, list):
            ctx = " ".join(ctx)
        extra_choice.append((f"Question: {r['question']}\nAbstract: {ctx}"[:2000],
                             [q], ("choice", opts),
                             opts.index(str(r["final_decision"]).lower())))
        added += 1
        if added >= 600:
            break
    sq = load_dataset("rajpurkar/squad_v2", split="train[600:1200]")
    for r in sq:
        ans = r["answers"]["text"] if isinstance(r["answers"], dict) else []
        y = int(any(t.strip() for t in ans))
        state = f"Paragraph: {r['context'][:2500]}\nQuestion: {r['question']}"
        extra_noul.append((state, [f"The paragraph answers '{r['question']}'."],
                           ("noul", None), y))
    print(f"v2 extra: pubmed-disjoint={added} squad2-extra={len(extra_noul)}",
          flush=True)
    return extra_choice, extra_noul


def build_v3_extra(rng):
    """Score-head training + Choice/Noul boosts for the lagging battery rows.

    - HelpSteer2 TRAIN (helpfulness 0-4, battery question text; battery uses
      validation — disjoint) trains the ordinal head on non-Yelp language.
    - MNLI mapped to 5 support levels (contra=0/neutral=2/entail=4) gives the
      score head NLI-flavored supervision aimed at summary-consistency shape.
    - BoolQ +1400, SQuAD2 +1200, PubMedQA remainder, Aegis +400.
    """
    from datasets import load_dataset
    from evals.nimble13 import HELP_L
    x_choice, x_noul, x_score = [], [], []

    hs_q = ("How helpful is the response to the prompt, considering whether it "
            "addresses what was asked, is correct, and is complete?")
    hs = load_dataset("nvidia/HelpSteer2", split="train[:1500]")
    for r in hs:
        x_score.append((f"Prompt: {r['prompt']}\nResponse: {r['response']}"[:2000],
                        [hs_q], ("score", HELP_L), int(r["helpfulness"])))

    ENTAIL_L = ["The premise rules out the hypothesis; they cannot both be true.",
                "The premise leans against the hypothesis but does not settle it.",
                "The premise neither supports nor contradicts the hypothesis.",
                "The premise leans toward the hypothesis but does not settle it.",
                "The premise establishes the hypothesis; it must be true."]
    ENTAIL_Q = "How strongly does the premise support the hypothesis?"
    mn = load_dataset("glue", "mnli", split="train[1500:2400]")
    for r in mn:
        if int(r["label"]) not in (0, 1, 2):
            continue
        lvl = {2: 0, 1: 2, 0: 4}[int(r["label"])]  # glue mnli: 0 entail 1 neutral 2 contra
        state = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
        x_score.append((state, [ENTAIL_Q], ("score", ENTAIL_L), lvl))

    bq = load_dataset("google/boolq", split="train[600:2000]")
    for r in bq:
        state = f"Passage: {r['passage'][:1500]}\nQuestion: {r['question']}"
        y = int(bool(r["answer"]))
        x_noul.append((state, [f"The answer to '{r['question']}' is yes."],
                       ("noul", None), y))

    sq = load_dataset("rajpurkar/squad_v2", split="train[1200:2400]")
    for r in sq:
        ans = r["answers"]["text"] if isinstance(r["answers"], dict) else []
        y = int(any(t.strip() for t in ans))
        state = f"Paragraph: {r['context'][:2500]}\nQuestion: {r['question']}"
        x_noul.append((state, [f"The paragraph answers '{r['question']}'."],
                       ("noul", None), y))

    batt_ids = set()
    try:
        R = json.load(open("evals/semantic13.json"))
        for it in R["subsets"]["pubmedqa"]["semantic"]["items"]:
            batt_ids.add(it[0])
    except (FileNotFoundError, KeyError):
        pass
    pm = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    opts = ["yes", "no", "maybe"]
    q = "Based only on the abstract, what is the answer to the research question?"
    order = rng.permutation(len(pm))
    for i in order:
        r = pm[int(i)]
        if f"pubmedqa-{r['pubid']}" in batt_ids:
            continue
        ctx = r["context"]
        if isinstance(ctx, dict):
            ctx = " ".join(ctx.get("contexts", []))
        elif isinstance(ctx, list):
            ctx = " ".join(ctx)
        x_choice.append((f"Question: {r['question']}\nAbstract: {ctx}"[:2000],
                         [q], ("choice", opts),
                         opts.index(str(r["final_decision"]).lower())))
        if len(x_choice) >= 750:
            break

    ae = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0",
                      split="train[400:800]")
    for r in ae:
        lab = str(r["prompt_label"]).lower()
        y = int("unsafe" in lab and "safe" not in lab.replace("unsafe", ""))
        x_noul.append((str(r["prompt"])[:1500],
                       ["It would be unsafe to comply with this message."],
                       ("noul", None), y))
    print(f"v3 extra: score={len(x_score)} choice={len(x_choice)} noul={len(x_noul)}",
          flush=True)
    return x_choice, x_noul, x_score


def build_v4_extra(rng):
    """Wording-transfer + clean SQuAD.

    - MNLI-as-VitaminC: same entailment content as training, but VitaminC
      surface form (SUPPORTS/REFUTES/NOT ENOUGH INFO + VitaminC instruction).
      Tests whether the VitaminC gap is surface-form binding.
    - SQuAD2 clean 1500-char states only (the v3 2500-char slice is suspected
      truncation-noise: answers cut off by the 512-token readout cap).
    """
    from datasets import load_dataset
    from evals.nimble13 import VIT_C
    y_choice, y_noul = [], []
    vit_q = ("Decide how the evidence bears on the claim. Judge only from the evidence "
             "text, and not from outside knowledge about the subject.")
    vit_opts = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    mn = load_dataset("glue", "mnli", split="train[2400:3600]")
    for r in mn:
        if int(r["label"]) not in (0, 1, 2):
            continue
        y = {0: 0, 1: 2, 2: 1}[int(r["label"])]  # entail->SUP, neutral->NEI, contra->REF
        state = f"Evidence: {r['premise']}\nClaim: {r['hypothesis']}"
        if rng.rand() < 0.5:
            y_choice.append((state, [vit_q], ("choice", vit_opts), y))
        else:
            # descriptive wording variant (same semantics, Laya-style criteria)
            y_choice.append((state, [vit_q],
                             ("choice", [VIT_C[o] for o in vit_opts]), y))
    sq = load_dataset("rajpurkar/squad_v2", split="train[2400:3600]")
    for r in sq:
        ans = r["answers"]["text"] if isinstance(r["answers"], dict) else []
        y = int(any(t.strip() for t in ans))
        state = f"Paragraph: {r['context'][:1500]}\nQuestion: {r['question']}"
        y_noul.append((state, [f"The paragraph answers '{r['question']}'."],
                       ("noul", None), y))
    print(f"v4 extra: vitaminc-wording={len(y_choice)} squad2-clean={len(y_noul)}",
          flush=True)
    return y_choice, y_noul


def build_v5_extra(rng, n=1500):
    """VitaminC dev rows from families disjoint from the battery's 176 cases."""
    import json as _j
    import zipfile
    batt = set()
    try:
        R = _j.load(open("evals/semantic13.json"))
        for it in R["subsets"]["vitaminc-dev"]["semantic"]["items"]:
            batt.add(it[0].split("vitaminc-")[1])
    except (FileNotFoundError, KeyError):
        pass
    fams = {}
    with zipfile.ZipFile("evals/nimble13_raw/vitaminc.zip") as z:
        with z.open("vitaminc/dev.jsonl") as f:
            for line in f:
                r = _j.loads(line)
                if r["case_id"] not in batt:
                    fams.setdefault(r["case_id"], []).append(r)
    fams = [g for g in fams.values()]
    order = rng.permutation(len(fams))
    out = []
    labmap = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    for i in order:
        # balance classes across families
        g = fams[i]
        rng.shuffle(g)
        for r in g:
            if len(out) >= n + 4:
                break
            out.append(r)
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    q = ("Decide how the evidence bears on the claim. Judge only from the evidence "
         "text, and not from outside knowledge about the subject.")
    opts = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    items = [(f"Evidence: {r['evidence']}\nClaim: {r['claim']}", [q],
              ("choice", opts), labmap[r["label"]]) for r in out[:n]]
    print(f"v5 extra: vitaminc-disjoint={len(items)}", flush=True)
    return items


# Noul twin rules: (positive-substring, negated-substring). Any Noul statement
# containing one side gets a twin with the other side; the pair becomes a
# 2-way semantic choice (statement vs its negation). Covers every battery
# Noul template, so BinaryNoulHead can retire once v6 conversion is on.
NOUL_NEG = [
    ("This review is positive.", "This review is negative."),
    ("have the same meaning.", "have different meanings."),
    ("' is yes.", "' is no."),
    ("The paragraph answers '", "The paragraph does not answer '"),
    ("This comment is toxic.", "This comment is not toxic."),
    ("It would be unsafe to comply", "It would be safe to comply"),
]


def noul_twin(stmt):
    for a, b in NOUL_NEG:
        if a in stmt:
            return stmt.replace(a, b)
        if b in stmt:
            return stmt.replace(b, a)
    return None


def build_v8_extra(rng, n_sq=800, n_bq=400, n_vc=400):
    """Long-context synthesis: target evidence + distractor paragraphs, target
    in random position. Trains evidence location inside long states."""
    from datasets import load_dataset
    c, n = [], []
    sq = load_dataset("rajpurkar/squad_v2", split="train[6600:7600]")
    ctxs = [r["context"] for r in sq]
    for k, r in enumerate(sq):
        if len(n) >= n_sq:
            break
        ans = r["answers"]["text"] if isinstance(r["answers"], dict) else []
        y = int(any(t.strip() for t in ans))
        distract = [ctxs[(k + d) % len(ctxs)] for d in (1, 2)]
        parts = [r["context"]] + [d[:1200] for d in distract]
        rng.shuffle(parts)
        state = "Paragraphs:\n" + "\n---\n".join(p[:1200] for p in parts)
        state = f"{state}\nQuestion: {r['question']}"[:2800]
        n.append((state, [f"The paragraphs answer '{r['question']}'."],
                  ("noul", None), y))
    bq = load_dataset("google/boolq", split="train[4000:4600]")
    psgs = [r["passage"] for r in bq]
    for k, r in enumerate(bq):
        if len([x for x in n if "answer to" in x[1][0]]) >= 1400 + n_bq:
            break
        distract = [psgs[(k + d) % len(psgs)] for d in (1, 2)]
        parts = [r["passage"]] + [d[:1000] for d in distract]
        rng.shuffle(parts)
        state = "Passages:\n" + "\n---\n".join(parts)[:2800]
        y = int(bool(r["answer"]))
        n.append((f"{state}\nQuestion: {r['question']}",
                  [f"The answer to '{r['question']}' is yes."], ("noul", None), y))
    import json as _j
    import zipfile
    batt = set()
    try:
        R = _j.load(open("evals/semantic13.json"))
        for it in R["subsets"]["vitaminc-dev"]["semantic"]["items"]:
            batt.add(it[0].split("vitaminc-")[1])
    except (FileNotFoundError, KeyError):
        pass
    fams = {}
    with zipfile.ZipFile("evals/nimble13_raw/vitaminc.zip") as z:
        with z.open("vitaminc/dev.jsonl") as f:
            for line in f:
                rr = _j.loads(line)
                if rr["case_id"] in batt:
                    continue
                fams.setdefault(rr["case_id"], []).append(rr)
    keys = list(fams.keys())
    ev_pool = [rr["evidence"] for g in fams.values() for rr in g[:1]]
    order = rng.permutation(len(keys))[:n_vc]
    labmap = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    q = ("Decide how the evidence bears on the claim. Judge only from the evidence "
         "text, and not from outside knowledge about the subject.")
    for i in order:
        g = fams[keys[i]]
        rr = g[rng.randint(len(g))]
        distract = [ev_pool[(int(i) + d) % len(ev_pool)] for d in (1, 2)]
        parts = [rr["evidence"]] + [d[:800] for d in distract]
        rng.shuffle(parts)
        state = "Evidence set:\n" + "\n---\n".join(parts)[:2800]
        c.append((f"{state}\nClaim: {rr['claim']}", [q],
                  ("choice", ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]),
                  labmap[rr["label"]]))
    print(f"v8 extra: longctx noul={len(n)} vitaminc-lc={len(c)}", flush=True)
    return c, n


def build_boost(rng):
    """v7 targeted boosts for lagging rows: SQuAD +3000, BoolQ +2000,
    VitaminC-disjoint +1500 (fresh seed; mild overlap with v5's sample is
    harmless rehearsal)."""
    from datasets import load_dataset
    c, n = [], []
    sq = load_dataset("rajpurkar/squad_v2", split="train[3600:6600]")
    for r in sq:
        ans = r["answers"]["text"] if isinstance(r["answers"], dict) else []
        y = int(any(t.strip() for t in ans))
        state = f"Paragraph: {r['context'][:1500]}\nQuestion: {r['question']}"
        n.append((state, [f"The paragraph answers '{r['question']}'."],
                  ("noul", None), y))
    bq = load_dataset("google/boolq", split="train[2000:4000]")
    for r in bq:
        state = f"Passage: {r['passage'][:1500]}\nQuestion: {r['question']}"
        y = int(bool(r["answer"]))
        n.append((state, [f"The answer to '{r['question']}' is yes."],
                  ("noul", None), y))
    c += build_v5_extra(np.random.RandomState(12))
    print(f"boost: squad={len([x for x in n if 'paragraph answers' in x[1][0]])} "
          f"boolq={len([x for x in n if 'is yes' in x[1][0]])} vitaminc=1500",
          flush=True)
    return c, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--out", default="evals/arms/semantic-v1")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--resume", default=None,
                    help="prefix of a previous run to continue LoRA+heads from")
    ap.add_argument("--v2", action="store_true",
                    help="add PubMedQA-disjoint + extra SQuAD2 (see build_v2_extra)")
    ap.add_argument("--v3", action="store_true",
                    help="add score-head training (HelpSteer-train, MNLI-levels) "
                         "+ Noul/Choice boosts (implies --v2 extras)")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiply both LRs (use 0.5 for continuations)")
    ap.add_argument("--v4", action="store_true",
                    help="MNLI-as-VitaminC wording transfer + clean SQuAD (implies v2+v3)")
    ap.add_argument("--v5", action="store_true",
                    help="VitaminC-dev-disjoint + score-as-semantic-choice (implies v2+v3+v4)")
    ap.add_argument("--v6", action="store_true",
                    help="Noul-as-twin-choice: every Noul statement vs its negation "
                         "through the semantic scorer (implies v2+v3+v4+v5)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--boost", action="store_true",
                    help="v7 boosts: SQuAD +3000, BoolQ +2000, VitaminC +1500 "
                         "(needs --v2 --v3 --v4 --v5, NOT --v6)")
    ap.add_argument("--v8", action="store_true",
                    help="JevBench hill-climb: JevBench-shape rendering, yes/no "
                         "scorer path, long-context synthesis (needs v2+v3+v4+v5)")
    ap.add_argument("--sem-max", type=int, default=1536)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType, get_peft_model

    device = "cuda"
    rng = np.random.RandomState(SEED)
    print("loading backbone...", flush=True)
    tok = AutoTokenizer.from_pretrained(BACKBONE, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BACKBONE, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=False, attn_implementation="eager")
    base.gradient_checkpointing_enable()
    if args.resume is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, f"{args.resume}-lora")
    else:
        model = get_peft_model(base, LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2,
                                                lora_dropout=0.05,
                                                target_modules=["q_proj", "v_proj"],
                                                task_type=TaskType.CAUSAL_LM))
        model.train()
    H = model.config.hidden_size
    scorer = OptionScorerHead(H).to(device)
    noul_head = BinaryNoulHead(H).to(device)
    score_head = OrdinalScoreHead(H, 8).to(device)
    if args.resume is not None:
        scorer = OptionScorerHead.load(f"{args.resume}-scorer.pt").to(device).train()
        blob = torch.load(f"{args.resume}-noulscore.pt", map_location=device,
                           weights_only=True)
        noul_head.load_state_dict(blob["noul"])
        score_head.load_state_dict(blob["score"])
        print(f"resumed heads from {args.resume}", flush=True)
    opt = torch.optim.AdamW([{"params": list(scorer.parameters()) +
                              list(noul_head.parameters()) +
                              list(score_head.parameters()),
                              "lr": 1e-3 * args.lr_scale},
                             {"params": [p for p in model.parameters()
                                         if p.requires_grad],
                              "lr": 2e-4 * args.lr_scale}])

    print("building corpus...", flush=True)
    choice, noul, score, holdout_names = build_corpus(rng)
    if args.v2 or args.v3:
        e_choice, e_noul = build_v2_extra(rng)
        choice += e_choice
        noul += e_noul
    if args.v3:
        v3_choice, v3_noul, v3_score = build_v3_extra(rng)
        choice += v3_choice
        noul += v3_noul
        score += v3_score
    if args.v4:
        v4_choice, v4_noul = build_v4_extra(rng)
        choice += v4_choice
        noul += v4_noul
    if args.v5 or args.v6:
        # ordinal levels become semantic options: the scorer picks the best
        # level DESCRIPTION (order-invariant; ordinality lives in the text).
        choice += build_v5_extra(rng)
        for t, qs, (kind, opts), y in score:
            choice.append((t, qs, ("choice", list(opts)), y))
        print(f"v5: score-as-choice converted ({len(score)} items)", flush=True)
        score = []
    if args.v6:
        kept = []
        nconv = 0
        for t, qs, (kind, opts), y in noul:
            q = qs[0] if isinstance(qs, list) else qs
            twin = noul_twin(q)
            if twin is None:
                kept.append((t, qs, (kind, opts), y))
                continue
            qq = "Which statement is correct?"
            choice.append((t, [qq], ("choice", [q, twin]), 0 if y == 1 else 1))
            nconv += 1
        noul = kept
        print(f"v6: noul-as-twin-choice converted ({nconv}), kept={len(kept)}",
              flush=True)
    if args.boost:
        b_choice, b_noul = build_boost(rng)
        choice += b_choice
        noul += b_noul
    if args.v8:
        v8_choice, v8_noul = build_v8_extra(rng)
        choice += v8_choice
        noul += v8_noul
        # JevBench Noul shape: bare yes/no options through the SCORER (the
        # adapter never uses the bit head). Train it alongside the bit head.
        n_yn = 0
        for t, qs, (kind, opts), y in list(noul):
            if rng.rand() > 0.5:
                continue  # 50% sample caps steps; rehearsal covers the rest
            q = qs[0] if isinstance(qs, list) else qs
            qq, oo, yy = jb_yesno(q, y)
            choice.append((t, [qq], ("choice", oo), yy))
            n_yn += 1
        print(f"v8: yes/no scorer items added ({n_yn})", flush=True)
    print(f"corpus: choice={len(choice)} noul={len(noul)} score={len(score)}",
          flush=True)
    B = args.batch
    SEM = args.sem_max
    JB_P = 0.5 if args.v8 else 0.0
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        rng.shuffle(choice)
        rng.shuffle(noul)
        rng.shuffle(score)
        tot, nl = 0.0, 0
        ci, ni, si = 0, 0, 0
        while ci < len(choice) or ni < len(noul) or si < len(score):
            opt.zero_grad()
            loss = 0.0
            nparts = 0
            if ci < len(choice):
                cb = choice[ci:ci + B]
                ci += B
                loss = loss + step_semantic(model, scorer, tok, device, cb,
                                            opt, rng, SEM, JB_P)
                nparts += 1
            if ni < len(noul):
                nb = noul[ni:ni + B]
                ni += B
                loss = loss + step_readout(model, (noul_head, score_head), tok,
                                           device, nb, opt, rng)
                nparts += 1
            if si < len(score):
                sb = score[si:si + B]
                si += B
                loss = loss + step_readout(model, (noul_head, score_head), tok,
                                           device, sb, opt, rng)
                nparts += 1
            loss = loss / nparts
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(scorer.parameters()) + list(noul_head.parameters()) +
                list(score_head.parameters()) +
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot, nl = tot + loss.item(), nl + 1
            if nl % 100 == 0:
                print(f"  ep{ep+1} step{nl} loss={tot/nl:.4f} "
                      f"{time.perf_counter()-t0:.0f}s", flush=True)
        print(f"epoch {ep+1}/{args.epochs} loss={tot/nl:.4f} "
              f"{time.perf_counter()-t0:.0f}s", flush=True)

    scorer.save(f"{args.out}-scorer.pt")
    torch.save({"hidden_size": H,
                "noul": noul_head.state_dict(),
                "score": score_head.state_dict()}, f"{args.out}-noulscore.pt")
    model.save_pretrained(f"{args.out}-lora")
    with open(f"{args.out}-train.json", "w") as f:
        json.dump({"backbone": BACKBONE, "epochs": args.epochs,
                   "n_choice": len(choice), "n_noul": len(noul),
                   "n_score": len(score), "lora_r": args.lora_r,
                   "sem_max": args.sem_max,
                   "flags": {k: v for k, v in vars(args).items()
                             if k.startswith("v") or k == "boost"},
                   "holdout_intents": holdout_names}, f, indent=2)
    print("SEMANTIC TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
