"""Smoke: MASSIVE-18 pred distribution (slot-0 check) + AGNews accuracy."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

from evals.eval_semantic import load_system, semantic_choice_probs
from evals.nimble13 import build_massive, SUBSET_QUESTIONS
from evals.train_heads import readout_hidden
from smalljev.model import render_prompt
import torch

rng = np.random.RandomState(11)
C = load_system("evals/arms/semantic-v1")
tok, model, scorer, noul, score_head, device = C
print("loaded", flush=True)

recs = build_massive("en-US", rng)[:40]
q = SUBSET_QUESTIONS["massive-en-US"]
triplets = [(r["state"], q, r["options"]) for r in recs]
P = semantic_choice_probs(tok, model, scorer, triplets, device)
pred = np.array([p.argmax() for p in P])
print("massive pred dist:", Counter(int(x) for x in pred), flush=True)
print("massive true dist:", Counter(r["y"] for r in recs), flush=True)
print("massive acc40:", float((pred == np.array([r["y"] for r in recs])).mean()),
      flush=True)

from datasets import load_dataset
ag = load_dataset("ag_news", split="test[:100]")
from evals.train_heads_v2 import AG_L, AG_QS
triplets = [(r["text"][:800], AG_QS[0], AG_L) for r in ag]
P = semantic_choice_probs(tok, model, scorer, triplets, device)
pred = np.array([p.argmax() for p in P])
y = np.array([int(r["label"]) for r in ag])
print("agnews acc100:", float((pred == y).mean()), flush=True)
print("SMOKE DONE", flush=True)
