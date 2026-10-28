# smalljev

**smalljev semantic-v9** — the open TypeSafe Jev that runs on your mama's phone. 2.5B params, one forward pass, zero generated tokens.

[![🤗 HF](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-isHeSatoshi%2Fsmalljev--semantic--v9-yellow)](https://huggingface.co/isHeSatoshi/smalljev-semantic-v9)

---

## install

```bash
# step 1: clone + install the library
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e .

# step 2: download the base model + the smalljev LoRA adapter from HuggingFace
hf download openbmb/MiniCPM5-2B-Base --local-dir ./weights/base
hf download isHeSatoshi/smalljev-semantic-v9 --local-dir ./weights/semantic-v9
```

two downloads: one for the base model (`openbmb/MiniCPM5-2B-Base`, ~5 GB, Apache-2.0), one for the smalljev LoRA adapter + heads (~17 MB, Apache-2.0). both live on HuggingFace Hub.

verified environment: python 3.11, torch 2.8.0, transformers 4.57.6, peft 0.15.2.

---

## what is this, in one paragraph

[TypeSafe built Jev](https://docs.typesafe.ai/). it's a small classifier that looks at your state, reads your list of allowed options, and returns the probability of each. never generates text, never hallucinates a label. really clean idea, really good product.

it's also closed source and runs on TypeSafe's GPUs. fine for production with a budget, less fine if you want the weights on your own machine.

smalljev is the same idea, open weights. the v9 model is a 2.5B backbone (`openbmb/MiniCPM5-2B-Base`, Apache-2.0) fine-tuned on a few thousand decision examples. Apache-2.0 throughout. ships as a python package + a JevBench harness.

## the score

**65.53 / 100 on JevBench v1.2, our own measurement.** the official benchmark has 534 items; we ran on 231 public items and got Intelligence 62.19 / Calibration 63.19 / Speed 79.67 / Cost 58.90.

Benchmark Heaven has since published the system. the maintainer posted our results on 2026-09-23: #40 on v1.3, then #57 at 12.31 on v1.4. it is a live ranked row on the open board now, last measured 2026-10-02: https://benchmarkheaven.com/jev-models

the official score sits below 65.53 because the official runs score all 534 items, held-out slice included. on v1.4 sealed accuracy was 26.9% against 60.6% on the public slice, and both the >25-point gap adjustment and the below-50 Intelligence gate applied.

so 65.53 is our public-slice number and it is not comparable with a board row. for the official score, read the board.

---

## what it actually does

three primitives, mixable in one call:

```python
from smalljev import decide
from smalljev.model import HFBackend

backend = HFBackend(
    model_id="./weights/base",
    adapter_id="./weights/semantic-v9/semantic-v9-lora",
    sem_ckpt="./weights/semantic-v9/semantic-v9-scorer.pt",
)

out = decide(
    "the package arrived broken and the customer wants a refund",
    {"intent":   {"type": "choice", "question": "primary issue?",
                  "choices": ["damaged", "wrong_item", "late", "other"]},
     "escalate": {"type": "noul",   "question": "needs a human right now"},
     "urgency":  {"type": "score",  "question": "how urgent?",
                  "levels": ["low", "medium", "high"]}},
    backend=backend,
)
# out["intent"]   -> {"values": [...], "probabilities": [...]}
# out["escalate"] -> {"probability": 0.78}
# out["urgency"]  -> {"value": ..., "levels": [...], "probabilities": [...]}
```

three things you can ask:

| you give | you get |
|---|---|
| a state and a list of allowed options | which one + the full probability distribution |
| a state and a yes/no question | a probability from 0 to 1 |
| a state and an ordinal rubric | which level + the full distribution |

the model can't pick a label that isn't in your list. the model can't write text. you don't parse json and pray.

## why i built it

i was building a customer-service triage bot. the first version asked GPT-4 to pick one of three options. it worked, mostly. every few hours it would pick "billing" when the right answer was "billing_inquiry" because the prompt wasn't strict enough, and i'd get a bug report at 2am.

then i read about TypeSafe's Jev. the trick is: don't let the model generate the answer. read the probability of each allowed option off the model's next-token distribution, pick the highest one, return the whole distribution. no generation, no parsing, no hallucinations. clean idea.

so i built the open version. 2.5B parameters instead of whatever Jev is. fine-tuned on a few thousand labeled decisions. smalljev fits on the 4060 and on the steam deck.

## how it works (the part that's actually interesting)

the naive version of this idea binds options to readout positions. position 0 = option 0, position 1 = option 1. works for two or three options. at 18-way routing it collapses: 31 of 40 test predictions piled onto position 0 because nothing told the model those positions were semantically different.

smalljev v4 binds options to their text instead. when you ask "which team: support / billing / logistics," the model tokenizes all three options in one forward pass, mean-pools each option's tokens into a vector, runs them through a shared head, gets a softmax. the model doesn't know which option is "position 0" because there are no positions, just text.

```
state:     the package arrived broken.
question:  which team?
options:   support | billing | logistics
                                ↓
            one forward, mean-pool each option's tokens
                                ↓
            shared OptionScorerHead → softmax
                → P(support), P(billing), P(logistics)
```

the heads (option scorer, yes/no, ordinal) are ~200 lines in `smalljev/heads.py`. the span binding is in `smalljev/semantic.py`. full library is ~700 lines.

## things worth knowing

the official benchmark has 534 items. 303 are held out and not in the public repo. the headline score here is on 231 public items. Benchmark Heaven has since rerun it on their own infrastructure and the number moved, for the reasons above; that's fine, that's how benchmarks work.

MASSIVE en/de in nimble13 is disjoint-split same-source, not zero-shot. the +0.18 over zero-shot Laya is the architectural contribution. the +0.41 raw delta is the train/test overlap by design.

cost is estimated at $0.04/m-input × measured tokens, same basis the leaderboard uses for self-hosted rows. self-hosting doesn't have a tariff; we don't invent one.

full per-tier write-up lives outside this repo (the public repo only ships what's needed to run smalljev itself; per-tier eval + score reproducibility are private).

**license** apache-2.0. backbone is apache-2.0. nothing proprietary in here.

**cite**

```bibtex
@software{smalljev2026,
  title  = {smalljev: the open TypeSafe Jev that runs on your mama's phone},
  year   = {2026},
  url    = {https://github.com/isHeSatoshi/smalljev}
}
```

**thanks** TypeSafe for the Jev idea and the public JevBench spec. OpenBMB for the MiniCPM5 backbone. the RTX 4060 Ti that ran this for two days straight while i slept.

**what's in the repo**

```
smalljev/                 # the library (~700 LOC)
tests/                    # unit tests, ~30 s on cpu
```

**benchmark** the JevBench submission lives at https://github.com/fstandhartinger/jevbench/issues/11, with the maintainer's published results in that thread. weights live on HuggingFace: [`isHeSatoshi/smalljev-semantic-v9`](https://huggingface.co/isHeSatoshi/smalljev-semantic-v9).
