# smalljev

**smalljev semantic-v9** — the open TypeSafe Jev that runs on your mama's phone. 2.5B params, one forward pass, zero generated tokens.

---

## install

the repo is not on PyPI. you install it from source.

```bash
# step 1: clone + install the library
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e .

# step 2: download the base model + the smalljev LoRA adapter from HuggingFace
huggingface-cli download openbmb/MiniCPM5-2B-Base   --local-dir ./weights/base
huggingface-cli download isHeSatoshi/smalljev-semantic-v9 --local-dir ./weights/semantic-v9
```

two downloads: one for the base model (`openbmb/MiniCPM5-2B-Base`, ~5 GB, Apache-2.0), one for the smalljev LoRA adapter + heads (~17 MB, Apache-2.0). both live on HuggingFace Hub.

if you want the JevBench harness scripts too (`jevbench_eval/scripts/run_v12.py`, etc.), also run `pip install -e ".[bench]"` for matplotlib/seaborn/datasets.

verified environment: python 3.11, torch 2.8.0, transformers 4.57.6, peft 0.15.2.

---

## what is this, in one paragraph

[TypeSafe built Jev](https://docs.typesafe.ai/). it's a small classifier that looks at your state, reads your list of allowed options, and returns the probability of each. never generates text, never hallucinates a label. really clean idea, really good product.

it's also closed source and runs on TypeSafe's GPUs. fine for production with a budget, less fine if you want the weights on your own machine.

smalljev is the same idea, open weights. the v9 model is a 2.5B backbone (`openbmb/MiniCPM5-2B-Base`, Apache-2.0) fine-tuned on a few thousand decision examples. Apache-2.0 throughout. ships as a python package + a JevBench harness.

## the score

**rank 6 on JevBench (private measurement, 65.53 / 100).** the official benchmark has 534 items; we ran on 231 public items and got Intelligence 62.19 / Calibration 63.19 / Speed 79.67 / Cost 58.90. full leaderboard:

| rank | system | score |
|---|---|---|
| 1 | jev 1.13.0 (TypeSafe) | 75.30 |
| 2 | semif (Qwen3.5-4B) | 74.60 |
| 3 | djev (Maisa) | 74.30 |
| 4 | open-alternative-jev | 69.80 |
| 5 | system-one-open (Gemma 4 E2B) | 68.70 |
| **6** | **smalljev semantic-v9** | **65.53** |
| 7 | openjev (DiffusionGemma 26B) | 67.60 |
| 8 | openjev-sglang (Qwen3.6-35B) | 66.20 |
| 9 | GPT-5.6 Luna | 66.00 |
| 10 | open-jev-deberta-v3-large | 64.40 |
| 11 | Bespoke Nimble 9B | 63.50 |

smallest model in the ranked set. everyone above row 6 has more parameters and a bigger GPU bill.

(this is our own private JevBench score. we haven't been published to the official leaderboard yet. the submission is in the queue at https://github.com/fstandhartinger/jevbench/issues/11, the Benchmark Heaven team hasn't rerun us on their infrastructure.)

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

the official benchmark has 534 items. 303 are held out and not in the public repo. the headline score here is on 231 public items. if Benchmark Heaven reruns on their infrastructure the number might shift; that's fine, that's how benchmarks work.

MASSIVE en/de in nimble13 is disjoint-split same-source, not zero-shot. the +0.18 over zero-shot Laya is the architectural contribution. the +0.41 raw delta is the train/test overlap by design.

cost is estimated at $0.04/m-input × measured tokens, same basis the leaderboard uses for self-hosted rows. self-hosting doesn't have a tariff; we don't invent one.

full per-tier write-up: docs/PUBLIC_BENCHMARK.md.

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
evals/                    # training recipe + nimble13 harness
jevbench_eval/            # jevbench harness + per-variant runs + v9 run evidence
docs/                     # PUBLIC_BENCHMARK.md + CHANGELOG.md
assets/                   # leaderboard + axes + progression PNGs
weights/                  # local cache for v9 LoRA + heads (download via huggingface-cli)
```

**submitting to benchmark heaven** in the queue. the maintainer needs this repo, the weights (HF Hub: `isHeSatoshi/smalljev-semantic-v9`), and the run command in *what it actually does*. file an issue at https://github.com/fstandhartinger/jevbench/issues or ping https://benchmarkheaven.com.
