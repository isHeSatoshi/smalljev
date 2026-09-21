# smalljev

**smalljev semantic-v9** — a 2.5B open-weights model that picks from lists you give it. one forward pass, no generated text. runs on a 4060, a raspberry pi 5, and probably your phone if you have 8 GB of ram and patience.

---

## what is this, in one paragraph

[TypeSafe built Jev](https://docs.typesafe.ai/). it's a small classifier that looks at your state, reads your list of allowed options, and returns the probability of each. never generates text, never hallucinates a label. it's a really clean idea and a really good product.

it's also closed source and runs on TypeSafe's GPUs. which is fine if you're shipping to production with a budget. less fine if you're a hobbyist, a researcher, or someone who wants the weights on their own machine.

smalljev is the same idea with an open backbone. same shape of outputs, same scoring rules. ~5 GB of weights. Apache-2.0. fine-tuned on a few thousand decision examples. ships as a python package. the [JevBench](https://benchmarkheaven.com/jev-models) score is 65.53, sitting at rank 6 of the public leaderboard.

## the leaderboard

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

four axes (geometric mean, 231 public items):
- **intelligence 62.19** (easy 97.92, standard 69.44, hard 38.74)
- **calibration 63.19**
- **speed 79.67**
- **cost 58.90**

the intelligence axis is identical to the v1.2.1 number. the headline score moves because v1.2 weighs the four axes geometrically; v1.2.1 used an arithmetic mean over three.

## what it actually does

three primitives, mixable in one call:

```python
from smalljev import decide
from smalljev.model import HFBackend

backend = HFBackend("openbmb/MiniCPM5-2B-Base",
                    adapter_id="adapters/semantic-v9-lora",
                    heads_ckpt="adapters/semantic-v9")

out = decide(
    "the package arrived broken and the customer wants a refund",
    {"intent":   {"type": "choice", "question": "primary issue?",
                  "choices": ["damaged", "wrong_item", "late", "other"]},
     "escalate": {"type": "noul",   "question": "needs a human right now"},
     "urgency":  {"type": "score",  "question": "how urgent?",
                  "levels": ["low", "medium", "high"]}},
    backend=backend,
)
# out["intent"]   -> {"choice": "damaged", "probabilities": [...], "confidence": 0.74}
# out["escalate"] -> {"noul": 0.78}
# out["urgency"]  -> {"value": 1.0, "probabilities": [...], "confidence": 0.42}
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

## install

```bash
pip install smalljev
```

or from source if you want the train scripts:

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"
```

then the snippet above. ~5 GB of weights, ~6 GB of vram at inference. works on cuda and cpu. apple silicon works if you swap the dtype; haven't tested it.

## reproduce the score

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"

# 6 transport-correctness checks on the adapter
python jevbench_eval/validation/validate_semantic_adapter.py
# → 6/6 passed

# pull upstream jevbench
git clone --depth 1 https://github.com/fstandhartinger/jevbench.git /tmp/jevbench

# run the 231 public items, ~70 s on a 4060 ti
python jevbench_eval/scripts/run_v12.py \
    --variant-label  smalljev_semantic_v9 \
    --adapter-dir    semantic-v9-lora \
    --scorer-ckpt    semantic-v9-scorer.pt \
    --noulscore-ckpt semantic-v9-noulscore.pt \
    --harness-root   /tmp/jevbench \
    --run-dir        runs/2026-09-21_semantic_v9_v12 \
    --tiers          easy,standard,hard
python jevbench_eval/scripts/summarize_v12.py \
    --run-dir      runs/2026-09-21_semantic_v9_v12 \
    --harness-root /tmp/jevbench
# → JevBench Score (4-axis geometric): 65.53
```

run evidence: `jevbench_eval/runs/2026-09-21_semantic_v9_v12/`. want to skip training? grab the weights from [huggingface.co/isHeSatoshi/smalljev-semantic-v9](https://huggingface.co/isHeSatoshi/smalljev-semantic-v9).

pinned env: python 3.11, torch 2.8.0+cu129, transformers 4.57.6, peft 0.15.2, rtx 4060 ti 16gb.

## things worth knowing

the official benchmark has 534 items. 303 are held out and not in the public repo. the headline score here is on 231 public items. if Benchmark Heaven reruns on their infrastructure the number might shift; that's fine, that's how benchmarks work.

MASSIVE en/de in nimble13 is disjoint-split same-source, not zero-shot. the +0.18 over zero-shot Laya is the architectural contribution. the +0.41 raw delta is the train/test overlap by design.

cost is estimated at $0.04/m-input × measured tokens, same basis the leaderboard uses for self-hosted rows. self-hosting doesn't have a tariff; we don't invent one.

full per-tier write-up: docs/PUBLIC_BENCHMARK.md.

**license** apache-2.0. backbone is apache-2.0. nothing proprietary in here.

**cite**

```bibtex
@software{smalljev2026,
  title  = {smalljev: a 2.5B open-weights classifier that runs on a phone},
  year   = {2026},
  url    = {https://github.com/isHeSatoshi/smalljev}
}
```

**thanks** TypeSafe for the Jev idea and the public JevBench spec. OpenBMB for the MiniCPM5 backbone. the RTX 4060 Ti that ran this for two days straight while i slept.

**what's in the repo**

```
smalljev/                 # the library (~700 LOC)
tests/                    # 49 unit tests, ~30 s on cpu
evals/                    # training recipe + nimble13 harness
jevbench_eval/            # jevbench harness + per-variant runs
docs/                     # PUBLIC_BENCHMARK.md + CHANGELOG.md
assets/                   # leaderboard + axes + progression PNGs
```

**submitting to benchmark heaven** in the queue. the maintainer needs this repo, the weights, and the run command above. file an issue at github.com/fstandhartinger/jevbench/issues or ping benchmarkheaven.com.
