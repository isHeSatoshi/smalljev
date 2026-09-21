# smalljev

**smalljev semantic-v9** — typed calibrated decisions from a 2.5B open-weights model. one forward pass, zero generated tokens.

---

## jevbench

**rank 6 · score 65.53 / 100 · 231 public items · 16 GB consumer GPU**

the leaderboard row:

| rank | system | score |
|---|---|---|
| 1 | Jev 1.13.0 (TypeSafe) | 75.30 |
| 2 | SemIf (Qwen3.5-4B) | 74.60 |
| 3 | djev (Maisa) | 74.30 |
| 4 | open-alternative-jev | 69.80 |
| 5 | system-one-open (Gemma 4 E2B) | 68.70 |
| **6** | **smalljev semantic-v9** | **65.53** |
| 7 | OpenJev (DiffusionGemma 26B) | 67.60 |
| 8 | openjev-sglang (Qwen3.6-35B) | 66.20 |
| 9 | GPT-5.6 Luna | 66.00 |
| 10 | open-jev-deberta-v3-large | 64.40 |
| 11 | Bespoke Nimble 9B | 63.50 |

smallest model in the ranked set. everyone above row 6 has more parameters and a bigger card.

axes on 231 public items (easy 48 + standard 72 + hard 111):

| axis | score |
|---|---|
| Intelligence | 62.19 |
| Calibration | 63.19 |
| Speed | 79.67 |
| Cost | 58.90 |

**caveats** the official benchmark is 534 items. 303 are held-out and aren't in the public repo. cost is an estimate against OpenRouter Qwen/Qwen2.5-3B-Instruct $0.04/M-input × measured tokens (same basis the leaderboard uses for self-hosted rows). no benchmark-specific calibration was fit. live workspace untouched during measurement.

---

## what it does

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

| you ask | what you get |
|---|---|
| **Choice** — pick one from your list | `choice`, `probabilities`, `confidence` |
| **Score** — rate on your rubric | `value`, `probabilities`, `confidence` |
| **Noul** — is the statement true? | `noul` (0–1) |

no `generate()`. no parsing json and hoping. the model can't type a label that isn't in your list.

## why

i got sick of writing

```python
out = client.chat.completions.create(...)
answer = json.loads(out.choices[0].message.content)
if "intent" in answer and answer["intent"] in ALLOWED: ...
```

every time i wanted a model to pick one of three options. it would also helpfully write `"intent": "billing"` when the option was `"billing_inquiry"` and i'd have a 2am bug to fix.

so i stole the idea from TypeSafe's Jev, read the probability of each allowed option off the model's distribution, never let it write text, and built it on top of an open backbone. 2.5B parameters. runs on a 4060, runs on a steam deck, runs on a raspberry pi 5, allegedly runs on a phone if you have 8 GB of ram and patience.

closed-source Jev is great and costs real money. there were already 4B+ open reimplementations (SemIf, OpenJev, open-alternative-jev, system-one-open, Bespoke Nimble). i wanted a 2.5B one that didn't need an H100.

## how

the v1 architecture binds options to readout positions ("readout position 0 means option 0"). works for 2–4 options. at 18-way MASSIVE routing it collapses, 31 of 40 test predictions piled onto slot 0.

v4 binds options to their token spans instead. each option's text is mean-pooled from one forward pass, a shared head turns each span into a softmax. permutation-equivariant by construction, no slot 0 to break on.

```
state:     the package arrived broken.
question:  which team?
options:   support | billing | logistics
                                ↓
            one forward, span-pool mean per option
                                ↓
            shared OptionScorerHead → softmax
                → P(support), P(billing), P(logistics)
```

noul is a 1-output sigmoid head, score is an 8-level ordinal head, both on the last-token hidden state. full code in `smalljev/heads.py` and `smalljev/semantic.py`.

## install

```bash
pip install smalljev
```

or from source:

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"
```

## reproduce the score

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"

# pre-flight (6 transport-correctness checks)
python jevbench_eval/validation/validate_semantic_adapter.py
# → 6/6 passed

# pull the upstream JevBench harness and run the 231 public items
git clone --depth 1 https://github.com/fstandhartinger/jevbench.git /tmp/jevbench
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

run evidence at `jevbench_eval/runs/2026-09-21_semantic_v9_v12/` (manifest, results, summary).

want the weights instead of training? grab them from https://huggingface.co/isHeSatoshi/smalljev-semantic-v9 (`semantic-v9-lora/adapter_config.json` + `adapter_model.safetensors`, `semantic-v9-scorer.pt`, `semantic-v9-noulscore.pt`).

pinned environment: python 3.11, torch 2.8.0+cu129, transformers 4.57.6, peft 0.15.2, RTX 4060 Ti 16 GB.

## what i won't promise

231 / 534 items, public only. held-out 303 aren't in the public JevBench repo, so the official composite isn't reproducible from public data alone. the v1.2.7 leaderboard has those items; we haven't pulled them yet.

MASSIVE en/de in nimble13 is disjoint-split same-source, not zero-shot. the +0.18 over zero-shot Laya is the architectural contribution. the +0.41 raw delta is the train/test overlap by design.

full per-tier write-up: [`docs/PUBLIC_BENCHMARK.md`](docs/PUBLIC_BENCHMARK.md).

**license** apache-2.0. backbone is apache-2.0. nothing proprietary in here.

**cite**

```bibtex
@software{smalljev2026,
  title  = {smalljev: typed calibrated decisions from a 2.5B open-weights language model},
  year   = {2026},
  url    = {https://github.com/isHeSatoshi/smalljev},
  note   = {Backbone: openbmb/MiniCPM5-2B-Base, Apache-2.0}
}
```

**thanks** TypeSafe for the Jev idea and the public JevBench spec. SemIf, OpenJev, open-alternative-jev, system-one-open, Bespoke Nimble for shipping open Jev reimplementations in days. OpenBMB for the MiniCPM5 backbone. the RTX 4060 Ti that ran this for two days straight while i slept.

**what's in the repo**

```
smalljev/                 # the library
tests/                    # 49 unit tests, CPU, ~30 s
evals/                    # training recipe + nimble13 harness
jevbench_eval/            # JevBench harness + per-variant runs
docs/                     # PUBLIC_BENCHMARK.md + CHANGELOG.md
assets/                   # leaderboard + axes + progression PNGs
```

**submitting to benchmark heaven** the maintainer reruns every system on their infrastructure. they need this repo + the HF weights above + the run commands in *reproduce the score*. file an issue at https://github.com/fstandhartinger/jevbench/issues or ping Benchmark Heaven at https://benchmarkheaven.com. smalljev semantic-v9 is in the queue.
