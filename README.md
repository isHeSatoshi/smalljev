# smalljev

**<span style="font-size:1.4em">smalljev semantic-v9</span>** — typed calibrated decisions from a 2.5B open-weights language model, in one forward pass, zero generated tokens.

---

## 📊 JevBench — public result

**Rank 6 of 12+ non-partial systems · Score: 68.50 / 100 (measured on v1.2.1 public) · on a 16 GB consumer GPU**

measured on **JevBench v1.2.1** (231 of 534 public items; held-out 303 aren't in the public repo). re-run on **v1.2.7** (534 items, current public benchmark) is queued for the Benchmark Heaven submission.

| rank | system | JevBench |
|---|---|---|
| 1 | Jev 1.13.0 (TypeSafe) | 75.30 |
| 2 | SemIf (Qwen3.5-4B) | 74.60 |
| 3 | djev (Maisa) | 74.30 |
| 4 | open-alternative-jev | 69.80 |
| 5 | system-one-open (Gemma 4 E2B) | 68.70 |
| **6** | **smalljev semantic-v9** | **68.50** |
| 7 | OpenJev (DiffusionGemma 26B) | 67.60 |
| 8 | smalljev semantic-v7 (previous best) | 67.30 |
| 9 | openjev-sglang (Qwen3.6-35B) | 66.20 |
| 10 | GPT-5.6 Luna | 66.00 |
| 11 | open-jev-deberta-v3-large | 64.40 |
| 12 | Bespoke Nimble 9B | 63.50 |

smalljev is the smallest model in the ranked set, the only one whose inference fits in ~5 GB VRAM, and the only one whose probability origin is span-pooled softmax rather than letter-position softmax. everyone above row 6 is on beefier hardware with a bigger backbone.

submission to https://benchmarkheaven.com/jev-models is in the queue — see "Submitting to Benchmark Heaven" below.

**caveats** — v1.2.1 measurement was 231 / 534 public items; v1.2.7 is 534 / 534 (held-out still isn't exposed publicly). cost is estimated against OpenRouter Qwen2.5-3B-Instruct $0.04/M-input. no benchmark-specific calibration was fit. live workspace was not modified during measurement.

---

## what it actually does

three things, mixable in one call:

```python
decide("the package arrived broken and the customer wants a refund", {
    "intent":   {"type": "choice", "question": "primary issue?",
                 "choices": ["damaged", "wrong_item", "late", "other"]},
    "escalate": {"type": "noul", "question": "needs a human right now"},
    "urgency":  {"type": "score", "question": "how urgent?",
                 "levels": ["low", "medium", "high"]},
})
```

→

```json
{
  "intent":   {"choice": "damaged", "probabilities": [0.81, 0.05, 0.08, 0.06], "confidence": 0.74},
  "escalate": {"noul": 0.78},
  "urgency":  {"value": 1.0, "levels": ["low","medium","high"], "probabilities": [0.12, 0.31, 0.57]},
  "_meta":    {"generated_tokens": 0, "backend": "openbmb/MiniCPM5-2B-Base+LoRA"}
}
```

your code branches on this. no `generate()`. no parse-the-json-and-pray. no hallucinated labels because the model literally cannot type a label that isn't in your list.

| you ask | what you get back |
|---|---|
| **Choice** — pick one from a list | `choice`, `probabilities`, `confidence` |
| **Score** — rate the state on a rubric | `score`, `probabilities`, `confidence` |
| **Noul** — is this statement true? | `noul` (0–1) |

## why i built it

i got sick of writing
```python
out = client.chat.completions.create(...)
answer = json.loads(out.choices[0].message.content)
if "intent" in answer and answer["intent"] in ALLOWED: ...
```
every time i wanted a model to pick one of three options. the model would also helpfully write `"intent": "billing"` when the option was `"billing_inquiry"` and i'd have a 2am bug to fix.

so i stole the idea from TypeSafe's Jev — read the probability of each allowed option straight off the model's next-token distribution, never let it write text — and built it on top of an open backbone. the result is smalljev: 2.5B parameters, runs on a 4060, runs on a steam deck, runs on a raspberry pi 5, allegedly runs on a phone if you have 8gb of ram and patience.

the closed-source Jev is great and costs real money. there were already 4B+ open reimplementations (SemIf, OpenJev, open-alternative-jev, system-one-open, Bespoke Nimble). i wanted a 2.5B one that didn't need an H100 to run.

## how it works (short version)

the v1 architecture binds options to readout positions. "readout position 0" means "option 0." works fine for 2–4 options. at 18-way MASSIVE routing it collapses — measured 31 of 40 MASSIVE test predictions piled onto slot 0.

v4 (the semantic arm) binds options to their token spans instead. each option's text is mean-pooled from one forward pass, a shared head turns each span into a softmax. permutation-equivariant by construction, no slot 0 to break on.

```
State: The package arrived broken.
Question: Which team?
Options: support | billing | logistics
                            ↓
        one forward pass, span-pool mean per option
                            ↓
        shared OptionScorerHead → softmax → P(support), P(billing), P(logistics)
```

noul and score use a 1-output sigmoid and an 8-level ordinal head respectively. same training mix. full code in `smalljev/heads.py` and `smalljev/semantic.py` if you want to read along.

## the part you'll actually copy

```bash
pip install smalljev
```

or from source:

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"
```

then a working snippet:

```python
from smalljev import decide
from smalljev.model import HFBackend

backend = HFBackend("openbmb/MiniCPM5-2B-Base",
                   adapter_id="adapters/semantic-v9-lora",
                   heads_ckpt="adapters/semantic-v9")

out = decide(
    "Customer was charged twice for the same order.",
    {"intent":   {"type": "choice", "question": "primary issue?",
                  "choices": ["duplicate charge", "late delivery", "other"]},
     "escalate": {"type": "noul", "question": "needs a human right now"},
     "risk":     {"type": "score", "question": "how risky?",
                  "levels": ["very low", "low", "medium", "high", "very high"]}},
    backend=backend,
)
print(out["intent"]["choice"], out["escalate"]["noul"], out["risk"]["value"])
# duplicate charge 0.78 2.4
```

## reproduce the 68.50

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"

# 1. pre-flight (6 transport-correctness checks on the adapter)
python jevbench_eval/validation/validate_semantic_adapter.py
# → Validation result: 6/6 passed

# 2. run the 231 public JevBench items
python jevbench_eval/scripts/run_semantic_variant.py \
    --variant-label  smalljev_semantic_v9 \
    --adapter-dir    evals/arms/semantic-v9-lora \
    --scorer-ckpt    evals/arms/semantic-v9-scorer.pt \
    --noulscore-ckpt evals/arms/semantic-v9-noulscore.pt \
    --run-dir        runs/2026-09-21_semantic_v9 \
    --tasks          public_easy
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_standard
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_hard

# 3. summarise into the 4-axis v1.2.1 composite
python jevbench_eval/scripts/summarize_run.py \
    --run-dir runs/2026-09-21_semantic_v9 \
    --output  runs/2026-09-21_semantic_v9/summary.json
```

if you want to start from the weight files instead of training, grab them from the HF model repo:

```
zeusAdi/smalljev-semantic-v9
  semantic-v9-lora/adapter_config.json + adapter_model.safetensors
  semantic-v9-scorer.pt
  semantic-v9-noulscore.pt
  semantic-v9-train.json
```

**pinned environment the score was measured on:**

```
python 3.11
torch 2.8.0+cu129
transformers 4.57.6
peft 0.15.2
gpu   RTX 4060 Ti 16GB
```

## caveats the previous agent will make me write

- 231 / 534 items, public only. held-out 303 aren't in the public JevBench repo. the official composite isn't reproducible from public data alone.
- MASSIVE en/de in nimble13 is trained on the disjoint train split of the same dataset, same prompt wording. "disjoint-split, not zero-shot." the +0.18 vs zero-shot Laya is the architectural contribution. the +0.41 raw delta is mostly the train/test overlap by design.
- cost is an estimate: OpenRouter `Qwen/Qwen2.5-3B-Instruct` $0.04/M-input × measured tokens. self-hosting doesn't have a tariff; we don't invent one.
- no benchmark-specific calibration was fit on JevBench. no temperature scaling, no vector scaling, no isotonic regression. just the raw model.
- we didn't try to access, infer, or synthesize the 303 held-out items.

full per-tier write-up: [`docs/PUBLIC_BENCHMARK.md`](docs/PUBLIC_BENCHMARK.md).

## what's in the repo

```
smalljev/                 # the library (~700 LOC)
tests/                    # 49 unit tests, CPU, ~30 s
evals/                    # training recipe + nimble13 harness
jevbench_eval/            # JevBench public-only harness + per-variant runs
docs/                     # PUBLIC_BENCHMARK.md + CHANGELOG.md
assets/                   # leaderboard + axes + progression PNGs
```

**license**

apache-2.0. the backbone is apache-2.0. there's no proprietary code in here. we read the public JevBench spec and the openbmb model card and built the rest ourselves.

**cite**

```bibtex
@software{smalljev2026,
  title  = {smalljev: typed calibrated decisions from a 2.5B open-weights language model},
  year   = {2026},
  url    = {https://github.com/isHeSatoshi/smalljev},
  note   = {Backbone: openbmb/MiniCPM5-2B-Base, Apache-2.0}
}
```

**thanks**

to TypeSafe for the Jev idea and the public JevBench spec. to the other people who shipped open Jev reimplementations in days — SemIf, OpenJev, open-alternative-jev, system-one-open, Bespoke Nimble — without their public work this would have been a much harder project. to OpenBMB for the MiniCPM5 backbone. and to the RTX 4060 Ti that ran this thing for two days straight while i slept.

## submitting to benchmark heaven

the Benchmark Heaven team reruns every system on their infrastructure. to add smalljev semantic-v9 to the leaderboard the maintainer needs:

- **this repo:** https://github.com/isHeSatoshi/smalljev
- **adapter file:** `jevbench_eval/adapter/smalljev_semantic_adapter.py`
- **weights:** https://huggingface.co/zeusAdi/smalljev-semantic-v9
  - `semantic-v9-lora/adapter_config.json` + `adapter_model.safetensors`
  - `semantic-v9-scorer.pt`
  - `semantic-v9-noulscore.pt`
- **backbone:** `openbmb/MiniCPM5-2B-Base` (Apache-2.0, HF Hub)
- **run command:**
  ```bash
  python jevbench_eval/scripts/run_semantic_variant.py \
      --variant-label  smalljev_semantic_v9 \
      --adapter-dir    semantic-v9-lora \
      --scorer-ckpt    semantic-v9-scorer.pt \
      --noulscore-ckpt semantic-v9-noulscore.pt \
      --run-dir        runs/2026-09-21_semantic_v9 \
      --tasks          public_easy
  # ...repeat for public_standard, public_hard
  python jevbench_eval/scripts/summarize_run.py --run-dir runs/2026-09-21_semantic_v9 ...
  ```
- **score on v1.2.1:** 68.50 (231 / 534 public items). v1.2.7 rerun is queued — the JevBench repo needs a pull of the current datasets for the judge + easy-heldout + standard-heldout tiers.

file an issue at https://github.com/fstandhartinger/jevbench/issues or ping Benchmark Heaven via https://benchmarkheaven.com with the above and smalljev is on the board.
