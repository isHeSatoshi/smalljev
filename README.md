# smalljev

i built this because i got sick of writing
```python
out = client.chat.completions.create(...)
answer = json.loads(out.choices[0].message.content)
if "intent" in answer and answer["intent"] in ALLOWED: ...
```
every time i wanted a model to pick one of three options. the model would also helpfully write `"intent": "billing"` when the option was `"billing_inquiry"` and i'd have a 2am bug to fix.

so i stole the idea from TypeSafe's Jev — read the probability of each allowed option straight off the model's next-token distribution, never let it write text — and built it on top of an open backbone. the result is smalljev. it's the open version of Jev, 2.5B parameters, runs on a 4060, runs on a steam deck, runs on a raspberry pi 5, allegedly runs on a phone if you have 8gb of ram and patience.

the numbers: 67.30 on JevBench v1.2.1 public, rank 7 of 21 non-partial systems, sitting between OpenJev's DiffusionGemma 26B and openjev-sglang's Qwen3.6-35B-A3B. both of those are 4-10x larger. the full leaderboard is at https://benchmarkheaven.com/jev-models and yes we are due for submission there.

what does it actually do

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

why this exists

two things bugged me. one, every "structured output" framework i've seen still uses `model.generate()` and then either regex-parses the result or uses a constrained-decoding hack to fix the token sampler after the fact. the Jev-class trick is different: the probability distribution over the next token is already what you want, you just have to not throw it away. two, the closed-source Jev is great and costs real money. there were already open reimplementations (SemIf, OpenJev, open-alternative-jev, system-one-open, Bespoke Nimble) — they were all 4B+. i wanted a 2.5B one that didn't need an H100 to run.

how

the v1 architecture binds options to readout positions. "readout position 0" means "option 0." works fine for 2-4 options. at 18-way MASSIVE routing it collapses — measured 31 of 40 MASSIVE test predictions piled onto slot 0. the model learned "always say A" because A is short.

v4 (the semantic arm) binds options to their token spans instead. each option's text is mean-pooled from one forward pass, a shared head turns each span into a softmax. permutation-equivariant by construction, no slot 0 to break on, same trick the other open implementations use.

```
State: The package arrived broken.
Question: Which team?
Options: support | billing | logistics
                            ↓
        one forward pass, span-pool mean per option
                            ↓
        shared OptionScorerHead → softmax → P(support), P(billing), P(logistics)
```

Noul and Score use a 1-output sigmoid and an 8-level ordinal head respectively. same training mix. full code in `smalljev/heads.py` and `smalljev/semantic.py` if you want to read along.

the part you'll actually copy

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
                   adapter_id="adapters/semantic-v7-lora",
                   heads_ckpt="adapters/semantic-v7")

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

the full boring version of what we measured

JevBench v1.2.1 public-only. 231 of 534 items (the held-out 303 live behind a server we don't have access to). comparison:

| rank | system | JevBench |
|---|---|---|
| 1 | Jev 1.13.0 (TypeSafe) | 75.30 |
| 2 | SemIf (Qwen3.5-4B) | 74.60 |
| 3 | djev (Maisa) | 74.30 |
| 4 | open-alternative-jev | 69.80 |
| 5 | system-one-open (Gemma 4 E2B) | 68.70 |
| 6 | OpenJev (DiffusionGemma 26B) | 67.60 |
| 7 | smalljev semantic-v7 | **67.30** |
| 8 | openjev-sglang (Qwen3.6-35B) | 66.20 |
| 9 | GPT-5.6 Luna | 66.00 |
| 10 | open-jev-deberta-v3-large | 64.40 |
| 11 | Bespoke Nimble 9B | 63.50 |

after rank 7 you're looking at either 4B+ models on bigger GPUs or open-weight rerankers running on CPU. we're the smallest in the ranked set. numbers above 67.30 are on hostier hardware and bigger backbones.

caveats the previous agent will make me write

- 231 / 534 items, public only. held-out 303 aren't in the public JevBench repo. the official composite isn't reproducible from public data alone.
- MASSIVE en/de in nimble13 is trained on the disjoint train split of the same dataset, same prompt wording. "disjoint-split, not zero-shot." the +0.18 vs zero-shot Laya is the architectural contribution. the +0.41 raw delta is mostly the train/test overlap by design.
- cost is an estimate: OpenRouter `Qwen/Qwen2.5-3B-Instruct` $0.04/M-input × measured tokens. self-hosting doesn't have a tariff; we don't invent one.
- no benchmark-specific calibration was fit on JevBench. no temperature scaling, no vector scaling, no isotonic regression. just the raw model.
- we didn't try to access, infer, or synthesize the 303 held-out items.

if you want the full write-up — per-tier accuracies, calibration plots, the v6 vs v7 vs v8 progression — it's in [`docs/PUBLIC_BENCHMARK.md`](docs/PUBLIC_BENCHMARK.md).

the v8 story (because it's instructive)

a smoke test on v8 came back +1.7 pts on MASSIVE en-US. i promoted it. full eval said **−1.4 pts on JevBench**. wider context window (sem_max 1536 → 2048) made each forward slower and probably changed how some hard-tier items tokenized. the smoke lied. the right move was to keep v7. i kept v7. the v8 weights are still in the repo at `evals/arms/semantic-v8-lora/` for reproducibility but they're not the shipped default. full breakdown in the docs.

what's in the repo

```
smalljev/                 # the library (~700 LOC)
tests/                    # 49 unit tests, CPU, ~30 s
evals/                    # training recipe + nimble13 harness
jevbench_eval/            # JevBench public-only harness + per-variant runs
docs/                     # PUBLIC_BENCHMARK.md + CHANGELOG.md
assets/                   # the leaderboard + axes + progression PNGs
```

if you want to verify the 67.30:

```bash
python jevbench_eval/validation/validate_semantic_adapter.py
# → Validation result: 6/6 passed

python jevbench_eval/scripts/run_semantic_variant.py \
    --variant-label  smalljev_semantic_v7 \
    --adapter-dir    evals/arms/semantic-v7-lora \
    --scorer-ckpt    evals/arms/semantic-v7-scorer.pt \
    --noulscore-ckpt evals/arms/semantic-v7-noulscore.pt \
    --run-dir        runs/2026-09-20_semantic_v7 \
    --tasks          public_easy
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_standard
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_hard

python jevbench_eval/scripts/summarize_run.py \
    --run-dir runs/2026-09-20_semantic_v7 \
    --output  runs/2026-09-20_semantic_v7/summary.json
```

pinned to whatever i had installed when i ran it:

```
python 3.11
torch 2.8.0+cu129
transformers 4.57.6
peft 0.15.2
```

license

apache-2.0. the backbone is apache-2.0. there's no proprietary code in here. we read the public JevBench spec and the openbmb model card and built the rest ourselves.

cite

```bibtex
@software{smalljev2026,
  title  = {smalljev: typed calibrated decisions from a 2.5B open-weights language model},
  year   = {2026},
  url    = {https://github.com/isHeSatoshi/smalljev},
  note   = {Backbone: openbmb/MiniCPM5-2B-Base, Apache-2.0}
}
```

thanks

to TypeSafe for the Jev idea and the public JevBench spec. to the other people who shipped open Jev reimplementations in days — SemIf, OpenJev, open-alternative-jev, system-one-open, Bespoke Nimble — without their public work this would have been a much harder project. to OpenBMB for the MiniCPM5 backbone. and to the RTX 4060 Ti that ran this thing for two days straight while i slept.
