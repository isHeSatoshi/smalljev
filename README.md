# smalljev

**smalljev semantic-v9** — the open TypeSafe Jev that runs on your mama's phone. 2.5B params, one forward pass, zero generated tokens.

two things live in this repo:

1. **the `smalljev` python library** — `decide()` API that turns a state + question spec into typed probabilities. CPU works, no GPU, no weights. ~700 LOC.
2. **the JevBench harness** — CLI scripts that load the v9 weights and run them against the upstream JevBench public benchmark. needs a CUDA GPU.

the README below covers both.

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

## install

the repo is not on PyPI. you install it from source:

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench,test]"
```

`pip install -e ".[bench,test]"` pulls:
- the library itself
- pytest (so you can run the test suite)
- the `bench` extras (datasets, matplotlib, scikit-learn) needed for the JevBench harness scripts

no backbone downloads yet. no LoRA downloads. no GPU required to install.

verified environment: python 3.11, torch 2.8.0, transformers 4.57.6, peft 0.15.2. other 3.10/3.12 versions likely work but aren't pinned in CI.

## try it — two paths

### path A: 30-second sanity check (no GPU, no weights, no download)

sanity check only. proves the install works and the test suite passes.

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench,test]"
python -m pytest tests/ -q
```

expected: `64 passed in ~30s` (CPU only, no model download).

### path B: full v9 on GPU (downloads ~5 GB backbone + 17 MB LoRA)

this is the real thing. needs CUDA. ~5 GB backbone + 17 MB LoRA + 86 KB heads downloaded from HuggingFace on first run.

```bash
# pull the v9 weights from HF Hub into ./weights/semantic-v9/
huggingface-cli download isHeSatoshi/smalljev-semantic-v9 \
    --local-dir ./weights/semantic-v9
```

then in python:

```python
from smalljev import decide
from smalljev.model import HFBackend

backend = HFBackend(
    model_id="openbmb/MiniCPM5-2B-Base",      # ~5 GB, auto-downloads to HF cache
    adapter_id="isHeSatoshi/smalljev-semantic-v9",  # LoRA, auto-downloads
    sem_ckpt="isHeSatoshi/smalljev-semantic-v9",     # scorer.pt, auto-downloads
)

out = decide(
    "the package arrived broken and the customer wants a refund",
    {"intent": {"type": "choice", "question": "primary issue?",
                "choices": ["damaged", "wrong_item", "late", "other"]},
     "escalate": {"type": "noul", "question": "needs a human right now"},
     "urgency": {"type": "score", "question": "how urgent?",
                 "levels": ["low", "medium", "high"]}},
    backend=backend,
)
print(out)
```

expected: same shape, with real probabilities from the v9 model. first call: ~60 s load + ~150 ms per call. subsequent calls: ~150 ms each.

> **caveat:** the library in this repo is v0/v1 (slot-position). the v9 weights use span-pooled scoring (`OptionScorerHead`), which the library's `HFBackend` supports via the `sem_ckpt=` argument, but the full v9 noul/score pipeline lives in `jevbench_eval/adapter/smalljev_semantic_adapter.py`, not in the public `smalljev/` library. for JevBench scoring use the harness (path C below), not `decide()` directly.

## what it actually does — three primitives

| you give | you get |
|---|---|
| a state + a list of allowed options | `{"values": [...], "probabilities": [...]}` |
| a state + a yes/no question | `{"probability": 0.0–1.0}` |
| a state + an ordinal rubric | `{"value": <float>, "levels": [...], "probabilities": [...]}` |

plus `out["_meta"] = {"backend": "...", "generated_tokens": 0, "n_questions": N}`. generated_tokens is always 0 — the model never calls `.generate()`.

mixable in one call:

```python
decide(state, {
    "intent":   {"type": "choice", "question": "...", "choices": [...]},
    "escalate": {"type": "noul",   "question": "..."},
    "urgency":  {"type": "score",  "question": "...", "levels": [...]},
}, backend=backend)
```

## why i built it

i was building a customer-service triage bot. the first version asked GPT-4 to pick one of three options. worked, mostly. every few hours it would pick "billing" when the right answer was "billing_inquiry" because the prompt wasn't strict enough. 2am bug reports.

then i read about TypeSafe's Jev. the trick: don't let the model generate the answer. read the probability of each allowed option off the model's next-token distribution, pick the highest one, return the whole distribution. no generation, no parsing, no hallucinations. clean idea.

so i built the open version. 2.5B parameters instead of whatever Jev is. fine-tuned on a few thousand labeled decisions. fits on the 4060 and on the steam deck.

## how it works (the part that's actually interesting)

the naive version binds options to readout positions. position 0 = option 0, position 1 = option 1. works for two or three options. at 18-way routing it collapses: 31 of 40 test predictions piled onto position 0 because nothing told the model those positions were semantically different.

v4 binds options to their text instead. tokenize all the options in one forward pass, mean-pool each option's tokens into a vector, run them through a shared head, get a softmax. permutation-equivariant by construction, no slot 0 to break on.

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

heads (option scorer, yes/no, ordinal) are ~200 lines in `smalljev/heads.py`. the span binding is in `smalljev/semantic.py`. full library is ~700 lines.

---

## reproduce the JevBench score (path C)

this is the v9 → JevBench harness. CUDA GPU required. downloads ~5 GB backbone + 17 MB LoRA + 86 KB heads on first run. ~70 s on a 4060 Ti for all 231 public items.

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench,test]"

# 1. download the v9 weights
huggingface-cli download isHeSatoshi/smalljev-semantic-v9 \
    --local-dir ./weights/semantic-v9

# 2. pre-flight: 6 transport-correctness checks on the adapter
python jevbench_eval/validation/validate_semantic_adapter.py
# → 6/6 passed

# 3. clone upstream jevbench (uses a path that works on linux/mac/windows)
git clone --depth 1 https://github.com/fstandhartinger/jevbench.git ./jevbench_upstream

# 4. run the 231 public items
python jevbench_eval/scripts/run_v12.py \
    --variant-label  smalljev_semantic_v9 \
    --adapter-dir    ./weights/semantic-v9/semantic-v9-lora \
    --scorer-ckpt    ./weights/semantic-v9/semantic-v9-scorer.pt \
    --noulscore-ckpt ./weights/semantic-v9/semantic-v9-noulscore.pt \
    --harness-root   ./jevbench_upstream \
    --run-dir        ./runs/2026-09-21_semantic_v9_v12 \
    --tiers          easy,standard,hard

# 5. summarise into the 4-axis v1.2 composite
python jevbench_eval/scripts/summarize_v12.py \
    --run-dir      ./runs/2026-09-21_semantic_v9_v12 \
    --harness-root ./jevbench_upstream
# → JevBench Score (4-axis geometric): 65.53
```

committed run evidence lives at `jevbench_eval/runs/2026-09-21_semantic_v9_v12/` (`manifest.json`, `results.jsonl`, `summary.json`, `summary.md`).

## things worth knowing

- **the library in this repo is v0/v1 (slot-position).** the v9 weights use span-pooled scoring and need the JevBench adapter for full evaluation. the `decide()` API works with both via the `backend` parameter; the JevBench harness uses the adapter directly.
- **the official benchmark has 534 items.** 303 are held out and not in the public repo. the headline score here is on 231 public items.
- **MASSIVE en/de in nimble13** is disjoint-split same-source, not zero-shot. the +0.18 over zero-shot Laya is the architectural contribution.
- **cost** is estimated at $0.04/m-input × measured tokens, same basis the leaderboard uses for self-hosted rows.

full per-tier write-up: [docs/PUBLIC_BENCHMARK.md](docs/PUBLIC_BENCHMARK.md).

## what's in the repo

```
smalljev/                 # the library (~700 LOC)
tests/                    # 49 unit tests, ~30 s on cpu
evals/                    # training recipe + nimble13 harness
jevbench_eval/            # jevbench harness + per-variant runs + v9 run evidence
docs/                     # PUBLIC_BENCHMARK.md + CHANGELOG.md
assets/                   # leaderboard + axes + progression PNGs
weights/                  # local cache for v9 LoRA + heads (download via huggingface-cli)
```

**license** Apache-2.0. backbone is Apache-2.0. nothing proprietary in here.

**cite**

```bibtex
@software{smalljev2026,
  title  = {smalljev: the open TypeSafe Jev that runs on your mama's phone},
  year   = {2026},
  url    = {https://github.com/isHeSatoshi/smalljev}
}
```

**thanks** TypeSafe for the Jev idea and the public JevBench spec. OpenBMB for the MiniCPM5 backbone. the RTX 4060 Ti that ran this for two days straight while i slept.

**submitting to benchmark heaven** in the queue. the maintainer needs this repo, the weights (HF Hub: `isHeSatoshi/smalljev-semantic-v9`), and the run command in *reproduce the JevBench score*. file an issue at https://github.com/fstandhartinger/jevbench/issues or ping https://benchmarkheaven.com.
