# smalljev

A tiny model that turns messy real-world text into typed probabilities — no JSON parsing, no decoding loop, no hallucinated labels. One forward pass, zero generated tokens.

smalljev is the open-source answer to TypeSafe's Jev, built on the same idea (read option probabilities straight off a frozen language model) but with weights you can actually download. The fastest publicly-measured open reimplementation on the JevBench leaderboard, on a 2.5B backbone that fits on a 16 GB consumer GPU.

```text
state + questions
        ↓
MiniCPM5-2B-Base (frozen + LoRA)
        ↓
OptionScorerHead + BinaryNoulHead + OrdinalScoreHead
        ↓
typed JSON, probabilities, confidence — no generation
```

Independent project. Not affiliated with TypeSafe AI.

## What it actually does

Three primitives, one call:

| Question type | What it asks | What you get back |
|---|---|---|
| **Choice** | "Pick one option from this list" | `choice`, `probabilities`, `confidence` |
| **Score** | "Score the state on this rubric" | `score`, `probabilities`, `confidence` |
| **Noul** | "Is this statement true about the state?" | `noul` (0–1) |

Mix them in a single request. Each question is evaluated independently in one forward pass — adding more questions barely changes the latency.

```python
from smalljev import decide
from smalljev.model import HFBackend

backend = HFBackend("openbmb/MiniCPM5-2B-Base",
                   adapter_id="adapters/semantic-v7-lora",
                   heads_ckpt="adapters/semantic-v7")

out = decide(
    "Customer was charged twice for the same order.",
    {
        "intent":   {"type": "choice", "question": "What is the primary issue?",
                     "choices": ["duplicate charge", "late delivery",
                                 "account problem", "other"]},
        "escalate": {"type": "noul",
                     "question": "This case should be escalated to a human."},
        "risk":     {"type": "score", "question": "How risky is this?",
                     "levels": ["very low", "low", "medium", "high", "very high"]},
    },
    backend=backend,
)
```

`out["intent"]` → `{"choice": "duplicate charge", "probabilities": [...], "confidence": 0.74}`
`out["escalate"]` → `{"noul": 0.78}`
`out["risk"]` → `{"value": 2.4, "levels": [...], "probabilities": [...]}`
`out["_meta"]` → `{"backend": "...", "generated_tokens": 0, "n_questions": 3}`

That's it. No `generate()`. No JSON repair.

## Where it ranks

Measured against the official [JevBench v1.2.1](https://benchmarkheaven.com/jev-models) harness on the public half (231 of 534 items; the held-out 303 aren't in the public repo):

| # | System | JevBench |
|---|---|---|
| 1 | Jev 1.13.0 (TypeSafe AI) | 75.30 |
| 2 | SemIf (Qwen3.5-4B) | 74.60 |
| 3 | djev (Maisa) | 74.30 |
| 4 | open-alternative-jev (Qwen3.5-4B) | 69.80 |
| 5 | system-one-open (Gemma 4 E2B) | 68.70 |
| 6 | OpenJev (DiffusionGemma 26B) | 67.60 |
| **7** | **smalljev semantic-v7** | **67.30** |
| 8 | openjev-sglang (Qwen3.6-35B-A3B) | 66.20 |
| 9 | GPT-5.6 Luna | 66.00 |
| 10 | open-jev-deberta-v3-large | 64.40 |
| 11 | Bespoke Nimble 9B | 63.50 |

We're the smallest model in the ranked set (2.5B), the only one whose inference fits in ~5 GB VRAM, and the only one whose probability origin is span-pooled softmax rather than letter-position softmax.

A v8 attempt — wider context window, longer training — regressed by 1.4 points (full write-up in [`docs/PUBLIC_BENCHMARK.md`](docs/PUBLIC_BENCHMARK.md)). v7 is the champion we're shipping.

The honest caveats are at the top of that write-up. Short version: 231 / 534 items, MASSIVE numbers are disjoint-split not zero-shot, cost is estimated against OpenRouter Qwen2.5-3B-Instruct, no benchmark-specific calibration was fit, live workspace was not modified.

## Why a different head

The v1 architecture bound options to positional slots: `SlotChoiceHead` reads the readout vector at position k as "option k". That works for 2–4 options. At 18-way MASSIVE routing it collapses — measured 31 of 40 MASSIVE predictions piled onto slot 0.

v4 (and onward, the "semantic" arm) replaces positional binding with semantic binding: each option's *token span* is mean-pooled from one forward pass, then a shared `OptionScorerHead` turns each span into a softmax. The head is permutation-equivariant by construction — there is no slot 0 to collapse onto. Same trick the public leaderboard systems all use; we wrote it out long-hand so you can read the code.

```
State: The package arrived broken.
Question: Which team?
Options: support | billing | logistics
                            ↓
        one forward pass, span-pool mean per option
                            ↓
        shared OptionScorerHead → softmax → P(support), P(billing), P(logistics)
```

For the Noul bit and Score ordinal levels the heads are still 1-output sigmoid and 8-level ordinal respectively, trained against the same corpus.

## Install

```bash
pip install smalljev
```

CPU works. GPU is much faster.

## Try it

The HuggingFace Space is the live demo: paste a state, write 1–3 questions in plain English, get probabilities back. No install.

[<link added when Space is published>](https://huggingface.co/) · [`hf_space/app.py`](hf_space/app.py) is the Space source.

A notebook-friendly local demo lives in [`scripts/demo_trajectory.py`](scripts/demo_trajectory.py).

## Run the JevBench public eval yourself

The harness that produced the 67.30 score is checked in:

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"

# pre-flight
python jevbench_eval/validation/validate_semantic_adapter.py
# → Validation result: 6/6 passed

# run the 231 public items
python jevbench_eval/scripts/run_semantic_variant.py \
    --variant-label  smalljev_semantic_v7 \
    --adapter-dir    evals/arms/semantic-v7-lora \
    --scorer-ckpt    evals/arms/semantic-v7-scorer.pt \
    --noulscore-ckpt evals/arms/semantic-v7-noulscore.pt \
    --run-dir        runs/2026-09-20_semantic_v7 \
    --tasks          public_easy
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_standard
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_hard

# summarize
python jevbench_eval/scripts/summarize_run.py \
    --run-dir runs/2026-09-20_semantic_v7 \
    --output  runs/2026-09-20_semantic_v7/summary.json
```

Pinned environment the score was measured on:

| Library | Version |
|---|---|
| Python | 3.11 |
| torch | 2.8.0+cu129 |
| transformers | 4.57.6 |
| peft | 0.15.2 |

Every `results.jsonl` carries a `runtime` block with the device, threads, backbone id, adapter id, scorer / noulscore paths, generated tokens (always 0), and library versions — so a manifest can reproduce the run.

## Repo layout

```
smalljev/
├── smalljev/                 # the library (~700 LOC)
│   ├── api.py                # decide(state, questions, backend)
│   ├── spec.py               # ChoiceQuestion / NoulQuestion / ScoreQuestion
│   ├── model.py              # HFBackend (frozen + LoRA + heads)
│   ├── heads.py              # OptionScorerHead + BinaryNoulHead + OrdinalScoreHead
│   ├── semantic.py           # span-pooled semantic scoring
│   ├── parallel.py           # shared-prefix block-isolation mask
│   ├── calibration.py        # softmax, temperature, NLL/Brier/ECE
│   └── confidence.py         # confidence bands, OOD detection
├── tests/                    # 49 unit tests, CPU-only, ~30 s
├── evals/                    # training + benchmark scripts
│   ├── train_semantic.py     # the semantic-arm trainer
│   ├── eval_semantic.py      # per-task nimble13 harness
│   └── nimble13.py           # 13-task battery builder
├── jevbench_eval/            # JevBench v1.2.1 public-only measurements
│   ├── adapter/              # smalljev-semantic JevBench adapter
│   ├── scripts/              # run + summarize
│   ├── runs/                 # per-variant results + summaries
│   └── JEVBENCH_DETAILED_TABLE.md
├── docs/
│   ├── PUBLIC_BENCHMARK.md    # full JevBench write-up
│   ├── ARCHITECTURE.md        # design + decisions
│   └── METHOD.md              # measurement methodology
├── adapters/                 # HF + JevBench adapter implementations
├── benchmarks/               # stub vs sklearn smoke harness
├── assets/                   # PNG figures (leaderboard, axes, etc.)
├── scripts/                  # end-to-end smoke demos
└── hf_space/                 # Gradio Space source (app.py + requirements)
```

## License

Apache-2.0. The backbone `openbmb/MiniCPM5-2B-Base` is Apache-2.0 too; we did not retrain or fine-tune the base. No proprietary code.

## Cite

```bibtex
@software{smalljev2026,
  title  = {smalljev: typed calibrated decisions from a 2.5B open-weights language model},
  year   = {2026},
  url    = {https://github.com/isHeSatoshi/smalljev},
  note   = {Backbone: openbmb/MiniCPM5-2B-Base, Apache-2.0}
}
```

## Thanks

- TypeSafe for the Jev / System One Models idea and the public benchmark protocol at [github.com/fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench).
- OpenBMB for the MiniCPM5 backbone.
- The independent JevBench evaluator whose harness in `jevbench_eval/` made the 67.30 number reproducible.
- The other Jev-class projects — [SemIf](https://github.com/TheoLeeCJ/SemIf), [OpenJev (razorback16)](https://github.com/razorback16/openjev), [open-alternative-jev](https://github.com/ikermoel/open-alternative-jev), [system-one-open](https://github.com/mithalouni/system-one-open), [Bespoke Nimble](https://github.com/bespokelabsai/nimble) — shipped public reproductions in days, set the bar for what honest measurement looks like, and made this benchmark category a community instead of a single-vendor thing.
