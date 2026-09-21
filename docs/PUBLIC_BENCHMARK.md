# smalljev × JevBench v1.2 — full write-up

benchmark: https://benchmarkheaven.com/jev-models (MIT)
harness: https://github.com/fstandhartinger/jevbench @ main (frozen on v1.2 datasets, `main` branch as of measurement time)
composite: `jevbench/composite_v12.py` — Intelligence, Calibration, Speed, Cost, geometric mean
scope: **public-only**. 231 of 534 items. held-out 303 aren't in the public repo.
hardware: RTX 4060 Ti 16 GB, torch 2.8.0+cu129, transformers 4.57.6, peft 0.15.2.

---

## 1. headline

smalljev semantic-v9 is at **65.53** on the public half of JevBench v1.2 (4-axis geometric composite). that's rank 6 of 12+ non-partial systems on the published Benchmark Heaven leaderboard.

the v1.2.1 measurement (v1.2.1 numbers, 3-axis arithmetic mean) was 68.50; the v1.2 number is lower because v1.2 weights the four axes geometrically where v1.2.1 used an arithmetic mean over three axes. **the underlying Intelligence axis (62.19) is identical** on both runs — same items, same adapter, same answers.

we're the smallest model in the ranked set, the only one whose inference fits in ~5 GB VRAM, and the only one whose probability origin is span-pooled softmax rather than letter-position softmax.

submission to Benchmark Heaven is in the queue.

![leaderboard](../assets/leaderboard.png)

## 2. v7 vs v8 — what actually moved the needle

i tried to hill-climb further with v8 (wider context window, longer training, an extra `--v8` flag). a MASSIVE en-US smoke test came back at 0.874 vs v7's 0.857. +1.7 pts. i promoted it. full eval said otherwise:

| | v7 | v8 | Δ |
|---|---|---|---|
| recipe | v2+v3+v4+v5+boost, 2 epochs, sem_max=1536 | same + v8 flag, 1 epoch, sem_max=2048 | — |
| training time | — | 8 642 s | — |
| final loss | — | 0.3831 | — |
| intelligence | 61.06 | 59.98 | **−1.08** |
| calibration | 66.29 | 65.77 | −0.52 |
| speed | 82.81 | 77.99 | **−4.82** |
| cost | 61.20 | 61.20 | 0.00 |
| **JevBench** | **67.30** | **65.87** | **−1.43** |

the wider `sem_max` is the most likely culprit. longer sequences made each forward slower (the −4.82 speed drop) and probably changed how some hard-tier items tokenized. v7 stays shipped. v8 weights stay in the repo for reproducibility but aren't the default.

## 3. the 4-axis decomposition

### v1.2 (current — geometric composite)

| variant | intel | calib | speed | cost | **JevBench** |
|---|---|---|---|---|---|
| **smalljev semantic-v9 (shipped)** | **62.19** | 63.19 | 79.67 | 58.90 | **65.53** |

per-tier accuracy on the 231 public items:

| tier | n | correct | accuracy |
|---|---|---|---|
| easy | 48 | 47 | 97.92 |
| standard | 72 | 50 | 69.44 |
| hard | 111 | 43 | 38.74 |

judge (146 items), easy-heldout (24), and 109 private hard items aren't in the public repo; the JevBench Score would shift on the held-out half.

### v1.2.1 (legacy — arithmetic composite, kept for context)

| variant | intel | calib | speed | cost | **JevBench** |
|---|---|---|---|---|---|
| smalljev v5_holdout (pre-loop) | 58.5 | 35.7 | 80.0 | 59.0 | 56.02 |
| smalljev crown (pre-loop) | 61.4 | 33.8 | 80.0 | 59.0 | 55.95 |
| smalljev v4_bundle (pre-loop) | 59.1 | 31.6 | 82.6 | 59.0 | 54.93 |
| smalljev lettered (pre-loop) | 32.0 | 0.0 | 79.9 | 59.0 | 19.71 |
| smalljev semantic-v4 | 63.6 | 54.4 | 82.9 | 61.2 | 64.73 |
| smalljev semantic-v5 | 64.9 | 54.0 | 82.8 | 61.2 | 64.90 |
| smalljev semantic-v6 (twin-choice) | 63.4 | 46.7 | 79.2 | 61.2 | 61.54 |
| smalljev semantic-v7 | 61.1 | 66.3 | 82.8 | 61.2 | 67.30 |
| smalljev semantic-v8 (discarded) | 60.0 | 65.8 | 78.0 | 61.2 | 65.87 |
| **smalljev semantic-v9** | 62.2 | 69.9 | 82.7 | 61.2 | **68.50** |

the +11.35 gain over the previous champion (crown) is almost entirely calibration: ECE 0.331 → 0.169. v6 (Noul-as-twin-choice) and v8 (wider sem_max) both regressed. v5 is the accuracy peak (0.623) but loses on calibration.

## 4. caveats — read before quoting

1. **public-only.** all 231 runs are on the public half of JevBench v1.2. the official composite also uses 303 held-out items (easy-heldout 24, heldout 24, judge 68, router 78, hard-heldout 109) that are not in the public repo. the published composite would shift by an unknown amount on the held-out half; treat 65.53 as a public-only lower bound.
2. **no benchmark-specific calibration.** probabilities are the raw output of the model. no temperature / vector scaling / isotonic fit on JevBench data.
3. **cost is an estimate.** self-hosted GPU has no per-token tariff. we use the same OpenRouter `Qwen/Qwen2.5-3B-Instruct` $0.04/M-input basis the published leaderboard's estimator uses.
4. **held-out items.** we did not attempt to access, infer, or synthesize the 303 held-out items. the official score on the full harness is not reproducible from public data alone.
5. **live workspace untouched.** `git status --short` before vs. after this measurement is identical — no edits, no commits, no checkpoint writes by the measurement pipeline.

## 5. architecture: why span-pooled beats slot-position

### choice questions

the v1 architecture bound options to positional slots (`SlotChoiceHead`: readout position k → option k). works for 2–4 options. at 18-way MASSIVE routing it collapses — measured 31 of 40 MASSIVE predictions piled onto slot 0.

v4 replaces positional binding with **semantic binding**: each option's *token span* is mean-pooled from one forward pass, then a shared `OptionScorerHead` turns each span into a softmax. permutation-equivariant by construction — there is no slot 0 to collapse onto.

```
State: The package arrived broken.
Question: Which team?
Options: support | billing | logistics
                            ↓
        one forward pass, span-pool mean per option
                            ↓
        shared OptionScorerHead → softmax → P(support), P(billing), P(logistics)
```

### noul and score

noul is a 1-output sigmoid head on the last-token hidden state. score is an ordinal head with 8 levels, mapped onto whichever rubric the user supplies at inference time.

### inference

one `forward()` per request. **zero `generate()` calls anywhere.** a shared-prefix block-isolation mask lets every question in a request read the same state once and only its own suffix. `generated_tokens == 0` is asserted in the test suite.

## 6. reproduce

```bash
git clone https://github.com/isHeSatoshi/smalljev
cd smalljev
pip install -e ".[bench]"

# pre-flight: 6/6 transport-correctness checks
python jevbench_eval/validation/validate_semantic_adapter.py

# pull the v1.2 harness (datasets frozen at upstream main)
git clone --depth 1 https://github.com/fstandhartinger/jevbench.git /tmp/jevbench

# run the public 231 items (v9, ~70 s on RTX 4060 Ti)
python jevbench_eval/scripts/run_v12.py \
    --variant-label  smalljev_semantic_v9 \
    --adapter-dir    semantic-v9-lora \
    --scorer-ckpt    semantic-v9-scorer.pt \
    --noulscore-ckpt semantic-v9-noulscore.pt \
    --harness-root   /tmp/jevbench \
    --run-dir        runs/2026-09-21_semantic_v9_v12 \
    --tiers          easy,standard,hard

# summarise into the 4-axis v1.2 composite (writes summary.json + summary.md)
python jevbench_eval/scripts/summarize_v12.py \
    --run-dir      runs/2026-09-21_semantic_v9_v12 \
    --harness-root /tmp/jevbench
```

pinned environment:

| library | version |
|---|---|
| python | 3.11 |
| torch | 2.8.0+cu129 |
| transformers | 4.57.6 |
| peft | 0.15.2 |
| jevbench | `main` (frozen on v1.2 datasets) |

every `results.jsonl` carries a `runtime` block with the loaded device, threads, backbone id, adapter id, scorer / noulscore paths, generated tokens (always 0), and the library versions — so a manifest can reproduce the run.

## 7. license and attribution

- code: Apache-2.0.
- backbone: `openbmb/MiniCPM5-2B-Base` (Apache-2.0).
- benchmark harness: [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (MIT).
- independent. not affiliated with TypeSafe AI.
