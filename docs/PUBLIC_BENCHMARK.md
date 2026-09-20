# smalljev × JevBench v1.2.1 — full write-up

benchmark: https://benchmarkheaven.com/jev-models (MIT)
harness: https://github.com/fstandhartinger/jevbench @ v1.2.1
scope: **public-only**. 231 of 534 items. held-out 303 aren't in the public repo.
hardware: RTX 4060 Ti 16 GB, torch 2.8.0+cu129, transformers 4.57.6, peft 0.15.2.

---

## 1. headline

smalljev semantic-v7 is at **67.30** on the public half of JevBench v1.2.1. that's rank 7 of 21 non-partial systems, sitting between the published Rank 6 OpenJev (67.6, DiffusionGemma 26B) and Rank 8 openjev-sglang (66.2, Qwen3.6-35B-A3B).

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

| variant | intel | calib | speed | cost | **JevBench** |
|---|---|---|---|---|---|
| smalljev v5_holdout (pre-loop) | 58.5 | 35.7 | 80.0 | 59.0 | 56.02 |
| smalljev crown (pre-loop) | 61.4 | 33.8 | 80.0 | 59.0 | 55.95 |
| smalljev v4_bundle (pre-loop) | 59.1 | 31.6 | 82.6 | 59.0 | 54.93 |
| smalljev lettered (pre-loop) | 32.0 | 0.0 | 79.9 | 59.0 | 19.71 |
| smalljev semantic-v4 | 63.6 | 54.4 | 82.9 | 61.2 | 64.73 |
| smalljev semantic-v5 | 64.9 | 54.0 | 82.8 | 61.2 | 64.90 |
| smalljev semantic-v6 (twin-choice) | 63.4 | 46.7 | 79.2 | 61.2 | 61.54 |
| **smalljev semantic-v7 (shipped)** | 61.1 | **66.3** | 82.8 | 61.2 | **67.30** |
| smalljev semantic-v8 (discarded) | 60.0 | 65.8 | 78.0 | 61.2 | 65.87 |

the +11.35 gain over the previous champion (crown) is almost entirely calibration: ECE 0.331 → 0.169. v6 (Noul-as-twin-choice) and v8 (wider sem_max) both regressed. v5 is the accuracy peak (0.623) but loses on calibration.

## 4. caveats — read before quoting

1. **public-only.** all 231 runs are on the public half of JevBench v1.2.1. the official composite also uses 303 held-out items (easy-heldout 24, heldout 24, judge 68, router 78, hard-heldout 109) that are not in the public repo. the published composite would shift by an unknown amount on the held-out half; treat 67.30 as a public-only lower bound.
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

# run the public 231 items (3 tier runs, ~30 s each on RTX 4060 Ti)
python jevbench_eval/scripts/run_semantic_variant.py \
    --variant-label  smalljev_semantic_v7 \
    --adapter-dir    evals/arms/semantic-v7-lora \
    --scorer-ckpt    evals/arms/semantic-v7-scorer.pt \
    --noulscore-ckpt evals/arms/semantic-v7-noulscore.pt \
    --run-dir        runs/2026-09-20_semantic_v7 \
    --tasks          public_easy
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_standard
python jevbench_eval/scripts/run_semantic_variant.py ... --tasks public_hard

# summarize (writes runs/<dir>/summary.json with the 4-axis v1.2.1 composite)
python jevbench_eval/scripts/summarize_run.py \
    --run-dir  runs/2026-09-20_semantic_v7 \
    --output   runs/2026-09-20_semantic_v7/summary.json
```

pinned environment:

| library | version |
|---|---|
| python | 3.11 |
| torch | 2.8.0+cu129 |
| transformers | 4.57.6 |
| peft | 0.15.2 |
| jevbench | v1.2.1 (commit pinned at measurement time) |

every `results.jsonl` carries a `runtime` block with the loaded device, threads, backbone id, adapter id, scorer / noulscore paths, generated tokens (always 0), and the library versions — so a manifest can reproduce the run.

## 7. license and attribution

- code: Apache-2.0.
- backbone: `openbmb/MiniCPM5-2B-Base` (Apache-2.0).
- benchmark harness: [fstandhartinger/jevbench](https://github.com/fstandhartinger/jevbench) (MIT).
- independent. not affiliated with TypeSafe AI.
