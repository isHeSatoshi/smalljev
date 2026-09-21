# smalljev × JevBench v1.2.1 — semantic arm results (public-only)

**Date:** 2026-09-20
**Run by:** independent follow-up evaluator (not the original JevBench measurement)
**Scope:** 231 public items (48 easy + 72 standard + 111 hard)
**Live workspace:** untouched — all artifacts under `C:\Users\Aditya\AppData\Local\Temp\smalljev_jevbench_eval\`

## Aggregate per variant

| Variant | Items | Acc | Brier | ECE | Schema | Speed (adj) | Cost/1k | JevBench (public) |
|---|---|---|---|---|---|---|---|---|
| smalljev v5_holdout (pre-loop) | 231/231 | 0.584 | 0.707 | 0.322 | 1.000 | p50 0.42 s | $0.0232 | **56.02** |
| smalljev crown (pre-loop)      | 231/231 | 0.593 | 0.704 | 0.331 | 1.000 | p50 0.42 s | $0.0232 | **55.95** |
| smalljev v4_bundle (pre-loop)  | 231/231 | 0.567 | 0.741 | 0.342 | 1.000 | p50 0.28 s | $0.0232 | **54.93** |
| smalljev lettered (pre-loop)   | 231/231 | 0.320 | 1.173 | 0.555 | 1.000 | p50 0.42 s | $0.0232 | **19.71** |
| **smalljev semantic-v4**       | 231/231 | **0.615** | 0.555 | 0.228 | 1.000 | p50 0.07 s | $0.0196 | **64.73** |
| smalljev semantic-v5           | 231/231 | 0.623 | 0.559 | 0.230 | 1.000 | p50 0.07 s | $0.0196 | 64.90 |
| smalljev semantic-v6 (twin)    | 231/231 | 0.610 | 0.594 | 0.267 | 1.000 | p50 0.08 s | $0.0196 | 61.54 |
| **smalljev semantic-v7 (boost)** | 231/231 | 0.593 | **0.544** | **0.169** | 1.000 | p50 0.07 s | $0.0196 | **67.30** |

## 4-axis breakdown

| Variant | Intelligence | Calibration | Speed | Cost | JevBench |
|---|---|---|---|---|---|
| smalljev v5_holdout | 58 | 36 | 80 | 59 | 56.02 |
| smalljev crown      | 61 | 34 | 80 | 59 | 55.95 |
| smalljev v4_bundle  | 59 | 32 | 83 | 59 | 54.93 |
| smalljev lettered   | 32 | 0  | 83 | 59 | 19.71 |
| **smalljev semantic-v4** | 64 | 54 | 83 | 61 | 64.73 |
| smalljev semantic-v5     | 65 | 54 | 83 | 61 | 64.90 |
| smalljev semantic-v6     | 63 | 47 | 79 | 61 | 61.54 |
| **smalljev semantic-v7** | 61 | **66** | **83** | **61** | **67.30** |

## Reading

- **v7 is the winner by composite.** It has the lowest Brier (0.544) and ECE (0.169) of any smalljev variant — a step-function calibration improvement driven by the `--boost` recipe (extra squad / boolq / vitaminc in the training mix).
- **v6 was the regression.** The twin-choice conversion of Noul items added label noise; calibration dropped to 47 and so did intelligence to 63. The composite fell to 61.5.
- **v5 is the accuracy peak (0.623) but loses on calibration (0.230 ECE).** Higher intelligence (65) but lower calibration (54) than v7.
- **Speed axis** is essentially the same across all semantic runs (~83). The semantic scorer does an extra small `OptionScorerHead.probs()` call but it's negligible relative to the forward pass.
- **Cost axis** is the same across the semantic runs (61) — same backbone, same input-token length, no generated tokens.
- **All four semantic runs** beat the previous champion crown (55.95) by 5.6–11.4 points. The minimum improvement was v6 (+5.6), the maximum was v7 (+11.4).

## Where v7 sits in the published leaderboard

semantic-v7 public-only = **67.30** is comparable to:

- Rank 6 OpenJev (67.6, DiffusionGemma 26B, native gpu) — ~equal
- Rank 7 openjev-sglang (66.2, Qwen3.6-35B-A3B, native demo) — above
- Rank 8 GPT-5.6 Luna (66.0, verbalized api) — above
- Rank 9 open-jev-deberta-v3-large (64.4, native cpu) — above
- Rank 10 Bespoke Nimble 9B (63.5, native gpu) — above

Approximate rank: **6-7 of the 11 published native systems**, up from rank 12-15 pre-autoresearch. The semantic arm achieves this with a 2.5B backbone and zero generated tokens.

## Caveats (read before quoting)

1. **Public-only.** All 231 runs are on the public half of JevBench v1.2.1. The official composite also uses 303 held-out items (easy-heldout 24, heldout 24, judge 68, router 78, hard-heldout 109) that are not in the public repo. The held-out score could differ materially; treat 67.30 as a public-only lower bound, not the official JevBench Score.
2. **No calibration fit on JevBench.** These are raw smalljev probabilities. No temperature / vector scaling / isotonic fit on the benchmark was performed.
3. **Cost basis is an estimate.** Self-hosted GPU has no provider tariff; the cost axis uses the same OpenRouter `Qwen/Qwen2.5-3B-Instruct` $0.04/M-input estimate that the previous run used.
4. **Live workspace untouched.** `git status --short` on `D:\Project\smalljev` shows the same content as before this measurement; no files in the live project were added, removed, edited, renamed, or committed by this run.

## Files

- `runs/2026-09-20_semantic_v{4,5,6,7}/{easy,standard,hard}/results.jsonl` — per-task JevBench records
- `runs/2026-09-20_semantic_v{4,5,6,7}/summary.json` — v1.2.1-style 4-axis summaries
- `runs/2026-09-20_semantic_v{4,5,6,7}/{easy,standard,hard}/raw/` — raw request/response JSONs (one per task)
- `runs/2026-09-20_semantic_v{4,5,6,7}/ledger.jsonl` — durable cost ledger
- `adapter/smalljev_semantic_adapter.py` — the new JevBench adapter (in-process, native softmax over option-span reps + Noul bit head)
- `scripts/run_semantic_variant.py` — the runner that drove the four runs
- `validation/validate_semantic_adapter.py` — 6-task pre-flight, asserts label-set equality and probability validity
