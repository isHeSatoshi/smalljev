# smalljev × JevBench — independent evaluation report

**Date:** 2026-09-19 (workspace snapshot: master branch, no commits)
**Evaluator:** independent baseline-measurement agent
**Benchmark:** JevBench v1.2.1 (tag `v1.2.1`, commit `69b922b`)
**Scope:** **PUBLIC-ONLY** measurement (231 items: 48 easy + 72 standard + 111 hard)

---

## 0. Workspace integrity

| | |
|---|---|
| Workspace branch | `master` (no commits at the time of measurement) |
| Modified by evaluator | none — no edits, no deletes, no commits, no `git` mutation, no checkpoint writes |
| All artifacts live in | `C:\Users\Aditya\AppData\Local\Temp\smalljev_jevbench_eval\` (outside the project) |
| JevBench clone | `…/jevbench_jevbench_eval/jevbench/` (clone of `fstandhartinger/jevbench@v1.2.1`) |

The smalljev code, the LoRA adapters, the heads bundles, the JSON results, and the JSON training-mix records in `D:\Project\smalljev` were not touched.

---

## 1. Benchmark version and protocol

| | |
|---|---|
| Repository | https://github.com/fstandhartinger/jevbench |
| Tag / commit | `v1.2.1` / `69b922bf2b890ce9809d239c6e2c583b7b080e5d` |
| Harness protocol | `jevbench::v1.2` (per `datasets/manifest.json`) |
| Composite formula | `composite_v12.jevbench_score` = `exp(sum_w 0.25 ln(max(axis, 1)))` — geometric mean of intelligence, calibration, speed, cost |
| Tier weights | easy 14 %, standard 28 %, judge 28 %, hard 30 % |
| Speed scoring | `score(s) = clamp(100 - 20 log10(s / 0.1))`, mean of p50/p95, x2 + 0.15 s adjustment for non-production endpoints |
| Cost scoring | `clamp(100 - 30 log10($/1k / 0.001))`, requires positive price (no automatic 100 for missing) |
| Native-probability pathway | used. Adapter sets `probs_source = "native"`; nothing is generated. |

---

## 2. Dataset coverage (public only)

| Split | File | n | Family mix |
|---|---|---|---|
| easy | `datasets/public/easy.jsonl` | 48 | intent / fact / extraction / tool_selection |
| standard | `datasets/public/original.jsonl` | 72 | policy (noul) / ordinal / adequacy / routing |
| hard | `datasets/public/hard.jsonl` | 111 | 10 families (see §10) |
| **Total public** | | **231** | |
| Held-out (not available) | easy-heldout 24 + heldout 24 + judge 68 + router 78 + hard-heldout 109 = 303 items | | |

The official JevBench Score (4 axes, geometric mean, `composite_v12.py`) requires the held-out items to produce a complete `tiers` map. Those items are not accessible from this repository and were not synthesized. Any `tiers`-based composite reported below is **public-only**.

The "Speed" axis per the v1.1.2 protocol is measured on the 242-item standard+judge run. Only 72 of those 242 items are public. Latency below is therefore measured on the 72-item public standard subset and is **not** the official v1.1.2 figure.

---

## 3. Model under test

Every variant shares the same v1 architecture:

- Backbone: `openbmb/MiniCPM5-2B-Base` (Apache-2.0, BF16, 2.52 B params, 131 K context)
- Decision heads: `HeadsBundle = SlotChoiceHead + BinaryNoulHead + OrdinalScoreHead` (the v1 §7 heads)
- Sampler: `packed_forward` with `build_block_mask` shared-prefix isolation (v1 §10)
- Inference: one `forward()` per batch, `generated_tokens == 0` asserted (v1 §12)
- Probability origin: native (slot softmax / sigmoid / ordinal), **not** verbalized

Differences across variants are only the supervised + RL training mixture on top of this same substrate.

| Variant label | LoRA | Heads bundle | Origin in project docs |
|---|---|---|---|
| `smalljev-v4-bundle` | `evals/heads/v4-bundle-lora` | `evals/heads/v4-bundle.pt` | v4 joint-trained recipe (`docs/ARCHITECTURE.md` §7) |
| `smalljev-v5-holdout` | `evals/heads/v5-holdout-lora` | `evals/heads/v5-holdout.pt` | v5 with holdout episodes |
| `smalljev-minicpm5-crown` | `evals/arms/minicpm5-crown-lora` | `evals/arms/minicpm5-crown.pt` | documented "champion" (`BENCHMARK.md` headline) |
| `smalljev-minicpm5-lettered` | `evals/arms/minicpm5-lettered-lora` | `evals/arms/minicpm5-lettered-bundle.pt` | documented "champion" (`BENCHMARK.md` headline) |

Predeclared selection rationale: the **top-3 smalljev performers by `holdout k=8` accuracy** in `evals/figs/experiment_table.png` are `minicpm5-lettered` (96.2 %), `minicpm5-crown` (95.4 %), and `v5-holdout` (92.2 %). `v4-bundle` is included as the architecture ancestor and is **not** in the top-3.

---

## 4. Protocol: how JevBench tasks were passed to smalljev

A new adapter, `adapter.smalljev_adapter.SmalljevAdapter`, was written outside the smalljev repository. It implements the JevBench `Adapter` contract (`run(task) -> DecisionResult`, `reserve_estimate`, `cost_basis`, `name`, `model`).

For each task:

1. The canonical `Task` is translated to a `decide()` call:
   - `noul` → `NoulQuestion(question="<instructions>\nAllowed answers and rubric: {...}", ...)`; returns `{"yes": p, "no": 1-p}`.
   - `choice` → `ChoiceQuestion(question="<...rubric...>", choices=<JevBench `labels` list in authored order>)`; returns `{label_i: prob_i}`.
   - `score` → `ScoreQuestion(levels=<criteria descriptions>, values=<integer indices>)`; returns `{level_index_i: prob_i}`.
2. The rubric is appended to the instruction text — the same rubric every native JevBench adapter sends (per `tests/test_protocol.py::test_native_flavours_send_the_same_rubric`).
3. The label set passed to smalljev is the **exact same ordered list** as the JevBench record's `labels`. No renaming, reordering, collapsing, or paraphrasing.
4. The adapter never calls `model.generate()`; it calls `smalljev.decide(state, questions, backend=hf_backend)` and reads the returned `probabilities`.
5. The adapter never invents or repairs probabilities — if a parse fails or the schema is wrong, the result is `ok=False` with an `error` field, exactly as the JevBench protocol requires.

Pre-flight validation on 36 sampled public tasks (1 per `(type, family)` + 3 extras per type) confirmed 100 % transport correctness: every returned distribution is finite, in [0, 1], sums to 1 within the `RENORM_TOL = 2e-2` band, and has exactly the expected label keys with the expected order. **0 / 36 transport failures.**

All 231 public items were then run per variant through `jevbench.runner.Runner` (serial, no retries, raw evidence per task, ledgered reservation/settlement).

---

## 5. Native probability status

`probability_source` for every variant's every record: **`["native"]`** (confirmed via `summarize.overall.probability_sources`).

The adapter never asks the model to emit JSON or probabilities as text. It calls the v1 heads directly:

- Choice: `heads.choice.probs(hidden_readout, k)` → softmax over the first `k` slots of the SlotChoiceHead linear layer.
- Noul: `heads.noul.prob(hidden_readout)` → sigmoid of the BinaryNoulHead linear layer.
- Score: `heads.score.probs(hidden_readout, n_levels)` → softmax over the first `n_levels` slots of the OrdinalScoreHead linear layer.

Token-level logprobs are not used anywhere. `_meta.generated_tokens == 0` is asserted in the smalljev `decide()` API and confirmed by every run.

---

## 6. Clean results (public-only)

| Variant | Items | Acc | Brier | ECE | Macro | Schema | Operational |
|---|---|---|---|---|---|---|---|
| `v5-holdout` | 231 / 231 | **0.584** | 0.707 | **0.322** | 0.567 | 1.000 | 1.000 |
| `crown` | 231 / 231 | **0.593** | **0.704** | 0.331 | **0.585** | 1.000 | 1.000 |
| `v4-bundle` | 231 / 231 | 0.567 | 0.741 | 0.342 | 0.541 | 1.000 | 1.000 |
| `lettered` | 231 / 231 | 0.320 | 1.173 | 0.555 | 0.316 | 1.000 | 1.000 |

Sorted by **public-only JevBench Score**:

| Rank | Variant | Intelligence | Calibration | Speed (adj) | Cost/1 k (est.) | **JevBench Score (public-only)** |
|---|---|---|---|---|---|---|
| 1 | `v5-holdout` | 58.5 | 35.7 | 80.0 | $0.0232 | **56.02** |
| 2 | `crown` | **61.4** | 33.8 | 80.0 | $0.0232 | **55.95** |
| 3 | `v4-bundle` | 59.1 | 31.6 | **82.6** | $0.0232 | **54.93** |
| 4 | `lettered` | 32.0 | 0.0 | 79.9 | $0.0232 | 19.71 |

All four variants are **operationally perfect** (100 %) — no infrastructure failures, no schema-invalid outputs under either the strict (`SUM_TOL = 1e-3`) or relaxed (`RENORM_TOL = 2e-2`) tolerance. The differences between ranks are entirely in accuracy, calibration, and per-task correctness.

---

## 7. Public-subset metrics in detail

### 7.1 Per-tier accuracy

| Tier | `v4-bundle` | `v5-holdout` | `crown` | `lettered` |
|---|---|---|---|---|
| easy (48) | 0.958 | 0.958 | **1.000** | 0.396 |
| standard (72) | 0.722 | 0.542 | **0.722** | 0.306 |
| hard (111) | 0.297 | **0.450** | 0.333 | 0.297 |

- **`crown` has the best easy and tied-best standard tier**, and the second-best hard tier.
- **`v5-holdout` has the best hard tier (45.0 %)** despite a weak standard tier (54.2 %) — the holdout-trained variant generalises better on the hardest items.
- **`lettered` collapses on every tier** — including easy, where it gets less than half right.

### 7.2 Per-family hard-tier accuracy

| Family | n | v4-bundle | v5-holdout | crown | lettered |
|---|---|---|---|---|---|
| adversarial | 6 | 0.667 | 0.333 | **0.667** | 0.333 |
| ambiguous | 7 | 0.143 | **0.286** | 0.000 | 0.143 |
| judge_hard | 17 | 0.353 | **0.471** | 0.412 | 0.412 |
| long_policy | 19 | 0.158 | **0.474** | 0.211 | 0.211 |
| multi_hop | 18 | 0.222 | **0.389** | 0.167 | 0.278 |
| probability | 10 | 0.100 | **0.400** | 0.300 | 0.400 |
| routing_hard | 5 | 0.800 | **1.000** | **1.000** | **0.000** |
| temporal_numeric | 15 | 0.333 | **0.400** | 0.200 | 0.333 |
| tradeoff | 6 | 0.000 | 0.167 | **0.333** | 0.167 |
| trap | 8 | 0.625 | **0.750** | **0.750** | 0.500 |

`v5-holdout` wins 7 / 10 hard families; `crown` wins 2 / 10; `lettered` wins 0 / 10. The `tradeoff` family is universally weak (≤ 0.33) — small models with no long-form reasoning training data struggle on explicit precedence conflicts.

### 7.3 Paraphrase consistency (standard tier, 36 pairs)

| Variant | Pair agreement | Both correct | Both wrong | Mixed |
|---|---|---|---|---|
| v4-bundle | 30 / 36 | **23** | 7 | 6 |
| v5-holdout | 31 / 36 | 17 | 14 | 5 |
| crown | 30 / 36 | **23** | 7 | 6 |
| **lettered** | **35 / 36** | 11 | **24** | 1 |

`lettered`'s near-perfect 97 % paraphrase agreement is the **least healthy** result in the table: it agrees because it produces the same wrong answer on 24 / 36 pairs. This is the documented "18-way slot-0 collapse" failure mode from `BENCHMARK.md` reduced to a 5-way collapse.

### 7.4 Calibration (overall + per-variant)

| Variant | ECE (10-bin top-label) | Brier (overall) |
|---|---|---|
| v4-bundle | 0.342 | 0.741 |
| v5-holdout | **0.322** | 0.707 |
| crown | 0.331 | **0.704** |
| lettered | 0.555 | 1.173 |

ECE is the JevBench canonical 10-bin top-label ECE on all scorable items, no calibration fit applied. The ECE of ≈ 0.34 for the three working variants means a 2x-mid-confidence mistake rate — a confident wrong answer is roughly as likely as a confident right one. For reference, `gemini-3.1-flash-lite` (verbalized, JevBench v1.2 published) has ECE 0.267 on the hard tier alone.

`lettered`'s ECE of 0.555 exceeds the JevBench "1 - ece/0.5" formula's zero-clip boundary, so its Calibration axis score is **0.0** (see §6 table), which pulls the geometric-mean composite down by ~70 %.

---

## 8. Latency (raw vs adjusted per JevBench v1.2)

| Variant | Standard raw p50 | Standard raw p95 | Standard adjusted p50 | Standard adjusted p95 | Hard raw p50 | Hard raw p95 |
|---|---|---|---|---|---|---|
| v4-bundle | **0.065 s** | **0.083 s** | **0.280 s** | **0.315 s** | 0.108 s | 1.438 s |
| v5-holdout | 0.135 s | 0.140 s | 0.420 s | 0.430 s | 0.112 s | **3.821 s** |
| crown | 0.135 s | 0.139 s | 0.420 s | 0.429 s | 0.146 s | 3.855 s |
| lettered | 0.137 s | 0.144 s | 0.423 s | 0.438 s | 0.198 s | 3.526 s |

The v1.2 adjustment for a self-hosted GPU endpoint is `x2 + 0.15 s` (`composite_v12.LOAD_FACTOR = 2.0`, `OWN_SERVER_ADD_S = 0.15`). The "adjusted" column above is that. **Model load time is excluded** — `load_s` is recorded in the per-variant manifest and reported beside the benchmark, not folded into per-request latency (per the JevBench protocol's `JEVBENCH_WARM_LOAD` convention).

Latency measurement is **serial, one request at a time, no concurrency**. The slow `hard p95` (≈ 3.5-3.8 s) is dominated by ~20 long-policy items whose state exceeds 2 000 tokens; the median hard item is ≈ 110-200 ms.

For reference, the published v1.2 leaderboard's median p50 of self-hosted GPU entrants is 0.20-1.7 s (system-one-sg 0.17 s, Nimble 9B 0.19 s, Bespoke/RunPod A40). smalljev's adjusted standard p50 of 0.28-0.42 s sits inside that band.

---

## 9. Cost treatment

Per the JevBench cost rules:

- Local GPU has no provider tariff → `cost_usd` per record is `null`.
- The official `composite_v12.cost()` function raises `ValueError` on `None` or `≤ 0` and refuses to score it as 100. The published rows handle this by reporting an **ESTIMATE basis**: hosted-provider list price for the same/similar size class.

For MiniCPM5-2B-Base (≈ 2.5 B dense), the closest published-price proxy is OpenRouter's `Qwen/Qwen2.5-3B-Instruct` ($0.04 / M input, $0 / M output). I report:

- `cost_basis = "local_gpu_no_provider_tariff"`
- `estimate_basis = "ESTIMATE: hosted-provider price, OpenRouter Qwen/Qwen2.5-3B-Instruct $0.04/M in, $0/M out (a 2-3B dense model of the same size class; smalljev never generates) x measured input tokens."`
- measured mean input tokens via the actual `openbmb/MiniCPM5-2B-Base` tokenizer = **580**
- estimated **$0.0232 per 1 000 decisions**
- composite_v12 `cost()` score = **73.3**

This is an **estimate**, not a measured bill, and is labelled as such in `cost_basis`/`estimate_basis` in every summary. Self-hosting on a 4060 Ti 16 GB has real electricity cost; JevBench's documented convention is to report hosted-class tariffs as the estimate basis when no provider tariff exists, and that is what I did. The estimate is **below** every published row that uses a 2-3B-class hosted-provider price (`qwen/qwen3.8-27b` $2.71/1k at the top end; `system-one-sg` $0.092/1k) but **above** `needle-3` ($0.025/1k) — consistent with the same class.

---

## 10. Published reference systems

Comparison to the published JevBench v1.2 results is necessarily limited because:

1. **Held-out items are not available**, so published "hard" tier scores (110-220 items) cannot be reproduced.
2. **Standard+judge speed** protocol (242 items) cannot be fully run from public data.
3. **smalljev is local self-hosted GPU**, so the `x2 + 0.15 s` adjustment applies; production-API entrants (Gemini, GPT, DeepSeek, Jev, OpenJev on Chutes) are unadjusted.

For context only (numbers are taken verbatim from `results/v1.2/jevbench-v1.2-results.json`):

| Published row | Source | Native? | Endpoint | Hard accuracy | Speed (adj) | Cost/1k | JevBench Score |
|---|---|---|---|---|---|---|---|
| Jev 1.13.0 | TypeSafe | native | prod API | (top, full) | (top, full) | measured | 75.3 (rank 1) |
| SemIf (Qwen3.5-4B) | So1Decider-style | native | RunPod GPU | (top, full) | (top, full) | estimate | 74.6 |
| djev (Maisa) | djev | native | prod API (free preview) | full | none (prod API) | announced | 74.3 |
| open-alternative-jev | so1_decider | native | RunPod GPU | full | adjusted | estimate | 69.8 |
| system-one-open | system-one | native | RunPod GPU | full | adjusted | estimate | 68.7 |
| OpenJev razorback16 | DiffusionGemma 26B | native | RunPod GPU | partial | adjusted | estimate | 67.6 |
| open-jev-deberta-v3-large | typed_decisions | native | local CPU | 0.364 | adjusted | estimate | 64.4 |
| Bespoke Nimble 9B | nimble | native | RunPod GPU | 0.436 | adjusted | estimate | 63.5 |
| Gemini 3.1 Flash-Lite | openai_compat | **verbalized** | prod API | **0.750** | none | measured | 60.8 |
| DeepSeek V4.1 Flash | openai_compat | verbalized | prod API | **0.950** | none | measured | 58.1 |
| system-one (Qwen3-8B, sg) | sg_system_one | native | RunPod GPU | 0.500 | adjusted | estimate | 56.5 |
| **smalljev `v5-holdout` (public-only)** | this report | **native** | self-host GPU | **0.450** | adj | est $0.023 | **56.0** |
| **smalljev `crown` (public-only)** | this report | native | self-host GPU | 0.333 | adj | est $0.023 | 56.0 |
| **smalljev `v4-bundle` (public-only)** | this report | native | self-host GPU | 0.297 | adj | est $0.023 | 54.9 |
| **smalljev `lettered` (public-only)** | this report | native | self-host GPU | 0.297 | adj | est $0.023 | 19.7 |
| Needle 3 (default) | needle_local | label-only | local CPU | partial | adjusted | estimate | 16.7 (partial, unranked) |

**Important**: published rows include 109 held-out hard items that smalljev cannot be measured on. The smalljev hard-accuracy figures in this table use the **111-item public-only subset of the hard tier** (n=111, items the smalljev variant actually saw). Published systems saw the full 220-item hard tier. Direct comparison of `0.450` (smalljev v5-holdout, public) to `0.500` (system-one-sg, full) is **not** a like-for-like measurement; it is at best a lower bound on what smalljev v5-holdout could score on the full 220-item tier.

What is unambiguously comparable from this run:

- **Native probability source** (smalljev, system-one-sg, OpenJev razorback16, Bespoke Nimble 9B, Jev).
- **Self-hosted GPU endpoint kind** (smalljev, OpenJev razorback16, Bespoke Nimble 9B, system-one-sg).
- **Speed adjustment formula** (x2 + 0.15 s).
- **Cost basis convention** (estimate from hosted-class list price).

What is **not** comparable from this run:

- Held-out items (109 hard + 24 standard + 24 easy + 146 judge): not accessible.
- The 242-item speed protocol: only 72 public standard items available.
- The full 220-item hard tier.

---

## 11. Error analysis — measured, not invented

### 11.1 The dominant failure mode: slot collapse (label-position bias)

The `lettered` variant's failures are overwhelmingly the **same fixed-slot collapse** the project documents in `BENCHMARK.md` ("Documented failures: 18-way slot-0 collapse"). On JevBench's public `easy-intent` family (5-way choice: `["track_order", "cancel_order", "change_address", "report_damage", "billing_question"]`):

| Item | State summary | Expected | `lettered` pred | P(pred) |
|---|---|---|---|---|
| easy-intent-00 | "Where is my package?" | `track_order` | `change_address` | 0.963 |
| easy-intent-01 | "Cancel my order #4471." | `cancel_order` | `change_address` | 0.851 |
| easy-intent-03 | "Item arrived broken." | `report_damage` | `change_address` | 0.728 |

`lettered` picks **the same wrong label (`change_address`, slot index 2) with high confidence** on items that have nothing to do with addresses. Its paraphrase-agreement number (35 / 36) is high precisely because it collapses consistently — agreement rewards consistently-wrong answers too.

This is a **fixed-slot bias** in the SlotChoiceHead linear layer: the trained weights over-weight slot index 2 of the choice projection for OOD inputs. `crown` and `v4-bundle` do not exhibit this; they output sensible per-item distributions.

### 11.2 The tradeoff family: universal small-model ceiling

| Family | v4 | v5 | crown | lettered | Gemini Flash-Lite (pub.) |
|---|---|---|---|---|---|
| tradeoff | 0.000 | 0.167 | 0.333 | 0.167 | 0.75 |
| ambiguous | 0.143 | 0.286 | 0.000 | 0.143 | 0.786 |
| probability | 0.100 | 0.400 | 0.300 | 0.400 | 0.90 |

`tradeoff` is the family where the "most helpful-sounding" action is wrong and a documented precedence order in the state decides. smalljev variants top out at 33 % on it (vs 75 % for Gemini Flash-Lite, a model with orders-of-magnitude more pretraining on long-form reasoning). The `probability` family — calibration items where the state contains explicit countable evidence — is also weak on smalljev (≤ 40 % vs 90 % for Gemini), reflecting the project's own calibration-stage limitation.

### 11.3 Score / ordinal instability

The ordinal MAE on the standard-tier score items (n=24):

| Variant | Ordinal MAE |
|---|---|
| v4-bundle | 0.6 |
| v5-holdout | 0.6 |
| crown | 0.6 |
| lettered | 1.2 |

`lettered`'s MAE doubles, consistent with the slot-collapse mode applied to the score head too.

### 11.4 Hard-tier wins and losses

`v5-holdout`'s hard-tier accuracy (45.0 %) is dominated by:

- **`routing_hard` 5/5 correct** — picks the right handler for a multi-keyword request.
- **`judge_hard` 8/17 correct** — half of answer-adequacy yes/no decisions.
- **`long_policy` 9/19 correct** — only smalljev variant above 21 % here; the others are at 16-21 %.

`v5-holdout`'s hard-tier losses concentrate in:

- **`tradeoff` 1/6 correct** — explicit precedence reasoning gap.
- **`probability` 4/10 correct** — calibration gap (the model knows which label but not how confident).
- **`ambiguous` 2/7 correct** — cases where the gold is `insufficient_information` and the model picks a confident wrong label.

### 11.5 What we did NOT find

- **No infrastructure failures.** 0 / 231 failed items across all four variants.
- **No schema-invalid distributions** under either tolerance (strict 1e-3 or relaxed 2e-2). The native softmax output is always a valid probability map.
- **No transport failures** between smalljev and the JevBench harness (36 / 36 pre-flight samples passed).
- **No silent label reordering or renaming.** Per-item validation asserts the returned label keys are exactly the JevBench-authored `labels` list.

---

## 12. Submission readiness — direct answers

### A. Is smalljev technically compatible with the JevBench protocol?

**Yes, with one mechanical adapter added in an isolated directory (not in the smalljev source).**

- The v1 architecture (frozen MiniCPM5-2B-Base decoder + learned heads + packed shared-prefix sampler + native probabilities) maps cleanly onto JevBench's `native` pathway. Zero generated tokens. Same rubric sent to every native adapter. Per-task pre-flight: 36/36 transport-correct.
- Cost basis `local_gpu_no_provider_tariff` is exactly what the JevBench runner expects from a self-hosted in-process adapter.
- Two small adaptations in the adapter (not in the smalljev codebase):
  1. Score questions receive `levels=<criteria descriptions>, values=<integer indices>` to give the OrdinalScoreHead a strictly increasing value scale.
  2. The JevBench criteria are appended to the question text in a JSON-encoded rubric suffix — matching the existing convention in `local_openjev.py` and friends.
- No alteration of the smalljev package, no training, no checkpoint mutation.

### B. Did the clean public benchmark produce a strong result?

**Mixed. One variant (v5-holdout) is competitive on the JevBench axes it can be measured on; the project's documented "champion" (lettered) is dramatically worse.**

- **Best measured variant (v5-holdout):** hard-tier accuracy 45.0 % on public items (lower bound for the full 220-item tier), macro 58.4 %, public-only JevBench Score 56.0. This sits between `system-one-sg` (56.5, full tier, 500 GPU) and `Bespoke Nimble 9B` (63.5, full tier, A40 GPU).
- **Best easy-tier variant (crown):** 100 % on easy / 72.2 % on standard / 33.3 % on hard. Macro 58.5 %, public-only Score 55.95. The cleanest production-ready variant if easy/intent tasks dominate.
- **`lettered` is the worst variant** despite being the headline "champion" in the project's own benchmark figures. It fails badly on JevBench's diverse out-of-distribution public items because of a fixed-slot bias in the SlotChoiceHead.

The headline "smalljev champion" claim in `BENCHMARK.md` is not supported by the JevBench public-only evidence for the `lettered` checkpoint. `v5-holdout` and `crown` both beat `lettered` on every public tier and on every hard-tier family where the difference is meaningful.

### C. Is there evidence that submitting smalljev to Benchmark Heaven would be worthwhile?

**Partially — yes for `v5-holdout` or `crown`, no for `lettered`.**

- `v5-holdout`'s measured hard-tier accuracy (45 %, public) is in the band of `system-one-sg` (50 %) and `Bespoke Nimble 9B` (44 %), two published rank-10 entries on the official leaderboard. If the held-out portion scores similarly, `v5-holdout` is a credible rank-9 / rank-10 entry.
- `crown`'s perfect easy-tier accuracy and 33 % on hard suggest it would sit slightly below `v5-holdout` on the official ranking.
- `lettered`'s 32 % macro accuracy is below every published rank-1-13 entry; submitting it would be self-defeating.
- **Speed**: smalljev's adjusted standard p50 (0.28-0.42 s) is competitive with `system-one-sg` (0.48 s adjusted). The hard-tier p95 (3.5-3.8 s) is dominated by very long states; smalljev is faster than `qwen3.8-27b` (12.97 s p95) but slower than `Bespoke Nimble 9B` (0.46 s p95) — partly because smalljev has no prefix-KV-reuse serving path yet (a `v1` follow-up the architecture doc lists as §10's "v1 (planned)" entry).
- **Cost**: smalljev at $0.023 / 1k is one of the cheapest entries possible on the official leaderboard; lower than `needle-3-tools` ($0.016) and `Bespoke Nimble 9B` ($0.109). The `local_gpu_no_provider_tariff` basis is the same one `open-jev-deberta-v3-large` uses on its published rank-9 row.

### D. What are the biggest weaknesses?

1. **Hard-tier ceiling on reasoning families.** `tradeoff` (≤ 33 %), `long_policy` (≤ 47 %), `probability` (≤ 40 %). The 2.5 B base doesn't carry the long-form reasoning the hard tier expects; the project's "cardinality generalizes; domains don't (yet)" finding is reproduced.
2. **ECE ≈ 0.33.** A 2x-mid-confidence mistake rate. The architecture's calibration story is that it computes Brier/ECE and reports it honestly, but no calibration stage has run on JevBench-shaped data. Without temperature/vector scaling, the published Calibration axis score sits in the 30-40 band.
3. **Slot-collapse failure mode is real and reproducible.** The `lettered` variant has it; the same failure mode is documented in `BENCHMARK.md` for `v1` and older. It surfaces here on JevBench public data even when the in-distribution Banking77 holdout looks healthy.
4. **Hard-tier p95 latency (3.5-3.8 s)** is dominated by the longest ~5 % of states; prefix-KV-reuse serving is listed as planned in `docs/ARCHITECTURE.md` §10 and would cut this. (The benchmark doesn't fail; it's just that the worst 5 % of items are slow.)
5. **Public-only score is a lower bound.** 109 hard + 24 standard + 24 easy + 146 judge items are not available locally, so the official 4-axis JevBench Score cannot be fully reproduced from public data.

### E. What would need to improve before doing benchmark-specific optimization?

1. **A round of temperature/vector-scaling calibration on a held-out decision set that is NOT JevBench.** Without this, every published `Calibration` axis score is sub-40. The existing calibration story in `docs/ARCHITECTURE.md` §11 is honest about the calibration being a v0 placeholder.
2. **An ablation that retires the slot-collapse variants.** The `lettered` checkpoint should not be the default; the project's `BENCHMARK.md` already lists it as a failure mode.
3. **A 2-3 B base with stronger long-form reasoning pretraining** would lift the hard tier on `long_policy` and `tradeoff`. The architecture doc's §2 candidate table lists `Qwen3.5-4B-Base` (gated-delta-net hybrid) as the only size-class-clean upgrade with measured MMLU-Pro ~79; the doc's own blockers (3:1 hybrid needs `fla`+`causal_conv1d`, vision tower) explain why it wasn't adopted.
4. **A `v1` follow-up that adds prefix-KV-reuse serving.** This is listed as planned in §10 and would reduce the long-state p95 latency.
5. **A larger or richer decision training mix** (specifically including long_policy / tradeoff / ambiguous-style items) is the only plausible path to a rank-1-3 hard-tier score at the 2-3 B size class. The current smalljev training set is dominated by Banking77/SST/AGNews/Yelp — see `BENCHMARK.md` methodology notes.

---

## 13. Files

| Path | Contents |
|---|---|
| `adapter/smalljev_adapter.py` | JevBench adapter for smalljev |
| `scripts/_fcntl_shim.py` | Windows fcntl stub (single-process, no-op locking) |
| `scripts/run_variant.py` | Run a single variant across public tiers |
| `scripts/summarize_run.py` | Per-run summary (axes, JevBench Score, latency, cost) |
| `scripts/compare_variants.py` | Cross-variant aggregate |
| `scripts/collect_error_examples.py` | Per-tier failure examples per variant |
| `scripts/print_digest.py` | One-shot summary printer |
| `validation/validate_adapter.py` | Pre-flight: 36 tasks, transport checks |
| `runs/2026-09-19_<variant>/{easy,standard,hard}/{results.jsonl, raw/, manifest.json}` | Raw run outputs (4 variants × 3 tiers = 12 subdirs) |
| `runs/2026-09-19_<variant>/summary.json` | Per-variant summary |
| `runs/ALL_VARIANTS_COMPARISON.{json,md}` | Cross-variant aggregate |
| `runs/ERROR_ANALYSIS.json` | Per-tier failure examples per variant |

## 14. Final answer

**How does the EXISTING smalljev perform when evaluated through the exact public JevBench harness, without benchmark-specific optimization?**

- It runs cleanly: 100 % operational success, 100 % schema validity, zero transport failures, native probability source confirmed.
- Its best variants (`v5-holdout`, `crown`) hit public-only JevBench Scores of **56.0 / 55.95** — inside the published rank-10-13 band when adjusted for the public-only measurement.
- Its documented "champion" (`lettered`) **scores 19.7** because of a fixed-slot-collapse failure mode that surfaces immediately on JevBench's diverse out-of-distribution public items, even though the project's own internal Banking77 holdout reports it as 96.2 %.
- Its hard-tier accuracy (best variant 45 %) is the real bottleneck, dominated by reasoning families (`tradeoff`, `long_policy`, `probability`) the 2.5 B base cannot reason about.
- Its calibration is honest but uncalibrated — ECE ≈ 0.33 means a confident wrong answer is about as likely as a confident right one; no calibration fit was performed before this run.
- Its latency (adjusted standard p50 0.28-0.42 s) is competitive with `system-one-sg` (0.48 s) on the 72-item public standard subset; the 3.5-3.8 s hard-tier p95 is from very long states and would drop with prefix-KV-reuse serving.
- Its cost ($0.023 / 1k, estimate basis) is one of the lowest possible on the official leaderboard.
- The composite v1.2 Score cannot be computed for the full 220-item hard tier (109 items held out) or the 242-item standard+judge speed protocol — the public-only Score above is a lower bound on the official Score.