# smalljev_semantic_v7 — v1.2 local summary

harness: `D:\Project\smalljev\jevbench_latest`
n_decisions: 231

| tier | n | correct | accuracy |
|---|---|---|---|
| easy | 48 | 47 | 97.92 |
| standard | 72 | 50 | 69.44 |
| judge | 0 | 0 | n/a |
| hard | 111 | 37 | 33.33 |

**Intelligence**: 59.93
**Calibration**: 60.69  (Brier=0.5523, ECE=0.1776, TVD=0.4311)
**Speed**: 81.91  (p50_adj=297.9ms, p95_adj=2159.9ms)
**Cost**: 58.90  ($0.0234 per 1k decisions; basis: reference_tariff_0.04_per_m_input)

## JevBench Score: **64.72**

Caveats:
- Public-only run (231 items: easy 48 + standard 72 + hard 111).
- v1.2.7 leaderboard uses 534 items including judge (146) + easy-heldout (24) + private hard (109).
- Cost basis: local GPU, no provider tariff. Estimated against OpenRouter Qwen/Qwen2.5-3B-Instruct $0.04/M-input × measured input tokens, the same estimator the public leaderboard uses for self-hosted systems without their own tariff.
- Speed is adjusted (×2 +0.15s) because the harness runs self-hosted.