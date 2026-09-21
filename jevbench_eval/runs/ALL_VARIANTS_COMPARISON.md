# smalljev × JevBench — variant comparison (public-only)
All variants use the same v1 architecture (openbmb/MiniCPM5-2B-Base + HeadsBundle + LoRA); only the supervised/RL training mixture differs.
Sorted by **public-only JevBench Score** (geometric mean of 4 axes).

## Aggregate per variant
| Variant | Items | Acc | Brier | ECE | Macro | Schema | Speed (adj) | Cost/1000 | JevBench |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-19_v5_holdout | 231/231 | 0.584 | 0.707 | 0.322 | 0.567 | 1.000 | p50=0.420s p95=0.430s | $0.0232 | **56.02** |
| 2026-09-19_crown | 231/231 | 0.593 | 0.704 | 0.331 | 0.585 | 1.000 | p50=0.420s p95=0.429s | $0.0232 | **55.95** |
| 2026-09-19_v4_bundle | 231/231 | 0.567 | 0.741 | 0.342 | 0.541 | 1.000 | p50=0.280s p95=0.315s | $0.0232 | **54.93** |
| 2026-09-19_f2llm_4b | 231/231 | 0.372 | 1.118 | 0.512 | 0.369 | 1.000 | p50=0.418s p95=5.317s | $0.0232 | **19.80** |
| 2026-09-19_lettered | 231/231 | 0.320 | 1.173 | 0.555 | 0.316 | 1.000 | p50=0.423s p95=0.438s | $0.0232 | **19.71** |

## Per-tier accuracy
| Tier | Variant | Accuracy | Brier | Schema |
|---|---|---|---|---|
| easy | 2026-09-19_crown | 1.000 | 0.000 | 1.000 |
| easy | 2026-09-19_f2llm_4b | 0.438 | 1.030 | 1.000 |
| easy | 2026-09-19_lettered | 0.396 | 1.069 | 1.000 |
| easy | 2026-09-19_v4_bundle | 0.958 | 0.055 | 1.000 |
| easy | 2026-09-19_v5_holdout | 0.958 | 0.080 | 1.000 |
| standard | 2026-09-19_crown | 0.722 | 0.511 | 1.000 |
| standard | 2026-09-19_f2llm_4b | 0.375 | 1.138 | 1.000 |
| standard | 2026-09-19_lettered | 0.306 | 1.154 | 1.000 |
| standard | 2026-09-19_v4_bundle | 0.722 | 0.526 | 1.000 |
| standard | 2026-09-19_v5_holdout | 0.542 | 0.824 | 1.000 |
| hard | 2026-09-19_crown | 0.333 | 1.133 | 1.000 |
| hard | 2026-09-19_f2llm_4b | 0.342 | 1.143 | 1.000 |
| hard | 2026-09-19_lettered | 0.297 | 1.231 | 1.000 |
| hard | 2026-09-19_v4_bundle | 0.297 | 1.176 | 1.000 |
| hard | 2026-09-19_v5_holdout | 0.450 | 0.902 | 1.000 |

## Per-family hard-tier accuracy (n=111)
| Family | n | 2026-09-19_crown | 2026-09-19_f2llm_4b | 2026-09-19_lettered | 2026-09-19_v4_bundle | 2026-09-19_v5_holdout |
|---|---|---|---|---|---|---|
| adversarial | 6 | 0.667 | 0.667 | 0.333 | 0.667 | 0.333 |
| ambiguous | 7 | 0.000 | 0.000 | 0.143 | 0.143 | 0.286 |
| judge_hard | 17 | 0.412 | 0.412 | 0.412 | 0.353 | 0.471 |
| long_policy | 19 | 0.211 | 0.421 | 0.211 | 0.158 | 0.474 |
| multi_hop | 18 | 0.167 | 0.222 | 0.278 | 0.222 | 0.389 |
| probability | 10 | 0.300 | 0.400 | 0.400 | 0.100 | 0.400 |
| routing_hard | 5 | 1.000 | 0.000 | 0.000 | 0.800 | 1.000 |
| temporal_numeric | 15 | 0.200 | 0.400 | 0.333 | 0.333 | 0.400 |
| tradeoff | 6 | 0.333 | 0.333 | 0.167 | 0.000 | 0.167 |
| trap | 8 | 0.750 | 0.375 | 0.500 | 0.625 | 0.750 |
