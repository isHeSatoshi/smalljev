# smalljev × JevBench — external evaluation

**Independent external evaluation of the existing smalljev implementation against the public JevBench benchmark.**

The live smalljev workspace at `D:\Project\smalljev` was not modified by this evaluation. All artifacts live in `C:\Users\Aditya\AppData\Local\Temp\smalljev_jevbench_eval\` (outside the project).

## Headline numbers (PUBLIC-ONLY JevBench v1.2.1, 231 items)

| Variant | Items | Acc | Brier | ECE | Macro | Schema | Speed (adj, public std) | Cost/1k (est.) | **JevBench Score (public-only)** |
|---|---|---|---|---|---|---|---|---|---|
| `v5-holdout` | 231/231 | 0.584 | 0.707 | **0.322** | 0.567 | 1.000 | p50 0.42 s, p95 0.43 s | $0.0232 | **56.02** |
| `crown` | 231/231 | **0.593** | **0.704** | 0.331 | **0.585** | 1.000 | p50 0.42 s, p95 0.43 s | $0.0232 | **55.95** |
| `v4-bundle` | 231/231 | 0.567 | 0.741 | 0.342 | 0.541 | 1.000 | **p50 0.28 s, p95 0.32 s** | $0.0232 | **54.93** |
| `lettered` | 231/231 | 0.320 | 1.173 | 0.555 | 0.316 | 1.000 | p50 0.42 s, p95 0.44 s | $0.0232 | 19.71 |

Full per-tier, per-family, calibration, latency, cost, error-analysis, and reference-row comparison: see [`jevbench_public_report.md`](jevbench_public_report.md).

## Layout

```
smalljev_jevbench_eval/
    jevbench/                           # clone of fstandhartinger/jevbench@v1.2.1
    adapter/
        smalljev_adapter.py             # JevBench adapter for smalljev (native)
    scripts/
        _fcntl_shim.py                  # Windows fcntl no-op shim (single-process)
        run_variant.py                  # Run one variant × all 3 public tiers
        summarize_run.py                # Per-variant axes + JevBench Score
        compare_variants.py             # Cross-variant aggregate
        collect_error_examples.py       # Per-tier failure examples per variant
        print_digest.py                 # One-shot summary printer
    validation/
        validate_adapter.py             # 36-task pre-flight (transport only)
    runs/
        2026-09-19_v4_bundle/           # one directory per variant
            easy/standard/hard/{results.jsonl, raw/, manifest.json}
            summary.json
        2026-09-19_v5_holdout/
        2026-09-19_crown/
        2026-09-19_lettered/
        ALL_VARIANTS_COMPARISON.{json,md}
        ERROR_ANALYSIS.json
    jevbench_public_report.md           # Full report (this is the entry point)
```

## Reproducing

```bash
# from this directory, with SMALLJEV_PATH pointing at the live smalljev checkout

export SMALLJEV_PATH=D:/Project/smalljev

python validation/validate_adapter.py
#   (uses the v4-bundle variant by default; 36 / 36 transport-valid)

python scripts/run_variant.py \
    --variant-label smalljev-v4-bundle \
    --adapter-id  D:/Project/smalljev/evals/heads/v4-bundle-lora \
    --heads-ckpt  D:/Project/smalljev/evals/heads/v4-bundle.pt \
    --run-dir     runs/2026-09-19_v4_bundle \
    --tasks       public_easy
python scripts/run_variant.py ... --tasks public_standard
python scripts/run_variant.py ... --tasks public_hard
python scripts/summarize_run.py --run-dir runs/2026-09-19_v4_bundle \
    --output runs/2026-09-19_v4_bundle/summary.json

# Repeat for each variant. Then:
python scripts/compare_variants.py
```

## Workspace integrity

- `git -C D:/Project/smalljev status --short` before and after this evaluation: identical (master branch, no commits).
- No file in `D:/Project/smalljev` was added, removed, edited, or renamed by this evaluation.
- No checkpoint was loaded for writing — every checkpoint was opened read-only by the smalljev `HFBackend.load()` path.
- All artifacts produced by this evaluation are under `C:\Users\Aditya\AppData\Local\Temp\smalljev_jevbench_eval\`.

## Limitations

- Held-out items (109 hard + 24 standard + 24 easy + 146 judge = 303) are not accessible; only the 231-item public half was run. The composite JevBench Score is therefore a **public-only lower bound**, not the official Score.
- The "Speed" axis per v1.1.2 uses a 242-item standard+judge run; only 72 of those 242 items are public. Latency above is measured on the 72-item public standard subset.
- Cost is an ESTIMATE basis (OpenRouter `Qwen/Qwen2.5-3B-Instruct` $0.04 / M input, $0 / M output) because self-hosting has no provider tariff; the estimate basis is documented per-record.
- smalljev was not optimised against JevBench — no calibration fit, no benchmark-specific prompts, no LoRA training. This is a clean baseline.