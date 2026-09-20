# Changelog

## 2026-09-20 — Initial public release

The first published version of smalljev. Ships the `semantic-v7` recipe and the JevBench v1.2.1 public-only measurement that puts it at rank 7 of 21 non-partial systems.

### What's in the box

- `semantic-v7` weights: MiniCPM5-2B-Base + LoRA + OptionScorerHead + BinaryNoulHead + OrdinalScoreHead.
- 49 unit tests covering the API, spec validation, calibration, parallel sampler, heads, training utilities.
- `evals/train_semantic.py` — the recipe to retrain from v7.
- `evals/eval_semantic.py` — the per-task nimble13 battery harness.
- `jevbench_eval/` — the official JevBench harness plus the smalljev adapter, with the public-only runs saved per tier and a `summary.json` per variant.
- `hf_space/` — the Gradio app backing the live HuggingFace Space.

### What we tried that didn't make it

| Version | Idea | Result | Verdict |
|---|---|---|---|
| v1 (pre-arm) | slot-position head | 19.7 JevBench, MASSIVE collapse | replaced |
| v3 | LoRA + joint training | 56.0 JevBench | superseded |
| v4 | first semantic head | 64.7 JevBench | shipped as baseline |
| v5 | episodic banking mix | 64.9 JevBench | superseded by v7 |
| v6 | Noul-as-twin-choice | 61.5 JevBench (-3.4) | regression — discarded |
| **v7** | **+ squad/boolq/vitaminc boost, 2 epochs** | **67.30 JevBench (rank 7)** | **shipped** |
| v8 | wider sem_max (1536→2048), v8 flag | 65.87 (-1.4 vs v7) | regression — discarded |

The `semantic-v7` weights stay as the recommended checkpoint. `semantic-v8` is left in the repo for reproducibility but is not the default and not on the leaderboard.

### Honest caveats baked in

- The JevBench number is **public-only**: 231 of 534 items. Held-out 303 items are not in the public JevBench repo. The official composite on the full harness is not reproducible from public data alone.
- The MASSIVE en/de numbers in nimble13 are **disjoint-split, not zero-shot**. The architectural contribution is the +0.18 vs zero-shot Laya; the larger delta is the train/test overlap by design.
- Cost is an **estimate**: OpenRouter `Qwen/Qwen2.5-3B-Instruct` $0.04/M-input × measured tokens.
- No benchmark-specific calibration (temperature / vector scaling / isotonic fit on JevBench data).
- Live workspace integrity was preserved during the measurement (`git status --short` identical before and after).
