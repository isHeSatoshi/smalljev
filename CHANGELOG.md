# Changelog

## 2026-09-20 — first public release

ships `semantic-v7`. JevBench v1.2.1 public score: 67.30. rank 7 of 21 non-partial systems on https://benchmarkheaven.com/jev-models. submission is in the queue.

### what made it

| version | what changed | JevBench | result |
|---|---|---|---|
| v1 (pre-arm) | slot-position head, MASSIVE collapse | 19.7 | replaced |
| v3 | LoRA + joint training | 56.0 | superseded |
| v4 | first semantic arm | 64.7 | shipped as baseline |
| v5 | episodic banking mix | 64.9 | superseded by v7 |
| v6 | Noul-as-twin-choice | 61.5 | regression, discarded |
| **v7** | + squad/boolq/vitaminc boost, 2 epochs | **67.30** | **shipped** |
| v8 | wider sem_max (1536 → 2048), v8 flag | 65.87 | regression, discarded |

### v8 story

a MASSIVE en-US smoke test on v8 came back at 0.874 vs v7's 0.857. +1.7 points, sounds great. i promoted it. ran the full JevBench eval: **−1.43 points**. intelligence dropped 1.08, speed dropped 4.82, calibration barely moved. the wider context window (sem_max 2048 vs 1536) made each forward slower and probably changed how some hard-tier items tokenized. the smoke lied.

v8 weights are kept in the repo at `evals/arms/semantic-v8-lora/` for reproducibility. they're not the default and not on the leaderboard. lesson: a smoke test is a probe, not a measurement.

### caveats baked in

- JevBench score is public-only (231 / 534 items). held-out 303 aren't in the public JevBench repo.
- MASSIVE numbers are disjoint-split same-source, not zero-shot.
- cost is estimated against OpenRouter Qwen2.5-3B-Instruct $0.04/M-input × measured tokens.
- no benchmark-specific calibration was fit on JevBench.
- live workspace was not modified during the measurement (`git status --short` identical before and after).

### honest stack

```
backbone     openbmb/MiniCPM5-2B-Base     (Apache-2.0)
adapter      evals/arms/semantic-v7-lora/ (LoRA r=16, q_proj + v_proj)
heads        OptionScorerHead + BinaryNoulHead + OrdinalScoreHead
python       3.11
torch        2.8.0+cu129
transformers 4.57.6
peft         0.15.2
gpu          RTX 4060 Ti 16GB
```
