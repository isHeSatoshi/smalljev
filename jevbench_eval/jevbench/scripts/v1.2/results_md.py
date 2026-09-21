"""RESULTS-v1.2.md from results/v1.2/jevbench-v1.2-results.json.   python3 scripts/v1.2/results_md.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "results/v1.2/jevbench-v1.2-results.json").read_text())
S = R["systems"]
ranked = [s for s in S if s["ranked"]]
partial = [s for s in S if s["partial"]]
f1 = lambda v: "—" if v is None else f"{v:.1f}"
pct = lambda v: "—" if v is None else f"{100 * v:.1f} %"
sec = lambda v: "—" if v is None else f"{v:.2f} s"
SHORT = {
    "jev-1.13.0": "Jev 1.13.0", "gpt-5.6-luna": "GPT-5.6 Luna (low)", "deepseek-flash": "DeepSeek V4.1 Flash",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "needle-3": "Needle 3", "needle-3-tools": "Needle 3, options as tools",
    "qwen3.8-27b": "Qwen3.8 27B", "semif-qwen3.5-4b": "SemIf (Qwen3.5-4B)", "openjev-razorback16": "OpenJev razorback16 (DiffusionGemma 26B)",
    "system-one-sg": "system-one (Qwen3-8B, Goedecke)", "nimble-9b": "Bespoke Nimble 9B", "open-alternative-jev": "open-alternative-jev (Qwen3.5-4B, IkerMoel)",
    "djev": "djev (Maisa, diffusion-gemma)",
}
name = lambda s: SHORT.get(s["key"], s["display"])


def usd(s):
    v, kind = s["cost"]["usd_per_1000"], s["cost"]["kind"]
    return f"${v:.4f}" + {"estimate": " est.", "announced": " (announced price, free preview)"}.get(kind, "")


def table(rows, ranked_rows=True):
    out = ["| # | System | **JevBench Score** | Intelligence | Calibration | Speed | Cost | $ / 1,000 | p50 raw → adjusted | Endpoint |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        a, sp = s["axes"], s["speed"]
        lat = sec(sp["p50_s_raw"]) + ("" if sp["p50_s_adjusted"] == sp["p50_s_raw"] else f" → {sec(sp['p50_s_adjusted'])}")
        out.append(f"| {s['rank'] or ''} | {name(s)}{' (partial run)' if s['partial'] else ''} | **{f1(s['jevbench_score'])}** | {f1(a['intelligence'])} | "
                   f"{f1(a['calibration']) if a['calibration'] is not None else 'none (label only)'} | {f1(a['speed'])} | {f1(a['cost'])} | {usd(s)} | {lat} | {s['endpoint_condition']} |")
    return "\n".join(out)


presets = list(R["presets"])
jev = next(s for s in ranked if s["key"] == "jev-1.13.0")
second = ranked[1]
log = "\n".join(f"- **{e['revision']}** ({e['date']}): {e['note']}" for e in R.get("revision_log", []))
md = f"""# JevBench {R['revision']} — results

**JevBench Score** = {R['score_one_liner']}

Artifact: [`results/v1.2/jevbench-v1.2-results.json`](results/v1.2/jevbench-v1.2-results.json) · scoring code:
[`jevbench/composite_v12.py`](jevbench/composite_v12.py) · built by [`scripts/v1.2/finalize.py`](scripts/v1.2/finalize.py) from the
v1.2-wip measurements (tag `v1.2-wip`; no measurement changed; v1.2.1 adds djev, measured later on the same frozen items) · interactive page: [benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models)

![JevBench Score](results/v1.2/charts/main-score.png)

> **Speed note.** {R['speed_note']}

## Ranking

{table(ranked)}

{name(jev)} is #1 with {f1(jev['jevbench_score'])}; {name(second)} is #2, {f1(round(jev['jevbench_score'], 1) - round(second['jevbench_score'], 1))} points behind (difference of the rounded scores).

**Partial runs** — shown, not ranked (a tier attempted for fewer than 95 % of its decisions):

{table(partial, False)}

Footnote — djev: {R['footnotes'].get('djev', '')}

Footnote — open-alternative-jev: {R['footnotes']['open-alternative-jev']} The ranked row uses the author's own order
(`A. yes, B. no`, as his `yes_no()` helper builds it); the reversed-order run was our adapter's mistake and is kept only as raw
files (`results/v1.2/wip/`, GPU round runs).

## How the score works

| Axis | Definition |
|---|---|
| **Intelligence** | {R['scoring']['intelligence']} |
| **Calibration** | {R['scoring']['calibration']} |
| **Speed** | {R['scoring']['speed']} |
| **Cost** | {R['scoring']['cost']} |
| **JevBench Score** | {R['scoring']['jevbench_score']} |

![The four axes](results/v1.2/charts/axes.png)

## Other views (not the JevBench Score)

{R['scoring']['presets']} Weights are Intelligence : Calibration : Speed : Cost.

| System | """ + " | ".join(f"{p} ({':'.join(str(round(100 * w)) for w in R['presets'][p].values())})" for p in presets) + " |\n|---|" + "---|" * len(presets) + "\n" + "\n".join(
    f"| {name(s)} | " + " | ".join(f"#{s['rank_under'][p]} {f1(s['presets'][p])}" for p in presets) + " |" for s in ranked) + f"""

## Hard tier

![Hard tier accuracy](results/v1.2/charts/hard-tier.png)
![Hard tier by family](results/v1.2/charts/hard-families.png)
![Calibration](results/v1.2/charts/calibration.png)

{R['scoring']['hard_tier']}

## Tier accuracies and raw latency

| System | easy | standard | judge | hard | p50 raw | p95 raw | hard-tier p50 | Adjustment |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join(f"| {name(s)} | {pct(s['tiers']['easy'])} | {pct(s['tiers']['standard'])} | {pct(s['tiers']['judge'])} | {pct(s['tiers']['hard'])} | "
                 f"{sec(s['speed']['p50_s_raw'])} | {sec(s['speed']['p95_s_raw'])} | {sec(s['speed']['hard_tier_p50_s'])} | {s['speed']['adjustment']} |" for s in S) + """

## Cost basis

""" + "\n".join(f"- **{name(s)}** — {usd(s)}: {s['cost']['basis']}" for s in S) + """

## Revision log

""" + log + """

## What changed from v1.2-wip

- Score: four axes (Intelligence, Calibration, Speed, Cost), 25 % each, geometric mean — replaces the Balanced 33:33:33 arithmetic Main Score.
- Intelligence weights hard 30 % (was 50 %); the rest 1 : 2 : 2 over easy : standard : judge.
- Speed scale 20 points per 10× (was 50); Cost scale 30 points per 10× from $0.001 (was 25). Latency of non-production endpoints adjusted (assumption, see the speed note).
- One open-alternative-jev row (author's option order), named plainly; the reversed-order run is a footnote.
- Needle 3 options-as-tools priced on Needle 3's per-token basis ($0.0162 est.; it had no price).
- Qwen3.8 27B on Chutes is treated as a production API (no latency adjustment).
"""
(ROOT / "RESULTS-v1.2.md").write_text(md)
print("RESULTS-v1.2.md written")
