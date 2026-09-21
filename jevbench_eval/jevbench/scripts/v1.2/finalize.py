"""JevBench v1.2 final: the JevBench Score (jevbench/composite_v12.py) applied to the frozen v1.2 measurements.

    python3 scripts/v1.2/finalize.py

Input : results/v1.2/wip/jevbench-v1.2-wip-results.json  (the v1.2-wip artifact: every measurement, tag v1.2-wip)
        results/v1.2/wip/jevbench-v1.2-wip-per-task.json
Output: results/v1.2/jevbench-v1.2-results.json, results/v1.2/jevbench-v1.2-per-task.json

v1.2.1 (19 Sep 2026): rows measured after the v1.2 freeze on the same frozen items and code are added from
results/v1.2/additions/<key>.json (+ <key>-per-task.json); nothing else changes. First addition: djev (Maisa).

No measurement changes here. What changes: the score (4 axes, geometric mean), one open-alternative-jev row instead of two,
and the Needle 3 options-as-tools price (it had none; now priced on Needle 3's per-token basis).
"""
import copy
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from jevbench import composite_v12 as C  # noqa: E402

V12 = ROOT / "results/v1.2"
WIP = json.loads((V12 / "wip/jevbench-v1.2-wip-results.json").read_text())
WIP_TASKS = json.loads((V12 / "wip/jevbench-v1.2-wip-per-task.json").read_text())
ADDITIONS = {p.stem: json.loads(p.read_text()) for p in sorted((V12 / "additions").glob("*.json")) if not p.stem.endswith("-per-task")}
REVISION = "v1.2.1" if ADDITIONS else "v1.2"
REVISION_LOG = [
    {"revision": "v1.2.1", "date": "2026-09-19", "note": "Added djev (Maisa, diffusion-gemma): full v1.2 set (534 decisions incl. held-out) "
     "through its production API, scored with the unchanged v1.2 rules. Cost at djev's announced price ($0.035/M input tokens, output free), "
     "which is not yet charged (free preview). No other row changed."},
    {"revision": "v1.2", "date": "2026-09-19", "note": "Final JevBench Score: 4 axes, geometric mean."},
]

# open-alternative-jev: the ranked row is the run with the author's own yes/no option order ("A. yes, B. no", as his
# yes_no() helper builds it). Our first adapter reversed it; that run is kept as raw files and one footnote, not ranked.
OAJ_RANKED, OAJ_REVERSED = "open-alternative-jev-yesfirst", "open-alternative-jev"
OAJ_KEY, OAJ_NAME = "open-alternative-jev", "open-alternative-jev (Qwen3.5-4B, IkerMoel)"
# 21 % / 72 %: the 68 yes/no answer-judging items (family "adequacy", imported judge set) of the v1.1 judge run — 20.6 % vs 72.1 %,
# recomputed from the GPU round's runs/open-alternative-jev{,-yesfirst}--v1/results.jsonl on 19 Sep 2026.
OAJ_FOOTNOTE = ("With the options in reverse order (A. no, B. yes) the same model scored 21 % instead of 72 % on yes/no "
                "answer-judging items — small models are very sensitive to option order.")


def endpoint_kind(cond):
    c = cond.lower()
    if c.startswith("production api") or c.startswith("chutes shared"):
        return "api"
    if "runpod gpu" in c:
        return "gpu"
    if "demo endpoint" in c:
        return "demo"
    if "our cpu" in c:
        return "cpu"
    raise ValueError(f"unknown endpoint condition: {cond}")


def needle_tools_price(row, needle):
    """Needle 3 options-as-tools never ran the hard tier: price its 314 v1.1 decisions on Needle 3's per-token basis."""
    c = row["cost"]
    usd = c["usd_per_1000_v11_tiers"]
    assert usd and "llama-3.2-1b-instruct list price $0.027/M in, $0.201/M out" in c["basis_v11"] and "llama-3.2-1b" in needle["cost"]["basis_v11"]
    c["usd_per_1000"] = usd
    c["basis_final"] = ("ESTIMATE: same per-token price as Needle 3 (openrouter meta-llama/llama-3.2-1b-instruct $0.027/M in, $0.201/M out) "
                        "x 452 input and 20 output tokens per decision, over the 314 easy/standard/judge decisions it ran (no hard-tier run). "
                        "The v1.2 score lab had no price for this row and scored it 100; fixed.")
    return row


def build_row(s):
    kind = endpoint_kind(s["endpoint_condition"])
    sb = s["speed_block"]
    p50, p95 = sb["p50_s"], sb["p95_s"]
    axes = {
        "intelligence": C.intelligence(s["tiers"]),
        "calibration": s["calibration"]["score"],
        "speed": C.speed(p50, p95, kind),
        "cost": C.cost(s["cost"]["usd_per_1000"]),
    }
    row = {k: s.get(k) for k in ("key", "display", "class", "open", "author", "repo", "licence", "underlying", "has_distribution", "probability_source")}
    row.update({
        "endpoint_condition": s["endpoint_condition"], "endpoint_kind": kind,
        "partial": s["partial"], "ranked": not s["partial"],
        "tiers": s["tiers"], "axes": axes, "jevbench_score": C.jevbench_score(axes),
        "speed": {
            "p50_s_raw": p50, "p95_s_raw": p95,
            "p50_s_adjusted": C.adjusted_latency(p50, kind), "p95_s_adjusted": C.adjusted_latency(p95, kind),
            "adjustment": "none (production API)" if kind == "api" else
                          f"x{C.LOAD_FACTOR:g}" + (f" + {C.OWN_SERVER_ADD_S} s" if kind in C.OWN_SERVERS else "") + " (assumption, not measured)",
            "run": sb.get("run"), "hardware": sb.get("hardware"), "measured_where": sb.get("measured_where"),
            "hard_tier_p50_s": (s.get("hard") or {}).get("latency_p50_s"), "hard_tier_p95_s": (s.get("hard") or {}).get("latency_p95_s"),
        },
        "cost": {"kind": s["cost"]["kind"], "usd_per_1000": s["cost"]["usd_per_1000"],
                 "basis": s["cost"].get("basis_final") or " | ".join(b for b in (s["cost"]["basis_v11"], s["cost"]["basis_hard"]) if b),
                 "usd_per_1000_v11_tiers": s["cost"]["usd_per_1000_v11_tiers"], "usd_per_1000_hard": s["cost"]["usd_per_1000_hard"],
                 "self_host_sensitivity": s["cost"]["self_host_sensitivity"]},
        "calibration": {**s["calibration"], "note": None if s["calibration"]["score"] is not None else
                        "returns a label, not a probability distribution: no calibration score (counts as 0 in the JevBench Score)"},
        "hard": s.get("hard"),
    })
    row["presets"] = {name: C.preset_score(axes, w) for name, w in C.PRESETS.items()}
    return row


def main():
    src = {s["key"]: copy.deepcopy(s) for s in WIP["systems"]}
    rev = src.pop(OAJ_REVERSED)
    own = src.pop(OAJ_RANKED)
    own.update(key=OAJ_KEY, display=OAJ_NAME, run_key=OAJ_RANKED)
    src[OAJ_KEY] = own
    needle_tools_price(src["needle-3-tools"], src["needle-3"])
    for k, row in ADDITIONS.items():
        assert k not in src, k
        src[k] = copy.deepcopy(row)

    rows = [build_row(s) for s in src.values()]
    rows[[r["key"] for r in rows].index(OAJ_KEY)]["run_key"] = OAJ_RANKED
    ranked = sorted((r for r in rows if r["ranked"]), key=lambda r: -r["jevbench_score"])
    partial = sorted((r for r in rows if not r["ranked"]), key=lambda r: -r["jevbench_score"])
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    for r in partial:
        r["rank"] = None
    for name in C.PRESETS:
        for i, r in enumerate(sorted(ranked, key=lambda r: -r["presets"][name]), 1):
            r.setdefault("rank_under", {})[name] = i

    footnote = OAJ_FOOTNOTE
    art = {
        "benchmark": "JevBench", "revision": REVISION, "revision_log": REVISION_LOG, "protocol": "jevbench::v1.2", "status": "final",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "measured_in": "v1.2-wip (tag v1.2-wip); no measurement changed for v1.2 final" + (
            "; v1.2.1 additions measured later on the same frozen items: " + ", ".join(ADDITIONS) if ADDITIONS else ""),
        "revision_note": "v1.2 final: 4 axes (Intelligence, Calibration, Speed, Cost), 25 % each, geometric mean; hard tier 30 % of Intelligence; "
                         "latency of non-production endpoints adjusted (assumption); one open-alternative-jev row (author's option order); "
                         "Needle 3 options-as-tools priced." + (" v1.2.1: added djev (Maisa, diffusion-gemma)." if ADDITIONS else ""),
        "score_name": "JevBench Score",
        "score_one_liner": "Intelligence, Calibration, Speed, Cost — 25 % each, geometric mean: a weak axis pulls the score down hard.",
        "tiers": WIP["tiers"], "tier_weights": C.TIER_WEIGHTS, "axis_weights": C.WEIGHTS,
        "presets": {k: dict(zip(C.AXES, v)) for k, v in C.PRESETS.items()}, "main": C.MAIN,
        "speed_note": C.SPEED_NOTE,
        "scoring": {
            "jevbench_score": "exp(sum over the four axes of 0.25 x ln(max(axis, 1))) — the geometric mean of Intelligence, Calibration, Speed and Cost. "
                              "A weak axis pulls the score down hard; a strong axis cannot buy it back.",
            "intelligence": "100 x weighted accuracy: hard 30 %, easy 14 %, standard 28 %, judge 28 %. Accuracy = correct / all items; failed, "
                            "timed-out or unparseable answers count as wrong.",
            "hard_tier": WIP["scoring"]["hard_tier"],
            "calibration": WIP["scoring"]["calibration"].split(" Label-only")[0] + " Label-only systems have none; it counts as 0 in the JevBench Score.",
            "speed": "Mean of score(p50) and score(p95) of the serial 242-decision standard+judge run; score(s) = 100 - 20 log10(s / 0.1 s), "
                     "clipped to 0..100 (0.1 s = 100, 1 s = 80, 10 s = 60). " + C.SPEED_NOTE +
                     " Production APIs (Jev, djev, OpenAI, Google, DeepSeek, Chutes) are not adjusted.",
            "cost": "Dollars per 1,000 decisions pooled over all 534 v1.2 decisions; score = 100 - 30 log10(usd / 0.001), clipped to 0..100 "
                    "($0.001 = 100, $0.01 = 70, $0.10 = 40, $1 = 10). Measured = public tariff x measured tokens. est. = hosted-provider list price "
                    "of the same weights or size class x tokens. announced = the provider's published price, not yet charged (free preview), x measured tokens.",
            "ranked": "Ranked: every tier attempted for >= 95 % of its decisions. Partial runs are shown below the ranking, marked, without a rank.",
            "presets": "Other views reweight the same four axes and combine them the same way (geometric mean). They are not the JevBench Score.",
        },
        "hard_dataset": WIP.get("hard_dataset"),
        "footnotes": {OAJ_KEY: footnote, **{k: r["footnote"] for k, r in ADDITIONS.items() if r.get("footnote")}},
        "excluded_runs": [{
            "key": "open-alternative-jev-reversed-order", "run_key": OAJ_REVERSED, "why_not_ranked":
            "Our first adapter put the options in reverse order (A. no, B. yes); the author's yes_no() helper builds A. yes, B. no. "
            "An adapter mistake, not a model weakness, so the author-order run is the ranked row. Raw run files are kept.",
            "tiers": rev["tiers"], "footnote": footnote,
        }],
        "systems": ranked + partial,
    }
    (V12 / "jevbench-v1.2-results.json").write_text(json.dumps(art, indent=1, ensure_ascii=False) + "\n")

    tasks = copy.deepcopy(WIP_TASKS)
    tasks["systems"].pop(OAJ_REVERSED)
    tasks["systems"][OAJ_KEY] = tasks["systems"].pop(OAJ_RANKED)
    if isinstance(tasks["systems"][OAJ_KEY], dict) and "display" in tasks["systems"][OAJ_KEY]:
        tasks["systems"][OAJ_KEY]["display"] = OAJ_NAME
    for k in ADDITIONS:
        tasks["systems"][k] = json.loads((V12 / "additions" / f"{k}-per-task.json").read_text())
    tasks.update(revision=REVISION, status="final")
    (V12 / "jevbench-v1.2-per-task.json").write_text(json.dumps(tasks, indent=1, ensure_ascii=False) + "\n")

    for r in ranked + partial:
        a = r["axes"]
        f = lambda v: "  -  " if v is None else f"{v:5.1f}"
        print(f"{str(r['rank'] or '-'):>2} {r['key'][:28]:28} {f(r['jevbench_score'])}  I {f(a['intelligence'])} C {f(a['calibration'])} "
              f"S {f(a['speed'])} K {f(a['cost'])}  ${r['cost']['usd_per_1000']:.4f} {r['cost']['kind'][:4]} {'PARTIAL' if r['partial'] else ''}")
    print(footnote)


if __name__ == "__main__":
    main()
