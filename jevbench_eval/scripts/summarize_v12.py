"""Summarize a semantic-v9 run on v1.2 PUBLIC items using composite_v12 scoring.

Reads:
    runs/<run_dir>/results.jsonl   produced by run_v12.py
    jevbench_latest/datasets/public/*.jsonl
Computes:
    per-tier accuracy (easy, standard, hard)
    v1.2 Intelligence (tier-weighted)
    Calibration (Brier + ECE)
    Speed (p50/p95 from results.jsonl latency_s; +2x +0.15s for self-hosted CPU)
    Cost ($/1k decisions, ledger-charged)
    JevBench Score = exp(sum 0.25 ln(max(axis, 1)))

Writes:
    runs/<run_dir>/summary.json
    runs/<run_dir>/summary.md  (markdown table)
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--harness-root", default="D:/Project/smalljev/jevbench_latest")
    return ap.parse_args()


def tier_of(task_id: str) -> str:
    """Map task_id prefix -> tier. public/easy.jsonl → 'easy', public/original → 'standard', public/hard → 'hard'."""
    tid = str(task_id)
    if tid.startswith("easy-") or tid.startswith("easy-"):
        return "easy"
    if tid.startswith("original-"):
        return "standard"
    if tid.startswith("hard-"):
        return "hard"
    if tid.startswith("judge-") or tid.startswith("legacy-answer-"):
        return "judge"
    return "unknown"


def tier_accuracy(records, tasks_by_id, tier):
    in_tier = [r for r in records if tier_of(r["task_id"]) == tier]
    scorable = [r for r in in_tier
                if r["task_id"] in tasks_by_id
                and tasks_by_id[r["task_id"]].expected is not None
                and not tasks_by_id[r["task_id"]].provenance.get("exclude_reason")]
    if not scorable:
        return None, 0, 0
    correct = sum(1 for r in scorable if r.get("correct"))
    return correct / len(scorable), len(scorable), correct


def intelligence(tier_accs):
    weights = {"easy": 0.14, "standard": 0.28, "judge": 0.28, "hard": 0.30}
    s = tw = 0.0
    for t, w in weights.items():
        a = tier_accs.get(t)
        if a is not None:
            s += w * a
            tw += w
    return 100 * s / tw if tw else None


def brier_ece(records, tasks_by_id):
    bs = []
    tvds = []
    pairs = []
    for r in records:
        t = tasks_by_id.get(r["task_id"])
        if t is None or t.expected is None or t.provenance.get("exclude_reason"):
            continue
        p = r.get("probs")
        if not p:
            continue
        gold = str(t.expected)
        if gold not in p:
            continue
        b = sum((p[k] - (1 if k == gold else 0)) ** 2 for k in t.labels)
        bs.append(b)
        tvd = 0.5 * sum(abs(p[k] - (1 if k == gold else 0)) for k in t.labels)
        tvds.append(tvd)
        # argmax for ECE
        argmax = max(p.items(), key=lambda kv: kv[1])[0]
        pairs.append((max(p.values()), argmax == gold))
    if not bs:
        return None, None, 0, 0.0
    brier_mean = sum(bs) / len(bs)
    tvd_mean = sum(tvds) / len(tvds)
    # ECE: 15-bin
    n = len(pairs)
    bins = 15
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        confs_in_bin = [conf for (conf, _) in pairs if lo <= conf < hi]
        if not confs_in_bin:
            continue
        corrects_in_bin = [c for (conf, c) in pairs if lo <= conf < hi]
        acc_in_bin = sum(corrects_in_bin) / len(corrects_in_bin)
        avg_conf = sum(confs_in_bin) / len(confs_in_bin)
        ece += len(confs_in_bin) / n * abs(avg_conf - acc_in_bin)
    return brier_mean, ece, n, tvd_mean


def calibration_axis(brier, ece, tvd_mean):
    """v1.2 calibration: mean of (1 - ECE/0.5)*100 (clipped) and (1 - mean_TVD)*100."""
    if brier is None and ece is None:
        return None, None
    if ece is None:
        return None, None
    ece_term = max(0.0, 100 * (1 - ece / 0.5))
    tvd_term = 100 * max(0.0, 1 - tvd_mean)
    return ece_term, tvd_term


def speed_axis(latencies):
    """v1.2 speed: clamp(100 - 20 * log10(s/0.1)), applied to adjusted latency.

    Self-hosted endpoints: latency * 2 + 0.15s.
    """
    if not latencies:
        return None, None
    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)]
    # Adjust for self-hosted
    p50a = p50 * 2 + 0.15
    p95a = p95 * 2 + 0.15
    def score(s):
        if s <= 0:
            return 100.0
        return max(0.0, min(100.0, 100 - 20 * math.log10(s / 0.1)))
    return (score(p50a) + score(p95a)) / 2, {"p50": p50, "p95": p95, "p50_adj": p50a, "p95_adj": p95a}


def cost_axis(charged_usd, n_decisions, total_input_tokens=None,
              reference_tariff_in_per_m=0.04):
    """v1.2 cost: clamp(100 - 30 * log10(usd_per_1000 / 0.001)).

    For self-hosted runs without a provider tariff (our case), cost is estimated
    as `reference_tariff_in_per_m * total_input_tokens / 1e6`. The reference
    tariff here is the OpenRouter Qwen/Qwen2.5-3B-Instruct rate ($0.04/M input)
    that the v1.2.1 submission and the published Benchmark Heaven leaderboard
    use for self-hosted systems without their own tariff.
    """
    if n_decisions == 0:
        return None
    if total_input_tokens is not None and total_input_tokens > 0:
        usd = total_input_tokens * reference_tariff_in_per_m / 1e6
        usd_per_1000 = usd * 1000 / n_decisions
        basis = f"reference_tariff_{reference_tariff_in_per_m}_per_m_input"
    else:
        usd_per_1000 = charged_usd * 1000 / n_decisions
        basis = "ledger_charged_usd"
    if usd_per_1000 <= 0:
        return 100.0
    score = max(0.0, min(100.0, 100 - 30 * math.log10(usd_per_1000 / 0.001)))
    return score, usd_per_1000, basis


def jevbench_score(intel, cal, spd, cst):
    axes = [intel, cal, spd, cst]
    axes = [a if a is not None else 1 for a in axes]
    return math.exp(sum(0.25 * math.log(max(a, 1)) for a in axes))


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    harness = Path(args.harness_root).resolve()
    sys.path.insert(0, str(harness))
    try:
        import fcntl  # noqa: F401
    except ImportError:
        import _fcntl_shim  # noqa: F401

    from jevbench.tasks import load_jsonl

    # Load all tasks we attempted
    tasks = []
    for tier_file in ["easy.jsonl", "original.jsonl", "hard.jsonl"]:
        tasks.extend(load_jsonl(str(harness / "datasets" / "public" / tier_file)))
    tasks_by_id = {t.id: t for t in tasks}

    # Load results
    records = []
    with open(run_dir / "results.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Per-tier accuracy
    tier_accs = {}
    tier_detail = {}
    for tier in ["easy", "standard", "hard", "judge"]:
        acc, n, k = tier_accuracy(records, tasks_by_id, tier)
        tier_accs[tier] = acc
        tier_detail[tier] = {"accuracy": acc, "n": n, "correct": k}

    intel = intelligence(tier_accs)

    # Calibration (Brier + ECE + TVD)
    brier, ece, n_dist, tvd_mean = brier_ece(records, tasks_by_id)
    ece_term, tvd_term = calibration_axis(brier, ece, tvd_mean)
    if ece_term is not None and tvd_term is not None:
        cal = (ece_term + tvd_term) / 2
    else:
        cal = None

    # Speed
    latencies = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    spd, spd_detail = speed_axis(latencies)

    # Cost
    manifest = json.loads((run_dir / "manifest.json").read_text())
    charged = manifest.get("charged_usd", 0.0)
    n = len(records)

    # Re-tokenize each prompt to get measured input tokens (v9 uses
    # openbmb/MiniCPM5-2B-Base tokenizer).
    total_input_tokens = 0
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B-Base",
                                            trust_remote_code=False)
        for t in tasks:
            from smalljev.semantic import build_semantic_ids
            labels, rubric = [], {}
            qtype = t.question["type"]
            crit = t.question.get("criteria") or {}
            if qtype == "noul":
                labels = ["no", "yes"]
                rubric = {"no": crit.get("false", "No"),
                          "yes": crit.get("true", "Yes")}
            elif qtype == "score":
                if not isinstance(crit, list):
                    continue
                labels = [str(i) for i in range(len(crit))]
                rubric = {labels[i]: (crit[i] or labels[i]) for i in range(len(crit))}
            else:
                labels = list(t.labels)
                rubric = {k: (crit.get(k) or k) for k in labels}
            state = t.state if isinstance(t.state, str) else json.dumps(t.state, ensure_ascii=False)
            instr = t.question["instructions"]
            rubric_str = json.dumps(rubric, ensure_ascii=False)
            question = f"{instr}\nAllowed answers and rubric: {rubric_str}"
            try:
                ids, _ = build_semantic_ids(tok, state, question, labels,
                                            max_len=2560)
                total_input_tokens += len(ids)
            except Exception:
                pass
    except Exception as e:
        print(f"[summary] token-counting failed: {type(e).__name__}: {e}", flush=True)

    cst_result = cost_axis(charged, n, total_input_tokens=total_input_tokens)
    if cst_result is None:
        cst = None
        cst_basis = None
        cst_usd_per_1000 = None
    else:
        cst, cst_usd_per_1000, cst_basis = cst_result

    score = jevbench_score(intel, cal, spd, cst)

    summary = {
        "run_label": manifest.get("run_label"),
        "harness_root": str(harness),
        "n_decisions": n,
        "tier_accuracies": tier_accs,
        "tier_detail": tier_detail,
        "intelligence": intel,
        "calibration": cal,
        "calibration_detail": {"brier_mean": brier, "ece": ece,
                                "tvd_mean": tvd_mean,
                                "ece_term": ece_term, "tvd_term": tvd_term,
                                "n_distributions": n_dist},
        "speed": spd,
        "speed_detail": spd_detail,
        "cost": cst,
        "cost_detail": {"charged_usd": charged,
                         "usd_per_1000": cst_usd_per_1000,
                         "basis": cst_basis,
                         "total_input_tokens": total_input_tokens,
                         "reference_tariff_in_per_m": 0.04},
        "jevbench_score": score,
        "scoring_rules": "composite_v12 (4 axes, geometric mean)",
    }

    out_json = run_dir / "summary.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=str)

    # Markdown
    md_lines = []
    md_lines.append(f"# {summary['run_label']} — v1.2 local summary\n")
    md_lines.append(f"harness: `{harness}`")
    md_lines.append(f"n_decisions: {n}")
    md_lines.append("")
    md_lines.append("| tier | n | correct | accuracy |")
    md_lines.append("|---|---|---|---|")
    for t in ["easy", "standard", "judge", "hard"]:
        d = tier_detail.get(t, {})
        a = d.get("accuracy")
        a_str = f"{a*100:.2f}" if a is not None else "n/a"
        md_lines.append(f"| {t} | {d.get('n', 0)} | {d.get('correct', 0)} | {a_str} |")
    md_lines.append("")
    md_lines.append(f"**Intelligence**: {intel:.2f}" if intel is not None else "**Intelligence**: n/a")
    if cal is not None:
        md_lines.append(f"**Calibration**: {cal:.2f}  (Brier={brier:.4f}, ECE={ece:.4f}, TVD={tvd_mean:.4f})")
    else:
        md_lines.append("**Calibration**: n/a")
    if spd is not None:
        md_lines.append(f"**Speed**: {spd:.2f}  (p50_adj={spd_detail['p50_adj']*1000:.1f}ms, p95_adj={spd_detail['p95_adj']*1000:.1f}ms)")
    if cst is not None:
        md_lines.append(f"**Cost**: {cst:.2f}  (${cst_usd_per_1000:.4f} per 1k decisions; basis: {cst_basis})")
    md_lines.append("")
    md_lines.append(f"## JevBench Score: **{score:.2f}**")
    md_lines.append("")
    md_lines.append("Caveats:")
    md_lines.append("- Public-only run (231 items: easy 48 + standard 72 + hard 111).")
    md_lines.append("- v1.2.7 leaderboard uses 534 items including judge (146) + easy-heldout (24) + private hard (109).")
    md_lines.append("- Cost basis: local GPU, no provider tariff. Estimated against OpenRouter Qwen/Qwen2.5-3B-Instruct $0.04/M-input × measured input tokens, the same estimator the public leaderboard uses for self-hosted systems without their own tariff.")
    md_lines.append("- Speed is adjusted (×2 +0.15s) because the harness runs self-hosted.")

    out_md = run_dir / "summary.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[summary] -> {out_json}")
    print(f"[summary] -> {out_md}")
    print(f"JevBench Score (4-axis geometric): {score:.2f}")


if __name__ == "__main__":
    main()
