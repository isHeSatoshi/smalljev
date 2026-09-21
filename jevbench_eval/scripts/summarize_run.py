"""Summarize a smalljev × JevBench run using the official JevBench scorers.

Loads the per-task results.jsonl files from one variant's run directory (all
tier subdirectories), then runs `jevbench.summarize.summarize()` to compute
the v1.2-style metrics (macro accuracy, Brier, ECE, latency, schema validity,
paraphrase consistency). Also computes per-tier accuracies and a per-family
breakdown for the hard tier.

Usage:
    python scripts/summarize_run.py --run-dir runs/2026-09-19_v4_bundle \
        --output runs/2026-09-19_v4_bundle/summary.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Windows: fcntl doesn't exist; install no-op shim before importing JevBench.
try:
    import fcntl  # noqa: F401
except ImportError:
    import _fcntl_shim  # noqa: F401

sys.path.insert(0, str(ROOT / "jevbench"))

from jevbench.summarize import summarize, agreement  # noqa: E402
from jevbench.metrics import brier_score, ece_top_label  # noqa: E402
from jevbench.tasks import load_jsonl, dataset_hash  # noqa: E402
from jevbench.composite_v12 import (
    speed as v12_speed, cost as v12_cost, intelligence as v12_intelligence,
    calibration as v12_calibration, jevbench_score as v12_score,
    speed_point, adjusted_latency, TIER_WEIGHTS, LOAD_FACTOR, OWN_SERVER_ADD_S,
)
from jevbench.scoring import argmax_label  # noqa: E402

PUBLIC = ROOT / "jevbench" / "datasets" / "public"


def load_records(run_dir):
    """Return (all_records_by_tier, all_tasks_by_id)."""
    recs = []
    tasks_by_id = {}
    tier_dirs = sorted([p for p in Path(run_dir).iterdir() if p.is_dir()
                        and p.name in ("easy", "standard", "hard", "all")])
    for tier_dir in tier_dirs:
        rp = tier_dir / "results.jsonl"
        if not rp.exists():
            print(f"WARN: missing {rp}")
            continue
        with open(rp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs, tasks_by_id


def load_tasks_for_records(records):
    """Reload tasks from the public datasets matching the records' task_ids."""
    files = [PUBLIC / "easy.jsonl", PUBLIC / "original.jsonl", PUBLIC / "hard.jsonl"]
    by_id = {}
    for fp in files:
        for t in load_jsonl(str(fp)):
            by_id[t.id] = t
    tasks = [by_id[r["task_id"]] for r in records if r["task_id"] in by_id]
    return tasks


def tier_of(task_id):
    if task_id.startswith("easy-"):
        return "easy"
    if task_id.startswith("original-"):
        return "standard"
    if task_id.startswith("hard-"):
        return "hard"
    return "unknown"


def per_tier_metric(tasks, records):
    out = {}
    for tier in ("easy", "standard", "hard"):
        ts = [t for t in tasks if tier_of(t.id) == tier]
        rs = [r for r in records if tier_of(r["task_id"]) == tier]
        out[tier] = summarize(ts, rs, headline_only=True)
    return out


def per_family_metric(tasks, records):
    out = {}
    families = sorted({t.family for t in tasks})
    for f in families:
        ts = [t for t in tasks if t.family == f]
        rs = [r for r in records if any(t.id == r["task_id"] for t in ts)]
        out[f] = summarize(ts, rs, headline_only=True)
    return out


def latency_summary(records):
    lats = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    lats = sorted(lats)
    if not lats:
        return None
    n = len(lats)
    def pct(p):
        if n == 1:
            return lats[0]
        k = (n - 1) * p
        f = int(k)
        c = min(f + 1, n - 1)
        return lats[f] * (c - k) + lats[c] * (k - f)
    return {
        "n": n,
        "p50_s": pct(0.5),
        "p95_s": pct(0.95),
        "min_s": lats[0],
        "max_s": lats[-1],
        "mean_s": sum(lats) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--endpoint-kind", default="gpu",
                    help="JevBench endpoint kind: 'gpu' (self-hosted), 'cpu', or 'api'")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    records, _ = load_records(run_dir)
    tasks = load_tasks_for_records(records)
    print(f"[summarize] loaded {len(records)} records, {len(tasks)} tasks")

    # Macro-level summary (all public items).
    overall = summarize(tasks, records, headline_only=True)

    # Per-tier breakdown.
    per_tier = per_tier_metric(tasks, records)

    # Hard tier per-family breakdown.
    hard_tasks = [t for t in tasks if tier_of(t.id) == "hard"]
    hard_records = [r for r in records if tier_of(r["task_id"]) == "hard"]
    per_family_hard = per_family_metric(hard_tasks, hard_records)

    # Latency summary: standard+judge-equivalent (we use the standard tier + judge-equivalent
    # items we DO have; the actual v1.1.2 protocol uses 242 items we can't access, so this is a
    # public-only proxy).
    standard_records = [r for r in records if tier_of(r["task_id"]) == "standard"]
    standard_lats = latency_summary(standard_records)
    hard_lats = latency_summary(hard_records)
    all_lats = latency_summary(records)

    # Adjusted latency per the v1.2 rules (composite_v12.adjusted_latency).
    if standard_lats:
        adj_p50 = adjusted_latency(standard_lats["p50_s"], args.endpoint_kind)
        adj_p95 = adjusted_latency(standard_lats["p95_s"], args.endpoint_kind)
        # Speed score using only standard+hard-as-judge (we cannot replicate the 242-item
        # standard+judge run; public proxy).
        speed_score = v12_speed(adj_p50, adj_p95, args.endpoint_kind)
    else:
        adj_p50 = adj_p95 = speed_score = None

    # Cost: local GPU, no provider tariff. Per the README, an unknown price is null,
    # not 0. We additionally report an ESTIMATE based on a same-class hosted-provider
    # price (a 2-3B dense model on OpenRouter, the same approach used by the
    # published 'open-jev-deberta-v3-large' row).
    cost_basis = "local_gpu_no_provider_tariff"
    price_in_per_m = None
    price_out_per_m = None
    cost_usd_per_1000 = None  # null because no billable provider tariff
    # Estimate basis: MiniCPM5-2B-Base class (2-3B dense) ~ $0.04/M input, $0/M output
    # on OpenRouter (Qwen/Qwen2.5-3B-Instruct listed). Output is 0 because smalljev
    # never generates tokens.
    estimate_basis = "ESTIMATE: hosted-provider price, OpenRouter Qwen/Qwen2.5-3B-Instruct $0.04/M in, $0/M out (a 2-3B dense model of the same size class; smalljev never generates) x measured input tokens."
    estimate_usd_per_1000 = None
    if records:
        # Compute input tokens via the smalljev tokenizer so the estimate is real.
        # The per-record raw request bodies live in raw_dir/<sha256(task_id)>.json,
        # written by the JevBench runner.
        try:
            from transformers import AutoTokenizer
            print(f"[summarize] loading tokenizer for cost estimate...", flush=True)
            tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM5-2B-Base",
                                               trust_remote_code=False)
            in_tokens = []
            import hashlib
            for r in records:
                # find the raw file by task_id hash
                tid = r["task_id"]
                tid_hash = hashlib.sha256(tid.encode()).hexdigest()
                raw_path = None
                for tier_dir in Path(run_dir).iterdir():
                    if not tier_dir.is_dir():
                        continue
                    cand = tier_dir / "raw" / (tid_hash + ".json")
                    if cand.exists():
                        raw_path = cand
                        break
                if raw_path is None:
                    continue
                blob = json.loads(raw_path.read_text(encoding="utf-8"))
                rb = blob.get("request") or {}
                if not isinstance(rb, dict):
                    continue
                state = rb.get("state", "") or ""
                qs = rb.get("questions", {}) or {}
                # Concatenate state + every question prompt text + the option/level
                # text that the adapter appended.
                pieces = [state]
                for qname, q in qs.items():
                    if isinstance(q, dict):
                        pieces.append(q.get("question", "") or "")
                        pieces.extend(q.get("choices", []) or [])
                        pieces.extend(q.get("levels", []) or [])
                n = len(tok(" ".join(pieces), add_special_tokens=False)["input_ids"])
                in_tokens.append(n)
            if in_tokens:
                mean_in = sum(in_tokens) / len(in_tokens)
                print(f"[summarize] mean input tokens: {mean_in:.0f} "
                      f"({len(in_tokens)} samples)", flush=True)
                estimate_usd_per_1000 = 1000 * mean_in * 0.04 / 1e6
        except Exception as e:
            print(f"[summarize] WARN: token-count estimate failed: {e}")

    # Intelligence: weighted accuracy using tier weights.
    tier_accs = {t: per_tier.get(t, {}).get("accuracy") for t in TIER_WEIGHTS}
    intel = v12_intelligence(tier_accs)

    # Calibration: ECE from the summarize() output + per-tier Brier.
    ece_dict = overall.get("ece")
    ece_overall = ece_dict.get("ece") if ece_dict else None

    # Brier hard: only valid on the hard tier's distribution-returning records.
    hard_brier = per_tier.get("hard", {}).get("brier_mean")
    calibration_score = v12_calibration(ece_overall)

    # Cost score: prefer the measured estimate if we have one; else null.
    if estimate_usd_per_1000 is not None:
        cost_score = v12_cost(estimate_usd_per_1000)
    else:
        cost_score = None

    # Composite JevBench Score.
    axes = {
        "intelligence": intel,
        "calibration": calibration_score,
        "speed": speed_score,
        "cost": cost_score,
    }
    js = v12_score(axes)

    summary = {
        "run_dir": str(run_dir),
        "endpoint_kind": args.endpoint_kind,
        "endpoint_adjustment": (
            f"x{int(LOAD_FACTOR)} "
            f"+ {OWN_SERVER_ADD_S} s"
            if args.endpoint_kind in ("gpu", "cpu")
            else "none (production API)"
        ),
        "overall": {
            "n_planned": overall["n_planned"],
            "n_attempted": overall["n_attempted"],
            "n_scorable": overall["n_scorable"],
            "n_valid": overall["n_valid"],
            "n_correct": overall["n_correct"],
            "accuracy": overall["accuracy"],
            "schema_validity": overall["schema_validity"],
            "schema_validity_strict": overall["schema_validity_strict"],
            "operational_success": overall["operational_success"],
            "brier_mean": overall["brier_mean"],
            "ece": ece_dict,
            "ordinal_mae": overall["ordinal_mae"],
            "macro_accuracy": overall["macro_accuracy"],
            "paraphrase_consistency": overall.get("paraphrase_consistency"),
            "model_identities": overall["model_identities"],
            "probability_sources": overall["probability_sources"],
            "n_renormalized": overall["n_renormalized"],
            "complete": overall["complete"],
        },
        "per_tier": per_tier,
        "per_family_hard": per_family_hard,
        "latency": {
            "all_items_raw": all_lats,
            "standard_items_raw": standard_lats,
            "hard_items_raw": hard_lats,
            "standard_p50_adjusted": adj_p50,
            "standard_p95_adjusted": adj_p95,
            "speed_score": speed_score,
            "note": (
                "Standard+judge is 242 items (held-out + imported) in the v1.1.2 protocol. "
                "Only 72 standard items are public; reported latency here is for the 72-item "
                "public standard subset (not the 242-item full set)."
            ),
        },
        "axes": axes,
        "cost_basis": cost_basis,
        "estimate_basis": estimate_basis,
        "price_input_per_m_estimated": 0.04 if estimate_usd_per_1000 else None,
        "estimate_usd_per_1000": estimate_usd_per_1000,
        "measured_usd_per_1000": cost_usd_per_1000,
        "jevbench_score": js,
        "split": "public-only",
        "protocol_note": (
            "JevBench Score requires four axes (intelligence, calibration, speed, cost). "
            "The v1.2 official score also requires the held-out items (109 hard + 24 easy + "
            "24 standard + 146 judge items); those are not accessible from public data, so "
            "this run reports a public-only subset. The 'speed' axis is computed on the public "
            "72-item standard subset, not the 242-item standard+judge run specified by v1.1.2."
        ),
    }

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(f"[summarize] -> {out}")
    print()
    print("=" * 60)
    print("JEVBENCH PUBLIC-ONLY SUMMARY")
    print("=" * 60)
    print(f"Variant: {run_dir.name}")
    print(f"Items attempted: {overall['n_attempted']}/{overall['n_planned']}")
    print(f"Schema validity (RENORM band): {overall['schema_validity']:.3f}")
    print(f"Schema validity (STRICT 1e-3):  {overall['schema_validity_strict']:.3f}")
    print(f"Accuracy (argmax, scorable):     {overall['accuracy']:.3f}")
    print(f"Brier mean:                      {overall['brier_mean']}")
    print(f"ECE (10-bin top-label):          {ece_overall}")
    print(f"Macro accuracy (per-family):      {overall['macro_accuracy']}")
    print(f"Operational success:             {overall['operational_success']:.3f}")
    print(f"Speed (adjusted, public standard): {speed_score}")
    print(f"Cost basis: {cost_basis}")
    print(f"Cost estimate USD/1000: {estimate_usd_per_1000}")
    print(f"Intelligence: {intel}")
    print(f"Calibration: {calibration_score}")
    print(f"JevBench Score (public-only): {js}")


if __name__ == "__main__":
    main()