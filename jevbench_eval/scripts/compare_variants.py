"""Build a cross-variant comparison report from all per-variant summaries.

Reads every runs/*/summary.json file and writes runs/ALL_VARIANTS_COMPARISON.json
plus a Markdown digest of the four axes per variant.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_all_summaries(runs_dir):
    out = {}
    for d in sorted(Path(runs_dir).iterdir()):
        if not d.is_dir():
            continue
        sp = d / "summary.json"
        if not sp.exists():
            continue
        out[d.name] = json.loads(sp.read_text(encoding="utf-8"))
    return out


def per_tier_compare(summaries):
    out = {}
    tiers = ("easy", "standard", "hard")
    for tier in tiers:
        out[tier] = {}
        for name, s in summaries.items():
            pt = s["per_tier"].get(tier, {})
            out[tier][name] = {
                "n_scorable": pt.get("n_scorable"),
                "n_attempted": pt.get("n_attempted"),
                "accuracy": pt.get("accuracy"),
                "brier_mean": pt.get("brier_mean"),
                "schema_validity": pt.get("schema_validity"),
            }
    return out


def main():
    runs_dir = ROOT / "runs"
    summaries = load_all_summaries(runs_dir)
    if not summaries:
        print("No summaries found", file=sys.stderr)
        return

    rows = []
    for name, s in summaries.items():
        rows.append({
            "variant": name,
            "n_attempted": s["overall"]["n_attempted"],
            "n_scorable": s["overall"]["n_scorable"],
            "accuracy": s["overall"]["accuracy"],
            "schema_validity": s["overall"]["schema_validity"],
            "schema_validity_strict": s["overall"]["schema_validity_strict"],
            "operational_success": s["overall"]["operational_success"],
            "brier_mean": s["overall"]["brier_mean"],
            "ece_overall": (s["overall"]["ece"] or {}).get("ece") if s["overall"]["ece"] else None,
            "macro_accuracy": s["overall"]["macro_accuracy"],
            "ordinal_mae": s["overall"]["ordinal_mae"],
            "paraphrase_consistency": (s["overall"]["paraphrase_consistency"] or {}).get("agreement") if s["overall"]["paraphrase_consistency"] else None,
            "axes": s["axes"],
            "intelligence": s["axes"]["intelligence"],
            "calibration": s["axes"]["calibration"],
            "speed": s["axes"]["speed"],
            "cost": s["axes"]["cost"],
            "jevbench_score_public_only": s["jevbench_score"],
            "latency_standard_p50_raw": (s["latency"]["standard_items_raw"] or {}).get("p50_s"),
            "latency_standard_p95_raw": (s["latency"]["standard_items_raw"] or {}).get("p95_s"),
            "latency_standard_p50_adjusted": s["latency"]["standard_p50_adjusted"],
            "latency_standard_p95_adjusted": s["latency"]["standard_p95_adjusted"],
            "latency_hard_p50_raw": (s["latency"]["hard_items_raw"] or {}).get("p50_s"),
            "latency_hard_p95_raw": (s["latency"]["hard_items_raw"] or {}).get("p95_s"),
            "estimate_usd_per_1000": s["estimate_usd_per_1000"],
            "measured_usd_per_1000": s["measured_usd_per_1000"],
            "cost_basis": s["cost_basis"],
        })

    # Sort by JevBench Score descending.
    rows.sort(key=lambda r: r["jevbench_score_public_only"], reverse=True)

    comparison = {
        "n_variants": len(rows),
        "rows": rows,
        "per_tier": per_tier_compare(summaries),
        "note": (
            "All variants are the same v1 architecture (openbmb/MiniCPM5-2B-Base + "
            "HeadsBundle + LoRA). Variants differ only in the supervised/RL training "
            "mixture. public-only: held-out items are not accessible from this repo."
        ),
    }

    out_path = runs_dir / "ALL_VARIANTS_COMPARISON.json"
    out_path.write_text(json.dumps(comparison, indent=2, sort_keys=True))
    print(f"-> {out_path}")

    # Markdown digest.
    md = []
    md.append("# smalljev × JevBench — variant comparison (public-only)\n")
    md.append("All variants use the same v1 architecture (openbmb/MiniCPM5-2B-Base + "
              "HeadsBundle + LoRA); only the supervised/RL training mixture differs.\n")
    md.append("Sorted by **public-only JevBench Score** (geometric mean of 4 axes).\n")
    md.append("\n## Aggregate per variant\n")
    md.append("| Variant | Items | Acc | Brier | ECE | Macro | Schema | Speed (adj) | "
              "Cost/1000 | JevBench |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        sp50 = r["latency_standard_p50_adjusted"]
        sp95 = r["latency_standard_p95_adjusted"]
        md.append(f"| {r['variant']} | "
                  f"{r['n_attempted']}/{r['n_scorable']} | "
                  f"{r['accuracy']:.3f} | "
                  f"{r['brier_mean']:.3f} | "
                  f"{r['ece_overall']:.3f} | "
                  f"{r['macro_accuracy']:.3f} | "
                  f"{r['schema_validity']:.3f} | "
                  f"p50={sp50:.3f}s p95={sp95:.3f}s | "
                  f"${r['estimate_usd_per_1000']:.4f} | "
                  f"**{r['jevbench_score_public_only']:.2f}** |\n")

    md.append("\n## Per-tier accuracy\n")
    tier_rows = []
    for tier in ("easy", "standard", "hard"):
        for vname, m in comparison["per_tier"][tier].items():
            tier_rows.append({
                "tier": tier,
                "variant": vname,
                "accuracy": m["accuracy"],
                "brier_mean": m["brier_mean"],
                "schema_validity": m["schema_validity"],
            })
    md.append("| Tier | Variant | Accuracy | Brier | Schema |\n")
    md.append("|---|---|---|---|---|\n")
    for r in tier_rows:
        md.append(f"| {r['tier']} | {r['variant']} | "
                  f"{r['accuracy']:.3f} | {r['brier_mean']:.3f} | "
                  f"{r['schema_validity']:.3f} |\n")

    md.append("\n## Per-family hard-tier accuracy (n=111)\n")
    # Collect all hard per-family results.
    fam_set = set()
    for s in summaries.values():
        for f in s["per_family_hard"]:
            fam_set.add(f)
    md.append("| Family | n | " + " | ".join(summaries.keys()) + " |\n")
    md.append("|---" * (2 + len(summaries)) + "|\n")
    for f in sorted(fam_set):
        n = None
        cells = []
        for name in summaries:
            m = summaries[name]["per_family_hard"].get(f, {})
            if m.get("n_scorable"):
                n = m["n_scorable"]
            cells.append(f"{m.get('accuracy'):.3f}" if m.get("accuracy") is not None else "—")
        md.append(f"| {f} | {n if n is not None else '—'} | " + " | ".join(cells) + " |\n")

    md_path = runs_dir / "ALL_VARIANTS_COMPARISON.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"-> {md_path}")


if __name__ == "__main__":
    main()