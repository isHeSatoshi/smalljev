"""Print a one-shot digest of a summary.json."""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-09-19_v4_bundle/summary.json"
s = json.load(open(p))
print("Per-tier accuracy:")
for t in ("easy", "standard", "hard"):
    pt = s["per_tier"].get(t, {})
    if pt:
        print(f"  {t}: accuracy={pt.get('accuracy')} n_attempted={pt.get('n_attempted')} n_scorable={pt.get('n_scorable')}")
print()
print("Per-family hard:")
for f, m in s["per_family_hard"].items():
    if m.get("n_scorable", 0) > 0:
        acc = m.get("accuracy")
        print(f"  {f}: accuracy={acc} n={m.get('n_scorable')}")
print()
print("Latency:")
lat = s["latency"]
print(f"  standard raw: p50={lat['standard_items_raw']['p50_s']:.4f} p95={lat['standard_items_raw']['p95_s']:.4f}")
print(f"  standard adj: p50={lat['standard_p50_adjusted']} p95={lat['standard_p95_adjusted']}")
print(f"  speed_score: {lat['speed_score']}")
print(f"  hard raw: p50={lat['hard_items_raw']['p50_s']:.4f} p95={lat['hard_items_raw']['p95_s']:.4f}")
print()
print("Cost:")
print(f"  estimate_usd_per_1000: {s['estimate_usd_per_1000']}")
print(f"  measured_usd_per_1000: {s['measured_usd_per_1000']}")
print(f"  estimate_basis: {s['estimate_basis']}")
print()
print("Axes:")
for k, v in s["axes"].items():
    print(f"  {k}: {v}")
print(f"JevBench Score (public-only): {s['jevbench_score']}")