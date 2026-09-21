"""Print top rows from the published JevBench v1.2 results."""
import json

data = json.load(open("jevbench/results/v1.2/jevbench-v1.2-results.json"))
rows = data["systems"]
rows_sorted = sorted(rows, key=lambda r: r.get("jevbench_score") or 0, reverse=True)
for i, r in enumerate(rows_sorted[:20], 1):
    score = r.get("jevbench_score")
    print(f"{i:2d} | {r['key']:30s} | jbs={score!r:>22} | hard={r['tiers'].get('hard')!r:>22} | rank={r.get('rank')!r:>4}")