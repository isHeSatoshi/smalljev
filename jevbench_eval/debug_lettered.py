"""Debug why lettered variant scores so poorly on easy tier."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "jevbench")
sys.path.insert(0, "scripts")
try:
    import fcntl  # noqa: F401
except ImportError:
    import _fcntl_shim  # noqa: F401

from jevbench.tasks import load_jsonl  # noqa: E402

runs = Path("runs")
for variant in ["lettered", "crown", "v5_holdout"]:
    print(f"\n=== {variant} ===")
    recs = []
    for line in open(runs / f"2026-09-19_{variant}" / "easy" / "results.jsonl"):
        recs.append(json.loads(line))
    correct = sum(1 for r in recs if r["correct"])
    print(f"easy tier: {correct}/{len(recs)} correct")
    wrong = [r for r in recs if not r["correct"]]
    # Look at top-3 wrong
    for r in wrong[:3]:
        tid = r["task_id"]
        pred = r["predicted"]
        ok = r["correct"]
        probs = r.get("probs", {})
        # Get expected from the task
        from jevbench.tasks import load_jsonl
        tasks = load_jsonl("jevbench/datasets/public/easy.jsonl")
        by_id = {t.id: t for t in tasks}
        t = by_id.get(tid)
        if t:
            exp = t.expected
            qtype = t.question["type"]
            print(f"  id={tid} family={r['family']} qtype={qtype} pred={pred} expected={exp} probs={probs}")
        else:
            print(f"  id={tid} family={r['family']} pred={pred} (task not found)")