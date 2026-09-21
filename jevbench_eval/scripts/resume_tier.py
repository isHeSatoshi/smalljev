"""Resume a partial JevBench run by skipping already-done task IDs.

Used when a previous run was killed mid-flight. Reads `results.jsonl` in the tier
dir to discover which task_ids are already done, then runs only the rest.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import fcntl  # noqa: F401
except ImportError:
    import _fcntl_shim  # noqa: F401
sys.path.insert(0, str(ROOT / "jevbench"))

from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.tasks import load_jsonl


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--tier", required=True,
                    help="easy | standard | hard")
    ap.add_argument("--tasks-jsonl", required=True)
    ap.add_argument("--variant-label", required=True)
    ap.add_argument("--adapter-id", default=None)
    ap.add_argument("--heads-ckpt", default=None)
    ap.add_argument("--backbone", default="openbmb/MiniCPM5-2B-Base")
    ap.add_argument("--reserve-usd", type=float, default=0.02)
    ap.add_argument("--cap-usd", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="Run only the first N remaining items")
    ap.add_argument("--delay-s", type=float, default=0.0)
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    tier_dir = run_dir / args.tier
    raw_dir = tier_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    results_path = tier_dir / "results.jsonl"
    ledger_path = run_dir / "ledger.jsonl"

    # Load already-done task ids
    done_ids = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)["task_id"])
    print(f"[resume] {len(done_ids)} tasks already done in {results_path}")

    all_tasks = load_jsonl(args.tasks_jsonl)
    todo = [t for t in all_tasks if t.id not in done_ids]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[resume] {len(todo)} remaining to run")

    if not todo:
        print("[resume] nothing to do")
        return

    from adapter.smalljev_adapter import SmalljevAdapter
    adapter = SmalljevAdapter(
        backbone=args.backbone,
        adapter_id=args.adapter_id,
        heads_ckpt=args.heads_ckpt,
        variant_label=args.variant_label,
    )

    ledger = Ledger(str(ledger_path), cap_usd=args.cap_usd)
    runner = Runner(adapter, ledger, raw_dir=str(raw_dir),
                    default_reserve_usd=args.reserve_usd)

    # Append to results.jsonl (file already exists; runner would refuse; do it manually)
    # Trick: have the runner write to a tmp file then merge.
    tmp_path = tier_dir / f"results.{int(time.time())}.jsonl"
    records = runner.run_all(todo, results_path=str(tmp_path),
                             delay_s=args.delay_s)
    print(f"[resume] got {len(records)} new records")

    # Append to existing results.jsonl
    with results_path.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, allow_nan=False) + "\n")
    tmp_path.unlink(missing_ok=True)

    # Remove any raw files for tasks we already had before (no-op safety).
    print(f"[resume] done. results -> {results_path}")


if __name__ == "__main__":
    main()