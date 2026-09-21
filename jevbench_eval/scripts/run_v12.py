"""Run smalljev semantic-v9 against JevBench v1.2 PUBLIC datasets using the
upstream fstandhartinger/jevbench harness (pulled fresh to jevbench_latest/).
Produces a v1.2 composite (4 axes, geometric mean) on the items we can actually
load from datasets/public/.

Usage:
    python run_v12.py \
        --variant-label  smalljev_semantic_v9 \
        --adapter-dir    D:/Project/smalljev/jevbench_latest/v9_weights/semantic-v9-lora \
        --scorer-ckpt    D:/Project/smalljev/jevbench_latest/v9_weights/semantic-v9-scorer.pt \
        --noulscore-ckpt D:/Project/smalljev/jevbench_latest/v9_weights/semantic-v9-noulscore.pt \
        --harness-root   D:/Project/smalljev/jevbench_latest \
        --run-dir        D:/Project/smalljev/runs/2026-09-21_semantic_v9_v12
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "smalljev"))
sys.path.insert(0, str(ROOT))  # for _fcntl_shim

# Windows: fcntl doesn't exist; install no-op shim BEFORE importing JevBench.
try:
    import fcntl  # noqa: F401
except ImportError:
    import _fcntl_shim  # noqa: F401

# Inject harness into path
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-label", default="smalljev_semantic_v9")
    ap.add_argument("--backbone", default="openbmb/MiniCPM5-2B-Base")
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--scorer-ckpt", required=True)
    ap.add_argument("--noulscore-ckpt", required=True)
    ap.add_argument("--harness-root", default="D:/Project/smalljev/jevbench_latest")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cap-usd", type=float, default=15.0)
    ap.add_argument("--reserve-usd", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay-s", type=float, default=0.0)
    ap.add_argument("--tiers", default="easy,standard,hard",
                    help="comma list of public tiers: easy,standard,hard,judge")
    return ap.parse_args()


def main():
    args = parse_args()
    harness = Path(args.harness_root).resolve()
    sys.path.insert(0, str(harness))

    # Drop the smalljev_semantic.py adapter we copied into the harness's adapters/ folder
    from jevbench.budget import Ledger
    from jevbench.runner import Runner
    from jevbench.tasks import dataset_hash, load_jsonl

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw"
    results_path = run_dir / "results.jsonl"
    ledger_path = run_dir / "ledger.jsonl"
    manifest_path = run_dir / "manifest.json"

    public = harness / "datasets" / "public"
    tier_map = {
        "easy":     public / "easy.jsonl",
        "standard": public / "original.jsonl",
        "hard":     public / "hard.jsonl",
        # judge requires third-party import (auto-model-router) — not in public/
    }
    files = [tier_map[t.strip()] for t in args.tiers.split(",") if t.strip() in tier_map]
    if not files:
        raise SystemExit(f"No valid tiers in {args.tiers}; available: {list(tier_map)}")
    tasks = []
    for f in files:
        tasks.extend(load_jsonl(str(f)))

    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[run] tasks loaded: {len(tasks)} from {[f.name for f in files]}")

    # Import the copied adapter
    from jevbench.adapters.smalljev_semantic import SmalljevSemanticAdapter
    adapter = SmalljevSemanticAdapter(
        backbone=args.backbone,
        adapter_dir=args.adapter_dir,
        scorer_ckpt=args.scorer_ckpt,
        noulscore_ckpt=args.noulscore_ckpt,
        variant_label=args.variant_label,
    )

    ledger = Ledger(str(ledger_path), cap_usd=args.cap_usd)
    runner = Runner(adapter, ledger, raw_dir=str(raw_dir),
                    default_reserve_usd=args.reserve_usd)

    print("[run] loading semantic model + heads...", flush=True)
    t0 = time.perf_counter()
    adapter.load()
    load_s = time.perf_counter() - t0
    print(f"[run] model loaded in {load_s:.1f}s", flush=True)

    print(f"[run] adapter={adapter.name} variant={args.variant_label} "
          f"tasks={len(tasks)} cap=${args.cap_usd} ledger={ledger_path}")
    started = time.time()
    records = runner.run_all(tasks, results_path=str(results_path),
                             delay_s=args.delay_s)
    finished = time.time()

    failed = sum(1 for r in records if r["status"] == "failed")
    print(f"[run] done: {len(records)}/{len(tasks)} attempted, "
          f"{failed} failed; charged=${ledger.charged:.4f}")
    print(f"[run] wall time: {finished - started:.1f}s")

    resolved_models = sorted({r.get("model") or "" for r in records})
    manifest = {
        "run_label": args.variant_label,
        "adapter": adapter.name,
        "variant_label": args.variant_label,
        "backbone": args.backbone,
        "adapter_dir": args.adapter_dir,
        "scorer_ckpt": args.scorer_ckpt,
        "noulscore_ckpt": args.noulscore_ckpt,
        "cost_basis": adapter.cost_basis,
        "load_s": load_s,
        "torch_version": adapter.torch_version,
        "transformers_version": adapter.transformers_version,
        "peft_version": adapter.peft_version,
        "device": str(adapter._device_real),
        "threads": adapter.threads,
        "dataset_files": [str(f) for f in files],
        "dataset_hash": dataset_hash(tasks),
        "n_planned": len(tasks),
        "n_attempted": len(records),
        "n_failed": failed,
        "charged_usd": ledger.charged,
        "delay_s": args.delay_s,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        "resolved_models": resolved_models,
        "harness_root": str(harness),
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"[run] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
