"""Run a single smalljev semantic variant (semantic-v4 .. semantic-v7) against the
JevBench public dataset suite, using the SmalljevSemanticAdapter.

Mirrors scripts/run_variant.py but loads the new (scorer.pt + noulscore.pt + lora/)
file layout produced by evals/train_semantic.py.

Usage:
    python scripts/run_semantic_variant.py \
        --variant-label  smalljev_semantic_v7 \
        --adapter-dir    D:/Project/smalljev/evals/arms/semantic-v7-lora \
        --scorer-ckpt    D:/Project/smalljev/evals/arms/semantic-v7-scorer.pt \
        --noulscore-ckpt D:/Project/smalljev/evals/arms/semantic-v7-noulscore.pt \
        --run-dir        runs/2026-09-20_semantic_v7
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adapter"))
sys.path.insert(0, str(ROOT / "scripts"))

# Windows: fcntl doesn't exist; install no-op shim before importing JevBench.
try:
    import fcntl  # noqa: F401
except ImportError:
    import _fcntl_shim  # noqa: F401

sys.path.insert(0, str(ROOT / "jevbench"))

from jevbench.budget import Ledger  # noqa: E402
from jevbench.runner import Runner  # noqa: E402
from jevbench.tasks import dataset_hash, load_jsonl  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="public_all",
                    help="public_all / public_easy / public_standard / public_hard "
                         "OR comma-separated jsonl paths")
    ap.add_argument("--variant-label", required=True)
    ap.add_argument("--backbone", default="openbmb/MiniCPM5-2B-Base")
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--scorer-ckpt", required=True)
    ap.add_argument("--noulscore-ckpt", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cap-usd", type=float, default=15.0)
    ap.add_argument("--reserve-usd", type=float, default=0.02)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay-s", type=float, default=0.0)
    return ap.parse_args()


def resolve_tasks(spec):
    public = ROOT / "jevbench" / "datasets" / "public"
    mapping = {
        "public_all":     [public / "easy.jsonl", public / "original.jsonl",
                           public / "hard.jsonl"],
        "public_easy":    [public / "easy.jsonl"],
        "public_standard":[public / "original.jsonl"],
        "public_hard":    [public / "hard.jsonl"],
    }
    if spec in mapping:
        files = mapping[spec]
    else:
        files = [Path(p) for p in spec.split(",")]
    tasks = []
    for fp in files:
        tasks.extend(load_jsonl(str(fp)))
    return tasks, files


def derive_tier_subdir(spec):
    mapping = {
        "public_all": "all",
        "public_easy": "easy",
        "public_standard": "standard",
        "public_hard": "hard",
    }
    return mapping.get(spec, "custom")


def main():
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    tier_subdir = derive_tier_subdir(args.tasks)
    tier_dir = run_dir / tier_subdir
    tier_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tier_dir / "raw"
    results_path = tier_dir / "results.jsonl"
    ledger_path = run_dir / "ledger.jsonl"
    manifest_path = tier_dir / "manifest.json"

    tasks, files = resolve_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[run] tasks loaded: {len(tasks)} from {[str(f.name) for f in files]}")

    from adapter.smalljev_semantic_adapter import SmalljevSemanticAdapter
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
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(f"[run] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
