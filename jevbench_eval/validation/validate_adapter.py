"""Validation script for the smalljev adapter against JevBench public tasks.

Picks one example from each question type and family present in the public datasets,
runs them through the adapter, and asserts:

  * The returned probability dict has EXACTLY the expected labels (no missing,
    no extra, no reordered).
  * Every probability is finite, in [0, 1].
  * Probabilities sum to 1 within the JevBench RENORM_TOL band.

This is a structural pre-flight. It does not score correctness, only that the
adapter's transport is faithful.
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adapter"))
sys.path.insert(0, str(ROOT / "jevbench"))

from jevbench.tasks import load_jsonl, Task

JEVBENCH_ROOT = Path(__file__).resolve().parent.parent / "jevbench"
PUBLIC = JEVBENCH_ROOT / "datasets" / "public"


def load_all_public():
    """Return all public tasks with their tier metadata."""
    out = []
    for tier, fname in [("easy", "easy.jsonl"),
                        ("standard", "original.jsonl"),
                        ("hard", "hard.jsonl")]:
        path = PUBLIC / fname
        tasks = load_jsonl(str(path))
        for t in tasks:
            out.append((tier, t))
    return out


def pick_validation_set(tasks):
    """Pick one example per (question_type, family) seen across the public suite."""
    seen = {}
    for tier, t in tasks:
        key = (t.question["type"], t.family)
        if key not in seen:
            seen[key] = (tier, t)
    # Guarantee we always cover the bare basics:
    by_type = {}
    for tier, t in tasks:
        by_type.setdefault(t.question["type"], []).append((tier, t))
    picks = list(seen.values())
    # Add a few extras so each type has 3+
    for qtype in ("noul", "choice", "score"):
        for tier, t in by_type.get(qtype, [])[:3]:
            if (tier, t) not in picks:
                picks.append((tier, t))
    return picks


def assert_distribution(probs, expected_labels, total_tol=2e-2):
    assert set(probs.keys()) == set(expected_labels), (
        f"label set mismatch: probs={set(probs.keys())} expected={set(expected_labels)}"
    )
    for k, v in probs.items():
        assert isinstance(v, float), f"{k}: not float ({type(v)})"
        assert 0.0 <= v <= 1.0, f"{k}: out of [0,1] -> {v}"
        assert v == v, f"{k}: NaN"  # noqa: PLR0124
    total = sum(probs.values())
    assert abs(total - 1.0) <= total_tol, (
        f"sum {total} not within {total_tol} of 1.0"
    )


def main():
    from adapter.smalljev_adapter import SmalljevAdapter  # noqa: E402

    print("=" * 60)
    print("smalljev adapter validation (pre-flight)")
    print("=" * 60)

    all_tasks = load_all_public()
    print(f"Loaded {len(all_tasks)} public tasks")

    picks = pick_validation_set(all_tasks)
    print(f"Picked {len(picks)} validation tasks (one per (type,family) + extras)")

    # Adapter is configured by env vars in this script.
    backbone = os.environ.get("SMALLJEV_BACKBONE", "openbmb/MiniCPM5-2B-Base")
    adapter_id = os.environ.get("SMALLJEV_ADAPTER_ID")
    heads_ckpt = os.environ.get("SMALLJEV_HEADS_CKPT")
    variant_label = os.environ.get("SMALLJEV_VARIANT_LABEL", "smalljev-validation")

    adapter = SmalljevAdapter(
        backbone=backbone,
        adapter_id=adapter_id,
        heads_ckpt=heads_ckpt,
        variant_label=variant_label,
    )

    n_ok, n_fail = 0, 0
    failures = []
    for i, (tier, t) in enumerate(picks, 1):
        try:
            res = adapter.run(t)
            if not res.ok:
                raise RuntimeError(f"adapter returned ok=False: error={res.error}")
            labels = list(t.labels)
            assert_distribution(res.probs, labels)
            print(f"  [{i}/{len(picks)}] {tier}/{t.question['type']}/{t.family} "
                  f"id={t.id}: ok (latency={res.latency_s*1000:.1f}ms) "
                  f"argmax={max(res.probs, key=res.probs.get)} expected={t.expected}")
            n_ok += 1
        except Exception as e:
            n_fail += 1
            failures.append((tier, t.id, str(e)))
            print(f"  [{i}/{len(picks)}] {tier}/{t.question['type']}/{t.family} "
                  f"id={t.id}: FAIL ({type(e).__name__}: {e})")

    print()
    print(f"RESULT: {n_ok} ok, {n_fail} fail out of {len(picks)}")
    if failures:
        print("Failures:")
        for tier, tid, err in failures:
            print(f"  - {tier}/{tid}: {err}")
        sys.exit(1)
    else:
        print("All adapter transports valid. Ready for full run.")


if __name__ == "__main__":
    main()