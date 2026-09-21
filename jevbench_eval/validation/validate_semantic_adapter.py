"""Validation: pick a small number of public JevBench tasks and run them through
the SmalljevSemanticAdapter. Asserts label-set equality, finiteness, [0,1]
range, sum-to-1, and that no task silently fails.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adapter"))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import fcntl  # noqa: F401
except ImportError:
    import _fcntl_shim  # noqa: F401

sys.path.insert(0, str(ROOT / "jevbench"))

from jevbench.tasks import load_jsonl  # noqa: E402

JEVBENCH_ROOT = ROOT / "jevbench"
PUBLIC = JEVBENCH_ROOT / "datasets" / "public"


def pick_validation_set():
    """Pick 6 tasks: 2 noul, 2 choice, 2 score from across tiers."""
    out = []
    by_type = {"noul": [], "choice": [], "score": []}
    for tier, fname in [("easy", "easy.jsonl"),
                        ("standard", "original.jsonl"),
                        ("hard", "hard.jsonl")]:
        for t in load_jsonl(str(PUBLIC / fname)):
            qtype = t.question["type"]
            if qtype in by_type and len([x for x in by_type[qtype] if x[0] == tier]) < 1:
                by_type[qtype].append((tier, t))
    for qtype in ("noul", "choice", "score"):
        out.extend(by_type[qtype][:2])
    return out


def assert_distribution(probs, expected_labels, total_tol=2e-2):
    assert set(probs.keys()) == set(expected_labels), (
        f"label set mismatch: probs={set(probs.keys())} expected={set(expected_labels)}"
    )
    for k, v in probs.items():
        assert isinstance(v, float), f"{k}: not float ({type(v)})"
        assert 0.0 <= v <= 1.0, f"{k}: out of [0,1] -> {v}"
        assert v == v, f"{k}: NaN"
    total = sum(probs.values())
    assert abs(total - 1.0) <= total_tol, f"sum {total} not within {total_tol} of 1.0"


def main():
    print("=" * 60)
    print("smalljev SEMANTIC adapter validation (pre-flight)")
    print("=" * 60)

    picks = pick_validation_set()
    print(f"Picked {len(picks)} validation tasks:")
    for tier, t in picks:
        print(f"  {tier}/{t.id}  type={t.question['type']}  family={t.family}  "
              f"labels={list(t.labels)}")

    from adapter.smalljev_semantic_adapter import SmalljevSemanticAdapter

    backbone = os.environ.get("SMALLJEV_BACKBONE", "openbmb/MiniCPM5-2B-Base")
    adapter_dir = os.environ.get("SMALLJEV_ADAPTER_DIR")
    scorer_ckpt = os.environ.get("SMALLJEV_SCORER_CKPT")
    noulscore_ckpt = os.environ.get("SMALLJEV_NOULSCORE_CKPT")
    variant_label = os.environ.get("SMALLJEV_VARIANT_LABEL", "smalljev_semantic_validation")
    if not (adapter_dir and scorer_ckpt and noulscore_ckpt):
        raise SystemExit("Set SMALLJEV_ADAPTER_DIR, SMALLJEV_SCORER_CKPT, "
                         "SMALLJEV_NOULSCORE_CKPT env vars first.")

    adapter = SmalljevSemanticAdapter(
        backbone=backbone, adapter_dir=adapter_dir, scorer_ckpt=scorer_ckpt,
        noulscore_ckpt=noulscore_ckpt, variant_label=variant_label,
    )

    print("\nLoading model...")
    adapter.load()
    print(f"  loaded in {adapter.load_s:.1f}s on {adapter._device_real}")

    n_ok = 0
    for tier, t in picks:
        try:
            r = adapter.run(t)
            print(f"\n{tier}/{t.id} type={t.question['type']}")
            print(f"  ok={r.ok} latency={r.latency_s*1000:.1f}ms error={r.error}")
            print(f"  probs={r.probs}")
            print(f"  expected_labels={list(t.labels)}")
            if not r.ok:
                print(f"  ERROR: {r.error}")
                continue
            assert_distribution(r.probs, list(t.labels))
            n_ok += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  EXCEPTION: {type(e).__name__}: {e}")

    print(f"\nValidation result: {n_ok}/{len(picks)} passed")
    if n_ok == len(picks):
        print("OK: ready for full JevBench run.")
    else:
        print("FAIL: fix the adapter before the full run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
