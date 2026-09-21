"""Save representative failure examples per variant for the report."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import fcntl  # noqa: F401
except ImportError:
    import _fcntl_shim  # noqa: F401
sys.path.insert(0, str(ROOT / "jevbench"))
from jevbench.tasks import load_jsonl  # noqa: E402


def load_records(run_dir, tier):
    p = Path(run_dir) / tier / "results.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_tasks():
    out = {}
    for fp in ["easy.jsonl", "original.jsonl", "hard.jsonl"]:
        for t in load_jsonl(str(ROOT / "jevbench" / "datasets" / "public" / fp)):
            out[t.id] = t
    return out


def main():
    tasks = load_tasks()
    failures = {}
    for variant in ["v4_bundle", "v5_holdout", "crown", "lettered"]:
        failures[variant] = {}
        for tier in ("easy", "standard", "hard"):
            recs = load_records(ROOT / "runs" / f"2026-09-19_{variant}", tier)
            wrong = [r for r in recs if not r["correct"] and r["valid"]]
            ex = []
            for r in wrong[:5]:
                t = tasks.get(r["task_id"])
                if t is None:
                    continue
                ex.append({
                    "id": r["task_id"],
                    "family": r["family"],
                    "qtype": t.question["type"],
                    "state_first_200": (t.state if isinstance(t.state, str) else json.dumps(t.state))[:200],
                    "instructions": t.question["instructions"][:150],
                    "labels": t.labels,
                    "expected": t.expected,
                    "predicted": r["predicted"],
                    "probs": r["probs"],
                    "confidence": max(r["probs"].values()) if r["probs"] else None,
                    "latency_s": r["latency_s"],
                })
            failures[variant][tier] = {
                "n_wrong": len(wrong),
                "n_total": len(recs),
                "examples": ex,
            }
    out = ROOT / "runs" / "ERROR_ANALYSIS.json"
    out.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()