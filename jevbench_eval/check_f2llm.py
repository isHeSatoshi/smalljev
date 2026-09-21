"""Check f2llm-4b hard tier partial results."""
import json
from pathlib import Path

p = Path(r"C:\Users\Aditya\AppData\Local\Temp\smalljev_jevbench_eval\runs\2026-09-19_f2llm_4b\hard\results.jsonl")
recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"records: {len(recs)}")
lats = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
print(f"latency p50={sorted(lats)[len(lats)//2]:.2f}s p95={sorted(lats)[int(len(lats)*0.95)]:.2f}s max={max(lats):.2f}s")
correct = sum(1 for r in recs if r["correct"])
print(f"correct: {correct}/{len(recs)}")
n_failed = sum(1 for r in recs if r["status"] == "failed")
print(f"failed: {n_failed}")
n_invalid = sum(1 for r in recs if r.get("status_code") == 422)
print(f"422 (over context): {n_invalid}")
# Estimate time to complete 111 items at this rate
remaining = 111 - len(recs)
total_time_so_far = sum(lats)
avg = total_time_so_far / len(lats) if lats else 0
print(f"avg latency: {avg:.2f}s/item")
print(f"estimated remaining time at avg: {remaining * avg / 60:.1f} min")