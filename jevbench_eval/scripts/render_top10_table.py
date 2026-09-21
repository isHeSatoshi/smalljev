"""Render a JevBench v1.2 top-10 + smalljev comparison table as a PNG.

Top-10 rows are taken verbatim from `jevbench/results/v1.2/jevbench-v1.2-results.json`
(published numbers). The smalljev rows are measured in this evaluation and are
labelled (public-only) to make the comparison transparent.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent

PUB = json.load(open(ROOT / "jevbench" / "results" / "v1.2" / "jevbench-v1.2-results.json"))
PUB_ROWS = sorted(PUB["systems"], key=lambda r: r.get("jevbench_score") or 0, reverse=True)
TOP10 = [r for r in PUB_ROWS if r.get("rank") is not None][:10]


def fmt_score(v):
    return f"{v:.1f}" if isinstance(v, (int, float)) else "—"


def fmt_pct(v):
    return f"{v*100:.0f}%" if isinstance(v, (int, float)) else "—"


def native_label(r):
    srcs = r.get("probability_source", [])
    if "verbalized" in srcs:
        return "verbalized"
    if "label_only_no_calibrated_distribution" in srcs:
        return "label-only"
    return "native"


rows_table = []
for r in TOP10:
    rows_table.append({
        "rank": r["rank"],
        "name": r["display"],
        "src": native_label(r),
        "endpoint": r.get("endpoint_kind", "?"),
        "hard": r["tiers"].get("hard"),
        "score": r.get("jevbench_score"),
    })

# Append smalljev variants from this evaluation (sorted by public-only score desc).
summaries = {}
for d in sorted((ROOT / "runs").iterdir()):
    sp = d / "summary.json"
    if sp.exists():
        summaries[d.name] = json.loads(sp.read_text())

smalljev_rows = []
for name, s in sorted(summaries.items(),
                     key=lambda kv: kv[1]["jevbench_score"], reverse=True):
    variant = name.replace("2026-09-19_", "")
    smalljev_rows.append({
        "rank": "—",
        "name": f"smalljev {variant}",
        "src": "native",
        "endpoint": "self-host GPU (ours)",
        "hard": s["per_tier"]["hard"]["accuracy"],
        "score": s["jevbench_score"],
    })


import pandas as pd
df_pub = pd.DataFrame(rows_table)
df_oj = pd.DataFrame(smalljev_rows)
df = pd.concat([df_pub, df_oj], ignore_index=True)

df["hard_str"] = df["hard"].apply(fmt_pct)
df["score_str"] = df["score"].apply(fmt_score)
df["rank_str"] = df["rank"].apply(lambda v: str(v) if v != "—" else "—")

display = df[["rank_str", "name", "src", "endpoint", "hard_str", "score_str"]].copy()
display.columns = ["#", "System", "Prob", "Endpoint", "Hard", "JevBench Score"]

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis("off")

n_pub = len(df_pub)
n_total = len(display)

table = ax.table(
    cellText=display.values,
    colLabels=list(display.columns),
    cellLoc="left",
    loc="center",
    colWidths=[0.04, 0.46, 0.07, 0.15, 0.07, 0.13],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.4)

for j in range(len(display.columns)):
    cell = table[(0, j)]
    cell.set_facecolor("#1F2A44")
    cell.set_text_props(color="white", weight="bold", fontsize=12)

for i in range(n_total):
    is_oj = i >= n_pub
    for j in range(len(display.columns)):
        cell = table[(i + 1, j)]
        if is_oj:
            cell.set_facecolor("#FFE7B0")
        else:
            cell.set_facecolor("#FFFFFF" if i % 2 == 0 else "#F4F6FB")
        cell.set_edgecolor("#CCCCCC")

for i in range(n_total):
    cell = table[(i + 1, 5)]
    cell.set_text_props(weight="bold")

for i in range(n_pub, n_total):
    cell = table[(i + 1, 1)]
    cell.set_text_props(weight="bold")

ax.set_title(
    "JevBench v1.2 — current top-10 (published) + smalljev (this evaluation, public-only)",
    fontsize=14,
    weight="bold",
    pad=14,
    loc="left",
)
fig.text(
    0.5, 0.02,
    "Top-10 numbers: jevbench v1.2 published leaderboard (results/v1.2/jevbench-v1.2-results.json), "
    "ranked on full 220-item hard tier + 242-item standard+judge speed. "
    "smalljev rows: 111 hard items only (109 held out, not accessible) — lower bound. "
    "Adapter: native (slot softmax / Noul bit / Score levels), zero generated tokens.",
    ha="center",
    fontsize=8,
    style="italic",
    color="#555555",
    wrap=True,
)

legend_handles = [
    mpatches.Patch(facecolor="#FFFFFF", edgecolor="#CCCCCC", label="Published v1.2 top-10"),
    mpatches.Patch(facecolor="#FFE7B0", edgecolor="#CCCCCC",
                   label="smalljev (this evaluation, public-only)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=False)

plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.10)
out = ROOT / "runs" / "TOP10_PLUS_SMALLJEV.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"-> {out}")