"""Laya-style benchmark figures: Jev vs Laya vs smalljev top-3.

All numbers are verified measurements or attributed publications (see BENCHMARK.md
and docs/ARCHITECTURE.md). N/A marks unevaluated cells — never interpolated.

Top-3 smalljev: minicpm5-lettered (champion), harrier-0.6b, f2llm-4b.
Outputs: evals/figs/benchmark_charts.png + evals/figs/benchmark_table.png
Usage: python evals/make_benchmark_figs.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import table as mpl_table
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

# ---- verified data (sources in BENCHMARK.md) ----
LAT = {  # k=1 / k=10 / k=50 ms
    "Jev (1Q avg/best)": (400.0, None, None, 150.0),
    "Laya": (38.4, 156.0, 721.4, None),
    "smalljev champion": (124.0, 141.7, 519.7, None),
    "smalljev mb-base": (39.1, 43.3, 51.7, None),
    "Laya (measured)": (51.9, 68.9, None, None),
}
ACC = {  # Overall / Intent / Moderation / FactCheck
    "Jev": (67.8, 96.0, 93.5, 80.0),
    "Laya": (83.8, 99.2, 96.7, 88.3),
    "smalljev best": (94.2, 96.25, 61.7, None),  # AG/SST macro (.936+.948)/2; holdout; TweetEval-hate
}
FAMILIES = [  # smalljev best-arm per family (system in parens in labels)
    ("AGNews topic (.936 harrier)", 93.6),
    ("SST-2 sentiment (.948 crown)", 94.8),
    ("DBpedia-14 (1.000 mb-base)", 100.0),
    ("Banking episodes (.9625 lettered)", 96.25),
    ("Noul SST (.973 ettin-1b)", 97.3),
    ("Yelp 5-star (.685 f2llm)", 68.5),
    ("Transfer unseen (.583)", 58.3),
    ("TweetEval hate (.617)", 61.7),
]
LAYA_CURVE = [(30, 95.0), (40, 93.7), (50, 92.2), (60, 91.5),
              (70, 90.7), (80, 89.5), (90, 86.5), (100, 83.8)]
OURS_CURVE = [(10, 100.0), (40, 97.0), (70, 97.0), (100, 92.0)]  # SST-2 slice


def charts():
    fig, ax = plt.subplots(2, 2, figsize=(17, 12))
    fig.suptitle("smalljev vs. Laya vs. TypeSafe Jev: Benchmark & Performance",
                 fontsize=18, fontweight="bold")
    # P1 latency
    a = ax[0, 0]
    labels = ["Jev\n(1Q avg)", "Jev\n(1Q best)", "Laya\n(1Q p50)", "Laya\n(10Q)",
              "Laya\n(50Q)", "Champ\n(1Q)", "Champ\n(10Q)", "Champ\n(50Q)",
              "MB-base\n(1Q)", "MB-base\n(10Q)", "MB-base\n(50Q)"]
    vals = [400.0, 150.0, 38.4, 156.0, 721.4, 124.0, 141.7, 519.7, 39.1, 43.3, 51.7]
    cols = (["#e74c3c", "#e67e22"] + ["#2980d9"] * 3 + ["#27ae60"] * 3 + ["#8e44ad"] * 3)
    bars = a.bar(labels, vals, color=cols, edgecolor="black")
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{v}",
               ha="center", fontsize=8, fontweight="bold")
    a.set_title("Inference Latency: 1Q and Batched Execution", fontweight="bold")
    a.set_ylabel("Latency in Milliseconds (Lower is Better)", fontweight="bold")
    a.tick_params(axis="x", labelsize=8, rotation=20)
    # P2 head-to-head
    a = ax[0, 1]
    groups = ["Overall Avg", "Intent Routing", "Moderation", "Fact Checking"]
    jev = [67.8, 96.0, 93.5, 80.0]
    laya = [83.8, 99.2, 96.7, 88.3]
    ours = [90.2, 96.25, 61.7, np.nan]
    x = np.arange(len(groups))
    w = 0.25
    for i, (s, c) in enumerate(zip([jev, laya, ours], ["#95a5a6", "#2980d9", "#27ae60"])):
        bars = a.bar(x + (i - 1) * w, s, w, label=["Jev (Published)", "Laya", "smalljev best"][i],
                     color=c, edgecolor="black")
        for b, v in zip(bars, s):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                a.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{v}",
                       ha="center", fontsize=8, fontweight="bold")
    a.set_xticks(x)
    a.set_xticklabels(groups, fontsize=9)
    a.set_title("Head-to-Head Accuracy (different tasks per column!)", fontweight="bold")
    a.set_ylabel("Accuracy (%)", fontweight="bold")
    a.legend(fontsize=9)
    a.text(3, 5, "smalljev FactCheck: N/A", fontsize=8, style="italic")
    # P3 task families (ours)
    a = ax[1, 0]
    names = [n for n, _ in FAMILIES][::-1]
    vals = [v for _, v in FAMILIES][::-1]
    bars = a.barh(names, vals, color="#27ae60", edgecolor="black")
    for b, v in zip(bars, vals):
        a.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v}%", va="center",
               fontsize=8, fontweight="bold")
    a.set_xlim(0, 112)
    a.set_title("smalljev Task Families (best arm each, n=200-1600)", fontweight="bold")
    a.set_xlabel("Accuracy (%)", fontweight="bold")
    a.tick_params(axis="y", labelsize=8)
    # P4 selective
    a = ax[1, 1]
    lx, ly = zip(*LAYA_CURVE)
    ox, oy = zip(*OURS_CURVE)
    a.plot(lx, ly, "o-", color="#2980d9", label="Laya Accuracy @ Coverage")
    a.plot(ox, oy, "s-", color="#27ae60", label="smalljev SST-2 slice @ Coverage")
    a.axhline(83.8, color="gray", linestyle=":", label="Laya baseline 100% (83.8%)")
    a.axvline(50, color="red", linestyle="--", label="50% gate")
    a.set_xlim(20, 105)
    a.set_ylim(80, 101)
    a.set_title("Selective Automation: Accuracy vs. Coverage", fontweight="bold")
    a.set_xlabel("Traffic Coverage / Automation Rate (%)", fontweight="bold")
    a.set_ylabel("Accuracy on Automated Decisions (%)", fontweight="bold")
    a.legend(fontsize=8)
    fig.text(0.5, 0.01, "Sources: TypeSafe blog + evals.typesafe.ai; Laya card + eval/results.json; "
             "smalljev evals/*.json. Cross-column tasks differ — shape, not ranking.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = os.path.join(OUT, "benchmark_charts.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


def table_fig():
    rows = [
        ["P50 latency (1Q)", "~400 ms avg", "38.4 ms", "124.0 ms", "106.0 ms", "39.1 ms"],
        ["Batched latency (10Q)", "~1,500 / ~400 ms", "156.0 ms", "141.7 ms", "—", "43.3 ms"],
        ["Batched latency (50Q)", "multi-s / limited", "721.4 ms", "519.7 ms", "—", "51.7 ms"],
        ["Benchmark accuracy", "67.8% (4 wf)", "83.8% macro", "90.2% AG/SST", "94.0% AG/SST", "83.4% AG/SST"],
        ["Intent & routing", "~95-98% agree", "99.1% (ECE .009)", "96.25% holdout", "—", "—"],
        ["Moderation", "~92-95% agree", "96.7% (ECE .061)", "61.7% TweetEval*", "—", "—"],
        ["Noul calibration", "w/ confidence", "churn-type ~0.99", "0.927 (v4 joint)", "0.927", "0.873"],
        ["Score ordinal", "rubric + dist", "—", "Yelp .60 / MAE .44 (v4 joint)", "Yelp .58", "Yelp .52"],
        ["Zero-shot transfer", "—", "0.651 / ECE .207", "holdout .941", "transfer .500", "transfer .333"],
        ["Selective @50% cov", "escalation", "92.2% (ECE .041)", "SST slice ~0.97", "—", "—"],
        ["Weights & code", "closed API", "Apache-2.0", "Apache-2.0/MIT", "MIT (+Apache base)", "Apache-2.0"],
        ["Inference cost", "$0.042/1M in", "$0 self-host", "$0 self-host", "$0 self-host", "$0 self-host"],
        ["VRAM (measured)", "—", "—", "5.0GB", "1.7GB", "0.6GB"],
    ]
    cols = ["Metric / Dimension", "TypeSafe Jev\n(Published)", "Laya\n(Checkpoint)",
            "smalljev\nChampion", "smalljev\nHarrier-0.6B", "smalljev\nMB-base"]
    fig, a = plt.subplots(figsize=(20, 9.5))
    a.axis("off")
    a.set_title("Benchmark table: Jev vs Laya vs smalljev top-3 (see BENCHMARK.md for methods)",
                fontsize=15, fontweight="bold", pad=12)
    t = mpl_table(a, cellText=[[c for c in r] for r in rows],
                  colLabels=cols, loc="center", colWidths=[0.24, 0.15, 0.15, 0.15, 0.15, 0.16])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold")
    fig.text(0.5, 0.02, "* TweetEval-hate is out-of-distribution for all open systems tried. "
             "AG/SST macro for smalljev; Jev/Laya columns use their own tasks. — = not evaluated.",
             ha="center", fontsize=9, style="italic")
    p = os.path.join(OUT, "benchmark_table.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    charts()
    table_fig()
