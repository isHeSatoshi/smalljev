"""Clean benchmark table: Nimble vs Jev (published) vs smalljev-crown vs Laya (measured).

Layout mirrors bespokelabsai/nimble's figure. Green = higher within each published
pair (Nimble-vs-Jev, transcribed) and within each measured pair (crown-vs-Laya,
computed). Stars: upcoming McNemar (crown vs Laya, paired same items); daggers:
their published Nimble-vs-Jev significance. Wilson CIs in the JSON, not the figure.

Usage: python evals/make_nimble13_fig.py
Out: evals/figs/nimble13_table.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import table as mpl_table
import seaborn as sns

sns.set_theme(style="whitegrid")

TASKS = [
    ("Intent routing, 18 scenarios", "MASSIVE en-US", "massive-en-US", 86.9, 87.4, False),
    ("Same utterances, German", "MASSIVE de-DE", "massive-de-DE", 83.4, 86.9, True),
    ("Entailment", "MultiNLI", "multinli", 85.3, 82.9, False),
    ("Medical yes/no/maybe", "PubMedQA", "pubmedqa", 75.6, 77.2, False),
    ("Contrastive fact verification", "VitaminC", "vitaminc-dev", 76.6, 80.1, True),
    ("Yes/no over a passage", "BoolQ", "boolq", 86.0, 89.7, False),
    ("Passage answerability", "SQuAD 2.0", "squad2", 80.6, 82.9, False),
    ("Adversarial paraphrase", "PAWS", "paws", 82.8, 89.2, True),
    ("Toxicity moderation", "Civil Comments", "civil_comments", 70.3, 81.0, True),
    ("Prompt safety", "Aegis 2.0", "aegis2", 81.2, 80.4, False),
    ("Helpfulness 0-4, exact", "HelpSteer2", "helpsteer2", 39.0, 34.1, False),
    ("Summary relevance 1-5", "SummEval", "summeval-relevance", 49.2, 35.0, True),
    ("Summary consistency 1-5", "SummEval", "summeval-consistency", 75.7, 81.2, False),
]


def main():
    R = json.load(open("evals/nimble13.json"))["subsets"]
    rows, winners = [], []
    for task, ds, key, nim, jev, pubstar in TASKS:
        s = R[key]
        n = s["n"]
        cr = s["crown"]["accuracy"] * 100
        le = s["lettered"]["accuracy"] * 100 if "lettered" in s else None
        v5 = s["v5"]["accuracy"] * 100 if "v5" in s else None
        ly = s["laya"]["accuracy"] * 100
        p = s.get("mcnemar", {}).get("crown_vs_laya_p")
        star = "*" if (p is not None and p < 0.05) else ""
        dag = "+" if pubstar else ""
        row = [task, ds, str(n), f"{nim:.1f}%", f"{jev:.1f}%{dag}",
               f"{cr:.1f}%", f"{le:.1f}%" if le is not None else "—",
               f"{v5:.1f}%" if v5 is not None else "—",
               f"{ly:.1f}%", star]
        rows.append(row)
        meas = [x for x in (cr, le, ly) if x is not None]
        winners.append((nim >= jev, max(meas) if meas else 0))
    fig, a = plt.subplots(figsize=(20, 9))
    a.axis("off")
    a.set_title("smalljev crown/lettered/v5 + Laya (measured) vs Nimble vs Jev (published) — 13 public benchmarks",
                fontsize=15, fontweight="bold", pad=12)
    cols = ["Task", "Dataset", "n", "Nimble\n(publ.)", "Jev\n(publ.)",
            "crown\n(meas.)", "lettered\n(meas.)", "v5\n(meas.)", "Laya\n(meas.)",
            "Sig*"]
    t = mpl_table(a, cellText=rows, colLabels=cols, loc="center",
                  colWidths=[0.24, 0.10, 0.05, 0.075, 0.075, 0.08, 0.08, 0.08, 0.08, 0.04])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r >= 1 and c in (3, 4, 5, 6, 7, 8):
            pub_win, meas_best = winners[r - 1]
            is_best = False
            try:
                vals = {"Nimble": TASKS[r - 1][3], "Jev": TASKS[r - 1][4]}
                if c == 3:
                    is_best = vals["Nimble"] >= vals["Jev"]
                elif c == 4:
                    is_best = vals["Jev"] > vals["Nimble"]
                else:
                    cellval = float(rows[r - 1][c].rstrip("%"))
                    is_best = abs(cellval - meas_best) < 1e-9
            except (ValueError, IndexError):
                is_best = False
            if is_best:
                cell.set_facecolor("#d5f5e3")
    p = "evals/figs/nimble13_table.png"
    fig.text(0.5, 0.02, "* = McNemar crown-vs-Laya p<0.05 (paired, same items); "
             "dagger = their published Nimble-vs-Jev significance; green = higher within pair. "
             "Nimble/Jev aggregates cannot be significance-tested without per-example data.",
             ha="center", fontsize=9, style="italic")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", p)


if __name__ == "__main__":
    import os as _os
    _os.makedirs("evals/figs", exist_ok=True)
    main()
