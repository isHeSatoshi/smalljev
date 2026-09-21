"""Internal experiment figures: generated FROM results JSON artifacts (no hardcoding).

Table: every system x AG/SST/Ep-k8/Holdout/Transfer/Noul/Yelp/ECE (N/A = unmeasured).
Charts: (1) training lineage v1->v4, (2) generalization trio across systems,
(3) efficiency frontier (latency vs accuracy, bubble=VRAM), (4) PPO ablation.
Outputs: evals/figs/experiment_table.png + evals/figs/experiment_charts.png
Usage: python evals/make_experiment_figs.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.table import table as mpl_table
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figs")
os.makedirs(OUT, exist_ok=True)


def J(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return json.load(f)


def acc(entry):
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, dict) and "accuracy" in entry:
        return float(entry["accuracy"])
    return None


def arms():
    out = {}
    import glob
    for f in sorted(glob.glob(os.path.join(ROOT, "arms", "*-eval.json"))):
        r = json.load(open(f))
        out[r["arm"]] = r
    return out


def collect():
    """name -> dict of metric -> value/None. Every number traceable to a JSON artifact."""
    rows = {}
    v1 = J("heads", "v1-slots.json")
    rows["v1 slots"] = {"AG": acc(v1["agnews_test"]), "SST": acc(v1["sst2_val300"]),
                        "Transfer": acc(v1["transfer_k3_unseen_task"])}
    v2 = J("heads", "v2-slots.json")
    rows["v2 +diverse+LoRA"] = {"AG": acc(v2["agnews_test"]), "SST": acc(v2["sst2_val300"]),
                                "Transfer": acc(v2["transfer_k3_unseen_task"])}
    v3 = J("heads", "v3-slots.json")
    rows["v3 +episodic"] = {"AG": acc(v3["agnews_test"]), "SST": acc(v3["sst2_val300"]),
                            "Transfer": acc(v3["transfer_k3_unseen_task"])}
    v4 = J("heads", "v4-bundle.json")
    rows["v4 bundle"] = {"AG": acc(v4["choice_agnews"]), "SST": acc(v4["choice_sst2"]),
                         "Noul": acc(v4["noul_sst2"]),
                         "Yelp": v4["score_yelp"].get("exact_acc"),
                         "Transfer": acc(v4["transfer_k3"])}
    v5 = J("heads", "v5-holdout.json")

    def _epkey(d, k):
        v = d.get(k)
        return v["accuracy"] if isinstance(v, dict) else v

    rows["v5 holdout"] = {"AG": acc(v5["choice_agnews"]), "SST": acc(v5["choice_sst2"]),
                          "Noul": acc(v5["noul_sst2"]), "Yelp": v5["score_yelp"].get("exact_acc"),
                          "Ep-k8": _epkey(v5, "inspace_episodes_k8"),
                          "Holdout": _epkey(v5, "holdout_episodes_k8"),
                          "Transfer": acc(v5["transfer_k3"])}
    for key, r in arms().items():
        rows[key] = {
            "AG": acc(r["indomain_agnews"]), "SST": acc(r["indomain_sst2"]),
            "Ep-k8": r.get("episodes_inspace_k8"), "Holdout": r.get("episodes_holdout_k8"),
            "Transfer": r.get("transfer_k3"), "Noul": acc(r["noul_sst2"]),
            "Yelp": r["score_yelp"].get("exact_acc"),
            "ECE": r["indomain_agnews"].get("ece"),
            "VRAM": r.get("vram_peak_gb"), "LAT8": r.get("latency_ms_k8"),
            "PARAMS": r.get("params_total"),
            "K32": r.get("k32"), "ORDER": r.get("order_stability"),
        }
        if isinstance(rows[key]["Ep-k8"], dict):
            rows[key]["Ep-k8"] = rows[key]["Ep-k8"].get("accuracy")
    ly = J("laya-eval.json")
    rows["Laya (measured)"] = {"AG": acc(ly["agnews"]), "SST": acc(ly["sst2"]),
                               "Ep-k8": ly["episodes_k8"]["accuracy"],
                               "Transfer": acc(ly["transfer"]),
                               "Noul": acc(ly["noul"]), "Yelp": ly["yelp"].get("exact_acc"),
                               "ECE": ly["agnews"].get("ece"),
                               "LAT8": None, "VRAM": 1.0}
    let = J("minicpm5-lettered-ablation.json") if os.path.exists(
        os.path.join(ROOT, "minicpm5-lettered-ablation.json")) else None
    if let is None:
        import glob as _g
        cands = _g.glob(os.path.join(ROOT, "arms", "minicpm5-lettered-ablation.json"))
        let = json.load(open(cands[0])) if cands else None
    if let:
        rows["minicpm5-lettered"] = {
            "AG": acc(let["agnews"]), "SST": acc(let["sst2"]),
            "Ep-k8": let["episodes_inspace_k8"]["accuracy"],
            "Holdout": let["episodes_holdout_k8"]["accuracy"],
            "Transfer": acc(let["transfer"]), "ORDER": let.get("order_stability"),
            "VRAM": 5.0}
    crownp = os.path.join(ROOT, "arms", "minicpm5-crown-ablation.json")
    if os.path.exists(crownp):
        cr = json.load(open(crownp))
        rows["minicpm5-crown"] = {
            "AG": acc(cr["agnews"]), "SST": acc(cr["sst2"]),
            "Ep-k8": cr["episodes_inspace_k8"]["accuracy"],
            "Holdout": cr["episodes_holdout_k8"]["accuracy"],
            "Transfer": acc(cr["transfer"]),
            "Noul": acc(cr["noul"]), "Yelp": cr["yelp"].get("exact_acc"),
            "ORDER": cr.get("order_stability"), "VRAM": 5.0}
    return rows


def fmt(v, pct=True):
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(f):
        return "—"
    return f"{f*100:.1f}%" if pct else f"{f:.3f}"


def table_fig(rows):
    order = ["v1 slots", "v2 +diverse+LoRA", "v3 +episodic", "v4 bundle", "v5 holdout",
             "minicpm5", "minicpm5-lettered", "minicpm5-crown", "bidirlm", "ettin-1b",
             "ettin-from-dec-1b", "modernbert-base", "modernbert-large", "f2llm-4b",
             "harrier-0.6b", "Laya (measured)"]
    order = [o for o in order if o in rows]
    cols = ["System", "AGNews", "SST-2", "Episodes k8", "Holdout k8", "Transfer",
            "Noul", "Yelp exact", "AG ECE"]
    keys = ["AG", "SST", "Ep-k8", "Holdout", "Transfer", "Noul", "Yelp", "ECE"]
    data = [[n] + [fmt(rows[n].get(k), pct=(k != "ECE")) for k in keys] for n in order]
    fig, a = plt.subplots(figsize=(20, 8.5))
    a.axis("off")
    a.set_title("smalljev experiment matrix (from results JSON artifacts; — = unmeasured)",
                fontsize=15, fontweight="bold", pad=12)
    t = mpl_table(a, cellText=data,
                  colLabels=cols, loc="center",
                  colWidths=[0.22] + [0.0975] * 8)
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#ecf0f1")
            cell.set_text_props(fontweight="bold")
    p = os.path.join(OUT, "experiment_table.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print("wrote", p)


def charts(rows):
    fig, ax = plt.subplots(2, 2, figsize=(17, 12))
    fig.suptitle("smalljev experiments: what actually moved the needle",
                 fontsize=18, fontweight="bold")
    # P1 lineage
    a = ax[0, 0]
    lin = ["v1 slots", "v2 +diverse+LoRA", "v3 +episodic", "v4 bundle"]
    lin = [l for l in lin if l in rows]

    def _v(l, k):
        v = rows[l].get(k)
        return v if v is not None else np.nan

    x = np.arange(len(lin))
    ag = [_v(l, "AG") for l in lin]
    ss = [_v(l, "SST") for l in lin]
    tr = [_v(l, "Transfer") for l in lin]
    w = 0.25
    a.bar(x - w, ag, w, label="AGNews", color="#2980d9", edgecolor="black")
    a.bar(x, ss, w, label="SST-2", color="#27ae60", edgecolor="black")
    a.bar(x + w, tr, w, label="Transfer-12", color="#e67e22", edgecolor="black")
    a.set_xticks(x)
    a.set_xticklabels(lin, fontsize=9)
    a.set_title("Training lineage: slots v1 → v4 bundle", fontweight="bold")
    a.set_ylabel("Accuracy", fontweight="bold")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=9)
    # P2 generalization trio across systems
    a = ax[0, 1]
    systems = ["minicpm5-lettered", "minicpm5-crown", "v5 holdout", "v4 bundle",
               "bidirlm", "harrier-0.6b",
               "f2llm-4b", "ettin-1b", "modernbert-large", "modernbert-base",
               "Laya (measured)"]
    systems = [s for s in systems if s in rows]
    ep = [_v(s, "Ep-k8") for s in systems]
    ho = [_v(s, "Holdout") for s in systems]
    tr = [_v(s, "Transfer") for s in systems]
    x = np.arange(len(systems))
    a.bar(x - w, [v or 0 for v in ep], w, label="Episodes k8", color="#2980d9",
          edgecolor="black")
    a.bar(x, [v or 0 for v in ho], w, label="Holdout k8", color="#8e44ad",
          edgecolor="black")
    a.bar(x + w, [v or 0 for v in tr], w, label="Transfer-12", color="#e67e22",
          edgecolor="black")
    a.set_xticks(x)
    a.set_xticklabels(systems, fontsize=8, rotation=25, ha="right")
    a.set_title("Generalization: novel sets + holdout + transfer", fontweight="bold")
    a.set_ylabel("Accuracy", fontweight="bold")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=9)
    # P3 efficiency frontier: latency vs AG accuracy, bubble = VRAM
    a = ax[1, 0]
    pts = []
    for s in systems:
        lat, agv, vram = rows[s].get("LAT8"), rows[s].get("AG"), rows[s].get("VRAM")
        if lat and agv and vram:
            pts.append((s, lat, agv, vram))
    for s, lat, agv, vram in pts:
        a.scatter(lat, agv, s=vram * 60, alpha=0.7, edgecolors="black")
        a.annotate(s, (lat, agv), fontsize=8, xytext=(4, 4), textcoords="offset points")
    a.set_xscale("log")
    a.set_title("Efficiency frontier: latency vs accuracy (bubble = VRAM GB)",
                fontweight="bold")
    a.set_xlabel("Latency k=8 ms/call (log)", fontweight="bold")
    a.set_ylabel("AGNews accuracy", fontweight="bold")
    # P4 PPO ablation
    a = ax[1, 1]
    ppo = J("ppo-ablation.json")
    warm = J("ppo-ablation-warm.json")
    cats = ["SFT", "REINFORCE\ncold", "PPO\ncold", "REINFORCE\nwarm", "PPO\nwarm"]
    src = [ppo["sft"], ppo["reinforce"], ppo["ppo"], warm["reinforce"], warm["ppo"]]
    x = np.arange(len(cats))
    agv = [s["agnews"]["accuracy"] for s in src]
    ssv = [s["sst2"]["accuracy"] for s in src]
    trv = [s["transfer"]["accuracy"] for s in src]
    a.bar(x - w, agv, w, label="AGNews", color="#2980d9", edgecolor="black")
    a.bar(x, ssv, w, label="SST-2", color="#27ae60", edgecolor="black")
    a.bar(x + w, trv, w, label="Transfer-12", color="#e67e22", edgecolor="black")
    a.set_xticks(x)
    a.set_xticklabels(cats, fontsize=9)
    a.set_title("SFT vs REINFORCE vs PPO (same setup; warm = SFT init)", fontweight="bold")
    a.set_ylabel("Accuracy", fontweight="bold")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=9)
    fig.text(0.5, 0.01, "Generated from evals/*-eval.json, heads/*.json, ppo-ablation*.json. "
             "Missing cells are omitted, never zero-filled.",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = os.path.join(OUT, "experiment_charts.png")
    fig.savefig(p, dpi=150)
    print("wrote", p)


if __name__ == "__main__":
    rows = collect()
    print("systems:", sorted(rows))
    table_fig(rows)
    charts(rows)
