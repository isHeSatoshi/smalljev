"""Final analysis: JSON battery results -> metrics, stats, calibration, selective.

Reads evals/final/{system}.json. Writes evals/final/analysis-{system}.json and
evals/final/cal-{system}.json (fitted temp+vec per k-class, dev ONLY).
Shareable helpers used by final_plots.py and FINAL_EVALUATION assembly.

Usage: python evals/final_analysis.py --system mb-base
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from smalljev.stats import (auroc_score, ece_conf, mcnemar_p, reliability_bins,
                            risk_coverage, wilson_ci)
from smalljev.calibration import (apply_temperature, apply_vector_scale,
                                  fit_temperature, fit_vector_scale)

COVERAGES = (0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0)
K_TASKS = {"agnews": 4, "ag_dev": 4, "sst2": 2, "ss_dev": 2, "dbpedia": 14,
           "k2": 2, "k14": 14, "k4": 4, "paraphrase": None, "transfer": 3,
           "world": None, "order": None}


def recs_of(R, task):
    return R["records"].get(task, [])


def as_arrays(recs):
    y = np.array([r["y"] for r in recs])
    pred = np.array([r["pred"] if r["pred"] is not None else -1 for r in recs])
    P = [np.asarray(r["probs"]) if r["probs"] is not None else None for r in recs]
    return y, pred, P


def acc_stats(y, pred):
    mask = pred >= 0
    c = int(((pred == y) & mask).sum())
    n = int(mask.sum())
    lo, hi = wilson_ci(c, n)
    return {"n": n, "correct": c, "accuracy": c / n if n else float("nan"),
            "wilson_lo": lo, "wilson_hi": hi}


def prob_stats(y, P):
    """NLL/Brier/ECE/AUROC over records with prob vectors (single k)."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    pred = P.argmax(1)
    correct = (pred == y).astype(int)
    conf = P.max(1)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1.0
    return {"nll": float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
            "brier": float(np.mean((P - onehot) ** 2)),
            "ece": ece_conf(conf, correct),
            "auroc": auroc_score(conf, correct),
            "mean_conf": float(conf.mean())}


def fit_calibration(R):
    """temp+vec per k-class on DEV records only. Returns params dict."""
    out = {}
    for dev_task, k in (("ag_dev", 4), ("ss_dev", 2)):
        recs = recs_of(R, dev_task)
        if len(recs) < 10:
            continue
        L = np.log(np.clip(np.stack([r["probs"] for r in recs]), 1e-12, 1.0))
        Y = np.array([r["y"] for r in recs])
        t = fit_temperature(L, Y)
        t2, b = fit_vector_scale(L, Y)
        out[f"k{k}"] = {"temp": t, "vec_temp": t2, "vec_bias": [float(x) for x in b],
                        "dev_n": len(recs)}
    # binary (noul_dev): temperature on 2-class logprobs
    recs = recs_of(R, "noul_dev") if "noul_dev" in R["records"] else []
    if len(recs) >= 10:
        P = np.stack([[1 - p[1], p[1]] if len(p) == 2 else p for p in
                      (np.asarray(r["probs"]) for r in recs)])
        # stored binary probs are [1-p, p]; rebuild 2-class matrix
        Y = np.array([r["y"] for r in recs])
        t = fit_temperature(np.log(np.clip(P, 1e-12, 1.0)), Y)
        out["binary"] = {"temp": t, "dev_n": len(recs)}
    # yelp_dev ordinal (5 levels)
    recs = recs_of(R, "yelp_dev") if "yelp_dev" in R["records"] else []
    if len(recs) >= 10:
        L = np.log(np.clip(np.stack([r["probs"] for r in recs]), 1e-12, 1.0))
        Y = np.array([r["y"] for r in recs])
        t = fit_temperature(L, Y)
        out["k5"] = {"temp": t, "dev_n": len(recs)}
    return out


def apply_cal(P, cal, kkey):
    P = np.asarray(P, dtype=float)
    if kkey not in cal:
        return P
    c = cal[kkey]
    if "vec_bias" in c:
        return apply_vector_scale(np.log(np.clip(P, 1e-12, 1.0)),
                                  c["vec_temp"], np.array(c["vec_bias"]))
    return apply_temperature(np.log(np.clip(P, 1e-12, 1.0)), c["temp"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    args = ap.parse_args()
    with open(f"evals/final/{args.system}.json") as f:
        R = json.load(f)
    A = {"system": args.system, "tasks": {}, "selective": {}, "reliability": {},
         "order": {}, "paraphrase": {}, "cardinality": {}, "latency": {},
         "calibration": {}, "auroc": {}, "world": {}, "transfer": {}}

    def task_acc(name):
        recs = recs_of(R, name)
        y = np.array([r["y"] for r in recs])
        pred = np.array([r["pred"] if r["pred"] is not None else -1 for r in recs])
        return acc_stats(y, pred), y, pred, recs

    for name in ("agnews", "sst2", "dbpedia", "paraphrase", "transfer", "world",
                 "k2", "k14", "k32", "k64", "ep_inspace", "ep_holdout",
                 "yelp", "noul_sst", "noul_hate"):
        if name in R["records"]:
            st, _, _, _ = task_acc(name)
            A["tasks"][name] = st
    # prob-based metrics where prob vectors exist (single-k tasks)
    for name, kk in (("agnews", "k4"), ("sst2", "k2"), ("dbpedia", None),
                     ("yelp", "k5"), ("noul_sst", "binary"), ("noul_hate", "binary")):
        if name in R["records"]:
            recs = [r for r in recs_of(R, name) if r["probs"] is not None]
            if recs:
                A["tasks"][name].update(
                    prob_stats([r["y"] for r in recs],
                               [r["probs"] for r in recs]))
    # calibration: fit dev, report test pre/post
    cal = fit_calibration(R)
    A["calibration"]["params"] = cal
    with open(f"evals/final/cal-{args.system}.json", "w") as f:
        json.dump(cal, f, indent=2)
    for name, kk in (("agnews", "k4"), ("sst2", "k2"), ("yelp", "k5")):
        if name in R["records"] and kk in cal:
            recs = [r for r in recs_of(R, name) if r["probs"] is not None]
            P = np.stack([r["probs"] for r in recs])
            Y = np.array([r["y"] for r in recs])
            c = cal[kk]
            Pt = apply_cal(P, {kk: {"temp": c["temp"]}}, kk) \
                if "temp" in c else P
            Pv = apply_cal(P, cal, kk)
            A["tasks"][name]["cal_temp"] = prob_stats(Y, Pt)
            A["tasks"][name]["cal_vec"] = prob_stats(Y, Pv)
    # selective + reliability + auroc on pooled choice tests
    pool = []
    for name in ("agnews", "sst2", "dbpedia"):
        pool += [r for r in recs_of(R, name) if r["probs"] is not None]
    if pool:
        conf = np.array([max(r["probs"]) for r in pool])
        y = np.array([r["y"] for r in pool])
        pred = np.array([int(np.argmax(r["probs"]) == r["y"]) for r in pool])
        A["selective"] = risk_coverage(conf, (np.array(
            [np.argmax(r["probs"]) for r in pool]) == y).astype(int), COVERAGES)
        A["reliability"] = reliability_bins(conf, (np.array(
            [np.argmax(r["probs"]) for r in pool]) == y).astype(int))
        A["auroc"] = {"confidence_vs_correct": auroc_score(
            conf, (np.array([np.argmax(r["probs"]) for r in pool]) == y).astype(int))}
    # order invariance
    recs = recs_of(R, "order")
    if recs:
        by_base = {}
        for r in recs:
            k = len(r["probs"]) if r["probs"] is not None else -1
            by_base.setdefault((str(r.get("base_idx")), k), []).append(r)
        stable, diffs_max, diffs_mean = 0, [], []
        for _, rows in by_base.items():
            winners = [r["pred"] for r in rows]
            stable += len(set(winners)) == 1
            if all(r["probs"] is not None for r in rows):
                P = np.stack([r["probs"] for r in rows])
                # align option identities via stored perm
                perms = [r.get("perm") for r in rows]
                if all(p is not None for p in perms) and len(set(map(len, perms))) == 1:
                    k = len(perms[0])
                    Q = np.zeros((len(rows), k))
                    for i, (p, perm) in enumerate(zip(P, perms)):
                        inv = np.argsort(perm)
                        Q[i] = np.asarray(p)[inv]
                    d = np.abs(Q[:, None, :] - Q[None, :, :]).max(-1)
                    iu = np.triu_indices(len(rows), 1)
                    diffs_max.append(float(d[iu].max()))
                    diffs_mean.append(float(d[iu].mean()))
        A["order"] = {"n_items": len(by_base), "stability": stable / max(1, len(by_base)),
                      "max_prob_diff_mean": float(np.mean(diffs_max)) if diffs_max else None,
                      "mean_prob_diff_mean": float(np.mean(diffs_mean)) if diffs_mean else None}
    # paraphrase agreement (positional pairing: para[i] <-> base[i])
    recs = recs_of(R, "paraphrase")
    if recs:
        n2 = len(recs) // 2
        ag_p, ss_p = recs[:n2], recs[n2:]
        ag_b = recs_of(R, "agnews")[:n2]
        ss_b = recs_of(R, "sst2")[:n2]
        agree, var = [], []
        for a, b in list(zip(ag_p, ag_b)) + list(zip(ss_p, ss_b)):
            agree.append((a["pred"] == b["pred"]) if a["pred"] is not None and b["pred"] is not None else None)
            if a["probs"] is not None and b["probs"] is not None:
                pa, pb = np.asarray(a["probs"]), np.asarray(b["probs"])
                if len(pa) == len(pb):
                    var.append(float(np.abs(pa - pb).max()))
        agree = [x for x in agree if x is not None]
        A["paraphrase"] = {"n_pairs": len(agree),
                           "agreement": float(np.mean(agree)) if agree else None,
                           "mean_maxprob_diff": float(np.mean(var)) if var else None}
    # cardinality table (k4 proxied by agnews; identical 4-way decisions)
    A["cardinality"] = {"2": A["tasks"]["k2"]["accuracy"] if "k2" in A["tasks"] else None,
                        "4": A["tasks"]["agnews"]["accuracy"] if "agnews" in A["tasks"] else None,
                        "8": A["tasks"]["ep_inspace"]["accuracy"] if "ep_inspace" in A["tasks"] else None,
                        "14": A["tasks"]["k14"]["accuracy"] if "k14" in A["tasks"] else None,
                        "32": A["tasks"]["k32"]["accuracy"] if "k32" in A["tasks"] else None,
                        "64": A["tasks"]["k64"]["accuracy"] if "k64" in A["tasks"] else None}
    # latency summary
    lat = R.get("latency_raw_ms", {})
    A["latency"] = {k: {"p50": float(np.percentile(v, 50)),
                        "p95": float(np.percentile(v, 95)),
                        "n": len(v)} for k, v in lat.items()}
    A["vram_peak_gb"] = R.get("vram_peak_gb")
    A["params"] = R.get("params")
    A["gen_tokens_total"] = R.get("gen_tokens_total", 0)
    A["gen_parse_fails"] = R.get("gen_parse_fails", 0)
    with open(f"evals/final/analysis-{args.system}.json", "w") as f:
        json.dump(A, f, indent=2)
    print(f"analysis-{args.system}: tasks={sorted(A['tasks'])} "
          f"vram={A['vram_peak_gb']}GB")


if __name__ == "__main__":
    main()
