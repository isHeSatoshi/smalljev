"""14-point eval battery, identical for every backbone arm.

Loads evals/arms/{arm}-bundle.pt + {arm}-lora under the joint-text protocol and
reports: in-domain, novel sets, paraphrase, order invariance, k-scaling
(2/4/8/14/32), holdout intents, cross-domain transfer, calibration trio,
confidence AUROC, latency, VRAM, params, forward-pass count, throughput,
world-knowledge probe, joint-vs-independent ablation.

Usage: python evals/backbone_evals.py --arm bidirlm
Out: evals/arms/{arm}-eval.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from smalljev.backbones import BACKBONES, Backbone
from smalljev.heads import HeadsBundle
from evals.train_heads_v2 import AG_L, AG_QS, SS_L, SS_QS, DB_L, DB_QS
from evals.train_heads_v4 import YELP_L, YELP_QS, BQ
from evals.train_heads import TRANSFER

N_HOLDOUT, HOLDOUT_SEED = 12, 11
YELP_V = [1, 2, 3, 4, 5]

WORLD = [  # synthetic diagnostic probe (hand labels): does pretraining knowledge
    # transfer into arbitrary typed decisions? NOT a benchmark.
    ("The platypus lays eggs yet produces milk to feed its young.",
     "Which category best fits this animal?", ["mammal", "reptile", "bird", "fish"], 0),
    ("The animal has feathers, hollow bones, and lays hard-shelled eggs.",
     "Which category best fits this animal?", ["mammal", "reptile", "bird", "fish"], 2),
    ("Marie Curie won Nobel Prizes in both Physics and Chemistry.",
     "What best describes her?", ["physicist and chemist", "biologist", "mathematician", "engineer"], 0),
    ("The treaty was signed in 1783, ending eight years of war.",
     "What kind of event is described?", ["treaty", "election", "discovery", "festival"], 0),
    ("After the substation failed, traffic shifted to the backup region.",
     "What happened first?", ["the substation failed", "traffic shifted", "both at once", "unclear"], 0),
    ("She left at 6pm carrying an umbrella though the sky was perfectly clear.",
     "Most likely reason?", ["expects rain later", "forgot it was in her hand",
                             "bought it as a gift", "uses it as a walking stick"], 0),
    ("Water boils at a lower temperature on top of a mountain than at sea level.",
     "Why?", ["lower air pressure", "colder air", "different water", "stronger gravity"], 0),
    ("The defendant was acquitted after the key witness recanted.",
     "What was the outcome?", ["acquittal", "conviction", "mistrial", "settlement"], 0),
    ("Habeas corpus was suspended during the emergency.",
     "What does this concern?", ["detention law", "tax law", "maritime law", "patent law"], 0),
    ("The triage nurse tagged the patient red and called the trauma team.",
     "What does red mean here?", ["immediate care", "walking wounded",
                                  "deceased", "discharged"], 0),
    ("The startup's burn multiple rose from 1.2 to 3.8 in two quarters.",
     "What does this suggest?", ["worsening efficiency", "improving efficiency",
                                 "stable operations", "a pivot"], 0),
    ("Schrödinger's cat is both alive and dead until observed.",
     "What is this about?", ["quantum superposition", "relativity",
                             "thermodynamics", "evolution"], 0),
]


def load_arm(arm, device):
    from peft import PeftModel
    bb = Backbone(arm, device=device)
    model = PeftModel.from_pretrained(bb.model, f"evals/arms/{arm}-lora").eval()
    bb.model = model
    bundle = HeadsBundle.load(f"evals/arms/{arm}-bundle.pt").to(device).eval()
    return bb, bundle


def probs_choice(bb, bundle, state, question, options):
    H = bb.encode([bb.joint_text(state, question, options)]).float()
    with torch.no_grad():
        return bundle.choice.probs(H.to(bb.device), len(options))[0].float().cpu().numpy()


def probs_noul(bb, bundle, state, statement):
    H = bb.encode([bb.joint_text(state, statement, ["yes", "no"])]).float()
    with torch.no_grad():
        return float(bundle.noul.prob(H.to(bb.device)).float().cpu().numpy().reshape(-1)[0])


def probs_score(bb, bundle, state, question, levels):
    H = bb.encode([bb.joint_text(state, question, levels)]).float()
    with torch.no_grad():
        return bundle.score.probs(H.to(bb.device), len(levels))[0].float().cpu().numpy()


def acc_nll_ece_brier(records):
    """records: list of (true_label, prob_vector); k may vary per item."""
    from sklearn.metrics import accuracy_score
    y = np.array([r[0] for r in records])
    Ps = [np.asarray(r[1], dtype=float) for r in records]
    pred = np.array([p.argmax() for p in Ps])
    correct = (pred == y).astype(float)
    conf = np.array([p.max() for p in Ps])
    nll = float(np.mean([-np.log(max(p[yy], 1e-12)) for p, yy in zip(Ps, y)]))
    brier = float(np.mean([
        float(((p - np.eye(len(p))[yy]) ** 2).mean()) for p, yy in zip(Ps, y)]))
    out = {"accuracy": float(accuracy_score(y, pred)), "brier": brier,
           "nll": nll, "ece": _ece_conf(conf, correct)}
    return out


def _ece_conf(conf, correct, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum():
            e += abs(correct[m].mean() - conf[m].mean()) * m.sum() / len(conf)
    return float(e)


def episodes(bb, bundle, by_te, names, intents, rng, n, k=8):
    correct, total = 0, 0
    for ep in range(n):
        ks = list(rng.choice(intents, k, replace=False))
        q = BQ[ep % len(BQ)]
        labels = [names[k] for k in ks]
        queries = [by_te[k][rng.randint(len(by_te[k]))][:800] for k in ks]
        H = bb.encode([bb.joint_text(queries[i], q, labels) for i in range(k)]).float()
        with torch.no_grad():
            P = bundle.choice.probs(H.to(bb.device), k).float().cpu().numpy()
        for row in range(k):
            total += 1
            correct += int(P[row].argmax()) == row
    return correct / total, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(BACKBONES))
    args = ap.parse_args()
    device = "cuda"
    t_all = time.perf_counter()
    bb, bundle = load_arm(args.arm, device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    from datasets import load_dataset
    rng = np.random.RandomState(HOLDOUT_SEED)
    R = {}

    ag_t = load_dataset("ag_news", split="test[:250]")
    ss_v = load_dataset("glue", "sst2", split="validation")
    db_t = load_dataset("dbpedia_14", split="test[:200]")
    yp_t = load_dataset("yelp_review_full", split="test[:200]")
    bk_te = load_dataset("banking77", split="test")
    names = [n.replace("_", " ") for n in bk_te.features["label"].names]
    by_te = {}
    for t, y in zip(bk_te["text"], bk_te["label"]):
        by_te.setdefault(int(y), []).append(t)
    holdout = sorted(rng.choice(len(names), N_HOLDOUT, replace=False).tolist())
    inspace = rng.choice([i for i in range(len(names)) if i not in holdout],
                         N_HOLDOUT, replace=False).tolist()

    ag = [(r["text"][:800], AG_QS[0], AG_L, int(r["label"])) for r in ag_t]
    ss = [(r["sentence"], SS_QS[0], SS_L, int(r["label"])) for r in ss_v][:250]
    db = [(r["content"][:800], DB_QS[0], DB_L, int(r["label"])) for r in db_t]
    yp = [(r["text"][:800], YELP_QS[0], YELP_L, int(r["label"])) for r in yp_t]
    ss_noul = [(t, "This review is positive.", None, y) for t, _, _, y in ss[:150]]

    def run_choice(data):
        return [(y, probs_choice(bb, bundle, t, q, c)) for t, q, c, y in data]

    # 1. in-domain
    R["indomain_agnews"] = acc_nll_ece_brier(run_choice(ag))
    R["indomain_sst2"] = acc_nll_ece_brier(run_choice(ss))
    R["indomain_dbpedia"] = acc_nll_ece_brier(run_choice(db))
    R["noul_sst2"] = _noul_eval(bb, bundle, ss_noul)
    R["score_yelp"] = _score_eval(bb, bundle, yp)
    # 2/6. novel sets + holdout
    R["episodes_inspace_k8"], _ = episodes(bb, bundle, by_te, names, inspace, rng, 40)
    R["episodes_holdout_k8"], _ = episodes(bb, bundle, by_te, names, holdout, rng, 60)
    # 3. paraphrase
    para = ([(t, "Which section does this article belong in?", AG_L, y) for t, _, _, y in ag[:75]] +
            [(t, "Is this review positive or negative?", SS_L, y) for t, _, _, y in ss[:75]])
    R["paraphrase"] = acc_nll_ece_brier(run_choice(para))
    # 4. order invariance (60 items x 3 shuffles): winner-identity stability
    stable, accs = 0, []
    sub = (ag[:30] + ss[:30])
    for t, q, c, y in sub:
        winners = []
        for s in range(3):
            order = rng.permutation(len(c))
            opts = [c[i] for i in order]
            p = probs_choice(bb, bundle, t, q, opts)
            winners.append(opts[int(p.argmax())])
            accs.append(winners[-1] == c[y])
        stable += len(set(winners)) == 1
    R["order_stability"] = stable / len(sub)
    R["order_acc_mean"] = float(np.mean(accs))
    # 5. variable k
    ag2 = [(t, AG_QS[0], AG_L[:2], y) for t, _, _, y in ag if y < 2][:80]
    R["k2"] = acc_nll_ece_brier(run_choice(ag2))["accuracy"]
    R["k4"] = R["indomain_agnews"]["accuracy"]
    R["k8"] = R["episodes_inspace_k8"]
    R["k14"] = acc_nll_ece_brier(run_choice(db))["accuracy"]
    R["k32"] = _episodes_k(bb, bundle, by_te, names, list(range(len(names))),
                           rng, n_eps=5, k=32)
    # 7. cross-domain transfer
    R["transfer_k3"] = acc_nll_ece_brier(run_choice(TRANSFER))["accuracy"]
    # 9. confidence AUROC (pooled in-domain)
    P_all, Y_all = [], []
    for data in (ag[:100], ss[:100]):
        for t, q, c, y in data:
            p = probs_choice(bb, bundle, t, q, c)
            P_all.append(p.max())
            Y_all.append(int(p.argmax()) == y)
    R["conf_auroc"] = float(roc_auc_score(Y_all, P_all))
    # 8b. calibration, fit PER-k (vector scaling is dim-dependent):
    # temp+vec on AG-dev50 (k=4) and SS-dev50 (k=2), applied to held tests.
    from smalljev.calibration import (apply_temperature, apply_vector_scale,
                                      fit_temperature, fit_vector_scale)
    dev4 = ag[:50]
    P4 = np.stack([probs_choice(bb, bundle, t, q, c) for t, q, c, y in dev4])
    Y4 = np.array([y for _, _, _, y in dev4])
    T4 = fit_temperature(np.log(np.clip(P4, 1e-12, 1.0)), Y4)
    T4v, B4 = fit_vector_scale(np.log(np.clip(P4, 1e-12, 1.0)), Y4)
    dev2 = ss[:50]
    P2 = np.stack([probs_choice(bb, bundle, t, q, c) for t, q, c, y in dev2])
    Y2 = np.array([y for _, _, _, y in dev2])
    T2 = fit_temperature(np.log(np.clip(P2, 1e-12, 1.0)), Y2)
    T2v, B2 = fit_vector_scale(np.log(np.clip(P2, 1e-12, 1.0)), Y2)
    tst4 = ag[50:150]
    Pt4 = np.stack([probs_choice(bb, bundle, t, q, c) for t, q, c, y in tst4])
    Yt4 = [y for _, _, _, y in tst4]
    tst2 = ss[50:150]
    Pt2 = np.stack([probs_choice(bb, bundle, t, q, c) for t, q, c, y in tst2])
    Yt2 = [y for _, _, _, y in tst2]
    R["cal_temp"] = {"k4_T": T4, "k2_T": T2}
    R["calibrated_k4"] = acc_nll_ece_brier(
        [(y, p) for y, p in zip(
            Yt4, apply_temperature(np.log(np.clip(Pt4, 1e-12, 1.0)), T4))])
    R["calibrated_vec_k4"] = acc_nll_ece_brier(
        [(y, p) for y, p in zip(
            Yt4, apply_vector_scale(np.log(np.clip(Pt4, 1e-12, 1.0)), T4v, B4))])
    R["calibrated_k2"] = acc_nll_ece_brier(
        [(y, p) for y, p in zip(
            Yt2, apply_temperature(np.log(np.clip(Pt2, 1e-12, 1.0)), T2))])
    R["calibrated_vec_k2"] = acc_nll_ece_brier(
        [(y, p) for y, p in zip(
            Yt2, apply_vector_scale(np.log(np.clip(Pt2, 1e-12, 1.0)), T2v, B2))])
    # 13. world probe
    R["world_probe"] = acc_nll_ece_brier(run_choice(
        [(t, q, c, y) for t, q, c, y in WORLD]))["accuracy"]
    R["world_detail"] = _world_detail(bb, bundle)
    # 10/11/12/14. latency/VRAM/params/throughput
    for k, nrep in ((1, 5), (8, 3), (32, 2)):
        sp_q = BQ[0]
        labs = [names[i] for i in range(k)]
        qs = ["sample query text for latency probe"] * k
        t0 = time.perf_counter()
        for _ in range(nrep):
            H = bb.encode([bb.joint_text(qq, sp_q, labs) for qq in qs])
        R[f"latency_ms_k{k}"] = (time.perf_counter() - t0) / nrep * 1000
    R["throughput_qps_k8"] = 8 / (R["latency_ms_k8"] / 1000)
    n_params = sum(p.numel() for p in bb.model.parameters()) + \
        sum(p.numel() for p in bundle.parameters())
    R["params_total"] = n_params
    R["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) \
        if torch.cuda.is_available() else 0.0
    R["forward_passes_per_call"] = 1  # joint encode, all arms
    R["elapsed_min"] = round((time.perf_counter() - t_all) / 60, 1)

    with open(f"evals/arms/{args.arm}-eval.json", "w") as f:
        json.dump({"arm": args.arm, "spec": BACKBONES[args.arm], **R}, f, indent=2)
    print(json.dumps({"arm": args.arm, **{k: v for k, v in R.items()
                                          if not isinstance(v, dict)}}, indent=2))


def _noul_eval(bb, bundle, data):
    p = np.array([probs_noul(bb, bundle, t, q) for t, q, _, y in data])
    y = np.array([y for _, _, _, y in data])
    pred = (p >= 0.5).astype(int)
    return {"accuracy": float((pred == y).mean()),
            "nll": float(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean()),
            "brier": float(np.mean((p - y) ** 2))}


def _score_eval(bb, bundle, data):
    P = np.stack([probs_score(bb, bundle, t, q, c) for t, q, c, y in data])
    y = np.array([y for _, _, _, y in data])
    pred = P.argmax(1)
    return {"exact_acc": float((pred == y).mean()),
            "mae": float(np.abs(pred - y).mean()),
            **acc_nll_ece_brier([(yy, pp) for yy, pp in zip(y, P)])}


def _episodes_k(bb, bundle, by_te, names, intents, rng, n_eps, k):
    correct, total = 0, 0
    for ep in range(n_eps):
        ks = list(rng.choice(intents, k, replace=False))
        q = BQ[ep % len(BQ)]
        labels = [names[i] for i in ks]
        queries = [by_te[i][rng.randint(len(by_te[i]))][:800] for i in ks]
        H = bb.encode([bb.joint_text(queries[i], q, labels) for i in range(k)]).float()
        with torch.no_grad():
            P = bundle.choice.probs(H.to(bb.device), k).float().cpu().numpy()
        for row in range(k):
            total += 1
            correct += int(P[row].argmax()) == row
    return correct / total


def _world_detail(bb, bundle):
    rows = []
    for t, q, c, y in WORLD:
        p = probs_choice(bb, bundle, t, q, c)
        rows.append({"q": q[:70], "true": c[y], "pred": c[int(p.argmax())],
                     "maxprob": round(float(p.max()), 3)})
    return rows


if __name__ == "__main__":
    main()
