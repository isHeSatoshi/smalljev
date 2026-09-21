"""Validate paper-sourced confidence signals on the v4 bundle (forward-only).

Signals: maxprob (baseline), layer-convergence (2510.01237 Eq.2),
OOD-kNN distance to dev readout states (2503.23303 meta-learning idea).
Metrics: AUROC for error detection, selective accuracy/coverage + fitted
precision bands, transfer-12 behavior. No training.

Usage: python evals/confidence_signals.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

from smalljev.model import HFBackend, render_prompt
from smalljev.confidence import (OODBank, coverage_curve, layer_convergence,
                                 threshold_for_precision)

BACKBONE = "openbmb/MiniCPM5-2B-Base"
device = "cuda"
be = HFBackend(BACKBONE, adapter_id="evals/heads/v4-bundle-lora",
               heads_ckpt="evals/heads/v4-bundle.pt")
rng = np.random.RandomState(0)

AG_Q = "What is the topic of this news article?"
AG_L = ["World", "Sports", "Business", "Sci/Tech"]
SS_Q = "What is the sentiment of this review?"
SS_L = ["negative", "positive"]


def collect(data):
    """Per-sample: pred, maxprob, convergence, readout hidden (CPU)."""
    recs, H = [], []
    for t, q, c, y in data:
        suf = be._branch_suffix(q, c)
        bounds, layers = be.packed_hidden_all(t, [suf])
        a, b = bounds[0]
        hl = [h[b - 1, :].float().cpu() for h in layers]
        h_last = hl[-1]
        with torch.no_grad():
            p = be.heads.choice.probs(h_last.to(device).unsqueeze(0), len(c)
                                      )[0].float().cpu().numpy()
        pred = int(p.argmax())
        recs.append({"y": y, "pred": pred, "conf": float(p.max()),
                     "probs": p.astype(float),
                     "conv": layer_convergence([x.unsqueeze(0) for x in hl])})
        H.append(h_last.numpy())
    return recs, np.stack(H)


ag_t = load_dataset("ag_news", split="test[:300]")
ss_v = load_dataset("glue", "sst2", split="validation")
ag = [(r["text"][:800], AG_Q, AG_L, int(r["label"])) for r in ag_t]
ss = [(r["sentence"], SS_Q, SS_L, int(r["label"])) for r in ss_v][:300]

# fit bank on AGNews dev readout states (first 100 of a held slice)
dev_sub = ag[:100]
recs_dev_ag, H_dev = collect(dev_sub)
bank = OODBank().fit(H_dev)

for name, data in (("agnews", ag[100:]), ("sst2", ss)):
    recs, H = collect(data)
    y = np.array([r["y"] for r in recs])
    correct = np.array([r["pred"] == r["y"] for r in recs])
    conf = np.array([r["conf"] for r in recs])
    conv = np.array([r["conv"] for r in recs])
    ood = bank.score(H)
    print(f"== {name} acc={correct.mean():.3f} n={len(y)} ==")
    for feat_name, f, higher_means in (("maxprob", conf, True),
                                       ("convergence", conv, True),
                                       ("ood_dist", -ood, True)):
        try:
            auc = roc_auc_score(correct, f)
        except ValueError:
            auc = float("nan")
        print(f"  error-detection AUROC({feat_name})={auc:.3f}", flush=True)
    cov, acc = coverage_curve(conf, correct)
    print(f"  selective acc@cov: " +
          " ".join(f"{c:.1f}:{a:.2f}" for c, a in zip(cov[::3], acc[::3])), flush=True)

# bands fit on sst2-dev slice, reported on rest + transfer
from smalljev.calibration import apply_temperature, fit_temperature
from evals.train_heads import TRANSFER
recs_tr, _ = collect(TRANSFER)
recs_ss, _ = collect(ss[:100])
# fit temperature on SST-2 dev TRUE labels (log-probs are valid scaler input;
# pooling across tasks is impossible: 4-way vs 2-way shapes differ).
P_sst = np.stack([r["probs"] for r in recs_ss])
y_sst = np.array([r["y"] for r in recs_ss])
T = fit_temperature(np.log(np.clip(P_sst, 1e-12, 1.0)), y_sst)


def cal_conf(recs):
    P = np.stack([r["probs"] for r in recs])
    return apply_temperature(np.log(np.clip(P, 1e-12, 1.0)), T).max(1)


c_d = cal_conf(recs_ss)
y_d = np.array([r["pred"] == r["y"] for r in recs_ss])
t_hi = threshold_for_precision(c_d, y_d, 0.95)
t_md = threshold_for_precision(c_d, y_d, 0.80)
print(f"bands on calibrated probs (T={T:.2f}): act>={t_hi:.3f} flag>={t_md:.3f}")
recs_rest, _ = collect(ss[100:300])
for nm, rr in (("sst2-rest", recs_rest), ("transfer-12", recs_tr)):
    c = cal_conf(rr)
    ok = np.array([r["pred"] == r["y"] for r in rr])
    for band, m in (("act", c >= t_hi), ("flag", (c >= t_md) & (c < t_hi)),
                    ("escalate", c < t_md)):
        n = m.sum()
        print(f"  {nm} {band}: n={n} acc={ok[m].mean() if n else float('nan'):.3f}",
              flush=True)
