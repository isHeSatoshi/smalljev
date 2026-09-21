"""Adversarial-grade statistics: Wilson CIs, McNemar paired test, ECE, AUROC,
risk-coverage. Pure numpy (no sklearn) so results never depend on lib versions."""
import numpy as np


def wilson_ci(correct, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((center - margin) / denom), float((center + margin) / denom))


def mcnemar_p(b, c):
    """Two-sided p-value: b = A-right/B-wrong, c = A-wrong/B-right.
    Exact binomial for b+c < 25, chi-square w/ continuity correction otherwise."""
    from math import comb, sqrt, erf
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        k = min(b, c)
        return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))
    chi2 = (abs(b - c) - 1) ** 2 / n
    # chi-square(1) survival via erf: P = 1 - erf(sqrt(chi2/2))... use normal equiv
    z = sqrt(chi2)
    return float(1 - erf(z / sqrt(2)))


def ece_conf(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, total = 0.0, len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        if m.sum():
            ece += abs(correct[m].mean() - conf[m].mean()) * m.sum() / total
    return float(ece)


def auroc_score(scores, labels):
    """Mann-Whitney U / Wilcoxon AUROC, pure numpy. Higher score = positive."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based: P(pos > neg) + 0.5 P(tie)
    together = np.concatenate([pos, neg])
    order = together.argsort(kind="mergesort")
    ranks = np.empty(len(together))
    ranks[order] = np.arange(1, len(together) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(together, return_inverse=True, return_counts=True)
    rank_sum = np.bincount(inv, weights=ranks)
    avg_ranks = rank_sum / counts
    r = avg_ranks[inv]
    rpos = r[:len(pos)]
    u = rpos.sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def risk_coverage(conf, correct, coverages=(0.1, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0)):
    """Accuracy at top-c coverage + threshold achieving it."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    out = []
    for c in coverages:
        t = float(np.quantile(conf, 1.0 - c))
        sel = conf >= t
        out.append({"coverage": c, "threshold": t, "n": int(sel.sum()),
                    "accuracy": float(correct[sel].mean()) if sel.sum() else float("nan")})
    return out


def reliability_bins(conf, correct, n_bins=10):
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i else (conf >= lo) & (conf <= hi)
        rows.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                     "mean_conf": float(conf[m].mean()) if m.sum() else float("nan"),
                     "accuracy": float(correct[m].mean()) if m.sum() else float("nan")})
    return rows
