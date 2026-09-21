"""Confidence signals beyond softmax probabilities.

Grounded in two external papers (see docs/ARCHITECTURE.md survey):
- layer_convergence: variance-reduction across layers (arXiv 2510.01237, Eq.2).
  Convergent processing (late layers settle) scores high; no training needed.
- OODBank: kNN distance to training readout states (cf. 2503.23303 meta-learning
  "know when it doesn't know" via training similarity). Abstention feature.
- threshold_for_precision / coverage_curve: precision bands for act/flag/escalate
  routing (cf. 2510.01237 deterministic routing).
- order_easy_first: curriculum ordering (easy = correct with margin) for training.
"""
import numpy as np


def layer_convergence(layer_hiddens, eps=1e-6):
    """Variance(early-half layers) / Variance(late-half layers), mean over batch.

    Args: list of tensors shaped (..., H) — same position across layers.
    Higher = representations settle in late layers (convergent processing).
    """
    import torch
    flat = [h.detach().float().reshape(-1) for h in layer_hiddens]
    L = len(flat)
    early = torch.cat(flat[:L // 2]).var(unbiased=False)
    late = torch.cat(flat[L // 2:]).var(unbiased=False)
    return float((early / (late + eps)).item())


class OODBank:
    """kNN distance to reference (training/dev) readout states. Larger = stranger."""

    def __init__(self, k=5):
        self.k = k
        self.mean_ = None
        self.std_ = None
        self.refs_ = None

    def fit(self, refs):
        refs = np.asarray(refs, dtype=np.float64)
        self.mean_ = refs.mean(0)
        self.std_ = refs.std(0) + 1e-6
        self.refs_ = (refs - self.mean_) / self.std_
        return self

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        Z = (X - self.mean_) / self.std_
        d2 = ((Z[:, None, :] - self.refs_[None, :, :]) ** 2).sum(-1)
        k = min(self.k, len(self.refs_))
        return np.sort(d2, axis=1)[:, :k].mean(1) ** 0.5


def threshold_for_precision(conf, correct, target=0.9):
    """Smallest threshold with precision(conf >= t) >= target on dev. 1.0 if none."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    # smallest t achieving target: iterate ascending, first hit
    for t in sorted(set(conf.tolist())):
        sel = conf >= t
        if sel.sum() > 0 and correct[sel].mean() >= target - 1e-12:
            return float(t)
    return 1.0


def coverage_curve(conf, correct, n=10):
    """Accuracy at top-c coverage for c in linspace(0.1, 1.0)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    covs = np.linspace(0.1, 1.0, n)
    accs = []
    for c in covs:
        t = np.quantile(conf, 1.0 - c)
        sel = conf >= t
        accs.append(float(correct[sel].mean()) if sel.sum() else float("nan"))
    return covs, np.array(accs)


def order_easy_first(p_true, correct):
    """Indices, confident-correct first, wrong-and-unsure last (curriculum)."""
    p_true = np.asarray(p_true, dtype=float)
    correct = np.asarray(correct, dtype=float)
    keys = list(zip(-correct, -p_true, range(len(p_true))))
    keys.sort()
    return np.array([i for _, _, i in keys])
