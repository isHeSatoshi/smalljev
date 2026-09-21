"""Calibration math: softmax, temperature scaling, NLL, Brier, ECE. Numpy only."""
import numpy as np


def softmax(logits, axis=-1):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def apply_temperature(logits, temperature):
    return softmax(np.asarray(logits, dtype=np.float64) / float(temperature))


def negative_log_likelihood(probs, labels):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    eps = 1e-12
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)).mean())


def brier_score(prob_positive, labels):
    p = np.asarray(prob_positive, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(prob_positive, labels, n_bins=10):
    p = np.asarray(prob_positive, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p > lo) & (p <= hi) if i else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += abs(acc - conf) * mask.sum() / len(p)
    return float(ece)


def apply_vector_scale(logits, temperature, bias):
    """Softmax over bias-corrected logits: p = softmax((z - b) / T).

    The bias vector absorbs systematic verbalizer preference (e.g. always
    favoring option A); temperature then handles over/under-confidence.
    Standard post-hoc family (cf. contextual calibration, vector scaling).
    """
    z = ((np.asarray(logits, dtype=np.float64)
          - np.asarray(bias, dtype=np.float64)) / float(temperature))
    return softmax(z)


def fit_vector_scale(logits, labels, l2=1e-3, steps=100):
    """Fit (T, b) minimizing NLL + light L2 on b. Returns (temperature, bias)."""
    import torch
    L = torch.tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    y = torch.tensor(np.asarray(labels, dtype=np.int64))
    logT = torch.zeros((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((L.shape[1],), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy((L - b) / torch.exp(logT), y)
        loss = loss + l2 * ((b ** 2).sum() + logT ** 2) / 2.0
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(logT).detach()), b.detach().numpy()


def fit_temperature(logits, labels):
    """Grid-search the temperature minimizing held-out NLL (softens overconfidence)."""
    logits = np.asarray(logits, dtype=np.float64)
    best_t, best_nll = 1.0, float("inf")
    for temp in np.linspace(0.05, 10.0, 400):
        nll = negative_log_likelihood(apply_temperature(logits, temp), labels)
        if nll < best_nll:
            best_nll, best_t = nll, float(temp)
    return best_t
