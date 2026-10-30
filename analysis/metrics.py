"""
analysis/metrics.py
--------------------
Proper scoring rules and calibration metrics for ordinal/categorical
rating distributions over classes {1, 2, 3, 4, 5}.

Public API
----------
nll(probs, y_true)          → float  (negative log-likelihood, nats)
rps(probs, y_true)          → float  (ranked probability score, lower=better)
ece(probs, y_true, n_bins)  → float  (expected calibration error, 15-bin default)
sharpness(probs)            → float  (mean max probability across rows)

All functions accept:
    probs   : np.ndarray of shape (N, 5), column k corresponds to rating k+1
    y_true  : np.ndarray of shape (N,)   with integer ratings 1-5

Conventions
-----------
- NLL and RPS are *mean* values across the N samples.
- ECE uses equal-width bins over the confidence (max-prob) axis.
- Sharpness = mean(max_k p_k) — high sharpness means the model is peaked.
"""

from __future__ import annotations

import numpy as np

RATINGS = np.array([1, 2, 3, 4, 5], dtype=int)   # class labels in column order
N_CLASSES = 5
EPSILON   = 1e-12   # floor to avoid log(0)


def _validate(probs: np.ndarray, y_true: np.ndarray) -> None:
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise ValueError(f"probs must be shape (N, {N_CLASSES}), got {probs.shape}")
    if y_true.ndim != 1 or len(y_true) != probs.shape[0]:
        raise ValueError("y_true must be 1-D with length == probs.shape[0]")
    if not np.all(np.isin(y_true, RATINGS)):
        raise ValueError("y_true must contain only values in {1, 2, 3, 4, 5}")


def nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean negative log-likelihood (nats).

    NLL = -1/N * sum_i log p(y_i | x_i)
    """
    _validate(probs, y_true)
    idx = y_true - 1                          # 0-indexed column
    p   = np.clip(probs[np.arange(len(y_true)), idx], EPSILON, 1.0)
    return float(-np.mean(np.log(p)))


def rps(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Ranked Probability Score (Epstein 1969).

    RPS = 1/N * sum_i  1/(K-1) * sum_{k=1}^{K-1} (CDF_i(k) - H(k-y_i))^2

    where CDF_i(k) = sum_{j<=k} p_ij and H(.) is the Heaviside step.
    Lower is better; RPS=0 for perfect deterministic predictions.
    """
    _validate(probs, y_true)
    N = len(y_true)
    K = N_CLASSES
    cum_pred = np.cumsum(probs, axis=1)          # (N, 5)
    # Build step function H(k - y_i) for k in {1,...,K-1}: shape (N, K-1)
    # entry (i, k) = 1 if y_true[i] <= k+1, else 0  (k is 0-indexed step)
    step = (y_true[:, None] <= np.arange(1, K)[None, :]).astype(float)  # (N, K-1)
    diff = (cum_pred[:, :K-1] - step) ** 2        # (N, K-1)
    return float(np.mean(diff.sum(axis=1) / (K - 1)))


def ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error over n_bins equal-width confidence bins.

    Confidence = max_k p_k for each sample.
    For each bin b:
        acc_b  = fraction of samples in bin b where argmax = y_true
        conf_b = mean confidence in bin b
    ECE = sum_b |bin_b| / N * |acc_b - conf_b|
    """
    _validate(probs, y_true)
    N = len(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1) + 1        # back to 1-indexed
    correct = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc_b  = correct[mask].mean()
        conf_b = confidences[mask].mean()
        ece_val += mask.sum() / N * abs(acc_b - conf_b)
    return float(ece_val)


def sharpness(probs: np.ndarray) -> float:
    """Mean maximum predicted probability across all samples.

    A perfectly sharp model always assigns probability 1 to one class → sharpness=1.
    A uniform model has sharpness = 1/K.
    """
    if probs.ndim != 2 or probs.shape[1] != N_CLASSES:
        raise ValueError(f"probs must be shape (N, {N_CLASSES}), got {probs.shape}")
    return float(probs.max(axis=1).mean())
