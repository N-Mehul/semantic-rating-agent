"""Experimental SSR decision rules and calibrators (Phase 7). canonical_ssr.py and improved_ssr.py are NOT modified;
their PMFs are the input here. Every function is deterministic. Fitted calibrators must be fit on a calibration set
that is disjoint from the evaluation set; nothing here sees evaluation labels.

Candidates
  round_expected     r = clip(round(E[r]))                                   (current rule)
  pmf_argmax         r = argmax_r p(r)                                       (mode)
  pmf_median         r = min{r : sum_{j<=r} p(j) >= 0.5}                     (ordinal median; minimises expected |error|)
  nearest_anchor     r = argmax_r cos(response, anchor_r)                    (k = 1 limit of the PMF family)
  BiasShift          E' = E - mean_cal(E - y)                                (global additive calibration)
  ConstructShift     E' = E - mean_cal,c(E - y), shrunk toward global with weight n_c/(n_c + m)
  OrdinalCutpoints   r = 1 + #{t_j <= E}: monotone thresholds fit on calibration data to maximise QWK
  VarianceMatch      E' = mu_y + (E - mu_E) * sd_y / sd_E                    (affine moment matching, fit on calibration)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

RATINGS = np.arange(1, 6)


def round_expected(E) -> np.ndarray:
    return np.clip(np.floor(np.asarray(E, float) + 0.5), 1, 5).astype(int)   # half-up, not banker's rounding


def pmf_argmax(P) -> np.ndarray:
    return np.asarray(P, float).argmax(axis=1) + 1          # ties -> lower rating (np.argmax convention)


def pmf_median(P) -> np.ndarray:
    cdf = np.cumsum(np.asarray(P, float), axis=1)
    return (cdf >= 0.5 - 1e-12).argmax(axis=1) + 1


def nearest_anchor(cos) -> np.ndarray:
    return np.asarray(cos, float).argmax(axis=1) + 1


def expected(P) -> np.ndarray:
    return np.asarray(P, float) @ RATINGS


@dataclass(frozen=True)
class BiasShift:
    shift: float

    @classmethod
    def fit(cls, E_cal, y_cal) -> "BiasShift":
        return cls(float(np.mean(np.asarray(E_cal, float) - np.asarray(y_cal, float))))

    def transform(self, E) -> np.ndarray:
        return np.clip(np.asarray(E, float) - self.shift, 1, 5)


@dataclass(frozen=True)
class ConstructShift:
    """Per-construct additive shift, shrunk toward the global shift (empirical-Bayes style, prior strength m)."""
    global_shift: float
    shifts: Mapping[str, float] = field(default_factory=dict)
    m: float = 8.0

    @classmethod
    def fit(cls, E_cal, y_cal, constructs_cal: Sequence[str], m: float = 8.0) -> "ConstructShift":
        E, y, c = np.asarray(E_cal, float), np.asarray(y_cal, float), np.asarray(constructs_cal)
        g = float(np.mean(E - y)); out = {}
        for k in np.unique(c):
            r = E[c == k] - y[c == k]; n = len(r)
            out[str(k)] = float((n * r.mean() + m * g) / (n + m))
        return cls(g, out, m)

    def transform(self, E, constructs: Sequence[str]) -> np.ndarray:
        return np.clip(np.asarray(E, float) - np.array([self.shifts.get(str(c), self.global_shift) for c in constructs]), 1, 5)


@dataclass(frozen=True)
class VarianceMatch:
    mu_E: float
    sd_E: float
    mu_y: float
    sd_y: float

    @classmethod
    def fit(cls, E_cal, y_cal) -> "VarianceMatch":
        E, y = np.asarray(E_cal, float), np.asarray(y_cal, float)
        return cls(float(E.mean()), float(E.std(ddof=1)), float(y.mean()), float(y.std(ddof=1)))

    def transform(self, E) -> np.ndarray:
        return np.clip(self.mu_y + (np.asarray(E, float) - self.mu_E) * self.sd_y / self.sd_E, 1, 5)


@dataclass(frozen=True)
class OrdinalCutpoints:
    """Thresholds t_1 <= ... <= t_4 on E: rating = 1 + #{j : E >= t_j}. Default thresholds are 1.5/2.5/3.5/4.5 (= rounding).
    fit() searches only the two thresholds the calibration labels can identify (3|4 and 4|5 when labels are 3-5),
    over midpoints of sorted calibration E values, maximising quadratic-weighted kappa (ties: lower MAE, then closest
    to the rounding thresholds). Unidentified thresholds keep their rounding defaults."""
    thresholds: tuple[float, float, float, float] = (1.5, 2.5, 3.5, 4.5)

    def predict(self, E) -> np.ndarray:
        E = np.asarray(E, float)
        return 1 + (E[:, None] >= np.asarray(self.thresholds)[None, :]).sum(axis=1)

    @classmethod
    def fit(cls, E_cal, y_cal) -> "OrdinalCutpoints":
        E, y = np.asarray(E_cal, float), np.asarray(y_cal, int)
        labels = sorted(set(y.tolist()))
        free = [j for j in range(4) if (j + 1) in labels and (j + 2) in labels]   # threshold between r=j+1 and j+2
        s = np.sort(np.unique(E)); cand = np.concatenate([[s[0] - 1e-6], (s[:-1] + s[1:]) / 2, [s[-1] + 1e-6]])
        default = np.array([1.5, 2.5, 3.5, 4.5]); best, best_key = tuple(default), None
        import itertools
        for combo in itertools.product(cand, repeat=len(free)):
            t = default.copy()
            for j, v in zip(free, combo):
                t[j] = v
            if np.any(np.diff(t) < 0):
                continue
            pred = 1 + (E[:, None] >= t[None, :]).sum(axis=1)
            q = cohen_kappa_score(y, pred, weights="quadratic", labels=list(range(1, 6))) if len(set(pred) | set(y)) > 1 else -1.0
            key = (-q, float(np.abs(pred - y).mean()), float(np.abs(t - default).sum()))
            if best_key is None or key < best_key:
                best, best_key = tuple(float(x) for x in t), key
        return cls(best)
