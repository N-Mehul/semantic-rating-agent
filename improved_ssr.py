"""Enhanced Semantic Similarity Rating (E-SSR) implementation (primary scoring method).

This module provides an additional, scientifically optimized SSR scoring variant
that implements top-k truncation (k=3) and mild temperature sharpening (T=0.75).

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. CANONICAL SSR IS NOT REPLACED. Canonical SSR (canonical_ssr.py) remains the
   original, unconstrained reference baseline (k=3/4/5 paper minimum-subtraction, T=1.0).
2. MATHEMATICAL REUSE: Reuses cosine_similarities(), expected_rating(), RATINGS,
   and result dataclasses directly from canonical_ssr.py without duplicating code.
3. COMPONENT SEPARATION:
   - Component A: Canonical SSR (canonical_ssr.py, unchanged)
   - Component B: E-SSR, Enhanced Semantic Similarity Rating (this module: k=3, T=0.75)
   - Component C: Optional Linear Variance Rescaling (disabled by default; intended
     for held-out human benchmark comparison, NOT production rating generation)
4. CONTINUOUS PROBABILISTIC RATINGS: Produces continuous expected ratings E[R] in [1, 5]
   and full 5-point probability mass functions (PMF). Does NOT generate ratings via LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

import numpy as np

from canonical_ssr import (
    PAPER_EPSILON,
    PROBABILITY_TOLERANCE,
    RATINGS,
    SingleConstructSSRResult,
    SSRResult,
    cosine_similarities,
    expected_rating,
)

# Scientifically validated default parameters from the Pareto optimization audit
DEFAULT_IMPROVED_K: Final[int] = 3
DEFAULT_IMPROVED_TEMPERATURE: Final[float] = 0.75
DEFAULT_EPSILON: Final[float] = PAPER_EPSILON


def improved_similarities_to_pmf(
    similarities: np.ndarray,
    k: int = DEFAULT_IMPROVED_K,
    temperature: float = DEFAULT_IMPROVED_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Convert raw cosine similarities into a 5-point PMF with top-k truncation and temperature.

    Mathematical Formulation:
    1. Minimum subtraction:
       w(r) = gamma(r) - min_j gamma(j) + epsilon * delta_{min, r}
    2. Top-k truncation:
       Retain only the k largest weights, zeroing the remaining (5 - k) noise anchors.
       For k=1, collapses to a hard argmax indicator vector.
    3. Normalization:
       p_raw(r) = w_topk(r) / sum_j w_topk(j)
    4. Temperature scaling (for T != 1.0):
       p(r) = p_raw(r)^(1/T) / sum_j p_raw(j)^(1/T) for p_raw(r) > 0

    Args:
        similarities: Array of exactly 5 cosine similarity values.
        k: Truncation threshold (1 <= k <= 5). Defaults to 3.
        temperature: Softmax temperature parameter (T > 0). Defaults to 0.75.
        epsilon: Small positive constant for tie-breaking. Defaults to 0.0.

    Returns:
        np.ndarray: Valid 5-point probability distribution summing to 1.0 within tolerance.
    """
    scores = np.asarray(similarities, dtype=float)
    if scores.shape != (len(RATINGS),):
        raise ValueError(f"similarities must contain exactly {len(RATINGS)} values, got {scores.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("similarities must contain only finite values")
    if not (1 <= k <= len(RATINGS)):
        raise ValueError(f"k must be an integer between 1 and {len(RATINGS)}, got {k}")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature must be finite and positive, got {temperature}")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"epsilon must be finite and non-negative, got {epsilon}")

    # Case k=1: Hard nearest-anchor argmax
    if k == 1:
        pmf = np.zeros(len(RATINGS), dtype=float)
        pmf[int(np.argmax(scores))] = 1.0
        return pmf

    # Step 1: Minimum subtraction
    min_idx = int(np.argmin(scores))
    weights = scores - scores[min_idx]
    weights[min_idx] += epsilon

    # Step 2: Top-k truncation (zero out the bottom 5-k anchors)
    if k < len(RATINGS):
        top_k_indices = np.argsort(weights)[-k:]
        mask = np.zeros(len(RATINGS), dtype=bool)
        mask[top_k_indices] = True
        weights[~mask] = 0.0

    denom = float(weights.sum())
    if denom <= 0:
        # Fallback to uniform distribution if all weights are zero (identical similarities)
        return np.ones(len(RATINGS), dtype=float) / float(len(RATINGS))

    pmf = weights / denom

    # Step 3: Temperature scaling
    if temperature != 1.0:
        pos = pmf > 0
        log_w = np.full(len(RATINGS), -np.inf, dtype=float)
        log_w[pos] = np.log(pmf[pos]) / temperature
        log_w -= np.max(log_w)
        pmf = np.exp(log_w)
        pmf /= pmf.sum()

    # Numerical verification
    if not np.isclose(pmf.sum(), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0.0):
        pmf /= pmf.sum()

    return pmf


def score_embedding_improved(
    response_embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
    k: int = DEFAULT_IMPROVED_K,
    temperature: float = DEFAULT_IMPROVED_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> SingleConstructSSRResult:
    """Score a response embedding against 5 ordered anchor embeddings using Improved SSR.

    Reuses canonical_ssr.cosine_similarities and canonical_ssr.expected_rating.
    """
    sims = cosine_similarities(response_embedding, anchor_embeddings)
    pmf = improved_similarities_to_pmf(sims, k=k, temperature=temperature, epsilon=epsilon)
    exp_rating = expected_rating(pmf)
    return SingleConstructSSRResult(
        similarities=sims,
        probabilities=pmf,
        expected_rating=exp_rating,
    )


# =============================================================================
# Component C: Optional Linear Variance Rescaler
# =============================================================================
@dataclass(frozen=True)
class LinearVarianceRescaler:
    """Optional post-processing affine rescaler for benchmarking against human survey distributions.

    IMPORTANT SCIENTIFIC NOTICE:
    This transformation is an optional evaluation and reporting tool designed to compensate
    for the natural variance compression of continuous expectation E[R] when benchmarking
    against discrete human survey responses.

    It is DISABLED BY DEFAULT and MUST NOT be applied automatically to production ratings.
    """

    mu_train: float
    sigma_train: float
    mu_synth: float
    sigma_synth: float
    clip_min: float = 1.0
    clip_max: float = 5.0
    enabled: bool = False

    def __post_init__(self):
        if self.sigma_synth <= 0:
            raise ValueError(f"sigma_synth must be strictly positive, got {self.sigma_synth}")
        if self.sigma_train < 0:
            raise ValueError(f"sigma_train must be non-negative, got {self.sigma_train}")
        if self.clip_min >= self.clip_max:
            raise ValueError(f"clip_min ({self.clip_min}) must be strictly less than clip_max ({self.clip_max})")

    @classmethod
    def fit_from_splits(
        cls,
        human_train_ratings: Sequence[float],
        synthetic_baseline_ratings: Sequence[float],
        enabled: bool = False,
    ) -> LinearVarianceRescaler:
        """Fit rescaling parameters strictly from a human training partition and synthetic baseline."""
        h_arr = np.asarray(human_train_ratings, dtype=float)
        s_arr = np.asarray(synthetic_baseline_ratings, dtype=float)

        mu_tr = float(np.mean(h_arr))
        sig_tr = float(np.std(h_arr, ddof=1))
        mu_sy = float(np.mean(s_arr))
        sig_sy = float(np.std(s_arr, ddof=1))

        return cls(
            mu_train=mu_tr,
            sigma_train=sig_tr,
            mu_synth=mu_sy,
            sigma_synth=sig_sy,
            enabled=enabled,
        )

    def rescale(self, rating: float | np.ndarray) -> float | np.ndarray:
        """Apply affine rescaling with boundary clipping if enabled, or return unchanged if disabled.

        Formula:
        y = mu_train + (x - mu_synth) * (sigma_train / sigma_synth)
        y_clipped = clip(y, clip_min, clip_max)
        """
        if not self.enabled:
            return rating

        scale = self.sigma_train / self.sigma_synth
        rescaled = self.mu_train + (np.asarray(rating, dtype=float) - self.mu_synth) * scale
        clipped = np.clip(rescaled, self.clip_min, self.clip_max)

        if isinstance(rating, (int, float)):
            return float(clipped)
        return clipped

