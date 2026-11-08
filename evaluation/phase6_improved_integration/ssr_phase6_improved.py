"""SSR_Phase6_Improved: Standalone Module for Phase 6 Improved SSR Integration.

Integrates the validated Phase 6 Improved Semantic Similarity Rating (SSR) method
into the Digital Persona project architecture as a selectable, modular component.

Phase 6 Parameters:
- Top-k Truncation: k = 3 (zeros out the 5 - 3 = 2 noise anchors after min-subtraction)
- Temperature Scaling: T = 0.75 (mild sharpening of probabilities)
- Epsilon: 0.0 (tie-breaking default)

Architectural Guarantees:
- Canonical SSR (canonical_ssr.py) remains completely untouched and functional.
- Zero modifications to anchors, codebooks, or original embedding configurations.
- Reuses cosine_similarities(), expected_rating(), and data structures from canonical_ssr.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence, Literal
import numpy as np

from canonical_ssr import (
    PAPER_EPSILON,
    PAPER_TEMPERATURE,
    PROBABILITY_TOLERANCE,
    RATINGS,
    SingleConstructSSRResult,
    cosine_similarities,
    expected_rating,
    similarities_to_pmf as canonical_similarities_to_pmf,
)

# Phase 6 exact parameters
PHASE6_K: Final[int] = 3
PHASE6_TEMPERATURE: Final[float] = 0.75
PHASE6_EPSILON: Final[float] = PAPER_EPSILON

SSRMethodName = Literal["canonical", "SSR_Phase6_Improved"]


def phase6_similarities_to_pmf(
    similarities: np.ndarray | Sequence[float],
    k: int = PHASE6_K,
    temperature: float = PHASE6_TEMPERATURE,
    epsilon: float = PHASE6_EPSILON,
) -> np.ndarray:
    """Convert raw 5-anchor cosine similarities into a 5-point PMF using Phase 6 methodology.

    Steps:
    1. Minimum subtraction: w(r) = gamma(r) - min_j gamma(j) + epsilon * delta_{min, r}
    2. Top-k truncation: retain top-k weights; set remaining (5-k) anchors to 0.0.
    3. Normalization: p_raw(r) = w_topk(r) / sum_j w_topk(j)
    4. Temperature scaling (T = 0.75):
       p(r) = p_raw(r)^(1/T) / sum_j p_raw(j)^(1/T) for p_raw(r) > 0.
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

    # Numerical stabilization
    if not np.isclose(pmf.sum(), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0.0):
        pmf /= pmf.sum()

    return pmf


def score_embedding_phase6(
    response_embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
    k: int = PHASE6_K,
    temperature: float = PHASE6_TEMPERATURE,
    epsilon: float = PHASE6_EPSILON,
) -> SingleConstructSSRResult:
    """Score a response embedding against 5 ordered anchor embeddings using Phase 6 Improved SSR."""
    sims = cosine_similarities(response_embedding, anchor_embeddings)
    pmf = phase6_similarities_to_pmf(sims, k=k, temperature=temperature, epsilon=epsilon)
    exp_rating = expected_rating(pmf)
    return SingleConstructSSRResult(
        similarities=sims,
        probabilities=pmf,
        expected_rating=exp_rating,
    )


class SSR_Phase6_Improved:
    """Class wrapper providing a unified interface and selector for Phase 6 Improved SSR."""

    name: Final[str] = "SSR_Phase6_Improved"
    k: Final[int] = PHASE6_K
    temperature: Final[float] = PHASE6_TEMPERATURE
    epsilon: Final[float] = PHASE6_EPSILON

    @staticmethod
    def similarities_to_pmf(similarities: np.ndarray | Sequence[float]) -> np.ndarray:
        return phase6_similarities_to_pmf(similarities, k=PHASE6_K, temperature=PHASE6_TEMPERATURE, epsilon=PHASE6_EPSILON)

    @staticmethod
    def score_embedding(response_embedding: np.ndarray, anchor_embeddings: np.ndarray) -> SingleConstructSSRResult:
        return score_embedding_phase6(response_embedding, anchor_embeddings, k=PHASE6_K, temperature=PHASE6_TEMPERATURE, epsilon=PHASE6_EPSILON)

    @staticmethod
    def expected_rating(probabilities: np.ndarray) -> float:
        return expected_rating(probabilities)

    @staticmethod
    def discrete_rating(expected_val: float) -> int:
        return int(np.clip(np.round(expected_val), 1, 5))


def get_ssr_scorer(method: str = "canonical"):
    """Unified selector function to choose between canonical SSR and Phase 6 Improved SSR.

    Supported methods:
      - 'canonical'
      - 'SSR_Phase6_Improved' (or 'improved_ssr', 'phase6')
    """
    m = method.strip().lower()
    if m in ("ssr_phase6_improved", "improved_ssr", "phase6", "improved"):
        return {
            "name": "SSR_Phase6_Improved",
            "similarities_to_pmf": SSR_Phase6_Improved.similarities_to_pmf,
            "score_embedding": SSR_Phase6_Improved.score_embedding,
            "expected_rating": SSR_Phase6_Improved.expected_rating,
            "discrete_rating": SSR_Phase6_Improved.discrete_rating,
            "config": {"k": PHASE6_K, "temperature": PHASE6_TEMPERATURE, "epsilon": PHASE6_EPSILON},
        }
    elif m == "canonical":
        return {
            "name": "canonical",
            "similarities_to_pmf": lambda s: canonical_similarities_to_pmf(s, epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE),
            "score_embedding": lambda r, a: SingleConstructSSRResult(
                similarities=cosine_similarities(r, a),
                probabilities=canonical_similarities_to_pmf(cosine_similarities(r, a), epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE),
                expected_rating=expected_rating(canonical_similarities_to_pmf(cosine_similarities(r, a), epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE)),
            ),
            "expected_rating": expected_rating,
            "discrete_rating": lambda e: int(np.clip(np.round(e), 1, 5)),
            "config": {"k": 5, "temperature": PAPER_TEMPERATURE, "epsilon": PAPER_EPSILON},
        }
    else:
        raise ValueError(f"Unknown SSR method: '{method}'. Choose 'canonical' or 'SSR_Phase6_Improved'.")

