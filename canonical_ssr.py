"""Paper-method SSR scoring, isolated from the frozen review-rating baseline.

This module implements the paper's cosine/minimum-subtraction PMF and six-set
averaging. Supply the original five-anchor text sets and an encoder configured
for OpenAI ``text-embedding-3-small`` to reproduce its reported setup.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

RATINGS = (1, 2, 3, 4, 5)
ANCHOR_SET_COUNT = 6
EMBEDDING_MODEL = "text-embedding-3-small"
PAPER_EPSILON = 0.0
PAPER_TEMPERATURE = 1.0
PROBABILITY_TOLERANCE = 1e-12
EmbeddingFunction = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class SSRResult:
    """Similarity and probability outputs for a single free-text response."""

    similarities: np.ndarray
    per_set_probabilities: np.ndarray
    probabilities: np.ndarray
    expected_rating: float


@dataclass(frozen=True)
class SingleConstructSSRResult:
    """Similarity and PMF probability outputs for a single construct (5 anchors)."""

    similarities: np.ndarray
    probabilities: np.ndarray
    expected_rating: float


def score_embedding_single_construct(
    response_embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
    epsilon: float = PAPER_EPSILON,
    temperature: float = PAPER_TEMPERATURE,
) -> SingleConstructSSRResult:
    """Score one response embedding against one construct's five ordered anchor embeddings.

    Implements Maier et al. (arXiv:2510.08338v3) Equations 7 & 8:
    1. Raw cosine similarity: gamma(r) = (v_r . v_t) / (||v_r|| * ||v_t||)
    2. Minimum subtraction: w(r) = gamma(r) - min_j gamma(j) + epsilon * delta_{l,r}
    3. 5-point PMF: p(r) = w(r) / sum_j w(j)
    4. Expected Likert rating: E[r] = sum_{r=1}^5 r * p(r)

    Methodological Note on k=3:
    The original Maier et al. paper does NOT use top-k nearest neighbors or k=3.
    It calculates cosine similarities across all five anchor statements and normalizes.
    No Softmax, top-k selection, or cross-construct averaging is used.
    """
    sims = cosine_similarities(response_embedding, anchor_embeddings)
    pmf = similarities_to_pmf(sims, epsilon=epsilon, temperature=temperature)
    exp = expected_rating(pmf)
    return SingleConstructSSRResult(
        similarities=sims,
        probabilities=pmf,
        expected_rating=exp,
    )


def cosine_similarities(
    response_embedding: np.ndarray,
    anchor_embeddings: np.ndarray,
) -> np.ndarray:
    """Return raw cosine similarities to five ordered anchor embeddings."""
    response = np.asarray(response_embedding, dtype=float)
    anchors = np.asarray(anchor_embeddings, dtype=float)
    if response.ndim != 1:
        raise ValueError("response_embedding must be a one-dimensional vector")
    if anchors.ndim != 2 or anchors.shape[0] != len(RATINGS):
        raise ValueError("anchor_embeddings must have shape (5, embedding_dim)")
    if anchors.shape[1] != response.shape[0]:
        raise ValueError("response and anchor embedding dimensions must match")
    if not np.all(np.isfinite(response)) or not np.all(np.isfinite(anchors)):
        raise ValueError("embeddings must contain only finite values")

    response_norm = np.linalg.norm(response)
    anchor_norms = np.linalg.norm(anchors, axis=1)
    if response_norm == 0 or np.any(anchor_norms == 0):
        raise ValueError("cosine similarity is undefined for zero-norm embeddings")

    similarities = anchors @ response / (anchor_norms * response_norm)
    return np.clip(similarities, -1.0, 1.0)


def similarities_to_pmf(
    similarities: np.ndarray,
    epsilon: float = PAPER_EPSILON,
    temperature: float = PAPER_TEMPERATURE,
) -> np.ndarray:
    """Apply the paper's minimum-subtracted PMF and optional T scaling.

    At epsilon=0, equal similarities make the paper's denominator zero; this
    function raises rather than inventing a fallback distribution.
    """
    scores = np.asarray(similarities, dtype=float)
    if scores.shape != (len(RATINGS),):
        raise ValueError("similarities must contain exactly five rating scores")
    if not np.all(np.isfinite(scores)):
        raise ValueError("similarities must contain only finite values")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and greater than zero")

    minimum_index = int(np.argmin(scores))
    weights = scores - scores[minimum_index]
    weights[minimum_index] += epsilon
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError(
            "SSR PMF is undefined for equal similarities when epsilon is zero"
        )

    probabilities = weights / denominator
    if temperature != 1.0:
        positive = probabilities > 0
        log_weights = np.full(probabilities.shape, -np.inf, dtype=float)
        log_weights[positive] = np.log(probabilities[positive]) / temperature
        log_weights -= np.max(log_weights)
        probabilities = np.exp(log_weights)
        probabilities /= probabilities.sum()

    return probabilities


def expected_rating(probabilities: np.ndarray) -> float:
    """Calculate E[r] for a five-rating PMF."""
    pmf = np.asarray(probabilities, dtype=float)
    if pmf.shape != (len(RATINGS),):
        raise ValueError("probabilities must contain exactly five values")
    if not np.all(np.isfinite(pmf)) or np.any(pmf < 0):
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isclose(pmf.sum(), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0.0):
        raise ValueError("probabilities must sum to one")
    return float(np.dot(np.asarray(RATINGS, dtype=float), pmf))


def score_embedding_over_six_sets(
    response_embedding: np.ndarray,
    anchor_embedding_sets: Sequence[np.ndarray],
    epsilon: float = PAPER_EPSILON,
    temperature: float = PAPER_TEMPERATURE,
) -> SSRResult:
    """Score one embedding using six ordered groups of five anchors each.

    The six per-set PMFs are averaged rating-wise. The paper's reported
    temperature is 1.0, for which per-set temperature scaling is the identity.
    """
    if len(anchor_embedding_sets) != ANCHOR_SET_COUNT:
        raise ValueError("the paper method requires exactly six anchor sets")

    similarity_rows = []
    probability_rows = []
    for anchor_embeddings in anchor_embedding_sets:
        similarities = cosine_similarities(response_embedding, anchor_embeddings)
        similarity_rows.append(similarities)
        probability_rows.append(
            similarities_to_pmf(similarities, epsilon=epsilon, temperature=temperature)
        )

    all_similarities = np.stack(similarity_rows)
    per_set_probabilities = np.stack(probability_rows)
    probabilities = per_set_probabilities.mean(axis=0)
    if (
        not np.all(np.isfinite(per_set_probabilities))
        or np.any(per_set_probabilities < 0.0)
        or np.any(per_set_probabilities > 1.0)
        or not np.allclose(
            per_set_probabilities.sum(axis=1),
            1.0,
            atol=PROBABILITY_TOLERANCE,
            rtol=0.0,
        )
    ):
        raise ArithmeticError("per-set SSR probabilities are not valid PMFs")
    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.isclose(
            probabilities.sum(), 1.0, atol=PROBABILITY_TOLERANCE, rtol=0.0
        )
    ):
        raise ArithmeticError("aggregated SSR probabilities are not a valid PMF")

    return SSRResult(
        similarities=all_similarities,
        per_set_probabilities=per_set_probabilities,
        probabilities=probabilities,
        expected_rating=expected_rating(probabilities),
    )


def score_text_over_six_sets(
    response_text: str,
    anchor_text_sets: Sequence[Sequence[str]],
    embed_texts: EmbeddingFunction,
    epsilon: float = PAPER_EPSILON,
    temperature: float = PAPER_TEMPERATURE,
) -> SSRResult:
    """Embed a response and six ordered five-text anchor sets in one encoder call.

    ``embed_texts`` must use the paper's ``text-embedding-3-small`` model for
    paper-compatible use. The callable is injected so tests can use fixed,
    deterministic vectors without API calls.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        raise ValueError("response_text must be a non-empty string")
    if isinstance(anchor_text_sets, (str, bytes)) or not isinstance(
        anchor_text_sets, Sequence
    ):
        raise ValueError("anchor_text_sets must be a sequence of six anchor sets")
    if len(anchor_text_sets) != ANCHOR_SET_COUNT:
        raise ValueError("the paper method requires exactly six anchor text sets")
    for anchor_set in anchor_text_sets:
        if isinstance(anchor_set, (str, bytes)) or not isinstance(
            anchor_set, Sequence
        ):
            raise ValueError("each anchor set must be a sequence of five statements")
        if len(anchor_set) != len(RATINGS):
            raise ValueError("each anchor text set must contain five ordered statements")
        if any(not isinstance(text, str) or not text.strip() for text in anchor_set):
            raise ValueError("anchor statements must be non-empty strings")

    texts = [response_text]
    for anchor_set in anchor_text_sets:
        texts.extend(anchor_set)
    embeddings = np.asarray(embed_texts(texts), dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError("embed_texts must return one embedding row per input text")

    return score_embedding_over_six_sets(
        embeddings[0],
        [embeddings[1 + set_index * 5: 1 + (set_index + 1) * 5]
         for set_index in range(ANCHOR_SET_COUNT)],
        epsilon=epsilon,
        temperature=temperature,
    )


def openai_text_embedding_3_small(texts: Sequence[str], client: Any) -> np.ndarray:
    """Encode text using an injected OpenAI client and the paper's model name.

    No client or credentials are created implicitly. The caller controls API
    access and can pass this function as ``embed_texts`` via a small closure.
    """
    if not texts:
        return np.empty((0, 0), dtype=float)
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=list(texts))
    try:
        indexed_data = list(response.data)
    except (AttributeError, TypeError) as exc:
        raise ValueError("embedding API response must contain a data sequence") from exc
    if len(indexed_data) != len(texts):
        raise ValueError("embedding API returned a different number of vectors")

    indices = []
    for item in indexed_data:
        try:
            index = item.index
        except AttributeError as exc:
            raise ValueError("each embedding API item must contain an index") from exc
        if isinstance(index, (bool, np.bool_)) or not isinstance(
            index, (int, np.integer)
        ):
            raise ValueError("embedding API indices must be integers")
        indices.append(int(index))

    expected_indices = list(range(len(texts)))
    if sorted(indices) != expected_indices:
        raise ValueError(
            "embedding API indices must be unique and cover the requested input range"
        )

    ordered_data = sorted(indexed_data, key=lambda item: item.index)
    vectors = []
    dimension = None
    for item in ordered_data:
        try:
            vector = np.asarray(item.embedding, dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("each embedding API item must contain a numeric vector") from exc
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("embedding API vectors must be non-empty one-dimensional arrays")
        if not np.all(np.isfinite(vector)):
            raise ValueError("embedding API vectors must contain only finite values")
        if not np.any(vector):
            raise ValueError("embedding API vectors must have non-zero norm")
        if dimension is None:
            dimension = vector.size
        elif vector.size != dimension:
            raise ValueError("embedding API vectors must have consistent dimensions")
        vectors.append(vector)

    return np.vstack(vectors)
