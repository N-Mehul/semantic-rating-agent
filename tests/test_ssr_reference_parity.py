"""Deterministic parity tests comparing canonical_ssr.py with pymc_compute.py.

This test suite verifies:
1. Cosine similarity calculations: raw cosine in canonical_ssr vs (1+cos)/2 in pymc_compute.
2. Probability Mass Function (PMF) equivalence: when epsilon=0 (the paper's baseline),
   both implementations produce identical distributions (within 1e-12).
3. Temperature scaling parity at T=1.0 and across various T values.
4. Expected rating calculation parity.
5. Mathematical differences when epsilon > 0 are documented and verified.
"""

from __future__ import annotations

import numpy as np
import pytest

from canonical_ssr import (
    cosine_similarities as canonical_cosine_similarities,
    expected_rating as canonical_expected_rating,
    similarities_to_pmf as canonical_similarities_to_pmf,
)
import pymc_compute


TOLERANCE = 1e-12


def _make_synthetic_fixtures() -> tuple[np.ndarray, np.ndarray]:
    """Create normalized response vector and five distinct anchor vectors."""
    rng = np.random.default_rng(42)
    response_vec = rng.standard_normal(128)
    response_vec /= np.linalg.norm(response_vec)

    # 5 anchors ordered with increasing similarity to response
    anchor_vecs = rng.standard_normal((5, 128))
    # inject deliberate alignment gradient
    for i in range(5):
        anchor_vecs[i] = (i * 0.4) * response_vec + anchor_vecs[i]
        anchor_vecs[i] /= np.linalg.norm(anchor_vecs[i])

    return response_vec, anchor_vecs


def test_cosine_similarity_relationship():
    """Verify that pymc_compute computes (1 + cos_raw)/2."""
    response_vec, anchor_vecs = _make_synthetic_fixtures()

    # canonical_ssr: raw cosine in [-1, 1]
    cos_raw = canonical_cosine_similarities(response_vec, anchor_vecs)

    # pymc_compute: (1 + cos) / 2
    # matrix_responses shape: (1, 128), matrix_likert_sentences shape: (128, 5)
    cos_pymc = pymc_compute.cosine_similarity_matrix(
        response_vec[None, :], anchor_vecs.T
    )[0]

    assert cos_raw.shape == (5,)
    assert cos_pymc.shape == (5,)

    expected_pymc = (1.0 + cos_raw) / 2.0
    assert np.allclose(cos_pymc, expected_pymc, atol=TOLERANCE, rtol=0.0)


def test_pmf_exact_parity_at_epsilon_zero():
    """Verify exact PMF identity when epsilon=0 (paper baseline).

    Mathematical proof of parity:
    w_raw = cos - min(cos)
    w_pymc = (1+cos)/2 - min((1+cos)/2) = (cos - min(cos))/2 = w_raw / 2
    p_pymc = (w_raw / 2) / sum(w_raw / 2) = w_raw / sum(w_raw) = p_raw.
    """
    response_vec, anchor_vecs = _make_synthetic_fixtures()

    cos_raw = canonical_cosine_similarities(response_vec, anchor_vecs)
    cos_pymc = pymc_compute.cosine_similarity_matrix(
        response_vec[None, :], anchor_vecs.T
    )

    pmf_canonical = canonical_similarities_to_pmf(cos_raw, epsilon=0.0, temperature=1.0)
    pmf_pymc = pymc_compute.similarities_to_pmf(cos_pymc, epsilon=0.0)[0]

    assert pmf_canonical.shape == (5,)
    assert pmf_pymc.shape == (5,)
    assert np.all(pmf_canonical >= 0.0)
    assert np.all(pmf_pymc >= 0.0)
    assert np.isclose(pmf_canonical.sum(), 1.0, atol=TOLERANCE)
    assert np.isclose(pmf_pymc.sum(), 1.0, atol=TOLERANCE)

    # Must be strictly identical to floating point tolerance
    diff = np.abs(pmf_canonical - pmf_pymc).max()
    assert diff < TOLERANCE, f"Max absolute difference {diff} exceeds tolerance {TOLERANCE}"


@pytest.mark.parametrize("temp", [0.5, 1.0, 1.5, 2.0])
def test_temperature_scaling_parity(temp: float):
    """Verify that temperature scaling produces identical PMFs across implementations."""
    response_vec, anchor_vecs = _make_synthetic_fixtures()

    cos_raw = canonical_cosine_similarities(response_vec, anchor_vecs)
    cos_pymc = pymc_compute.cosine_similarity_matrix(
        response_vec[None, :], anchor_vecs.T
    )

    # Canonical SSR with temperature
    pmf_canonical = canonical_similarities_to_pmf(cos_raw, epsilon=0.0, temperature=temp)

    # PyMC SSR with scale_pmf
    pmf_pymc_raw = pymc_compute.similarities_to_pmf(cos_pymc, epsilon=0.0)[0]
    pmf_pymc_scaled = pymc_compute.scale_pmf(pmf_pymc_raw, temperature=temp)

    assert np.allclose(pmf_canonical, pmf_pymc_scaled, atol=TOLERANCE, rtol=0.0)


def test_expected_rating_parity():
    """Verify that expected rating E[r] matches exactly."""
    response_vec, anchor_vecs = _make_synthetic_fixtures()

    cos_raw = canonical_cosine_similarities(response_vec, anchor_vecs)
    pmf_canonical = canonical_similarities_to_pmf(cos_raw, epsilon=0.0, temperature=1.0)

    # Canonical E[r]
    e_canonical = canonical_expected_rating(pmf_canonical)

    # PyMC style E[r] = sum(r * p(r)) for r in 1..5
    ratings = np.array([1, 2, 3, 4, 5], dtype=float)
    e_pymc = float(np.dot(ratings, pmf_canonical))

    assert np.isclose(e_canonical, e_pymc, atol=TOLERANCE)
    assert 1.0 <= e_canonical <= 5.0


def test_epsilon_convention_documentation_and_behavior():
    """Document and test behavior when epsilon > 0.

    When epsilon > 0, canonical_ssr operates on raw cosine scale,
    while pymc_compute operates on (1+cos)/2 scale.
    If epsilon_pymc is scaled by 0.5 (or vice-versa), they remain mathematically close.
    """
    response_vec, anchor_vecs = _make_synthetic_fixtures()

    cos_raw = canonical_cosine_similarities(response_vec, anchor_vecs)
    cos_pymc = pymc_compute.cosine_similarity_matrix(
        response_vec[None, :], anchor_vecs.T
    )

    eps_raw = 0.02
    eps_pymc = eps_raw / 2.0  # scaled to match affine factor of 1/2

    pmf_canonical = canonical_similarities_to_pmf(cos_raw, epsilon=eps_raw, temperature=1.0)
    pmf_pymc = pymc_compute.similarities_to_pmf(cos_pymc, epsilon=eps_pymc)[0]

    # When scaled properly, the PMFs match within tight numerical precision
    assert np.allclose(pmf_canonical, pmf_pymc, atol=1e-10)


def test_edge_case_uniform_similarities_handling():
    """Test behavior when all 5 similarities are exactly equal."""
    equal_anchors = np.ones((5, 10))
    resp = np.ones(10)

    # canonical_ssr at eps=0 raises ValueError as denominator is zero
    cos_raw = canonical_cosine_similarities(resp, equal_anchors)
    with pytest.raises(ValueError, match="SSR PMF is undefined for equal similarities"):
        canonical_similarities_to_pmf(cos_raw, epsilon=0.0)

    # canonical_ssr with eps > 0 handles equal similarities gracefully
    pmf_with_eps = canonical_similarities_to_pmf(cos_raw, epsilon=0.01)
    assert np.isclose(pmf_with_eps.sum(), 1.0)
