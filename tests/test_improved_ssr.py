"""Unit tests for improved_ssr.py and verification of canonical SSR invariance."""

import numpy as np
import pytest

from canonical_ssr import (
    PAPER_EPSILON,
    PAPER_TEMPERATURE,
    RATINGS,
    SingleConstructSSRResult,
    cosine_similarities,
    expected_rating,
    similarities_to_pmf,
)
from improved_ssr import (
    DEFAULT_IMPROVED_K,
    DEFAULT_IMPROVED_TEMPERATURE,
    LinearVarianceRescaler,
    improved_similarities_to_pmf,
    score_embedding_improved,
)


class TestCanonicalSSRInvariance:
    """Ensure canonical SSR implementation remains 100% intact and unchanged."""

    def test_canonical_parameters_unchanged(self):
        assert PAPER_EPSILON == 0.0
        assert PAPER_TEMPERATURE == 1.0
        assert RATINGS == (1, 2, 3, 4, 5)

    def test_canonical_pmf_calculation(self):
        # Sample cosine similarities
        sims = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        pmf_canon = similarities_to_pmf(sims, epsilon=0.0, temperature=1.0)

        # Expected weights: [0.0, 0.1, 0.2, 0.3, 0.4] -> sum = 1.0
        assert np.allclose(pmf_canon, [0.0, 0.1, 0.2, 0.3, 0.4], atol=1e-6)
        assert np.isclose(pmf_canon.sum(), 1.0)

        exp_canon = expected_rating(pmf_canon)
        # Expected: 1*0 + 2*0.1 + 3*0.2 + 4*0.3 + 5*0.4 = 0.2 + 0.6 + 1.2 + 2.0 = 4.0
        assert np.isclose(exp_canon, 4.0)


class TestImprovedSSR:
    """Verify improved SSR behavior, top-k truncation, temperature scaling, and validity."""

    def test_improved_defaults(self):
        assert DEFAULT_IMPROVED_K == 3
        assert DEFAULT_IMPROVED_TEMPERATURE == 0.75

    def test_top_k_truncation(self):
        # sims: lowest is anchor 1 (0.50), next is anchor 2 (0.60)
        # For k=3, top 3 are anchors 3, 4, 5
        sims = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        pmf_imp = improved_similarities_to_pmf(sims, k=3, temperature=1.0)

        # Anchors 1 and 2 must have exactly zero probability
        assert pmf_imp[0] == 0.0
        assert pmf_imp[1] == 0.0
        assert pmf_imp[2] > 0.0
        assert pmf_imp[3] > 0.0
        assert pmf_imp[4] > 0.0
        assert np.isclose(pmf_imp.sum(), 1.0)

    def test_temperature_sharpening(self):
        sims = np.array([0.50, 0.60, 0.70, 0.80, 0.90])
        pmf_t1 = improved_similarities_to_pmf(sims, k=3, temperature=1.0)
        pmf_t075 = improved_similarities_to_pmf(sims, k=3, temperature=0.75)

        # T=0.75 sharpens the probability toward the dominant anchor (anchor 5)
        assert pmf_t075[4] > pmf_t1[4]
        assert np.isclose(pmf_t075.sum(), 1.0)

    def test_expected_rating_within_bounds(self):
        # Test various random similarity profiles
        rng = np.random.default_rng(seed=42)
        for _ in range(50):
            sims = rng.uniform(0.3, 0.9, size=5)
            pmf = improved_similarities_to_pmf(sims, k=3, temperature=0.75)
            exp = expected_rating(pmf)
            assert 1.0 <= exp <= 5.0
            assert np.isclose(pmf.sum(), 1.0, atol=1e-6)

    def test_score_embedding_improved_interface(self):
        resp_vec = np.array([0.1, 0.2, 0.3, 0.4])
        anch_vecs = np.array([
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3],
            [0.4, 0.4, 0.4, 0.4],
            [0.5, 0.5, 0.5, 0.5],
        ])
        res = score_embedding_improved(resp_vec, anch_vecs, k=3, temperature=0.75)
        assert isinstance(res, SingleConstructSSRResult)
        assert res.probabilities.shape == (5,)
        assert np.isclose(res.probabilities.sum(), 1.0)
        assert 1.0 <= res.expected_rating <= 5.0

    def test_non_direct_llm_rating_property(self):
        """Verify ratings are derived via semantic projection over anchor vectors, not LLM token scoring."""
        resp_vec = np.array([1.0, 0.0])
        anch_vecs = np.array([
            [0.0, 1.0],  # 1
            [0.2, 0.8],  # 2
            [0.5, 0.5],  # 3
            [0.8, 0.2],  # 4
            [1.0, 0.0],  # 5 (identical)
        ])
        res = score_embedding_improved(resp_vec, anch_vecs, k=3, temperature=0.75)
        # Anchor 5 must receive highest similarity and probability
        assert np.argmax(res.probabilities) == 4
        assert res.expected_rating > 4.0


class TestLinearVarianceRescaler:
    """Verify optional component C: Linear Variance Rescaler."""

    def test_rescaler_disabled_by_default(self):
        rescaler = LinearVarianceRescaler(
            mu_train=3.6,
            sigma_train=1.0,
            mu_synth=3.3,
            sigma_synth=0.6,
            enabled=False,
        )
        assert rescaler.enabled is False
        assert rescaler.rescale(3.5) == 3.5

    def test_rescaler_enabled_rescaling(self):
        rescaler = LinearVarianceRescaler(
            mu_train=3.6,
            sigma_train=1.2,
            mu_synth=3.3,
            sigma_synth=0.6,
            enabled=True,
        )
        # Scale factor = 1.2 / 0.6 = 2.0
        # y = 3.6 + (3.3 - 3.3)*2.0 = 3.6
        assert np.isclose(rescaler.rescale(3.3), 3.6)
        # y = 3.6 + (3.8 - 3.3)*2.0 = 3.6 + 1.0 = 4.6
        assert np.isclose(rescaler.rescale(3.8), 4.6)

    def test_rescaler_clipping(self):
        rescaler = LinearVarianceRescaler(
            mu_train=3.5,
            sigma_train=2.0,
            mu_synth=3.0,
            sigma_synth=0.5,
            clip_min=1.0,
            clip_max=5.0,
            enabled=True,
        )
        # Scale factor = 4.0
        # x = 4.0 -> y = 3.5 + 4.0 = 7.5 -> clipped to 5.0
        assert rescaler.rescale(4.0) == 5.0
        # x = 2.0 -> y = 3.5 - 4.0 = -0.5 -> clipped to 1.0
        assert rescaler.rescale(2.0) == 1.0
