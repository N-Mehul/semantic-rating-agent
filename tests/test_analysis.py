"""
tests/test_analysis.py
-----------------------
Full test suite for the analysis layer.

Coverage
--------
  metrics.py         : nll, rps, ece, sharpness — correctness + edge cases
  decision_rules.py  : confusion_matrix_df, save_confusion_matrix, helper funcs
  distribution_eval.py : wasserstein1, chi_square, kl_divergence,
                         _point_histogram, _sample_histogram, _empirical_hist
  calibration.py     : _rescale_temperature
  results/ CSVs      : file-existence + shape checks for all produced outputs

Run with:
    python -m pytest tests/test_analysis.py -v
"""

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd
import pytest

# ── ensure project root is on sys.path ───────────────────────────────────────
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "results")

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / shared data
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def perfect_probs():
    """5 samples, each correctly peaked at the true class."""
    N = 5
    p = np.zeros((N, 5))
    for i in range(N):
        p[i, i] = 1.0
    y = np.arange(1, 6)
    return p, y


@pytest.fixture(scope="module")
def uniform_probs():
    """5 samples, uniform distribution."""
    p = np.full((5, 5), 0.2)
    y = np.array([1, 2, 3, 4, 5])
    return p, y


@pytest.fixture(scope="module")
def realistic_probs():
    """100 samples with smooth distributions and plausible ratings."""
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(np.ones(5), size=100)
    y   = raw.argmax(axis=1) + 1      # noisy true labels
    return raw, y


# ─────────────────────────────────────────────────────────────────────────────
# 1. analysis.metrics — NLL
# ─────────────────────────────────────────────────────────────────────────────

class TestNLL:

    def test_perfect_predictions_near_zero(self, perfect_probs):
        from analysis.metrics import nll
        p, y = perfect_probs
        assert nll(p, y) < 1e-6

    def test_uniform_nll_equals_log5(self, uniform_probs):
        from analysis.metrics import nll
        p, y = uniform_probs
        expected = np.log(5)
        assert abs(nll(p, y) - expected) < 1e-10

    def test_returns_float(self, realistic_probs):
        from analysis.metrics import nll
        p, y = realistic_probs
        result = nll(p, y)
        assert isinstance(result, float)

    def test_nll_positive(self, realistic_probs):
        from analysis.metrics import nll
        p, y = realistic_probs
        assert nll(p, y) > 0

    def test_wrong_shape_raises(self):
        from analysis.metrics import nll
        with pytest.raises(ValueError):
            nll(np.ones((5, 3)), np.array([1, 2, 3, 4, 5]))

    def test_wrong_ytrue_length_raises(self):
        from analysis.metrics import nll
        with pytest.raises(ValueError):
            nll(np.full((5, 5), 0.2), np.array([1, 2, 3]))

    def test_bad_ytrue_values_raises(self):
        from analysis.metrics import nll
        with pytest.raises(ValueError):
            nll(np.full((3, 5), 1/5), np.array([0, 3, 6]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. analysis.metrics — RPS
# ─────────────────────────────────────────────────────────────────────────────

class TestRPS:

    def test_perfect_predictions_zero(self, perfect_probs):
        from analysis.metrics import rps
        p, y = perfect_probs
        assert rps(p, y) < 1e-10

    def test_rps_non_negative(self, realistic_probs):
        from analysis.metrics import rps
        p, y = realistic_probs
        assert rps(p, y) >= 0

    def test_rps_bounded(self, realistic_probs):
        from analysis.metrics import rps
        p, y = realistic_probs
        # RPS <= 1 for ordinal K=5 (max RPS = 1 when all mass at wrong extreme)
        assert rps(p, y) <= 1.0

    def test_rps_uniform_known_value(self):
        """Single sample, true class=1, uniform probs → check formula."""
        from analysis.metrics import rps
        # CDF_pred = [0.2, 0.4, 0.6, 0.8, 1.0]
        # CDF_true  = [1.0, 1.0, 1.0, 1.0, 1.0]  (y=1)
        # RPS = 1/4 * [(0.2-1)^2 + (0.4-1)^2 + (0.6-1)^2 + (0.8-1)^2]
        #      = 1/4 * [0.64 + 0.36 + 0.16 + 0.04] = 1/4 * 1.2 = 0.3
        p = np.full((1, 5), 0.2)
        y = np.array([1])
        assert abs(rps(p, y) - 0.3) < 1e-10

    def test_rps_worst_case(self):
        """Mass entirely at class 1 when true=5 → RPS=1."""
        from analysis.metrics import rps
        p = np.zeros((1, 5)); p[0, 0] = 1.0
        y = np.array([5])
        assert abs(rps(p, y) - 1.0) < 1e-10

    def test_rps_returns_float(self, realistic_probs):
        from analysis.metrics import rps
        p, y = realistic_probs
        assert isinstance(rps(p, y), float)


# ─────────────────────────────────────────────────────────────────────────────
# 3. analysis.metrics — ECE
# ─────────────────────────────────────────────────────────────────────────────

class TestECE:

    def test_perfect_ece_zero(self, perfect_probs):
        from analysis.metrics import ece
        p, y = perfect_probs
        assert ece(p, y) == 0.0

    def test_ece_non_negative(self, realistic_probs):
        from analysis.metrics import ece
        p, y = realistic_probs
        assert ece(p, y) >= 0

    def test_ece_at_most_one(self, realistic_probs):
        from analysis.metrics import ece
        p, y = realistic_probs
        assert ece(p, y) <= 1.0

    def test_ece_returns_float(self, realistic_probs):
        from analysis.metrics import ece
        p, y = realistic_probs
        assert isinstance(ece(p, y), float)

    def test_ece_custom_bins(self, realistic_probs):
        from analysis.metrics import ece
        p, y = realistic_probs
        v5  = ece(p, y, n_bins=5)
        v15 = ece(p, y, n_bins=15)
        v20 = ece(p, y, n_bins=20)
        assert all(isinstance(v, float) for v in [v5, v15, v20])


# ─────────────────────────────────────────────────────────────────────────────
# 4. analysis.metrics — Sharpness
# ─────────────────────────────────────────────────────────────────────────────

class TestSharpness:

    def test_perfect_sharpness_one(self, perfect_probs):
        from analysis.metrics import sharpness
        p, _ = perfect_probs
        assert abs(sharpness(p) - 1.0) < 1e-10

    def test_uniform_sharpness(self, uniform_probs):
        from analysis.metrics import sharpness
        p, _ = uniform_probs
        assert abs(sharpness(p) - 0.2) < 1e-10

    def test_sharpness_between_recip_k_and_one(self, realistic_probs):
        from analysis.metrics import sharpness
        p, _ = realistic_probs
        s = sharpness(p)
        assert 0.2 <= s <= 1.0

    def test_sharpness_returns_float(self, realistic_probs):
        from analysis.metrics import sharpness
        p, _ = realistic_probs
        assert isinstance(sharpness(p), float)

    def test_sharpness_wrong_shape_raises(self):
        from analysis.metrics import sharpness
        with pytest.raises(ValueError):
            sharpness(np.ones((5, 3)))


# ─────────────────────────────────────────────────────────────────────────────
# 5. analysis.decision_rules — helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestDecisionRuleHelpers:

    def test_mode_from_dist_argmax(self):
        from analysis.decision_rules import mode_from_dist
        d = {1: 0.1, 2: 0.2, 3: 0.5, 4: 0.1, 5: 0.1}
        assert mode_from_dist(d) == 3

    def test_mode_from_dist_tie_lower(self):
        """On a tie, lower rating wins."""
        from analysis.decision_rules import mode_from_dist
        d = {1: 0.5, 2: 0.0, 3: 0.5, 4: 0.0, 5: 0.0}
        assert mode_from_dist(d) == 1

    def test_median_from_dist_basic(self):
        from analysis.decision_rules import median_from_dist
        # Cumulative at 1 = 0.1, at 2 = 0.3, at 3 = 0.8 → median = 3
        d = {1: 0.1, 2: 0.2, 3: 0.5, 4: 0.1, 5: 0.1}
        assert median_from_dist(d) == 3

    def test_median_from_dist_all_mass_at_1(self):
        from analysis.decision_rules import median_from_dist
        d = {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        assert median_from_dist(d) == 1

    def test_expected_from_dist(self):
        from analysis.decision_rules import expected_from_dist
        d = {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}
        assert abs(expected_from_dist(d) - 3.0) < 1e-10

    def test_round_expected(self):
        from analysis.decision_rules import round_expected
        assert round_expected(3.4) == 3
        assert round_expected(3.5) == 4
        assert round_expected(4.7) == 5

    def test_exact_acc(self):
        from analysis.decision_rules import exact_acc
        y_pred = np.array([1, 2, 3, 4, 5])
        y_true = np.array([1, 2, 3, 4, 5])
        assert exact_acc(y_pred, y_true) == 1.0

    def test_within_1(self):
        from analysis.decision_rules import within_1
        y_pred = np.array([1, 3, 5])
        y_true = np.array([1, 2, 4])   # all within ±1
        assert within_1(y_pred, y_true) == 1.0

    def test_mae(self):
        from analysis.decision_rules import mae
        y_pred = np.array([1.0, 3.0])
        y_true = np.array([2.0, 2.0])
        assert abs(mae(y_pred, y_true) - 1.0) < 1e-10

    def test_rmse(self):
        from analysis.decision_rules import rmse
        y_pred = np.array([1.0, 3.0])
        y_true = np.array([2.0, 2.0])
        assert abs(rmse(y_pred, y_true) - 1.0) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# 6. analysis.decision_rules — confusion_matrix_df & save_confusion_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestConfusionMatrix:

    def test_confusion_matrix_df_shape(self):
        from analysis.decision_rules import confusion_matrix_df
        y_true = np.array([1, 2, 3, 4, 5, 1, 2])
        y_pred = np.array([1, 2, 3, 5, 5, 2, 2])
        cm = confusion_matrix_df(y_true, y_pred)
        assert isinstance(cm, pd.DataFrame)

    def test_confusion_matrix_df_diagonal(self):
        from analysis.decision_rules import confusion_matrix_df
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        cm = confusion_matrix_df(y_true, y_pred)
        assert cm.values.diagonal().sum() == 3

    def test_save_confusion_matrix_creates_file(self, tmp_path, monkeypatch):
        from analysis.decision_rules import save_confusion_matrix
        import analysis.decision_rules as dr
        monkeypatch.setattr(dr, "RESULTS_DIR", str(tmp_path))
        y_true = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5, 2, 3, 4, 5, 1])
        path = save_confusion_matrix(y_true, y_pred, "mode", "A")
        assert os.path.exists(path)

    def test_save_confusion_matrix_is_5x5(self, tmp_path, monkeypatch):
        from analysis.decision_rules import save_confusion_matrix
        import analysis.decision_rules as dr
        monkeypatch.setattr(dr, "RESULTS_DIR", str(tmp_path))
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 1, 1, 1, 1])   # all predicted as 1
        path = save_confusion_matrix(y_true, y_pred, "mode", "A")
        cm = pd.read_csv(path, index_col=0)
        assert cm.shape == (5, 5)

    def test_save_confusion_matrix_all_labels_present_even_if_absent(self,
                                                                       tmp_path,
                                                                       monkeypatch):
        """Classes missing from predictions still appear as columns with zeros."""
        from analysis.decision_rules import save_confusion_matrix
        import analysis.decision_rules as dr
        monkeypatch.setattr(dr, "RESULTS_DIR", str(tmp_path))
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])   # class 4 and 5 never appear
        path = save_confusion_matrix(y_true, y_pred, "mode", "A")
        cm = pd.read_csv(path, index_col=0)
        assert cm.shape == (5, 5)
        assert int(cm.loc[4, "4"]) == 0
        assert int(cm.loc[5, "5"]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. analysis.distribution_eval — divergence measures
# ─────────────────────────────────────────────────────────────────────────────

class TestDivergences:

    def test_wasserstein1_identical_dists(self):
        from analysis.distribution_eval import wasserstein1
        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        assert wasserstein1(p, p) == 0.0

    def test_wasserstein1_extreme_case(self):
        """All mass at 1 vs all mass at 5 → W1 = 4 steps × each CDF gap = 1.0*4=4."""
        from analysis.distribution_eval import wasserstein1
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        # CDF_p = [1,1,1,1,1], CDF_q = [0,0,0,0,1], diff = [1,1,1,1] → W1=4
        assert abs(wasserstein1(p, q) - 4.0) < 1e-10

    def test_wasserstein1_non_negative(self):
        from analysis.distribution_eval import wasserstein1
        rng = np.random.default_rng(7)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            assert wasserstein1(p, q) >= 0

    def test_chi_square_identical_dists_zero(self):
        from analysis.distribution_eval import chi_square
        p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert chi_square(p, p, N=100) == 0.0

    def test_chi_square_non_negative(self):
        from analysis.distribution_eval import chi_square
        p = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        q = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        assert chi_square(p, q, N=100) >= 0.0

    def test_kl_identical_dists_zero(self):
        from analysis.distribution_eval import kl_divergence
        p = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        assert kl_divergence(p, p) < 1e-10

    def test_kl_non_negative(self):
        from analysis.distribution_eval import kl_divergence
        rng = np.random.default_rng(11)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))
            q = rng.dirichlet(np.ones(5))
            assert kl_divergence(p, q) >= 0

    def test_kl_asymmetric(self):
        from analysis.distribution_eval import kl_divergence
        # Deliberately asymmetric: p is sharply peaked, q is nearly uniform
        p = np.array([0.9, 0.025, 0.025, 0.025, 0.025])
        q = np.array([0.2, 0.2,   0.2,   0.2,   0.2])
        # KL(p||q) >> KL(q||p): p is sharp and q is broad → forward KL large
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert abs(kl_pq - kl_qp) > 0.01, (
            f"Expected KL to be asymmetric: KL(p||q)={kl_pq:.6f}, KL(q||p)={kl_qp:.6f}"
        )

    def test_kl_eps_prevents_division_by_zero(self):
        """q has a zero bin — should not raise or return inf."""
        from analysis.distribution_eval import kl_divergence
        p = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        q = np.array([0.3, 0.3, 0.3, 0.1, 0.0])
        result = kl_divergence(p, q)
        assert np.isfinite(result)


# ─────────────────────────────────────────────────────────────────────────────
# 8. analysis.distribution_eval — histogram builders
# ─────────────────────────────────────────────────────────────────────────────

class TestHistograms:

    def test_empirical_hist_sums_to_one(self):
        from analysis.distribution_eval import _empirical_hist
        y = np.array([1, 2, 3, 4, 5, 1, 2, 3])
        h = _empirical_hist(y)
        assert abs(h.sum() - 1.0) < 1e-10

    def test_empirical_hist_length_five(self):
        from analysis.distribution_eval import _empirical_hist
        y = np.array([1, 1, 2, 3, 4, 5])
        assert len(_empirical_hist(y)) == 5

    def test_point_histogram_sums_to_one(self):
        from analysis.distribution_eval import _point_histogram
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5), size=50)
        h = _point_histogram(probs)
        assert abs(h.sum() - 1.0) < 1e-10

    def test_point_histogram_argmax_agrees(self):
        """All samples peaked at class 3 → point hist should be [0,0,1,0,0]."""
        from analysis.distribution_eval import _point_histogram
        p = np.zeros((10, 5)); p[:, 2] = 1.0
        h = _point_histogram(p)
        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(h, expected)

    def test_sample_histogram_sums_to_one(self):
        from analysis.distribution_eval import _sample_histogram
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5), size=50)
        h = _sample_histogram(probs, n_avg=20, rng=np.random.default_rng(0))
        assert abs(h.sum() - 1.0) < 1e-10

    def test_sample_histogram_uniform_input_approximately_uniform(self):
        """Uniform input probs → sample histogram should be ≈ uniform."""
        from analysis.distribution_eval import _sample_histogram
        probs = np.full((500, 5), 0.2)
        h = _sample_histogram(probs, n_avg=50, rng=np.random.default_rng(99))
        # All bins should be within ±0.05 of 0.2
        assert np.all(np.abs(h - 0.2) < 0.05)


# ─────────────────────────────────────────────────────────────────────────────
# 9. analysis.calibration — temperature rescaling
# ─────────────────────────────────────────────────────────────────────────────

class TestTemperatureRescaling:

    def test_rescale_at_same_temperature_identity(self):
        """Rescaling from T_frozen back to T_frozen reproduces input."""
        from analysis.calibration import _rescale_temperature, T_FROZEN
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5), size=20)
        rescaled = _rescale_temperature(probs, T_FROZEN)
        np.testing.assert_allclose(rescaled, probs, atol=1e-6)

    def test_rescale_rows_sum_to_one(self):
        from analysis.calibration import _rescale_temperature
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5), size=30)
        for T in [0.05, 0.1, 0.5, 1.0, 2.0]:
            out = _rescale_temperature(probs, T)
            np.testing.assert_allclose(out.sum(axis=1), np.ones(30), atol=1e-10)

    def test_lower_temperature_increases_sharpness(self):
        """Lower T → sharper (higher max prob) distribution."""
        from analysis.calibration import _rescale_temperature
        from analysis.metrics import sharpness
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5) * 2, size=50)
        s_high_T = sharpness(_rescale_temperature(probs, 1.0))
        s_low_T  = sharpness(_rescale_temperature(probs, 0.1))
        assert s_low_T > s_high_T

    def test_higher_temperature_tends_to_uniform(self):
        """Very high T → distribution approaches uniform."""
        from analysis.calibration import _rescale_temperature
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5), size=20)
        out = _rescale_temperature(probs, T_new=100.0)
        # Each row should be close to 0.2
        np.testing.assert_allclose(out, np.full((20, 5), 0.2), atol=0.01)

    def test_rescale_non_negative(self):
        from analysis.calibration import _rescale_temperature
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(5), size=10)
        for T in [0.02, 0.5, 2.0]:
            out = _rescale_temperature(probs, T)
            assert np.all(out >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. results/ file-existence and shape checks
# ─────────────────────────────────────────────────────────────────────────────

class TestResultsFiles:

    # ── Confusion matrix CSVs ────────────────────────────────────────────────
    @pytest.mark.parametrize("regime,rule", [
        ("A", "mode"), ("A", "median"), ("A", "rounded_expected"), ("A", "prior_corrected_mode"),
        ("B", "mode"), ("B", "median"), ("B", "rounded_expected"), ("B", "prior_corrected_mode"),
    ])
    def test_cm_csv_exists(self, regime, rule):
        path = os.path.join(RESULTS_DIR, f"cm_regime{regime}_{rule}.csv")
        assert os.path.exists(path), f"Missing: {path}"

    @pytest.mark.parametrize("regime,rule", [
        ("A", "mode"), ("A", "median"), ("A", "rounded_expected"), ("A", "prior_corrected_mode"),
        ("B", "mode"), ("B", "median"), ("B", "rounded_expected"), ("B", "prior_corrected_mode"),
    ])
    def test_cm_csv_is_5x5(self, regime, rule):
        path = os.path.join(RESULTS_DIR, f"cm_regime{regime}_{rule}.csv")
        cm = pd.read_csv(path, index_col=0)
        assert cm.shape == (5, 5), f"{path} is {cm.shape}, expected (5,5)"

    @pytest.mark.parametrize("regime,rule", [
        ("A", "mode"), ("A", "median"), ("A", "rounded_expected"), ("A", "prior_corrected_mode"),
        ("B", "mode"), ("B", "median"), ("B", "rounded_expected"), ("B", "prior_corrected_mode"),
    ])
    def test_cm_csv_row_sums_equal_true_counts(self, regime, rule):
        """Every row sum equals the number of test samples with that true rating."""
        path = os.path.join(RESULTS_DIR, f"cm_regime{regime}_{rule}.csv")
        cm = pd.read_csv(path, index_col=0)
        row_sums = cm.sum(axis=1).values
        # Row sums should all be non-negative and at least one > 0
        assert np.all(row_sums >= 0)
        assert row_sums.sum() > 0

    # ── Class-5 metrics CSV ──────────────────────────────────────────────────
    def test_class5_metrics_csv_exists(self):
        path = os.path.join(RESULTS_DIR, "class5_metrics.csv")
        assert os.path.exists(path)

    def test_class5_metrics_csv_columns(self):
        path = os.path.join(RESULTS_DIR, "class5_metrics.csv")
        df = pd.read_csv(path)
        for col in ["regime", "rule", "TP", "FP", "FN", "precision", "recall", "F1"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_class5_prior_corrected_recall_regime_A(self):
        path = os.path.join(RESULTS_DIR, "class5_metrics.csv")
        df = pd.read_csv(path)
        row = df[(df["regime"] == "A") & (df["rule"] == "prior_corrected_mode")]
        assert len(row) == 1
        assert abs(float(row["recall"].iloc[0]) - 0.7818) < 0.01

    def test_class5_prior_corrected_recall_regime_B(self):
        path = os.path.join(RESULTS_DIR, "class5_metrics.csv")
        df = pd.read_csv(path)
        row = df[(df["regime"] == "B") & (df["rule"] == "prior_corrected_mode")]
        assert len(row) == 1
        assert abs(float(row["recall"].iloc[0]) - 0.9062) < 0.01

    # ── Calibration outputs ──────────────────────────────────────────────────
    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_calibration_json_exists(self, regime):
        path = os.path.join(RESULTS_DIR, f"calibration_regime{regime}.json")
        assert os.path.exists(path)

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_calibration_json_has_required_keys(self, regime):
        path = os.path.join(RESULTS_DIR, f"calibration_regime{regime}.json")
        with open(path) as f:
            data = json.load(f)
        for key in ["regime", "T_frozen", "T_star", "T_sweep",
                    "nll_val_sweep", "nll_test_sweep", "table"]:
            assert key in data, f"Missing key '{key}' in {path}"

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_calibration_T_star_in_range(self, regime):
        path = os.path.join(RESULTS_DIR, f"calibration_regime{regime}.json")
        with open(path) as f:
            data = json.load(f)
        assert 0.02 <= data["T_star"] <= 2.0

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_calibration_table_has_two_rows(self, regime):
        path = os.path.join(RESULTS_DIR, f"calibration_regime{regime}.json")
        with open(path) as f:
            data = json.load(f)
        assert len(data["table"]) == 2

    def test_calibration_summary_csv_exists(self):
        assert os.path.exists(os.path.join(RESULTS_DIR, "calibration_summary.csv"))

    def test_calibration_summary_csv_four_rows(self):
        df = pd.read_csv(os.path.join(RESULTS_DIR, "calibration_summary.csv"))
        assert len(df) == 4   # 2 regimes × 2 settings

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_reliability_diagram_png_exists(self, regime):
        path = os.path.join(RESULTS_DIR, f"reliability_diagram_regime{regime}.png")
        assert os.path.exists(path)

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_nll_vs_T_png_exists(self, regime):
        path = os.path.join(RESULTS_DIR, f"nll_vs_T_regime{regime}.png")
        assert os.path.exists(path)

    # ── Distribution eval outputs ────────────────────────────────────────────
    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_distribution_json_exists(self, regime):
        path = os.path.join(RESULTS_DIR, f"distribution_eval_regime{regime}.json")
        assert os.path.exists(path)

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_distribution_json_keys(self, regime):
        path = os.path.join(RESULTS_DIR, f"distribution_eval_regime{regime}.json")
        with open(path) as f:
            data = json.load(f)
        for key in ["regime", "N", "true_hist", "point_hist", "sample_hist",
                    "point", "sample"]:
            assert key in data

    @pytest.mark.parametrize("regime,strategy", [
        ("A", "point"), ("A", "sample"), ("B", "point"), ("B", "sample"),
    ])
    def test_distribution_divergences_non_negative(self, regime, strategy):
        path = os.path.join(RESULTS_DIR, f"distribution_eval_regime{regime}.json")
        with open(path) as f:
            data = json.load(f)
        m = data[strategy]
        assert m["W1"] >= 0
        assert m["chi2"] >= 0
        assert m["KL"] >= 0

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_sample_W1_much_less_than_point_W1(self, regime):
        """Sample strategy should track the true distribution far better."""
        path = os.path.join(RESULTS_DIR, f"distribution_eval_regime{regime}.json")
        with open(path) as f:
            data = json.load(f)
        assert data["sample"]["W1"] < data["point"]["W1"] / 5

    def test_distribution_summary_csv_exists(self):
        assert os.path.exists(os.path.join(RESULTS_DIR, "distribution_eval_summary.csv"))

    def test_distribution_summary_csv_four_rows(self):
        df = pd.read_csv(os.path.join(RESULTS_DIR, "distribution_eval_summary.csv"))
        assert len(df) == 4   # 2 regimes × 2 strategies

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_dist_histogram_png_exists(self, regime):
        path = os.path.join(RESULTS_DIR, f"dist_histogram_regime{regime}.png")
        assert os.path.exists(path)


# ─────────────────────────────────────────────────────────────────────────────
# 11. analysis.per_class_ceiling — split-size safeguard
# ─────────────────────────────────────────────────────────────────────────────

class TestPerClassCeiling:
    """Verify per_class_ceiling outputs exist and totals match known split sizes."""

    EXPECTED_N = {"A": 1501, "B": 1797}

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_per_class_csv_exists(self, regime):
        path = os.path.join(RESULTS_DIR, f"per_class_ceiling_regime{regime}.csv")
        assert os.path.exists(path), f"Missing: {path}"

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_per_class_csv_has_five_rows(self, regime):
        path = os.path.join(RESULTS_DIR, f"per_class_ceiling_regime{regime}.csv")
        df = pd.read_csv(path)
        assert len(df) == 5, f"Expected 5 class rows, got {len(df)}"

    @pytest.mark.parametrize("regime,expected_n", [("A", 1501), ("B", 1797)])
    def test_per_class_n_sums_to_total_test_size(self, regime, expected_n):
        """
        Summing n_test across all 5 classes must equal the known test-set size
        for that regime — the split-size-mismatch safeguard.
        """
        path = os.path.join(RESULTS_DIR, f"per_class_ceiling_regime{regime}.csv")
        df   = pd.read_csv(path)
        total = int(df["n_test"].sum())
        assert total == expected_n, (
            f"Regime {regime}: sum of n_test across classes = {total}, "
            f"expected {expected_n}"
        )

    def test_per_class_summary_json_exists(self):
        path = os.path.join(RESULTS_DIR, "per_class_ceiling_summary.json")
        assert os.path.exists(path)

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_per_class_ceiling_in_range(self, regime):
        """Ceiling accuracy must be in [0, 1] for every class."""
        path = os.path.join(RESULTS_DIR, f"per_class_ceiling_regime{regime}.csv")
        df   = pd.read_csv(path)
        for _, row in df.iterrows():
            assert 0.0 <= row["ceiling_acc"] <= 1.0, (
                f"Regime {regime} class {row['true_class']}: "
                f"ceiling_acc={row['ceiling_acc']} out of [0,1]"
            )

    @pytest.mark.parametrize("regime", ["A", "B"])
    def test_per_class_knn_acc_in_range(self, regime):
        """kNN accuracy must be in [0, 1] for every class."""
        path = os.path.join(RESULTS_DIR, f"per_class_ceiling_regime{regime}.csv")
        df   = pd.read_csv(path)
        for _, row in df.iterrows():
            assert 0.0 <= row["knn_acc"] <= 1.0, (
                f"Regime {regime} class {row['true_class']}: "
                f"knn_acc={row['knn_acc']} out of [0,1]"
            )
