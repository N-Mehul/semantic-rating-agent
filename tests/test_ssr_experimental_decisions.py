"""Deterministic tests for scripts/ssr_experimental_decisions.py (Phase 7 candidates)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ssr_experimental_decisions as X  # noqa: E402

P = np.array([[0.0, 0.1, 0.2, 0.3, 0.4],     # E = 4.0, mode 5, median 4
              [0.0, 0.0, 0.0, 0.5, 0.5],     # E = 4.5, mode 4 (tie -> lower), median 4
              [0.6, 0.0, 0.0, 0.0, 0.4]])    # E = 2.6, mode 1, median 1


def test_expected_and_round_half_up():
    assert np.allclose(X.expected(P), [4.0, 4.5, 2.6])
    assert X.round_expected([4.5, 3.5, 2.49, 0.2, 6.0]).tolist() == [5, 4, 2, 1, 5]


def test_argmax_ties_to_lower_and_median():
    assert X.pmf_argmax(P).tolist() == [5, 4, 1]
    assert X.pmf_median(P).tolist() == [4, 4, 1]


def test_nearest_anchor():
    assert X.nearest_anchor([[0.1, 0.2, 0.3, 0.5, 0.4], [0.9, 0.1, 0.1, 0.1, 0.1]]).tolist() == [4, 1]


def test_bias_shift_and_clip():
    b = X.BiasShift.fit([4.0, 4.0], [3, 5])          # mean(E - y) = 0
    assert b.shift == 0.0
    b2 = X.BiasShift.fit([4.5, 4.5], [4, 4])          # +0.5 bias removed
    assert np.allclose(b2.transform([4.5, 1.2]), [4.0, 1.0])


def test_construct_shift_shrinks_toward_global():
    cs = X.ConstructShift.fit([5, 5, 3, 3], [4, 4, 3, 3], ["PE", "PE", "EE", "EE"], m=2.0)
    assert cs.global_shift == pytest.approx(0.5)
    assert cs.shifts["PE"] == pytest.approx((2 * 1.0 + 2 * 0.5) / 4)
    assert cs.shifts["EE"] == pytest.approx((2 * 0.0 + 2 * 0.5) / 4)
    assert cs.transform([4.0], ["UNSEEN"])[0] == pytest.approx(3.5)   # unseen construct -> global


def test_variance_match_maps_moments():
    vm = X.VarianceMatch.fit([3.0, 4.0, 5.0], [2.0, 4.0, 6.0])
    out = vm.transform([3.0, 4.0, 5.0])
    assert np.allclose(out, [2.0, 4.0, 5.0])            # 6 clipped to 5


def test_cutpoints_default_equals_rounding_and_fit_is_monotone():
    E = np.array([1.4, 2.6, 3.49, 3.5, 4.49, 4.5])
    assert X.OrdinalCutpoints().predict(E).tolist() == X.round_expected(E).tolist()
    E_cal = np.array([3.1, 3.3, 3.9, 4.0, 4.2, 4.3]); y_cal = np.array([3, 3, 4, 4, 5, 5])
    cp = X.OrdinalCutpoints.fit(E_cal, y_cal)
    assert list(cp.thresholds) == sorted(cp.thresholds)
    assert cp.predict(E_cal).tolist() == y_cal.tolist()            # separable calibration data recovered exactly
    assert cp.thresholds[:2] == (1.5, 2.5)                          # unidentified thresholds keep defaults


def test_cutpoint_fit_is_deterministic():
    E_cal = np.linspace(3, 4.5, 12); y_cal = np.array([3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 4])
    assert X.OrdinalCutpoints.fit(E_cal, y_cal).thresholds == X.OrdinalCutpoints.fit(E_cal, y_cal).thresholds
