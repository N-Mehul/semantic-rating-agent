"""Deterministic tests for scripts/ssr_hybrid_experiments.py (Phase 8)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ssr_hybrid_experiments as H  # noqa: E402

# E = 4.0 (median 4); E = 4.45 (median 5); E = 2.6 (median 1); uniform E = 3 (median 3)
P = np.array([[0.0, 0.1, 0.2, 0.3, 0.4],
              [0.0, 0.0, 0.1, 0.35, 0.55],
              [0.6, 0.0, 0.0, 0.0, 0.4],
              [0.2, 0.2, 0.2, 0.2, 0.2]])


def test_base_rules():
    assert np.allclose(H.expected(P), [4.0, 4.45, 2.6, 3.0])
    assert H.rule_rounded_E(P).tolist() == [4, 4, 3, 3]
    assert H.rule_median(P).tolist() == [4, 5, 1, 3]
    assert H.round_half_up([3.5, 4.5, 2.49]).tolist() == [4, 5, 2]


def test_weighted_average_and_rounding():
    # 0.5*4.45 + 0.5*5 = 4.725 -> 5 ; 0.5*2.6 + 0.5*1 = 1.8 -> 2
    assert H.rule_weighted(P, 0.5).tolist() == [4, 5, 2, 3]
    assert H.rule_weighted(P, 1.0).tolist() == H.rule_rounded_E(P).tolist()
    assert H.rule_weighted(P, 0.0).tolist() == H.rule_median(P).tolist()
    # 0.75*4.45 + 0.25*5 = 4.5875 -> 5 ; 0.25*4.45 + 0.75*5 = 4.86 -> 5
    assert H.rule_weighted(P, 0.75)[1] == 5 and H.rule_weighted(P, 0.25)[1] == 5


def test_integer_average_tie_breaks_toward_E():
    # row 1: round(E)=4, median=5 -> 4.5 tie; E=4.45 is closer to 4 -> 4
    # row 2: round(E)=3, median=1 -> 2 exactly
    assert H.rule_integer_average(P).tolist() == [4, 4, 2, 3]


def test_upper_gate():
    assert H.rule_upper_gate(P, 0.5).tolist() == [4, 5, 3, 3]        # only row 1 has P5 >= 0.5
    assert H.rule_upper_gate(P, 0.99).tolist() == H.rule_rounded_E(P).tolist()


def test_confidence_measures_and_weighting():
    ce, cm = H.confidence_entropy(P), H.confidence_maxprob(P)
    assert ce[3] == pytest.approx(0.0) and cm[3] == pytest.approx(0.0)       # uniform -> zero confidence
    assert np.all((ce >= 0) & (ce <= 1)) and np.all((cm >= 0) & (cm <= 1))
    one_hot = np.array([[0, 0, 0, 1.0, 0]])
    assert H.confidence_entropy(one_hot)[0] == pytest.approx(1.0) and H.confidence_maxprob(one_hot)[0] == pytest.approx(1.0)
    w = H.rule_confidence_weighted(P, H.confidence_maxprob)
    c = cm; expected = H.round_half_up(c * H.pmf_median(P) + (1 - c) * H.expected(P))
    assert w.tolist() == expected.tolist()


def test_threshold_rule_regions():
    assert H.rule_conf_threshold(P, 1.01, 1.01).tolist() == H.rule_rounded_E(P).tolist()   # never switches
    assert H.rule_conf_threshold(P, 0.0, 0.0).tolist() == H.rule_median(P).tolist()        # always median
    mid = H.rule_conf_threshold(P, 0.0, 1.01)                                             # always integer average
    assert mid.tolist() == H.rule_integer_average(P).tolist()


def test_threshold_and_tau_selection_deterministic_and_tie_to_current():
    y = H.rule_rounded_E(P)                       # labels that the current rule already gets right
    assert H.select_tau(P, y) == max(H.TAU_GRID)  # tie -> keep closest to current rule
    assert H.select_thresholds(P, y) == (1.01, 1.01)
    y2 = H.rule_median(P)
    t = H.select_thresholds(P, y2)
    assert H.rule_conf_threshold(P, *t).tolist() == y2.tolist()
    assert H.select_thresholds(P, y2) == t        # deterministic


def test_construct_selection():
    cons = np.array(["PE", "PE", "HM", "HM"])
    y = np.array([4, 5, 3, 3])                    # PE: median right on both; HM: E right on both
    choice = H.select_construct_rules(P, y, cons)
    assert choice == {"HM": "E", "PE": "median"}
    pred = H.rule_by_construct(P, cons, choice)
    assert pred.tolist() == [4, 5, 3, 3]
    assert H.rule_by_construct(P, ["XX"] * 4, choice).tolist() == H.rule_rounded_E(P).tolist()   # unseen -> E


def test_calibration_separation_and_sufficiency():
    with pytest.raises(ValueError):
        H.assert_disjoint(["a", "b"], ["b", "c"])
    H.assert_disjoint(["a"], ["b"])
    small = H.calibration_sufficient(np.array([3.0, 4.0, 4.5]), np.array([3, 4, 5]))
    assert small["sufficient"] is False
    rng = np.random.default_rng(0); y = np.repeat([3, 4, 5], 40); E = y + rng.normal(0, 0.3, len(y))
    assert H.calibration_sufficient(E, y)["sufficient"] is True


def test_canonical_ssr_behaviour_unchanged():
    from canonical_ssr import similarities_to_pmf, expected_rating
    from improved_ssr import improved_similarities_to_pmf
    p = similarities_to_pmf(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert np.allclose(p, [0.0, 0.1, 0.2, 0.3, 0.4]) and expected_rating(p) == pytest.approx(4.0)
    pi = improved_similarities_to_pmf(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), k=3, temperature=0.75)
    w = np.array([0.2, 0.3, 0.4]) ** (1 / 0.75); w /= w.sum()
    assert np.allclose(pi, [0, 0, *w])


def test_saved_pmfs_reproduce_phase6_expected_ratings():
    T, C = H.load_data()
    assert np.allclose(H.expected(T["P"]), T["E_stored"], atol=1e-3)
    assert len(T["y"]) == 24 and len(C["y"]) == 48
    H.assert_disjoint(T["text"], C["text"])
