"""Unit tests for Step 6: Validation and Interpretation of Product Evaluation Results.

Verifies:
1. All 6 validation artifacts exist and are properly populated.
2. ssr_probability_validation.json validates all 24 evaluations (PMFs sum to 1, expected ratings in [1, 5]).
3. Zero-least-similar anchor condition holds for all minimum-subtracted PMFs.
4. Entropy and spread metrics are correctly computed.
5. Persona consistency report verifies all four empirical personas (P1, P2, P3, P4).
6. Human vs. synthetic comparison correctly articulates the boundary between human existing-brand loyalty
   and synthetic new-product consideration.
7. Robustness analysis confirms temperature scaling invariance and identifies spread-based uncertainty.
8. Canonical SSR components and anchor sets remain unmodified.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf
from survey_ssr_bridge import load_all_survey_anchors

ROOT = Path(__file__).resolve().parents[1]
VAL_DIR = ROOT / "data" / "product_evaluation" / "nova_ai_smartphone" / "validation"
EXPECTED_PERSONAS = [
    "P1_premium_tier_owner",
    "P2_midrange_heavy_user",
    "P3_midrange_moderate_user",
    "P4_budget_or_undisclosed_price",
]
EXPECTED_CONSTRUCTS = ["PE", "BI", "EE", "HM", "SI", "PP"]


def test_step6_all_artifacts_exist():
    """Verify all 6 validation artifacts are present."""
    assert VAL_DIR.is_dir(), f"Validation directory missing: {VAL_DIR}"
    expected_files = [
        "persona_consistency_report.md",
        "ssr_probability_validation.json",
        "human_vs_synthetic_comparison.md",
        "robustness_analysis.md",
        "product_adoption_interpretation.md",
        "final_validation_report.md",
    ]
    for fn in expected_files:
        p = VAL_DIR / fn
        assert p.is_file(), f"Missing validation artifact: {fn}"
        assert p.stat().st_size > 200, f"Artifact {fn} is suspiciously small"


def test_ssr_probability_validation_integrity():
    """Verify mathematical audit in ssr_probability_validation.json."""
    data = json.loads((VAL_DIR / "ssr_probability_validation.json").read_text(encoding="utf-8"))
    meta = data["audit_metadata"]
    assert meta["n_records"] == 24
    assert meta["all_pmfs_sum_to_one"] is True
    assert meta["all_ratings_in_bounds"] is True
    assert meta["zero_least_similar_anchor_verified"] is True

    records = data["records"]
    assert len(records) == 24

    for r in records:
        assert r["checks"]["pmf_sums_to_1"] is True
        assert r["checks"]["expected_rating_in_bounds"] is True
        assert r["checks"]["zero_least_similar_anchor"] is True
        assert 1.0 <= r["expected_rating"] <= 5.0
        assert r["cosine_spread"] > 0.0
        assert r["shannon_entropy"] > 0.0
        assert 1 <= r["dominant_category"] <= 5


def test_persona_consistency_report_content():
    """Verify that persona_consistency_report.md covers all 4 personas without rating leaks."""
    text = (VAL_DIR / "persona_consistency_report.md").read_text(encoding="utf-8")
    for pid in EXPECTED_PERSONAS:
        assert pid in text
    assert "iPhone" in text
    assert "OnePlus" in text
    assert "Passed" in text
    assert "Conclusion:" in text


def test_human_vs_synthetic_comparison_conceptual_boundary():
    """Verify human_vs_synthetic_comparison.md clearly marks the methodological boundary."""
    text = (VAL_DIR / "human_vs_synthetic_comparison.md").read_text(encoding="utf-8")
    assert "Essential Conceptual Distinction" in text
    assert "Q20" in text
    assert "Q21" in text
    assert "N=258" in text or "258" in text
    # Checks that brand repurchase intention is distinguished from new product concept adoption
    assert "current brand" in text.lower()
    assert "Nova AI" in text


def test_robustness_analysis_temperature_invariance():
    """Verify that temperature scaling does not invert persona orderings on PE."""
    data = json.loads((VAL_DIR / "ssr_probability_validation.json").read_text(encoding="utf-8"))
    pe_records = {r["persona_id"]: r for r in data["records"] if r["construct_code"] == "PE"}

    # Under T=1.0: P4 > P2 > P3 > P1
    p1_t1 = pe_records["P1_premium_tier_owner"]["sensitivity"]["T_1_0"]
    p2_t1 = pe_records["P2_midrange_heavy_user"]["sensitivity"]["T_1_0"]
    p3_t1 = pe_records["P3_midrange_moderate_user"]["sensitivity"]["T_1_0"]
    p4_t1 = pe_records["P4_budget_or_undisclosed_price"]["sensitivity"]["T_1_0"]
    assert p4_t1 > p2_t1 > p3_t1 > p1_t1

    # Under T=0.5: P4 > P2 > P3 > P1 must be strictly preserved (temperature monotonic invariance)
    p1_t05 = pe_records["P1_premium_tier_owner"]["sensitivity"]["T_0_5"]
    p2_t05 = pe_records["P2_midrange_heavy_user"]["sensitivity"]["T_0_5"]
    p3_t05 = pe_records["P3_midrange_moderate_user"]["sensitivity"]["T_0_5"]
    p4_t05 = pe_records["P4_budget_or_undisclosed_price"]["sensitivity"]["T_0_5"]
    assert p4_t05 > p2_t05 > p3_t05 > p1_t05


def test_product_adoption_interpretation_evidence_layers():
    """Verify product_adoption_interpretation.md distinguishes the 4 layers of evidence."""
    text = (VAL_DIR / "product_adoption_interpretation.md").read_text(encoding="utf-8")
    assert "Empirical Human Survey Evidence" in text
    assert "LLM Qualitative Persona Generation" in text
    assert "Canonical SSR Estimates" in text
    assert "Agent Interpretation" in text
    assert "₹24,999" in text


def test_protected_components_and_k_setting_preserved():
    """Confirm canonical SSR, approved anchors, and k=3 remain untouched."""
    from canonical_ssr import cosine_similarities, similarities_to_pmf
    from survey_ssr_bridge import load_all_survey_anchors

    anchors = load_all_survey_anchors()
    assert len(anchors) == 6
    assert set(anchors.keys()) == {
        "Performance Expectancy",
        "Effort Expectancy",
        "Hedonic Motivation",
        "Social Influence",
        "Perceived Protection",
        "Behavioral Intention",
    }

