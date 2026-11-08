"""Regression test for evaluation CSV DictWriter fieldnames mismatch.

Verifies that saving evaluation CSV outputs (summary and intermediate) succeeds without
raising ValueError: dict contains fields not in fieldnames when dual SSR comparison
fields ('canonical_expected_rating', 'improved_expected_rating', 'delta') are present.
"""

import csv
import io
import pytest


def test_summary_csv_writer_accepts_dual_ssr_fields():
    """Verify that summary CSV DictWriter correctly writes dual SSR fields without error."""
    fieldnames = [
        "persona_id", "product_name", "construct_code", "construct_name",
        "is_primary_outcome", "expected_rating", "canonical_expected_rating", "improved_expected_rating", "delta",
        "p1", "p2", "p3", "p4", "p5", "embedding_model"
    ]

    sample_row = {
        "persona_id": "P1_premium_tier_owner",
        "product_name": "Test Gadget Pro",
        "construct_code": "BI",
        "construct_name": "Behavioral Intention",
        "is_primary_outcome": True,
        "expected_rating": 3.45,
        "canonical_expected_rating": 3.45,
        "improved_expected_rating": 3.60,
        "delta": 0.15,
        "p1": 0.0,
        "p2": 0.1,
        "p3": 0.35,
        "p4": 0.45,
        "p5": 0.1,
        "embedding_model": "nomic-embed-text"
    }

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    # This must succeed without raising ValueError: dict contains fields not in fieldnames
    writer.writerows([sample_row])

    content = buf.getvalue()
    assert "canonical_expected_rating" in content
    assert "improved_expected_rating" in content
    assert "delta" in content
    assert "3.6" in content
    assert "0.15" in content


def test_intermediate_csv_writer_accepts_dual_ssr_fields():
    """Verify that intermediate CSV DictWriter correctly writes dual SSR fields without error."""
    fieldnames = [
        "persona_id", "product_name", "anchor_set_code", "anchor_set_name",
        "cos_sim_1", "cos_sim_2", "cos_sim_3", "cos_sim_4", "cos_sim_5", "min_similarity",
        "p1", "p2", "p3", "p4", "p5", "set_expected_rating",
        "canonical_expected_rating", "improved_expected_rating", "delta",
        "embedding_model"
    ]

    sample_inter_row = {
        "persona_id": "P2_midrange_heavy_user",
        "product_name": "Test Gadget Pro",
        "anchor_set_code": "PE",
        "anchor_set_name": "Performance Expectancy",
        "cos_sim_1": 0.312,
        "cos_sim_2": 0.456,
        "cos_sim_3": 0.678,
        "cos_sim_4": 0.812,
        "cos_sim_5": 0.543,
        "min_similarity": 0.312,
        "p1": 0.0,
        "p2": 0.15,
        "p3": 0.35,
        "p4": 0.45,
        "p5": 0.05,
        "set_expected_rating": 3.4,
        "canonical_expected_rating": 3.4,
        "improved_expected_rating": 3.55,
        "delta": 0.15,
        "embedding_model": "nomic-embed-text"
    }

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    # Must succeed without raising ValueError
    writer.writerows([sample_inter_row])

    content = buf.getvalue()
    assert "cos_sim_1" in content
    assert "canonical_expected_rating" in content
    assert "improved_expected_rating" in content
    assert "delta" in content
    assert "3.55" in content


def test_csv_writer_resilience_to_unexpected_extra_keys():
    """Verify that extrasaction='ignore' prevents crashes if any future experimental keys are added."""
    fieldnames = [
        "persona_id", "product_name", "construct_code", "construct_name",
        "is_primary_outcome", "expected_rating", "canonical_expected_rating", "improved_expected_rating", "delta",
        "p1", "p2", "p3", "p4", "p5", "embedding_model"
    ]

    sample_row_with_extra = {
        "persona_id": "P3_budget_user",
        "product_name": "Test Gadget Pro",
        "construct_code": "EE",
        "construct_name": "Effort Expectancy",
        "is_primary_outcome": False,
        "expected_rating": 3.1,
        "canonical_expected_rating": 3.1,
        "improved_expected_rating": 3.2,
        "delta": 0.1,
        "p1": 0.0,
        "p2": 0.2,
        "p3": 0.5,
        "p4": 0.3,
        "p5": 0.0,
        "embedding_model": "nomic-embed-text",
        "future_experimental_field": "some_value"  # unexpected extra key
    }

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    # With extrasaction='ignore', it does not raise ValueError
    writer.writerows([sample_row_with_extra])
    assert "future_experimental_field" not in buf.getvalue()

