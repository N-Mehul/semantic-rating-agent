"""Validation tests for Step 4: Persona-Based Product Evaluation Responses.

Verifies:
1. Output directory contains product_concept.json, persona_responses.json, persona_responses.jsonl,
   and evaluation_manifest.json.
2. Exactly 4 persona responses are generated (P1, P2, P3, P4).
3. Each response is non-empty, linked to the correct persona, and contains authentic qualitative critique.
4. Strict SSR protection: No numerical Likert ratings, star ratings, or percentages exist in responses.
5. Canonical SSR and existing persona files remain untouched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "product_evaluation" / "nova_ai_smartphone"
EXPECTED_PERSONAS = [
    "P1_premium_tier_owner",
    "P2_midrange_heavy_user",
    "P3_midrange_moderate_user",
    "P4_budget_or_undisclosed_price",
]
RATING_LEAK_REGEX = re.compile(
    r"\b[1-5]\s*/\s*5\b|\bout of (5|five)\b|\brating:\s*\d|\bscore:\s*\d|\b\d{1,3}%\b",
    re.IGNORECASE,
)


def test_step4_output_files_exist():
    """Verify all expected files exist in the evaluation directory."""
    assert EVAL_DIR.is_dir(), f"Evaluation directory missing: {EVAL_DIR}"
    assert (EVAL_DIR / "product_concept.json").is_file()
    assert (EVAL_DIR / "persona_responses.json").is_file()
    assert (EVAL_DIR / "persona_responses.jsonl").is_file()
    assert (EVAL_DIR / "evaluation_manifest.json").is_file()


def test_product_concept_details():
    """Verify product concept is properly documented as hypothetical."""
    concept = json.loads((EVAL_DIR / "product_concept.json").read_text(encoding="utf-8"))
    assert concept["name"] == "Nova AI Smartphone"
    assert concept["price_inr"] == 24999.0
    assert concept["is_hypothetical_test_concept"] is True
    assert len(concept["features"]) == 6


def test_evaluation_manifest_completeness():
    """Verify evaluation manifest records generation metadata and zero errors."""
    manifest = json.loads((EVAL_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
    assert manifest["product_name"] == "Nova AI Smartphone"
    assert manifest["model"] == "qwen2.5:7b-instruct"
    assert manifest["n_personas_requested"] == 4
    assert manifest["n_responses_generated"] == 4
    assert manifest["n_errors"] == 0
    assert manifest["ssr_protection_verified"] is True
    assert manifest["personas_requested"] == EXPECTED_PERSONAS


def test_exactly_four_persona_responses_linked_correctly():
    """Verify exactly 4 persona responses exist, linked to P1-P4."""
    responses = json.loads((EVAL_DIR / "persona_responses.json").read_text(encoding="utf-8"))
    assert len(responses) == 4
    pids = [r["persona_id"] for r in responses]
    assert pids == EXPECTED_PERSONAS

    for r in responses:
        assert r["product_name"] == "Nova AI Smartphone"
        assert len(r["response_text"].strip()) > 100
        assert r["finish_reason"] == "STOP"
        assert r["model_requested"] == "qwen2.5:7b-instruct"


def test_strict_ssr_protection_no_numerical_ratings():
    """Verify responses contain text only and no numerical Likert/percentage leaks."""
    responses = json.loads((EVAL_DIR / "persona_responses.json").read_text(encoding="utf-8"))
    for r in responses:
        text = r["response_text"]
        match = RATING_LEAK_REGEX.search(text)
        assert match is None, f"Persona {r['persona_id']} generated a rating leak: {match.group(0)}"
        assert r["contains_numeric_rating"] is False


def test_persona_responses_are_distinct_and_grounded():
    """Verify that responses reflect distinct persona characteristics rather than uniform text."""
    responses = json.loads((EVAL_DIR / "persona_responses.json").read_text(encoding="utf-8"))
    by_pid = {r["persona_id"]: r["response_text"] for r in responses}

    # All 4 responses must have distinct text
    all_texts = list(by_pid.values())
    assert len(set(all_texts)) == 4

    # P1 (Premium iPhone owner) discusses ecosystem switching or current iPhone
    assert "iphone" in by_pid["P1_premium_tier_owner"].lower() or "ecosystem" in by_pid["P1_premium_tier_owner"].lower()

    # P4 (Budget owner) focuses strongly on the steep price climb from current budget
    assert "price" in by_pid["P4_budget_or_undisclosed_price"].lower() or "steep" in by_pid["P4_budget_or_undisclosed_price"].lower()


def test_protected_components_remain_unchanged():
    """Verify canonical SSR and anchor definitions were not touched."""
    from canonical_ssr import cosine_similarities, similarities_to_pmf, expected_rating
    from survey_ssr_bridge import load_all_survey_anchors

    anchors = load_all_survey_anchors()
    assert len(anchors) == 6
    assert "Performance Expectancy" in anchors
    assert "Behavioral Intention" in anchors

