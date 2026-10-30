"""Tests for Step 3: Product-Evaluation Prompt Design and Template Generation.

Verifies:
1. Construct audit integrity: Supported constructs are verified against survey codebook;
   unsupported/invented constructs (trust, perceived risk, facilitating conditions) are rejected.
2. Persona context generation: All four empirical personas (P1-P4) format correctly.
3. Feature and product concept integration.
4. Methodological protection: Prompts mandate text-only output and strictly prohibit
   numerical ratings, Likert scores, percentages, and probabilities.
5. Critical consumer feedback: Prompts instruct the model to evaluate features critically
   (attractions, unappealing features, hesitation, barriers, improvements).
6. Alignment with canonical SSR anchor sets.
"""
from __future__ import annotations

import pytest

from scripts.product_evaluation_prompts import (
    ProductConcept,
    ProductFeature,
    SUPPORTED_CONSTRUCTS,
    UNAVAILABLE_CONSTRUCTS,
    build_persona_context,
    build_product_evaluation_prompt,
    get_supported_constructs,
    get_unavailable_constructs,
    validate_prompt_for_ssr,
    validate_requested_constructs,
)
from survey_ssr_bridge import CONSTRUCT_CODE_MAP


@pytest.fixture
def sample_product() -> ProductConcept:
    return ProductConcept(
        name="NovaAir X1",
        description="A lightweight productivity smartphone with long battery life and an AI-enhanced camera.",
        price_inr=24999.0,
        target_group="College students and young working professionals",
        brand_or_ecosystem="Android ecosystem",
        features=[
            ProductFeature(
                name="6000 mAh Silicon-Carbon Battery",
                explanation="Offers up to 2 full days of moderate use with 65W fast charging.",
                category="Battery",
            ),
            ProductFeature(
                name="AI Portrait & Low-Light Camera",
                explanation="50 MP main sensor with automatic noise reduction for social media reels.",
                category="Camera",
            ),
            ProductFeature(
                name="120 Hz AMOLED Display",
                explanation="Bright, high-refresh display optimized for outdoor readability.",
                category="Display",
            ),
            ProductFeature(
                name="Clean Android OS with 3 Years Updates",
                explanation="Zero bloatware, fast app launches, and guaranteed security patches.",
                category="Software",
            ),
        ],
    )


def test_supported_constructs_registered():
    """Verify supported constructs match the survey instrument and codebook."""
    supported = get_supported_constructs()
    # Must contain the core constructs
    assert "PE" in supported
    assert "BI" in supported
    assert "EE" in supported
    assert "HM" in supported
    assert "SI" in supported
    assert "PP" in supported
    assert "PV" in supported

    # Check mapping fields
    for code, meta in supported.items():
        assert meta["construct_code"] == code
        assert len(meta["canonical_name"]) > 0
        assert len(meta["prompt_focus"]) > 0
        assert isinstance(meta["has_approved_ssr_anchors"], bool)


def test_ssr_anchors_alignment_with_survey_ssr_bridge():
    """Verify that constructs flagged as SSR-ready align with approved anchors in survey_ssr_bridge."""
    supported = get_supported_constructs()
    anchored_codes = {k for k, v in supported.items() if v["has_approved_ssr_anchors"]}
    assert anchored_codes == set(CONSTRUCT_CODE_MAP.keys())
    # PV is in supported for open-ended prompt evaluation, but not approved for SSR scoring
    assert supported["PV"]["has_approved_ssr_anchors"] is False


@pytest.mark.parametrize(
    "unsupported_name",
    [
        "trust",
        "perceived_risk",
        "perceived_usefulness",
        "perceived_ease_of_use",
        "facilitating_conditions",
        "willingness_to_pay",
    ],
)
def test_unavailable_constructs_rejected(unsupported_name: str):
    """Confirm that invented or unmeasured constructs are explicitly rejected with reasons."""
    with pytest.raises(ValueError) as excinfo:
        validate_requested_constructs([unsupported_name])
    err_msg = str(excinfo.value)
    assert "unsupported" in err_msg.lower() or "not supported" in err_msg.lower()


def test_persona_context_generation_for_all_four_personas():
    """Verify that all four empirical personas generate structured context with demographics and behaviors."""
    persona_ids = [
        "P1_premium_tier_owner",
        "P2_midrange_heavy_user",
        "P3_midrange_moderate_user",
        "P4_budget_or_undisclosed_price",
    ]
    for pid in persona_ids:
        context = build_persona_context(pid)
        assert pid in context
        assert "Segment definition:" in context
        assert "Demographics:" in context
        assert "Current smartphone:" in context
        assert "Usage intensity:" in context
        assert "Spending capacity:" in context
        assert "Technology comfort:" in context


def test_product_evaluation_prompt_structure(sample_product: ProductConcept):
    """Verify prompt creation with product features, persona, and focus areas."""
    prompt = build_product_evaluation_prompt(
        product=sample_product,
        persona_id_or_dict="P2_midrange_heavy_user",
        constructs=["PE", "BI"],
    )

    assert prompt["persona_id"] == "P2_midrange_heavy_user"
    assert prompt["product_name"] == "NovaAir X1"
    assert "system" in prompt
    assert "user" in prompt

    user_text = prompt["user"]
    # Checks persona profile presence
    assert "CONSUMER PERSONA PROFILE: P2_midrange_heavy_user" in user_text
    # Checks product concept details
    assert "NovaAir X1" in user_text
    assert "Proposed Price: ₹24,999 INR" in user_text
    assert "6000 mAh Silicon-Carbon Battery" in user_text
    assert "AI Portrait & Low-Light Camera" in user_text
    assert "Clean Android OS" in user_text
    # Checks constructs
    assert "Performance Expectancy (PE)" in user_text
    assert "Behavioral Intention (BI)" in user_text


def test_prompt_mandates_text_only_and_prohibits_ratings(sample_product: ProductConcept):
    """Critical test: The prompt MUST enforce text-only and forbid Likert/numbers to protect SSR."""
    prompt = build_product_evaluation_prompt(
        product=sample_product,
        persona_id_or_dict="P1_premium_tier_owner",
        constructs=["PE", "BI"],
    )

    sys_text = prompt["system"]
    assert "TEXT ONLY" in sys_text
    assert "NEVER provide numerical ratings" in sys_text
    assert "Likert scores" in sys_text
    assert "probabilities" in sys_text
    assert "percentages" in sys_text
    assert "do not mention being an ai" in sys_text.lower()

    # Run the SSR prompt validator
    issues = validate_prompt_for_ssr(prompt)
    assert issues == []


def test_prompt_requires_meaningful_critical_feedback(sample_product: ProductConcept):
    """Verify prompt instructs Qwen to provide critical, realistic consumer feedback."""
    prompt = build_product_evaluation_prompt(
        product=sample_product,
        persona_id_or_dict="P3_midrange_moderate_user",
        constructs=["PE", "BI"],
    )

    sys_text = prompt["system"]
    # Must instruct on features that attract
    assert "Which specific features attract you" in sys_text
    # Must instruct on unappealing features
    assert "unappealing, gimmicky, or unnecessary" in sys_text
    # Must ask for adoption reasons
    assert "adopt" in sys_text.lower() or "purchas" in sys_text.lower()
    # Must ask for hesitation or rejection
    assert "hesitation" in sys_text.lower() or "rejection" in sys_text.lower()
    # Must ask for barriers
    assert "concerns or barriers" in sys_text.lower()
    # Must ask for concrete improvements
    assert "improvements" in sys_text.lower()
    # Must ask for relevance
    assert "relevance" in sys_text.lower() or "relevant" in sys_text.lower()
    # Must forbid defaulting to praise
    assert "Do not default to polite praise" in sys_text


def test_metadata_tracks_ssr_compatibility(sample_product: ProductConcept):
    """Verify that metadata accurately distinguishes SSR-anchored vs unanchored constructs."""
    # When requesting an unanchored construct like PV (Price Value)
    prompt = build_product_evaluation_prompt(
        product=sample_product,
        persona_id_or_dict="P4_budget_or_undisclosed_price",
        constructs=["PE", "BI", "PV"],
    )
    meta = prompt["metadata"]
    assert meta["selected_constructs"] == ["PE", "BI", "PV"]
    assert meta["ssr_compatible_constructs"] == ["PE", "BI"]
    assert meta["has_unanchored_constructs"] is True
