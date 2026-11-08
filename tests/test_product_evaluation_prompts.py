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


def test_boat_airdopes_audio_prompts_and_budget_reasoning():
    """Verify boAt Airdopes 141 is correctly classified as Audio Device with audio construct instructions."""
    from scripts.product_evaluation_prompts import (
        build_persona_multiconstruct_prompt,
        parse_multiconstruct_response,
        sanitize_invented_facts,
        validate_and_detect_category,
    )

    boat = ProductConcept(
        name="boAt Airdopes 141 Bluetooth Earbuds",
        description="Wireless in-ear earbuds with 42 hours playback, ENx mic technology, 8mm drivers, and IPX4 water resistance.",
        price_inr=1299.0,
        features=[],
    )

    cat, noun, role, eco = validate_and_detect_category(boat.name, boat.description)
    assert cat == "Audio Device"
    assert noun == "wireless earbuds"
    assert "earbuds" in role

    # P1 Prompt: ₹1,299 must be completely affordable, NOT "too expensive" or "lack top-end build quality hesitate"
    p1_prompt = build_persona_multiconstruct_prompt(boat, "P1_premium_tier_owner")
    u_p1 = p1_prompt["user"]
    assert "₹1,299" in u_p1
    assert "completely affordable" in u_p1.lower()
    assert "NEVER claim you already own a device of this type" in u_p1
    assert "music" in u_p1.lower()
    assert "bluetooth" in u_p1.lower()

    # Construct instructions for audio must not mention cameras or alarms
    assert "Do NOT mention home security, surveillance cameras, alarms" in u_p1

    # P4 Prompt: ₹1,299 must be pocket-friendly for budget user, NOT auto-rejected
    p4_prompt = build_persona_multiconstruct_prompt(boat, "P4_budget_or_undisclosed_price")
    u_p4 = p4_prompt["user"]
    assert "pocket-friendly" in u_p4.lower()

    # Verify sanitizer removes invented ownership and awkward words
    raw_with_hallucination = "I already have a great set of headphones that work fine. I can listen to music without feeling fatiguing."
    cleaned = sanitize_invented_facts(raw_with_hallucination)
    assert "already have a great set of headphones" not in cleaned
    assert "fatiguing" not in cleaned
    assert "without feeling tired" in cleaned


def test_laptop_naturalness_and_category_affordability():
    """Verify ASUS Vivobook 15 evaluation prompts enforce natural English and laptop-specific affordability."""
    from scripts.product_evaluation_prompts import (
        build_persona_multiconstruct_prompt,
        parse_multiconstruct_response,
        sanitize_invented_facts,
        validate_and_detect_category,
        get_category_tier_info,
    )

    laptop = ProductConcept(
        name="ASUS Vivobook 15 Laptop",
        description="Everyday computing laptop with 15.6-inch Full HD display, Intel Core i5 processor, and SSD storage.",
        price_inr=55000.0,
        features=[],
    )

    cat, noun, role, eco = validate_and_detect_category(laptop.name, laptop.description)
    assert cat == "Laptop Computer"
    assert noun == "laptop"
    assert "computer" in role

    # Technology-specific tier: ₹55,000 is mainstream mid-range for laptops in India
    tier, tier_desc = get_category_tier_info(cat, noun, 55000.0)
    assert tier == "mainstream mid-range"
    assert "laptop price tier" in tier_desc

    # P3 prompt should evaluate laptop in mainstream mid-range terms, not smartphone budget
    p3_prompt = build_persona_multiconstruct_prompt(laptop, "P3_midrange_moderate_user")
    u_p3 = p3_prompt["user"]
    assert "₹55,000" in u_p3
    assert "mainstream mid-range laptop price tier" in u_p3
    # Check that pricing perspective does not tell P3 that price is prohibitive
    perspective_block = u_p3.split("--- YOUR TASK ---")[0].lower()
    assert "prohibitive" not in perspective_block
    assert "financial barrier" not in perspective_block
    # Check that smartphone benchmark was not injected into general tech persona block
    assert "historical primary device investment > ₹35,000" not in perspective_block

    # Check construct isolation instructions: PE, EE, HM, SI, PP must forbid price
    assert "CRITICAL: Do NOT mention price, cost, or budget here" in u_p3
    # HM for non-gaming laptop must forbid gaming
    assert "CRITICAL: Do NOT mention gaming unless this is a gaming device" in u_p3

    # Test natural language sanitization of robotic words
    test_text = "The processor is top-notch and provides adequate battery life for demanding tasks with a refined build."
    sanitized = sanitize_invented_facts(test_text, category="Laptop Computer")
    assert "top-notch" not in sanitized
    assert "adequate" not in sanitized
    assert "demanding tasks" not in sanitized
    assert "refined build" not in sanitized
    assert "great" in sanitized or "very good" in sanitized
    assert "decent" in sanitized

    # Test stripping accidental price complaints from PE
    pe_with_price = "The display is bright and clear for office tasks, but the high price is a significant barrier."
    pe_cleaned = sanitize_invented_facts(pe_with_price, category="Laptop Computer", construct="PE")
    assert "barrier" not in pe_cleaned
    assert "display is bright and clear for office tasks" in pe_cleaned

    # Test gaming replacement for non-gaming laptops in HM
    hm_with_gaming = "It could be enjoyable for gaming and entertainment on daily work routines."
    hm_cleaned = sanitize_invented_facts(hm_with_gaming, category="Laptop Computer", construct="HM")
    assert "gaming and entertainment" not in hm_cleaned
    assert "media and entertainment" in hm_cleaned



def test_smartphone_is_never_relabelled_as_another_category():
    """A phone stays a phone even if its description mentions laptops, TVs or watches."""
    from scripts.product_evaluation_prompts import build_persona_multiconstruct_prompt, validate_and_detect_category

    cases = [
        ("one me", "A mid-range smartphone designed for students.", "Technology Product"),
        ("Apex Ultra", "Phone with laptop-class processor and PC-level gaming.", "Smartphone"),
        ("Nova 5G", "Casts to your smart TV and syncs with your smartwatch.", "Smartphone"),
    ]
    for name, desc, cat in cases:
        assert validate_and_detect_category(name, desc, cat)[0] == "Smartphone"
        product = ProductConcept(name=name, description=desc, category=cat, price_inr=27000.0, features=[])
        user = build_persona_multiconstruct_prompt(product, "P3")["user"]
        assert "Always call it a smartphone" in user
        assert "NEVER assume it is a smartphone" not in user
        assert "Given the price" in user  # listed only as a banned opening

    assert validate_and_detect_category("Galaxy Tab S10 FE", "tablet with S Pen", "Tablet Computer")[0] == "Tablet Computer"
    assert validate_and_detect_category("ZenBook 14", "thin laptop that syncs with your phone", "Laptop")[0] == "Laptop Computer"
