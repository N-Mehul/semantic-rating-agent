"""Product-Evaluation Prompt Architecture & Reusable Template Generator.

Step 3 of the Digital Persona-Based Synthetic Survey Data Generation Pipeline.

This module provides reusable prompt templates and builders for evaluating new mobile
products with local LLMs (specifically Qwen 2.5 7B Instruct) role-playing empirical
digital personas derived from human survey data.

METHODOLOGICAL PROTECTIONS (Canonical SSR Compliance):
1. Textual Responses Only: The LLM generates qualitative descriptive text. It NEVER
   outputs Likert ratings, numerical scores (e.g., 4/5), probabilities, or adoption percentages.
2. The canonical SSR pipeline (cosine similarity against approved 5-anchor sets) converts
   the generated text into ratings where approved anchor sets exist.
3. Unsupported constructs (e.g., trust, perceived risk, perceived usefulness, facilitating
   conditions) are rejected rather than guessed or fabricated.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Sequence

ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# -----------------------------------------------------------------------------
# Construct Audit: Supported vs. Unavailable Constructs
# -----------------------------------------------------------------------------

SUPPORTED_CONSTRUCTS: Final[dict[str, dict[str, Any]]] = {
    "PE": {
        "construct_code": "PE",
        "canonical_name": "Performance Expectancy",
        "theoretical_framework": "UTAUT",
        "survey_evidence_type": "theoretical_codebook_and_qualitative_themes",
        "codebook_items": ["PE01", "PE02", "PE03", "PE04", "FT_PE01"],
        "fielded_survey_links": [
            "Q19 (What do you mainly use your smartphone for?)",
            "Q25 (How would you describe your smartphone's performance, speed, and display quality?)",
        ],
        "has_approved_ssr_anchors": True,
        "anchor_source": "smartphone_construct_anchor_statements.csv (PE)",
        "prompt_focus": (
            "Evaluate how effectively the product helps accomplish everyday tasks, productivity, "
            "and work/study activities given this persona's daily routines."
        ),
    },
    "BI": {
        "construct_code": "BI",
        "canonical_name": "Behavioral Intention",
        "theoretical_framework": "UTAUT",
        "survey_evidence_type": "direct_empirical_proxy_and_theoretical_codebook",
        "codebook_items": ["BI01", "BI02", "BI03", "BI04", "FT_BI01"],
        "fielded_survey_links": [
            "Q21 (How likely are you to buy the same smartphone brand again? [1-5 Likert])",
            "Q26 (Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?)",
            "Q16 (How often do you usually replace your smartphone?)",
            "Q17 (What was the main reason for upgrading to your current smartphone?)",
        ],
        "has_approved_ssr_anchors": True,
        "anchor_source": "smartphone_construct_anchor_statements.csv (BI)",
        "prompt_focus": (
            "State whether you would actually consider purchasing or switching to this product, "
            "and explain the primary drivers or barriers to adoption."
        ),
    },
    "EE": {
        "construct_code": "EE",
        "canonical_name": "Effort Expectancy",
        "theoretical_framework": "UTAUT",
        "survey_evidence_type": "theoretical_codebook_and_tech_comfort_proxy",
        "codebook_items": ["EE01", "EE02", "EE03", "EE04"],
        "fielded_survey_links": [
            "Q20 (How comfortable are you with using new technology? [1-5 Likert])",
            "Q23 (How would you describe your overall experience with your current smartphone?)",
        ],
        "has_approved_ssr_anchors": True,
        "anchor_source": "smartphone_construct_anchor_statements.csv (EE)",
        "prompt_focus": (
            "Evaluate how clear, understandable, and easy to learn the product appears, "
            "considering this persona's tech comfort level and operating system background."
        ),
    },
    "HM": {
        "construct_code": "HM",
        "canonical_name": "Hedonic Motivation",
        "theoretical_framework": "UTAUT",
        "survey_evidence_type": "theoretical_codebook_and_usage_patterns",
        "codebook_items": ["HM01", "HM02", "HM03", "HM04"],
        "fielded_survey_links": [
            "Q18 (Daily usage hours)",
            "Q19 (Usage for gaming, entertainment, social media)",
            "Q25 (Display quality and gaming performance)",
        ],
        "has_approved_ssr_anchors": True,
        "anchor_source": "smartphone_construct_anchor_statements.csv (HM)",
        "prompt_focus": (
            "Evaluate the enjoyment, fun, entertainment value, and media experience offered by the product."
        ),
    },
    "SI": {
        "construct_code": "SI",
        "canonical_name": "Social Influence",
        "theoretical_framework": "UTAUT",
        "survey_evidence_type": "theoretical_codebook_and_info_sources",
        "codebook_items": ["SI01", "SI02", "SI03", "SI04"],
        "fielded_survey_links": [
            "Q22 (What sources did you use when choosing your current smartphone? [friends, family, social media, tech reviews])",
        ],
        "has_approved_ssr_anchors": True,
        "anchor_source": "smartphone_construct_anchor_statements.csv (SI)",
        "prompt_focus": (
            "Evaluate the social perception, peer recommendations, and brand prestige of the product "
            "among people in this persona's peer circle."
        ),
    },
    "PP": {
        "construct_code": "PP",
        "canonical_name": "Perceived Protection",
        "theoretical_framework": "UTAUT Extension",
        "survey_evidence_type": "theoretical_codebook_only",
        "codebook_items": ["PP01", "PP02", "PP03", "PP04"],
        "fielded_survey_links": [
            "Working interpretation: protection of personal data and security on the device.",
        ],
        "has_approved_ssr_anchors": True,
        "anchor_source": "smartphone_construct_anchor_statements.csv (PP)",
        "prompt_focus": (
            "Evaluate data privacy, security controls, and personal information protection."
        ),
    },
    "PV": {
        "construct_code": "PV",
        "canonical_name": "Price Value",
        "theoretical_framework": "UTAUT2 (Open-ended only)",
        "survey_evidence_type": "fielded_text_and_spending_capacity",
        "codebook_items": ["FT_PV01"],
        "fielded_survey_links": [
            "Q10 (What is your approximate monthly personal spending capacity?)",
            "Q13 (Approximately how much did you pay for your current smartphone?)",
            "Q26 (Do you think your smartphone is worth the price you paid...)",
        ],
        "has_approved_ssr_anchors": False,
        "anchor_source": "None approved (blocked for SSR scoring; qualitative feedback only)",
        "prompt_focus": (
            "Evaluate whether the proposed price represents good value relative to this persona's "
            "spending capacity and current smartphone investment."
        ),
    },
}

UNAVAILABLE_CONSTRUCTS: Final[dict[str, str]] = {
    "trust": "Not measured: The survey contains no questions or anchors measuring institutional, vendor, or AI trust.",
    "perceived_risk": "Not measured: Financial, privacy, and physical risk are not operationalised in the instrument.",
    "perceived_usefulness": "TAM construct: The project adopts UTAUT Performance Expectancy (PE) instead.",
    "perceived_ease_of_use": "TAM construct: The project adopts UTAUT Effort Expectancy (EE) instead.",
    "facilitating_conditions": (
        "Explicitly excluded: The FC alias was deliberately removed in survey_ssr_bridge.py because "
        "no FC items exist in the codebook and no approved anchors exist in the CSV."
    ),
    "willingness_to_pay": (
        "Not measured as an econometric curve: The survey records only historical price paid brackets (Q13) "
        "and monthly personal spending capacity (Q10)."
    ),
    "brand_loyalty_affective": (
        "Not separately measured: Brand repurchase intention is captured as a single ordinal item (Q21), "
        "not a multi-item psychological loyalty battery."
    ),
}

# Regex to detect rating leaks in prompts and responses
RATING_LEAK_PATTERN: Final[re.Pattern] = re.compile(
    r"\b[1-5]\s*/\s*5\b|\bout of 5\b|\bout of five\b|\brating\b|\bscore\b|\bstars?\b|\bprobabilities\b|\bpercentage\b",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ProductFeature:
    """A feature of the proposed product with name and explanation."""

    name: str
    explanation: str
    category: str | None = None

    def format_line(self) -> str:
        cat_prefix = f"[{self.category}] " if self.category else ""
        return f"- {cat_prefix}{self.name}: {self.explanation}"


@dataclass(frozen=True)
class ProductConcept:
    """Full product concept specification for evaluation."""

    name: str
    description: str
    features: Sequence[ProductFeature]
    price_inr: float | None = None
    target_group: str | None = None
    brand_or_ecosystem: str | None = None

    def format_block(self) -> str:
        lines = [
            f"Product Name: {self.name}",
            f"Product Description: {self.description}",
        ]
        if self.price_inr is not None:
            lines.append(f"Proposed Price: ₹{self.price_inr:,.0f} INR")
        if self.target_group:
            lines.append(f"Target Consumer Group: {self.target_group}")
        if self.brand_or_ecosystem:
            lines.append(f"Brand / Ecosystem: {self.brand_or_ecosystem}")
        if self.features:
            lines.append("Key Features & Specifications:")
            for feat in self.features:
                lines.append(f"  {feat.format_line()}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Reusable Prompt System Templates
# -----------------------------------------------------------------------------

QWEN_SYSTEM_PROMPT_TEMPLATE: Final[str] = (
    "You role-play one real consumer from an empirical market-research panel reacting to a proposed mobile product concept. "
    "Respond in your own authentic voice from the perspective of your assigned persona profile.\n\n"
    "EVALUATION INSTRUCTIONS:\n"
    "1. Honestly evaluate the proposed product against your current smartphone, daily routine, and personal budget.\n"
    "2. State clearly:\n"
    "   - Which specific features attract you and why.\n"
    "   - Which features feel unappealing, gimmicky, or unnecessary for your needs.\n"
    "   - Reasons why you would consider adopting or purchasing this product.\n"
    "   - Reasons for hesitation, skepticism, or rejection.\n"
    "   - Any concerns or barriers (e.g. price vs. spending capacity, operating system transition, battery life, display).\n"
    "   - Concrete improvements that would make the product substantially more appealing to you.\n"
    "   - Whether the product feels genuinely relevant to your daily lifestyle.\n"
    "3. Maintain a natural, conversational, critical tone. Do not default to polite praise; give mixed, critical, or skeptical feedback if the product does not fit your profile.\n\n"
    "STRICT METHODOLOGICAL RESTRICTIONS (MANDATORY):\n"
    "- Output TEXT ONLY: 4 to 8 sentences in continuous prose.\n"
    "- NEVER provide numerical ratings, Likert scores, points, stars, percentages, or probabilities (e.g., do NOT say '4/5', '8/10', '70% chance').\n"
    "- NEVER state an adoption percentage or conversion score.\n"
    "- Do NOT mention being an AI, an LLM, or a language model."
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_supported_constructs() -> dict[str, dict[str, Any]]:
    """Return dictionary of supported constructs with mapping and anchor details."""
    return dict(SUPPORTED_CONSTRUCTS)


def get_unavailable_constructs() -> dict[str, str]:
    """Return dictionary of unavailable constructs with documented reasons."""
    return dict(UNAVAILABLE_CONSTRUCTS)


def validate_requested_constructs(construct_codes: Sequence[str]) -> list[str]:
    """Check that requested constructs are supported by the survey instrument.

    Returns a list of validated canonical codes (e.g., ['PE', 'BI']).
    Raises ValueError if an unsupported or invented construct is requested.
    """
    validated = []
    for c in construct_codes:
        normalized = c.strip().upper()
        if normalized in SUPPORTED_CONSTRUCTS:
            validated.append(normalized)
            continue
        # Check by canonical name
        found = False
        for code, meta in SUPPORTED_CONSTRUCTS.items():
            if meta["canonical_name"].lower() == c.strip().lower():
                validated.append(code)
                found = True
                break
        if not found:
            lower_c = c.strip().lower().replace(" ", "_")
            unavail_reason = UNAVAILABLE_CONSTRUCTS.get(lower_c)
            if unavail_reason:
                raise ValueError(f"Construct '{c}' is unsupported: {unavail_reason}")
            raise ValueError(
                f"Construct '{c}' is not supported by the survey instrument. "
                f"Available constructs are: {list(SUPPORTED_CONSTRUCTS.keys())}."
            )
    return validated


def load_persona_data(persona_id: str, root: Path = ROOT) -> dict[str, Any]:
    """Load profile card and spec data for the specified persona."""
    profiles_path = root / "data/personas/persona_profiles.json"
    if not profiles_path.exists():
        raise FileNotFoundError(f"Missing persona profiles: {profiles_path}")
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    profile = next((p for p in profiles if p["persona_id"] == persona_id or p["persona_id"].startswith(persona_id)), None)
    if not profile:
        raise KeyError(f"Persona '{persona_id}' not found in {profiles_path}")
    return profile


def build_persona_context(persona_id_or_dict: str | dict[str, Any], root: Path = ROOT) -> str:
    """Format an empirical persona into a structured prompt context block."""
    if isinstance(persona_id_or_dict, str):
        profile = load_persona_data(persona_id_or_dict, root)
    else:
        profile = persona_id_or_dict

    def top_cat(k: str, default: str = "") -> str:
        v = profile.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict) and "category" in v[0]:
            return v[0]["category"]
        if isinstance(v, str) and v:
            return v
        return default

    pid = profile.get("persona_id", "Unknown")
    rule = profile.get("defining_rule", "")
    med_age = profile.get("median_age", "20")
    gender = top_cat("gender", "Mixed")
    status = top_cat("current_status", "Student")
    area = top_cat("area_type", "Urban")
    os_name = top_cat("operating_system", "Android")
    brand = top_cat("brand", "Samsung")
    spending = top_cat("monthly_spending", "Moderate")
    daily_use = top_cat("daily_usage", "4-6 hours")
    comfort = profile.get("tech_comfort_mean_1to5", 4.0)

    # Experience themes if present
    themes = profile.get("top_experience_themes_from_Q23_Q25", [])
    if themes and isinstance(themes, list) and isinstance(themes[0], dict):
        themes_str = "; ".join(
            f"{t['theme']} ({t.get('mentions', t.get('n', ''))} mentions)" for t in themes[:3]
        )
    else:
        themes_str = "General satisfaction"

    lines = [
        f"CONSUMER PERSONA PROFILE: {pid}",
        f"- Segment definition: {rule}",
        f"- Demographics: ~{med_age} years old, {gender}, status: {status}, location: {area}",
        f"- Current smartphone: {brand} running {os_name}",
        f"- Usage intensity: {daily_use} daily screen time",
        f"- Spending capacity: {spending}",
        f"- Technology comfort: {comfort:.1f} on a 1-5 scale",
        f"- Key priorities in current phone: {themes_str}",
    ]
    return "\n".join(lines)


def build_product_evaluation_prompt(
    product: ProductConcept,
    persona_id_or_dict: str | dict[str, Any],
    constructs: Sequence[str] = ("PE", "BI"),
    root: Path = ROOT,
) -> dict[str, Any]:
    """Generate a complete, reusable product-evaluation prompt for Qwen 2.5 7B.

    Args:
        product: The product concept and feature specifications.
        persona_id_or_dict: Persona identifier ('P1_premium_tier_owner', etc.) or profile dict.
        constructs: Supported construct codes (default: Performance Expectancy and Behavioral Intention).
        root: Workspace root directory.

    Returns:
        A dict with 'system', 'user', and 'metadata' ready for local LLM inference.
    """
    valid_constructs = validate_requested_constructs(constructs)
    persona_block = build_persona_context(persona_id_or_dict, root)
    product_block = product.format_block()

    construct_lines = []
    for c in valid_constructs:
        meta = SUPPORTED_CONSTRUCTS[c]
        construct_lines.append(f"- {meta['canonical_name']} ({c}): {meta['prompt_focus']}")
    construct_section = "\n".join(construct_lines)

    user_message = (
        f"{persona_block}\n\n"
        f"--- PROPOSED PRODUCT CONCEPT ---\n"
        f"{product_block}\n\n"
        f"--- EVALUATION FOCUS AREAS ---\n"
        f"Please specifically address these adoption dimensions in your feedback:\n"
        f"{construct_section}\n\n"
        f"Write an authentic consumer review of this product in 4 to 8 sentences. "
        f"Describe what you like, what worries you, whether it fits your budget and lifestyle, "
        f"and whether you would genuinely consider replacing your current phone with it. "
        f"Remember: Use words only; do not include numerical scores, star ratings, or percentages."
    )

    pid = persona_id_or_dict if isinstance(persona_id_or_dict, str) else persona_id_or_dict.get("persona_id", "custom")
    ssr_ready_constructs = [c for c in valid_constructs if SUPPORTED_CONSTRUCTS[c]["has_approved_ssr_anchors"]]

    return {
        "persona_id": pid,
        "product_name": product.name,
        "system": QWEN_SYSTEM_PROMPT_TEMPLATE,
        "user": user_message,
        "metadata": {
            "selected_constructs": valid_constructs,
            "ssr_compatible_constructs": ssr_ready_constructs,
            "has_unanchored_constructs": len(valid_constructs) > len(ssr_ready_constructs),
            "profile_source": "data/personas/persona_profiles.json",
        },
    }


def validate_prompt_for_ssr(prompt: dict[str, Any]) -> list[str]:
    """Validate that the prompt contains no rating leakage and enforces text-only constraints."""
    issues = []
    system = prompt.get("system", "")
    user = prompt.get("user", "")

    if "TEXT ONLY" not in system and "words only" not in system.lower():
        issues.append("System prompt does not explicitly demand text-only responses.")
    if "NEVER provide numerical ratings" not in system:
        issues.append("System prompt does not explicitly forbid numerical ratings.")

    # Check that user prompt does not ask for Likert score
    if re.search(r"rate (this|the) \w+ on a \d", user, re.I):
        issues.append("User prompt inappropriately requests a numerical rating.")

    return issues


def build_single_construct_prompt(
    product: ProductConcept,
    persona_id_or_dict: str | dict[str, Any],
    construct_code: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Generate a construct-specific product evaluation prompt for Qwen 2.5 7B.

    Focuses exclusively on one construct (PE, EE, HM, SI, PP, BI, or PV) so that
    the generated natural language response directly evaluates that latent dimension.
    """
    code = construct_code.strip().upper()
    if code not in SUPPORTED_CONSTRUCTS:
        raise ValueError(f"Construct '{construct_code}' not supported. Supported: {list(SUPPORTED_CONSTRUCTS.keys())}")

    meta = SUPPORTED_CONSTRUCTS[code]
    persona_block = build_persona_context(persona_id_or_dict, root)
    product_block = product.format_block()

    user_message = (
        f"{persona_block}\n\n"
        f"--- PROPOSED PRODUCT CONCEPT ---\n"
        f"{product_block}\n\n"
        f"--- EVALUATION FOCUS: {meta['canonical_name']} ({code}) ---\n"
        f"{meta['prompt_focus']}\n\n"
        f"Write an authentic consumer review focusing specifically on this aspect in 3 to 6 sentences. "
        f"Be direct, critical, and candid from your persona's perspective. "
        f"MANDATORY: Words only. NEVER provide numerical scores, ratings (e.g. 4/5), or percentages."
    )

    pid = persona_id_or_dict if isinstance(persona_id_or_dict, str) else persona_id_or_dict.get("persona_id", "custom")
    return {
        "persona_id": pid,
        "product_name": product.name,
        "construct_code": code,
        "construct_name": meta["canonical_name"],
        "system": QWEN_SYSTEM_PROMPT_TEMPLATE,
        "user": user_message,
        "metadata": {
            "construct_code": code,
            "construct_name": meta["canonical_name"],
            "has_approved_ssr_anchors": meta["has_approved_ssr_anchors"],
        },
    }
