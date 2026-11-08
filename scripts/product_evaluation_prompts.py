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
    category: str | None = None

    def format_block(self) -> str:
        lines = [
            f"Product Name: {self.name}",
        ]
        if self.category:
            lines.append(f"Product Category: {self.category}")
        lines.append(f"Product Description: {self.description}")
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


def validate_and_detect_category(name: str, description: str = "", input_category: str = "") -> tuple[str, str, str, str]:
    """Identify the product category, giving the user-entered category and name priority over the description.

    Descriptions often mention other devices ("laptop-class chip", "cast to your TV", "syncs with
    your smartwatch"), so they are only used when the category and name do not identify the product.
    """
    cat = (input_category or "").strip()
    if cat and cat.lower() not in ("technology product", "technology", "product", "gadget", "other"):
        detected = _detect_category(name, cat, cat)
        if detected[3] != "Technology Platform":
            return detected
    phone_from_name = _detect_category(name, "", "")
    if phone_from_name[0] == "Smartphone":
        return phone_from_name
    detected = _detect_category(name, description, input_category)
    if detected[3] == "Technology Platform" and re.search(
        r"\b(smartphone|smartphones|smart\s*phone|mobile\s*phone|android\s*phone|5g\s*phone|phone)\b", (description or "").lower()
    ):
        return _detect_category(name, "", "Smartphone")
    return detected


def _detect_category(name: str, description: str = "", input_category: str = "") -> tuple[str, str, str, str]:
    """Identify, validate, and return the correct product category, noun, usage role, and ecosystem.
    
    Category detection prioritizes the product name and explicit domain terminology,
    strictly preventing false categorization (e.g. security cameras having 'two-way audio'
    are never classified as 'Audio Device'). Never relies on price.
    
    Returns:
        (category, product_noun, primary_usage_role, brand_or_ecosystem)
    """
    name_clean = (name or "").strip()
    name_lower = name_clean.lower()
    desc_clean = (description or "").strip()
    desc_lower = desc_clean.lower()
    full_text = f"{name_lower} {desc_lower}"
    input_cat = (input_category or "").strip()
    input_cat_lower = input_cat.lower()

    # 1. Smart Glasses / Wearable AI / AR / VR
    is_glasses = bool(re.search(r"\b(smart\s*glasses|glasses|eyewear|spectacles|ar\s*glasses|smart\s*lens|vr\s*headset|vision\s*glasses|meta\s*glasses)\b", full_text)) or any(k in input_cat_lower for k in ["glasses", "eyewear"])
    if is_glasses:
        return (
            "Smart Glasses / Wearable AI",
            "smart glasses",
            "daily smart glasses",
            "Wearable AI Platform"
        )

    # 2. Smart Home Security / Surveillance Camera
    is_camera_in_name = bool(re.search(r"\b(security\s*camera|surveillance\s*camera|cctv|ip\s*camera|smart\s*camera|home\s*camera|indoor\s*camera|outdoor\s*camera|webcam|dashcam|doorbell|tapo|camera|cameras)\b", name_lower))
    is_camera_in_desc = bool(re.search(r"\b(security\s*camera|surveillance\s*camera|cctv|ip\s*camera|smart\s*camera|home\s*camera|indoor\s*camera|outdoor\s*camera|video\s*monitoring|home\s*surveillance|pan/tilt|night\s*vision)\b", desc_lower))
    is_not_phone = not any(k in name_lower for k in ["smartphone", "phone", "mobile", "iphone", "pixel", "galaxy s"])
    if (is_camera_in_name or (is_camera_in_desc and "camera" in full_text)) and is_not_phone:
        return (
            "Smart Home Security / Surveillance Camera",
            "security camera",
            "primary security camera",
            "Smart Home Platform (Tapo / Wi-Fi)" if "tapo" in full_text else "Smart Home Platform"
        )

    # 3. Wearable Device (Smartwatch / Fitness Tracker / Smart Ring)
    is_wearable = bool(re.search(r"\b(smartwatch|smartwatches|smart\s*watch|fitness\s*band|fitness\s*tracker|smartband|wristband|wrist\s*wear|smart\s*ring|galaxy\s*watch|apple\s*watch)\b", full_text)) or (re.search(r"\b(watch|wearable|ring)\b", name_lower) and "tv" not in name_lower) or any(k in input_cat_lower for k in ["wearable", "smartwatch", "ring", "fitness band"])
    if is_wearable:
        w_noun = "smart ring" if "ring" in full_text else "smartwatch"
        return (
            "Wearable Device",
            w_noun,
            f"daily {w_noun}",
            "Wearable Platform"
        )

    # 4. Gaming Console
    if re.search(r"\b(playstation|ps5|ps4|ps3|xbox|nintendo|switch|gaming\s*console|handheld\s*console|console|consoles)\b", full_text) or "console" in input_cat_lower:
        eco = "PlayStation Platform" if any(k in full_text for k in ["playstation", "ps5", "ps4", "sony"]) else (
            "Xbox Platform" if "xbox" in full_text else (
                "Nintendo Platform" if "nintendo" in full_text or "switch" in full_text else "Console Gaming Platform"
            )
        )
        return (
            "Gaming Console",
            "gaming console",
            "primary gaming console",
            eco
        )

    # 5. Television / Smart TV
    if re.search(r"\b(smart\s*tv|television|televisions|oled\s*tv|qled\s*tv|4k\s*tv|android\s*tv|google\s*tv|bravia|home\s*theater)\b", full_text) or re.search(r"\b(tv|tvs)\b", name_lower) or "tv" in input_cat_lower:
        return (
            "Television",
            "smart TV",
            "primary television",
            "Smart TV Platform"
        )

    # 6. Laptop Computer
    if re.search(r"\b(laptop|laptops|notebook|notebooks|macbook|ultrabook|chromebook|thinkpad|desktop|workstation|pc|computer|computers)\b", full_text) and "tablet" not in input_cat_lower or "laptop" in input_cat_lower:
        return (
            "Laptop Computer",
            "laptop",
            "primary computer",
            "PC Platform (Windows / macOS / Linux)"
        )

    # 7. Tablet Computer
    if re.search(r"\b(tablet|tablets|ipad|ipads|tab|kindle|ereader|e-reader)\b", full_text) and not any(k in name_lower for k in ["phone", "smartphone"]):
        return (
            "Tablet Computer",
            "tablet",
            "primary tablet",
            "Tablet Platform"
        )

    # 8. Audio Device (Headphones / Earbuds / Speakers)
    # Strictly checks audio hardware keywords in the product name or primary description;
    # NEVER triggers on incidental "two-way audio", "sound", or "audio" in a camera/TV/laptop!
    is_audio_in_name = bool(re.search(r"\b(headphone|headphones|earphone|earphones|earbud|earbuds|airpod|airpods|airdopes|tws|soundbar|speaker|speakers|headset|headsets|audio\s*device)\b", name_lower))
    is_audio_in_desc = bool(re.search(r"\b(wireless\s*earbuds|bluetooth\s*earbuds|in-ear\s*monitors|noise-cancelling\s*headphones|over-ear\s*headphones|bluetooth\s*speaker)\b", desc_lower))
    if is_audio_in_name or is_audio_in_desc or "audio" in input_cat_lower:
        if bool(re.search(r"\b(earbud|earbuds|airpod|airpods|airdopes|tws|in-ear)\b", full_text)):
            a_noun = "wireless earbuds"
            a_role = "daily earbuds"
        elif bool(re.search(r"\b(speaker|speakers|soundbar)\b", full_text)):
            a_noun = "Bluetooth speaker"
            a_role = "home audio speaker"
        else:
            a_noun = "headphones"
            a_role = "daily headphones"
        return (
            "Audio Device",
            a_noun,
            a_role,
            "Bluetooth / Smartphone Audio"
        )

    # 9. Smart Home / Robotics / Cleaning Appliance
    if re.search(r"\b(robot|robotics|robotic|cleaner|vacuum|mop|mop\s*robot|cleaning\s*robot|ecobot|smart\s*home|automation|smart\s*plug|smart\s*bulb)\b", full_text) or any(k in input_cat_lower for k in ["robot", "vacuum", "cleaning"]):
        is_cleaner = any(k in full_text for k in ["clean", "vacuum", "mop", "sweep", "floor", "robot", "ecobot"])
        c_noun = "cleaning robot" if is_cleaner else "smart home device"
        c_role = "daily cleaning robot" if is_cleaner else "daily smart home device"
        return (
            "Smart Home / Robotics" if is_cleaner else "Smart Home Device",
            c_noun,
            c_role,
            "Smart Home Ecosystem"
        )

    # 10. Home Appliance
    if re.search(r"\b(refrigerator|fridge|washing\s*machine|air\s*conditioner|ac|microwave|purifier|dishwasher|geyser|water\s*heater|appliance|appliances)\b", full_text):
        return (
            "Home Appliance",
            "appliance",
            "home appliance",
            "Home Appliance Ecosystem"
        )

    # 11. Software / AI Tool / Application
    if re.search(r"\b(software|ai\s*tool|saas|application|apps|mobile\s*app|web\s*app|cloud\s*service|model|extension|assistant\s*tool)\b", full_text):
        return (
            "Software / AI Tool",
            "software tool",
            "primary software tool",
            "Cloud / Web Platform"
        )

    # 12. Smartphone / Mobile Phone (STRICT CHECK: MUST be in name or explicit phone category)
    # NEVER triggers merely because a description mentions syncing with or integrating with a smartphone!
    is_phone_in_name = bool(re.search(r"\b(smartphone|smartphones|smart\s*phone|mobile\s*phone|handset|iphone|android\s*phone|cellphone)\b", name_lower)) or (
        bool(re.search(r"\b(phone|cellphone)\b", name_lower)) and not any(k in name_lower for k in ["headphone", "earphone", "glasses", "case", "holder", "charger", "robot", "vacuum", "ring", "watch", "camera"])
    )
    is_phone_in_cat = bool(re.search(r"\b(smartphone|mobile\s*phone|phone)\b", input_cat_lower)) and not any(k in input_cat_lower for k in ["headphone", "earphone", "speakerphone"])
    if is_phone_in_name or is_phone_in_cat:
        eco = "Android" if "android" in full_text else ("iOS" if "iphone" in full_text or "apple" in full_text else "Android")
        return (
            "Smartphone",
            "smartphone",
            "primary phone",
            eco
        )

    # 13. Preserved input category if valid and non-generic
    if input_cat and input_cat_lower not in ("technology product", "technology", "product", "gadget", "other"):
        clean_noun = input_cat_lower
        if not any(k in clean_noun for k in ["device", "product", "system", "tool", "appliance", "robot", "camera", "phone", "glasses"]):
            clean_noun = f"{clean_noun} device"
        return (
            input_cat,
            clean_noun,
            f"daily {clean_noun}",
            "Technology Platform"
        )

    # 14. Fallback
    return (
        "Technology Product",
        "technology product",
        "daily device",
        "Technology Platform"
    )


def detect_product_noun(product: ProductConcept) -> tuple[str, str]:
    """Detect product noun and primary usage phrase (e.g., ('smartwatch', 'daily driver')).
    
    Identifies the product correctly from its name, description, and category,
    without relying on price.
    
    Returns:
        (product_noun, primary_usage_role)
    """
    _, noun, role, _ = validate_and_detect_category(
        name=product.name,
        description=product.description,
        input_category=getattr(product, "category", "") or ""
    )
    return (noun, role)


# Audience, persona, marketing and generic-suitability wording is never a product attribute.
_AUDIENCE_OR_GENERIC = re.compile(
    r"\b(students?|professionals?|consumers?|users?|enthusiasts?|customers?|buyers?|people|persons?|famil(?:y|ies)|teens?|kids?|"
    r"seniors?|households?|everyone|anyone|budget-conscious|tech-savvy|suitab\w*|suitable|designed|ideal|target\w*|"
    r"overall|needs?|balance|evaluat\w*|may|might|those|who|everyday|productivity|entertainment|communication|"
    r"performance|smartphones?|phones?|devices?|products?|gadgets?|mid-range|flagship|entry-level|offers?|provides?)\b",
    re.IGNORECASE)
# A clause counts as a product attribute only if it names a concrete part, capability, specification or number.
_PRODUCT_ATTRIBUTE = re.compile(
    r"(\d|display|screen|camera|battery|charg\w*|processor|chip|cpu|gpu|\bram\b|storage|memory|speaker|audio|sound|sensor|"
    r"lens|build|body|water|dust|resist\w*|\b5g\b|\b4g\b|wi-?fi|bluetooth|nfc|fingerprint|unlock|android|\bios\b|"
    r"\bos\b|update|gaming|touch|keyboard|port|\bsim\b|microphone|connect\w*|\bapps?\b|assistant|tracking|"
    r"\bgb\b|\btb\b|\bmp\b|\bhz\b|inch|mah|oled|amoled|lcd|refresh)",
    re.IGNORECASE)


def _is_product_attribute_clause(clause: str) -> bool:
    """True only for clauses that name a concrete product attribute (no audience, persona or generic wording)."""
    c = clause.strip()
    return bool(_PRODUCT_ATTRIBUTE.search(c)) and not _AUDIENCE_OR_GENERIC.search(c)


def extract_features_from_concept(name: str, category: str, description: str) -> list[ProductFeature]:
    """Derive product features strictly from product name, category, and user-provided description.
    
    Never invents AI features if none are mentioned in the description.
    Strictly prevents cross-category feature leakage (e.g., security cameras will never receive audio headphone features).
    """
    text = f"{name} {category} {description}".lower()
    features: list[ProductFeature] = []

    # 1. Smart Home Security / Surveillance Camera
    if any(k in category.lower() for k in ["security", "surveillance", "camera"]) or any(k in name.lower() for k in ["camera", "cctv", "tapo", "doorbell"]):
        if any(w in text for w in ["2k", "4mp", "1080p", "hd", "resolution", "video", "recording", "clarity", "sensor"]):
            res_label = "2K High-Definition Video" if ("2k" in text or "4mp" in text) else "High-Definition Video Recording"
            features.append(ProductFeature(res_label, "Crisp video recording and live stream clarity for home monitoring."))
        if any(w in text for w in ["pan", "tilt", "360", "motion tracking", "rotation", "tracking"]):
            features.append(ProductFeature("360° Pan & Tilt Motion Tracking", "Full rotational room coverage with automated subject tracking."))
        if any(w in text for w in ["night vision", "infrared", "low light", "dark"]):
            features.append(ProductFeature("Infrared Night Vision", "Clear low-light surveillance and visibility in complete darkness."))
        if any(w in text for w in ["motion detection", "detection", "alert", "smart detection", "ai detection", "motion"]):
            features.append(ProductFeature("Smart Motion Detection & Alerts", "Instant activity notifications and human/motion detection."))
        if any(w in text for w in ["two-way audio", "two way audio", "speaker", "mic", "talk"]):
            features.append(ProductFeature("Two-Way Audio Communication", "Built-in microphone and speaker for real-time remote communication."))
        if any(w in text for w in ["cloud", "sd card", "storage", "local storage", "microsd"]):
            features.append(ProductFeature("Flexible Storage Options", "Support for local microSD card storage and cloud backup."))
        if any(w in text for w in ["privacy", "shutter", "encryption", "secure", "protection"]):
            features.append(ProductFeature("Privacy Protection & Encryption", "Physical privacy mode and encrypted transmission for personal security."))

    # 2. Gaming Console
    elif any(k in category.lower() for k in ["gaming", "console"]) or re.search(r"\b(playstation|ps5|ps4|xbox|nintendo)\b", text):
        if any(w in text for w in ["graphic", "4k", "visual", "ray tracing", "hdr", "fps", "frame rate", "display"]):
            features.append(ProductFeature("High-Quality Graphics", "High-fidelity visuals, ray tracing, and 4K output for realistic graphics."))
        if any(w in text for w in ["load", "ssd", "fast", "speed", "storage", "quick"]):
            features.append(ProductFeature("Fast Loading", "High-speed SSD architecture for quick game boot times and minimal loading pauses."))
        if any(w in text for w in ["gameplay", "immersive", "dualsense", "controller", "haptic", "trigger", "feedback"]):
            features.append(ProductFeature("Immersive Gameplay", "Responsive controller feedback and smooth frame rates for deep gameplay immersion."))
        if any(w in text for w in ["multiplayer", "online", "co-op", "friends", "network"]):
            features.append(ProductFeature("Multiplayer Gaming", "Online multiplayer support and co-op gaming with friends and community."))
        if any(w in text for w in ["audio", "sound", "3d", "spatial"]):
            features.append(ProductFeature("3D Spatial Audio", "Immersive directional audio engineering for realistic game environments."))

    # 3. Wearable Device
    elif any(k in category.lower() for k in ["wearable", "smartwatch", "watch"]) or any(k in text for k in ["smartwatch", "smart watch", "fitness band", "tracker"]):
        if any(w in text for w in ["health", "heart", "spo2", "fitness", "tracking", "sleep", "step", "sensor"]):
            features.append(ProductFeature("Health & Fitness Tracking", "Comprehensive biometric monitoring including heart rate, sleep, and activity tracking."))
        if any(w in text for w in ["display", "screen", "amoled", "retina", "bright"]):
            features.append(ProductFeature("High-Clarity Display", "Vibrant, high-contrast display designed for clear indoor and outdoor visibility."))
        if any(w in text for w in ["battery", "charging", "days", "long-lasting"]):
            features.append(ProductFeature("Long Battery Life", "Optimized power management providing extended usage on a single charge."))
        if any(w in text for w in ["water", "resistant", "swim", "ip68", "durable", "tough"]):
            features.append(ProductFeature("Durable & Water Resistant", "Rugged build quality certified for water resistance and daily active wear."))
        if any(w in text for w in ["call", "notification", "message", "sync", "bluetooth"]):
            features.append(ProductFeature("Smart Notifications", "Real-time sync for phone calls, messages, and app notifications."))

    # 4. Laptop Computer
    elif any(k in category.lower() for k in ["laptop", "computer"]) or re.search(r"\b(laptops?|notebooks?|macbooks?|pcs?|desktops?)\b", text):
        if any(w in text for w in ["processor", "cpu", "chip", "performance", "speed", "ram", "multitask"]):
            features.append(ProductFeature("Processing Performance", "High-speed multi-core processing for demanding productivity and multitasking."))
        if any(w in text for w in ["display", "screen", "resolution", "oled", "retina", "anti-glare"]):
            features.append(ProductFeature("High-Resolution Display", "Crisp visual workspace with accurate color reproduction."))
        if any(w in text for w in ["battery", "all-day", "hours", "charging"]):
            features.append(ProductFeature("Extended Battery Life", "Long-lasting battery life for sustained mobile productivity."))
        if any(w in text for w in ["keyboard", "trackpad", "type", "comfort", "build"]):
            features.append(ProductFeature("Ergonomic Keyboard & Trackpad", "Comfortable typing experience and responsive precision trackpad."))
        if any(w in text for w in ["port", "thunderbolt", "usb", "hdmi", "connect"]):
            features.append(ProductFeature("Versatile Connectivity", "Comprehensive ports for external displays, accessories, and fast transfers."))

    # 5. Television
    # Whole words only: "oled" inside "Super AMOLED" must not turn a phone into a TV.
    elif re.search(r"\b(television|tv)\b", category.lower()) or re.search(r"\b(televisions?|smart tv|oled tv|qled tv|qled)\b", text):
        if any(w in text for w in ["display", "picture", "4k", "8k", "uhd", "hdr", "color"]):
            features.append(ProductFeature("Vibrant 4K Display", "Ultra-high definition panel with HDR support for vivid contrast and colors."))
        if any(w in text for w in ["smart", "streaming", "apps", "ott", "netflix", "youtube", "os"]):
            features.append(ProductFeature("Smart Streaming Hub", "Integrated operating system with popular streaming apps and easy navigation."))
        if any(w in text for w in ["sound", "audio", "speaker", "dolby", "atmos"]):
            features.append(ProductFeature("Dolby Audio System", "Integrated high-clarity sound system with rich dialogue and bass."))

    # 6. Audio Device (ONLY if category is Audio Device or name explicitly indicates headphones/earbuds)
    elif "audio" in category.lower() or any(k in name.lower() for k in ["headphone", "earphone", "earbud", "airpod", "airdopes", "tws", "soundbar", "speaker"]):
        if any(w in text for w in ["noise cancel", "anc", "ambient", "isolation"]):
            features.append(ProductFeature("Active Noise Cancellation", "Acoustic isolation that cuts out ambient background noise."))
        if any(w in text for w in ["sound", "bass", "driver", "fidelity", "acoustic", "audio", "clarity"]):
            features.append(ProductFeature("Clear Balanced Audio", "Crisp sound quality with punchy bass for songs, videos, and daily calls."))
        if any(w in text for w in ["battery", "playback", "hours", "charge", "case"]):
            features.append(ProductFeature("Extended Playback Time", "Long-lasting battery life with portable charging case for all-day use."))
        if any(w in text for w in ["comfort", "fit", "cushion", "lightweight", "ear"]):
            features.append(ProductFeature("Comfortable Lightweight Fit", "Snug and lightweight design for easy listening without ear discomfort."))
        if any(w in text for w in ["bluetooth", "wireless", "pair", "connectivity", "range"]):
            features.append(ProductFeature("Seamless Bluetooth Pairing", "Fast, stable wireless connection with smartphones and tablets."))
        if any(w in text for w in ["call", "mic", "microphone", "enx", "voice"]):
            features.append(ProductFeature("Clear Voice Calls", "Built-in microphone for hands-free calling and clear voice pickup."))

    # 7. Smart Glasses / Wearable AI / AR Eyewear
    elif any(k in category.lower() for k in ["glasses", "eyewear", "vision", "headset"]) or any(k in name.lower() for k in ["glasses", "eyewear", "spectacles", "lens"]):
        if any(w in text for w in ["voice", "assistant", "command", "ai", "speech"]):
            features.append(ProductFeature("Hands-Free Voice AI Assistant", "Instant voice-activated smart assistant for daily inquiries and hands-free control."))
        if any(w in text for w in ["audio", "speaker", "sound", "playback", "music", "mic"]):
            features.append(ProductFeature("Open-Ear Directional Audio", "Integrated directional speakers and dual-mic array for private listening and clear voice calls."))
        if any(w in text for w in ["navigation", "turn-by-turn", "gps", "direction", "maps"]):
            features.append(ProductFeature("Real-Time Audio Navigation", "Audio-guided turn-by-turn directions directly into the ear."))
        if any(w in text for w in ["recogni", "object", "text", "camera", "visual", "vision"]):
            features.append(ProductFeature("Visual AI Recognition", "Real-time optical object and text translation and recognition."))
        if any(w in text for w in ["frame", "lightweight", "comfort", "wear", "fit", "eyewear"]):
            features.append(ProductFeature("Lightweight Ergonomic Frame", "Stylish, featherlight frame engineered for all-day facial comfort."))
        if any(w in text for w in ["privacy", "indicator", "shutter", "control", "sensor"]):
            features.append(ProductFeature("Physical Privacy Controls", "LED capture indicator and hardware microphone mute for user privacy protection."))
        if any(w in text for w in ["battery", "charging", "case", "hours", "recharge"]):
            features.append(ProductFeature("All-Day Battery & Quick Charge", "Extended battery life with portable charging support for continuous daily use."))

    # 8. Smart Home / Robotics / Cleaning Appliance
    elif any(k in category.lower() for k in ["robot", "vacuum", "cleaning", "smart home", "appliance"]) or any(k in text for k in ["robot", "vacuum", "cleaner", "mop", "ecobot", "purifier"]):
        if any(w in text for w in ["lidar", "navigation", "mapping", "slam", "sensor", "laser"]):
            features.append(ProductFeature("Intelligent LiDAR Navigation", "Precision room mapping and obstacle avoidance for efficient floor coverage."))
        if any(w in text for w in ["suction", "pa", "clean", "vacuum", "motor", "power", "deep"]):
            features.append(ProductFeature("High-Efficiency Deep Cleaning", "Strong suction power and dual-brush system for deep carpet and floor cleaning."))
        if any(w in text for w in ["dock", "auto-empty", "self-empty", "charge", "station", "empty"]):
            features.append(ProductFeature("Auto-Empty Dock & Self-Maintenance", "Automated dust collection and self-docking recharge station."))
        if any(w in text for w in ["mop", "mopping", "water", "wet", "scrub"]):
            features.append(ProductFeature("Smart Mopping System", "Controlled water flow and active surface mopping for hard floors."))
        if any(w in text for w in ["app", "schedule", "wifi", "smart", "remote", "zone"]):
            features.append(ProductFeature("Smart App Control & Scheduling", "Custom zone cleaning, no-go barriers, and automated schedules via companion app."))
        if any(w in text for w in ["hepa", "filter", "allergen", "air", "dust"]):
            features.append(ProductFeature("Multi-Stage HEPA Filtration", "High-efficiency particulate filtration trapping fine dust and allergens."))
        if any(w in text for w in ["battery", "runtime", "minutes", "hours", "mah"]):
            features.append(ProductFeature("Extended Battery Runtime", "High-capacity battery ensuring complete multi-room coverage on a single charge."))

    # 8. Software / AI Tool / Productivity Platform
    elif any(k in category.lower() for k in ["software", "ai tool", "saas", "application"]) or re.search(r"\b(software|ai tool|saas|assistant tool)\b", text):
        if any(w in text for w in ["ai", "generate", "intelligence", "automation", "smart"]):
            features.append(ProductFeature("AI-Powered Automation", "Intelligent workflow automation and smart content/task assistance."))
        if any(w in text for w in ["speed", "fast", "efficient", "productivity", "workflow"]):
            features.append(ProductFeature("High-Efficiency Workflow", "Streamlined operation designed to cut repetitive manual tasks."))
        if any(w in text for w in ["cloud", "sync", "collaborate", "team", "share"]):
            features.append(ProductFeature("Cloud Sync & Collaboration", "Seamless multi-device synchronization and team sharing capabilities."))
        if any(w in text for w in ["interface", "ui", "dashboard", "clean", "simple"]):
            features.append(ProductFeature("Intuitive Interface", "Clean, user-friendly layout with minimal onboarding friction."))
        if any(w in text for w in ["security", "privacy", "encrypt", "protection", "safe"]):
            features.append(ProductFeature("Data Privacy & Encryption", "Robust security standards ensuring sensitive user data remains protected."))

    # 9. General / Fallback clause extraction from description text
    if not features:
        # Split on sentence full stops and list separators, but not inside decimals such as "6.6-inch".
        clauses = re.split(r"(?:(?<!\d)\.|\.(?!\d)|[\n;•])+|(?:\band\b)|(?:\bwith\b)", description)
        valid_clauses = [c.strip() for c in clauses if 10 <= len(c.strip()) <= 80 and _is_product_attribute_clause(c)]
        for idx, c in enumerate(valid_clauses[:4]):
            title = c.split(",")[0].strip().title()
            if len(title) > 35:
                words = title.split()[:4]
                title = " ".join(words)
            features.append(ProductFeature(title, c))

    return features


# -----------------------------------------------------------------------------
# Reusable Prompt System Templates
# -----------------------------------------------------------------------------

QWEN_SYSTEM_PROMPT_TEMPLATE: Final[str] = (
    "You role-play one real consumer from an empirical market-research panel reacting to a proposed technology product concept. "
    "Respond in your own authentic voice from the perspective of your assigned persona profile.\n\n"
    "EVALUATION INSTRUCTIONS:\n"
    "1. Honestly evaluate the proposed product against your practical needs, daily routine, and personal budget.\n"
    "2. State clearly:\n"
    "   - Which specific features attract you and why.\n"
    "   - Which features feel unappealing, gimmicky, or unnecessary for your needs.\n"
    "   - Reasons why you would consider adopting or purchasing this product.\n"
    "   - Reasons for hesitation, skepticism, or rejection.\n"
    "   - Any concerns or barriers (e.g. price vs. spending capacity, build quality, usability, reliability, feature fit).\n"
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


def build_persona_context(persona_id_or_dict: str | dict[str, Any], root: Path = ROOT, category: str = "") -> str:
    """Format an empirical persona into a structured prompt context block derived strictly from survey data.

    Preserves 100% empirical grounding from survey codebook & clustering:
    - Never assumes personal income, spending capacity, or exact personal budget.
    - Never invents study habits, college work, office jobs, or personal needs.
    - Omits smartphone-specific screen time and age to ensure clean generalization.
    """
    if isinstance(persona_id_or_dict, str):
        profile = load_persona_data(persona_id_or_dict, root)
    else:
        profile = persona_id_or_dict

    pid = profile.get("persona_id", "Unknown")
    rule = profile.get("defining_rule", "")
    n_resp = profile.get("n", 0)
    pct = profile.get("pct_of_primary", 0.0)
    comfort = profile.get("tech_comfort_mean_1to5", 3.9)

    if "P1" in pid:
        segment_title = "P1 · Premium Tier Consumer"
        survey_benchmark = "Survey Benchmark: Historical primary device investment > ₹35,000"
        segment_focus = "Prefers solid build quality, reliable performance, and careful finish based on empirical survey clustering."
    elif "P2" in pid:
        segment_title = "P2 · Mid-Range Intensive User"
        survey_benchmark = "Survey Benchmark: Device investment ₹10,000–₹35,000 with intensive usage"
        segment_focus = "Prioritizes sustained speed, multitasking responsiveness, and battery endurance based on empirical survey clustering."
    elif "P3" in pid:
        segment_title = "P3 · Mid-Range Balanced User"
        survey_benchmark = "Survey Benchmark: Device investment ₹10,000–₹35,000 with moderate usage"
        segment_focus = "Values dependable everyday utility, balanced functionality, and reliable day-to-day operation based on empirical survey clustering."
    else:
        segment_title = "P4 · Entry-Level Value User"
        survey_benchmark = "Survey Benchmark: Entry-level device investment < ₹10,000 or undisclosed"
        segment_focus = "Focuses on essential functionality, practical durability, and straightforward value based on empirical survey clustering."

    themes = profile.get("top_experience_themes_from_Q23_Q25", [])
    if themes and isinstance(themes, list) and isinstance(themes[0], dict):
        themes_str = "; ".join(
            f"{t['theme'].replace('_', ' ')} ({t.get('mentions', '')} mentions)" for t in themes[:3]
        )
    else:
        themes_str = "Reliable performance and durability"

    is_general_tech = bool(category and not any(k in category.lower() for k in ["phone", "smartphone"]))

    if is_general_tech:
        lines = [
            f"CONSUMER PERSONA PROFILE: {pid} ({segment_title})",
            f"- Segment definition: {rule} ({n_resp} respondents, {pct:.1f}% of cohort)",
            f"- Technology comfort: {comfort:.2f} on a 1-5 scale",
            f"- Segment priority: {segment_focus}",
            f"- Survey experience priorities: {themes_str}",
            f"- Technology category orientation: Evaluated in the context of this specific product category, not smartphone spending.",
            f"- FACTUAL GROUNDING MANDATE: Evaluate ONLY the proposed product specifications, hardware capabilities, and price.",
            f"- DO NOT invent personal needs, study habits, college work, office tasks, pre-owned devices, or assumed personal income.",
            f"- Replace unsupported generic claims like 'offers everything I need' or 'suits my needs perfectly' with concrete, evidence-grounded evaluations of actual features.",
        ]
    else:
        # Legacy/smartphone evaluation context (preserving test compatibility)
        if "P1" in pid:
            device_tier_desc = "Flagship-tier smartphone"
            price_tier = "Historical device investment > ₹35,000"
            budget_role = "High performance and build quality priority"
        elif "P2" in pid:
            device_tier_desc = "Upper mid-range performance smartphone"
            price_tier = "Device investment ₹10,000–₹35,000"
            budget_role = "Heavy multitasking and sustained performance priority"
        elif "P3" in pid:
            device_tier_desc = "Mid-range balanced smartphone"
            price_tier = "Device investment ₹10,000–₹35,000"
            budget_role = "Balanced utility and practical functionality priority"
        else:
            device_tier_desc = "Entry-level smartphone"
            price_tier = "Entry-level investment < ₹10,000 or undisclosed"
            budget_role = "Essential functionality and practical value priority"

        lines = [
            f"CONSUMER PERSONA PROFILE: {pid} ({segment_title})",
            f"- Segment definition: {rule}",
            f"- Demographics: survey cohort representing {n_resp} respondents ({pct:.1f}%)",
            f"- Current smartphone: {device_tier_desc}",
            f"- Baseline phone tier: {price_tier}",
            f"- Persona archetype role: {budget_role}",
            f"- Usage intensity: empirical survey cluster benchmark",
            f"- Spending capacity: {price_tier}",
            f"- Technology comfort: {comfort:.1f} on a 1-5 scale",
            f"- Persona priorities & preferences: {themes_str}",
            f"- FACTUAL GROUNDING MANDATE: Evaluate ONLY the proposed product specifications and features.",
            f"- DO NOT invent personal study habits, work requirements, or unsupported claims like 'offers everything I need'.",
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
    cat_detected, _, _, _ = validate_and_detect_category(
        name=product.name,
        description=product.description,
        input_category=getattr(product, "category", "") or "",
    )
    persona_block = build_persona_context(persona_id_or_dict, root, category=cat_detected)
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


# -----------------------------------------------------------------------------
# Multi-Construct Natural Indian English Prompting (Universal Tech Evaluation)
# -----------------------------------------------------------------------------

def get_category_tier_info(cat: str, prod_noun: str, price: float) -> tuple[str, str]:
    """Provide realistic, technology-specific price tier context in the Indian market.

    Evaluates pricing relative to the actual product category, preventing false smartphone analogies.
    """
    cat_lower = (cat or "").lower()
    noun_lower = (prod_noun or "").lower()

    if any(k in noun_lower or k in cat_lower for k in ["laptop", "computer", "notebook", "pc"]):
        # Laptops in India: Entry: < 35k, Mainstream Mid-Range: 35k-75k, Premium: > 75k
        if price < 35000:
            return ("entry-level", "an entry-level laptop price tier in the Indian market")
        elif price <= 75000:
            return ("mainstream mid-range", "a standard mainstream mid-range laptop price tier in the Indian market")
        else:
            return ("premium", "a premium high-performance laptop price tier in the Indian market")

    if any(k in noun_lower or k in cat_lower for k in ["tablet", "ipad", "tab"]):
        if price < 20000:
            return ("entry-level", "an entry-level tablet price tier")
        elif price <= 45000:
            return ("mainstream mid-range", "a mainstream mid-range tablet price tier")
        else:
            return ("premium", "a premium tablet price tier")

    if any(k in noun_lower or k in cat_lower for k in ["smartwatch", "wearable", "band", "ring"]):
        if price < 4000:
            return ("entry-level", "an affordable wearable price tier")
        elif price <= 15000:
            return ("mainstream mid-range", "a mid-range smartwatch price tier")
        else:
            return ("premium", "a premium smartwatch price tier")

    if any(k in noun_lower or k in cat_lower for k in ["glasses", "eyewear", "smart glasses"]):
        if price < 8000:
            return ("entry-level", "an accessible entry-level smart glasses price tier")
        elif price <= 25000:
            return ("mainstream mid-range", "a mainstream smart glasses price tier in the Indian market")
        else:
            return ("premium", "a premium luxury smart glasses price tier")

    if any(k in noun_lower or k in cat_lower for k in ["earbuds", "headphone", "audio", "speaker"]):
        if price < 2500:
            return ("entry-level", "an affordable entry-level audio price tier")
        elif price <= 8000:
            return ("mainstream mid-range", "a mid-range audio price tier")
        else:
            return ("premium", "a premium audio price tier")

    if any(k in noun_lower or k in cat_lower for k in ["tv", "television"]):
        if price < 25000:
            return ("entry-level", "an entry-level television price tier")
        elif price <= 55000:
            return ("mainstream mid-range", "a standard mid-range television price tier")
        else:
            return ("premium", "a premium television price tier")

    if any(k in noun_lower or k in cat_lower for k in ["camera", "security", "surveillance"]):
        if price < 3000:
            return ("entry-level", "an accessible security camera price tier")
        elif price <= 7000:
            return ("mainstream mid-range", "a standard smart home security camera price tier")
        else:
            return ("premium", "an advanced security system price tier")

    if any(k in noun_lower or k in cat_lower for k in ["console", "playstation", "xbox", "gaming"]):
        return ("dedicated gaming console", "a standard dedicated gaming console price tier")

    if any(k in noun_lower or k in cat_lower for k in ["robot", "vacuum", "cleaner", "mop", "ecobot", "cleaning robot"]):
        if price < 15000:
            return ("entry-level", "an accessible entry-level cleaning robot price tier")
        elif price <= 40000:
            return ("mainstream mid-range", "a mainstream mid-range cleaning robot price tier")
        else:
            return ("premium", "a premium high-end cleaning robot price tier")

    if any(k in noun_lower or k in cat_lower for k in ["software", "ai tool", "saas", "application", "tool"]):
        if price < 2000:
            return ("entry-level", "an accessible entry-level software/tool price tier")
        elif price <= 10000:
            return ("mainstream mid-range", "a professional mid-range software price tier")
        else:
            return ("premium", "a premium professional/enterprise software price tier")

    if any(k in noun_lower or k in cat_lower for k in ["phone", "smartphone"]):
        if price < 15000:
            return ("entry-level", "an entry-level smartphone price tier")
        elif price <= 35000:
            return ("mainstream mid-range", "a mainstream mid-range smartphone price tier")
        else:
            return ("premium", "a premium flagship smartphone price tier")

    # General tech fallback
    if price < 5000:
        return ("entry-level", "an accessible entry-level price tier")
    elif price <= 35000:
        return ("mainstream mid-range", "a mid-range price tier")
    else:
        return ("premium", "a higher-tier price tier")


def get_category_persona_price_zones(category: str, prod_noun: str) -> dict[str, dict[str, Any]]:
    """Return realistic price ranges (min_inr, max_inr) for each persona in the specific product category.

    Aligns with empirical persona archetypes:
    - P1 Premium Consumer: expects premium quality, flagship performance, advanced features
    - P2 Tech Enthusiast: prioritizes performance, specifications, and advanced technology
    - P3 Practical Everyday User: values reliability, practical features, ease of use, and balanced pricing
    - P4 Budget-Conscious User: prioritizes affordability, essential features, and value for money
    """
    cat_lower = (category or "").lower()
    noun_lower = (prod_noun or "").lower()

    if any(k in noun_lower or k in cat_lower for k in ["earbuds", "headphone", "audio", "speaker"]):
        return {
            "P1": {"min": 5000.0, "max": 12000.0, "desc": "premium audiophile audio tier (₹12,000 and above)"},
            "P2": {"min": 5000.0, "max": 15000.0, "desc": "advanced features and high-res audio tier (₹5,000–₹15,000)"},
            "P3": {"min": 2000.0, "max": 6000.0, "desc": "balanced everyday audio tier (₹2,000–₹6,000)"},
            "P4": {"min": 500.0, "max": 2500.0, "desc": "accessible budget audio tier (below ₹2,500)"},
        }

    if any(k in noun_lower or k in cat_lower for k in ["laptop", "computer", "notebook", "pc"]):
        return {
            "P1": {"min": 60000.0, "max": 100000.0, "desc": "premium flagship laptop tier (₹1,00,000 and above)"},
            "P2": {"min": 60000.0, "max": 120000.0, "desc": "high-performance multitasking laptop tier (₹60,000–₹1,20,000)"},
            "P3": {"min": 35000.0, "max": 65000.0, "desc": "dependable everyday productivity laptop tier (₹35,000–₹65,000)"},
            "P4": {"min": 15000.0, "max": 35000.0, "desc": "entry-level budget laptop tier (below ₹35,000)"},
        }

    if any(k in noun_lower or k in cat_lower for k in ["robot", "vacuum", "cleaner", "mop", "ecobot", "cleaning robot"]):
        return {
            "P1": {"min": 50000.0, "max": 100000.0, "desc": "ultra-luxury automated cleaning station tier (₹1,00,000 and above)"},
            "P2": {"min": 40000.0, "max": 100000.0, "desc": "advanced LiDAR and smart automation cleaning tier (₹40,000–₹1,00,000)"},
            "P3": {"min": 20000.0, "max": 45000.0, "desc": "practical automated floor cleaning tier (₹20,000–₹45,000)"},
            "P4": {"min": 5000.0, "max": 20000.0, "desc": "entry-level basic automated cleaner tier (below ₹20,000)"},
        }

    if any(k in noun_lower or k in cat_lower for k in ["tablet", "ipad", "tab"]):
        return {
            "P1": {"min": 35000.0, "max": 60000.0, "desc": "flagship pro workstation tablet tier (₹60,000 and above)"},
            "P2": {"min": 35000.0, "max": 65000.0, "desc": "fast performance multitasking tablet tier (₹35,000–₹65,000)"},
            "P3": {"min": 15000.0, "max": 35000.0, "desc": "balanced media and everyday tablet tier (₹15,000–₹35,000)"},
            "P4": {"min": 7000.0, "max": 15000.0, "desc": "budget entry-level tablet tier (below ₹15,000)"},
        }

    if any(k in noun_lower or k in cat_lower for k in ["smartwatch", "wearable", "band", "ring"]):
        return {
            "P1": {"min": 10000.0, "max": 25000.0, "desc": "luxury smartwatch & premium sports tier (₹25,000 and above)"},
            "P2": {"min": 10000.0, "max": 25000.0, "desc": "feature-rich OS smartwatch tier (₹10,000–₹25,000)"},
            "P3": {"min": 3000.0, "max": 10000.0, "desc": "reliable daily fitness & notifications tier (₹3,000–₹10,000)"},
            "P4": {"min": 1000.0, "max": 3000.0, "desc": "accessible budget fitness band tier (below ₹3,000)"},
        }

    if any(k in noun_lower or k in cat_lower for k in ["glasses", "eyewear", "smart glasses"]):
        return {
            "P1": {"min": 18000.0, "max": 35000.0, "desc": "luxury designer smart eyewear tier (₹35,000 and above)"},
            "P2": {"min": 18000.0, "max": 35000.0, "desc": "advanced hands-free AI smart glasses tier (₹18,000–₹35,000)"},
            "P3": {"min": 8000.0, "max": 18000.0, "desc": "everyday audio & call smart glasses tier (₹8,000–₹18,000)"},
            "P4": {"min": 3000.0, "max": 8000.0, "desc": "accessible entry-level smart eyewear tier (below ₹8,000)"},
        }

    if any(k in noun_lower or k in cat_lower for k in ["tv", "television"]):
        return {
            "P1": {"min": 45000.0, "max": 80000.0, "desc": "premium OLED/flagship home cinema tier (₹80,000 and above)"},
            "P2": {"min": 45000.0, "max": 85000.0, "desc": "120Hz gaming and high-end display TV tier (₹45,000–₹85,000)"},
            "P3": {"min": 20000.0, "max": 45000.0, "desc": "balanced 4K family smart TV tier (₹20,000–₹45,000)"},
            "P4": {"min": 8000.0, "max": 20000.0, "desc": "entry-level budget TV tier (below ₹20,000)"},
        }

    if any(k in noun_lower or k in cat_lower for k in ["camera", "security", "surveillance"]):
        return {
            "P1": {"min": 5000.0, "max": 10000.0, "desc": "advanced multi-camera / 4K security tier (₹10,000 and above)"},
            "P2": {"min": 5000.0, "max": 12000.0, "desc": "smart pan-tilt AI tracking security tier (₹5,000–₹12,000)"},
            "P3": {"min": 2500.0, "max": 5000.0, "desc": "reliable everyday indoor monitoring tier (₹2,500–₹5,000)"},
            "P4": {"min": 1200.0, "max": 2500.0, "desc": "accessible entry-level Wi-Fi camera tier (below ₹2,500)"},
        }

    # Default / Smartphone:
    # - P1: ₹50,000 and above
    # - P2: ₹25,000–₹50,000
    # - P3: ₹10,000–₹25,000
    # - P4: Below ₹15,000
    return {
        "P1": {"min": 50000.0, "max": 100000.0, "desc": "premium flagship smartphone tier (₹50,000 and above)"},
        "P2": {"min": 25000.0, "max": 50000.0, "desc": "high-performance tech enthusiast smartphone tier (₹25,000–₹50,000)"},
        "P3": {"min": 10000.0, "max": 25000.0, "desc": "practical everyday balanced smartphone tier (₹10,000–₹25,000)"},
        "P4": {"min": 3000.0, "max": 15000.0, "desc": "budget-conscious essential smartphone tier (below ₹15,000)"},
    }


PERSONA_VOICES: Final[dict[str, str]] = {
    "P1": (
        "You care most about looks, build quality, camera and a premium feel. Cheap-feeling products put you off even if they are affordable. "
        "You sound confident and a bit picky. Answer style: start with your first impression of the quality, then say if it feels premium enough for you."
    ),
    "P2": (
        "You care most about speed, processor, RAM, display and latest tech. You use your phone very heavily, so you notice heating, lag and battery drain under gaming and multitasking, and you are quick to look at other brands if performance disappoints. "
        "You sound curious, analytical and demanding. Answer style: start by naming one or two specs, then say whether they would hold up under your heavy use."
    ),
    "P3": (
        "You care most about whether it works reliably every day: battery, ease of use, durability and fair price. "
        "You sound calm and sensible, not excited by fancy features. Answer style: start with how it would fit a normal day, then give your simple verdict."
    ),
    "P4": (
        "You care most about price and value for money. You want the basics to work and you worry about spending too much, but when a product covers your daily needs at a fair price you are genuinely satisfied and accept its limits. "
        "You sound careful and practical. Answer style: start with the price, then say whether the basic features make it worth that money."
    ),
}


PERSONA_BI_FOCUS: Final[dict[str, str]] = {
    "P1": "You buy for quality and feel. A cheap price does not attract you by itself; you ask whether it feels premium enough to be your main device.",
    "P2": "You buy for specs and performance. You check whether the processor, display, battery and features will hold up under heavy daily use without heating or lag.",
    "P3": "You buy for daily use. You check whether it will be reliable, easy and fairly priced for normal everyday tasks.",
    "P4": "You buy for value. You think hard about the money first; if the basic features are worth that money, you are happy to buy.",
}


def _price_fit_note(persona_key: str, price: float, zone: dict[str, Any], prod_noun: str) -> str:
    """Describe, in plain words, how the price compares with what this survey segment usually pays."""
    if price <= 0:
        return ""
    lo, hi = zone["min"], zone["max"]
    if price < lo * 0.6:
        fit = "much cheaper than"
    elif price < lo:
        fit = "a bit cheaper than"
    elif price <= hi:
        fit = "about the same as"
    elif price <= hi * 1.5:
        fit = "more than"
    else:
        fit = "far more than"
    if persona_key == "P1" and price >= lo:
        fit = "in line with" if fit in ("about the same as", "more than") else fit
    return f"For context, people like you usually buy a {prod_noun} in the {zone['desc']}, so this price is {fit} that."


def _other_category_examples(prod_noun: str) -> str:
    """List common device types other than the evaluated one, so the prompt can forbid substituting them."""
    nouns = ["smartphone", "laptop", "tablet", "smartwatch", "TV", "earbuds"]
    return ", ".join(n for n in nouns if n.lower() not in prod_noun.lower())


INDIAN_SURVEY_PERSONA_SYSTEM_PROMPT: Final[str] = (
    "You role-play an authentic respondent from an Indian consumer technology survey panel. "
    "Respond in natural, simple everyday Indian English, matching how normal consumers write in genuine survey responses. "
    "Use everyday words and clear, conversational sentences. "
    "Avoid robotic, overly formal, academic, or repetitive language.\n\n"
    "FORBIDDEN WORDS & ROBOTIC PHRASES (DO NOT USE):\n"
    "- NEVER use words such as: 'adequate', 'steep', 'top-notch', 'high-end', 'prohibitive', 'paramount', 'furthermore', 'moreover', 'leverage', 'utilize', 'refined build', 'demanding tasks', 'fatiguing', or 'acoustic isolation'.\n"
    "- NEVER speak as if you already own a prospective product or repeat formulaic phrases. Always express realistic prospective purchase or adoption intentions (such as considering adoption, feeling hesitant, exploring alternatives, or deciding against buying) in natural, varied everyday language.\n"
    "- Evaluate the exact product category named in the task. Never describe it as a different kind of device.\n"
    "- Use simple, natural everyday words instead (e.g., 'good', 'decent', 'very good', 'smooth', 'on the higher side', 'solid build', 'heavy daily work', 'tiring').\n\n"
    "STRICT FACTUAL GROUNDING ON PERSONA:\n"
    "- NEVER invent personal backstories, pre-owned items, or fictional possessions (e.g., NEVER say 'I already have a laptop', 'I already own earbuds', or 'I am loyal to brand X').\n"
    "- You do NOT own the evaluated product or any competing device in that category. Evaluate the proposed product purely on its listed specifications, features, and price.\n"
    "- DO NOT invent personal financial facts, bank savings, family allowances, or a personal rupee budget. You MAY say a price feels too high, fair, or cheap for you, based on the price context given. Evaluate value for money based on the product category.\n"
    "- DO NOT invent personal needs, study habits, college work, office jobs, or personal daily routines.\n\n"
    "PROHIBITION OF UNGROUNDED CLAIMS & CLICHES:\n"
    "- NEVER use unsupported generic claims like 'This tablet offers everything I need', 'It suits my needs perfectly', 'It matches my requirements', or 'fits my study/work needs'.\n"
    "- Provide neutral, evidence-grounded evaluations of the actual product features, display, processing, battery, build, and price.\n\n"
    "CONSTRUCT RELEVANCE & SEPARATION (MANDATORY):\n"
    "- PE (Performance Expectancy), EE (Effort Expectancy), HM (Hedonic Motivation), SI (Social Influence), and PP (Perceived Protection) must evaluate ONLY product features. NEVER discuss price, cost, or budget in PE, EE, HM, SI, or PP!\n"
    "- Save all price and adoption evaluation strictly for BI (Behavioral Intention) and overall.\n"
    "- In HM (Hedonic Motivation), evaluate media and enjoyment based ONLY on listed features. NEVER mention gaming unless the product is specifically a gaming console or gaming device!\n"
    "- In PP (Perceived Protection), evaluate physical build quality, hardware durability, and reliability. Do NOT fabricate data privacy, encryption, surveillance, or alarm concerns unless explicitly stated in the product description.\n\n"
    "STRICT METHODOLOGICAL RESTRICTIONS (MANDATORY):\n"
    "- Output TEXT ONLY in valid JSON format.\n"
    "- NEVER mention, compare with, or name any real commercial brands or competing product models in your responses. Focus strictly on the evaluated product and its features.\n"
    "- NEVER provide numerical ratings, Likert scores, points, stars, or percentages (e.g., do NOT say '4/5', '8/10', '70% chance').\n"
    "- Do NOT mention being an AI, an LLM, or a language model."
)


def build_persona_multiconstruct_prompt(
    product: ProductConcept,
    persona_id_or_dict: str | dict[str, Any],
    root: Path = ROOT,
    product_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a multi-construct evaluation prompt producing authentic Indian English responses.

    ``product_context`` (from ``scripts.response_validation.build_product_context``) is the single,
    pre-validated description of the product; when given it is placed first in every persona's prompt.

    Produces separate natural-language responses for PE, EE, HM, SI, PP, BI, and an overall summary
    with strict persona consistency and category-grounded price evaluation.
    """
    cat_detected, prod_noun, usage_role, eco_detected = validate_and_detect_category(
        name=product.name,
        description=product.description,
        input_category=getattr(product, "category", "") or ""
    )

    persona_block = build_persona_context(persona_id_or_dict, root, category=cat_detected)
    product_block = product.format_block()
    pid = persona_id_or_dict if isinstance(persona_id_or_dict, str) else persona_id_or_dict.get("persona_id", "custom")

    price_str = f"₹{int(product.price_inr):,}" if product.price_inr else "the proposed price"
    price = float(product.price_inr or 0.0)

    # Category-specific flags
    is_camera = "camera" in cat_detected.lower() or "surveillance" in cat_detected.lower() or "security camera" in prod_noun
    is_audio = "audio" in cat_detected.lower() or any(k in prod_noun for k in ["earbuds", "headphones", "speaker", "audio"])
    is_console = prod_noun in ["gaming console", "console"]
    is_gaming_device = is_console or "gaming" in cat_detected.lower() or "gaming" in product.name.lower()
    is_phone = any(k in prod_noun.lower() or k in cat_detected.lower() for k in ["phone", "smartphone"])

    tier_name, tier_desc = get_category_tier_info(cat_detected, prod_noun, price)
    category_zones = get_category_persona_price_zones(cat_detected, prod_noun)
    z1 = category_zones["P1"]
    z2 = category_zones["P2"]
    z3 = category_zones["P3"]
    z4 = category_zones["P4"]

    if "P1" in pid:
        if price < 20000.0:
            p1_price_eval = f"The proposed price is {price_str}. This is completely affordable and inexpensive, but represents an entry-level tier with low premium appeal for a {prod_noun} (your baseline expectation is {z1['desc']})."
            p1_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and features. "
                f"As a premium-oriented consumer (baseline expectation: {z1['desc']}), evaluate whether this entry-level {prod_noun} meets basic expectations, "
                f"but recognize that entry-level products typically use basic materials, compromise on refinement, and lack premium appeal. "
                f"Even though the price is low and completely affordable, an entry-level device does not satisfy your premium expectations. While the monetary cost is low, you notice basic materials, compromised finish, and lack of top-tier performance. "
                f"A low price does not automatically make it attractive to you. Express realistic hesitation or skepticism about adopting this basic {prod_noun} as your primary {usage_role}. "
                f"State clearly that you are undecided or hesitant to adopt it unless it genuinely exceeds basic expectations in build quality. (Soft calibration target: 2.5 to 3.5)."
            )
        elif price < z1["min"]:
            p1_price_eval = f"The proposed price is {price_str}, which represents {tier_desc}."
            p1_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and features. "
                f"As a premium-oriented consumer whose baseline expectation is {z1['desc']}, a {price_str} product sits in the mid-range tier and should not automatically be considered premium. "
                f"Evaluate whether the craftsmanship, reliable performance, and materials actually deliver premium quality, distinctive features, and superior experience. Do not automatically approve it. "
                f"If the product cuts corners with mid-tier build quality, plastic chassis, or lacks distinctive flagship refinement, express clear skepticism or hesitation about adopting it. "
                f"Only if it genuinely delivers exceptional craftsmanship, distinctive refinement, and premium experience would you consider adopting it."
            )
        elif price < z1["max"]:
            p1_price_eval = f"The proposed price is {price_str}, which represents {tier_desc}."
            p1_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and features. "
                f"As a premium-oriented consumer, weigh whether the craftsmanship, reliable performance, and premium experience justify this price tier. "
                f"If the product delivers solid build quality and refined performance that justifies the cost, express positive adoption; "
                f"if you notice plastic build shortcuts, compromises, or lack of refinement, state clearly why you would hesitate or prefer a higher-end option."
            )
        else:
            p1_price_eval = f"The proposed price is {price_str}, which represents a premium flagship tier."
            p1_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and features. "
                f"At this premium tier, your premium expectations increase substantially. You expect uncompromising top-tier materials, cutting-edge speed, and exceptional finish. "
                f"If the product delivers true flagship prestige, solid build quality, and flawless capabilities, express confident positive adoption; "
                f"if there are any hardware flaws, thermal issues, or unrefined aspects, reject it or express clear hesitation despite its high price."
            )

        pricing_guide = (
            f"- Persona Profile: You belong to the Premium survey segment (values solid build quality, reliable performance, and careful finish).\n"
            f"- Category Context: {p1_price_eval}\n"
            f"- Evaluation Focus: Evaluate whether the build quality, material durability, and functional performance meet your high standards for a {prod_noun} in your daily use as a {usage_role}.\n"
            f"{p1_bi_guide}"
        )
        selected_bi_guide = p1_bi_guide
    elif "P2" in pid:
        if price < z2["min"]:
            p2_price_eval = f"The proposed price is {price_str}, which represents an entry-level tier with modest technical specifications (your baseline expectation is {z2['desc']})."
            p2_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and technical performance. "
                f"As a tech enthusiast who prioritizes performance, specifications, and advanced technology (baseline expectation: {z2['desc']}), a low price alone does not automatically make a product attractive. "
                f"Critically examine the processor, RAM, display, and hardware specifications. At this entry-level tier, modest processors, limited RAM, and basic displays will likely face bottlenecks under intensive multitasking or heavy daily workloads. "
                f"You might find it decent as an entry-level secondary device or basic backup, but would be cautious, hesitant, or undecided about adopting it as your primary daily machine for heavy workloads. (Soft calibration target: 3.0 to 4.0)."
            )
        elif price <= z2["max"]:
            p2_price_eval = f"The proposed price is {price_str}, which represents {z2['desc']}."
            p2_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and performance. "
                f"As a tech enthusiast, evaluate specifications, technical capabilities, speed, multitasking responsiveness, and whether the price justifies the technology. Do not automatically approve it. "
                f"If the product genuinely delivers the speed, multitasking responsiveness, and endurance needed for intensive daily operation, express decisive positive adoption explaining how it supports your daily tasks. "
                f"If it falls short of heavy workloads, throttles, or cuts corners on specs for this price, state your technical concerns and hesitation clearly."
            )
        else:
            p2_price_eval = f"The proposed price is {price_str}, which represents a high-end luxury tier above standard enthusiast pricing."
            p2_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and technical capabilities. "
                f"At this high-end price point, you expect cutting-edge technology, peak processing power, and excellent sustained performance without thermal throttling or hardware compromises. "
                f"If it delivers bleeding-edge specs and uncompromising power, express confident positive adoption; "
                f"if the hardware falls short of its premium price, state your technical rejection clearly."
            )

        pricing_guide = (
            f"- Persona Profile: You belong to the Mid-Range Intensive User survey segment (prioritizes sustained speed, multitasking, and battery endurance for heavy daily use).\n"
            f"- Category Context: {p2_price_eval}\n"
            f"- Evaluation Focus: Evaluate whether the hardware and capabilities can comfortably handle continuous, heavy daily workloads as a {usage_role} without slowing down or failing.\n"
            f"{p2_bi_guide}"
        )
        selected_bi_guide = p2_bi_guide
    elif "P3" in pid:
        if price < z3["min"]:
            p3_price_eval = f"The proposed price is {price_str}, which is an ultra-budget entry price point."
            p3_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and everyday utility. "
                f"Check whether it covers everyday communication, browsing, and core daily tasks reliably without annoying bugs or poor battery life. "
                f"If it proves dependable for routine everyday use, express practical positive adoption. If it feels too fragile or struggles with basic daily tasks, explain your hesitation."
            )
        elif price <= z3["max"]:
            p3_price_eval = f"The proposed price is {price_str}, which represents {tier_desc} ({z3['desc']})."
            p3_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and everyday balance. "
                f"Values reliability, practical features, ease of use, and balanced pricing (your baseline expectation is {z3['desc']}). "
                f"Evaluate whether the product meets daily needs, whether features are practical and useful, and whether the price is practical. "
                f"This product is in your balanced price sweet spot if it is reliable and sufficient for everyday needs like calling, messaging, browsing, and dependable battery life. "
                f"If it covers ordinary daily essentials reliably without annoying bugs or poor battery, express sensible, practical positive adoption. "
                f"If it feels too fragile, overpriced for everyday utility, or struggles with basic daily tasks, explain your hesitation. (Soft calibration target: 3.5 to 4.5)."
            )
        else:
            p3_price_eval = f"The proposed price is {price_str}, which represents a luxury/flagship tier well above standard everyday budgets ({z3['desc']})."
            p3_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and everyday usefulness. "
                f"Strong price justification needed. For routine everyday tasks, this price represents a very steep expense that requires strong justification. "
                f"Unless the product offers indispensable everyday convenience or unmatched longevity that clearly warrants the high cost, express clear hesitation or reluctance, noting that simpler, more affordable alternatives satisfy your everyday needs."
            )

        pricing_guide = (
            f"- Persona Profile: You belong to the Mid-Range Balanced User survey segment (values dependable everyday utility, balanced features, and sensible value).\n"
            f"- Category Context: {p3_price_eval}\n"
            f"- Evaluation Focus: Evaluate whether the product offers a well-rounded, dependable experience for everyday tasks as a {usage_role} and whether the features represent sensible value for money.\n"
            f"{p3_bi_guide}"
        )
        selected_bi_guide = p3_bi_guide
    else:
        # P4
        if price <= z4["max"]:
            p4_price_eval = f"The proposed price is {price_str}, which is pocket-friendly, relatively accessible, and within your financial comfort zone ({z4['desc']})."
            p4_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and essential functionality. "
                f"This price is relatively accessible and attractive provided essential daily needs are met (baseline expectation: {z4['desc']}). "
                f"If the product covers your core daily requirements reliably and durably at this affordable price point, express confident positive adoption and strong purchase intent. "
                f"If it feels cheaply made or fails at basic essentials even at this price, explain why you would hesitate. (Soft calibration target: 4.0 to 5.0)."
            )
        else:
            p4_price_eval = f"The proposed price is {price_str}, which is more than people like you usually spend ({z4['desc']})."
            p4_bi_guide = (
                f"- Behavioral Intention (BI): Evaluate your realistic adoption intention taking into account the price ({price_str}) and essential functionality. "
                f"As a budget-conscious consumer who prioritizes affordability and value for money (baseline expectation: {z4['desc']}), evaluate affordability, essential features, and whether spending {price_str} exceeds your comfortable budget. "
                f"This price represents a serious financial expense with increasing affordability concerns and a major, insurmountable financial barrier. "
                f"For expensive products (such as a ₹25,000+ or ₹50,000+ smartphone, ₹80,000 laptop, or ₹1,50,000 cleaning robot), this price is far out of reach and generally receives low adoption intention. "
                f"Express strong hesitation, low adoption intention, or definitive rejection, explaining that the product is far too expensive for your budget, regardless of its features."
            )

        pricing_guide = (
            f"- Persona Profile: You belong to the Entry-Level Value User survey segment (focuses on essential functionality, practical durability, and straightforward value).\n"
            f"- Category Context: {p4_price_eval}\n"
            f"- Evaluation Focus: Evaluate whether the product covers core daily essentials reliably as a {usage_role} and provides durable, practical everyday value.\n"
            f"{p4_bi_guide}"
        )
        selected_bi_guide = p4_bi_guide

    persona_key = next((p for p in ("P1", "P2", "P3", "P4") if p in pid), "P4")
    calibrated_bi_guide = (
        f"- Behavioral Intention (BI): Would you actually buy this {prod_noun} at {price_str}? "
        f"{_price_fit_note(persona_key, price, category_zones[persona_key], prod_noun)} "
        f"{PERSONA_BI_FOCUS[persona_key]} "
        f"Decide for yourself: you may want it, be unsure, or not want it. Give your honest adoption intention and the main reason, "
        f"mentioning a real feature or the price. Talk about buying this product; never say 'continue using'."
    )
    pricing_guide = pricing_guide.replace(selected_bi_guide, calibrated_bi_guide)
    pricing_guide += f"\n- Your Voice: {PERSONA_VOICES[persona_key]}"
    selected_bi_guide = calibrated_bi_guide
    bi_prompt_text = selected_bi_guide.replace("- Behavioral Intention (BI): ", "").strip()

    # Domain-specific construct instruction builders
    is_glasses = any(k in cat_detected.lower() or k in prod_noun for k in ["glasses", "eyewear", "spectacles"])
    is_robotics = any(k in cat_detected.lower() or k in prod_noun for k in ["robot", "vacuum", "cleaner", "mop", "ecobot"])
    is_wearable = any(k in cat_detected.lower() or k in prod_noun for k in ["smartwatch", "wearable", "fitness band", "smart ring", "ring"])
    is_computing = any(k in cat_detected.lower() or k in prod_noun for k in ["laptop", "computer", "pc", "desktop", "notebook"])

    if is_camera:
        construct_instructions = (
            f"1. PE (Performance Expectancy): How effective this security camera is for home monitoring, motion detection, video recording, and night vision based ONLY on its listed features. (CRITICAL: Do NOT mention price or cost here; do NOT discuss music or headphones).\n"
            f"2. EE (Effort Expectancy): How simple and easy it feels to set up, mount, connect to Wi-Fi, and monitor through the companion app. (CRITICAL: Do NOT mention price or cost here).\n"
            f"3. HM (Hedonic Motivation): The reassurance, satisfaction, and peace of mind of having clear home monitoring. (CRITICAL: Do NOT mention price, music, entertainment, or gaming).\n"
            f"4. SI (Social Influence): What family members, housemates, or peers interested in home safety might think about this camera. (CRITICAL: Do NOT mention price or personal budgets here).\n"
            f"5. PP (Perceived Protection): Hardware reliability, camera access control, feed privacy, and data protection. (CRITICAL: Do NOT mention price here).\n"
            f"6. BI (Behavioral Intention): {bi_prompt_text}\n"
        )
    elif is_audio:
        construct_instructions = (
            f"1. PE (Performance Expectancy): How effective and clear these {prod_noun} are for listening to music, watching videos, and taking phone calls based ONLY on listed features. (CRITICAL: Do NOT mention price or cost here; save price for BI).\n"
            f"2. EE (Effort Expectancy): How simple it is to pair via Bluetooth, use controls, and charge in the case. (CRITICAL: Do NOT mention price or cost here).\n"
            f"3. HM (Hedonic Motivation): Personal enjoyment and fun when listening to favourite songs and videos. (CRITICAL: Do NOT mention price or cost here).\n"
            f"4. SI (Social Influence): What peers, friends, or classmates who use audio gear think about these {prod_noun}. (CRITICAL: Do NOT mention price or personal budgets here).\n"
            f"5. PP (Perceived Protection): Connection stability, pairing safety, and physical build durability. (CRITICAL: Do NOT mention home security, surveillance cameras, alarms, or physical locks).\n"
            f"6. BI (Behavioral Intention): {bi_prompt_text}\n"
        )
    elif is_glasses:
        construct_instructions = (
            f"1. PE (Performance Expectancy): How effective and useful these {prod_noun} are for hands-free voice assistance, audio playback, real-time navigation, and AI visual recognition based ONLY on listed features. (CRITICAL: Do NOT mention price or cost here; do NOT discuss smartphone screens or cameras).\n"
            f"2. EE (Effort Expectancy): How comfortable, lightweight, and natural they feel to wear on the face all day and operate via voice or touch controls. (CRITICAL: Do NOT mention price or cost here).\n"
            f"3. HM (Hedonic Motivation): The personal excitement, modern lifestyle appeal, and satisfaction of wearing hands-free smart eyewear. (CRITICAL: Do NOT mention price or gaming; do NOT mention cost here).\n"
            f"4. SI (Social Influence): What friends, peers, housemates, or colleagues think about wearing smart glasses in public or social settings. (CRITICAL: Do NOT mention price or personal budgets here).\n"
            f"5. PP (Perceived Protection): Frame durability, hardware reliability, privacy LED indicators, and microphone safety controls. (CRITICAL: Do NOT mention price or cost here).\n"
            f"6. BI (Behavioral Intention): {bi_prompt_text}\n"
        )
    elif is_robotics:
        construct_instructions = (
            f"1. PE (Performance Expectancy): How effective and thorough this {prod_noun} is for floor cleaning, navigation, suction, and dust pickup based ONLY on its listed specifications. (CRITICAL: Do NOT mention price or cost here; do NOT discuss smartphone features or headphones).\n"
            f"2. EE (Effort Expectancy): How easy it feels to set up, schedule, empty dust, and control via app or buttons. (CRITICAL: Do NOT mention price or cost here).\n"
            f"3. HM (Hedonic Motivation): The relief, convenience, and modern satisfaction of automated hands-free floor cleaning. (CRITICAL: Do NOT mention gaming or music; do NOT mention price or cost here).\n"
            f"4. SI (Social Influence): What family members, housemates, or visitors would think about having this {prod_noun} in the home. (CRITICAL: Do NOT mention price or personal budgets here).\n"
            f"5. PP (Perceived Protection): Mechanical durability, sensor reliability, obstacle safety, and floor protection over time. (CRITICAL: Do NOT mention price or cost here).\n"
            f"6. BI (Behavioral Intention): {bi_prompt_text}\n"
        )
    elif is_wearable:
        construct_instructions = (
            f"1. PE (Performance Expectancy): How effective and accurate this {prod_noun} is for health tracking, workout monitoring, and daily notifications based ONLY on its listed features. (CRITICAL: Do NOT mention price or cost here).\n"
            f"2. EE (Effort Expectancy): How comfortable it is to wear daily, read the screen, and sync with the companion app. (CRITICAL: Do NOT mention price or cost here).\n"
            f"3. HM (Hedonic Motivation): Lifestyle satisfaction, fitness motivation, and enjoyment of wearing this device. (CRITICAL: Do NOT mention price or cost here).\n"
            f"4. SI (Social Influence): What friends, fitness peers, or colleagues think about wearing this {prod_noun}. (CRITICAL: Do NOT mention price or personal budgets here).\n"
            f"5. PP (Perceived Protection): Water resistance, build toughness, scratch resistance, and health data privacy. (CRITICAL: Do NOT mention price or cost here).\n"
            f"6. BI (Behavioral Intention): {bi_prompt_text}\n"
        )
    elif is_computing:
        construct_instructions = (
            f"1. PE (Performance Expectancy): How effective and useful this {prod_noun} is for computing tasks, speed, multitasking, and battery life based ONLY on its listed hardware specifications (processor, display, battery, ports). (CRITICAL: Do NOT mention price, cost, or budget here; do NOT make unsupported claims like 'offers everything I need').\n"
            f"2. EE (Effort Expectancy): How simple and easy it feels to type, navigate the touchpad/OS, and carry for daily use. (CRITICAL: Do NOT mention price, cost, or budget here).\n"
            f"3. HM (Hedonic Motivation): How pleasant, satisfying, and enjoyable it feels to use for daily tasks and media viewing based ONLY on listed features. (CRITICAL: Do NOT mention gaming unless this is a gaming device; do NOT mention price or cost here).\n"
            f"4. SI (Social Influence): What peers, colleagues, or friends who use laptops might think about it. (CRITICAL: Do NOT mention price, cost, or personal budgets here).\n"
            f"5. PP (Perceived Protection): Physical build durability, hinge reliability, thermal cooling, and longevity over time. (CRITICAL: Do NOT mention price or cost here).\n"
            f"6. BI (Behavioral Intention): {bi_prompt_text}\n"
        )
    else:
        gaming_hm_instruction = (
            f"3. HM (Hedonic Motivation): Enjoyment and engaging gameplay experience based on listed features. (CRITICAL: Do NOT mention price or cost here).\n"
            if is_gaming_device else
            f"3. HM (Hedonic Motivation): How pleasant, satisfying, and enjoyable it feels to use for daily tasks and media viewing based ONLY on listed features. (CRITICAL: Do NOT mention gaming unless this is a gaming device; do NOT mention price or cost here).\n"
        )

        construct_instructions = (
            f"1. PE (Performance Expectancy): How effective, capable, and useful this {prod_noun} is for its primary purpose based ONLY on its listed specifications and features. (CRITICAL: Do NOT mention price, cost, or budget here; do NOT make unsupported claims like 'offers everything I need').\n"
            f"2. EE (Effort Expectancy): How simple and easy it feels to set up, navigate, operate, and maintain based on its interface and controls. (CRITICAL: Do NOT mention price, cost, or budget here).\n"
            f"{gaming_hm_instruction}"
            f"4. SI (Social Influence): What peers, colleagues, housemates, or friends who use this category of product might think about it. (CRITICAL: Do NOT mention price, cost, or personal budgets here).\n"
            f"5. PP (Perceived Protection): Physical build durability, hardware reliability, safety, and longevity over time. Discuss data security ONLY if the product explicitly lists security features. (CRITICAL: Do NOT mention price or cost here).\n"
            f"6. BI (Behavioral Intention): {bi_prompt_text}\n"
        )

    context_block = ""
    if product_context:
        from scripts.response_validation import product_context_block
        context_block = product_context_block(product_context) + "\n\n"

    user_message = (
        f"{context_block}"
        f"{persona_block}\n\n"
        f"--- PROPOSED PRODUCT CONCEPT ---\n"
        f"{product_block}\n\n"
        f"--- PERSONA EVALUATION PERSPECTIVE ---\n"
        f"{pricing_guide}\n\n"
        f"--- YOUR TASK ---\n"
        f"Evaluate this {prod_noun} across 6 survey constructs and an overall summary from your persona perspective.\n"
        f"CRITICAL GROUNDING RULES:\n"
        f"- Use ONLY the actual features and specifications listed in the product description above.\n"
        f"- NEVER invent, assume, or hallucinate features, specs, graphics cards, gaming frame rates, AI features, cameras, or accessories not stated above.\n"
        f"- NEVER claim you already own a device of this type (e.g. NEVER say 'I already have a laptop', 'I already own earbuds', or 'my current laptop is fine'). Evaluate THIS proposed product directly.\n"
        f"- NEVER make unsupported generic statements like 'offers everything I need', 'suits my needs perfectly', 'matches all my needs', or 'fits my study/work requirements'. Evaluate concrete hardware features (display, battery, speed, build).\n"
        f"- DO NOT invent personal study habits, college work, homework, office meetings, or assumed personal income.\n"
        f"- When evaluating audio earbuds or headphones, focus STRICTLY on audio, music, calls, Bluetooth pairing, and daily listening. NEVER discuss home security cameras, surveillance, or physical alarms!\n"
        f"- When evaluating a security camera, focus STRICTLY on home security, video monitoring, and privacy. NEVER discuss headphones, music, sound quality, smartphone cameras, battery optimization, or gaming!\n"
        f"- If evaluating a tablet, gaming console, TV, smartwatch, laptop, or appliance, focus ONLY on what that product actually does. Do not treat it as a smartphone or headphones.\n"
        f"- You are evaluating {product.name}, a {prod_noun} in the {cat_detected} category. Focus STRICTLY on what this specific product actually does. "
        f"Always call it a {prod_noun}; NEVER call it a {_other_category_examples(prod_noun)} or discuss any other kind of product.\n"
        f"- Only BI and overall may talk about price or whether you would buy it. PE, EE, HM, SI and PP talk only about their own topic.\n"
        f"- Start every answer in a different way. Do NOT start with 'Given the price', 'Given its', 'Considering', 'I am certain', 'Overall' or the product name.\n"
        f"- Express authentic, diverse consumer sentiments: positive, neutral, hesitant, or critical. Do not default to polite praise. If the product is expensive, lacks needed features, or feels unproven, state your genuine skepticism or hesitation.\n"
        f"- Avoid repetitive sentence structures. Vary your vocabulary and sentence flow naturally across the constructs.\n"
        f"- Speak naturally in simple everyday Indian English, like an ordinary consumer.\n"
        f"- Do NOT use complicated academic words or robotic repetitive phrases (e.g. avoid 'adequate', 'steep', 'top-notch', 'high-end', 'prohibitive', 'fatiguing', 'acoustic isolation', 'furthermore').\n\n"
        f"Construct Instructions:\n"
        f"{construct_instructions}"
        f"7. overall: A 1 to 2 sentence natural summary of your thoughts on this {prod_noun}.\n\n"
        f"MANDATORY FORMAT:\n"
        f"Return ONLY a valid JSON object with keys \"PE\", \"EE\", \"HM\", \"SI\", \"PP\", \"BI\", and \"overall\". "
        f"Each value must be 1 to 3 short, simple sentences in very easy everyday English, like an ordinary person answering a university survey. "
        f"Use short common words, like talking to a friend. Give your OWN opinion based on your priorities; do not sound like other buyers. "
        f"Do not say which persona or type of buyer you are. "
        f"Bad (too formal): 'Given its price, I am hesitant to adopt it as a premium device. It lacks the distinctive craftsmanship I expect.' "
        f"Better: 'I think it is too expensive. I would buy it if the quality was better.' "
        f"Real survey answers sound like this (style only, do not copy): "
        f"'Worth the price, but I would not pick it again.' / 'Camera and battery are good, so yes I would buy it.' / "
        f"'Too costly for what it gives. I will look for something cheaper.' Mixed feelings are fine. "
        f"Never use the phrase 'continue using'. "
        f"WORDS ONLY. Do NOT include numerical ratings (e.g. 4/5), scores, or percentages."
    )

    return {
        "persona_id": pid,
        "product_name": product.name,
        "system": INDIAN_SURVEY_PERSONA_SYSTEM_PROMPT,
        "user": user_message,
        "metadata": {
            "evaluation_type": "multi_construct_indian_english",
            "evaluated_constructs": ["PE", "EE", "HM", "SI", "PP", "BI"],
            "profile_source": "data/personas/persona_profiles.json",
        },
    }


def sanitize_invented_facts(text: str, category: str = "", construct: str = "") -> str:
    """Clean out hallucinated personal possessions, ungrounded cliches, robotic phrasing, and price leaks.

    Ensures persona evaluates the product on its own merits without fabricating
    pre-existing personal inventories, study habits, vacuous unsupported claims, or robotic words.
    """
    if not text:
        return text

    # Remove invented ownership sentences: "I already have a great set of headphones...", "I already own a pair of earbuds..."
    cleaned = re.sub(
        r"(?:While\s+|Although\s+|Even\s+though\s+)?I\s+already\s+(?:have|own|use)\s+(?:a\s+|an\s+|my\s+)?(?:great|good|fine|another)?\s*(?:set\s+of|pair\s+of)?\s*(?:headphones|earbuds|camera|console|laptop|smartwatch|watch|tv|tablet|device)[^.,;!?]*[.,;!?]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    # Remove "my current headphones/earbuds/laptop are/is..."
    cleaned = re.sub(
        r"(?:My|my)\s+current\s+(?:headphones|earbuds|smartwatch|camera|console|laptop|tablet|device)\s+(?:are|is|handles|work)[^.,;!?]*[.,;!?]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    # Replace unsupported generic cliches like "offers everything I need" or "suits my needs perfectly"
    cleaned = re.sub(r"\b(?:offers?|provides?)\s+(?:everything|all)\s+I\s+need\b", "provides practical features for daily use", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bsuits?\s+my\s+needs\s+perfectly\b", "performs reliably for daily tasks", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:meets?|satisfies?)\s+all\s+my\s+needs\b", "provides balanced daily functionality", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfits?\s+my\s+needs\s+perfectly\b", "provides dependable functionality", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmatches?\s+my\s+needs\s+well\b", "delivers practical features", cleaned, flags=re.IGNORECASE)

    # Clean invented study/work habits
    cleaned = re.sub(r"\bfor\s+my\s+(?:study|college|studies|class|classes|school)\s+(?:needs|tasks|sessions|work|routine|homework)\b", "for daily media and reading", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfor\s+my\s+work\s+(?:requirements|needs|tasks)\b", "for daily productivity", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmy\s+student\s+budget\b", "an accessible price", cleaned, flags=re.IGNORECASE)

    # Replace robotic, repetitive, or overly formal words with simple everyday English (User Request Item 1)
    cleaned = re.sub(r"\badequate\b", "decent", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bsteep\b", "on the higher side", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btop-notch\b", "great", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bprohibitive\b", "too high", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhigh-end\s+specifications\b", "strong specifications", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhigh-end\s+machine\b", "premium machine", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhigh-end\s+device\b", "capable device", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bhigh-end\b", "premium", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\brefined\s+build\b", "solid build", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bdemanding\s+tasks\b", "heavy daily work", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bwithout feeling fatiguing\b", "without feeling tired", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfatiguing\b", "tiring", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bacoustic isolation\b", "noise isolation", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bmoreover,?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfurthermore,?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bleverage\b", "use", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\butilize\b", "use", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bparamount\b", "important", cleaned, flags=re.IGNORECASE)

    # Clean invented budget/financial assumptions (User Request Item 3)
    cleaned = re.sub(r"\bfor\s+my\s+budget\b", "for the price", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bout\s+of\s+my\s+budget\b", "on the higher side", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bbeyond\s+my\s+(?:current\s+)?budget\b", "on the higher side", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btoo\s+high\s+for\s+my\s+budget\b", "on the higher side", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btoo\s+expensive\s+for\s+my\s+budget\b", "on the higher side", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bfor\s+an\s+entry-level\s+user\b", "for basic daily use", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\ba\s+significant\s+barrier\b", "a key consideration", cleaned, flags=re.IGNORECASE)

    # Clean non-gaming references when the product is not a gaming device (User Request Item 5)
    is_gaming = "gaming" in category.lower() or "console" in category.lower()
    if not is_gaming:
        cleaned = re.sub(r"\bfor\s+gaming\s+and\s+entertainment\b", "for media and entertainment", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bgaming\s+and\s+entertainment\b", "media and entertainment", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\benjoyable\s+for\s+gaming\b", "enjoyable for entertainment", cleaned, flags=re.IGNORECASE)

    # Neutralize robotic "continue using" phrasing into prospective evaluation without forcing positive adoption
    cleaned = re.sub(r"\bI am certain I will continue using this\b", "I will consider adopting this", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcontinue using this\b", "adopt this", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bcontinue using\b", "adopt", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bkeep using this\b", "purchase this", cleaned, flags=re.IGNORECASE)
    if category and not any(k in category.lower() for k in ["phone", "smartphone"]):
        cleaned = re.sub(r"\b(?:smartphone|phone|handset)\b", "product", cleaned, flags=re.IGNORECASE)

    # If this is PE, EE, HM, SI, or PP, strip accidental price complaint clauses so construct remains purely feature-focused
    construct_code = construct.upper().strip()
    if construct_code in ("PE", "EE", "HM", "SI", "PP"):
        cleaned = re.sub(r",?\s*(?:but|however|although)?\s*(?:the\s+)?(?:high\s+)?(?:price|cost)\s+(?:is|feels)\s+(?:a\s+significant\s+barrier|a\s+barrier|too\s+high|a\s+major\s+concern|a\s+concern|a\s+drawback|a\s+deterrent|prohibitive|quite\s+expensive)[^.;!?]*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:^|[.!?]\s*)(?:However,?\s+|Although,?\s+|But,?\s+)?(?:The|the)\s+(?:high\s+)?(?:price|cost)\s+(?:is|feels)\s+[^.!?]+[.!?]", ".", cleaned, flags=re.IGNORECASE)

    # Clean leading lowercase, dangling commas, or repeated periods
    cleaned = re.sub(r"\s+([,.;!?])", r"\1", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = cleaned.strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned or text


def parse_multiconstruct_response(raw_text: str, category: str = "") -> dict[str, str]:
    """Parse JSON or structured construct responses from the LLM output."""
    raw = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fence_match.group(1).strip() if fence_match else raw

    brace_match = re.search(r"(\{.*\})", candidate, re.DOTALL)
    if brace_match:
        try:
            data = json.loads(brace_match.group(1))
            if isinstance(data, dict):
                return {str(k).upper(): sanitize_invented_facts(str(v).strip(), category=category, construct=str(k)) for k, v in data.items() if v}
        except Exception:
            pass

    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return {str(k).upper(): sanitize_invented_facts(str(v).strip(), category=category, construct=str(k)) for k, v in data.items() if v}
    except Exception:
        pass

    res = {}
    for code in ["PE", "EE", "HM", "SI", "PP", "BI", "OVERALL"]:
        pattern = rf'["\']?{code}["\']?\s*:\s*["\']([^"\'\n\r]+(?:(?:\n|\r|\n)[^"\'\n\r]+)*)["\']'
        m = re.search(pattern, raw, re.IGNORECASE)
        if m:
            res[code.upper()] = sanitize_invented_facts(m.group(1).strip(), category=category, construct=code)

    if not res:
        clean_raw = sanitize_invented_facts(raw, category=category, construct="OVERALL")
        res["OVERALL"] = clean_raw
        for code in ["PE", "EE", "HM", "SI", "PP", "BI"]:
            res[code] = sanitize_invented_facts(raw, category=category, construct=code)

    return res


# =============================================================================
# Plain-English survey prompt (used by the dashboard product evaluation)
# =============================================================================
# The long rule-heavy prompt above is kept for reference and its tests, but its own wording
# ("craftsmanship", "premium flagship tier", "steep expense that requires strong justification")
# was being copied into the answers. This prompt is short and written in the style we want back.

SIMPLE_SYSTEM_PROMPT: Final[str] = (
    "You are one particular consumer in India answering a short survey about one product, with your own needs and spending habits. "
    "Answer the way you would naturally type: simple everyday English, short sentences, common words. "
    "No formal, business, marketing or academic language. Your answers do not need to sound polished. "
    "Use words only (TEXT ONLY): NEVER provide numerical ratings, scores, stars or percentages. "
    "Never say what type of buyer or persona you are. Return only valid JSON."
)

SIMPLE_PERSONA_PRIORITIES: Final[dict[str, str]] = {
    "P1": ("You care most about quality: how good the display, camera, build and overall experience feel. "
           "You evaluate whether this device gives enough refinement for its price. "
           "Do not call it a flagship or premium unless the description says so. "
           "A reasonably priced product can still satisfy you if it truly meets your standards; name its limits honestly."),
    "P2": ("You like technology and how things work. When you look at this product you think about its specifications, processor capabilities, RAM, display refresh rate, charging speed, and how fast and capable it is. "
           "You evaluate whether the listed features are actually useful and what the listed hardware should be able to handle. "
           "Mention a spec only if it is written in the description; never make up specs or benchmark scores. "
           "Talk like a knowledgeable tech user."),
    "P3": ("You think about your normal day: whether this product is easy to use, saves you time, works reliably and fits your routine, "
           "and whether someone who is not techy could use it. You value balanced daily utility, good battery endurance, and practical convenience. "
           "You describe things in everyday words, not technical numbers."),
    "P4": ("You watch your money closely. You think about whether the price is affordable for you, whether it has the basic features you need, "
           "and whether it is good value. In the other answers, judge the product itself fairly on its features; your budget only decides the buying answer."),
}

SIMPLE_POSITIVE_LENS: Final[dict[str, str]] = {
    "P1": "When something is genuinely good, say what you like about its build, quality or premium feel.",
    "P2": "When something is genuinely good, say which listed features or specs work well and why they matter to you.",
    "P3": "When something is genuinely good, say how it makes your day easier or simpler.",
    "P4": ("When something is genuinely good, say which essential features work well. "
           "Talk about value for money only in your buying answer, never in the other answers."),
}

# How each persona weighs price. Qualitative on purpose: no persona budget, income or fixed price threshold,
# so the model compares the actual price with the actual product for any category or price range.
SIMPLE_PRICE_STANCE: Final[dict[str, str]] = {
    "P1": ("On price: you have relatively high spending power and can comfortably afford premium and upper-mid-range products, "
           "so a price in that range is not too expensive for you by itself. "
           "You are comfortable paying more when the quality truly justifies it; "
           "you judge it against other premium options, and a low price alone does not win you over if the product feels basic."),
    "P2": ("On price: you have fairly flexible spending and will pay more for genuinely strong specs and capabilities, "
           "but you judge the hardware itself, not the price tag, and strong specs do not justify any price."),
    "P3": ("On price: you have a moderate budget and like sensible spending. You don't need premium extras. "
           "If it covers your daily needs well at a price that feels sensible for what it does, that is a good reason to buy it; "
           "don't hold back just because it isn't top-of-the-line or isn't cheap. A pricier one is worth it if its benefits really help your day."),
    "P4": ("On price: you have limited spare money and compare prices carefully, more than most buyers. "
           "You are careful with money and judge whether the essentials are worth it; you can admit a product is excellent and still "
           "decide it costs too much for you, but you don't reject something only because it costs more."),
}

# How each persona reasons, so answers differ in reasoning and not only in wording.
SIMPLE_REASONING_STYLE: Final[dict[str, str]] = {
    "P1": "You reason by comparing it with other premium options you could pick instead: is it refined and reliable enough to be worth choosing?",
    "P2": "You reason from the specific specs and features listed: what they let it do, and where their limits are.",
    "P3": "You reason from moments in a normal day: whether it would make those simpler, or get in the way.",
    "P4": "You reason by separating what is essential from what is extra, and whether the essentials are covered well.",
}

# Who each persona's social circle is, without any money framing (used for SI so the answer stays about other people).
SIMPLE_SOCIAL_CIRCLE: Final[dict[str, str]] = {
    "P1": "Your friends and colleagues notice polished, high-quality devices, and you care how a device reflects your taste.",
    "P2": "Your friends follow technology and compare specifications, and you often discuss gadgets with them.",
    "P3": "Your family and coworkers want devices that are dependable and simple, and you listen to their everyday experience.",
    "P4": "Your family and close friends share practical tips about which devices work well for them, and their experience matters to you.",
}

SIMPLE_MAIN_CONCERN: Final[dict[str, str]] = {
    "P1": "quality and reliability, and whether it feels like a good experience",
    "P2": "the specs, the features and how well it performs",
    "P3": "whether it is useful and easy for everyday life",
    "P4": "whether it is affordable and gives good value for the money",
}

SIMPLE_QUESTIONS: Final[dict[str, str]] = {
    "PE": "Would this {noun} help you with the things you need to do every day? How useful would it be?",
    "EE": "Would this {noun} be easy for you to set up and use?",
    "HM": "Would you enjoy using this {noun}? Would it be fun or pleasant to use?",
    "SI": "What would your friends, family or colleagues think of you using this {noun}, and would their views matter to you?",
    "PP": ("How well would this {noun} protect your personal information and accounts? "
           "Would you feel safe using it for things involving personal information?"),
    "BI": "Would you actually buy this {noun} at {price}? State your direct purchase decision first (Yes, No, or Unsure), and explain why based on suitability and price.",
}

SIMPLE_CONSTRUCT_FOCUS: Final[dict[str, str]] = {
    "PE": ("Say how it helps you get things done faster, better or with less effort (speed, multitasking, productivity). "
           "If the product information shows a real strength for this, say so plainly and positively. "
           "Mention specific features that matter and note any real limitation. "
           "Describe what the listed specs are designed to support; do not claim real-world lag, slowdowns, heat or benchmark results."),
    "EE": ("Talk about how easy it is to learn, set up and use day to day, in your own words. "
           "If the information supports it, say it is easy and positive. "
           "Focus strictly on interface simplicity, learning curve, and navigation. "
           "Do not confuse fingerprint or security features with ease of use, and do not treat raw specs as ease of use. "
           "Avoid stock phrases like 'user-friendly interface' or 'straightforward UI'."),
    "HM": ("Talk about how it feels to use it, tied to what you personally enjoy: fun, pleasant, annoying or dull, and why. "
           "Focus on enjoyment, entertainment, media consumption, and lifestyle satisfaction. "
           "Do not default to the camera or display every time; explain what makes it pleasant or dull for your lifestyle."),
    "SI": ("Talk about the social side of the choice: how friends, family or colleagues might see you using it, which listed features "
           "they might notice, and whether their views would matter to you. If the listed features give a real reason others would "
           "approve, say so positively. You have not heard their real opinions, so acknowledge that "
           "naturally instead of claiming they recommend it. Never invent recommendations, reviews or popularity, and never say what "
           "a specific friend, family member or colleague said. Keep it about other people, not buying decisions or the price."),
    "PP": ("Talk about how well it would protect your personal information, accounts and privacy, using only security features named in the "
           "product information. A fingerprint sensor or face unlock controls who can unlock the device; it does not by itself prove that "
           "your data, accounts or privacy are fully protected, so say what the information does not show. If no security features are "
           "named, state clearly that none are described. Never invent encryption, certifications, security updates or privacy policies. "
           "Do not talk about physical build or durability, and do not talk about price."),
    "BI": ("Say whether you would buy it and give your reason. "
           "If the product information gives a clear reason to want it, say you would buy it; if it gives a clear reason not to, say that. "
           "Answer with your decision first, then one specific reason, in about 25 to 40 words, "
           "using how well it fits you and, where relevant, the price. "
           "Make it match the buying_decision in your assessment. You may talk about the price here."),
}

SIMPLE_STYLE_EXAMPLES: Final[str] = (
    "STYLE EXAMPLES (style only; do not copy):\n"
    "- Bad: \"Given the processor specifications, I am hesitant to adopt this primary unit.\"\n"
    "- Good PE (~40 words): "
    "\"With 8GB RAM I could keep my work apps and chats open together, and the 120Hz display should make scrolling smoother. "
    "I can't tell from the specs how it copes with long gaming sessions, so that part is unclear to me.\"\n"
    "- Good EE (~35 words): "
    "\"Setting up my accounts looks straightforward and familiar with the standard interface. Navigating the menus feels intuitive for daily use, though learning some deeper custom settings might take a couple of days.\"\n"
    "- Good BI (~30 words): "
    "\"No, the big battery and display sound good on paper, but my current phone already covers what I need, so I would not switch for this one.\"\n"
    "- Good PP (~40 words): "
    "\"The fingerprint sensor would stop others from unlocking it, which helps. But nothing here tells me how my accounts or personal "
    "data are protected beyond that, so I would not assume it is fully secure.\""
)


_SECURITY_FEATURE = re.compile(r"\b(fingerprint( sensor| unlock| scanner| reader)?|face (id|unlock|recognition)|biometric\w*|encrypt\w*|"
                               r"secure folder|knox|security updates?|privacy (features?|dashboard|mode|protection)|passcode|secure enclave)\b", re.I)


def named_security_features(description: str) -> list[str]:
    """Security / unlock features literally named in the product information (used to ground PP answers)."""
    seen = []
    for m in _SECURITY_FEATURE.finditer(description or ""):
        f = m.group(0).strip()
        if f.lower() not in [x.lower() for x in seen]:
            seen.append(f)
    return seen


def build_simple_persona_prompt(product: ProductConcept, persona_id: str, product_context: dict[str, Any]) -> dict[str, Any]:
    """Plain-English prompt: one validated product, six plain questions, then one persona's priorities.

    Everything shared by all four personas comes first and the persona section comes last, so Ollama can reuse its
    cached processing of the shared prefix for the next persona (large time saving on CPU-only machines).
    """
    noun = product_context["noun"]
    pkey = next((p for p in ("P1", "P2", "P3", "P4") if p in (persona_id or "")), "P3")
    price = product_context.get("price_inr")
    price_str = f"₹{int(price):,}" if price else "its price"
    feats = "; ".join(f.name for f in (product.features or []) if getattr(f, "name", ""))
    security = named_security_features(product_context["description"])

    questions = "\n".join(
        f"\"{c}\": {q.format(noun=noun, price=price_str)} Focus: {SIMPLE_CONSTRUCT_FOCUS[c]}" for c, q in SIMPLE_QUESTIONS.items())
    # Everything up to "ABOUT YOU" is identical for all four personas, so Ollama reuses its processed prefix for P2-P4
    # (measured on this CPU-only machine: ~1,300 cached prompt tokens take 0.4 s instead of 34 s). The persona goes last.
    # The price appears only inside the BI question, so the other answers are not primed to talk about it.
    shared = (
        f"THE PRODUCT (the only product you are answering about):\n"
        f"- Name: {product_context['name']}\n"
        f"- Type: {noun}\n"
        f"- Description: {product_context['description']}\n"
        + (f"- Features: {feats}\n" if feats else "")
        + f"- Security or unlock features named: {', '.join(security) if security else 'none (do not assume any)'}\n"
        f"Only talk about this {noun}. Do not talk about any other kind of product.\n\n"
        f"USING THE PRODUCT FACTS:\n"
        f"- Confirmed facts are only what is written above. Use them as your evidence; do not make up specs.\n"
        f"- You may give a careful reading of a spec, worded as what it supports or should do "
        f"(\"the 120Hz display should make scrolling smoother\").\n"
        f"- Never state untested real-world results as fact: no lag, slowdowns, heating, actual battery hours, photo quality, "
        f"fingerprint speed, gaming performance or benchmark scores. If one matters to you, say the information does not show it.\n\n"
        f"QUESTIONS (answer each one separately and only that question):\n{questions}\n"
        f"\"overall\": Your overall feeling about this {noun} in one sentence.\n\n"
        f"HOW TO ANSWER:\n"
        f"- PE, EE, HM, SI and PP: 35 to 55 words each (answers of only 15 to 30 words are too short), written as 3 sentences: "
        f"(1) your view and the listed feature it rests on, (2) what that means for you in your own life, "
        f"(3) a real limitation, or what the information does not show. Do not pad with filler.\n"
        f"- BI: 25 to 40 words. Start with your direct decision (\"Yes\", \"No\" or \"Unsure\"), then your reason based on how well it "
        f"suits you and the price. If unsure, name the specific thing you still need to check.\n"
        f"- PE, EE, HM, SI and PP must never mention price, cost, affordability, discounts, budget or buying. "
        f"Judge PE, EE, HM, SI and PP on the product itself, not on its price; only BI and overall may talk about price or buying.\n"
        f"- Simple everyday words, like one real consumer answering a survey, not a product review or spec sheet.\n"
        f"- For PE, EE, HM, SI and PP: don't open with a one-word reply such as \"Maybe\" or \"Yes\", and don't start with "
        f"\"I think\", \"Given\", \"Considering\", \"While\", \"Overall\", \"As a\", \"The product\", \"It seems\" or \"Friends\", "
        f"and don't start two of your answers the same way.\n"
        f"- Mention a spec only where it matters for that question; don't repeat the same specs in every answer.\n"
        f"- If something you care about is not stated in the product information, say it isn't mentioned instead of guessing. "
        f"Never invent reviews, recommendations or durability.\n"
        f"- Be honest: positive, mixed, unsure or negative are all fine. Hedge only where it really does not decide it. "
        f"A product can be very good and still not right for you to buy.\n\n"
        f"{SIMPLE_STYLE_EXAMPLES}\n\n"
        f"BEFORE ANSWERING, assess the product for yourself in an \"assessment\" object with short notes: \"strengths\", "
        f"\"limitations\" (what falls short of what you expect), \"what_matters_to_me\", \"fits_my_needs\" (yes, no or partly), "
        f"\"price_justified_for_me\" (yes, no or partly), \"comfortable_to_spend\" (yes, no or a stretch), "
        f"\"better_than_my_usual_options\" (yes, no or not sure) and \"buying_decision\" (yes, no or unsure). "
        f"Choose yes or no whenever the information lets you decide; use unsure only if a specific question is still open, and name it in \"still_need_to_check\". "
        f"If the price is not comfortable or not justified for you and nothing specific is left to check, the decision is no. "
        f"Your BI answer must state the same decision as buying_decision.\n\n"
        f"Return ONLY a JSON object with the keys \"assessment\", \"PE\", \"EE\", \"HM\", \"SI\", \"PP\", \"BI\", \"overall\".\n\n"
    )
    user = shared + (
        f"ABOUT YOU:\n"
        f"{SIMPLE_PERSONA_PRIORITIES[pkey]} {SIMPLE_PRICE_STANCE[pkey]}\n"
        f"Your main concern is {SIMPLE_MAIN_CONCERN[pkey]}. Let it shape your answers, and use your own words for it.\n"
        f"{SIMPLE_POSITIVE_LENS[pkey]} {SIMPLE_REASONING_STYLE[pkey]} {SIMPLE_SOCIAL_CIRCLE[pkey]}\n"
        f"Other buyers may feel differently from you."
    )
    return {
        "persona_id": persona_id,
        "product_name": product.name,
        "system": SIMPLE_SYSTEM_PROMPT,
        "user": user,
        "metadata": {"evaluation_type": "simple_plain_english_v1", "evaluated_constructs": list(SIMPLE_QUESTIONS)},
    }
