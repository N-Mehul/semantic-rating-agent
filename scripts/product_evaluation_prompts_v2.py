"""Generation prompts v2 (generation_quality_improvement_v1, Phase 2).

New, separate module. The legacy builder in product_evaluation_prompts.py (used by the dashboard and its tests)
is left untouched. SSR, anchors, encoder, k and T are not touched here: this module only builds text prompts,
sizes the Ollama context window and parses the model's JSON answer.

Fixes (see evaluation/generation_quality_improvement_v1/phase1_generation_audit.md):
  W1 PP restored to protection of personal information (codebook PP01-PP04, instrument working interpretation)
  W2 no forced BI sentence; the persona chooses its own stance
  W3 no shared "pocket-friendly" price line; each persona judges the price from its own survey price tier
  W4 one consistent ownership frame: the survey's "your current smartphone" = the Nova AI Smartphone (hypothetical)
  W5 HM = enjoyment / fun / entertainment (HM01-HM04), kept distinct from PE
  W6 prompt sized to fit an explicit num_ctx, with a recorded token budget
  W7/W8 one short rule list; concerns (including price) are allowed; no positivity default
  W9 each construct in its own clearly separated block (name, definition, survey items, scale, focus, exclusion)
  W10 parse failures leave a construct missing instead of copying the raw text into every construct
  W11 one respondent spec of the persona is drawn (seeded) from respondent_specs_extended_v2.jsonl
"""
from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Final, Sequence

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
CODEBOOK: Final[Path] = ROOT / "docs/survey_instrument/survey_codebook.csv"
PROFILES: Final[Path] = ROOT / "data/personas/persona_profiles.json"
SPECS: Final[Path] = ROOT / "data/persona_generator/respondent_specs_extended_v2.jsonl"
PROMPT_VERSION: Final[str] = "product_eval_v2_2026-11-04"
CONSTRUCT_ORDER: Final[tuple[str, ...]] = ("PE", "EE", "HM", "SI", "PP", "BI")

# Definitions paraphrase the survey instrument / codebook; survey items are read verbatim from the codebook.
CONSTRUCTS: Final[dict[str, dict[str, str]]] = {
    "PE": {"name": "Performance Expectancy",
           "definition": "How useful the phone is for getting everyday tasks done and how much it improves effectiveness and productivity.",
           "focus": "practical usefulness of the listed features (assistant, camera, battery, processing) for tasks that matter to you",
           "exclude": "enjoyment or fun (that is HM), ease of learning (EE), price (BI)"},
    "EE": {"name": "Effort Expectancy",
           "definition": "How easy the phone is to learn, understand and operate without much effort.",
           "focus": "how easy the interface, AI suggestions and setup would be for someone with your technology comfort",
           "exclude": "usefulness or speed (PE), enjoyment (HM)"},
    "HM": {"name": "Hedonic Motivation",
           "definition": "How enjoyable, fun, entertaining and pleasurable using the phone is.",
           "focus": "fun and pleasure: taking and enhancing photos and short videos, casual gaming, entertainment, the smooth display",
           "exclude": "productivity, task usefulness or work/study benefits (that is PE)"},
    "SI": {"name": "Social Influence",
           "definition": "Whether people important to you (family, friends, peers) would support or encourage your using this phone.",
           "focus": "what your family, friends or classmates/colleagues would likely say about you using it, and why",
           "exclude": "your own view of features (PE/EE/HM)"},
    "PP": {"name": "Perceived Protection",
           "definition": "How well the phone protects your personal information and accounts from unauthorized access and misuse, and how much control its privacy and security controls give you.",
           "focus": "privacy and data protection: personal data sent to the cloud versus processed on the device, location-based suggestions, account and payment safety, control over your information",
           "exclude": "physical durability, build quality or hardware reliability (not part of this construct)"},
    "BI": {"name": "Behavioral Intention",
           "definition": "Whether you intend to keep using this phone as your main phone and rely on it going forward.",
           "focus": "your honest intention to keep using it (or not), weighing the features against the price of the phone",
           "exclude": "nothing; price belongs here"},
}
# Scale labels are deliberately NOT shown to the model: the Phase 2 pilot showed that printing them made Qwen
# prefix its answer with "3 Neither agree nor disagree" (a direct rating). Kept here only for documentation.
SCALE_LABELS: Final[str] = "1 Strongly disagree · 2 Disagree · 3 Neither agree nor disagree · 4 Agree · 5 Strongly agree"

SYSTEM_PROMPT_V2: Final[str] = (
    "You are one respondent in an Indian consumer survey about smartphones. Write the way an ordinary person writes in a "
    "survey: simple, natural Indian English, first person, honest.\n"
    "Rules:\n"
    "1. Use only the product facts given. Do not invent features, specifications, incidents, repairs or long usage history.\n"
    "2. Take your own honest position. It can be positive, mixed or negative; mention a real concern when you have one.\n"
    "3. Answer each construct only about that construct.\n"
    "4. Write words only. Never state an answer option, scale point or scale label (for example a number such as 3, or "
    "'Neither agree nor disagree'), and no scores, stars or percentages. Do not name other brands.\n"
    "5. Return valid JSON only."
)


# ------------------------------------------------------------------ data
def survey_items(code: str, path: Path = CODEBOOK) -> list[str]:
    """Verbatim Likert items for a construct (e.g. PP01-PP04) from the original codebook."""
    with path.open(encoding="utf-8") as f:
        return [r["question_text"] for r in csv.DictReader(f) if re.fullmatch(rf"{code}0[1-9]", r["question_id"] or "")]


def load_profile(persona_id: str, path: Path = PROFILES) -> dict[str, Any]:
    profiles = json.loads(path.read_text(encoding="utf-8"))
    return next(p for p in profiles if p["persona_id"] == persona_id or p["persona_id"].startswith(persona_id))


def load_spec(persona_id: str, spec_index: int | None = None, seed: int = 20261104, path: Path = SPECS) -> dict[str, Any]:
    specs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [s for s in specs if s["persona_id"] == persona_id or s["persona_id"].startswith(persona_id)]
    if not mine:
        raise KeyError(f"no respondent spec for {persona_id}")
    return mine[spec_index % len(mine)] if spec_index is not None else random.Random(f"{seed}:{persona_id}").choice(mine)


# ------------------------------------------------------------------ prompt blocks
PERSONA_TIER: Final[dict[str, str]] = {
    "P1": "Premium segment: your phones have usually cost above ₹35,000.",
    "P2": "Mid-range, heavy-use segment: your phones have usually cost ₹10,000-₹35,000 and you use them intensively.",
    "P3": "Mid-range, moderate-use segment: your phones have usually cost ₹10,000-₹35,000.",
    "P4": "Entry-level segment: your phones have usually cost below ₹10,000 (or you preferred not to say).",
}


def persona_block(persona_id: str, spec: dict[str, Any]) -> str:
    prof = load_profile(persona_id)
    key = persona_id[:2]
    themes = ", ".join(t["theme"].replace("_", " ") for t in (prof.get("top_experience_themes_from_Q23_Q25") or [])[:3])
    fields = [("Segment", PERSONA_TIER.get(key, prof.get("defining_rule", ""))),
              ("You", f"{spec.get('gender', '')}, {spec.get('age', '')}, {spec.get('current_status', '')}, {spec.get('area_type', '')}".strip(", ")),
              ("Price you paid for your last phone", spec.get("price_paid")), ("Monthly spending", spec.get("monthly_spending")),
              ("Daily phone use", spec.get("daily_usage")), ("Main uses", spec.get("main_uses")),
              ("Usually replaces phone", spec.get("replacement_frequency")), ("Learns about phones from", spec.get("info_sources")),
              ("What people in your segment talk about most", themes)]
    return "PERSONA (from survey data)\n" + "\n".join(f"- {k}: {v}" for k, v in fields if v not in (None, ""))


def product_block(product: Any) -> str:
    lines = [f"PRODUCT: {product.name} - {product.description}", f"Price: ₹{product.price_inr:,.0f}"]
    lines += [f"- {f.name}: {f.explanation}" for f in product.features]
    return "\n".join(lines)


SCENARIO: Final[str] = ("SCENARIO: The survey asks about \"your current smartphone\". Answer as if the {name} described above were "
                        "your current smartphone. This is a hypothetical: judge it from its listed features and price and your own "
                        "profile; do not describe events that did not happen.")


def construct_block(code: str) -> str:
    c = CONSTRUCTS[code]
    items = "; ".join(survey_items(code))
    return (f"[{code}] {c['name']}\nDefinition: {c['definition']}\nSurvey statements: {items}\n"
            f"Write about: {c['focus']}.\nDo not write about: {c['exclude']}.")


def build_prompt_v2(product: Any, persona_id: str, constructs: Sequence[str] = CONSTRUCT_ORDER,
                    spec: dict[str, Any] | None = None, spec_index: int | None = None) -> dict[str, Any]:
    """One generation call covering `constructs` (one or many), each in its own separated block."""
    codes = [c.upper() for c in constructs]
    bad = [c for c in codes if c not in CONSTRUCTS]
    if bad:
        raise ValueError(f"unsupported constructs: {bad}")
    spec = spec or load_spec(persona_id, spec_index)
    keys = ", ".join(f'"{c}"' for c in codes)
    user = "\n\n".join([
        persona_block(persona_id, spec), product_block(product), SCENARIO.format(name=product.name),
        "CONSTRUCTS (answer each separately)", *[construct_block(c) for c in codes],
        (f"TASK: For each construct, write 2-4 sentences in your own everyday words (not an answer option or scale label) that "
         f"show how much you would agree or disagree with its survey statements, and why, referring to the {product.name} by name "
         f"or as \"this phone\". "
         f"Return a JSON object with exactly these keys: {keys}."),
    ])
    return {"system": SYSTEM_PROMPT_V2, "user": user, "persona_id": persona_id, "constructs": codes,
            "respondent_spec_id": spec.get("synthetic_id"), "prompt_version": PROMPT_VERSION}


# ------------------------------------------------------------------ context window
OUTPUT_TOKENS_PER_CONSTRUCT: Final[int] = 110   # 2-4 sentences + JSON overhead, measured upper range for Qwen 2.5
CHARS_PER_TOKEN_ESTIMATE: Final[float] = 4.2     # calibrated: Phase 2 pilot measured ~4.9 chars/token (Ollama prompt_eval_count,
                                                 # incl. chat template); 4.2 keeps a ~15% over-estimate as a safety margin
CONTEXT_STEPS: Final[tuple[int, ...]] = (2048, 3072, 4096, 6144, 8192)


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN_ESTIMATE) + 1


def plan_context(prompt: dict[str, Any], margin: float = 0.15) -> dict[str, Any]:
    """Choose the smallest standard num_ctx that holds prompt + output budget + safety margin, and record the budget."""
    chat_overhead = 16  # chat template tokens per message
    prompt_tokens = estimate_tokens(prompt["system"]) + estimate_tokens(prompt["user"]) + 2 * chat_overhead
    max_output = OUTPUT_TOKENS_PER_CONSTRUCT * len(prompt.get("constructs", CONSTRUCT_ORDER)) + 40
    need = int((prompt_tokens + max_output) * (1 + margin))
    num_ctx = next((c for c in CONTEXT_STEPS if c >= need), CONTEXT_STEPS[-1])
    return {"prompt_tokens_estimate": prompt_tokens, "max_output_tokens": max_output, "num_ctx": num_ctx,
            "generation_token_budget": num_ctx - prompt_tokens, "required_with_margin": need,
            "truncation_likely": prompt_tokens + max_output > num_ctx}


def context_diagnostic(plan: dict[str, Any], prompt_eval_count: int | None = None, eval_count: int | None = None) -> dict[str, Any]:
    """Merge the plan with the token counts Ollama actually reports for a call."""
    d = dict(plan)
    if prompt_eval_count is not None:
        d["prompt_tokens_actual"] = int(prompt_eval_count)
        d["generation_token_budget_actual"] = plan["num_ctx"] - int(prompt_eval_count)
        d["truncation_likely"] = int(prompt_eval_count) + plan["max_output_tokens"] > plan["num_ctx"]
    if eval_count is not None:
        d["output_tokens_actual"] = int(eval_count)
    return d


def ollama_request(prompt: dict[str, Any], plan: dict[str, Any], timeout_s: float = 300.0) -> dict[str, Any]:
    """Prompt dict in the shape scripts/ollama_utils.generate expects (it already honours num_ctx/max_output_tokens)."""
    return {"system": prompt["system"], "user": prompt["user"], "json": True, "timeout_s": timeout_s,
            "max_output_tokens": plan["max_output_tokens"], "num_ctx": plan["num_ctx"]}


# ------------------------------------------------------------------ parsing (no rewriting of content)
RATING_LEAK: Final[re.Pattern] = re.compile(
    r"\b[1-5]\s*(/|out of)\s*(5|five)\b|\b\d+\s*stars?\b|\b\d{1,3}\s*%|\b(rating|score)\s*[:=]?\s*\d"
    r"|^\s*[1-5]\b"                                                   # answer starting with a scale point ("3 Neither ...")
    r"|\b(strongly agree|strongly disagree|neither agree nor disagree)\b",  # verbatim scale labels
    re.I)


def parse_response_v2(raw: str, constructs: Sequence[str]) -> dict[str, Any]:
    """Return {"responses": {code: text|None}, "missing": [...], "rating_leak": [...], "parse_ok": bool}. Never copies text across constructs."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    data: dict[str, Any] = {}
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    norm = {str(k).strip().upper(): v for k, v in data.items()} if isinstance(data, dict) else {}
    out = {c: (str(norm[c]).strip() or None) if c in norm and norm[c] else None for c in constructs}
    return {"responses": out, "missing": [c for c, v in out.items() if v is None],
            "rating_leak": [c for c, v in out.items() if v and RATING_LEAK.search(v)], "parse_ok": bool(norm)}


# =====================================================================================================================
# Phase 3: one construct per call, data-grounded persona drivers, rating-leak guard (max one corrective retry),
# deterministic quality flags. Nothing here rewrites text or looks at SSR scores.
# =====================================================================================================================
import time  # noqa: E402

PROMPT_VERSION_V3: Final[str] = "product_eval_v3_2026-11-04"
SYSTEM_PROMPT_V3: Final[str] = (
    "You are one respondent in an Indian consumer survey about smartphones. Write the way an ordinary person writes in a "
    "survey: simple, natural Indian English, first person, honest, in your own words.\n"
    "Rules:\n"
    "1. Use only the product facts given. Do not invent features, specifications, incidents, repairs or long usage history.\n"
    "2. Take your own honest position, which may be positive, mixed or negative, and give your own reasons.\n"
    "3. Write only about the one topic you are asked about.\n"
    "4. Words only: never give a number, score, rating, answer option or scale label, and no stars or percentages. "
    "Do not name other brands.\n"
    "5. Return valid JSON only: {\"response\": \"...\"}."
)

_PRICE_NUM = re.compile(r"₹\s*([\d,]+)")


def price_band(price_paid: str | None) -> tuple[float | None, float | None]:
    """'₹10,000–₹20,000' -> (10000, 20000); 'Less than ₹10,000' -> (None, 10000); 'More than ₹60,000' -> (60000, None)."""
    t = price_paid or ""
    nums = [float(n.replace(",", "")) for n in _PRICE_NUM.findall(t)]
    if not nums:
        return (None, None)
    if t.lower().startswith("less than"):
        return (None, nums[0])
    if t.lower().startswith("more than"):
        return (nums[0], None)
    return (nums[0], nums[-1])


def price_position(price: float, price_paid: str | None) -> str:
    """Factual comparison of the product price with what THIS respondent paid last time (no assumed income)."""
    lo, hi = price_band(price_paid)
    p = f"₹{price:,.0f}"
    if lo is None and hi is None:
        return f"You did not say what you paid for your last phone; judge whether {p} is reasonable for what this phone offers."
    if hi is not None and price > hi:
        return f"{p} is more than you paid for your last phone ({price_paid}); decide honestly whether the features justify the extra cost."
    if lo is not None and price < lo:
        return f"{p} is less than you paid for your last phone ({price_paid}); decide whether it feels like enough phone for you."
    return f"{p} is in the same range you paid for your last phone ({price_paid})."


def persona_drivers(persona_id: str, spec: dict[str, Any], product: Any) -> str:
    """Needs, concerns, adoption habits, tech experience and price position from the respondent spec + segment profile.
    Q21/Q26 outcomes are excluded (they are evaluation targets)."""
    prof = load_profile(persona_id)
    seg_tc = prof.get("tech_comfort_mean_1to5")
    tc = spec.get("tech_comfort_1to5")
    neg = next((t["pct_of_usable_answers"] for t in prof.get("top_experience_themes_from_Q23_Q25") or [] if t["theme"] == "negative_issue"), None)
    # Words, not numbers, so the prompt does not prime numeric (rating-like) output.
    seg_up = "; ".join(u["category"].rstrip(".").lower() for u in (prof.get("upgrade_reason") or [])[:2] if u["category"] != "Other")
    eco = str(getattr(product, "brand_or_ecosystem", "") or "")
    os_line = None
    if spec.get("operating_system") and eco and spec["operating_system"].lower() not in eco.lower():
        os_line = f"You currently use {spec['operating_system']}; this phone is {eco}, so switching would be new for you."
    comfort_words = {1: "very low", 2: "low", 3: "moderate", 4: "high", 5: "very high"}
    rel = None if not (tc and seg_tc) else ("higher than" if tc > seg_tc + 0.5 else "lower than" if tc < seg_tc - 0.5 else "about the same as")
    neg_words = None if neg is None else ("fairly often" if neg >= 14 else "sometimes")
    lines = [
        ("Technology experience", f"your comfort with new technology is {comfort_words.get(int(tc), tc)}, {rel} most people in your segment" if tc else None),
        ("What you use a phone for", spec.get("main_uses")),
        ("Why you last changed phones", spec.get("upgrade_reason")),
        ("How often you replace phones", spec.get("replacement_frequency")),
        ("Common reasons in your segment for changing phones", seg_up or None),
        ("Problems", f"people in your segment {neg_words} mention problems with their phones in the survey" if neg_words else None),
        ("Price", price_position(float(product.price_inr), spec.get("price_paid"))),
        ("Platform", os_line),
    ]
    return "WHAT MATTERS TO YOU (from your survey answers)\n" + "\n".join(f"- {k}: {v}" for k, v in lines if v)


def build_construct_prompt_v3(product: Any, persona_id: str, code: str, spec: dict[str, Any] | None = None,
                              spec_index: int | None = None, avoid_openings: Sequence[str] = ()) -> dict[str, Any]:
    """One call = one construct: only that construct's definition and codebook statements are included."""
    code = code.upper()
    if code not in CONSTRUCTS:
        raise ValueError(f"unsupported construct: {code}")
    spec = spec or load_spec(persona_id, spec_index)
    c = CONSTRUCTS[code]
    avoid = [a for a in avoid_openings if a]
    task = ("TASK: In 2-4 sentences, say in your own everyday words how much these statements are true for you with the "
            f"{product.name}, and why, using the reasons that matter to you above. Refer to it by name or as \"this phone\". "
            "Start the way you naturally would; do not open with a stock phrase.")
    if avoid:
        task += " Do not begin with any of these openings already used by other respondents: " + "; ".join(repr(a) for a in avoid) + "."
    parts = [
        persona_block(persona_id, spec), persona_drivers(persona_id, spec, product), product_block(product),
        SCENARIO.format(name=product.name),
        (f"TOPIC: {c['name']}\nDefinition: {c['definition']}\nSurvey statements: {'; '.join(survey_items(code))}\n"
         f"Write about: {c['focus']}.\nDo not write about: {c['exclude']}."),
        task,
    ]
    return {"system": SYSTEM_PROMPT_V3, "user": "\n\n".join(parts), "persona_id": persona_id, "constructs": [code],
            "construct": code, "respondent_spec_id": spec.get("synthetic_id"), "prompt_version": PROMPT_VERSION_V3}


# ------------------------------------------------------------------ rating leakage
RATING_LEAK_V3: Final[list[tuple[str, re.Pattern]]] = [
    ("leading_scale_point", re.compile(r"^\W*[1-5](?:\s*[.):·\-–]|\s+(?:strongly|agree|disagree|neither)\b|\s*$)", re.I)),
    ("scale_label", re.compile(r"\b(strongly agree|strongly disagree|neither agree nor disagree|somewhat agree|somewhat disagree)\b", re.I)),
    ("rate_statement", re.compile(r"\b(?:i(?:'d| would)?\s+(?:rate|give|score)\w*|rated|rating it)\b[^.]{0,25}?\b(?:[1-5]|one|two|three|four|five)\b", re.I)),
    ("label_prefix", re.compile(r"\b(?:rating|answer|likert(?:\s+score)?|score|my answer)\s*[:=]", re.I)),
    ("x_out_of_y", re.compile(r"\b(?:[1-9]|10|one|two|three|four|five)\s*(?:/|out of)\s*(?:5|10|five|ten)\b", re.I)),
    ("stars_or_percent", re.compile(r"\b\d+(?:\.\d+)?\s*(?:%|percent\b|stars?\b)|\b(?:one|two|three|four|five)[ -]stars?\b", re.I)),
]


def detect_rating_leak(text: str | None) -> list[str]:
    t = (text or "").strip()
    return [name for name, rx in RATING_LEAK_V3 if rx.search(t)]


CORRECTIVE_NOTE: Final[str] = ("Your previous answer stated a rating, number or answer option. Rewrite it in plain everyday words only, "
                               "with no number, score, rating or scale label anywhere. Keep your same opinion and reasons.")


def parse_single_v3(raw: str | None) -> dict[str, Any]:
    """{"response": text} -> text (a construct-code key is accepted too). Never rewrites content."""
    m = re.search(r"\{.*\}", raw or "", re.S)
    text, ok = None, False
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and d:
                ok = True
                v = d.get("response") or next((v for v in d.values() if isinstance(v, str) and v.strip()), None)
                text = v.strip() if isinstance(v, str) and v.strip() else None
        except json.JSONDecodeError:
            ok = False
    return {"text": text, "parse_ok": ok}


# ------------------------------------------------------------------ deterministic quality flags
CONSTRUCT_LEXICON: Final[dict[str, re.Pattern]] = {
    "PE": re.compile(r"\b(useful|usefulness|productiv\w*|tasks?|effective\w*|get (?:things|work) done|helps? me|reminders?|schedul\w*|study support|work)\b", re.I),
    "EE": re.compile(r"\b(eas(?:y|ier|ily)|simple|learn\w*|intuitive|clear|understand\w*|confus\w*|complicated|effort\w*|navigat\w*|set ?up|interface)\b", re.I),
    "HM": re.compile(r"\b(enjoy\w*|fun|entertain\w*|pleasur\w*|pleasant|gaming|games?|movies?|videos?|music|photos?|selfies?|exciting|boring)\b", re.I),
    "SI": re.compile(r"\b(friends?|family|parents?|peers?|classmates?|colleagues?|people|others|recommend\w*|encourag\w*|support\w*|opinions?)\b", re.I),
    "PP": re.compile(r"\b(privacy|private|personal (?:data|information|info)|data|secur\w*|safe\w*|protect\w*|cloud|on-device|locally|locations?|tracking|permissions?|accounts?|payments?|banking|misuse|hack\w*|unauthori[sz]ed)\b", re.I),
    "BI": re.compile(r"\b(keep using|continue|stick with|switch\w*|buy|purchase|replace\w*|upgrade|go for|choose|worth|price|cost|money|rely on|next phone|in future)\b", re.I),
}
OFF_CONSTRUCT_CUES: Final[dict[str, re.Pattern]] = {
    "PP": re.compile(r"\b(durab\w*|build quality|sturdy|robust|wear and tear|scratch\w*)\b", re.I),
    "HM": re.compile(r"\b(productiv\w*|tasks? done|work efficiently|schedul\w*|study support)\b", re.I),
    "PE": re.compile(r"\b(fun|enjoy\w*|entertain\w*)\b", re.I),
}
UNSUPPORTED_SPEC: Final[re.Pattern] = re.compile(
    r"\b\d+\s*(?:mah|gb|tb|mp|hz|watt|inch|inches|nm|ghz)\b|\b(?:5g|amoled|oled|snapdragon|dimensity|ip6\d|wireless charging|fast charging|stylus|face unlock|fingerprint)\b", re.I)
OTHER_BRANDS: Final[re.Pattern] = re.compile(r"\b(apple|iphone|samsung|galaxy|xiaomi|redmi|oneplus|vivo|oppo|realme|google|pixel|motorola|nokia|poco|iqoo)\b", re.I)
OTHER_DEVICES: Final[re.Pattern] = re.compile(r"\b(laptop|tablet|smart ?glasses|earbuds|smartwatch|television|tv)\b", re.I)


def opening(text: str | None, n: int = 4) -> str:
    return " ".join(re.findall(r"[a-z']+", (text or "").lower())[:n])


def _ngrams(text: str, n: int = 5) -> set[tuple[str, ...]]:
    w = re.findall(r"[a-z']+", (text or "").lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def quality_flags(text: str | None, code: str, product: Any, spec: dict[str, Any] | None = None,
                  other_texts: Sequence[str] = ()) -> dict[str, Any]:
    """Flags only (never rewrites, never uses SSR). `other_texts` = same-construct responses of the other personas."""
    t = (text or "").strip()
    f: dict[str, Any] = {"empty_or_malformed": not t or len(t.split()) < 8 or not re.search(r"[.!?]['\"]?$", t)}
    if not t:
        return {**f, "rating_leak": [], "issues": ["empty_or_malformed"]}
    first = product.name.split()[0].lower()
    feat_words = {w for feat in product.features for w in re.findall(r"[a-z]{5,}", (feat.name + " " + feat.explanation).lower())}
    f["refers_to_product"] = bool(re.search(rf"\b{re.escape(first)}\b|\bthis (?:phone|smartphone)\b", t, re.I))
    f["mentions_listed_feature"] = bool(feat_words & set(re.findall(r"[a-z]{5,}", t.lower())))
    f["other_device_mentioned"] = bool(OTHER_DEVICES.search(t))
    f["other_brand_mentioned"] = bool(OTHER_BRANDS.search(t))
    hits = {k: len(rx.findall(t)) for k, rx in CONSTRUCT_LEXICON.items()}
    f["construct_cue_hits"] = hits
    f["construct_aligned"] = hits[code] > 0
    others = [k for k in hits if k != code and hits[k] > hits[code]]
    f["dominant_other_construct"] = max(others, key=lambda k: hits[k]) if others else None
    f["off_construct_content"] = bool(OFF_CONSTRUCT_CUES.get(code) and OFF_CONSTRUCT_CUES[code].search(t))
    f["rating_leak"] = detect_rating_leak(t)
    f["unsupported_spec_claims"] = sorted({m.group(0).lower() for m in UNSUPPORTED_SPEC.finditer(t)})
    prices = [float(p.replace(",", "")) for p in _PRICE_NUM.findall(t)]
    f["unsupported_price"] = [p for p in prices if abs(p - float(product.price_inr)) > 1]
    contra = []
    if spec:
        st = (spec.get("current_status") or "").lower()
        if "student" in st and re.search(r"\b(my (?:office|boss|job|company|clients?))\b", t, re.I):
            contra.append("work_context_for_student")
        if "working" in st and re.search(r"\b(my (?:college|classes|exams|professor|hostel))\b", t, re.I):
            contra.append("student_context_for_professional")
        if (spec.get("operating_system") or "").lower() == "android" and re.search(r"\b(my iphone|on ios)\b", t, re.I):
            contra.append("ios_claim_for_android_user")
    if re.search(r"\b(for (?:the )?(?:past|last) (?:few )?(?:weeks?|months?|years?)|since i bought|after (?:a|one|two|six) months?|was hacked|got hacked|repaired)\b", t, re.I):
        contra.append("invented_usage_history_or_incident")
    f["persona_or_history_contradiction"] = contra
    mine = _ngrams(t)
    f["shared_5grams_with_other_personas"] = len({g for o in other_texts for g in (mine & _ngrams(o))})
    f["same_opening_as_other_persona"] = bool(opening(t) and opening(t) in {opening(o) for o in other_texts})
    issues = [k for k in ("other_device_mentioned", "other_brand_mentioned", "off_construct_content", "same_opening_as_other_persona") if f[k]]
    issues += ["empty_or_malformed"] if f["empty_or_malformed"] else []
    issues += ["not_about_product"] if not (f["refers_to_product"] or f["mentions_listed_feature"]) else []
    issues += ["construct_not_addressed"] if not f["construct_aligned"] else []
    issues += [f"possible_contamination:{f['dominant_other_construct']}"] if f["dominant_other_construct"] else []
    issues += ["rating_leak"] if f["rating_leak"] else []
    issues += ["unsupported_claims"] if f["unsupported_spec_claims"] or f["unsupported_price"] else []
    issues += contra
    issues += ["repeated_phrasing_across_personas"] if f["shared_5grams_with_other_personas"] >= 2 else []
    f["issues"] = issues
    return f


# ------------------------------------------------------------------ one construct response (max 1 corrective retry)
def generate_construct_response(generate_fn: Any, model: str, product: Any, persona_id: str, code: str,
                                spec: dict[str, Any] | None = None, avoid_openings: Sequence[str] = (),
                                temperature: float = 0.7, settings: dict | None = None) -> dict[str, Any]:
    """generate_fn = scripts.ollama_utils.generate (injected so tests can stub it); settings = ollama_utils.SETTINGS.
    Returns a metadata record (text, status, every attempt with leak/parse/context info). At most 2 calls."""
    prompt = build_construct_prompt_v3(product, persona_id, code, spec=spec, avoid_openings=avoid_openings)
    plan = plan_context(prompt)
    if settings is not None:
        settings["temperature"] = temperature
    attempts, text, user = [], None, prompt["user"]
    for attempt in (1, 2):
        t0 = time.perf_counter()
        rec, err = generate_fn(model, ollama_request({**prompt, "user": user}, plan), timeout=300.0)
        usage = (rec or {}).get("usage") or {}
        parsed = parse_single_v3(rec["response_text"]) if rec else {"text": None, "parse_ok": False}
        leak = detect_rating_leak(parsed["text"])
        attempts.append({"attempt": attempt, "elapsed_s": round(time.perf_counter() - t0, 1), "error": err,
                         "finish_reason": (rec or {}).get("finish_reason"), "parse_ok": parsed["parse_ok"],
                         "rating_leak": leak, "raw": (rec or {}).get("response_text"),
                         "context": context_diagnostic(plan, usage.get("prompt_tokens"), usage.get("output_tokens"))})
        text = parsed["text"]
        if err or not leak:   # success, or a transport error (reported, not retried)
            break
        user = prompt["user"] + "\n\n" + CORRECTIVE_NOTE   # the single targeted corrective regeneration
    status = ("generation_error" if attempts[-1]["error"] else "parse_failed" if not text
              else "rating_leak_unresolved" if detect_rating_leak(text) else "ok_after_correction" if len(attempts) == 2 else "ok")
    return {"persona_id": persona_id, "construct": code, "respondent_spec_id": prompt["respondent_spec_id"],
            "prompt_version": prompt["prompt_version"], "model": model, "temperature": temperature,
            "text": text, "status": status, "n_calls": len(attempts), "attempts": attempts,
            "prompt": {"system": prompt["system"], "user": prompt["user"]}, "avoid_openings": list(avoid_openings)}
