"""Product-context validation (before generation) and response validation (before SSR scoring).

Two gates sit around the LLM call:

1. ``build_product_context`` runs BEFORE any persona is prompted. It resolves the exact product
   (name, category, product noun, price) from the user's input and refuses to continue when the
   product type is unclear or the name / category / description contradict each other, so that no
   persona has to guess what the product is.
2. ``validate_construct_response`` runs AFTER generation and BEFORE E-SSR / Canonical SSR. A response
   that talks about another kind of product, leaks a numeric rating, mixes constructs, or is too long
   or formal is regenerated with ``build_correction_prompt`` (at most twice) and is flagged, not
   scored, if it still fails.

These checks are lexical and deliberately conservative. They do not judge whether a response
reflects its persona; that is not something a keyword check can verify.
"""
from __future__ import annotations

import re
from typing import Any

from scripts.product_evaluation_prompts import (
    _detect_category,
    validate_and_detect_category,
)

SCORED_CONSTRUCTS = ("PE", "EE", "HM", "SI", "PP", "BI")
MAX_CORRECTION_RETRIES = 2

GENERIC_CATEGORIES = {"", "technology product", "technology", "product", "gadget", "other"}

# Words that identify a kind of product. Bare "watch" and "tv" are left out on purpose because
# "watch videos" and "TV shows" are normal things to say about phones, tablets and laptops.
PRODUCT_FAMILIES: dict[str, str] = {
    "smartphone": r"\b(smartphones?|smart\s*phones?|phones?|mobile\s*phones?|handsets?)\b",
    "laptop": r"\b(laptops?|notebook\s*computers?|macbooks?|chromebooks?|ultrabooks?)\b",
    "tablet": r"\b(tablets?|ipads?)\b",
    "smartwatch": r"\b(smartwatch(es)?|smart\s*watch(es)?|fitness\s*bands?|fitness\s*trackers?|smart\s*rings?)\b",
    "television": r"\b(smart\s*tvs?|televisions?)\b",
    "audio": r"\b(earbuds|earphones|headphones|headsets?|bluetooth\s*speakers?|soundbars?)\b",
    "glasses": r"\b(smart\s*glasses|eyewear|ar\s*glasses)\b",
    "camera": r"\b(security\s*cameras?|cctv|surveillance\s*cameras?)\b",
    "robot": r"\b(robot\s*vacuums?|cleaning\s*robots?|robots?|robotic\s*vacuums?)\b",
    "console": r"\b(gaming\s*consoles?|consoles?|playstation|xbox)\b",
}

NOUN_TO_FAMILY: dict[str, str] = {
    "smartphone": "smartphone", "laptop": "laptop", "tablet": "tablet", "smartwatch": "smartwatch",
    "smart ring": "smartwatch", "smart TV": "television", "wireless earbuds": "audio", "headphones": "audio",
    "Bluetooth speaker": "audio", "smart glasses": "glasses", "security camera": "camera",
    "cleaning robot": "robot", "gaming console": "console",
}

# Accessories and smart devices are normally used with a phone, so a phone mention is allowed
# when the sentence is about pairing, apps or connecting.
PHONE_COMPANION_FAMILIES = {"smartwatch", "audio", "glasses", "camera", "robot", "television", "console", "laptop", "tablet"}
COMPANION_CONTEXT = re.compile(r"\b(pair|pairs|paired|pairing|connect|connects|connected|sync|syncs|synced|app|apps|bluetooth|with my phone|from my phone|on my phone)\b", re.I)

HARDWARE_CATEGORIES = {
    "Smartphone", "Laptop Computer", "Tablet Computer", "Wearable Device", "Television", "Audio Device",
    "Smart Glasses / Wearable AI", "Smart Home Security / Surveillance Camera", "Gaming Console", "Smart Home / Robotics",
}


class ProductContextError(ValueError):
    """The product cannot be identified safely; the user must clarify before generation."""


def _family(noun: str) -> str | None:
    return NOUN_TO_FAMILY.get(noun)


def _mentions(text: str, family: str) -> bool:
    return bool(re.search(PRODUCT_FAMILIES[family], text, re.I))


def other_product_mentions(text: str, noun: str) -> list[str]:
    """Return product families mentioned in ``text`` that are not the evaluated product."""
    own = _family(noun)
    found = []
    for fam in PRODUCT_FAMILIES:
        if fam == own or not _mentions(text, fam):
            continue
        if fam == "smartphone" and own in PHONE_COMPANION_FAMILIES:
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if _mentions(s, "smartphone")]
            if sentences and all(COMPANION_CONTEXT.search(s) for s in sentences):
                continue
        found.append(fam)
    return found


KNOWN_BRANDS = ("samsung", "apple", "iphone", "oneplus", "xiaomi", "redmi", "poco", "vivo", "oppo", "realme", "motorola", "google",
                "pixel", "nokia", "iqoo", "infinix", "tecno", "honor", "huawei", "asus", "lenovo", "dell", "acer", "msi", "sony", "lg",
                "boat", "noise", "jbl", "bose", "amazfit", "garmin", "fitbit", "tapo", "ecovacs", "roborock", "dyson")
_GENERIC_NAME_TOKENS = {"pro", "max", "plus", "lite", "ultra", "mini", "neo", "prime", "phone", "smartphone", "the", "new", "edition"}


def _name_description_mismatch(name: str, description: str) -> str | None:
    """Return the brand the description names when it clearly describes a different product than ``name``.

    Fires only when the description names a known brand that is absent from the product name AND none of the
    product name's distinctive words appear in the description (e.g. name 'reno 1' with a Samsung description).
    """
    n, d = (name or "").lower(), (description or "").lower()
    brands = [b for b in KNOWN_BRANDS if re.search(rf"\b{b}\b", d) and not re.search(rf"\b{b}\b", n)]
    if not brands:
        return None
    tokens = [t for t in re.findall(r"[a-z0-9]+", n) if len(t) >= 3 and t not in _GENERIC_NAME_TOKENS and not re.fullmatch(r"\d+g?", t)]
    if not tokens or any(re.search(rf"\b{re.escape(t)}\b", d) for t in tokens):
        return None
    return brands[0]


def build_product_context(name: str, description: str, category: str = "", price_inr: float | None = None) -> dict[str, Any]:
    """Resolve the exact product once, before generation, or raise ProductContextError."""
    name = (name or "").strip()
    description = (description or "").strip()
    category = (category or "").strip()

    detected_cat, noun, usage_role, eco = validate_and_detect_category(name, description, category)
    if detected_cat == "Technology Product":
        raise ProductContextError(
            f"I could not tell what kind of product '{name}' is. Please set the Product Category "
            f"(for example Smartphone, Laptop Computer, Smartwatch or Cleaning Robot) or say what it is in the description."
        )

    if category.lower() not in GENERIC_CATEGORIES:
        from_name = _detect_category(name, "", "")
        from_category = _detect_category("", category, category)
        if (from_name[0] in HARDWARE_CATEGORIES and from_category[0] in HARDWARE_CATEGORIES
                and from_name[0] != from_category[0]):
            raise ProductContextError(
                f"The product name '{name}' looks like a {from_name[1]}, but the category is '{category}'. "
                f"Please correct the name or the category so every persona evaluates the same product."
            )

    own = _family(noun)
    if own and description:
        others = other_product_mentions(description, noun)
        if others and not _mentions(description, own) and not _mentions(name, own):
            other_noun = next((n for n, f in NOUN_TO_FAMILY.items() if f == others[0]), others[0])
            raise ProductContextError(
                f"The product is set as a {noun}, but the description talks about a {other_noun} and never mentions a {noun}. "
                f"Please update the description (it may still be from a previous product) or the category."
            )

    mismatch = _name_description_mismatch(name, description)
    if mismatch:
        raise ProductContextError(
            f"The product name '{name}' does not appear in the description, which describes a {mismatch.title()} product. "
            f"Please make the name and description refer to the same product (the description may be from a previous product)."
        )

    return {
        "name": name,
        "category": detected_cat,
        "noun": noun,
        "usage_role": usage_role,
        "ecosystem": eco,
        "price_inr": price_inr,
        "description": description,
    }


def product_context_block(ctx: dict[str, Any]) -> str:
    price = f"₹{int(ctx['price_inr']):,}" if ctx.get("price_inr") else "not given"
    return (
        "--- PRODUCT BEING EVALUATED (fixed for every persona and every question) ---\n"
        f"- Product name: {ctx['name']}\n"
        f"- Product type: {ctx['noun']} (category: {ctx['category']})\n"
        f"- Price: {price}\n"
        f"- Every answer must be about this {ctx['noun']}. Do not talk about any other kind of product "
        f"and do not reuse details from other products."
    )


NUMERIC_RATING = re.compile(r"\b[1-5](\.\d)?\s*(/|out of)\s*(5|10)\b|\b\d+(\.\d)?\s*stars?\b|\b\d+\s*%", re.I)
PERSONA_LABEL = re.compile(r"\bas an? (premium|budget|budget-conscious|tech|practical|everyday)\b[^.]{0,30}\b(consumer|user|buyer|enthusiast|person)\b|\bpersona\b|\bP[1-4]\b", re.I)
PRICE_WORDS = re.compile(
    r"(₹|\brs\.?\s*\d|pric(e|es|ed|ey|ing)\b|overpric\w*|cost(s|ly)?\b|expensive|cheap(er)?\b|afford\w*|budget\w*|"
    r"value for money|good value|worth the money|money\b)", re.I)
PURCHASE_OUTSIDE_BI = re.compile(r"\b(would|will|won'?t|wouldn'?t|might|may)\s+(not\s+|definitely\s+|probably\s+)?(buy|purchase)\b", re.I)
BI_INTENT = re.compile(
    r"\b(buy|buying|bought|purchase|purchasing|get it|getting it|go for|pick it|choose it|consider|considering|recommend|spend|"
    r"switch|adopt|order|own it|skip it|skip|pass on|pass|wait|waiting|think about|not for me|for me|"
    r"maybe|perhaps|not sure|unsure|might|may|probably|possibly|depends|if|undecided|hesitant|not really|no way|won'?t|wouldn'?t|would not|will not)\b",
    re.I)
HARD_WORDS = re.compile(
    r"\b(adequate|steep|top-notch|high-end|prohibitive|paramount|furthermore|moreover|leverage|utili[sz]e|robust|seamless(ly)?|"
    r"commendable|substantial|meticulous|craftsmanship|refine(d|ment)|compelling|justifiable|investment|excels?|unparalleled|"
    r"cutting-edge|insurmountable|warrant(s|ed)?|distinctive|exceptional|adoption|adopt(ing)?|hesitant|primary device|flagship-grade|"
    r"commensurate|optimal|consequently|nevertheless|notwithstanding|discerning|aforementioned|ergonomic)\b", re.I)
BANNED_OPENING = re.compile(r"^(given\b|considering\b|while (the|it|this|its)\b|overall\b|as an? \b|i am certain\b)", re.I)
PERFORMANCE_WORDS = re.compile(r"\b(fast\w*|speed\w*|quick\w*|multitask\w*|productiv\w*|efficien\w*|smooth\w*|lag\w*|time|perform\w*|handl\w*|run\w*|load\w*|faster|process\w*|saves?)\b", re.I)
QUOTED_OPINION = re.compile(r"\b(they say|people say|friends? says?|family says?|colleagues? says?|told me|said (that|it|they|he|she)|says it)\b", re.I)
PP_UNSUPPORTED = re.compile(r"\b(durable|durability|waterproof|water-resistant|dust-proof|drop[- ]proof|shockproof|unbreakable|will last|will never|scratch-resistant|lasts? for years|crack-proof)\b", re.I)
_SPEC_PART = r"(battery|display|screen|camera|processor|chip|ram|storage|charging|refresh rate)"
_BUILD_WORD = r"(sturdy|sturdier|durable|solid|tough|rugged|strong|robust|well[- ]built|hold up)"
PP_INFERRED_FROM_SPEC = re.compile(
    rf"\b{_SPEC_PART}\b[^.]{{0,40}}\b(makes?|means?|shows?|suggests?|so)\b[^.]{{0,25}}\b{_BUILD_WORD}\b|"
    rf"\b{_BUILD_WORD}\b[^.]{{0,30}}\b(because of|thanks to|due to|given)\b[^.]{{0,25}}\b{_SPEC_PART}\b", re.I)
SI_SOCIAL_REFERENCE = re.compile(
    r"\b(friends?|family|famil\w+|people|others?|colleagues?|classmates?|peers?|anyone|someone|everyone|nobody|no one|"
    r"reviews?|reviewers?|recommend\w*|opinions?|social|image|impress\w*|show off|status|circle|group|community|online|"
    r"parents?|siblings?|co-?workers?|neighbou?rs?|influenc\w*|sway\w*|word of mouth)\b", re.I)
SI_ASSERTED_SOCIAL = re.compile(
    r"\b((friends?|family|people|colleagues?|classmates?|everyone|others) (recommend|recommended|recommends|suggested|swear by)|"
    r"(is|it's|very|really|quite) popular|best[- ]?sell\w*|trending|"
    r"reviews? (say|says|are|praise\w*|show\w*|rate\w*)|(everyone|most people) (has|have|uses|use|loves?|likes?|bought|buys))\b", re.I)
DIRECT_DECISION = re.compile(r"^\W*(yes|yeah|yep|no|nope|definitely|absolutely)\b", re.I)
BI_POSITIVE = re.compile(
    r"(^\W*(yes|yeah|yep|definitely|absolutely)\b|\b(i'?d|i would|i will|i'?ll)\s+(definitely\s+|probably\s+|happily\s+)?(buy|get|consider|go for|pick)\b|"
    r"\bhappy to buy\b|\bworth buying\b)", re.I)
BI_NEGATIVE = re.compile(
    r"(^\W*(no|nope)\b|\b(won'?t|wouldn'?t|would not|will not|not going to)\b|\bnot (buy|get|purchase)\b|\b(pass|skip)\b|"
    r"\bnot for me\b|\banother (option|phone|model)\b|\bsomething else\b|\blook elsewhere\b|\bother options\b)", re.I)
BI_UNSURE = re.compile(r"\b(maybe|perhaps|unsure|not sure|might|depends|undecided|first check|compare|still deciding|think about)\b", re.I)
BRAND_UI = re.compile(r"\b(one ?ui|coloros|miui|hyperos|oxygenos|funtouch ?os|realme ui|harmonyos|magic ?os|ios|oneui)\b", re.I)
BUILD_EVIDENCE = re.compile(r"\b(build|built|material|metal|aluminium|aluminum|glass|gorilla|frame|plastic|polycarbonate|ip\d\d|"
                            r"water|dust|drop|rugged|durab\w*|sturdy|body|chassis|military|mil-std)\b", re.I)
PP_BUILD_CLAIM = re.compile(r"\b(well[- ]built|well[- ]made|sturdy|solid|durable|tough|robust(?! (protection|security|privacy|data))|premium build|strong build|built to last)\b", re.I)
UNCERTAIN = re.compile(r"\b(can'?t|cannot|not sure|unsure|doesn'?t say|don'?t know|isn'?t (described|stated|mentioned)|not (described|stated|mentioned)|"
                       r"no information|unclear|(isn'?t|not) clear|hard to (tell|judge)|wouldn'?t know|no details?)\b", re.I)


def bi_decision_mismatch(text: str, decision: str | None) -> str | None:
    """Issue text when a BI answer contradicts the persona's own assessed buying decision, else None."""
    if decision is None:
        return None
    t = text or ""
    pos, neg, unsure = bool(BI_POSITIVE.search(t)), bool(BI_NEGATIVE.search(t)), bool(BI_UNSURE.search(t) or HEDGE_OPENING.match(t))
    if decision == "no" and (not neg or HEDGE_OPENING.match(t)):
        return "your assessment decided not to buy; say that clearly with one reason"
    if decision == "yes" and (not pos or neg or HEDGE_OPENING.match(t)):
        return "your assessment decided to buy; say that clearly with one reason"
    # "No, unsure." / "No, ... make me unsure": a yes/no opening followed by uncertainty is not a clear decision.
    if re.match(r"^\W*(yes|yeah|no|nope)\b", t, re.I) and re.search(r"\b(unsure|not sure|undecided)\b", t, re.I):
        return "the answer mixes a yes/no opening with 'unsure'; give one clear decision first (Yes, No or Unsure)"
    if decision == "unsure" and not (unsure and SPECIFIC_CHECK.search(t)):
        return "your assessment was unsure; say what you still need to check before deciding"
    return None


SPECIFIC_CHECK = re.compile(r"\b(check|compare|see|wait (for|until)|test|try|find out|know more|look at|read|confirm|depends on|if)\b", re.I)


def acceptable_bi_decisions(assessment: dict[str, Any] | None, decision: str | None) -> set[str]:
    """Buying decisions consistent with the persona's whole assessment, not only its decision label.

    An 'unsure' label whose own fields already say the price is not comfortable or not justified also supports
    a direct 'no'; one whose fields are all positive also supports a direct 'yes'. 'yes' and 'no' never swap.
    """
    if decision is None:
        return set()
    allowed = {decision}
    if decision == "unsure" and assessment:
        field = lambda k: str(assessment.get(k, "")).strip().lower()
        comfortable, justified, better, fits = (field("comfortable_to_spend"), field("price_justified_for_me"),
                                                field("better_than_my_usual_options"), field("fits_my_needs"))
        # Word-boundary match: "not sure" must not count as "no" (it did, which let a direct No contradict an unsure assessment).
        is_no = lambda f: bool(re.match(r"(no|nope|not really)\b", f))
        is_yes = lambda f: bool(re.match(r"(yes|yeah)\b", f))
        if is_no(comfortable) or is_no(justified) or (is_no(better) and not is_yes(justified)):
            allowed.add("no")
        if is_yes(comfortable) and is_yes(justified) and not is_no(fits):
            allowed.add("yes")
    return allowed


def bi_assessment_mismatch(text: str, assessment: dict[str, Any] | None, decision: str | None) -> str | None:
    """None when the BI answer matches any decision the assessment supports, else the reason it does not."""
    allowed = acceptable_bi_decisions(assessment, decision)
    if not allowed or any(bi_decision_mismatch(text, d) is None for d in allowed):
        return None
    return bi_decision_mismatch(text, decision)


GENERIC_EE = re.compile(r"\b(easy to (set up|setup) and (to )?use|user-friendly interface|straightforward (ui|interface))\b", re.I)
HEDGE_OPENING = re.compile(r"^\W*(maybe|perhaps|possibly|it depends|not sure|i'?m not sure)\b", re.I)
PP_TOPIC = re.compile(r"\b(privacy|private|personal (information|data|info|details|stuff)|my data|data|secur\w*|protect\w*|safe\w*|"
                      r"face id|face unlock|fingerprint|passcode|password|pin|lock\w*|encrypt\w*|accounts?|hack\w*|theft|stolen|"
                      r"unauthori[sz]ed|biometric\w*|payments?|banking)\b", re.I)
SECURITY_EVIDENCE = re.compile(r"\b(face id|face unlock|fingerprint|biometric\w*|encrypt\w*|secur\w*|privacy|passcode|password|"
                               r"lock|security updates?|knox|secure enclave)\b", re.I)
PROTECTION_CLAIM = re.compile(r"\b(protects?|protected|secure|safe|safely|keeps? (my|your) (data|information|info|details) (safe|private|secure))\b", re.I)
PRIVACY_TOPIC = re.compile(r"\b(privacy|private|personal data|my data|your data|data security|encrypt\w*|hack\w*|surveillance|spy\w*)\b", re.I)
STOCK_PHRASES = re.compile(r"\b(given (its|the) (price|cost)|considering (its|the) (price|cost)|while the device has)\b", re.I)
SPEC_TOKEN = re.compile(r"\b\d+(\.\d+)?\s*(gb|tb|mah|hz|mp|megapixels?|inch(es)?|w|nm|ghz)\b|\b(snapdragon|dimensity|helio|exynos|bionic|tensor|amoled|oled|lcd|gorilla glass|ip6[78])\b", re.I)
RUPEE_AMOUNT = re.compile(r"(?:₹|rs\.?\s*|inr\s*)(\d[\d,]*)(\s*k\b)?", re.I)

# Price / purchase leakage outside BI. Broader than PRICE_WORDS: also discounts, deals, offers and buying decisions.
PRICE_LEAK = re.compile(
    r"(₹|\brs\.?\s*\d|\bpric(e|es|ed|ey|ier|ing)\b|overpric\w*|\bcost(s|ly|lier)?\b|expensive|inexpensive|\bcheap\w*|afford\w*|"
    r"budget\w*|value for (the )?money|good value|great value|worth the (money|price|cost)|\bmoney\b|discount\w*|\bon sale\b|"
    r"\b(bank|festive|exchange) offers?\b|\bdeals?\b|\bemi\b|\bpurchas\w*|\bbuy(s|ing)?\b|\bbought\b|spend(ing)?\b|\bspent\b)", re.I)

# --- Product-evidence grounding --------------------------------------------------------------------------------
# A sentence that explicitly says the information does not show something is an honest limitation, not a claim.
EVIDENCE_LIMIT = re.compile(
    r"\b(can'?t|cannot|couldn'?t) (tell|know|say|judge|confirm|be sure)|\b(not|isn'?t|aren'?t|wasn'?t) "
    r"(stated|mentioned|described|listed|given|shown|clear|specified|known)|\bdoesn'?t (say|tell|mention|describe|list|show|specify|prove|guarantee|cover|detail|explain)|"
    r"\b(don'?t|do not) know\b|\bno (details?|information|info|data|mention|word)\b|\bnothing (about|on|is said)\b|\bunclear\b|"
    r"\bwithout (testing|trying|using|reviews?|seeing)|\b(i'?d|i would|would) (need|want|have) to (see|check|test|try|know|read)|"
    r"\bon paper\b|\bunknown\b|\bhard to (tell|judge|say)\b|\bnot sure\b|\bunsure\b", re.I)
# Real-world performance claims that no specification sheet can establish (invented weaknesses or absolute strengths).
UNSUPPORTED_PERFORMANCE = re.compile(
    r"\b(lags?|lagg(y|ing|ed)|stutter\w*|slow(s|ed)? down|slowdowns?|sluggish|bottleneck\w*|overheat\w*|heats? up|"
    r"(gets?|runs?|becomes?) (hot|warm)|throttl\w*|freez(e|es|ing)|crash(es|ing)?|hangs?\b|"
    r"drains? (it |the battery |quickly |fast)?(fast|faster|quickly)|battery drain|"
    r"benchmarks?|antutu|geekbench|\bfps\b|frame ?rates?|bulky|cumbersome|weighs?|weight|heavy to (hold|carry)|"
    r"(battery|phone|it) (might|may|could|would|will) be (a bit |too |quite )?heavy|low[- ]light|struggles? with)\b", re.I)
# "No lag" style absolutes are also unsupported, but "without lag" after a hedge like "should" is a fair reading of specs.
QUALIFIED = re.compile(r"\b(should|likely|probably|expect|expected|suggests?|on paper|seems? (built|designed|meant)|designed for|meant for|"
                       r"supports?|helps?|can help|is made for|is built for)\b", re.I)
ABSOLUTE_CLAIM = re.compile(r"\b(never (lags?|slows?|stutters?|heats?|hangs?|freezes?)|zero lag|no lag at all|lasts? (for )?(two|2|three|3) days|"
                            r"flawless\w*|perfect(ly)? (smooth|fast)|handles? (any|every|all) (game|app|task)s?)\b", re.I)
BATTERY_ENDURANCE = re.compile(r"\bbattery\b[^.!?]{0,40}\b(lasts?|lasted|gets? me through|easily (lasts|covers)|runs? out|dies)\b|"
                               r"\b(lasts?|lasted) (well|long|all day|the whole day|a full day)\b", re.I)
CAMERA_QUALITY = re.compile(r"\b(photos?|pictures?|pics|images?|shots|selfies?|videos?)\b[^.!?]{0,25}\b(are|is|look|looks|come out|turn out)\s+"
                            r"(really |very |quite |a bit |pretty |so )?(great|sharp|amazing|stunning|clear|crisp|good|bad|blurry|grainy|poor|dull)\b|"
                            r"\bcamera quality is\b", re.I)
RAM_CLAIM = re.compile(r"\b(ram|memory)\b[^.!?]{0,30}\b(limits?|limiting|bottleneck\w*|not enough|isn'?t enough|too little|holds? (it )?back|struggles?)\b", re.I)
FINGERPRINT_BEHAVIOUR = re.compile(r"\b(fingerprint|face unlock|biometric)\w*\b[^.!?]{0,40}\b(slow|sluggish|laggy|unreliable|inaccurate|fails?|"
                                   r"instant(ly)?|lightning|super fast|very fast|always works)\b", re.I)


def unsupported_claims(text: str) -> list[str]:
    """Real-world performance claims that the supplied specifications cannot establish.

    A sentence is excused when it explicitly says the information does not show the point (EVIDENCE_LIMIT). Modal guesses
    such as "might lag" are NOT excused: they still invent a weakness the product facts never mention.
    """
    found = []
    for s in re.split(r"(?<=[.!?])\s+", text or ""):
        limited = bool(EVIDENCE_LIMIT.search(s))
        for rx, label in ((UNSUPPORTED_PERFORMANCE, "real-world performance"), (ABSOLUTE_CLAIM, "absolute performance"),
                          (RAM_CLAIM, "RAM limitation"), (FINGERPRINT_BEHAVIOUR, "fingerprint speed or reliability")):
            m = rx.search(s)
            if m and not limited:
                found.append(f"{label}: '{m.group(0).strip()}'")
        for rx, label in ((BATTERY_ENDURANCE, "actual battery life"), (CAMERA_QUALITY, "actual photo quality")):
            m = rx.search(s)
            if m and not limited and not QUALIFIED.search(s):
                found.append(f"{label}: '{m.group(0).strip()}'")
    return found


# --- PP: protection of personal information, accounts and privacy -------------------------------------------------
PP_AUTH = re.compile(r"\b(fingerprint\w*|face (unlock|id|recognition)|biometric\w*|passcode|pin lock|screen lock)\b", re.I)
# Words showing the answer treats authentication as device access only, or names what is not established.
PP_AUTH_SCOPE = re.compile(r"\b(only|just|basic|mainly|simply|unlock\w*|access|lock(s|ing)? the (phone|device|screen)|who (can )?(gets? in|opens?|uses?)|"
                           r"(stops?|keeps?) (others|people|strangers|someone) (out|from)|beyond (that|this|it|unlocking)|"
                           r"not (the same as|enough|everything|full|complete)|by itself|on its own|alone)\b", re.I)
PP_AUTH_OVERCLAIM = re.compile(r"\b(fingerprint|face unlock|biometric)\w*[^.!?]{0,40}\b(seems?|is|feels?|looks?|makes? (it|the phone|my data))\s+"
                               r"(very |quite |really |fairly |pretty )?(secure|safe|protected|private)\b|"
                               r"\b(fingerprint|face unlock|biometric)\w*[^.!?]{0,30}\b(protects?|secures?|keeps?) (all |everything|my (data|information|info|accounts?|privacy))", re.I)
PP_INVENTED_MECHANISM = re.compile(r"\b(encrypt\w*|certif\w*|security (updates?|patch\w*)|software updates?|monthly updates?|privacy polic\w*|"
                                   r"two[- ]factor|2fa|knox|secure folder|app lock\w*|privacy (dashboard|features?|settings|mode|controls?)|"
                                   r"data protection|anti[- ]?(virus|malware|theft)|find my (phone|device)|remote (wipe|lock)|vpn|secure enclave|tee\b)", re.I)
NEGATED = re.compile(r"\b(no|not|isn'?t|aren'?t|doesn'?t|don'?t|without|nothing|none|never|unknown|unclear|can'?t|cannot|if)\b", re.I)


def pp_issues(text: str, source: str) -> list[str]:
    """PP-specific grounding: authentication is not full privacy protection, and no security mechanism may be invented."""
    out = []
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    if PP_AUTH.search(text) and (PP_AUTH_OVERCLAIM.search(text) or not (PP_AUTH_SCOPE.search(text) or EVIDENCE_LIMIT.search(text))):
        out.append("treats the fingerprint/unlock feature as full privacy protection; say it controls who can unlock the phone, "
                   "and that the information does not show how well personal data and accounts are protected")
    invented = sorted({m.group(0).lower() for s in sentences for m in PP_INVENTED_MECHANISM.finditer(s)
                       if m.group(0).lower() not in source and not NEGATED.search(s) and not EVIDENCE_LIMIT.search(s)})
    if invented:
        out.append(f"describes security or privacy protection the product information does not list ({', '.join(invented)}); "
                   f"say it is not stated instead")
    return out


# Minimum words. Below these an answer cannot carry reasoning + feature + implication (PE..PP) or decision + reason (BI).
MIN_WORDS = {"BI": 15}
# Measured on real Qwen 2.5 7B output: a 26-word floor made it pad answers with invented claims (weight, bulk, low-light
# photos), so the hard floor is 20 words; the 35-55 word target is carried by the prompt's three-sentence plan.
MIN_WORDS_DEFAULT = 20

# Words that show a buying-intention answer is reasoning from what this persona cares about.
PERSONA_BI_CUES = {
    "P1": r"\b(quality|camera|performance|feel|build|experience|premium|worth|reliable|smooth|fast|look|price|pric|expensive|cost|money|value|afford|feature)\w*",
    "P2": r"\b(processor|chip|spec|specs|camera|performance|feature|display|screen|battery|tech|fast|speed|5g|ram|storage|price|pric|expensive|cost|money|value|worth)\w*",
    "P3": r"\b(daily|everyday|day|battery|easy|simple|calls?|apps?|use|work|reliable|practical|need|price|cost|worth|money|value|afford|sensible)\w*",
    "P4": r"(₹|\b(price|money|afford|budget|worth|cost|spend|value|cheap|expensive|save|saving)\w*)",
}


_STOPWORDS = set("""a an the and or but so if of to for in on at by with from as is are was were be been it its it's this that these those
i i'm i'd i'll i've me my you your we our they their them he she his her not no yes would could should will can may might do does
did have has had just very really quite more most less some any all also than then too there here what which who how about""".split())
UNLISTED_FEATURE = re.compile(r"\b(fingerprint( unlock| sensor| lock| scanner)?|face (unlock|id|recognition)|iris scan\w*|knox|secure folder|"
                              r"wireless charging|reverse charging|stereo speakers|headphone jack|optical zoom|telephoto|periscope|"
                              r"expandable storage|micro ?sd)\b", re.I)


def _content_words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z']+", (t or "").lower()) if w not in _STOPWORDS and len(w) > 2}


def _word_set(t: str) -> set[str]:
    return set(re.findall(r"[a-z']+", t.lower()))


def _opening(t: str, n: int = 2) -> str:
    return " ".join(re.findall(r"[a-z']+", (t or "").lower())[:n])


def validate_construct_response(text: str, construct: str, ctx: dict[str, Any], persona_id: str | None = None,
                                earlier_answers: list[str] | None = None, extra_prices: list[float] | None = None,
                                sibling_answers: list[str] | None = None, assessment_decision: str | None = None,
                                assessment: dict[str, Any] | None = None) -> list[str]:
    """Return the problems found in one construct response (empty list = valid).

    ``persona_id`` enables the persona check on BI; ``earlier_answers`` are other personas' accepted
    answers to the same question (near-copies and repeated openings are rejected); ``extra_prices`` are
    rupee amounts the persona may legitimately mention; ``sibling_answers`` are this persona's accepted
    answers to the other questions (a repeated opening is rejected).
    """
    t = (text or "").strip()
    if len(t.split()) < 3:
        return ["missing or too short"]
    if len(t.split()) < 6:
        return ["too short; add one short natural reason and keep the same opinion"]
    issues = []
    if len(t.split()) < MIN_WORDS.get(construct, MIN_WORDS_DEFAULT):
        issues.append("too short; " + ("give your decision and your reason in about 25 to 40 words" if construct == "BI" else
                                       "write 2 or 3 full sentences (about 35 to 55 words): your reasoning, the features that matter, "
                                       "what that means for you, and any real limitation"))
    source = f"{ctx.get('description', '')} {ctx.get('name', '')}".lower()
    invented = sorted({m.group(0).lower() for m in SPEC_TOKEN.finditer(t) if m.group(0).lower() not in source})
    if invented:
        issues.append(f"mentions specs that are not in the product description ({', '.join(invented)})")
    allowed = [p for p in [ctx.get("price_inr"), *(extra_prices or [])] if p]
    for m in RUPEE_AMOUNT.finditer(t):
        amount = float(m.group(1).replace(",", "") or 0) * (1000 if m.group(2) else 1)
        if amount and not any(abs(amount - p) <= max(1.0, 0.01 * p) for p in allowed):
            issues.append(f"uses a price (₹{int(amount):,}) that is not the product's price")
            break
    if persona_id and construct == "BI":
        pkey = next((p for p in ("P1", "P2", "P3", "P4") if p in persona_id), None)
        if pkey and not re.search(PERSONA_BI_CUES[pkey], t, re.I):
            issues.append("does not give a reason based on what this buyer cares about")
    # Duplicates are judged on content words, so answers that merely share words like "the", "price" or "for"
    # are not treated as copies.
    for other in earlier_answers or []:
        a, b = _content_words(t), _content_words(other)
        if a and b and len(a & b) / len(a | b) > (0.7 if construct == "BI" else 0.6):
            issues.append("is almost the same as another buyer's answer; give your own reason in your own words")
            break
    # Repeated-structure check: natural shared openings such as "I would ..." are fine; a reused sentence
    # template (same first three words and mostly the same content) or a repeated hedge opening is not.
    compared = list(earlier_answers or []) + list(sibling_answers or [])
    head = _opening(t, 3)
    for other in compared:
        a, b = _content_words(t), _content_words(other)
        if head and head == _opening(other, 3) and a and b and len(a & b) / len(a | b) >= 0.5:
            if construct == "BI" and len(a & b) / len(a | b) < 0.75:
                continue
            issues.append(f"reuses the same sentence pattern as another answer ('{head} ...'); say it your own way")
            break
    # BI must open with a decision word, so a shared "Yes"/"No"/"Unsure"/"Maybe" opening is expected there, not a template.
    if construct != "BI" and HEDGE_OPENING.match(t) and any(HEDGE_OPENING.match(o) for o in compared):
        issues.append("opens with a hedge such as 'Maybe' like another answer; give your view or decision first")
    # "no face unlock" / "the lack of face unlock" names a missing feature honestly; only asserted extras are invented.
    unlisted = sorted({m.group(0).lower() for s in re.split(r"(?<=[.!?])\s+", t) for m in UNLISTED_FEATURE.finditer(s)
                       if m.group(0).lower() not in source
                       and not re.search(r"\b(lack of|no|without|missing|doesn'?t have|isn'?t)\b", s, re.I)})
    if unlisted:
        issues.append(f"mentions a feature the product information does not list ({', '.join(unlisted)})")
    others = other_product_mentions(t, ctx["noun"])
    if others:
        issues.append(f"talks about a different product type ({', '.join(others)}) instead of the {ctx['noun']}")
    if NUMERIC_RATING.search(t):
        issues.append("contains a numeric rating or score")
    if PERSONA_LABEL.search(t):
        issues.append("names a persona label instead of speaking naturally")
    # "continue using my current phone" is a valid BI 'No'; "continue using it/this phone" implies owning the product.
    if re.search(r"\bcontinue using\b(?!\s+(my|our)\s+(current|old|existing|present)\b)", t, re.I):
        issues.append("says 'continue using' although the person does not own the product")
    leak = sorted({m.group(0).lower().strip() for m in PRICE_LEAK.finditer(t)}) if construct != "BI" else []
    if leak:
        issues.append(f"price leakage: mentions price, cost, affordability, discounts or buying ({', '.join(leak)}), "
                      f"which belong only in the buying-intention answer")
    elif construct != "BI" and PURCHASE_OUTSIDE_BI.search(t):
        issues.append("talks about buying, which belongs only in the buying-intention answer")
    claims = unsupported_claims(t)
    if claims:
        issues.append(f"unsupported product claim ({'; '.join(claims)}); the specifications do not show this, so only describe "
                      f"what the listed specs support, or say real-world results are not shown")
    if construct == "PE" and not PERFORMANCE_WORDS.search(t):
        issues.append("does not say how it helps you get things done (speed, multitasking, productivity)")
    if construct == "SI" and QUOTED_OPINION.search(t):
        issues.append("quotes what a specific person said or thinks; use uncertain wording such as 'might' or 'not sure'")
    if construct == "PP":
        # Approved PP (codebook PP01-PP04 and its anchors): protection of personal information, accounts and privacy.
        if not PP_TOPIC.search(t):
            issues.append("does not talk about protecting your personal information, accounts or privacy")
        if PP_BUILD_CLAIM.search(t) or PP_UNSUPPORTED.search(t) or PP_INFERRED_FROM_SPEC.search(t):
            issues.append("talks about physical build or durability; this question is about protecting your personal information")
        elif not SECURITY_EVIDENCE.search(source) and any(PROTECTION_CLAIM.search(s) and not UNCERTAIN.search(s)
                                                          for s in re.split(r"(?<=[.!?])\s+", t)):
            issues.append("claims protection that the product information does not describe; say it isn't stated")
        issues.extend(pp_issues(t, source))
    if construct == "SI":
        if not SI_SOCIAL_REFERENCE.search(t):
            issues.append("does not talk about other people's influence; keep the social answer about others")
        invented = sorted({m.group(0).lower() for m in SI_ASSERTED_SOCIAL.finditer(t) if m.group(0).lower() not in source})
        if invented:
            issues.append(f"states a recommendation, review or popularity the product information does not give ({', '.join(invented)})")
    if construct == "EE" and GENERIC_EE.search(t):
        issues.append("uses a stock phrase such as 'easy to set up and use' or 'user-friendly interface'; say what is easy in your own words")
    if construct != "PP" and PRIVACY_TOPIC.search(t) and not PRIVACY_TOPIC.search(source):
        issues.append("talks about privacy or data, which is not in the product description")
    ui = sorted({m.group(0).lower() for m in BRAND_UI.finditer(t) if m.group(0).lower() not in source})
    if ui:
        issues.append(f"mentions a manufacturer interface that is not in the product information ({', '.join(ui)})")
    if construct == "BI":
        if not (BI_INTENT.search(t) or DIRECT_DECISION.match(t)):
            issues.append("does not say whether you would buy it (a yes, no, maybe or 'not sure' with a reason is fine)")
        mismatch = bi_assessment_mismatch(t, assessment, assessment_decision)
        if mismatch:
            issues.append(mismatch)
        bi_sentences = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        if len(t.split()) > 50 or len(bi_sentences) > 3 or max(len(s.split()) for s in bi_sentences) > 40:
            issues.append("keep the buying answer short: one short sentence or concise statement with your decision first, then your reason (about 25 to 40 words)")
    words = len(t.split())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    longest = max(len(s.split()) for s in sentences)
    if construct != "BI" and (words > 75 or len(sentences) > 4 or longest > 40):
        issues.append("too long; use 2 or 3 sentences (about 35 to 55 words)")
    hard = sorted({m.group(0).lower() for m in HARD_WORDS.finditer(t)})
    if hard:
        issues.append(f"uses formal words ({', '.join(hard)}); use simple everyday words")
    if BANNED_OPENING.match(t) or STOCK_PHRASES.search(t):
        issues.append("uses a stock phrase such as 'Given its price' or 'Considering'; start naturally")
    return issues


CONSTRUCT_QUESTIONS = {
    "PE": "How useful would this {noun} be for you? Talk about how well it does its main job, using its real features.",
    "EE": "How easy would this {noun} be to set up and use?",
    "HM": "Would you enjoy using this {noun}? Would it be fun or pleasant?",
    "SI": "What would friends, family or people around you think about you using this {noun}?",
    "PP": "How well would this {noun} protect your personal information, accounts and privacy? (Focus on personal data and security only, not physical build.)",
    "BI": "Would you buy this {noun} at {price}? Say yes, no or maybe, and give your main reason.",
}


def extract_assessment(raw_text: str) -> dict[str, Any] | None:
    """Pull the persona's private suitability assessment out of the raw JSON reply (None if absent)."""
    import json
    m = re.search(r"\{.*\}", raw_text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    a = next((v for k, v in data.items() if str(k).lower() == "assessment"), None)
    return a if isinstance(a, dict) else None


def stated_buying_decision(assessment: dict[str, Any] | None) -> str | None:
    """'yes', 'no' or 'unsure' from the assessment, else None."""
    d = str((assessment or {}).get("buying_decision", "")).strip().lower()
    if re.match(r"(maybe|perhaps|not sure|undecided|depends)", d):
        return "unsure"
    for key in ("yes", "no", "unsure"):
        if d.startswith(key):
            return key
    return None


def _assessment_reminder(assessment: dict[str, Any] | None) -> str:
    decision = stated_buying_decision(assessment)
    if not decision:
        return ""
    notes = "; ".join(f"{k}: {v}" for k, v in (assessment or {}).items() if k != "buying_decision" and v)[:300]
    return f"Your own assessment of this product: {notes}. Your buying decision was '{decision}'; keep the answer consistent with it.\n"


# Short, construct-specific rules used by the correction prompts (the long main prompt is never resent).
CORRECTION_RULES: dict[str, str] = {
    "PE": "Discuss only how useful it is for getting your tasks done, using the listed specs and what they reasonably allow.",
    "EE": "Discuss only how easy it is to set up, learn and use day to day. Fingerprint or specs are not ease of use.",
    "HM": "Discuss only whether using it would be enjoyable or pleasant for you, and why.",
    "SI": ("Discuss only social influence: how friends, family or colleagues might see you using it and whether their views would matter "
           "to you. You may say which listed features they might notice, but you have not heard their real opinions, so acknowledge that "
           "naturally, never claim they recommend it, and never say what a specific friend, family member or colleague said."),
    "PP": ("Discuss only protection of personal information, accounts and privacy. A fingerprint sensor or face unlock only controls who "
           "can unlock the device; it does not prove full data protection. Do not invent encryption, certifications, security updates, "
           "policies or other protections; if they are not listed, say the information does not show them. No physical build or durability."),
    "BI": "Give your purchase decision first (Yes, No or Unsure), then your reason, using how well it fits you and the price.",
}
PRICE_LEAK_INSTRUCTION = ("Rewrite this {code} answer without any price, cost, affordability, discount, budget or purchase-decision "
                          "references. Discuss only {topic}.")
_TOPIC = {"PE": "usefulness", "EE": "ease of use", "HM": "enjoyment", "SI": "social influence", "PP": "protection of personal information"}
UNSUPPORTED_INSTRUCTION = ("Use only the listed specifications. Do not claim real-world lag, slowdowns, heat, battery life, photo quality, "
                           "fingerprint speed or benchmark results; you may say what a spec is designed to support, or that real-world "
                           "results are not shown.")


def _persona_line(pkey: str, construct: str) -> str:
    """Short persona context for a correction. SI and PP use a money-free line so the persona cannot drift back to price."""
    from scripts.product_evaluation_prompts import SIMPLE_PERSONA_PRIORITIES, SIMPLE_REASONING_STYLE, SIMPLE_SOCIAL_CIRCLE
    if construct == "BI":
        return SIMPLE_PERSONA_PRIORITIES[pkey]
    if construct == "SI":
        return SIMPLE_SOCIAL_CIRCLE[pkey]
    return SIMPLE_REASONING_STYLE[pkey]


def _fix_block(ctx: dict[str, Any], code: str, bad_text: str, issues: list[str], attempt: int) -> str:
    """Question + construct rules + the exact validation failure for one answer."""
    from scripts.product_evaluation_prompts import SIMPLE_QUESTIONS
    price = f"₹{int(ctx['price_inr']):,}" if ctx.get("price_inr") else "its price"
    lines = [f"QUESTION ({code}): {SIMPLE_QUESTIONS[code].format(noun=ctx['noun'], price=price)}", f"Focus for this question: {CORRECTION_RULES[code]}"]
    price_leak = any(i.startswith("price leakage") or "talks about price" in i or "talks about buying" in i for i in issues)
    if price_leak and code != "BI":
        lines.append(PRICE_LEAK_INSTRUCTION.format(code=code, topic=_TOPIC[code]))
    if any(i.startswith("unsupported product claim") for i in issues):
        lines.append(UNSUPPORTED_INSTRUCTION)
    # The final attempt (and any price leak) does not see the rejected wording: echoing it made Qwen paraphrase the same leak.
    if bad_text.strip() and attempt == 1 and not price_leak:
        lines.append(f"Rejected answer: \"{bad_text.strip()}\"")
    elif bad_text.strip():
        lines.append("Write a fresh answer with new wording; do not reuse the rejected one.")
    # The leaked words stay in the saved diagnostics but are not repeated to the model, which tends to reuse words it is shown.
    shown = ["it mentioned price, cost or buying" if i.startswith("price leakage") else i for i in issues]
    lines.append(f"Problem to fix: {'; '.join(shown)}.")
    lines.append("Length: your decision first, then your reason, about 25 to 40 words." if code == "BI" else
                 "Length: 35 to 55 words in 3 sentences: (1) your view and the listed feature it rests on, (2) what that means "
                 "for you, (3) a real limitation or what the information does not show.")
    return "\n".join(lines)


def _fix_header(ctx: dict[str, Any], pkey: str, codes: list[str]) -> str:
    price = f", price ₹{int(ctx['price_inr']):,}" if ctx.get("price_inr") and "BI" in codes else ""
    about = " ".join(dict.fromkeys(_persona_line(pkey, c) for c in codes))
    return (f"PRODUCT FACTS (the only facts you may use): {ctx['name']}, a {ctx['noun']}{price}.\n{ctx['description']}\n\n"
            f"ABOUT YOU: {about}\n"
            f"Write in simple, natural everyday English as one consumer in a survey. No number ratings, and do not say what type of buyer you are.\n\n")


def build_correction_prompt(ctx: dict[str, Any], persona_id: str, construct: str, bad_text: str, issues: list[str], system: str,
                            assessment: dict[str, Any] | None = None, attempt: int = 1) -> dict[str, str]:
    """Short targeted correction for one failed answer: product facts, persona context, construct rules, exact failure.

    For BI, the persona's own private suitability assessment is repeated so the rewrite keeps the same decision.
    ``attempt`` is 1 for the first correction and 2 for the final one (which never sees the rejected wording).
    """
    pkey = next((p for p in ("P1", "P2", "P3", "P4") if p in (persona_id or "")), "P3")
    user = (_fix_header(ctx, pkey, [construct]) + _fix_block(ctx, construct, bad_text, issues, attempt) + "\n"
            + (_assessment_reminder(assessment) if construct == "BI" else "")
            + 'Return ONLY a JSON object: {"answer": "..."}')
    return {"system": system, "user": user}


def build_batch_correction_prompt(ctx: dict[str, Any], persona_id: str, items: list[tuple[str, str, list[str]]], system: str,
                                  assessment: dict[str, Any] | None = None, attempt: int = 1) -> dict[str, str]:
    """One correction prompt for several failed answers of the same persona (same blocks as the single version).

    The price is shown only when the buying question (BI) is among the items.
    """
    pkey = next((p for p in ("P1", "P2", "P3", "P4") if p in (persona_id or "")), "P3")
    codes = [c for c, _, _ in items]
    user = (_fix_header(ctx, pkey, codes)
            + "Rewrite each answer below so it fixes its problem. Answer each question separately and only that question.\n\n"
            + "\n\n".join(_fix_block(ctx, c, t, i, attempt) for c, t, i in items) + "\n"
            + (_assessment_reminder(assessment) if "BI" in codes else "")
            + f"\nReturn ONLY a JSON object with the keys {', '.join(chr(34) + c + chr(34) for c in codes)}.")
    return {"system": system, "user": user}
