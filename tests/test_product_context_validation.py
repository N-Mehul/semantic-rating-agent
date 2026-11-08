"""Product-context validation, response validation, and validated-only SSR scoring (offline: fake LLM + fake embedder)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from canonical_ssr import expected_rating, similarities_to_pmf
from dashboard import evaluation_service as es
from improved_ssr import improved_similarities_to_pmf
from scripts.product_evaluation_prompts import ProductConcept, build_simple_persona_prompt
from scripts.response_validation import (
    MAX_CORRECTION_RETRIES,
    ProductContextError,
    build_product_context,
)
from scripts.response_validation import validate_construct_response as _validate_full


def validate_construct_response(*args, **kwargs):
    """The validator minus the 35-55 / 25-40 word minimum, which these older unit tests predate.

    Their short example answers test other rules (price words, decisions, duplicates, constructs); the length minimum is
    tested on its own in tests/test_reliability_24.py, and the integration tests below run the full validator.
    """
    return [i for i in _validate_full(*args, **kwargs)
            if not i.startswith(("too short; write 2 or 3 full sentences", "too short; give your decision"))]

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ["P1_premium_tier_owner", "P2_midrange_heavy_user", "P3_midrange_moderate_user", "P4_budget_or_undisclosed_price"]

PRODUCTS = {
    "smartphone": ("Nova 5G", "A 5G smartphone with a 6.6-inch 120Hz display, 5000mAh battery and 50MP camera.", "Smartphone", 20000),
    "laptop": ("ZenBook 14", "A thin laptop with an Intel Core i5 processor, 16GB RAM and a 14-inch OLED screen.", "Laptop", 65000),
    "smartwatch": ("Fit Pulse", "A smartwatch with heart-rate tracking, sleep tracking and 10-day battery.", "Smartwatch", 4000),
    "cleaning robot": ("EcoBot Mini", "A robot vacuum that maps rooms with LiDAR and empties its own dust bin.", "Cleaning Robot", 30000),
}


def _prompt(ctx, pid, features=()):
    product = ProductConcept(name=ctx["name"], description=ctx["description"], category=ctx["category"], price_inr=ctx["price_inr"], features=list(features))
    return build_simple_persona_prompt(product, pid, ctx)["user"]


# ---------- 1-4: every persona receives the same, correct product context
@pytest.mark.parametrize("noun", list(PRODUCTS))
def test_each_category_reaches_all_four_personas(noun):
    name, desc, cat, price = PRODUCTS[noun]
    ctx = build_product_context(name, desc, cat, price)
    assert ctx["noun"] == noun
    for pid in PERSONAS:
        user = _prompt(ctx, pid)
        head = user.split("ABOUT YOU")[0]
        assert f"- Name: {name}" in head and f"- Type: {noun}" in head
        assert f"Only talk about this {noun}" in head and f"₹{price:,}" in head


# ---------- 5: switching laptop -> smartphone leaves nothing of the laptop behind
def test_laptop_then_smartphone_has_no_stale_laptop_context():
    lap = build_product_context(*PRODUCTS["laptop"])
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert lap["noun"] == "laptop" and phone["noun"] == "smartphone"
    for pid in PERSONAS:
        user = _prompt(phone, pid)
        assert "ZenBook" not in user and "₹65,000" not in user and "Intel Core i5" not in user
        assert "Only talk about this smartphone" in user
    # a laptop description left over in the form while the category says smartphone is refused
    with pytest.raises(ProductContextError, match="previous product"):
        build_product_context("Nova 5G", PRODUCTS["laptop"][1], "Smartphone", 20000)


def test_frontend_sends_no_hidden_features_and_clears_old_results():
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    assert "AI Personal Assistant" not in js and "features: []," in js
    run = js.split("async function handleRunEvaluation")[1].split("const res = await apiCall")[0]
    assert "state.activeEvaluation = null" in run and "state.form.features = []" in run


# ---------- 7: unclear or contradictory products stop before generation
def test_unclear_product_requires_clarification():
    with pytest.raises(ProductContextError, match="could not tell"):
        build_product_context("Zeta One", "A new gadget for young people.", "Technology Product", 9999)


def test_name_and_category_contradiction_requires_clarification():
    with pytest.raises(ProductContextError, match="looks like a smartwatch"):
        build_product_context("Galaxy Watch Ultra", "Rugged watch with GPS.", "Smartphone", 30000)


def test_phone_named_only_in_description_is_accepted():
    assert build_product_context("one me", "A mid-range smartphone designed for students.", "Technology Product", 27000)["noun"] == "smartphone"


# ---------- 6: construct-specific checks
def test_construct_rules():
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert validate_construct_response("The phone should run my apps fast and the 5000mAh battery should help me through the day.", "PE", phone) == []
    assert any("unsupported product claim" in i for i in validate_construct_response("The phone runs my apps fast and the battery lasts all day.", "PE", phone))
    assert validate_construct_response("I think it is a bit costly. I would buy it if the camera was better.", "BI", phone) == []
    assert any("price" in i for i in validate_construct_response("The screen is smooth and the price is fair.", "PE", phone))
    assert any("buying" in i for i in validate_construct_response("It looks fun and I would buy it.", "HM", phone))
    assert any("would buy" in i for i in validate_construct_response("The phone looks nice and the screen is bright.", "BI", phone))
    assert any("different product" in i for i in validate_construct_response("This laptop has a nice keyboard for typing.", "EE", phone))
    assert any("numeric" in i for i in validate_construct_response("I would give it 4/5 because it works well.", "BI", phone))
    assert any("persona" in i for i in validate_construct_response("As a budget-conscious user, I like the battery a lot.", "PE", phone))
    watch = build_product_context(*PRODUCTS["smartwatch"])
    assert validate_construct_response("It pairs with my phone app easily and setup is quick.", "EE", watch) == []


# ---------- 8-11: generation -> validation -> retry -> validated-only scoring
GOOD = {   # realistic 35-55 word answers grounded only in the Nova 5G description
    "PE": ("The 120Hz display should make scrolling through my messages and maps smoother, and the 5000mAh battery should help me "
           "get through a busy workday. I can't tell how it handles heavy apps from this information."),
    "EE": ("Setting it up looks simple because it is a regular 5G phone with nothing unusual to learn. Moving my contacts and apps "
           "across should take one evening, and the large screen makes the menus easy to read."),
    "HM": ("Scrolling and watching videos on the smooth 6.6-inch screen would feel nice in the evenings. The 50MP camera could make "
           "casual photos with friends more fun, though I would have to try it to know."),
    "SI": ("My friends would probably notice the big smooth screen and the camera when I use it. I have not heard their actual "
           "opinions about this phone, so I would not assume they would all recommend it to me."),
    "PP": ("The description doesn't say how it protects my personal data or accounts, so I'm not sure yet. No unlock or security "
           "features are listed either, so I would check the security settings before using banking apps on it."),
    "BI": "Yes, I would probably buy it because the price feels fair for the big battery, smooth screen and camera that I need every day.",
    "overall": "A decent phone for the money.",
}


class FakeLLM:
    def __init__(self, first: dict, fixes: dict[str, list[str]]):
        self.first, self.fixes, self.calls = first, {k: list(v) for k, v in fixes.items()}, []

    def __call__(self, model, prompt, timeout=None, **kw):
        user = prompt["user"]
        self.calls.append(user)
        if '{"answer": "..."}' in user:
            code = re.search(r"QUESTION \((\w\w)\):", user).group(1)        # the correction prompt names its construct
            return {"response_text": json.dumps({"answer": self.fixes[code].pop(0)})}, None
        first = self.first(user) if callable(self.first) else self.first
        return {"response_text": json.dumps(first)}, None


class FakeEmbedder:
    def _v(self, t):
        rng = np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
        return rng.normal(size=16)

    def embed_text(self, t):
        return self._v(t)

    def embed_texts(self, ts):
        return np.array([self._v(t) for t in ts])


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "EVAL_BASE_DIR", tmp_path)
    monkeypatch.setattr(es, "get_ollama_embedding_adapter", lambda *a, **k: FakeEmbedder())
    svc = es.ProductEvaluationService(ROOT)
    monkeypatch.setattr(svc, "check_ollama_backends", lambda: {"ollama_running": True, "generator_ready": True, "embedding_ready": True})
    return svc


def _run(svc, llm, method="improved", persona="P1"):
    svc.generate_fn = llm
    name, desc, cat, price = PRODUCTS["smartphone"]
    return svc.run_evaluation({"name": name, "description": desc, "category": cat, "price_inr": price,
                               "target_persona": persona, "ssr_method": method, "features": []})


def test_invalid_product_stops_before_any_generation(service):
    llm = FakeLLM(GOOD, {})
    service.generate_fn = llm
    with pytest.raises(ValueError, match="could not tell"):
        service.run_evaluation({"name": "Zeta One", "description": "A new gadget for young people.", "category": "Technology Product", "price_inr": 9999})
    assert llm.calls == []


def test_failed_answer_is_retried_at_most_twice_then_flagged_and_not_scored(service):
    bad = dict(GOOD, BI="This laptop is great for office work and I would buy it.")
    llm = FakeLLM(bad, {"BI": ["The laptop keyboard is nice, I would buy it.", "I want this laptop, I would buy it."]})
    out = _run(service, llm)
    ps = out["persona_summaries"][0]
    v = ps["response_validation"]["BI"]
    assert v["status"] == "flagged" and v["retries"] == MAX_CORRECTION_RETRIES == 2
    assert sum('{"answer": "..."}' in c for c in llm.calls) == 2
    assert "BI" not in ps["construct_scores"] and ps["primary_adoption_rating"] is None
    assert all(a["construct_code"] != "BI" for a in ps["anchor_sets"]) and ps["flagged_constructs"] == ["BI"]


def test_corrected_answer_is_scored_and_display_is_consistent(service):
    fixed_pe = ("The 120Hz display should make my apps feel quicker to scroll, and the 5000mAh battery should help me get through "
                "a full workday. I can't tell how it handles heavy multitasking from these specs.")
    bad = dict(GOOD, PE="The price is low and the battery is good.")
    llm = FakeLLM(bad, {"PE": [fixed_pe]})
    out = _run(service, llm)
    ps = out["persona_summaries"][0]
    assert ps["response_validation"]["PE"]["status"] == "corrected" and ps["response_validation"]["PE"]["retries"] == 1
    assert ps["construct_responses"]["PE"] == fixed_pe
    assert len(ps["anchor_sets"]) == 6 and ps["flagged_constructs"] == []
    for a in ps["anchor_sets"]:
        assert a["response_text"] == ps["construct_responses"][a["construct_code"]]           # shown text = scored text
        pmf = improved_similarities_to_pmf(np.array(a["cosine_similarities"]), k=3, temperature=0.75)
        assert a["pmf_probabilities"] == pytest.approx(pmf, abs=1e-5)                            # PMF from the shown cosines
        stored = np.array(a["pmf_probabilities"])          # saved rounded to 6 decimals; renormalise before re-using it
        assert a["expected_rating"] == pytest.approx(expected_rating(stored / stored.sum()), abs=1e-3)
        assert ps["construct_scores"][a["construct_code"]] == a["expected_rating"]
    assert out["concept"]["product_context"]["noun"] == "smartphone"


def test_canonical_baseline_still_available(service):
    out = _run(service, FakeLLM(GOOD, {}), method="canonical")
    a = next(x for x in out["persona_summaries"][0]["anchor_sets"] if x["construct_code"] == "BI")
    pmf = similarities_to_pmf(np.array(a["cosine_similarities"]))
    assert a["pmf_probabilities"] == pytest.approx(pmf, abs=1e-5) and out["ssr_method"] == "canonical"


PER_PERSONA = {   # four distinct, validator-clean 35-55 word answer sets (spec-generic, so they fit any phone description); keyed on each persona's ABOUT YOU text
    "quality": {
        "PE": ("The smooth display and big battery cover the basics I expect for calls, email and maps on a busy day. Nothing in the "
               "description shows a standout processor, so it may feel ordinary next to the phones I usually choose."),
        "EE": ("Getting started should be painless, since this is a regular 5G phone without unusual extras to learn. I would expect my "
               "accounts and photos to move over in one sitting, and the large screen keeps the settings readable."),
        "HM": ("I would enjoy the smooth screen for reading articles and watching shows in the evening. The camera sounds "
               "decent on paper, but I would want to see real photos before I call it a pleasure to use."),
        "SI": ("Colleagues who notice polished devices would probably see this as a sensible everyday phone rather than a statement "
               "piece. I have not heard what they actually think of it, so I would not assume they would admire it."),
        "PP": ("The description lists no unlock or security features, so I cannot tell how it would guard my email and banking accounts. "
               "Without that information I would keep sensitive work data on a device I already trust."),
        "BI": "No, it covers the basics well, but the build and camera do not sound polished enough for me to choose it over the options I usually pick.",
        "overall": "Decent but ordinary."},
    "technology": {
        "PE": ("A high refresh rate screen and a large battery are solid specs for keeping several apps open while I work. The processor and RAM "
               "are not listed, though, so I can't judge how far it can be pushed with demanding tasks."),
        "EE": ("Setup should be quick for me: I could copy my apps, adjust the display settings and be ready within an hour. "
               "Nothing listed suggests a complicated interface, so there is very little to learn."),
        "HM": ("Playing casual games and browsing on a smooth panel would be a lot of fun for me. The main camera gives me something to "
               "experiment with, although the description says nothing about video features I could play with."),
        "SI": ("My tech friends would look at the spec sheet first and probably notice the smooth display and big battery. I can't say "
               "whether they would rate it highly without hearing their views, so their opinions would matter only once I knew them."),
        "PP": ("I would want details on security patches and app permissions, but the description lists nothing of the kind. Since no "
               "unlock or security features are mentioned, I cannot judge how well my accounts and personal data would be protected."),
        "BI": "Unsure, the display and battery look good, but I need to check which processor it uses and how it compares with similar phones.",
        "overall": "Good specs, missing details."},
    "normal day": {
        "PE": ("On a normal day I mostly message family, pay bills and use maps, and the big battery should save me from charging at "
               "work. The smooth screen should make scrolling quicker and easier, which is nice but not essential for me."),
        "EE": ("My parents could probably learn this phone too, because the description shows nothing complicated about it. Moving my "
               "contacts and setting up chat apps should take an evening, and the large screen makes the icons easy to tap."),
        "HM": ("Watching short videos on the bus with the big screen would make my commute more relaxing. It is pleasant in an "
               "everyday way rather than exciting, and that suits me because I mainly want it to work."),
        "SI": ("My family wants a phone that works without fuss, so they would likely be comfortable seeing me use it. I have not asked "
               "them about this model, so I would not claim they would recommend it, though their everyday experience matters to me."),
        "PP": ("For banking and messages I need my personal details to stay private, but the description does not say how that is "
               "handled. Because no unlock or security features are listed, I would check the settings before trusting it with payments."),
        "BI": "Yes, I would buy it because the big battery and smooth screen cover my daily needs at a price that feels sensible for what it does.",
        "overall": "Practical."},
    "money": {
        "PE": ("The basics I rely on are covered: a big battery for a long day of calls and a decent camera for documents and family "
               "photos. Extras like the smooth screen are nice, but I mostly need it to run my essential apps."),
        "EE": ("Nothing here looks hard to learn, so I could set it up myself without asking my nephew for help. The standard layout of "
               "a 5G phone means my usual apps would be in familiar places from day one."),
        "HM": ("It would be pleasant enough for videos and chatting, and the smooth screen is a nice touch I would appreciate. I would "
               "not use it for gaming much, so the fun side matters less to me than the essentials."),
        "SI": ("My family and close friends share tips about phones that work well for them, so I would ask what they think first. Right "
               "now I have no idea of their real opinions on this model, so I would not assume they would approve."),
        "PP": ("I am not sure how well it would protect my bank and UPI accounts, since the description says nothing about security. "
               "With no unlock or security features listed, I would be careful about storing personal documents on it."),
        "BI": "No, the features are good for daily use, but the price is more than I can comfortably spend right now, so I would wait for a cheaper option.",
        "overall": "Costly for me."},
}


def _per_persona(user):
    about = user.split("ABOUT YOU:")[1][:200]
    return next(v for k, v in PER_PERSONA.items() if k in about)


def test_all_four_personas_get_the_same_product_context(service):
    llm = FakeLLM(_per_persona, {})
    out = _run(service, llm, persona="all")
    main_prompts = [c for c in llm.calls if '{"answer": "..."}' not in c]
    assert len(main_prompts) == 4
    heads = {c.split("ABOUT YOU")[0] for c in main_prompts}
    assert len(heads) == 1 and "- Type: smartphone" in heads.pop()
    assert all(v["status"] == "passed" for r in out["responses"] for v in r["response_validation"].values())
    assert {r["product_context"]["noun"] for r in out["responses"]} == {"smartphone"}
    assert not any(re.search(r"\blaptop", json.dumps(r["construct_responses"]), re.I) for r in out["responses"])


def test_hesitant_buying_answers_are_valid_bi():
    phone = build_product_context(*PRODUCTS["smartphone"])
    for t in ["Maybe, good features for the price.", "I'm not sure yet, I might wait for a sale to get a better price.",
              "Probably not, the price is too high for what I need.", "I would consider it if the price drops a bit.",
              "I need to think about the money for a while before I buy."]:
        assert validate_construct_response(t, "BI", phone, persona_id="P4_budget") == [], t
    assert validate_construct_response("The camera is blue and the box is white.", "BI", phone, persona_id="P4_budget") != []


def test_pricey_is_a_price_word_outside_bi():
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert any("price" in i for i in validate_construct_response("It seems reliable enough, but pricey for basic use.", "PP", phone))
    assert any("price" in i for i in validate_construct_response("It is overpriced for what it does well.", "PE", phone))
    assert validate_construct_response("It isn't clear from the description how it protects my personal data.", "PP", phone) == []


def test_very_short_answers_are_rejected():
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert any("too short" in i for i in validate_construct_response("Yes, it is easy.", "EE", phone))
    assert any("too short" in i for i in validate_construct_response("Maybe.", "BI", phone, persona_id="P3_x"))


def test_stricter_quality_checks():
    phone = build_product_context(*PRODUCTS["smartphone"])
    v = lambda t, c="PE", **k: validate_construct_response(t, c, phone, **k)
    assert any("specs" in i for i in v("The 12GB RAM and Snapdragon chip make it fast."))                 # not in the description
    assert v("The 120Hz display and 5000mAh battery should keep my apps running fast.") == []                     # in the description
    assert any("price" in i for i in v("I would buy it for ₹15,000 but not more.", "BI", persona_id="P4_x"))
    assert v("I would buy it for ₹20,000 if the battery is good.", "BI", persona_id="P4_x") == []
    assert any("formal" in i for i in v("The camera delivers exceptional results for daily photos."))
    assert any("stock phrase" in i for i in v("Considering everything, the camera is quite good for photos."))
    assert any("too long" in i for i in v(" ".join(["The camera is good"] * 12) + "."))
    assert any("what this buyer cares" in i for i in v("Yes, I would buy it soon.", "BI", persona_id="P4_budget"))
    assert any("almost the same" in i for i in v("The camera is good and the battery lasts all day.",
                                                earlier_answers=["The camera is good and the battery lasts all day long."]))


def test_simple_prompt_is_short_plain_and_persona_specific():
    phone = build_product_context(*PRODUCTS["smartphone"])
    prompts = {p: _prompt(phone, p) for p in PERSONAS}
    for u in prompts.values():
        assert "premium flagship tier" not in u and "craftsmanship" not in u and "15 to 30 words" in u
        assert len(u) < 9500        # grew with the evidence-grounding rules; P2-P4 reuse it from the Ollama prompt cache
    assert "comfortable paying more" in prompts[PERSONAS[0]] and "careful with money" in prompts[PERSONAS[3]]
    assert len({u.split("ABOUT YOU:")[1].split("QUESTIONS")[0] for u in prompts.values()}) == 4


def test_p2_and_p3_prompts_differ_in_reasoning_not_just_wording():
    phone = build_product_context(*PRODUCTS["smartphone"])
    p2 = _prompt(phone, "P2_midrange_heavy_user")
    p3 = _prompt(phone, "P3_midrange_moderate_user")
    p2_focus = p2.split("ABOUT YOU:")[1].split("Your main concern")[0]
    p3_focus = p3.split("ABOUT YOU:")[1].split("Your main concern")[0]
    assert "how fast and capable it is" in p2_focus and "how fast and capable it is" not in p3_focus
    assert "your normal day" in p3_focus and "your normal day" not in p2_focus
    assert "not technical numbers" in p3_focus
    assert "Maybe" in p2 and "one-word reply" in p2                       # opening rule is present for every persona
    assert "Your main concern is the specs, the features and how well it performs" in p2
    assert "Your main concern is whether it is useful and easy for everyday life" in p3


def test_all_personas_keep_the_same_product_block_after_persona_change():
    phone = build_product_context(*PRODUCTS["smartphone"])
    heads = {_prompt(phone, p).split("ABOUT YOU")[0] for p in PERSONAS}
    assert len(heads) == 1 and "Only talk about this smartphone" in heads.pop()


def test_user_specifications_are_the_only_spec_source(service):
    llm = FakeLLM(_per_persona, {})
    service.generate_fn = llm
    out = service.run_evaluation({"name": "Nova 5G", "category": "Smartphone", "price_inr": 20000,
                                  "description": "A mid-range smartphone with a good camera and battery.",
                                  "specifications": "6.6-inch 120Hz display; 5000mAh battery", "features": []})
    ctx_desc = out["concept"]["product_context"]["description"]
    assert "Specifications: 6.6-inch 120Hz display; 5000mAh battery" in ctx_desc
    mains = [c for c in llm.calls if '{"answer": "..."}' not in c]
    assert len(mains) == 4 and all("5000mAh battery" in c for c in mains)
    assert all("Nova 5G" in c and "₹20,000" in c for c in mains)


def test_missing_price_is_refused_before_generation(service):
    llm = FakeLLM(GOOD, {})
    service.generate_fn = llm
    with pytest.raises(ValueError, match="price in rupees"):
        service.run_evaluation({"name": "Nova 5G", "category": "Smartphone", "description": "A mid-range smartphone with a good camera.", "features": []})
    assert llm.calls == []


def test_page_has_no_hidden_product_defaults():
    JS = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    form = JS.split("function freshProductForm() {")[1].split("ssr_method")[0]
    for hidden in ("mid-range smartphone designed", "Students & young professionals", "Android", "25000", "Technology Product\""):
        assert hidden not in form, hidden
    assert 'name: ""' in form and "price_inr: null" in form and "specifications: \"\"" in form
    assert 'id="categorySuggestions"' in JS or "categorySuggestions" in JS
    assert "Specifications" in JS and "state.form.specifications" in JS


def test_persona_opening_rule_covers_i_think():
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert "don't start with \"I think\"" in _prompt(phone, "P2_midrange_heavy_user")


def test_pp_rejects_price_build_talk_and_unsupported_protection():
    phone = build_product_context(*PRODUCTS["smartphone"])        # no security features in this description
    v = validate_construct_response
    assert any("price" in i for i in v("My data feels safe enough, but the price is fair for it.", "PP", phone))
    assert any("physical build" in i for i in v("It feels sturdy and well built in my hand.", "PP", phone))
    assert any("physical build" in i for i in v("It is durable and will last for years.", "PP", phone))
    assert any("does not describe" in i for i in v("It keeps my personal data completely safe.", "PP", phone))
    assert v("I can't tell how well it protects my personal information, since it isn't mentioned.", "PP", phone) == []


def test_si_rejects_quoted_opinions_but_allows_uncertainty():
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert any("quotes" in i for i in validate_construct_response("My family says it looks modern and good.", "SI", phone))
    assert any("quotes" in i for i in validate_construct_response("They say it is a good phone for students.", "SI", phone))
    assert validate_construct_response("My friends might like the camera, but I'm not sure they would care much.", "SI", phone) == []


def test_pe_requires_a_performance_explanation():
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert any("get things done" in i for i in validate_construct_response("It is good for daily use.", "PE", phone))
    assert validate_construct_response("It opens my apps quickly, so I can finish my tasks faster.", "PE", phone) == []


def test_correction_prompt_carries_the_construct_focus():
    from scripts.response_validation import build_correction_prompt
    phone = build_product_context(*PRODUCTS["smartphone"])
    fix = build_correction_prompt(phone, "P3_midrange_moderate_user", "SI", "They say it is great.", ["quotes what a specific person said"], "sys")
    assert "Focus for this question" in fix["user"] and "never say what a specific friend" in fix["user"]


def test_each_construct_has_its_own_focus_in_the_prompt():
    phone = build_product_context(*PRODUCTS["smartphone"])
    u = _prompt(phone, "P2_midrange_heavy_user")
    assert "Focus: Say how it helps you get things done" in u
    assert "Focus: Talk about the social side of the choice" in u
    assert "Focus: Talk about how well it would protect your personal information" in u
    assert "Focus: Say whether you would buy it" in u


M35_DESC = ("The Samsung Galaxy M35 5G is a mid-range smartphone designed for users who need a balance of performance, entertainment, "
            "communication, and everyday productivity. It supports daily activities such as browsing the internet, using social media, "
            "attending online classes, watching videos, taking photographs, and running multiple applications. The phone is suitable for "
            "students, working professionals, technology enthusiasts, and budget-conscious consumers. Users may evaluate its performance, "
            "convenience, reliability, entertainment experience, and overall suitability for their needs.")
AUDIENCE_WORDS = ("budget", "conscious", "consumer", "student", "professional", "enthusiast", "suitab", "overall", "everyday",
                  "productivity", "user", "need", "persona", "premium", "practical", "tech")


def test_genuine_product_features_are_retained():
    from scripts.product_evaluation_prompts import extract_features_from_concept
    feats = extract_features_from_concept("Nova 5G", "Smartphone", "A 5G phone with a 6.6-inch 120Hz display, 5000mAh battery and 50MP camera.")
    names = " | ".join(f.name for f in feats)
    assert "6.6-Inch" in names and "120Hz" in names and "50Mp Camera" in names
    assert any("5000mAh" in f.explanation for f in feats)                       # the spec is kept whole, not split at the decimal


def test_audience_and_generic_phrases_are_excluded():
    from scripts.product_evaluation_prompts import extract_features_from_concept
    feats = extract_features_from_concept("Samsung Galaxy M35 5G", "Smartphone", M35_DESC)
    assert feats == []                                                            # nothing in this description is a real attribute
    assert extract_features_from_concept("X", "Smartphone", "Budget-conscious consumers and students buy this for their needs.") == []


def test_persona_wording_never_becomes_a_feature_and_nothing_is_invented():
    from scripts.product_evaluation_prompts import extract_features_from_concept
    for desc in ["A mid-range smartphone for everyday use.", "Designed for technology enthusiasts who want a good value.",
                 "Suitable for working professionals and students with overall suitability for their needs."]:
        feats = extract_features_from_concept("Nova", "Smartphone", desc)
        assert feats == [], desc
    feats = extract_features_from_concept("Nova", "Smartphone", "This model has a 5000mAh battery and a bright screen.")
    assert "AI-Powered Automation" not in [f.name for f in feats]                 # 'model' no longer triggers the software branch
    assert any("5000mAh" in f.explanation for f in feats)


def test_all_personas_receive_the_same_corrected_product_context(service):
    llm = FakeLLM(_per_persona, {})
    service.generate_fn = llm
    out = service.run_evaluation({"name": "Samsung Galaxy M35 5G", "category": "Smartphone", "price_inr": 19999,
                                  "description": M35_DESC, "features": []})
    assert out["concept"]["features"] == []
    mains = [c for c in llm.calls if '{"answer": "..."}' not in c]
    assert len(mains) == 4
    heads = {c.split("ABOUT YOU")[0] for c in mains}
    assert len(heads) == 1                                                         # same product block for every persona
    block = heads.pop()
    assert "Features:" not in block and not any(w in block for w in ("Budget-Conscious Consumers", "Overall Suitability"))


POSITIVE_CUE = {
    "PE": "say so plainly and positively", "EE": "say it is easy and positive", "HM": "say so",
    "SI": "say so positively", "PP": "say so positively", "BI": "say you would buy it",
}
POSITIVE_SAMPLE = {
    ("P1", "PE"): "It runs my apps smoothly and helps me finish my work faster.",
    ("P2", "PE"): "The processor runs my apps smoothly, so multitasking feels fast.",
    ("P3", "PE"): "It makes my calls and messages quick, so my day runs smoother.",
    ("P4", "PE"): "It runs my basic apps smoothly, so the essentials work well.",
    ("P1", "EE"): "Setting it up was quick, and the menus are simple to use.",
    ("P2", "EE"): "Setup was quick and the settings are easy to find.",
    ("P3", "EE"): "It was easy to set up, and my family can use it without help.",
    ("P4", "EE"): "The setup was quick, and the basic controls are simple.",
    ("P1", "HM"): "It is a pleasure to use, and the screen looks great for videos.",
    ("P2", "HM"): "It is fun to use, and the screen is smooth for games and videos.",
    ("P3", "HM"): "I enjoy using it for videos and music, and it feels pleasant.",
    ("P4", "HM"): "It is enjoyable to use, and the screen is pleasant for videos.",
    ("P1", "SI"): "People I respect would probably approve, since it is a solid choice.",
    ("P2", "SI"): "My classmates who follow tech would likely approve of the features.",
    ("P3", "SI"): "My family would probably approve, since it is easy for everyone.",
    ("P4", "SI"): "Friends who shop carefully would probably approve of it, since it covers the basics.",
    # Positive but correctly scoped: biometric unlock controls access; it is not proof of full data protection (PP rule).
    ("P1", "PP"): "Face ID means only I can unlock it, which I like, though the description doesn't say more about protecting my data.",
    ("P2", "PP"): "The fingerprint unlock and Face ID stop others from getting into my accounts on the phone, which I like a lot.",
    ("P3", "PP"): "With Face ID only I can open it, so I feel better about banking on it, although nothing else about privacy is stated.",
    ("P4", "PP"): "Fingerprint unlock keeps strangers out of my phone, which is good, but that alone doesn't tell me my data is protected.",
    ("P1", "BI"): "I would buy it, the build quality feels premium enough for me.",
    ("P2", "BI"): "I would buy it, the specs and performance are a good fit.",
    ("P3", "BI"): "I would buy it, it is easy for my daily needs.",
    ("P4", "BI"): "I would buy it, the price is fair for the essentials it gives.",
}
PERSONA_IDS = {"P1": "P1_premium_tier_owner", "P2": "P2_midrange_heavy_user", "P3": "P3_midrange_moderate_user", "P4": "P4_budget_or_undisclosed_price"}


def test_positive_guidance_reaches_every_persona_and_construct():
    phone = build_product_context(*PRODUCTS["smartphone"])
    for short, pid in PERSONA_IDS.items():
        user = _prompt(phone, pid)
        for code, cue in POSITIVE_CUE.items():
            assert cue in user, (short, code)
        assert "Hedge only where it really does not" in user                       # no reflex neutrality
        assert "Never invent reviews, recommendations" in user                     # no fabricated positivity
    lens = {short: _prompt(phone, pid).split("Your main concern")[1].split("\n")[1] for short, pid in PERSONA_IDS.items()}
    assert len(set(lens.values())) == 4                                            # four distinct positive lenses
    assert "only in your buying answer" in lens["P4"]                              # value talk stays out of other constructs


@pytest.mark.parametrize("short,code", sorted(POSITIVE_SAMPLE))
def test_genuinely_positive_answers_pass_validation(short, code):
    phone = build_product_context(*PRODUCTS["smartphone"])
    if code == "PP":   # positive protection claims are only valid when the product information names security features
        phone = build_product_context("Nova 5G", "A 5G smartphone with Face ID, fingerprint unlock and a 5000mAh battery.", "Smartphone", 20000)
    text = POSITIVE_SAMPLE[(short, code)]
    issues = validate_construct_response(text, code, phone, persona_id=PERSONA_IDS[short])
    assert issues == [], (short, code, issues)


def test_positive_answers_do_not_carry_value_talk_outside_bi():
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert any("price" in i for i in validate_construct_response("It runs my basic apps smoothly and is good value.", "PE", phone, persona_id="P4_x"))
    assert any("price" in i for i in validate_construct_response("It feels solid, and it is worth the money.", "PP", phone, persona_id="P4_x"))


# ---------- dynamic persona-specific suitability (no fixed budgets or price thresholds)
DYNAMIC_PRODUCTS = {
    "strong phone": ("Apex Pro", "A smartphone with a fast flagship processor, 12GB RAM, a 120Hz OLED display and a 50MP camera.", "Smartphone", 49999),
    "basic phone": ("Lite 4G", "A basic smartphone with a 6.5-inch display and a 5000mAh battery.", "Smartphone", 13999),
    "laptop": PRODUCTS["laptop"],
    "smartwatch": PRODUCTS["smartwatch"],
    "cleaning robot": PRODUCTS["cleaning robot"],
}


@pytest.mark.parametrize("key", list(DYNAMIC_PRODUCTS))
def test_price_reasoning_is_dynamic_with_no_persona_budget(key):
    from scripts.product_evaluation_prompts import SIMPLE_PRICE_STANCE
    name, desc, cat, price = DYNAMIC_PRODUCTS[key]
    ctx = build_product_context(name, desc, cat, price)
    abouts = set()
    for pid in PERSONAS:
        user = _prompt(ctx, pid)
        amounts = {int(a.replace(",", "")) for a in re.findall(r"₹([\d,]+)", user)}
        assert amounts == {price}                                       # only the product's own price; no persona budget or threshold
        assert "usually spend" not in user
        assert SIMPLE_PRICE_STANCE[pid[:2]] in user
        about = user.split("ABOUT YOU:")[1].split("QUESTIONS")[0]
        abouts.add(about)
        if ctx["noun"] != "smartphone":
            assert not re.search(r"\bphones?\b", about, re.I)            # persona reasoning is category-neutral
    assert len(abouts) == 4                                             # four distinct persona perspectives on the same product


@pytest.mark.parametrize("key", list(DYNAMIC_PRODUCTS))
def test_same_product_block_and_quality_vs_buying_rule_for_every_persona(key):
    ctx = build_product_context(*DYNAMIC_PRODUCTS[key])
    prompts = [_prompt(ctx, pid) for pid in PERSONAS]
    assert len({p.split("ABOUT YOU")[0] for p in prompts}) == 1
    for p in prompts:
        assert "Judge PE, EE, HM, SI and PP on the product itself, not on its price" in p
        assert "A product can be very good and still not right for you to buy" in p
        assert "say it isn't mentioned instead of guessing" in p


def test_quality_and_purchase_intention_stay_separate_in_validation():
    strong = build_product_context(*DYNAMIC_PRODUCTS["strong phone"])
    basic = build_product_context(*DYNAMIC_PRODUCTS["basic phone"])
    v = validate_construct_response
    # budget persona: praises a strong product's quality, then declines to buy it at its price
    assert v("The fast processor runs every app smoothly, so I get my work done quicker.", "PE", strong, persona_id="P4_budget") == []
    assert v("The specs are excellent, but ₹49,999 is more than I'm comfortable spending, so I won't buy it.", "BI", strong, persona_id="P4_budget") == []
    # premium persona: a cheap basic product is affordable but not good enough
    assert v("It's cheap at ₹13,999, but the build feels too basic for me, so I wouldn't buy it.", "BI", basic, persona_id="P1_premium") == []
    # practical persona: the same basic product meets daily needs
    assert v("I would buy it, it covers my daily calls and apps well for ₹13,999.", "BI", basic, persona_id="P3_practical") == []
    # tech persona: weak hardware judged on the hardware, without price
    assert v("The basic display is fine, but I can't tell from the specs how it handles heavy apps.", "PE", basic, persona_id="P2_tech") == []
    assert any("unsupported product claim" in i for i in v("The basic display is fine, but it would lag when I run heavy apps.", "PE", basic, persona_id="P2_tech"))
    # price never enters the quality constructs
    for code in ("PE", "EE", "HM", "SI", "PP"):
        assert any("price" in i for i in v("It works well, and it is worth the money at ₹49,999.", code, strong, persona_id="P4_budget")), code


def test_missing_specs_are_not_invented_for_any_category():
    v = validate_construct_response
    basic = build_product_context(*DYNAMIC_PRODUCTS["basic phone"])
    laptop = build_product_context(*DYNAMIC_PRODUCTS["laptop"])
    assert any("specs" in i for i in v("The 12GB RAM keeps my apps running fast.", "PE", basic))
    assert any("specs" in i for i in v("The 32GB RAM makes multitasking run fast.", "PE", laptop))
    assert v("The 16GB RAM makes multitasking run fast.", "PE", laptop) == []          # stated in the laptop description


def test_ssr_settings_and_anchors_are_unchanged():
    import canonical_ssr, improved_ssr
    from survey_ssr_bridge import get_construct_anchors
    assert improved_ssr.DEFAULT_IMPROVED_K == 3 and improved_ssr.DEFAULT_IMPROVED_TEMPERATURE == 0.75
    assert canonical_ssr.PAPER_EPSILON == 0.0 and canonical_ssr.PAPER_TEMPERATURE == 1.0
    for code in ("PE", "EE", "HM", "SI", "PP", "BI"):
        assert len(get_construct_anchors(code).statements) == 5


# ---------- diversity, SI/PP grounding and repeated openings
def test_four_personas_have_distinct_reasoning_styles():
    from scripts.product_evaluation_prompts import SIMPLE_REASONING_STYLE
    phone = build_product_context(*PRODUCTS["smartphone"])
    styles = {}
    for pid in PERSONAS:
        user = _prompt(phone, pid)
        assert SIMPLE_REASONING_STYLE[pid[:2]] in user
        styles[pid[:2]] = SIMPLE_REASONING_STYLE[pid[:2]]
    assert len(set(styles.values())) == 4
    assert "other premium options" in styles["P1"] and "specs and features listed" in styles["P2"]
    assert "normal day" in styles["P3"] and "essential from what is extra" in styles["P4"]
    user = _prompt(phone, PERSONAS[0])
    assert "\"The product\", \"It seems\" or \"Friends\"" in user and "don't start two of your answers the same way" in user
    assert "don't repeat the same specs in every answer" in user


def test_repeated_openings_are_rejected_across_and_within_personas():
    phone = build_product_context(*PRODUCTS["smartphone"])
    v = validate_construct_response
    first = "Maybe, the camera is good but I want to see more first."
    assert not any("hedge" in i or "pattern" in i for i in v(first, "BI", phone, persona_id="P1_premium"))   # first use is fine
    second = "Maybe, the battery is good enough for my daily needs."
    # BI must open with a decision word, so a shared "Maybe"/"Yes"/"No" opening is allowed there (common-opening exemption) ...
    assert not any("hedge" in i for i in v(second, "BI", phone, persona_id="P3_x", earlier_answers=[first]))
    # ... but a repeated hedge opener is still rejected in the other constructs
    assert any("hedge" in i for i in v("Maybe the menus are easy enough to learn.", "EE", phone,
                                       earlier_answers=["Maybe it is simple to set up for me."]))
    assert not any("hedge" in i or "pattern" in i for i in v("For my daily needs the battery is good enough, so I'd buy it.", "BI", phone,
                                                             persona_id="P3_x", earlier_answers=[first]))
    # a reused sentence template is rejected within one persona
    assert any("sentence pattern" in i for i in v("It seems quick to open my apps and the battery lasts all day.", "PE", phone,
                                                   sibling_answers=["It seems quick to open my apps and the screen looks good all day."]))


def test_natural_shared_openings_are_not_rejected():
    phone = build_product_context(*PRODUCTS["smartphone"])
    v = validate_construct_response
    p1 = "I would buy it, the camera quality feels good enough for me."
    p3 = "I would buy it because it handles my daily calls and apps without trouble."
    assert v(p3, "BI", phone, persona_id="P3_practical", earlier_answers=[p1]) == []          # both start 'I would': still fine
    assert not any("pattern" in i or "hedge" in i for i in v("It seems quick to set up, and the menus look simple.", "EE", phone,
                                                             sibling_answers=["It seems fast at opening my apps every day."]))


def test_si_rejects_invented_social_evidence_and_off_topic_answers():
    phone = build_product_context(*PRODUCTS["smartphone"])
    v = validate_construct_response
    for t in ["My friends recommend it to everyone I know.", "It is very popular with people my age.",
              "Reviews say it is the best phone around.", "Everyone has one of these in my class."]:
        assert any("states a recommendation" in i for i in v(t, "SI", phone)), t
    assert any("other people's influence" in i for i in v("The screen is bright and the battery lasts all day.", "SI", phone))
    assert v("I haven't heard what others think of it yet, so it wouldn't sway me much.", "SI", phone) == []
    assert v("My family might not care which one I pick, as long as it works for me.", "SI", phone) == []


def test_pp_stays_on_personal_information_protection():
    phone = build_product_context(*PRODUCTS["smartphone"])
    v = validate_construct_response
    assert any("personal information" in i for i in v("The battery lasts all day and the screen is bright.", "PP", phone))
    assert any("physical build" in i for i in v("The big battery makes it feel sturdy and strong.", "PP", phone))
    assert not any("privacy or data" in i for i in v("I'm not sure how private my data would be, as nothing is said.", "PP", phone))
    assert any("privacy or data" in i for i in v("My data would stay private, and it runs apps fast.", "PE", phone))   # privacy outside PP


def test_ee_rejects_the_stock_phrase_and_constructs_keep_their_meaning():
    phone = build_product_context(*PRODUCTS["smartphone"])
    v = validate_construct_response
    assert any("stock phrase" in i for i in v("It looks easy to set up and use for most people.", "EE", phone))
    assert v("Setting it up took only a few minutes, and the menus felt familiar.", "EE", phone) == []
    assert any("get things done" in i for i in v("It is nice to hold and looks good.", "PE", phone))   # PE keeps its meaning
    assert any("price" in i for i in v("Watching videos is fun, and it is cheap too.", "HM", phone))  # price stays out of HM


def test_same_product_allows_different_buying_decisions_per_persona():
    basic = build_product_context(*DYNAMIC_PRODUCTS["basic phone"])
    v = validate_construct_response
    assert v("For ₹13,999 it covers what I need and saves me money, so I would buy it.", "BI", basic, persona_id="P4_budget") == []
    assert v("Its build feels too basic for what I expect, so I would skip it.", "BI", basic, persona_id="P1_premium") == []


def test_service_retries_an_answer_that_repeats_a_hedge_opening(service):
    bad = dict(GOOD, EE=("Maybe it is simple to set up, and the menus look clear to me. Moving my contacts and apps across should "
                         "take one evening, and the large screen keeps the settings easy to read."),
               HM=("Maybe it would be fun for videos and music at night on the smooth screen. The camera could make casual photos "
                   "with friends more enjoyable, though I would have to try it to know."))
    llm = FakeLLM(bad, {"HM": ["Watching videos on the smooth screen at night sounds relaxing and fun to me. The camera could make "
                               "weekend photos with friends more enjoyable, though I would have to try it to know for sure."]})
    out = _run(service, llm)
    v = out["persona_summaries"][0]["response_validation"]
    assert v["EE"]["status"] == "passed"                                    # first hedge opening is allowed
    assert v["HM"]["status"] == "corrected" and v["HM"]["retries"] == 1      # the repeated one is retried
    assert "hedge" in v["HM"]["previous_attempts"][0]["issues"][0]


# ---------- persona suitability, private assessment and BI alignment diagnostics
def test_persona_buying_stances_are_balanced():
    from scripts.product_evaluation_prompts import SIMPLE_PERSONA_PRIORITIES, SIMPLE_PRICE_STANCE
    assert "A reasonably priced product can still satisfy you" in SIMPLE_PERSONA_PRIORITIES["P1"]     # P1 does not auto-reject mid-range
    assert "strong specs do not justify any price" in SIMPLE_PRICE_STANCE["P2"]                        # P2 judges hardware, not price
    assert "that is a good reason to buy it" in SIMPLE_PRICE_STANCE["P3"]                              # P3 buys what meets daily needs
    assert "isn't top-of-the-line" in SIMPLE_PRICE_STANCE["P3"]
    assert "don't reject something only because it costs more" in SIMPLE_PRICE_STANCE["P4"]           # P4 weighs value, not price alone


def test_bi_scenarios_validate_for_each_persona():
    mid = build_product_context("Samsung Galaxy M35 5G", "A 5G smartphone with a 6000mAh battery, a 120Hz AMOLED display and a 50MP camera.",
                                "Smartphone", 19999)
    strong = build_product_context(*DYNAMIC_PRODUCTS["strong phone"])
    v = validate_construct_response
    # P3: meets daily needs -> clear yes; misses them -> clear no
    assert v("It handles my calls and apps all day and lasts till night, so I would buy it.", "BI", mid, persona_id="P3_x") == []
    assert v("It can't get me through a full day of calls, so I would not buy it.", "BI", mid, persona_id="P3_x") == []
    # P1: reasonably priced but genuinely good enough
    assert v("The AMOLED display and camera feel good enough for me, so I'd buy it despite the mid-range build.", "BI", mid, persona_id="P1_x") == []
    # P2: strong specs, but not at any price
    assert v("The specs are strong, but not worth ₹49,999 to me, so I'll pass.", "BI", strong, persona_id="P2_x") == []
    # P4: expensive yet worth it when the benefits justify it
    assert v("₹19,999 is a stretch, but the big battery and display make it worth the money, so I'd buy it.", "BI", mid, persona_id="P4_x") == []


def test_prompt_asks_for_a_private_assessment_before_answers():
    phone = build_product_context(*PRODUCTS["smartphone"])
    from scripts.product_evaluation_prompts import ProductConcept, build_simple_persona_prompt, validate_prompt_for_ssr
    prompt = build_simple_persona_prompt(ProductConcept(name=phone["name"], description=phone["description"], category=phone["category"],
                                                        price_inr=phone["price_inr"], features=[]), "P3_midrange_moderate_user", phone)
    u = prompt["user"]
    assert "BEFORE ANSWERING, assess the product for yourself" in u and "\"buying_decision\" (yes, no or unsure)" in u
    assert "keys \"assessment\", \"PE\"" in u and "Make it match the buying_decision in your assessment" in u
    assert validate_prompt_for_ssr(prompt) == []
    assert "₹" not in u.split("ABOUT YOU:")[1].split("QUESTIONS")[0]                       # no persona budget numbers


def test_assessment_is_kept_out_of_answers_and_scoring(service):
    answers = dict(GOOD, assessment={"strengths": "battery, camera", "limitations": "none stated", "what_matters_to_me": "daily use",
                                     "price_justified_for_me": "yes", "buying_decision": "yes"})
    out = _run(service, FakeLLM(answers, {}))
    r = out["responses"][0]
    assert "ASSESSMENT" not in r["construct_responses"] and "assessment" not in r["construct_responses"]
    assert r["suitability_assessment"]["buying_decision"] == "yes"
    ps = out["persona_summaries"][0]
    assert sorted(a["construct_code"] for a in ps["anchor_sets"]) == sorted(["PE", "EE", "HM", "SI", "PP", "BI"])


def test_bi_alignment_flag_is_diagnostic_only():
    from dashboard.evaluation_service import _bi_alignment_flag
    assert "review semantic alignment" in _bi_alignment_flag("BI", "yes", 2.4)
    assert "review semantic alignment" in _bi_alignment_flag("BI", "no", 4.1)
    assert _bi_alignment_flag("BI", "yes", 4.1) is None and _bi_alignment_flag("BI", "unsure", 2.0) is None
    assert _bi_alignment_flag("PE", "yes", 1.5) is None and _bi_alignment_flag("BI", None, 1.5) is None


def test_bi_correction_keeps_the_persona_decision_and_ee_stock_phrases_are_caught():
    from scripts.response_validation import build_correction_prompt, extract_assessment, stated_buying_decision
    phone = build_product_context(*PRODUCTS["smartphone"])
    a = extract_assessment('{"assessment": {"strengths": "battery", "buying_decision": "Yes, it fits"}, "BI": "x"}')
    assert stated_buying_decision(a) == "yes" and extract_assessment("not json") is None
    fix = build_correction_prompt(phone, "P3_x", "BI", "Maybe.", ["too short"], "sys", assessment=a)
    assert "Your buying decision was 'yes'" in fix["user"]
    assert "buying decision" not in build_correction_prompt(phone, "P3_x", "PE", "x", ["too short"], "sys", assessment=a)["user"]
    for t in ["It has a user-friendly interface that anyone can learn.", "A straightforward UI makes it simple for me."]:
        assert any("stock phrase" in i for i in validate_construct_response(t, "EE", phone)), t


# ---------- reno 1 diagnosis: product context, feature extraction, interface references
RENO_RUN_DESC = ("The Samsung Galaxy M35 5G is a mid-range smartphone designed for users who want a balance of performance, entertainment, "
                 "connectivity and everyday usability. It offers a Super AMOLED display, high refresh rate, capable processor, high-capacity "
                 "battery and a camera with optical image stabilisation.\nSpecifications: - Display: 6.6-inch Super AMOLED\n- Refresh rate: 120Hz\n"
                 "- Processor: Exynos 1380\n- RAM: 8GB\n- Battery: 6000mAh\n- Operating system: Android with Samsung One UI")


def test_name_and_description_for_different_products_are_refused():
    with pytest.raises(ProductContextError, match="does not appear in the description"):
        build_product_context("reno 1", RENO_RUN_DESC, "Smartphone", 21000)            # the saved reno 1 input
    assert build_product_context("Samsung Galaxy M35 5G", RENO_RUN_DESC, "Smartphone", 21000)["noun"] == "smartphone"
    assert build_product_context("Galaxy M35", RENO_RUN_DESC, "Smartphone", 21000)["noun"] == "smartphone"
    assert build_product_context("one me", "A mid-range smartphone designed for students.", "Smartphone", 27000)["noun"] == "smartphone"


def test_amoled_and_switch_do_not_trigger_tv_or_console_features():
    from scripts.product_evaluation_prompts import extract_features_from_concept
    names = [f.name for f in extract_features_from_concept("Samsung Galaxy M35 5G", "Smartphone", RENO_RUN_DESC)]
    assert "Vibrant 4K Display" not in names and "Smart Streaming Hub" not in names        # the invented TV features in the saved run
    names2 = [f.name for f in extract_features_from_concept("Nova", "Smartphone", "A phone that lets you switch between apps with a 5000mAh battery.")]
    assert not any("Graphics" in n or "Gameplay" in n for n in names2)


def test_interface_references_must_come_from_the_product_information():
    plain = build_product_context("Nova 5G", "A 5G smartphone with a 6.6-inch 120Hz display and a 5000mAh battery.", "Smartphone", 20000)
    m35 = build_product_context("Samsung Galaxy M35 5G", RENO_RUN_DESC, "Smartphone", 21000)
    assert any("manufacturer interface" in i for i in validate_construct_response("Setup is quick thanks to Samsung One UI menus.", "EE", plain))
    assert not any("manufacturer interface" in i for i in validate_construct_response("Setup is quick thanks to Samsung One UI menus.", "EE", m35))


def test_context_reaches_every_persona_prompt_for_the_corrected_product():
    m35 = build_product_context("Samsung Galaxy M35 5G", RENO_RUN_DESC, "Smartphone", 21000)
    heads = {_prompt(m35, pid).split("ABOUT YOU")[0] for pid in PERSONAS}
    assert len(heads) == 1 and "Samsung Galaxy M35 5G" in heads.pop()


# ---------- BI: direct decisions, agreement with the assessment, length
DIRECT_BI = [
    ("P3_x", "no", "No, the price is too high for the features I need."),                       # the rejected P3 answer from the run
    ("P3_x", "no", "No, the price is too high for my needs."),
    ("P4_x", "no", "No, it is too expensive for the features I need."),                          # direct no + price reason
    ("P1_x", "no", "I wouldn't buy it because the performance doesn't meet my expectations."),   # direct no + performance reason
    ("P2_x", "no", "I would pass because the performance doesn't meet my expectations."),
    ("P4_x", "no", "No, I would choose another option with better value for money."),
    ("P3_x", "yes", "Yes, I would buy it because it meets my everyday needs at a sensible price."),   # direct yes + practical benefit
    ("P2_x", "yes", "I would consider buying it because its performance and features meet my expectations."),  # yes + technical benefit
    ("P2_x", "yes", "Yes, I would consider purchasing it because the specs and performance suit me."),
    ("P1_x", "unsure", "I'm still unsure because I need to compare its camera with other options."),  # genuine unsure + reason
    ("P4_x", "unsure", "I might buy it, but I would first check whether the price is worth it."),
]


@pytest.mark.parametrize("pid,decision,text", DIRECT_BI)
def test_direct_bi_decisions_pass_validation(pid, decision, text):
    phone = build_product_context(*PRODUCTS["smartphone"])
    assert validate_construct_response(text, "BI", phone, persona_id=pid) == [], text
    assert validate_construct_response(text, "BI", phone, persona_id=pid, assessment_decision=decision) == [], (decision, text)


def test_bi_contradicting_the_assessment_is_rejected():
    phone = build_product_context(*PRODUCTS["smartphone"])
    v = validate_construct_response
    assert any("decided not to buy" in i for i in v("Yes, I would buy it because it meets my everyday needs.", "BI", phone,
                                                     persona_id="P3_x", assessment_decision="no"))
    vague = "Maybe, the price seems high for the features, but the camera and battery might be worth it."   # P1's answer from the run
    assert any("decided to buy" in i for i in v(vague, "BI", phone, persona_id="P1_x", assessment_decision="yes"))
    assert any("decided not to buy" in i for i in v(vague, "BI", phone, persona_id="P1_x", assessment_decision="no"))
    assert any("was unsure" in i for i in v("No, it is too expensive for the features I need.", "BI", phone,
                                             persona_id="P4_x", assessment_decision="unsure"))


def test_bi_must_be_one_short_sentence():
    phone = build_product_context(*PRODUCTS["smartphone"])
    ok_bi = ("I would buy it because the battery is big and the screen is smooth and the camera is decent, and the price feels fair "
             "for what I get out of it every day.")                              # one 33-word sentence: inside the 25-40 word BI target
    assert validate_construct_response(ok_bi, "BI", phone, persona_id="P3_x") == []
    long_bi = ok_bi[:-1] + " because I mostly use my phone for calls, maps, music and photos of my family when we travel together on weekends."
    assert any("keep the buying answer short" in i for i in validate_construct_response(long_bi, "BI", phone, persona_id="P3_x"))


def test_p1_assessment_and_bi_guidance_are_evidence_based():
    from scripts.product_evaluation_prompts import SIMPLE_CONSTRUCT_FOCUS, SIMPLE_PERSONA_PRIORITIES
    user = _prompt(build_product_context(*PRODUCTS["smartphone"]), PERSONAS[0])
    assert "\"limitations\" (what falls short of what you expect)" in user
    assert "name its limits honestly" in SIMPLE_PERSONA_PRIORITIES["P1"]
    assert "your decision first, then one specific reason" in SIMPLE_CONSTRUCT_FOCUS["BI"]


def test_bi_correction_is_retried_to_match_the_assessment(service):
    answers = dict(GOOD, assessment={"strengths": "camera", "limitations": "build feels basic", "buying_decision": "no"})
    llm = FakeLLM(answers, {"BI": ["No, the camera is not good enough for the price, and the build sounds basic, so I would skip it and look elsewhere."]})
    out = _run(service, llm)
    v = out["persona_summaries"][0]["response_validation"]["BI"]
    assert v["status"] == "corrected" and "decided not to buy" in " ".join(v["previous_attempts"][0]["issues"])


# ---------- PP: no build claims without evidence
def test_pp_protection_claims_need_security_evidence():
    v = validate_construct_response
    plain = build_product_context(*PRODUCTS["smartphone"])
    secured = build_product_context("Nova 5G", "A 5G smartphone with Face ID, fingerprint unlock and a 5000mAh battery.", "Smartphone", 20000)
    for t in ["Well built, feels sturdy and reliable.", "Well-built with a sturdy feel and good battery life, which is reliable."]:
        assert any("physical build" in i for i in v(t, "PP", plain)), t                 # the saved-run PP answers are off-construct
    assert any("does not describe" in i for i in v("My accounts are well protected on this phone.", "PP", plain))
    assert any("full privacy protection" in i for i in v("Face ID makes me feel my accounts are well protected on this phone.", "PP", secured))
    assert v("Face ID means only I can unlock it, but the description does not say how my accounts are protected beyond that.", "PP", secured) == []
    assert v("The description doesn't say how it protects my data, so I can't tell yet.", "PP", plain) == []


# ---------- stale server guard
def test_stale_server_code_is_refused(service, monkeypatch):
    monkeypatch.setattr(es, "_LOADED_FINGERPRINT", "stale-code")
    with pytest.raises(RuntimeError, match="Restart the dashboard server"):
        _run(service, FakeLLM(GOOD, {}))


def test_saved_run_records_the_code_fingerprint(service):
    out = _run(service, FakeLLM(GOOD, {}))
    assert out["concept"]["pipeline_fingerprint"] == es._LOADED_FINGERPRINT


# ---------- "New Concept" must not carry over the previous product
def _run_js_form_flow(script_body: str) -> dict:
    """Run the real freshProductForm()/openModal() source from app.js in Node, with render() stubbed."""
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")

    def fn(name):
        start = js.index(f"function {name}(")
        depth, i = 0, js.index("{", start)
        while True:
            depth += {"{": 1, "}": -1}.get(js[i], 0)
            i += 1
            if depth == 0:
                return js[start:i]

    program = (fn("freshProductForm") + "\n" + fn("openModal") + "\n"
               "function render() {}\n"
               "const state = { activeModal: null, selectedPersonaId: null, errorMessage: null, activeEvaluation: null, form: freshProductForm() };\n"
               + script_body + "\nconsole.log(JSON.stringify(state));")
    out = subprocess.run(["node", "-e", program], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_new_concept_clears_the_previous_samsung_product():
    state = _run_js_form_flow(f"""
        openModal('new_eval');
        state.form.name = 'Samsung Galaxy M35 5G';
        state.form.category = 'Smartphone';
        state.form.price_inr = 21000;
        state.form.description = {json.dumps(RENO_RUN_DESC)};
        state.form.specifications = '6.6-inch Super AMOLED; 6000mAh battery; Exynos 1380';
        state.form.features = [{{name: 'Super AMOLED Display', explanation: 'x'}}];
        state.errorMessage = 'old error';
        state.activeEvaluation = {{concept: {{name: 'Samsung Galaxy M35 5G', description: 'saved Samsung run'}}}};
        state.activeModal = null;          // the Samsung evaluation finished
        openModal('new_eval');             // click "New Concept"
        state.form.name = 'apple';         // type the new product
    """)
    form = state["form"]
    assert form["name"] == "apple"
    assert form["description"] == "" and form["specifications"] == "" and form["features"] == []
    assert form["price_inr"] is None and form["category"] == ""
    assert "Samsung" not in json.dumps(form) and "AMOLED" not in json.dumps(form)
    assert state["errorMessage"] is None
    # the previously loaded evaluation (a historical record) is not modified by opening a new form
    assert state["activeEvaluation"]["concept"]["description"] == "saved Samsung run"


def test_other_modals_do_not_reset_the_form():
    state = _run_js_form_flow("""
        openModal('new_eval');
        state.form.name = 'Nova 5G';
        state.form.description = 'A 5G smartphone with a 5000mAh battery.';
        openModal('persona_detail', 'P1_premium_tier_owner');
    """)
    assert state["form"]["name"] == "Nova 5G" and state["form"]["description"].startswith("A 5G smartphone")


def test_loading_a_saved_evaluation_never_touches_the_form():
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    body = js.split("async function handleSelectSavedEvaluation")[1].split("\nasync function ")[0].split("\nfunction ")[0]
    assert "state.form" not in body


def test_apple_name_with_leftover_samsung_description_is_still_refused(service):
    llm = FakeLLM(GOOD, {})
    service.generate_fn = llm
    with pytest.raises(ProductContextError, match="does not appear in the description"):
        service.run_evaluation({"name": "apple", "category": "Smartphone", "price_inr": 79999,
                                "description": RENO_RUN_DESC, "features": []})
    assert llm.calls == []                     # nothing is generated from mismatched product data


# ---------- iPhone 17 (₹45,000): persona financial positions, decisions, PP construct
IPHONE_DESC = ("Apple iPhone 17 is a premium smartphone with a 6.3-inch OLED display, A19 processor, 48MP rear camera and 18MP front camera. "
               "It runs on iOS and supports 5G connectivity. It has Face ID, USB-C charging and water and dust resistance.")


def test_personas_have_distinct_qualitative_financial_positions():
    from scripts.product_evaluation_prompts import SIMPLE_PRICE_STANCE, SIMPLE_SYSTEM_PROMPT
    assert "can comfortably afford premium and upper-mid-range products" in SIMPLE_PRICE_STANCE["P1"]
    assert "not too expensive for you by itself" in SIMPLE_PRICE_STANCE["P1"]                  # P1 does not reject ₹45,000 on price alone
    assert "fairly flexible spending" in SIMPLE_PRICE_STANCE["P2"] and "strong specs do not justify any price" in SIMPLE_PRICE_STANCE["P2"]
    assert "moderate budget" in SIMPLE_PRICE_STANCE["P3"] and "isn't cheap" in SIMPLE_PRICE_STANCE["P3"]
    assert "limited spare money" in SIMPLE_PRICE_STANCE["P4"] and "more than most buyers" in SIMPLE_PRICE_STANCE["P4"]
    assert "student" not in SIMPLE_SYSTEM_PROMPT.lower()                                          # no shared low-budget framing
    for stance in SIMPLE_PRICE_STANCE.values():
        assert not re.search(r"₹|\d", stance)                                                     # qualitative only: no budgets or thresholds


def test_iphone_prompt_carries_price_and_full_private_assessment():
    ctx = build_product_context("Apple iPhone 17", IPHONE_DESC, "Smartphone", 45000)
    for pid in PERSONAS:
        user = _prompt(ctx, pid)
        assert "₹45,000" in user.split("ABOUT YOU")[0]
        about = user.split("ABOUT YOU:")[1].split("QUESTIONS")[0]
        assert "₹" not in about
        for field in ("\"comfortable_to_spend\"", "\"better_than_my_usual_options\"", "\"still_need_to_check\"",
                      "Choose yes or no whenever the information lets you decide"):
            assert field in user, (pid, field)
    from scripts.product_evaluation_prompts import SIMPLE_STYLE_EXAMPLES
    assert not re.search(r"price|spend|expensive|cost", SIMPLE_STYLE_EXAMPLES, re.I)            # style examples no longer prime price


def test_maybe_in_the_assessment_counts_as_unsure():
    from scripts.response_validation import stated_buying_decision
    assert stated_buying_decision({"buying_decision": "maybe"}) == "unsure"                      # P3's saved assessment
    assert stated_buying_decision({"buying_decision": "Unsure"}) == "unsure"                     # P4's saved assessment
    assert stated_buying_decision({"buying_decision": "Yes"}) == "yes" and stated_buying_decision({"buying_decision": "no"}) == "no"


IPHONE_BI = [
    ("P1_x", "yes", "Yes, I would consider buying it because its performance and camera features justify the price."),
    ("P2_x", "yes", "Yes, I would buy it because the processor and camera capabilities suit my technology needs."),
    ("P3_x", "yes", "I would consider buying it because it handles everyday tasks well, although it is a significant expense."),
    ("P4_x", "no", "No, I would choose a more affordable phone that meets my essential needs."),
    ("P1_x", "no", "No, too expensive for the features I want."),                    # saved-run P1 answer, now consistent with a 'no'
    ("P4_x", "no", "No, too expensive for daily tasks."),                            # saved-run P4 answer
    ("P2_x", "unsure", "I'm not sure yet; I'd compare its camera with other phones at this price first."),
]


@pytest.mark.parametrize("pid,decision,text", IPHONE_BI)
def test_iphone_bi_answers_pass_when_they_match_the_assessment(pid, decision, text):
    ctx = build_product_context("Apple iPhone 17", IPHONE_DESC, "Smartphone", 45000)
    assert validate_construct_response(text, "BI", ctx, persona_id=pid, assessment_decision=decision) == [], (decision, text)


def test_same_product_different_justified_decisions():
    ctx = build_product_context("Apple iPhone 17", IPHONE_DESC, "Smartphone", 45000)
    v = validate_construct_response
    assert v("Yes, I would buy it because the A19 processor and camera suit how I use my phone.", "BI", ctx,
             persona_id="P2_x", assessment_decision="yes") == []
    assert v("No, I would choose a more affordable phone that meets my essential needs.", "BI", ctx,
             persona_id="P4_x", assessment_decision="no") == []
    # an assessment of yes cannot produce a no, and the reverse
    assert any("decided to buy" in i for i in v("No, too expensive for the features I want.", "BI", ctx, persona_id="P1_x", assessment_decision="yes"))
    assert any("decided not to buy" in i for i in v("Yes, I would buy it for the camera quality.", "BI", ctx, persona_id="P4_x", assessment_decision="no"))


def test_price_is_kept_out_of_si_and_pp_and_their_retries():
    from scripts.product_evaluation_prompts import SIMPLE_CONSTRUCT_FOCUS
    from scripts.response_validation import build_correction_prompt
    ctx = build_product_context("Apple iPhone 17", IPHONE_DESC, "Smartphone", 45000)
    v = validate_construct_response
    for code, text in [("SI", "Friends might like it, but family could find it pricey."),          # saved-run SI answers
                       ("PP", "It feels sturdy and well-built, but the price is on the higher side.")]:  # saved-run PP answer
        assert any("price" in i for i in v(text, code, ctx)), code
    assert "or the price" in SIMPLE_CONSTRUCT_FOCUS["SI"] and "do not talk about price" in SIMPLE_CONSTRUCT_FOCUS["PP"]
    for code in ("PE", "EE", "HM", "SI", "PP"):
        fix = build_correction_prompt(ctx, "P4_budget", code, "x", ["talks about price"], "sys")["user"]
        assert "₹" not in fix and "careful with money" not in fix, code                             # retries no longer show the price
    assert "₹45,000" in build_correction_prompt(ctx, "P4_budget", "BI", "x", ["too short"], "sys")["user"]


def test_pp_follows_the_approved_personal_information_construct():
    from scripts.product_evaluation_prompts import SIMPLE_QUESTIONS
    ctx = build_product_context("Apple iPhone 17", IPHONE_DESC, "Smartphone", 45000)
    assert "protect your personal information" in SIMPLE_QUESTIONS["PP"]
    assert any("full privacy protection" in i for i in validate_construct_response(
        "Face ID makes me feel my personal information and accounts are well protected.", "PP", ctx))
    assert any("physical build" in i for i in validate_construct_response("The phone feels well-built and should last a long time.", "PP", ctx))
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    assert "protects personal information, accounts and privacy" in js
    assert "build quality, safety, and data security" not in js


def test_no_invented_iphone_specs():
    ctx = build_product_context("Apple iPhone 17", IPHONE_DESC, "Smartphone", 45000)
    assert any("specs" in i for i in validate_construct_response("The 8GB RAM keeps every app running fast.", "PE", ctx))
    assert validate_construct_response("The A19 processor keeps every app running fast for my work.", "PE", ctx) == []
