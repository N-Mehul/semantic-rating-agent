"""Reliability regression tests: 24/24 scored answers, SI/PP/BI construct rules, evidence grounding, SSR integrity.

Offline: fake Qwen + fake embedder. These prove the pipeline mechanics and the validators; they do not prove real Qwen
output quality (that is checked by a real evaluation run).
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from canonical_ssr import PAPER_EPSILON, expected_rating
from dashboard import evaluation_service as es
from improved_ssr import DEFAULT_IMPROVED_K, DEFAULT_IMPROVED_TEMPERATURE, improved_similarities_to_pmf
from scripts.response_validation import (
    MAX_CORRECTION_RETRIES,
    PRICE_LEAK_INSTRUCTION,
    build_batch_correction_prompt,
    build_correction_prompt,
    build_product_context,
    validate_construct_response as v,
)
from tests.test_product_context_validation import GOOD, PER_PERSONA, PRODUCTS, ROOT, FakeLLM, _per_persona, _run, service  # noqa: F401

CODES = ["PE", "EE", "HM", "SI", "PP", "BI"]
PIDS = ["P1_premium_tier_owner", "P2_midrange_heavy_user", "P3_midrange_moderate_user", "P4_budget_or_undisclosed_price"]

ONEPLUS_DESC = ("OnePlus Nord CE4 5G is a mid-range smartphone with a 6.7-inch AMOLED display with a 120Hz refresh rate, the Qualcomm "
                "Snapdragon 7 Gen 3 processor, 8GB RAM and 128GB storage. It has a 50MP main camera with OIS, a 5500mAh battery with "
                "100W SUPERVOOC fast charging, 5G and an in-display fingerprint sensor.")


@pytest.fixture()
def oneplus():
    return build_product_context("OnePlus Nord CE4 5G", ONEPLUS_DESC, "Smartphone", 24999)


@pytest.fixture()
def nova():
    return build_product_context(*PRODUCTS["smartphone"])


def _issues(text, code, ctx, **kw):
    return " | ".join(v(text, code, ctx, **kw))


class ScriptedLLM(FakeLLM):
    """FakeLLM that also records which prompts were main calls and which were corrections, in order."""

    def __init__(self, first, fixes):
        super().__init__(first, fixes)
        self.kinds = []

    def __call__(self, model, prompt, timeout=None, **kw):
        self.kinds.append("fix" if '{"answer": "..."}' in prompt["user"] or "Rewrite each answer" in prompt["user"] else "main")
        if "Rewrite each answer" in prompt["user"]:                       # batch correction: answer every listed construct
            self.calls.append(prompt["user"])
            codes = re.findall(r"QUESTION \((\w\w)\):", prompt["user"])
            return {"response_text": json.dumps({c: self.fixes[c].pop(0) for c in codes})}, None
        return super().__call__(model, prompt, timeout, **kw)


# ---------------------------------------------------------------- 24/24 combinations
def test_all_24_persona_construct_combinations_are_generated_validated_and_scored(service):
    out = _run(service, FakeLLM(_per_persona, {}), persona="all")
    c = out["ssr_results"]["completeness"]
    assert c["expected_responses"] == c["generated_responses"] == c["validated_responses"] == c["scored_responses"] == 24
    assert c["not_scored"] == [] and c["missing_combinations"] == [] and c["complete"] is True
    pairs = {(ps["persona_id"], a["construct_code"]) for ps in out["persona_summaries"] for a in ps["anchor_sets"]}
    assert pairs == {(p, code) for p in PIDS for code in CODES}
    assert len(out["ssr_results"]["records"]) == 24


def test_saved_results_on_disk_are_complete(service):
    out = _run(service, FakeLLM(_per_persona, {}), persona="all")
    d = es.EVAL_BASE_DIR / out["eval_id"]
    saved = json.loads((d / "ssr_results" / "ssr_results.json").read_text(encoding="utf-8"))
    assert saved["completeness"]["scored_responses"] == 24 and saved["completeness"]["complete"]
    responses = json.loads((d / "persona_responses.json").read_text(encoding="utf-8"))
    assert len(responses) == 4 and all(set(CODES) <= set(r["construct_responses"]) for r in responses)
    rows = (d / "ssr_results" / "ssr_results.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1 + 24
    assert saved["generation_stats"]["embedded_answers"] == 24 and "embedding_seconds" in saved["generation_stats"]


def test_all_main_calls_run_before_any_correction(service):
    """Two-phase generation keeps Ollama's prompt cache warm for P2-P4 (corrections would overwrite it)."""
    def first(user):
        ans = dict(_per_persona(user))
        if "You watch your money closely" in user:
            ans["SI"] = "Friends might recommend it, but family could find it pricey for an ordinary phone like this one in my opinion, honestly."
        return ans
    llm = ScriptedLLM(first, {"SI": [PER_PERSONA["money"]["SI"]]})
    out = _run(service, llm, persona="all")
    assert llm.kinds[:4] == ["main"] * 4 and llm.kinds[4:] == ["fix"]
    assert out["ssr_results"]["completeness"]["scored_responses"] == 24


# ---------------------------------------------------------------- SI
SI_PRICE_LEAKS = [
    "My friends might think it's a good everyday phone, but they could also suggest checking out other options with similar features at a lower price.",
    "Friends might recommend it for its fast charging and good camera, but family might think it's a bit pricey for an otherwise decent phone.",
    "My colleagues would notice the big display, and they might ask whether I got a good discount on it during the festive sale this month.",
    "Family members would probably say it fits my budget, and my friends might agree that the fast charging is handy for everyday use at work.",
    "My friends would discuss the purchase decision with me, since they always compare the display and camera before anyone buys a new phone.",
]


@pytest.mark.parametrize("text", SI_PRICE_LEAKS)
def test_si_price_leakage_is_rejected(oneplus, text):
    assert "price leakage" in _issues(text, "SI", oneplus)


@pytest.mark.parametrize("pid,text", [
    ("P1_x", "I cannot know whether my friends would recommend this phone without hearing their experiences. Its display and fast "
             "charging may appeal to people who value entertainment and convenience, but I would not assume that everyone would prefer it."),
    ("P2_x", "My colleagues may appreciate the large display and fast charging for everyday use. However, I do not have enough information "
             "about their actual opinions to claim that they would recommend this device."),
    ("P3_x", "My family mostly wants a phone that just works, so they might like the big battery and quick charging when they see me use it. "
             "I have not heard their actual views on this model, so I would not assume they approve."),
    ("P4_x", "My close friends trade tips about phones, and they would probably notice the fast charging first. I have no real idea what "
             "they think of this model yet, so their experience would matter to me before I made up my mind."),
])
def test_si_valid_answers_for_all_four_personas(oneplus, pid, text):
    assert v(text, "SI", oneplus, persona_id=pid) == []


def test_si_invented_recommendation_and_quotes_are_rejected(oneplus):
    t = "Everyone has this phone and my friends recommend it for the fast charging, so it is clearly the phone people around me love."
    assert "states a recommendation" in _issues(t, "SI", oneplus)
    t2 = "My brother told me it is great, and my colleagues would notice the 120Hz display when they see me scrolling at the office."
    assert "quotes what a specific person said" in _issues(t2, "SI", oneplus)


def test_si_correction_uses_the_mandated_instruction_and_does_not_echo_the_leak(oneplus):
    bad = SI_PRICE_LEAKS[1]
    issues = v(bad, "SI", oneplus)
    for attempt in (1, 2):
        user = build_correction_prompt(oneplus, "P4_budget", "SI", bad, issues, "sys", attempt=attempt)["user"]
        assert ("Rewrite this SI answer without any price, cost, affordability, discount, budget or purchase-decision references. "
                "Discuss only social influence.") in user
        assert "pricey" not in user and "₹" not in user and "24,999" not in user          # the leak is never shown back
        assert "careful with money" not in user and "limited spare money" not in user      # no money-framed persona line on SI
        assert "family and close friends" in user                                          # P4's money-free social context
    assert PRICE_LEAK_INSTRUCTION.format(code="SI", topic="social influence") in user


def test_si_correction_retry_reaches_a_scored_answer(service):
    bad = dict(GOOD, SI="My friends might think it is good, but they would suggest more affordable options for everyday use, honestly.")
    second = "Friends might like it, but they would wait for a discount before they told me anything about it at all, really."
    llm = FakeLLM(bad, {"SI": [second, GOOD["SI"]]})
    out = _run(service, llm)
    val = out["persona_summaries"][0]["response_validation"]["SI"]
    assert val["status"] == "corrected" and val["retries"] == 2
    assert all("price leakage" in " ".join(a["issues"]) for a in val["previous_attempts"])
    assert "SI" in out["persona_summaries"][0]["construct_scores"]
    fixes = [c for c in llm.calls if '{"answer": "..."}' in c]
    assert len(fixes) == 2 and "Write a fresh answer" in fixes[1]
    assert "wait for a discount" not in fixes[1] and "more affordable options" not in fixes[0]       # rejected wording not echoed


# ---------------------------------------------------------------- PP
def test_pp_must_be_about_personal_information(oneplus):
    t = ("The phone has a large 6.7-inch AMOLED screen and a 5500mAh battery, which means I can watch videos for a long time without "
         "worrying about anything at all during the day.")
    assert "does not talk about protecting your personal information" in _issues(t, "PP", oneplus)
    t2 = "It feels sturdy and well-built, so I am confident it will survive drops and keep working for years while I use it daily."
    assert "physical build" in _issues(t2, "PP", oneplus)


@pytest.mark.parametrize("text", [
    "It has in-display fingerprint, which seems secure, so my personal information and accounts are safe on this phone for daily banking use.",
    "The fingerprint sensor keeps my data safe, and with 8GB RAM and 5G I feel my personal information and accounts are well looked after here.",
    "The in-display fingerprint sensor protects all my information, so I would feel completely safe keeping my banking and email apps on it.",
])
def test_pp_fingerprint_is_not_treated_as_full_privacy_protection(oneplus, text):
    assert "treats the fingerprint/unlock feature as full privacy protection" in _issues(text, "PP", oneplus)


def test_pp_authentication_distinguished_from_data_protection_passes(oneplus):
    t = ("The in-display fingerprint sensor would stop other people from unlocking my phone, which helps. But the specifications do not "
         "say how my accounts or personal data are protected beyond that, so I would not assume full protection.")
    assert v(t, "PP", oneplus) == []


@pytest.mark.parametrize("term,text", [
    ("encryption", "Its encryption keeps my personal data private, and the in-display fingerprint sensor only unlocks the phone for me, which helps me feel calm."),
    ("security updates", "Regular security updates will keep my accounts protected, and the fingerprint sensor only controls who unlocks it when I leave it on my desk."),
    ("knox", "Knox protects my banking data, while the fingerprint sensor just controls access to the phone itself when someone else picks it up."),
])
def test_pp_invented_security_mechanisms_are_rejected(oneplus, term, text):
    assert f"does not list ({term}" in _issues(text, "PP", oneplus) or term in _issues(text, "PP", oneplus)


def test_pp_saying_a_mechanism_is_not_listed_is_allowed(oneplus):
    t = ("The fingerprint sensor only controls who can unlock it. The description does not mention encryption or security updates, "
         "so I can't tell how well my personal data and accounts would be protected over time.")
    assert v(t, "PP", oneplus) == []


def test_pp_fingerprint_speed_claims_are_rejected(oneplus):
    t = ("The in-display fingerprint sensor is a bit slow, which might be a security concern for my personal accounts, and that only "
         "controls access to the phone anyway when I unlock it.")
    assert "fingerprint speed or reliability" in _issues(t, "PP", oneplus)


# ---------------------------------------------------------------- product grounding
@pytest.mark.parametrize("text,label", [
    ("The 8GB RAM and 120Hz display make multitasking smooth, but heavy apps can sometimes lag when I keep many of them open during work.", "real-world performance"),
    ("The 5500mAh battery lasts well during normal use, and the Snapdragon 7 Gen 3 lets me switch between my work apps quickly through the day.", "actual battery life"),
    ("It never lags during heavy gaming, and the Snapdragon 7 Gen 3 lets me switch between work apps quickly while the 120Hz screen stays smooth.", "absolute performance"),
    ("The 8GB RAM limits multitasking for me, so switching between many apps at work would feel slow compared with what I usually expect from a phone.", "RAM limitation"),
    ("The 50MP camera with OIS and photos look sharp and clear in every light, which makes sharing my work and family pictures quick and easy.", "actual photo quality"),
    ("It scores well in benchmarks, and the 120Hz display with the Snapdragon 7 Gen 3 makes my daily work apps open quickly through the day.", "real-world performance"),
])
def test_unsupported_performance_claims_are_rejected(oneplus, text, label):
    assert f"unsupported product claim ({label}" in _issues(text, "PE", oneplus) or label in _issues(text, "PE", oneplus)


@pytest.mark.parametrize("text", [
    "The Snapdragon 7 Gen 3 and 8GB RAM should let me keep office apps, chats and maps open together, and the 120Hz display should make "
    "scrolling smoother. I can't tell from the specs how it copes with long gaming sessions.",
    "The 120Hz display supports smoother scrolling, and the 5500mAh battery should get me through a long workday. The specs do not show "
    "how fast it really is in daily use, so I would treat that as unknown.",
])
def test_qualified_spec_interpretations_are_allowed(oneplus, text):
    assert v(text, "PE", oneplus) == []


def test_invented_specs_are_rejected(oneplus):
    t = ("The 12GB RAM and 144Hz display should make multitasking quick for me, and the Snapdragon 7 Gen 3 should help my apps open "
         "without waiting through the day.")
    assert "specs that are not in the product description" in _issues(t, "PE", oneplus)


def test_unsupported_claim_correction_carries_the_grounding_instruction(oneplus):
    bad = "The 8GB RAM is fine, but heavy apps can sometimes lag and the battery drains fast when I play games all evening long."
    user = build_correction_prompt(oneplus, "P2_x", "PE", bad, v(bad, "PE", oneplus), "sys")["user"]
    assert "Do not claim real-world lag" in user and "PRODUCT FACTS" in user and "Snapdragon 7 Gen 3" in user


# ---------------------------------------------------------------- length
def test_short_answers_are_rejected_and_detailed_ones_accepted(oneplus):
    assert "too short" in _issues("Setting up is simple and the interface is familiar to me.", "EE", oneplus)
    assert "too short" in _issues("No, too expensive.", "BI", oneplus, persona_id="P4_x")
    long_ok = ("Setting it up should be simple because it is a standard Android phone, and moving my contacts and chats would take one "
               "evening. The large 6.7-inch screen makes the menus easy to read, though some settings may take a few days to learn.")
    assert v(long_ok, "EE", oneplus) == []


# ---------------------------------------------------------------- BI
BI_CASES = [
    ("P1_x", "no", "No, the display and charging are strong, but the 8GB RAM and camera are not special enough for me to pick it over the other phones I usually choose."),
    ("P2_x", "yes", "Yes, the Snapdragon 7 Gen 3, 120Hz AMOLED screen and 100W charging give me the specs I want at this price."),
    ("P3_x", "no", "No, I would continue using my current phone because it already meets my everyday needs. I would need a clear practical improvement to justify switching."),
    ("P4_x", "unsure", "Unsure, the features are useful for daily use but the price would stretch my budget, so I need to check for a discount or a cheaper variant first."),
]


@pytest.mark.parametrize("pid,decision,text", BI_CASES)
def test_bi_answers_for_all_four_personas_pass(oneplus, pid, decision, text):
    assert v(text, "BI", oneplus, persona_id=pid, assessment_decision=decision) == []


def test_bi_contradicting_its_assessment_is_rejected(oneplus):
    t = "No, the 8GB RAM is fine but I want a better camera than this for the money, so I would choose another phone instead."
    assert "assessment was unsure" in _issues(t, "BI", oneplus, persona_id="P1_x", assessment_decision="unsure",
                                              assessment={"comfortable_to_spend": "yes", "price_justified_for_me": "partly",
                                                          "better_than_my_usual_options": "not sure"})
    assert "decided to buy" in _issues(t, "BI", oneplus, persona_id="P1_x", assessment_decision="yes")


def test_bi_single_sentence_within_40_words_is_not_rejected_for_length(oneplus):
    t = "Yes, I would buy it because the 5500mAh battery and 100W charging suit my long workdays and the price feels fair for what I get every day."
    assert len(t.split()) > 25 and v(t, "BI", oneplus, persona_id="P3_x") == []


def test_common_decision_openings_are_not_treated_as_duplicates(oneplus):
    a = "Yes, the Snapdragon 7 Gen 3, 120Hz AMOLED screen and 100W charging give me the specs I want at this price."
    b = "Yes, I would buy it because the big battery and fast charging cover my daily needs at a price that feels sensible."
    assert v(b, "BI", oneplus, persona_id="P3_x", earlier_answers=[a]) == []
    c = "Maybe, I need to check how the camera compares with similar phones at this price before I decide to buy it."
    d = "Maybe, I would first check whether a discount brings the price closer to what I can spend on a phone."
    assert v(d, "BI", oneplus, persona_id="P4_x", earlier_answers=[c]) == []


def test_near_duplicate_answers_are_rejected(oneplus):
    a = ("The Snapdragon 7 Gen 3 and 8GB RAM should let me keep office apps, chats and maps open together, and the 120Hz display "
         "should make scrolling smoother. I can't tell from the specs how it copes with long gaming sessions.")
    b = ("The Snapdragon 7 Gen 3 and 8GB RAM should let me keep office apps, chats and maps open at once, and the 120Hz display "
         "should make scrolling smoother. I can't tell from the specs how it copes with long gaming sessions.")
    assert "almost the same as another buyer's answer" in _issues(b, "PE", oneplus, earlier_answers=[a])


def test_bi_correction_repeats_the_private_assessment(oneplus):
    a = {"strengths": "display", "comfortable_to_spend": "a stretch", "buying_decision": "No"}
    user = build_correction_prompt(oneplus, "P4_x", "BI", "Maybe.", ["too short"], "sys", assessment=a)["user"]
    assert "Your buying decision was 'no'" in user and "₹24,999" in user


def test_batch_correction_shows_price_only_when_bi_is_included(oneplus):
    items = [("SI", SI_PRICE_LEAKS[0], ["price leakage: x"]), ("PP", "short", ["too short"])]
    assert "₹" not in build_batch_correction_prompt(oneplus, "P4_x", items, "sys")["user"]
    assert "₹24,999" in build_batch_correction_prompt(oneplus, "P4_x", items + [("BI", "x", ["too short"])], "sys")["user"]


# ---------------------------------------------------------------- no default scores
def test_an_answer_that_never_validates_gets_no_score_and_full_diagnostics(service):
    bad = dict(GOOD, SI="Friends would say it is pricey, and they might suggest a cheaper option for my budget instead of this phone today.")
    llm = FakeLLM(bad, {"SI": ["They would want a discount first, and the price would worry them for an ordinary phone like this one.",
                               "My budget friends would say it costs too much for what it offers, so they might suggest something cheaper."]})
    out = _run(service, llm)
    ps = out["persona_summaries"][0]
    val = ps["response_validation"]["SI"]
    assert val["status"] == "flagged" and val["retries"] == MAX_CORRECTION_RETRIES and val["issues"]
    assert "SI" not in ps["construct_scores"] and all(a["construct_code"] != "SI" for a in ps["anchor_sets"])
    c = out["ssr_results"]["completeness"]
    assert c["scored_responses"] == 5 and c["complete"] is False
    assert c["not_scored"][0]["construct"] == "SI" and "price leakage" in " ".join(c["not_scored"][0]["issues"])
    assert all(r["construct_code"] != "SI" for r in out["ssr_results"]["records"])            # nothing substituted


# ---------------------------------------------------------------- SSR integrity
ANCHOR_HASH = "b05cce4bcfcae54cf97504029d8a8b41233efb5d91222f0a3c0ff8a56c3643be"


def test_ssr_parameters_and_anchor_sets_are_frozen():
    from survey_ssr_bridge import get_construct_anchors
    assert DEFAULT_IMPROVED_K == 3 and DEFAULT_IMPROVED_TEMPERATURE == 0.75 and PAPER_EPSILON == 0.0
    h = hashlib.sha256()
    for code in CODES:
        a = get_construct_anchors(code)
        assert len(a.statements) == 5
        h.update(("|".join(a.statements) + "#" + "|".join(map(str, a.labels))).encode())
    assert h.hexdigest() == ANCHOR_HASH


def test_every_saved_score_is_recomputable_from_its_cosines(service):
    out = _run(service, FakeLLM(_per_persona, {}), persona="all")
    for ps in out["persona_summaries"]:
        for a in ps["anchor_sets"]:
            pmf = improved_similarities_to_pmf(np.array(a["cosine_similarities"]), k=3, temperature=0.75, epsilon=0.0)
            assert a["pmf_probabilities"] == pytest.approx(pmf, abs=1e-5)
            assert a["expected_rating"] == pytest.approx(expected_rating(pmf), abs=1e-3)


def test_batched_embedding_prefetch_gives_the_same_vectors_as_single_calls():
    from scripts.ollama_embedding_adapter import OllamaEmbeddingAdapter
    adapter = OllamaEmbeddingAdapter()
    texts = ["first answer", "second answer"]
    adapter._fetch_embeddings = lambda ts: [np.arange(4, dtype=float) + len(t) for t in ts]   # stands in for /api/embed
    batch = adapter.embed_texts(texts)
    adapter._fetch_embeddings = lambda ts: pytest.fail("embed_text must reuse the exact-text cache")
    assert np.array_equal(adapter.embed_text("first answer"), batch[0])


def test_documented_qwen_benchmark_is_preserved():
    assert "79.17" in (ROOT / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------- background jobs
def test_background_job_completes_with_24_scored_answers(service):
    import time
    jobs = es.EvaluationJobs(service) if hasattr(es, "EvaluationJobs") else None
    if jobs is None:
        pytest.skip("job manager class not exposed under this name")
    service.generate_fn = FakeLLM(_per_persona, {})
    name, desc, cat, price = PRODUCTS["smartphone"]
    job = jobs.start({"name": name, "description": desc, "category": cat, "price_inr": price, "features": []})
    for _ in range(200):
        st = jobs.status(job["job_id"])
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert st["status"] == "completed", st.get("error")
    assert st["result"]["ssr_results"]["completeness"]["scored_responses"] == 24


@pytest.mark.parametrize("text", [
    "No, unsure. The specs are good for everyday use, but at ₹24,999, I'd like to see better gaming performance and long-term battery life before deciding.",
    "No, the price and lack of specific gaming and long-term battery details make me unsure. I need to check more about these aspects.",
])
def test_bi_mixing_no_with_unsure_is_rejected(oneplus, text):
    """The two contradictory BI answers from the real OnePlus run of 2026-11-06 12:36."""
    assert "mixes a yes/no opening" in _issues(text, "BI", oneplus, persona_id="P2_x", assessment_decision="unsure")
