"""Lightweight tests for scripts/product_evaluation_prompts_v2.py (no LLM calls)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import product_evaluation_prompts_v2 as v2  # noqa: E402
from scripts.run_product_evaluation import NOVA_AI_SMARTPHONE, PERSONA_IDS  # noqa: E402


def _p(pid="P1_premium_tier_owner", cons=v2.CONSTRUCT_ORDER):
    return v2.build_prompt_v2(NOVA_AI_SMARTPHONE, pid, cons)


def test_pp_is_privacy_not_durability():
    u = _p(cons=["PP"])["user"]
    assert "personal information" in u and "unauthorized access" in u
    assert "Do not write about: physical durability" in u
    for item in v2.survey_items("PP"):
        assert item in u  # verbatim codebook statements


def test_hm_is_enjoyment_and_distinct_from_pe():
    u = _p(cons=["HM"])["user"]
    assert "enjoyable, fun, entertaining and pleasurable" in u
    assert "productivity" in u.split("Do not write about:")[1]


def test_no_forced_bi_sentence_no_pocket_friendly_no_ownership_contradiction():
    for pid in PERSONA_IDS:
        p = _p(pid)
        text = p["system"] + p["user"]
        assert "I am certain I will continue" not in text
        assert "pocket-friendly" not in text.lower()
        assert "Current smartphone: Flagship" not in text
        assert "You do NOT own" not in text


def test_scale_labels_not_shown_to_model():
    assert "Neither agree nor disagree" not in _p()["user"]


def test_constructs_separated_and_complete():
    u = _p()["user"]
    for c in v2.CONSTRUCT_ORDER:
        assert f"[{c}] {v2.CONSTRUCTS[c]['name']}" in u


def test_context_plan_fits_and_flags_overflow():
    plan = v2.plan_context(_p())
    assert not plan["truncation_likely"] and plan["num_ctx"] in v2.CONTEXT_STEPS
    assert plan["prompt_tokens_estimate"] + plan["max_output_tokens"] <= plan["num_ctx"]
    d = v2.context_diagnostic(plan, prompt_eval_count=plan["num_ctx"] - 10)
    assert d["truncation_likely"] is True


def test_parser_never_copies_and_flags_leaks():
    r = v2.parse_response_v2('{"PP": "3 Neither agree nor disagree. Fine.", "HM": "I enjoy it."}', ["PP", "HM", "BI"])
    assert r["responses"]["BI"] is None and r["missing"] == ["BI"]
    assert r["rating_leak"] == ["PP"]
    bad = v2.parse_response_v2("not json at all", ["PE", "PP"])
    assert bad["parse_ok"] is False and all(v is None for v in bad["responses"].values())


# ------------------------------------------------------------------ Phase 3 (per-construct prompts, leak guard, flags)
import json  # noqa: E402

import pytest  # noqa: E402

P = NOVA_AI_SMARTPHONE


def _v3(pid, code, **kw):
    return v2.build_construct_prompt_v3(P, pid, code, spec=v2.load_spec(pid), **kw)


def test_persona_prompts_differ_and_price_is_not_pocket_friendly_for_all():
    users = {pid: _v3(pid, "BI")["user"] for pid in PERSONA_IDS}
    assert len(set(users.values())) == 4
    drivers = {pid: v2.persona_drivers(pid, v2.load_spec(pid), P) for pid in PERSONA_IDS}
    assert len(set(drivers.values())) == 4
    assert not any("pocket-friendly" in u.lower() for u in users.values())


@pytest.mark.parametrize("paid,expected", [
    ("₹10,000–₹20,000", "more than you paid"), ("More than ₹60,000", "less than you paid"),
    ("₹20,001–₹35,000", "same range"), ("Less than ₹10,000", "more than you paid"), ("Prefer not to say", "did not say")])
def test_price_position_is_relative_to_respondent(paid, expected):
    assert expected in v2.price_position(24999, paid)


def test_no_numeric_priming_in_persona_text():
    for pid in PERSONA_IDS:
        txt = v2.persona_block(pid, v2.load_spec(pid)) + v2.persona_drivers(pid, v2.load_spec(pid), P)
        assert "%" not in txt and "1-5" not in txt


@pytest.mark.parametrize("code", v2.CONSTRUCT_ORDER)
def test_construct_isolation_only_own_items(code):
    u = _v3("P2_midrange_heavy_user", code)["user"]
    for item in v2.survey_items(code):
        assert item in u
    for other in v2.CONSTRUCT_ORDER:
        if other != code:
            assert f"TOPIC: {v2.CONSTRUCTS[other]['name']}" not in u
            assert not any(item in u for item in v2.survey_items(other))
    assert "I am certain I will continue" not in u and "Neither agree nor disagree" not in u


def test_pp_hm_pe_focus():
    assert "privacy and data protection" in _v3("P1_premium_tier_owner", "PP")["user"]
    assert "physical durability" in _v3("P1_premium_tier_owner", "PP")["user"].split("Do not write about:")[1]
    assert "fun and pleasure" in _v3("P1_premium_tier_owner", "HM")["user"]
    assert "practical usefulness" in _v3("P1_premium_tier_owner", "PE")["user"]


def test_avoid_openings_passed_through():
    u = _v3("P3_midrange_moderate_user", "SI", avoid_openings=["my friends would say"])["user"]
    assert "'my friends would say'" in u


@pytest.mark.parametrize("text", [
    "3 Neither agree nor disagree · fine.", "4. I like it.", "4 - good phone.", "I rate this a 4.", "I would give it four out of five.",
    "Rating: 4", "Answer: agree", "Likert score: 3", "It is 4/5 for me.", "Easily 8 out of 10.", "Three stars from me.",
    "I strongly agree with this.", "80% sure I will keep it."])
def test_rating_leak_detected(text):
    assert v2.detect_rating_leak(text)


@pytest.mark.parametrize("text", [
    "I really enjoy clicking photos at night with this phone.", "Nova's price of ₹24,999 feels a bit much for me.",
    "My friends would probably support it, maybe two of them would even buy one.", "Five friends of mine use Android anyway."])
def test_rating_leak_not_flagged_for_normal_text(text):
    assert v2.detect_rating_leak(text) == []


def test_parse_single():
    assert v2.parse_single_v3('{"response": "This phone feels safe."}')["text"] == "This phone feels safe."
    assert v2.parse_single_v3('{"PP": "Okay for privacy."}')["text"] == "Okay for privacy."
    r = v2.parse_single_v3("no json")
    assert r["text"] is None and r["parse_ok"] is False


def _stub(outputs):
    calls = []

    def gen(model, prompt, timeout=0):
        calls.append(prompt)
        out = outputs[len(calls) - 1]
        return ({"response_text": json.dumps({"response": out}), "finish_reason": "STOP",
                 "usage": {"prompt_tokens": 900, "output_tokens": 60}}, None)
    return gen, calls


def test_one_corrective_retry_then_ok():
    gen, calls = _stub(["4 Agree. Nice phone.", "Privacy seems fine because data stays on this phone."])
    r = v2.generate_construct_response(gen, "m", P, "P1_premium_tier_owner", "PP")
    assert r["status"] == "ok_after_correction" and r["n_calls"] == 2 and len(calls) == 2
    assert v2.CORRECTIVE_NOTE in calls[1]["user"] and calls[1]["num_ctx"] == 2048


def test_leak_twice_is_flagged_not_retried_again():
    gen, calls = _stub(["Rating: 4", "I rate this a 4.", "never used"])
    r = v2.generate_construct_response(gen, "m", P, "P1_premium_tier_owner", "PP")
    assert r["status"] == "rating_leak_unresolved" and len(calls) == 2


def test_clean_first_answer_single_call():
    gen, calls = _stub(["The on-device processing on this phone makes me feel safer about my data."])
    r = v2.generate_construct_response(gen, "m", P, "P2_midrange_heavy_user", "PP")
    assert r["status"] == "ok" and len(calls) == 1 and r["attempts"][0]["context"]["prompt_tokens_actual"] == 900


def test_quality_flags():
    spec = v2.load_spec("P1_premium_tier_owner")
    good = v2.quality_flags("I like that the Nova AI Smartphone keeps some AI tasks on the device, so less of my personal data goes to the cloud.", "PP", P, spec)
    assert good["issues"] == [] and good["construct_aligned"]
    dur = v2.quality_flags("This phone feels sturdy and the build quality should last for years without wear and tear.", "PP", P, spec)
    assert "off_construct_content" in dur["issues"] and "construct_not_addressed" in dur["issues"]
    bad = v2.quality_flags("My Samsung laptop has 8 GB and 5G, I rate it a 4.", "PE", P, spec)
    assert {"other_brand_mentioned", "other_device_mentioned", "unsupported_claims", "rating_leak"} <= set(bad["issues"])
    rep = v2.quality_flags("I will keep using this phone because the camera is good for my photos.", "BI", P, spec,
                           other_texts=["I will keep using this phone because the camera is good for my reels."])
    assert "same_opening_as_other_persona" in rep["issues"] and "repeated_phrasing_across_personas" in rep["issues"]
    assert v2.quality_flags("", "BI", P)["issues"] == ["empty_or_malformed"]
    hist = v2.quality_flags("For the past few months this phone was hacked once, so I worry about my data.", "PP", P, spec)
    assert "invented_usage_history_or_incident" in hist["issues"]
