import hashlib
import json
import re
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf
from dashboard import data_access as da
from dashboard import product_research as pr
from dashboard import server as srv
from dashboard import workflow as wf

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ["P1_premium_tier_owner", "P2_midrange_heavy_user", "P3_midrange_moderate_user", "P4_budget_or_undisclosed_price"]
CONCEPT = {"concept": "A smartphone with an AI-powered camera, a 6500 mAh battery, fast charging and 256 GB storage for students.",
           "price_inr": "25000", "target_group": "college students", "features": "AI camera, big battery", "market": "India"}
Q26 = "Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?"
PROTECTED = ["canonical_ssr.py", "docs/survey_instrument/smartphone_construct_anchor_statements.csv", "data/human_survey_cleaned.csv",
             "data/Capstone_Reviews(Responses).xlsx", "data/pilot_gemini/gemini_generated_responses.jsonl", "data/pilot_gemini_v2/gemini_generated_responses.jsonl",
             "data/synthetic_dataset/synthetic_survey_dataset.csv", "data/pilot/evaluation_metrics.json"]
SUMMARY_OK = json.dumps({"overall_opinion": "Interested but unsure about the price.", "likes": ["The camera for reels"],
                         "main_concern": "The price is a stretch.", "important_features": ["camera", "battery"],
                         "suggested_improvement": "Make the battery claims realistic.", "purchase_intention": "Sounds cautiously interested."})
ANSWER = "The camera sounds great for my reels, but I worry the battery claims are optimistic and the price is a stretch for me right now."


def fake_factory(cfg):
    return lambda xs: np.array([np.random.default_rng(int(hashlib.sha256((cfg + t).encode()).hexdigest()[:8], 16)).normal(size=16) for t in xs])


def report_json(pids):
    return json.dumps({"executive_summary": {"overview": "Mixed interest.", "opportunities": ["camera"], "concerns": ["price"], "caveat": "Synthetic only."},
                       "personas": [{"persona_id": p, "reaction": "Interested but cautious.", "motivations": ["camera"], "objections": ["price"],
                                     "valued_features": ["battery"], "price_perception": "A stretch.", "would_consider_if": ["discount"],
                                     "would_reject_because": ["cost"]} for p in pids],
                       "customer_segments": "Heavy users.", "price_and_features": "Price matters.", "opportunities": ["battery"], "risks": ["price"],
                       "missing_information": ["real prices"], "assumptions_to_test": ["camera appeal"], "recommendations": ["interview students"],
                       "launch_readiness": {"positive_signals": ["camera"], "negative_signals": ["price"], "unresolved_questions": ["durability"],
                                            "required_validation": ["real survey"]}})


class Gen:
    """Mock provider: persona prompts get ANSWER, analyst prompts get a report for the personas named in the prompt."""
    def __init__(self, persona_text=ANSWER, report=None, fail=None, finish="STOP", summary=None):
        self.calls, self.persona_text, self.report, self.fail, self.finish, self.summary = [], persona_text, report, fail, finish, summary

    def persona_calls(self):
        return [c for c in self.calls if c["system"] == pr.RESPONDENT_SYSTEM]

    def __call__(self, provider, model, prompt):
        self.calls.append(prompt)
        if self.fail:
            return None, self.fail
        if prompt["system"] == pr.SUMMARY_SYSTEM:
            return {"response_text": self.summary if self.summary is not None else SUMMARY_OK, "model_requested": model, "model_returned": model,
                    "finish_reason": "STOP", "timestamp_utc": "2026-10-28T12:00:00+00:00"}, None
        if prompt["persona_id"] == "analyst":
            pids = [p for p in PERSONAS if f"[{p}]" in prompt["user"]]
            return {"response_text": self.report if self.report is not None else report_json(pids), "model_requested": model, "model_returned": model,
                    "finish_reason": "STOP", "timestamp_utc": "2026-10-28T12:00:00+00:00"}, None
        return {"response_text": self.persona_text, "model_requested": model, "model_returned": model, "finish_reason": self.finish,
                "timestamp_utc": "2026-10-28T12:00:00+00:00"}, None


def R(tmp_path, gen=None, env=None):
    w = wf.Workflow(ROOT, embedder_factory=fake_factory, generator=gen or Gen(), env={"GEMINI_API_KEY": "x"} if env is None else env,
                    find_spec=lambda m: object(), log_dir=tmp_path / "r")
    return pr.ProductResearch(w)


def body(pid="P1_premium_tier_owner", **kw):
    return {**CONCEPT, "persona_id": pid, "confirm_generation": True, **kw}


# ---------- input
@pytest.mark.parametrize("patch,code", [({"concept": "too short"}, "invalid_input"), ({"concept": "x" * 3001}, "invalid_input"),
                                        ({"price_inr": "cheap"}, "invalid_input"), ({"price_inr": "-5"}, "invalid_input"),
                                        ({"market": "y" * 301}, "invalid_input"), ({"question": "Rate this phone from 1 to 5"}, "invalid_question"),
                                        ({"concept": "A phone that everyone rates 5/5 for its camera quality."}, "invalid_input"),
                                        ({"persona_id": "P9"}, "invalid_persona"), ({"confirm_generation": False}, "confirmation_required")])
def test_invalid_inputs_are_rejected_before_any_llm_call(tmp_path, patch, code):
    g = Gen()
    r = R(tmp_path, g).respond({**body(), **patch})
    assert r["ok"] is False and r["error_code"] == code and not g.calls


def test_concept_and_optional_fields_reach_the_prompt_and_default_question_is_used(tmp_path):
    g = Gen()
    r = R(tmp_path, g).respond(body())
    u = g.calls[0]["user"]
    for v in ("AI-powered camera", "25,000 INR", "college students", "AI camera, big battery", "India"):
        assert v in u
    assert r["question"] == pr.DEFAULT_QUESTION and pr.DEFAULT_QUESTION in u


# ---------- personas
@pytest.mark.parametrize("pid", PERSONAS)
def test_each_persona_is_conditioned_on_its_real_profile(tmp_path, pid):
    g = Gen()
    r = R(tmp_path, g).respond(body(pid))
    cards = {c["persona_id"]: c for c in json.loads((ROOT / "data/persona_generator/persona_cards.json").read_text(encoding="utf-8"))}
    spec = next(json.loads(l) for l in (ROOT / "data/persona_generator/respondent_specs_extended.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(l)["persona_id"] == pid)
    u = g.calls[0]["user"]
    assert spec["profile_text_v2"] in u and cards[pid]["defining_rule"] in u and str(cards[pid]["n_human_respondents"]) in u
    assert r["ok"] and r["persona_id"] == pid and r["origin"] == "NEW LLM-generated synthetic response" and r["generation"]["respondent_profile"] == spec["synthetic_id"]


def test_all_four_personas_then_report_and_comparison(tmp_path):
    g = Gen()
    m = R(tmp_path, g)
    rs = [m.respond(body(p)) for p in PERSONAS]
    assert all(r["ok"] for r in rs) and len(g.persona_calls()) == 4 and len(g.calls) == 8      # one response + one summary per persona
    rep = m.report({"result_ids": [r["result_id"] for r in rs]})
    assert rep["ok"] and [c["persona_id"] for c in rep["comparison"]] == PERSONAS and len(g.calls) == 9
    analyst = g.calls[-1]
    assert analyst["system"] == pr.ANALYST_SYSTEM and all(ANSWER in analyst["user"] for _ in [0]) and analyst["max_output_tokens"] == 8192
    assert all(c["ssr_expected_rating"] is None and c["ssr_status"] == pr.UNRATED for c in rep["comparison"])
    assert rep["ratings_comparable"] is False and "not market evidence" in rep["labels"]["report"]


# ---------- no rating from the LLM; anchors
def test_llm_is_never_asked_for_a_rating(tmp_path):
    g = Gen()
    m = R(tmp_path, g)
    r = m.respond(body())
    m.report({"result_ids": [r["result_id"]]})
    for p in g.calls:
        assert "no ratings" in p["system"] or "never give numeric ratings" in p["system"] or "Never give numbers, scores or ratings" in p["system"]
        assert not re.search(r"\b(rate|score) (it|this|from)\b|out of 5|1 to 5", p["user"], re.I)


def test_new_product_question_is_unrated_with_the_required_message(tmp_path):
    for q in ("Would you consider buying this smartphone?", "Does the proposed price seem reasonable?", "How likely are you to recommend this product?"):
        r = R(tmp_path).respond(body(question=q))
        assert r["ok"] and r["ssr"] == {"available": False, "reason": "Likert rating unavailable: no compatible approved anchor set."}


def test_question_with_approved_anchors_is_scored_by_canonical_ssr(tmp_path):
    r = R(tmp_path).respond(body(question=Q26))
    s = r["ssr"]
    assert s["available"] and s["construct"] == "Behavioral Intention" and s["question_scored"] == Q26
    csv = pd.read_csv(ROOT / "docs/survey_instrument/smartphone_construct_anchor_statements.csv")
    stmts = list(csv[csv.construct == "Behavioral Intention"].sort_values("anchor_rating").anchor_statement)
    assert [a["statement"] for a in s["anchors"]] == stmts
    emb = fake_factory(wf.DEFAULT_ENCODER)
    sims = cosine_similarities(emb([ANSWER])[0], emb(stmts))
    pmf = similarities_to_pmf(sims)
    assert [a["cosine"] for a in s["anchors"]] == pytest.approx(list(sims)) and s["probabilities"] == pytest.approx(list(pmf))
    assert abs(sum(s["probabilities"]) - 1) <= 1e-12 and s["expected_rating"] == pytest.approx(expected_rating(pmf))
    assert s["expected_rating"] == pytest.approx(sum((i + 1) * p for i, p in enumerate(s["probabilities"])))


def test_rating_text_or_truncated_response_is_shown_but_not_rated(tmp_path):
    for text, fin in (("I would give it 4/5 because the camera is good.", "STOP"), ("The camera sounds good but the", "MAX_TOKENS")):
        r = R(tmp_path, Gen(persona_text=text, finish=fin)).respond(body(question=Q26))
        assert r["ok"] and r["ssr"]["available"] is False and "quality checks" in r["ssr"]["reason"]


# ---------- provider failures: no fake output
@pytest.mark.parametrize("kind", ["bad_request", "invalid_api_key", "rate_limit_or_quota", "model_not_found_or_no_access"])
def test_provider_failures_produce_errors_and_no_response(tmp_path, kind):
    err = {"kind": kind, "provider": {"http_status": {"bad_request": 400, "invalid_api_key": 400, "rate_limit_or_quota": 429, "model_not_found_or_no_access": 404}[kind],
                                      "provider_status": "X", "reason": "API_KEY_INVALID" if kind == "invalid_api_key" else None, "message": "provider said no"}}
    r = R(tmp_path, Gen(fail=err)).respond(body())
    assert r["ok"] is False and "response_text" not in r and "ssr" not in r and "No response was generated" in r["error"]
    if kind == "bad_request":
        assert "HTTP 400" in r["error"] and "provider said no" in r["error"]


def test_missing_key_is_reported_without_calling_the_llm(tmp_path):
    g = Gen()
    m = R(tmp_path, g, env={})
    assert m.options()["generation_available"] is False
    r = m.respond(body())
    assert r["error_code"] == "no_backend" and "GEMINI_API_KEY" in r["error"] and not g.calls


def test_report_failures_are_reported_not_filled_in(tmp_path):
    m = R(tmp_path, Gen(report="not json at all"))
    r = m.respond(body())
    rep = m.report({"result_ids": [r["result_id"]]})
    assert rep["ok"] is False and rep["error_code"] == "report_unparseable" and "report" not in rep
    m2 = R(tmp_path, Gen(report=json.dumps({"executive_summary": {}})))
    r2 = m2.respond(body())
    assert m2.report({"result_ids": [r2["result_id"]]})["error_code"] == "report_unparseable"
    assert m.report({"result_ids": ["nope"]})["error_code"] == "unknown_result" and m.report({"result_ids": []})["error_code"] == "invalid_input"


def test_report_numeric_market_claims_are_flagged(tmp_path):
    rep = json.loads(report_json(["P1_premium_tier_owner"]))
    rep["customer_segments"] = "About 40% of the market would buy it and market share could double."
    m = R(tmp_path, Gen(report=json.dumps(rep)))
    r = m.respond(body())
    out = m.report({"result_ids": [r["result_id"]]})
    assert out["ok"] and any("NOT supported by synthetic data" in w for w in out["report_warnings"])


def test_report_only_uses_responses_from_one_concept(tmp_path):
    m = R(tmp_path)
    a = m.respond(body())
    b = m.respond(body(concept="A rugged budget phone with a long battery life and a simple interface for older users."))
    assert m.report({"result_ids": [a["result_id"], b["result_id"]]})["error_code"] == "invalid_input"


def test_comparison_only_ranks_when_all_rated_with_the_same_construct(tmp_path):
    m = R(tmp_path)
    rs = [m.respond(body(p, question=Q26)) for p in PERSONAS[:2]]
    rep = m.report({"result_ids": [r["result_id"] for r in rs]})
    assert rep["ratings_comparable"] is True and all(isinstance(c["ssr_expected_rating"], float) for c in rep["comparison"])
    assert [c["ssr_expected_rating"] for c in rep["comparison"]] == [r["ssr"]["expected_rating"] for r in rs]


# ---------- server, page, protected files
def test_http_endpoints_and_page():
    httpd = srv.DashboardServer(("127.0.0.1", 0), srv.make_handler(da.Dashboard()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        o = json.loads(urllib.request.urlopen(base + "/api/product/options", timeout=30).read())
        assert o["available"] and o["title"] == "Digital Persona — Synthetic Consumer Research" and len(o["personas"]) == 4
        req = urllib.request.Request(base + "/api/product/respond", method="POST", data=json.dumps({**body(), "persona_id": "P9"}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.loads(urllib.request.urlopen(req, timeout=30).read())["error_code"] == "invalid_persona"
        html = urllib.request.urlopen(base + "/", timeout=30).read().decode("utf-8")
        assert "<title>Digital Persona — Synthetic Consumer Research</title>" in html
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_page_layout_and_error_handling_in_source():
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    assert "Analyze Product &amp; Generate Consumer Responses" in js and js.count('class="primary"') == 1
    for s in ("/api/product/options", "/api/product/respond", "/api/product/report", "All four", "Research question (optional)",
              "Proposed price (INR)", "Target customer group", "Main product features", "Intended market / geography",
              "LLM-generated synthetic response", "SSR-computed", "Persona comparison", "Executive summary", "Launch readiness", "No LLM is configured"):
        assert s in js
    assert not re.search(r"P[1-4]_[a-z]", js) and not re.search(r"\b[1-5]\.[0-9]{2,}\b", js) and not re.search(r"(sk-|AIza)[A-Za-z0-9]{8,}", js)


def test_protected_files_unchanged_by_the_workflow(tmp_path):
    h = lambda: {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PROTECTED}
    before = h()
    m = R(tmp_path)
    rs = [m.respond(body(p, question=Q26)) for p in PERSONAS]
    m.report({"result_ids": [r["result_id"] for r in rs]})
    assert h() == before


# ---------- simple summary, anchor comparison, explanation
def test_summary_is_shown_with_the_raw_response_kept_unchanged(tmp_path):
    g = Gen()
    r = R(tmp_path, g).respond(body())
    s = r["summary"]
    assert r["response_text"] == ANSWER and r["summary_error"] is None
    assert s["overall_opinion"] == "Interested but unsure about the price." and s["likes"] == ["The camera for reels"]
    assert s["important_features"] == ["camera", "battery"] and s["purchase_intention_note"] == pr.INTENTION_NOTE
    summary_prompt = next(c for c in g.calls if c["system"] == pr.SUMMARY_SYSTEM)
    assert ANSWER in summary_prompt["user"] and summary_prompt["json"] is True and "Use ONLY what the answer says" in summary_prompt["system"]


def test_summary_items_not_in_the_response_are_flagged_not_shown_as_fact(tmp_path):
    bad = json.dumps({"overall_opinion": "Interested.", "likes": ["The camera", "Waterproof titanium body"], "main_concern": "Price",
                      "important_features": ["camera", "satellite messaging"], "suggested_improvement": "Not mentioned", "purchase_intention": "Interested."})
    r = R(tmp_path, Gen(summary=bad)).respond(body())
    s = r["summary"]
    assert s["likes"] == ["The camera"] and s["important_features"] == ["camera"]
    assert set(s["not_in_response"]) == {"Waterproof titanium body", "satellite messaging"}


@pytest.mark.parametrize("summary", ["not json", json.dumps({"overall_opinion": "x"}),
                                     json.dumps({"overall_opinion": "I give it 4/5", "likes": [], "main_concern": "", "important_features": [],
                                                 "suggested_improvement": "", "purchase_intention": ""})])
def test_summary_failures_are_non_fatal_and_honest(tmp_path, summary):
    r = R(tmp_path, Gen(summary=summary)).respond(body())
    assert r["ok"] and r["summary"] is None and "Simple summary unavailable" in r["summary_error"] and r["response_text"] == ANSWER


def test_anchor_labels_and_explanation_come_from_actual_values(tmp_path):
    r = R(tmp_path).respond(body(question=Q26))
    s = r["ssr"]
    csv = pd.read_csv(ROOT / "docs/survey_instrument/smartphone_construct_anchor_statements.csv")
    bi = csv[csv.construct == "Behavioral Intention"].sort_values("anchor_rating")
    assert [a["label"] for a in s["anchors"]] == list(bi.anchor_label) and [a["statement"] for a in s["anchors"]] == list(bi.anchor_statement)
    top = max(s["anchors"], key=lambda a: a["cosine"])
    assert f"anchor {top['rating']} ({top['label']}" in s["explanation"] and f"{s['expected_rating']:.2f} out of 5" in s["explanation"]
    side = "agreeing" if s["expected_rating"] > 3.5 else "disagreeing" if s["expected_rating"] < 2.5 else "middle"
    assert side in s["explanation"]


def test_explanation_flags_nearly_equal_similarities():
    sc = {"anchors": [{"rating": i + 1, "label": l, "statement": f"s{i}", "cosine": c} for i, (l, c) in
                      enumerate(zip(["SD", "D", "N", "A", "SA"], [0.60, 0.61, 0.62, 0.615, 0.605]))], "expected_rating": 3.1}
    t = pr.explain_ssr(sc)
    assert "anchor 3 (N" in t and "almost equally similar" in t and "middle" in t


def test_unrated_question_still_gets_summary(tmp_path):
    r = R(tmp_path).respond(body(question="Does the proposed price seem reasonable?"))
    assert r["ssr"] == {"available": False, "reason": "Likert rating unavailable: no compatible approved anchor set."} and r["summary"]


def test_page_shows_the_six_parts_in_order():
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    order = ["1. Simple consumer response", "2. Key likes and concerns", "3. Comparison with the approved anchor statements",
             "4. Likert probabilities", "5. Expected SSR rating", "6. What this means"]
    fn = lambda name: js[js.index(f"function {name}("): js.index("\n}\n", js.index(f"function {name}("))]
    card, summ, ssr = fn("personaCard"), fn("summaryHtml"), fn("ssrHtml")
    assert card.index(order[0]) < card.index("${summaryHtml(r)}") < card.index("${ssrHtml(r.ssr)}")   # 1 -> summary (2) -> rating (3-6)
    assert order[1] in summ
    assert [ssr.index(x) for x in order[2:]] == sorted(ssr.index(x) for x in order[2:])
    for x in ("Overall opinion:", "Why they like it:", "Main concern:", "Most important features:", "Suggested improvement:", "Purchase intention:",
              "Original generated response (unchanged; this is the text SSR scores)"):
        assert x in js


def test_text_fields_and_numbers_not_in_the_response_are_replaced_and_flagged(tmp_path):
    bad = json.dumps({"overall_opinion": "Interested in the camera.", "likes": ["The camera"], "main_concern": "The price is a stretch.",
                      "important_features": ["camera", "8000 mAh battery"], "suggested_improvement": "Additional 1600 mAh for 256 GB storage",
                      "purchase_intention": "Recommend it to friends and family"})
    r = R(tmp_path, Gen(summary=bad)).respond(body())
    s = r["summary"]
    assert s["suggested_improvement"] == "Not clearly stated in the response." and s["important_features"] == ["camera"]
    assert "Additional 1600 mAh for 256 GB storage" in s["not_in_response"] and "8000 mAh battery" in s["not_in_response"]
    assert s["purchase_intention"] == "Recommend it to friends and family"      # interpretive field: only its numbers are checked
    assert s["main_concern"] == "The price is a stretch."


def test_explanation_reports_middle_for_a_neutral_closest_anchor():
    sc = {"anchors": [{"rating": i + 1, "label": l, "statement": f"s{i}", "cosine": c} for i, (l, c) in
                      enumerate(zip(["SD", "D", "N", "A", "SA"], [0.5916, 0.6116, 0.6611, 0.6074, 0.6096]))], "expected_rating": 3.258}
    t = pr.explain_ssr(sc)
    assert "anchor 3 (N" in t and "toward the middle" in t and "agreeing" not in t


def test_ollama_timeout_override_is_used(monkeypatch):
    from scripts import ollama_utils as ou
    seen = {}

    def fake_urlopen(req, timeout):
        seen["timeout"] = timeout
        raise TimeoutError()
    monkeypatch.setattr(ou.urllib.request, "urlopen", fake_urlopen)
    rec, err = ou.generate("m", {"system": "s", "user": "u", "timeout_s": 300})
    assert rec is None and err["kind"] == "network_or_timeout" and seen["timeout"] == 300


def test_repeated_summary_items_are_removed(tmp_path):
    rep = json.dumps({"overall_opinion": "Interested.", "likes": ["The camera", "the camera", "The camera"], "main_concern": "The price is a stretch.",
                      "important_features": ["camera", "battery", "Camera"], "suggested_improvement": "Not mentioned", "purchase_intention": "Interested."})
    s = R(tmp_path, Gen(summary=rep)).respond(body())["summary"]
    assert s["likes"] == ["The camera"] and s["important_features"] == ["camera", "battery"]

