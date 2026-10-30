import hashlib
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf
from dashboard import data_access as da
from dashboard import server as srv
from dashboard import workflow as wf

ROOT = Path(__file__).resolve().parents[1]
Q26 = "Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?"
TRYON = "Do you think AI virtual try-on technology will become widely used in the future?"


def fake_factory(cfg):
    return lambda xs: np.array([np.random.default_rng(int(hashlib.sha256((cfg + t).encode()).hexdigest()[:8], 16)).normal(size=16) for t in xs])


def gen(text="Honestly it was worth it for me, and I would probably stay with the same brand next time.", finish="STOP", calls=None):
    def g(provider, model, prompt):
        if calls is not None:
            calls.append(prompt)
        return {"response_text": text, "model_requested": model, "model_returned": model + "-001", "finish_reason": finish,
                "timestamp_utc": "2026-10-28T12:00:00+00:00"}, None
    return g


def W(tmp_path, generator=None, env=None):
    return wf.Workflow(ROOT, embedder_factory=fake_factory, generator=generator or gen(), env={"GEMINI_API_KEY": "x"} if env is None else env,
                       find_spec=lambda m: object(), log_dir=tmp_path / "r")


BODY = {"persona_id": "P1_premium_tier_owner", "question": Q26, "confirm_generation": True}


def test_options_use_the_real_personas_with_p1_default_and_the_requested_question(tmp_path):
    o = W(tmp_path).simple_options()
    cards = json.loads((ROOT / "data/persona_generator/persona_cards.json").read_text(encoding="utf-8"))
    assert o["title"] == "Digital Persona — Synthetic Consumer" and o["default_question"] == TRYON
    assert [p["persona_id"] for p in o["personas"]] == [c["persona_id"] for c in cards] and o["default_persona"] == "P1_premium_tier_owner"
    for p, c in zip(o["personas"], cards):
        assert c["defining_rule"] in p["description"] and str(c["n_human_respondents"]) in p["description"]
    assert o["generation_available"] is True
    assert W(tmp_path, env={}).simple_options()["generation_available"] is False


def test_survey_question_is_generated_and_rated_with_canonical_ssr_and_approved_anchors(tmp_path):
    calls = []
    r = W(tmp_path, gen(calls=calls)).simple_run(BODY)
    assert r["ok"] and r["rating_available"] and r["response_text"].startswith("Honestly")
    assert r["generation"]["model"] == __import__("scripts.gemini_utils", fromlist=["x"]).DEFAULT_MODEL + "-001" and r["construct"] == "Behavioral Intention"
    csv = pd.read_csv(ROOT / "docs/survey_instrument/smartphone_construct_anchor_statements.csv")
    stored = list(csv[csv.construct == "Behavioral Intention"].sort_values("anchor_rating").anchor_statement)
    assert [a["statement"] for a in r["anchors"]] == stored
    emb = fake_factory(wf.DEFAULT_ENCODER)
    sims = cosine_similarities(emb([r["response_text"]])[0], emb(stored))
    pmf = similarities_to_pmf(sims)
    assert [a["cosine"] for a in r["anchors"]] == pytest.approx(list(sims)) and r["probabilities"] == pytest.approx(list(pmf))
    assert sum(r["probabilities"]) == pytest.approx(1.0, abs=1e-12) and r["expected_rating"] == pytest.approx(expected_rating(pmf))
    assert "rating" not in calls[0]["user"].lower() and "no ratings" in calls[0]["system"]          # LLM asked for text only


def test_question_without_approved_anchors_gets_no_invented_rating(tmp_path):
    r = W(tmp_path).simple_run({**BODY, "question": TRYON})
    assert r["ok"] and r["rating_available"] is False and "No approved anchor statements" in r["rating_unavailable_reason"]
    assert "probabilities" not in r and "expected_rating" not in r and "anchors" not in r and r["response_text"]
    assert Q26 in r["questions_with_approved_anchors"]


@pytest.mark.parametrize("patch,code", [({"persona_id": "P9"}, "invalid_persona"), ({"question": ""}, "invalid_question"),
                                        ({"question": "Rate AI try-on from 1 to 5"}, "invalid_question"), ({"confirm_generation": False}, "confirmation_required")])
def test_bad_requests_rejected_without_calling_the_llm(tmp_path, patch, code):
    calls = []
    r = W(tmp_path, gen(calls=calls)).simple_run({**BODY, **patch})
    assert r["ok"] is False and r["error_code"] == code and not calls


def test_unavailable_llm_and_failures_give_errors_not_fake_results(tmp_path):
    r = W(tmp_path, env={}).simple_run(BODY)
    assert r["ok"] is False and r["error_code"] == "no_backend" and "GEMINI_API_KEY" in r["error"]
    r = W(tmp_path, generator=lambda *a: (None, {"kind": "rate_limit_or_quota"})).simple_run(BODY)
    assert r["error_code"] == "generation_failed" and "rate_limit_or_quota" in r["error"] and "response_text" not in r

    def boom(*a):
        raise RuntimeError("secret detail")
    r = W(tmp_path, generator=boom).simple_run(BODY)
    assert r["error_code"] == "generation_failed" and "secret detail" not in r["error"]


def test_truncated_or_rating_text_is_shown_but_not_rated(tmp_path):
    for text, fr in (("It was worth it and I would", "MAX_TOKENS"), ("I would say 4/5 overall, worth it.", "STOP")):
        r = W(tmp_path, gen(text=text, finish=fr)).simple_run(BODY)
        assert r["ok"] and r["rating_available"] is False and "quality checks" in r["rating_unavailable_reason"] and "expected_rating" not in r


def test_http_endpoints_for_the_page():
    httpd = srv.DashboardServer(("127.0.0.1", 0), srv.make_handler(da.Dashboard()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        o = json.loads(urllib.request.urlopen(base + "/api/simple/options", timeout=30).read())
        assert o["available"] and len(o["personas"]) == 4
        req = urllib.request.Request(base + "/api/simple/run", method="POST", data=json.dumps({**BODY, "persona_id": "P9"}).encode(),
                                     headers={"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=30).read())
        assert r["ok"] is False and r["error_code"] == "invalid_persona"         # validated before any provider call
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_page_shows_the_llm_warning_once_and_disables_the_button_without_a_backend():
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    assert js.count("No LLM is configured") == 1                                     # one warning, inside the form
    assert "run.busy || !opts.generation_available" in js and '${all ? "disabled" : ""}' in js   # button cannot request without a backend
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_GEN_MODEL"):
        assert name in js
    assert not __import__("re").search(r"(sk-|AIza)[A-Za-z0-9]{8,}", js)

