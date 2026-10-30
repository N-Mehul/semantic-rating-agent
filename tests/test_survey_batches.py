import csv
import hashlib
import io
import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf
from dashboard import data_access as da
from dashboard import server as srv
from dashboard import survey_batches as sb
from dashboard import workflow as wf

ROOT = Path(__file__).resolve().parents[1]
PROTECTED = ["data/human_survey_cleaned.csv", "data/Capstone_Reviews(Responses).xlsx", "docs/survey_instrument/smartphone_construct_anchor_statements.csv",
             "canonical_ssr.py", "data/pilot_gemini/gemini_generated_responses.jsonl", "data/pilot_gemini_v2/gemini_generated_responses.jsonl",
             "data/synthetic_dataset/synthetic_survey_dataset.csv", "data/personas/persona_assignments.csv"]
TEXTS = ["It helps me finish my college assignments quickly, though the battery barely lasts a full day.",
         "Honestly it has become slow, so even simple tasks take longer and that annoys me.",
         "It is fine for calls and social media but nothing special for anything else.",
         "I rely on it for payments and notes every day, and it rarely lets me down.",
         "Mostly it works, but the camera disappoints me whenever I need photos for class."]


def fake_factory(cfg):
    return lambda xs: np.array([np.random.default_rng(int(hashlib.sha256((cfg + t).encode()).hexdigest()[:8], 16)).normal(size=16) for t in xs])


def gen_from(texts, fail_at=None, kind="rate_limit_or_quota"):
    calls = []
    it = iter(texts)

    def gen(provider, model, prompt):
        calls.append(prompt)
        if fail_at is not None and len(calls) == fail_at:
            return None, {"kind": kind, "stop_run": True}
        return {"response_text": next(it), "model_requested": model, "model_returned": model + "-001", "finish_reason": "STOP",
                "timestamp_utc": "2026-10-28T10:00:00+00:00"}, None
    gen.calls = calls
    return gen


def manager(tmp_path, gen=None, env=None):
    w = wf.Workflow(ROOT, embedder_factory=fake_factory, generator=gen or gen_from(TEXTS), env={"GEMINI_API_KEY": "x"} if env is None else env,
                    find_spec=lambda m: object(), log_dir=tmp_path / "runs")
    return sb.BatchManager(w, store_dir=tmp_path / "batches")


REQ = {"persona_id": "P3_midrange_moderate_user", "task_id": "FT_PE01", "count": 5, "backend": "gemini", "confirm_generation": True, "encoder": "local_minilm"}


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PROTECTED}


# ---------- survey-data integration
def test_persona_context_is_aggregated_survey_evidence_without_individual_answers(tmp_path):
    m = manager(tmp_path)
    cards = {c["persona_id"]: c for c in json.loads((ROOT / "data/persona_generator/persona_cards.json").read_text(encoding="utf-8"))}
    ctx = sb.persona_context(cards["P2_midrange_heavy_user"])
    for t in cards["P2_midrange_heavy_user"]["experience_themes_Q23_Q25"]:
        assert f"{t['pct_of_usable_answers']}% of answers" in ctx
    assert str(cards["P2_midrange_heavy_user"]["n_human_respondents"]) in ctx
    r = m._resolve(REQ)
    prompts = [m.build_prompt(r, s) for s in r["specs"]]
    human = pd.read_csv(ROOT / "data/human_survey_cleaned.csv", keep_default_na=False, dtype=str)
    answers = {a.strip() for c in ("text_overall_experience", "text_battery_camera", "text_performance_display", "text_value_rebuy")
               for a in human[c] if len(a.split()) >= 5}
    blob = "\n".join(p["system"] + p["user"] for p in prompts)
    assert not any(a in blob for a in answers)                         # no individual respondent answer is reproduced
    assert not re.search(r"\bHS\d{3}\b", blob) and "Q21" not in blob and "rebuy" not in blob.lower()
    assert all(p["system"] == wf.SYSTEM for p in prompts)              # approved system prompt unchanged
    assert len({p["synthetic_id"] for p in prompts}) == 5              # a different real-data-sampled profile per response


def test_supported_questions_come_from_the_survey_instrument(tmp_path):
    m = manager(tmp_path)
    cb = pd.read_csv(ROOT / "docs/survey_instrument/survey_codebook.csv")
    q = cb[cb.question_id == "FT_PE01"].question_text.iloc[0]
    r = m._resolve(REQ)
    assert r["question"] == q and r["construct"] == "PE"
    assert m._resolve({**REQ, "task_id": "Q26_BI_proxy"})["question"].startswith("Do you think your smartphone is worth the price")
    rc = m._resolve({**REQ, "task_id": "custom", "custom_question": "How easy was it to learn your phone?", "custom_construct": "EE"})
    assert rc["construct"] == "EE"


@pytest.mark.parametrize("patch,code", [({"persona_id": "P9"}, "invalid_persona"), ({"task_id": "nope"}, "invalid_task"), ({"count": 0}, "invalid_count"),
                                        ({"count": 11}, "invalid_count"), ({"count": "x"}, "invalid_count"), ({"encoder": "bad"}, "invalid_encoder"),
                                        ({"confirm_generation": False}, "confirmation_required"), ({"backend": "nope"}, "invalid_backend"),
                                        ({"task_id": "custom", "custom_question": "Rate it from 1 to 5", "custom_construct": "PE"}, "invalid_question"),
                                        ({"task_id": "custom", "custom_question": "How is it?", "custom_construct": "PV"}, "invalid_construct")])
def test_invalid_requests_are_rejected_before_any_provider_call(tmp_path, patch, code):
    g = gen_from(TEXTS)
    r = manager(tmp_path, g).start({**REQ, **patch})
    assert r["ok"] is False and r["error_code"] == code and not g.calls


def test_missing_backend_is_reported_and_nothing_is_generated_or_saved(tmp_path):
    g = gen_from(TEXTS)
    m = manager(tmp_path, g, env={})
    r = m.start(REQ)
    assert r["error_code"] == "no_backend" and "GEMINI_API_KEY" in r["error"] and "saved responses remain available" in r["error"]
    assert not g.calls and not (tmp_path / "batches").exists()


# ---------- multiple responses, SSR on each
def test_batch_generates_multiple_scored_responses_with_canonical_ssr(tmp_path):
    m = manager(tmp_path)
    b = m.run_batch(REQ)
    assert b["counts"]["scored"] == 5 and len(b["items"]) == 5 and b["stopped_early_reason"] is None
    assert len({x["response_text"] for x in b["items"]}) == 5
    stmts = wf.anchor_info("PE")["statements"]
    assert b["anchors"]["statements"] == stmts and b["anchors"]["construct_name"] == "Performance Expectancy"
    emb = fake_factory("local_minilm")
    for x in b["items"]:
        assert x["origin"] == "NEW generation (this batch)" and x["model_requested"] == __import__("scripts.gemini_utils", fromlist=["x"]).DEFAULT_MODEL and x["model_returned"] == __import__("scripts.gemini_utils", fromlist=["x"]).DEFAULT_MODEL + "-001"
        sims = cosine_similarities(emb([x["response_text"]])[0], emb(stmts))
        pmf = similarities_to_pmf(sims)
        assert x["ssr"]["similarities"] == pytest.approx(list(sims)) and x["ssr"]["probabilities"] == pytest.approx(list(pmf))
        assert x["ssr"]["expected_rating"] == pytest.approx(expected_rating(pmf)) and x["ssr"]["expected_rating_check"]
        assert sum(x["ssr"]["probabilities"]) == pytest.approx(1.0, abs=1e-12) and all(p >= 0 for p in x["ssr"]["probabilities"])
        assert x["ssr"]["expected_rating"] == pytest.approx(sum((i + 1) * p for i, p in enumerate(x["ssr"]["probabilities"])))
        assert 1.0 <= x["ssr"]["expected_rating"] <= 5.0
    es = [x["ssr"]["expected_rating"] for x in b["items"]]
    assert b["expected_rating_summary"]["min"] == min(es) and b["expected_rating_summary"]["max"] == max(es)


def test_quota_failure_mid_batch_is_reported_honestly_and_partial_results_kept(tmp_path):
    g = gen_from(TEXTS, fail_at=3)
    m = manager(tmp_path, g)
    b = m.run_batch(REQ)
    st = [x["status"] for x in b["items"]]
    assert st == ["scored", "scored", "generation_failed", "not_attempted", "not_attempted"] and len(g.calls) == 3
    assert b["stopped_early_reason"] == "rate_limit_or_quota" and b["counts"]["scored"] == 2
    assert "ssr" not in b["items"][2] and "response_text" not in b["items"][2]          # nothing invented for the failed one
    assert b["saved"] and (tmp_path / "batches" / f"{b['batch_id']}.json").exists()


def test_batch_where_every_call_fails_is_not_saved(tmp_path):
    m = manager(tmp_path, gen_from(TEXTS, fail_at=1, kind="authentication_or_permission"))
    b = m.run_batch(REQ)
    assert b["counts"]["generation_failed"] == 1 and b["counts"]["not_attempted"] == 4 and not b["saved"]
    assert m.list_batches() == []


def test_provider_exception_becomes_a_failed_item(tmp_path):
    def boom(*a):
        raise RuntimeError("network down")
    b = manager(tmp_path, boom).run_batch({**REQ, "count": 2})
    assert [x["status"] for x in b["items"]] == ["generation_failed", "generation_failed"] and "RuntimeError" in b["items"][0]["warnings"][0]


def test_quality_failures_are_shown_but_not_scored_and_duplicates_flagged(tmp_path):
    texts = ["It helps a lot with my studies and the", "I would give it 4/5 for daily tasks.", TEXTS[0], TEXTS[0], TEXTS[1]]
    b = manager(tmp_path, gen_from(texts)).run_batch(REQ)
    st = [x["status"] for x in b["items"]]
    assert st[:2] == ["excluded_quality", "excluded_quality"] and "ssr" not in b["items"][0] and b["items"][0]["response_text"] == texts[0]
    assert any("Near-duplicate" in w for w in b["items"][3]["warnings"])


# ---------- display all and export
def test_all_generated_responses_are_listed_and_retrievable(tmp_path):
    m = manager(tmp_path, gen_from(TEXTS + TEXTS))
    b1 = m.run_batch(REQ)
    b2 = m.run_batch({**REQ, "persona_id": "P1_premium_tier_owner", "count": 3, "task_id": "Q26_BI_proxy"})
    ids = [x["batch_id"] for x in m.list_batches()]
    assert set(ids) == {b1["batch_id"], b2["batch_id"]}
    got = m.get_batch(b1["batch_id"])
    assert [x["response_id"] for x in got["items"]] == [x["response_id"] for x in b1["items"]] and len(got["items"]) == 5
    assert m.get_batch("../../etc") is None and m.get_batch("nope-not-there") is None


def test_export_contains_every_response_with_provenance_and_labels(tmp_path):
    b = manager(tmp_path, gen_from(TEXTS, fail_at=5)).run_batch(REQ)
    rows = sb.export_rows(b["items"], b["batch_id"])
    assert len(rows) == 5 and list(rows[0]) == sb.EXPORT_FIELDS
    text = sb.to_csv(rows)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert len(parsed) == 5 and all(r["origin"].startswith("NEW generation") for r in parsed)
    ok = [r for r in parsed if r["status"] == "scored"]
    assert len(ok) == 4 and all(abs(sum(float(r[f"p{k}"]) for k in range(1, 6)) - 1) < 1e-9 for r in ok)
    assert parsed[4]["status"] == "generation_failed" and parsed[4]["expected_rating"] == "" and parsed[4]["p1"] == ""   # blank, never 0
    assert all(r["model_requested"] and r["persona_id"] and r["question"] and r["anchor_sha"] for r in parsed)


def test_saved_responses_are_labelled_saved_with_stored_scores(tmp_path):
    m = manager(tmp_path)
    s = m.saved_pilot("local_bge_base")
    df = pd.read_csv(ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv")
    assert s["available"] and len(s["items"]) == len(df) == 20 and "not newly generated" in s["items"][0]["origin"]
    x = next(i for i in s["items"] if i["spec_id"] == "P1_S01")
    row = df[df.synthetic_id == "P1_S01"].iloc[0]
    assert x["ssr"]["expected_rating"] == pytest.approx(row["local_bge_base__E"]) and x["ssr"]["stored_scores"]
    assert x["anchor_statements"] == wf.anchor_info("BI")["statements"]
    assert {i["persona_id"][:2] for i in s["items"]} == {"P1", "P2"}           # the saved pilot covers only two personas


# ---------- HTTP endpoints (manager injected with a temporary store; no real provider call)
def test_http_endpoints_list_show_export_and_refuse_generation_without_backend(tmp_path):
    dash = da.Dashboard()
    m = manager(tmp_path, gen_from(TEXTS), env={"GEMINI_API_KEY": "x"})
    b = m.run_batch(REQ)
    old = srv._BATCHES.get(str(dash.root))
    srv._BATCHES[str(dash.root)] = m
    httpd = srv.DashboardServer(("127.0.0.1", 0), srv.make_handler(dash))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, method="GET", body=None):
        req = urllib.request.Request(base + path, method=method, data=body, headers={"Content-Type": "application/json"} if body else {})
        try:
            r = urllib.request.urlopen(req, timeout=60)
            return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers
    try:
        s, body, _ = call("/api/survey/batches")
        assert s == 200 and json.loads(body)["batches"][0]["batch_id"] == b["batch_id"]
        s, body, _ = call(f"/api/survey/batch?id={b['batch_id']}")
        assert s == 200 and len(json.loads(body)["items"]) == 5
        s, body, h = call(f"/api/survey/export?source=batch&id={b['batch_id']}&format=csv")
        assert s == 200 and "attachment" in h["Content-Disposition"] and len(list(csv.DictReader(io.StringIO(body.decode())))) == 5
        s, body, _ = call(f"/api/survey/export?source=batch&id={b['batch_id']}&format=json")
        assert s == 200 and len(json.loads(body)) == 5
        s, body, _ = call("/api/survey/export?source=saved&format=json")
        assert s == 200 and len(json.loads(body)) == 20 and all("SAVED" in r["origin"] for r in json.loads(body))
        assert call("/api/survey/export?source=batch&format=csv")[0] == 400 and call("/api/survey/export?source=batch&id=zzzzzz-1&format=csv")[0] == 404
        assert call("/api/survey/export?source=x&format=csv")[0] == 400 and call("/api/survey/job")[0] == 400
        s, body, _ = call("/api/survey/saved")
        assert s == 200 and len(json.loads(body)["items"]) == 20
        m.w._env = {}                                             # backend removed: generation must be refused, previous batch kept
        s, body, _ = call("/api/survey/generate", "POST", json.dumps(REQ).encode())
        assert s == 200 and json.loads(body)["error_code"] == "no_backend"
        assert len(json.loads(call("/api/survey/batches")[1])["batches"]) == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        if old is None:
            srv._BATCHES.pop(str(dash.root), None)
        else:
            srv._BATCHES[str(dash.root)] = old


def test_background_job_reports_progress_and_finishes(tmp_path):
    m = manager(tmp_path)
    r = m.start({**REQ, "count": 3})
    assert r["ok"]
    for _ in range(200):
        j = m.job(r["job_id"])
        if j["state"] != "running":
            break
        threading.Event().wait(0.05)
    assert j["state"] == "finished" and j["counts"]["scored"] == 3 and j["batch_id"]
    assert m.job("nope") is None


# ---------- no automatic generation, original files untouched
def test_reading_endpoints_never_generate_and_research_files_stay_unchanged(tmp_path):
    before = hashes()
    g = gen_from(TEXTS)
    m = manager(tmp_path, g)
    m.saved_pilot()
    m.list_batches()
    m.w.options()
    assert not g.calls
    b = m.run_batch(REQ)
    sb.to_csv(sb.export_rows(b["items"], b["batch_id"]))
    assert hashes() == before
