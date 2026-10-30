import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf
from dashboard import data_access as da
from dashboard import server as srv
from dashboard import workflow as wf

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CSV = ROOT / "docs/survey_instrument/smartphone_construct_anchor_statements.csv"
SECRET = "sk-SECRET-KEY-1234567890"


def fake_embed_factory(cfg):
    """Deterministic pseudo-embeddings (16 dims); no model is loaded."""
    def embed(texts):
        return np.array([np.random.default_rng(int(hashlib.sha256((cfg + t).encode()).hexdigest()[:8], 16)).normal(size=16) for t in texts])
    return embed


def no_spec(_):
    return object()


def W(tmp_path, env=None, generator=None, find_spec=None):
    return wf.Workflow(ROOT, embedder_factory=fake_embed_factory, generator=generator, env=env if env is not None else {},
                       log_dir=tmp_path / "runs", find_spec=find_spec or (lambda m: object()))


EXISTING = {"persona_id": "P1_premium_tier_owner", "task_id": "Q26_BI_proxy", "mode": "existing", "source_key": "gemini_v2", "synthetic_id": "P1_S01", "encoder": "local_bge_base"}


def run_all(w, body):
    r1 = w.run_step(1, body)
    assert r1["ok"], r1
    outs = [r1]
    for n in range(2, 7):
        outs.append(w.run_step(n, {"run_id": r1["run_id"]}))
    return outs


# ---------- backends
def test_backends_report_availability_without_exposing_keys():
    none = wf.backends_status({}, lambda m: object())
    assert all(not b["available"] for b in none)
    assert all("not set" in b["unavailable_reason"] for b in none if b["provider"] != "ollama")      # cloud providers: key missing
    ok = {b["provider"]: b for b in wf.backends_status({"GEMINI_API_KEY": SECRET, "OPENAI_API_KEY": SECRET}, lambda m: object())}
    env = {"GEMINI_API_KEY": SECRET, "OPENAI_API_KEY": SECRET}
    ok = {b["provider"]: b for b in wf.backends_status(env, lambda m: object())}
    assert ok["gemini"]["available"] and ok["gemini"]["model"] == __import__("scripts.gemini_utils", fromlist=["x"]).DEFAULT_MODEL
    assert not ok["openai"]["available"] and "no model chosen" in ok["openai"]["unavailable_reason"]       # OpenAI model is never assumed
    assert {b["provider"]: b for b in wf.backends_status({"OPENAI_API_KEY": SECRET, "OPENAI_GEN_MODEL": "m"}, lambda m: object())}["openai"]["available"]
    nosdk = wf.backends_status({"GEMINI_API_KEY": SECRET}, lambda m: None)
    g = {b["provider"]: b for b in nosdk}["gemini"]
    assert not g["available"] and "not installed" in g["unavailable_reason"]
    assert SECRET not in json.dumps(ok) + json.dumps(nosdk)


# ---------- validation helpers
@pytest.mark.parametrize("q,frag", [("", "empty"), ("   ", "empty"), ("x" * 500, "too long"), ("Rate your phone from 1 to 5", "rating"),
                                    ("Give it stars", "rating"), ("What is your Likert score?", "rating"), ("Tell me about Q21 please", "evaluation labels")])
def test_invalid_questions_rejected(q, frag):
    with pytest.raises(wf.StepError) as e:
        wf.validate_question(q)
    assert e.value.code == "invalid_question" and frag in e.value.message.lower()


def test_valid_question_accepted_and_trimmed():
    assert wf.validate_question("  How does your phone help you with studying?  ") == "How does your phone help you with studying?"


def test_probability_validation():
    good = similarities_to_pmf(np.array([0.5, 0.6, 0.7, 0.65, 0.62]))
    assert wf.validate_probabilities(good)["valid"] and wf.validate_probabilities(good)["sum"] == pytest.approx(1.0)
    for bad in ([0.2, 0.2, 0.2, 0.2], [0.5, 0.5, 0.2, 0.0, 0.0], [-0.1, 0.4, 0.3, 0.2, 0.2], [np.nan, 0.25, 0.25, 0.25, 0.25]):
        assert not wf.validate_probabilities(bad)["valid"]


# ---------- options
def test_options_come_from_saved_files_and_start_nothing(tmp_path):
    called = []
    w = W(tmp_path, generator=lambda *a: called.append(a))
    o = w.options()
    assert o["available"] and o["title"] == "Digital Persona-Based Synthetic Survey Response Generation using SSR"
    assert [p["persona_id"][:2] for p in o["personas"]] == ["P1", "P2", "P3", "P4"] and all(len(p["specs"]) == 10 for p in o["personas"])
    assert {p["persona_id"][:2]: p["saved_scored_responses"] for p in o["personas"]} == {"P1": 10, "P2": 10, "P3": 0, "P4": 0}
    assert [t["task_id"] for t in o["tasks"]] == ["Q26_BI_proxy", "FT_BI01", "FT_PE01"] and len(o["encoders"]) == 3
    assert len(o["saved_responses"]) == 20 and not called and not (tmp_path / "runs").exists()      # nothing generated or logged by loading options


def test_options_with_missing_files_report_unavailable(tmp_path):
    w = wf.Workflow(tmp_path, embedder_factory=fake_embed_factory, env={}, log_dir=tmp_path / "r")
    o = w.options()
    assert o["available"] is False and "Persona cards not found" in o["error"]


# ---------- full six-step chain on an existing saved response (fake embedder: tests the plumbing and the canonical calls)
def test_six_steps_use_canonical_functions_and_approved_anchors(tmp_path):
    w = W(tmp_path)
    r1, r2, r3, r4, r5, r6 = run_all(w, EXISTING)
    assert r1["provenance"]["origin"].startswith("EXISTING saved response") and "not newly generated" in r1["provenance"]["origin"]
    assert r1["text"] and r1["quality"]["safe_for_ssr"]
    assert r2["dimensions"] == 16 and r2["model"] == wf.ENCODERS["local_bge_base"] and len(r2["first_values"]) == 8       # from the real vector
    csv = pd.read_csv(ANCHOR_CSV)
    stored = list(csv[csv.construct == "Behavioral Intention"].sort_values("anchor_rating").anchor_statement)
    assert [a["statement"] for a in r3["anchors"]] == stored and [a["rating"] for a in r3["anchors"]] == [1, 2, 3, 4, 5]
    emb = fake_embed_factory("local_bge_base")
    vec = emb([r1["text"]])[0]
    sims = cosine_similarities(vec, emb(stored))
    assert [a["cosine"] for a in r3["anchors"]] == pytest.approx(list(sims))
    pmf = similarities_to_pmf(sims)
    assert [p["probability"] for p in r4["probabilities"]] == pytest.approx(list(pmf)) and r4["validation"]["valid"]
    assert r4["config"] == {"epsilon": 0.0, "temperature": 1.0, "probability_tolerance": 1e-12}      # canonical paper configuration, unchanged
    assert r4["zero_probability_ratings"] == [int(np.argmin(sims)) + 1]                              # min-subtraction: least similar anchor gets exactly 0
    assert r5["expected_rating"] == pytest.approx(expected_rating(pmf)) and r5["matches_canonical_function"]
    assert sum(t["product"] for t in r5["terms"]) == pytest.approx(r5["expected_rating"])
    assert r6["expected_rating"] == r5["expected_rating"] and r6["probabilities"] == pytest.approx(list(pmf)) and r6["similarities"] == pytest.approx(list(sims))
    assert r6["persona"]["persona_id"] == "P1_premium_tier_owner" and r6["embedding_model"] == wf.ENCODERS["local_bge_base"]
    texts = " ".join(x["text"] for x in r6["warnings"])
    assert "EXISTING saved response" in texts and "DIFFER" in texts            # fake embedder cannot reproduce the stored score: reported, not hidden
    assert json.dumps(r6, allow_nan=False)


def test_real_embedding_model_reproduces_stored_dataset_score(tmp_path):
    w = wf.Workflow(ROOT, log_dir=tmp_path / "runs", env={})                    # default (real, local) embedder factory
    r1 = w.run_step(1, EXISTING)
    r2 = w.run_step(2, {"run_id": r1["run_id"]})
    if not r2["ok"] and r2.get("error_code") == "embedding_unavailable":
        pytest.skip("local sentence-transformer model not available")
    assert r2["ok"] and r2["dimensions"] == 768
    for n in (3, 4, 5, 6):
        r = w.run_step(n, {"run_id": r1["run_id"]})
        assert r["ok"]
    df = pd.read_csv(ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv")
    row = df[df.synthetic_id == "P1_S01"].iloc[0]
    assert r["expected_rating"] == pytest.approx(row["local_bge_base__E"], abs=1e-6)
    assert [p for p in r["probabilities"]] == pytest.approx([row[f"local_bge_base__p{k}"] for k in range(1, 6)], abs=1e-6)
    assert any("they match" in x["text"] for x in r["warnings"])


def test_compare_encoders_option(tmp_path):
    r6 = run_all(W(tmp_path), {**EXISTING, "compare_encoders": True})[-1]
    assert [e["encoder"] for e in r6["encoder_comparison"]] == list(wf.ENCODERS) and r6["encoder_spread_sd"] is not None
    assert all(sum(e["probabilities"]) == pytest.approx(1.0) for e in r6["encoder_comparison"])


# ---------- existing-response mode errors
@pytest.mark.parametrize("patch,code", [({"persona_id": "P2_midrange_heavy_user"}, "persona_mismatch"), ({"task_id": "FT_PE01"}, "task_mismatch"),
                                        ({"synthetic_id": "NOPE"}, "no_such_response"), ({"source_key": ""}, "no_such_response"),
                                        ({"persona_id": "P9"}, "invalid_persona"), ({"task_id": "zzz"}, "invalid_task"), ({"encoder": "nope"}, "invalid_encoder"),
                                        ({"mode": "magic"}, "invalid_mode")])
def test_existing_mode_rejects_bad_selections(tmp_path, patch, code):
    r = W(tmp_path).run_step(1, {**EXISTING, **patch})
    assert r["ok"] is False and r["error_code"] == code and r["error"]


def test_custom_question_cannot_use_existing_response_and_needs_valid_construct(tmp_path):
    w = W(tmp_path)
    assert w.run_step(1, {**EXISTING, "task_id": "custom", "custom_question": "How is it?", "custom_construct": "PE"})["error_code"] == "task_mismatch"
    assert w.run_step(1, {**EXISTING, "task_id": "custom", "custom_question": "How is it?", "custom_construct": "XX"})["error_code"] == "invalid_construct"
    assert w.run_step(1, {**EXISTING, "task_id": "custom", "custom_question": "Rate it 1-5", "custom_construct": "PE"})["error_code"] == "invalid_question"


def test_excluded_saved_response_is_not_scored(tmp_path):
    root = tmp_path / "r"
    for rel in ("data/persona_generator/persona_cards.json", "data/persona_generator/respondent_specs_extended.jsonl"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text((ROOT / rel).read_text(encoding="utf-8"), encoding="utf-8")
    df = pd.read_csv(ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv", keep_default_na=False)
    df.loc[0, "status"] = "excluded_quality"
    (root / "data/synthetic_dataset").mkdir(parents=True)
    df.to_csv(root / "data/synthetic_dataset/synthetic_survey_dataset.csv", index=False)
    w = wf.Workflow(root, embedder_factory=fake_embed_factory, env={}, log_dir=tmp_path / "l")
    r = w.run_step(1, EXISTING)
    assert r["ok"] is False and r["error_code"] == "response_not_usable"
    assert w.run_step(1, {**EXISTING, "synthetic_id": "P1_S02"})["ok"]


def test_missing_dataset_reported(tmp_path):
    root = tmp_path / "r"
    (root / "data/persona_generator").mkdir(parents=True)
    (root / "data/persona_generator/persona_cards.json").write_text((ROOT / "data/persona_generator/persona_cards.json").read_text(encoding="utf-8"), encoding="utf-8")
    r = wf.Workflow(root, embedder_factory=fake_embed_factory, env={}, log_dir=tmp_path / "l").run_step(1, EXISTING)
    assert r["error_code"] == "missing_dataset"


# ---------- generation mode: missing backend, confirmation, mocked LLM
GEN = {"persona_id": "P3_midrange_moderate_user", "task_id": "FT_PE01", "mode": "generate", "backend": "gemini", "spec_id": "P3_S01", "encoder": "local_minilm",
       "confirm_generation": True}


def good_record(text="It helps me finish my college work much faster, and I use it for notes every day."):
    return {"synthetic_id": "P3_S01", "persona_id": "P3_midrange_moderate_user", "response_text": text, "model_requested": "gemini-2.5-flash",
            "model_returned": "gemini-2.5-flash", "finish_reason": "STOP", "timestamp_utc": "2026-01-01T00:00:00+00:00", "attempts": 1}


def test_generation_without_backend_reports_clearly_and_never_calls_llm(tmp_path):
    called = []
    w = W(tmp_path, env={}, generator=lambda *a: called.append(a))
    r = w.run_step(1, GEN)
    assert r["ok"] is False and r["error_code"] == "no_backend" and "GEMINI_API_KEY" in r["error"] and "existing saved response" in r["error"] and not called


def test_generation_requires_explicit_confirmation(tmp_path):
    called = []
    w = W(tmp_path, env={"GEMINI_API_KEY": SECRET}, generator=lambda *a: called.append(a))
    for conf in (False, None, "true", 1):
        r = w.run_step(1, {**GEN, "confirm_generation": conf})
        assert r["error_code"] == "confirmation_required"
    assert not called


def test_mocked_generation_full_chain_labelled_new_with_text_only_prompt(tmp_path):
    seen = []

    def gen(provider, model, prompt):
        seen.append((provider, model, prompt))
        return good_record(), None
    w = W(tmp_path, env={"GEMINI_API_KEY": SECRET}, generator=gen)
    outs = run_all(w, GEN)
    assert all(o["ok"] for o in outs)
    r1, r6 = outs[0], outs[-1]
    assert r1["provenance"]["origin"] == "NEW generation (this run)" and r1["provenance"]["model_requested"] == "gemini-2.5-flash"
    assert r1["provenance"]["spec_id"] == "P3_S01" and "did not provide any rating" in r1["llm_note"]
    prov, model, prompt = seen[0]
    assert (prov, model) == ("gemini", __import__("scripts.gemini_utils", fromlist=["x"]).DEFAULT_MODEL) and "no ratings" in prompt["system"]
    assert not re.search(r"Q21|likert|out of 5", prompt["system"] + prompt["user"], re.I) and "year-old" in prompt["user"]
    assert r6["construct"] == "PE" and r6["anchor"]["construct_name"] == "Performance Expectancy"      # FT_PE01 is scored with the PE anchors
    texts = " ".join(x["text"] for x in r6["warnings"])
    assert "single new generation" in texts and "No comparable human answers" in texts and "has no saved generated responses" in texts
    assert SECRET not in json.dumps(outs, default=str)


def test_custom_question_generation_uses_chosen_construct(tmp_path):
    seen = []
    w = W(tmp_path, env={"GEMINI_API_KEY": SECRET}, generator=lambda p, m, pr: (seen.append(pr), (good_record(), None))[1])
    outs = run_all(w, {**GEN, "task_id": "custom", "custom_question": "How easy was your phone to learn to use?", "custom_construct": "EE"})
    assert all(o["ok"] for o in outs) and outs[-1]["anchor"]["construct_name"] == "Effort Expectancy"
    assert "How easy was your phone to learn to use?" in seen[0]["user"]


def test_truncated_or_rating_response_is_flagged_and_not_scored(tmp_path):
    for text, frag in (("It helps me finish my college work and the", "no_terminal_punctuation"), ("I would say it is a 4/5 phone overall.", "contains_rating")):
        w = W(tmp_path, env={"GEMINI_API_KEY": SECRET}, generator=lambda *a, t=text: (good_record(t), None))
        r1 = w.run_step(1, GEN)
        assert r1["ok"] and r1["stop"] and "not scored" in r1["stop"]
        r2 = w.run_step(2, {"run_id": r1["run_id"]})
        assert r2["ok"] is False and r2["error_code"] == "not_scorable"


def test_provider_failures_are_reported_without_secrets(tmp_path):
    w = W(tmp_path, env={"GEMINI_API_KEY": SECRET}, generator=lambda *a: (None, {"kind": "rate_limit_or_quota"}))
    r = w.run_step(1, GEN)
    assert r["error_code"] == "generation_failed" and "rate_limit_or_quota" in r["error"]

    def boom(*a):
        raise RuntimeError(f"auth failed for {SECRET}")
    r = W(tmp_path, env={"GEMINI_API_KEY": SECRET}, generator=boom).run_step(1, GEN)
    assert r["error_code"] == "generation_failed" and SECRET not in json.dumps(r) and "RuntimeError" in r["error"]


def test_generation_rejects_bad_profile_or_backend(tmp_path):
    w = W(tmp_path, env={"GEMINI_API_KEY": SECRET}, generator=lambda *a: (good_record(), None))
    assert w.run_step(1, {**GEN, "spec_id": "P1_S01"})["error_code"] == "invalid_spec"          # profile of another persona
    assert w.run_step(1, {**GEN, "backend": "other"})["error_code"] == "invalid_backend"


# ---------- ordering, sessions, logging
def test_steps_must_run_in_order_and_unknown_runs_rejected(tmp_path):
    w = W(tmp_path)
    r1 = w.run_step(1, EXISTING)
    assert w.run_step(3, {"run_id": r1["run_id"]})["error_code"] == "step_order"
    assert w.run_step(5, {"run_id": r1["run_id"]})["error_code"] == "step_order"
    assert w.run_step(2, {"run_id": "nope"})["error_code"] == "unknown_run" and w.run_step(9, {})["error_code"] == "unknown_step"


def test_completed_run_is_logged_without_keys(tmp_path):
    w = W(tmp_path, env={"GEMINI_API_KEY": SECRET})
    run_all(w, EXISTING)
    log = (tmp_path / "runs" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 1 and SECRET not in log[0]
    rec = json.loads(log[0])
    assert rec["provenance"]["origin"].startswith("EXISTING") and rec["expected_rating"] and len(rec["probabilities"]) == 5


def test_nothing_in_project_data_was_written_by_workflow(tmp_path):
    before = hashlib.sha256((ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv").read_bytes()).hexdigest()
    run_all(W(tmp_path), EXISTING)
    assert hashlib.sha256((ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv").read_bytes()).hexdigest() == before


# ---------- server endpoints
def test_server_workflow_endpoints_and_post_protection():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(da.Dashboard()))
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"

    def call(path, method="GET", body=None, headers=None):
        req = urllib.request.Request(base + path, method=method, data=body, headers=headers or {})
        try:
            r = urllib.request.urlopen(req, timeout=15)
            return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    try:
        s, b = call("/api/workflow/options")
        assert s == 200 and json.loads(b)["available"] and len(json.loads(b)["personas"]) == 4
        js = {"Content-Type": "application/json"}
        s, b = call("/api/workflow/step/1", "POST", b'{"persona_id": "P9"}', js)
        assert s == 200 and json.loads(b)["ok"] is False and json.loads(b)["error_code"] == "invalid_persona"
        assert call("/api/workflow/step/1", "POST", b"not json", js)[0] == 400
        assert call("/api/workflow/step/1", "POST", b"[1,2]", js)[0] == 400
        assert call("/api/workflow/step/1", "POST", b"x" * 70000, js)[0] == 400
        assert call("/api/workflow/step/1", "POST", b"{}", {**js, "Origin": "http://evil.example"})[0] == 403
        assert call("/api/workflow/step/7", "POST", b"{}", js)[0] == 405 and call("/api/overview", "POST", b"{}", js)[0] == 405
        assert call("/api/workflow/step/1", "PUT", b"{}", js)[0] == 405 and call("/api/workflow/step/1", "DELETE")[0] == 405
        s, b = call("/api/workflow/step/2", "POST", b'{"run_id": "zzz"}', js)
        assert json.loads(b)["error_code"] == "unknown_run"
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------- front end
def test_frontend_page_uses_the_product_endpoints_and_hardcodes_nothing():
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    html = (ROOT / "dashboard/static/index.html").read_text(encoding="utf-8")
    assert "<title>Digital Persona — Synthetic Consumer Research</title>" in html
    assert "/api/product/options" in js and "/api/product/respond" in js and "/api/product/report" in js
    assert not re.search(r"P[1-4]_[a-z]", js) and not re.search(r"gemini-\d|gpt-\d|-flash\b", js.lower())

def test_workflow_module_never_exposes_keys_or_imports_providers_at_load():
    src = (ROOT / "dashboard/workflow.py").read_text(encoding="utf-8")
    head = src.split("class StepError")[0]
    assert "from google" not in head and "from openai" not in head and "import openai" not in head          # SDKs are imported lazily, inside generate
    assert "environ[" not in src                                      # keys are never read into results; only presence is checked via .get()
    assert not re.search(r"^\s*print\(", src, re.M) and "import logging" not in src       # nothing is printed or sent to a logger by the workflow
