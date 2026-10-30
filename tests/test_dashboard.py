import json
import re
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from dashboard import data_access as da
from dashboard import server as srv

ROOT = Path(__file__).resolve().parents[1]
ENC = da.ENCODERS


def dump(x):
    return json.dumps(x, allow_nan=False)           # raises on NaN/inf: every payload must be strict JSON


# ---------------- fixtures for synthetic roots
def write(root, rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def mini_root(tmp_path, with_dataset=True, partial=False, blank_scores=False, excluded=False):
    cards = [{"persona_id": "P1_a", "n_human_respondents": 5, "pct_of_primary_cohort": 50.0, "defining_rule": "r", "median_age": 20, "age_range": [18, 25]},
             {"persona_id": "P2_b", "n_human_respondents": 5, "pct_of_primary_cohort": 50.0, "defining_rule": "r", "median_age": 21, "age_range": [19, 26]}]
    write(tmp_path, da.FILES["persona_cards"], json.dumps(cards))
    write(tmp_path, da.FILES["respondent_specs"], "\n".join(json.dumps({"synthetic_id": f"P{i}_S01", "persona_id": p}) for i, p in ((1, "P1_a"), (2, "P2_b"))))
    cov = ("source_key,persona_id,planned,responses_on_disk,passed_quality_and_scored,not_generated,coverage_status\n"
           f"srcA,P1_a,2,{1 if partial else 2},{1 if partial else 2},{1 if partial else 0},{'partial' if partial else 'complete'}\n"
           "srcA,P2_b,2,0,0,2,none\n")
    write(tmp_path, da.FILES["coverage"], cov)
    if with_dataset:
        cols = ["source_key", "synthetic_id", "persona_id", "status", "response_text", "model_requested", "profile_version", "brand", "age"]
        for c in ENC[:1]:
            cols += [f"{c}__{s}" for s in ["E", "sd", "entropy", "argmax"] + [f"p{k}" for k in range(1, 6)] + [f"s{k}" for k in range(1, 6)]]
        vals = lambda e: [str(e), "1.0", "1.2", "3", "0.0", "0.1", "0.4", "0.3", "0.2", "0.5", "0.6", "0.7", "0.65", "0.6"]
        row1 = ["srcA", "P1_S01", "P1_a", "scored", "NA", "m1", "v1", "Samsung", ""] + ([""] * 14 if blank_scores else vals(3.8))
        rows = [row1]
        if excluded:
            rows.append(["srcA", "P1_S02", "P1_a", "excluded_quality", "cut off and", "m1", "v1", "Vivo", ""] + [""] * 14)
        write(tmp_path, da.FILES["dataset"], ",".join(cols) + "\n" + "\n".join(",".join(r) for r in rows) + "\n")
    write(tmp_path, da.FILES["dataset_manifest"], json.dumps({"sources": {"srcA": {"anchors": ["a1", "a2", "a3", "a4", "a5"]}}, "limitations": ["L1"]}))
    return tmp_path


# ---------------- actual saved outputs
def test_real_outputs_load_and_are_strict_json():
    d = da.Dashboard()
    for payload in (d.overview(), d.personas(), d.responses(), d.coverage(), d.validation(), d.limitations()):
        dump(payload)
    o = d.overview()
    assert o["n_files"] == o["n_files_ok"] == len(da.FILES) and o["scored_responses"] == 20 and o["models"] == ["gemini-2.5-flash"]


def test_real_personas_distinguish_with_and_without_responses():
    p = da.Dashboard().personas()
    assert [x["card"]["persona_id"][:2] for x in p["personas"]] == ["P1", "P2", "P3", "P4"]
    assert [x[:2] for x in p["personas_with_responses"]] == ["P1", "P2"] and [x[:2] for x in p["personas_without_responses"]] == ["P3", "P4"]
    for x in p["personas"]:
        assert (x["scored_responses_total"] > 0) == x["has_generated_responses"]
        assert x["card"]["n_human_respondents"] > 0


def test_real_response_detail_matches_dataset_and_probabilities_sum_to_one():
    import pandas as pd
    d = da.Dashboard()
    df = pd.read_csv(ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv", keep_default_na=False)
    r = d.responses()
    assert r["n"] == len(df) == 20 and all(x["model_requested"] == "gemini-2.5-flash" and x["source_key"] == "gemini_v2" for x in r["rows"])
    det = d.response_detail("gemini_v2", "P1_S01")
    row = df[df.synthetic_id == "P1_S01"].iloc[0]
    assert det["response_text"] == row.response_text and det["scored"] and len(det["anchors"]) == 5
    for c in ENC:
        e = det["encoders"][c]
        assert e["probability_sum"] == pytest.approx(1.0, abs=1e-9) and len(e["similarities"]) == 5
        assert e["expected_rating"] == pytest.approx(row[f"{c}__E"]) and e["probabilities"][2] == pytest.approx(row[f"{c}__p3"])
        assert e["expected_rating"] == pytest.approx(sum((i + 1) * p for i, p in enumerate(e["probabilities"])))
    assert det["persona_attributes"]["age"] is None            # v1 prompt version has no age: unavailable, not zero or blank


def test_real_coverage_warnings_name_missing_personas_and_sources():
    c = da.Dashboard().coverage()
    msgs = " ".join(w["message"] for w in c["warnings"])
    assert "P3_midrange_moderate_user has no generated responses in any source" in msgs and "P4_budget_or_undisclosed_price has no generated" in msgs
    assert "Source openai_v1 has zero saved responses" in msgs and "stability is not supported" in msgs and "Encoders disagree" in msgs
    gem = [r for r in c["rows"] if r["source_key"] == "gemini_v2"]
    assert {r["persona_id"][:2]: r["coverage_status"] for r in gem} == {"P1": "complete", "P2": "complete", "P3": "none", "P4": "none"}


def test_real_validation_results_loaded_without_equivalence_claim():
    v = da.Dashboard().validation()
    h = v["distribution_headline"]
    assert h["n_comparisons"] == 6 and h["equivalence_claimed_any"] is False and h["n_with_interval"] == 6
    assert v["stability"]["status"] == "not_supported_by_data" and len(v["encoder_agreement"]) == 3
    assert all(0 <= abs(x["spearman_E"]) < 0.5 for x in v["encoder_agreement"]) and len(v["consistency_summary"]) == 2
    assert {r["persona_id"][:2] for r in v["distribution"]} == {"P1", "P2"}


# ---------------- missing, empty, malformed, incomplete
def test_empty_root_degrades_gracefully_everywhere(tmp_path):
    d = da.Dashboard(tmp_path)
    o = d.overview()
    assert o["n_files_ok"] == 0 and o["scored_responses"] is None and dump(o)
    for fn, key in ((d.personas, "personas"), (d.responses, "rows")):
        r = fn()
        assert r["available"] is False and r["error"] and r[key] == []
    assert d.coverage()["available"] is False and d.validation()["available"] is False
    assert d.response_detail("a", "b")["available"] is False
    assert any("Coverage table unavailable" in w["message"] for w in d.coverage()["warnings"])
    dump(d.limitations())


def test_malformed_and_empty_files_do_not_raise(tmp_path):
    write(tmp_path, da.FILES["persona_cards"], "{not json")
    write(tmp_path, da.FILES["dataset"], "")
    write(tmp_path, da.FILES["coverage"], "")
    write(tmp_path, da.FILES["validation_summary"], "   ")
    write(tmp_path, da.FILES["distribution"], "x,y\n")
    d = da.Dashboard(tmp_path)
    assert d.personas()["available"] is False and d.responses()["available"] is False
    assert d.coverage()["available"] is False and d.validation()["available"] in (True, False)
    assert dump(d.overview())
    bad = tmp_path / da.FILES["dataset"]
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    assert "missing columns" in da.Dashboard(tmp_path).responses()["error"]


def test_no_responses_saved_gives_empty_state_not_zeros(tmp_path):
    root = mini_root(tmp_path, with_dataset=False)
    d = da.Dashboard(root)
    assert d.responses()["available"] is False
    p = d.personas()
    assert p["available"] and p["personas_with_responses"] == ["P1_a"]    # coverage file claims P1_a has scored responses, dataset absent
    assert d.overview()["scored_responses"] is None                       # unknown, not 0


def test_incomplete_persona_coverage_flagged(tmp_path):
    d = da.Dashboard(mini_root(tmp_path, partial=True))
    msgs = [w["message"] for w in d.coverage()["warnings"]]
    assert any("Partial coverage for P1_a" in m and "1 scored of 2" in m for m in msgs)
    assert any("No generated responses for P2_b" in m for m in msgs) and any("P2_b has no generated responses in any source" in m for m in msgs)
    p = d.personas()
    assert p["personas_without_responses"] == ["P2_b"] and p["personas_with_responses"] == ["P1_a"]


def test_blank_scores_become_none_not_zero(tmp_path):
    d = da.Dashboard(mini_root(tmp_path, blank_scores=True))
    det = d.response_detail("srcA", "P1_S01")
    e = det["encoders"][ENC[0]]
    assert e["expected_rating"] is None and e["probabilities"] == [None] * 5 and e["probability_sum"] is None and e["similarities"] is None
    assert d.responses()["rows"][0]["expected_rating"][ENC[0]] is None
    dump(det)


def test_excluded_response_has_no_scores_and_literal_NA_text_is_kept(tmp_path):
    d = da.Dashboard(mini_root(tmp_path, excluded=True))
    ex = d.response_detail("srcA", "P1_S02")
    assert ex["scored"] is False and ex["encoders"][ENC[0]] is None
    assert d.response_detail("srcA", "P1_S01")["response_text"] == "NA"      # literal NA is a real answer, not missing
    assert d.response_detail("srcA", "nope") is None


# ---------------- server
def test_route_function_status_codes():
    d = da.Dashboard()
    assert srv.route(d, "/api/overview", {})[0] == 200
    assert srv.route(d, "/api/response", {})[0] == 400
    assert srv.route(d, "/api/response", {"source": ["gemini_v2"], "id": ["nope"]})[0] == 404
    assert srv.route(d, "/api/unknown", {})[0] == 404
    status, body = srv.route(d, "/api/response", {"source": ["gemini_v2"], "id": ["P1_S01"]})
    assert status == 200 and body["synthetic_id"] == "P1_S01"


def test_http_server_serves_pages_api_and_refuses_writes_and_traversal():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), srv.make_handler(da.Dashboard()))
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{port}"

    def get(path, method="GET"):
        try:
            r = urllib.request.urlopen(urllib.request.Request(base + path, method=method), timeout=10)
            return r.status, r.read(), r.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers
    try:
        s, b, h = get("/")
        assert s == 200 and "Digital Persona — Synthetic Consumer" in b.decode("utf-8") and "text/html" in h["Content-Type"]
        assert get("/app.js")[0] == 200 and get("/styles.css")[0] == 200
        s, b, h = get("/api/personas")
        assert s == 200 and "application/json" in h["Content-Type"] and len(json.loads(b)["personas"]) == 4
        assert get("/api/overview", "POST")[0] == 405 and get("/api/overview", "DELETE")[0] == 405
        for bad in ("/../scripts/run_pipeline.py", "/data/human_survey_cleaned.csv", "/canonical_ssr.py", "/%2e%2e/README.md"):
            assert get(bad)[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


# ---------------- safety / no hardcoded research values
def test_frontend_has_no_hardcoded_personas_models_or_results():
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    html = (ROOT / "dashboard/static/index.html").read_text(encoding="utf-8")
    for text in (js, html):
        assert not re.search(r"P[1-4]_[a-z]", text), "persona ids must come from the data"
        no_env = re.sub(r"(GEMINI|OPENAI)_[A-Z_]+", "", text)          # env-var names in the "how to configure" help are instructions, not model names
        assert not re.search(r"gemini-\d|gpt-\d|-flash\b|-mini\b", no_env.lower()), "model names must come from the data"
        assert not re.search(r"\b3\.[0-9]{3}\b", text), "no literal rating values"


def test_dashboard_code_is_read_only_and_offline():
    for f in ("dashboard/data_access.py", "dashboard/server.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        for bad in ("import openai", "from openai", "google.genai", "import requests", "urllib.request", "canonical_ssr", "write_text", "write_bytes",
                    ".to_csv", "open(", "subprocess", "sentence_transformers", "ollama"):
            assert bad not in src, f"{f} must stay read-only/offline ({bad})"
