import json
import re
import threading
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

from dashboard import data_access as da
from dashboard import server as srv
from dashboard import workflow as wf

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
HTML = (ROOT / "dashboard/static/index.html").read_text(encoding="utf-8")


class Live:
    """Real dashboard server on a free port (same handler class the command line uses)."""
    def __init__(self, handler_factory=None, server_cls=None):
        dash = da.Dashboard()
        self.httpd = (server_cls or srv.DashboardServer)(("127.0.0.1", 0), (handler_factory or srv.make_handler)(dash) if handler_factory is None else handler_factory)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def call(self, path, method="GET", body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method, data=body, headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            r = urllib.request.urlopen(req, timeout=120)
            return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def live():
    s = Live()
    yield s
    s.close()


# ---------- the reported 404: frontend/backend route consistency
def test_every_api_path_used_by_the_page_is_served_by_the_server():
    used = set(re.findall(r"/api/[A-Za-z/]+", JS))
    assert {"/api/product/options", "/api/health", "/api/product/respond", "/api/product/report"} <= used
    dash = da.Dashboard()
    for path in sorted(used):
        if path == "/api/workflow/step/":                                   # called as POST /api/workflow/step/<n>
            assert all(f"/api/workflow/step/{n}" in srv.POST_ROUTES for n in range(1, 7))
            continue
        if path in srv.POST_ROUTES:                                         # e.g. POST /api/survey/generate
            continue
        status, _ = srv.route(dash, path, {})
        assert status != HTTPStatus.NOT_FOUND, f"{path} is used by app.js but the server answers 404"


def test_options_endpoint_works_over_http_with_the_schema_the_page_reads(live):
    s, b = live.call("/api/workflow/options")
    assert s == 200
    o = json.loads(b)                                                      # strict JSON
    assert o["available"] is True and o["title"] == "Digital Persona-Based Synthetic Survey Response Generation using SSR"
    assert {"personas", "tasks", "custom", "encoders", "backends", "saved_responses", "safety"} <= set(o)
    assert len(o["personas"]) == 4 and all({"persona_id", "n_human_respondents", "saved_scored_responses", "specs"} <= set(p) for p in o["personas"])
    assert all({"task_id", "label", "construct", "question_text", "profile_version"} <= set(t) for t in o["tasks"]) and len(o["tasks"]) == 3
    assert o["custom"]["id"] == "custom" and len(o["custom"]["constructs"]) == 6 and len(o["encoders"]) == 3
    assert sum(e["default"] for e in o["encoders"]) == 1
    assert all({"provider", "label", "available", "unavailable_reason", "model", "how_to_configure"} <= set(b_) for b_ in o["backends"])
    pids = {p["persona_id"] for p in o["personas"]}
    tids = {t["task_id"] for t in o["tasks"]}
    assert o["saved_responses"] and all(r["persona_id"] in pids and r["task_id"] in tids for r in o["saved_responses"])
    assert "sk-" not in b.decode("utf-8")


def test_health_endpoint_lists_the_routes_so_a_stale_server_can_be_detected(live):
    s, b = live.call("/api/health")
    h = json.loads(b)
    assert s == 200 and h["app"] == "digital-persona-dashboard" and "/api/workflow/options" in h["get_routes"] and h["server_started_utc"]
    for path in h["get_routes"]:
        assert live.call(path)[0] != 404, f"health advertises {path} but it answers 404"
    assert set(h["post_routes"]) == {f"/api/workflow/step/{n}" for n in range(1, 7)} | {"/api/survey/generate", "/api/simple/run", "/api/product/respond", "/api/product/report"}


def test_favicon_request_does_not_produce_a_404(live):
    assert live.call("/favicon.ico")[0] == 204


# ---------- root cause: a second server must not silently share the port
def test_second_server_cannot_bind_the_same_port(live):
    with pytest.raises(OSError):
        srv.DashboardServer(("127.0.0.1", live.port), srv.make_handler(da.Dashboard()))


def test_old_server_without_the_route_is_what_produces_the_404(live):
    class Old(BaseHTTPRequestHandler):                                      # simulates the server started before the route existed
        def do_GET(self):
            body = json.dumps({"error": "unknown endpoint"}).encode()
            self.send_response(404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    old = Live(handler_factory=Old, server_cls=srv.DashboardServer)
    try:
        assert old.call("/api/workflow/options")[0] == 404 and old.call("/api/health")[0] == 404
    finally:
        old.close()
    # the page reacts to exactly that situation with an explicit instruction (checked in the source) instead of a blank console error
    assert "older dashboard server" in JS and "Stop-Process" in JS and "/api/health" in JS


# ---------- single screen
def test_page_shows_only_the_requested_elements():
    assert "<nav" not in HTML and "data-view" not in HTML and "<title>Digital Persona — Synthetic Consumer Research</title>" in HTML
    assert JS.count('class="primary"') == 1 and "Analyze Product &amp; Generate Consumer Responses" in JS        # one primary action
    assert 'id="concept"' in JS and "<textarea" in JS and "p-one" in JS and 'id="question"' in JS
    for gone in ("encoder", "sw-survey", "Synthetic Survey Responses", "Validation", "Coverage", "c-enc", 'id="c-be"', 'id="s-be"', "LLM backend", "<select"):
        assert gone not in JS                                                                                    # no model/encoder selectors or extra pages

def test_six_steps_over_http_return_the_fields_the_page_displays(live):
    def post(n, body):
        s, b = live.call(f"/api/workflow/step/{n}", "POST", json.dumps(body).encode())
        assert s == 200
        return json.loads(b)
    r1 = post(1, {"persona_id": "P2_midrange_heavy_user", "task_id": "Q26_BI_proxy", "mode": "existing", "source_key": "gemini_v2", "synthetic_id": "P2_S01",
                  "encoder": "local_bge_base"})
    assert r1["ok"] and {"run_id", "persona", "question", "construct", "task", "mode", "text", "provenance", "quality", "stop", "llm_note"} <= set(r1)
    assert r1["provenance"]["origin"].startswith("EXISTING saved response") and r1["stop"] is None
    rid = {"run_id": r1["run_id"]}
    r2 = post(2, rid)
    if not r2["ok"] and r2.get("error_code") == "embedding_unavailable":
        pytest.skip("local sentence-transformer model not available")
    assert r2["ok"] and {"model", "dimensions", "norm", "first_values", "seconds", "note"} <= set(r2) and r2["dimensions"] == 768
    r3, r4, r5, r6 = (post(n, rid) for n in (3, 4, 5, 6))
    assert all(r["ok"] for r in (r3, r4, r5, r6))
    assert {"construct", "anchor_sha", "anchor_status", "anchors", "note"} <= set(r3) and len(r3["anchors"]) == 5 and all({"rating", "statement", "cosine"} <= set(a) for a in r3["anchors"])
    assert {"probabilities", "validation", "config", "zero_probability_ratings", "method"} <= set(r4) and r4["validation"]["valid"]
    assert {"expected_rating", "terms", "sum_of_terms", "matches_canonical_function", "formula", "who_does_what", "most_probable_rating"} <= set(r5) and r5["matches_canonical_function"]
    assert {"persona", "question", "text", "embedding_model", "similarities", "probabilities", "expected_rating", "provenance", "warnings", "anchor", "pmf_sd", "entropy"} <= set(r6)
    assert r6["expected_rating"] == r5["expected_rating"] and sum(r6["probabilities"]) == pytest.approx(1.0)
    assert any("they match" in w["text"] for w in r6["warnings"])            # recomputed with the real model: reproduces the stored dataset score


def test_workflow_failures_come_back_as_json_the_page_can_show(live):
    # Only selections that fail BEFORE any provider call are posted here: tests must never trigger a real (billed) generation request.
    for n, body, code in ((1, {"persona_id": "P1_premium_tier_owner", "task_id": "FT_PE01", "mode": "existing", "source_key": "gemini_v2", "synthetic_id": "P1_S01"}, "task_mismatch"),
                          (1, {"persona_id": "P1_premium_tier_owner", "task_id": "Q26_BI_proxy", "mode": "generate", "backend": "gemini"}, "confirmation_required"),
                          (3, {"run_id": "nope"}, "unknown_run")):
        s, b = live.call(f"/api/workflow/step/{n}", "POST", json.dumps(body).encode())
        r = json.loads(b)
        assert s == 200 and r["ok"] is False and r["error"] and r["error_code"] == code
