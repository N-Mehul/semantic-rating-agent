"""Background evaluation jobs: POST /api/evaluation/run returns at once, the page polls /api/evaluation/status/<id>.

These tests run the real HTTP server and the real ProductEvaluationService.run_evaluation (validation, E-SSR scoring,
saving) with a slow FAKE Qwen and a FAKE embedder. They prove the job mechanics, not real Qwen output quality.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from dashboard import data_access as da
from dashboard import server as srv

ROOT = Path(__file__).resolve().parents[1]

PRODUCT = {"name": "Nova 5G", "category": "Smartphone", "price_inr": 20000, "features": [],
           "description": "A 5G smartphone with a 6.6-inch 120Hz display, 5000mAh battery and 50MP camera."}
GOOD = {
    "PE": "The phone runs my apps fast and the battery lasts all day.",
    "EE": "It looks simple to set up and easy to use.",
    "HM": "Scrolling on the smooth screen would feel nice.",
    "SI": "My friends would probably like the camera.",
    "PP": "The description doesn't say how it protects my personal data, so I'm not sure yet.",
    "BI": "I think the price is fair for this camera, so I would probably buy it.",
    "overall": "A decent phone for the money.",
}


class SlowLLM:
    def __init__(self, delay=0.0, error=None):
        self.delay, self.error, self.calls = delay, error, 0

    def __call__(self, model, prompt, timeout=None, **kw):
        self.calls += 1
        time.sleep(self.delay)
        if self.error:
            return None, self.error
        if '{"answer": "..."}' in prompt["user"]:                # correction prompts are not expected with these answers
            return {"response_text": json.dumps({"answer": "Setting it up took only a few minutes for me."})}, None
        from tests.test_product_context_validation import _per_persona   # four distinct, validator-clean answer sets
        return {"response_text": json.dumps(_per_persona(prompt["user"]))}, None


class FakeEmbedder:
    def _v(self, t):
        return np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16)).normal(size=16)

    def embed_text(self, t):
        return self._v(t)

    def embed_texts(self, ts):
        return np.array([self._v(t) for t in ts])


READY = {"ollama_running": True, "generator_ready": True, "embedding_ready": True}


@pytest.fixture()
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "_EVALUATION", {})
    monkeypatch.setattr(srv, "_JOBS", {})
    dash = da.Dashboard()
    svc = srv.get_evaluation_service(dash)                      # the same service object the HTTP handler will use
    mod = sys.modules[type(svc).__module__]
    monkeypatch.setattr(mod, "EVAL_BASE_DIR", tmp_path)
    monkeypatch.setattr(mod, "get_ollama_embedding_adapter", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(svc, "check_ollama_backends", lambda: dict(READY))
    httpd = srv.DashboardServer(("127.0.0.1", 0), srv.make_handler(dash))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def call(path, body=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"} if body is not None else {})
        try:
            r = urllib.request.urlopen(req, timeout=30)
            return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def wait(job_id, timeout=60):
        seen, end = [], time.time() + timeout
        while time.time() < end:
            s, job = call(f"/api/evaluation/status/{job_id}")
            assert s == 200
            seen.append(job["status"])
            if job["status"] in ("completed", "failed"):
                return job, seen
            time.sleep(0.2)
        raise AssertionError(f"job {job_id} did not finish; saw {seen[-5:]}")

    yield type("Live", (), {"call": staticmethod(call), "wait": staticmethod(wait), "svc": svc, "dir": tmp_path})
    httpd.shutdown()
    httpd.server_close()


def test_run_returns_202_immediately_and_the_job_completes_in_the_background(live):
    live.svc.generate_fn = SlowLLM(delay=0.6)                   # 4 personas -> ~2.4 s of "Qwen" time
    t0 = time.time()
    status, payload = live.call("/api/evaluation/run", PRODUCT)
    assert status == 202 and payload["ok"] and time.time() - t0 < 2.0        # returns promptly, before generation finishes
    job_id = payload["job"]["job_id"]
    assert payload["job"]["status"] in ("queued", "running")
    assert live.call("/api/overview")[0] == 200                              # other endpoints keep working meanwhile
    assert live.call("/api/evaluation/options")[0] == 200
    s, active = live.call("/api/evaluation/active")
    assert s == 200 and active["job"]["job_id"] == job_id
    job, seen = live.wait(job_id)
    assert "running" in seen and job["status"] == "completed" and job["error"] is None
    assert job["progress"]["stage"] == "completed"
    result = job["result"]
    assert result["eval_id"] == job["eval_id"] and len(result["persona_summaries"]) == 4
    saved = live.dir / result["eval_id"]
    assert (saved / "product_concept.json").exists() and (saved / "ssr_results" / "ssr_results.json").exists()
    s, loaded = live.call(f"/api/evaluation/saved?id={result['eval_id']}")
    assert s == 200 and loaded["concept"]["name"] == "Nova 5G"               # saved exactly as before and reloadable
    assert live.call("/api/evaluation/active")[1]["job"] is None


def test_progress_is_reported_while_qwen_generates(live):
    live.svc.generate_fn = SlowLLM(delay=0.8)
    job_id = live.call("/api/evaluation/run", PRODUCT)[1]["job"]["job_id"]
    messages, end = set(), time.time() + 30
    while time.time() < end:
        job = live.call(f"/api/evaluation/status/{job_id}")[1]
        messages.add(job["progress"].get("message", ""))
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.15)
    assert any(m.startswith("Generating answers for P") for m in messages)
    assert "Evaluation complete" in messages


def test_duplicate_submission_is_refused_and_jobs_keep_separate_state(live):
    live.svc.generate_fn = SlowLLM(delay=0.5)
    first = live.call("/api/evaluation/run", PRODUCT)[1]["job"]["job_id"]
    s, dup = live.call("/api/evaluation/run", PRODUCT)
    assert s == 409 and dup["job"]["job_id"] == first                         # no second run while one is active
    job1, _ = live.wait(first)
    s, again = live.call("/api/evaluation/run", PRODUCT)
    assert s == 202 and again["job"]["job_id"] != first
    job2, _ = live.wait(again["job"]["job_id"])
    assert job1["status"] == job2["status"] == "completed"
    assert job1["eval_id"] != job2["eval_id"]                                      # each job saved its own evaluation
    assert live.call(f"/api/evaluation/status/{first}")[1]["status"] == "completed"          # first job unchanged


def test_failed_ollama_call_reports_the_real_error(live):
    live.svc.generate_fn = SlowLLM(error="Ollama request timed out after 360s")
    job_id = live.call("/api/evaluation/run", PRODUCT)[1]["job"]["job_id"]
    job, _ = live.wait(job_id)
    assert job["status"] == "failed" and "Ollama request timed out after 360s" in job["error"]
    assert live.call("/api/evaluation/active")[1]["job"] is None              # a failed job does not block the next one


def test_bad_input_and_unavailable_ollama_are_rejected_before_any_job(live):
    llm = SlowLLM()
    live.svc.generate_fn = llm
    s, p = live.call("/api/evaluation/run", dict(PRODUCT, name="x"))
    assert s == 400 and "at least 2 characters" in p["error"]
    s, p = live.call("/api/evaluation/run", dict(PRODUCT, name="apple",
                                                 description="The Samsung Galaxy M35 5G is a smartphone with a 6000mAh battery."))
    assert s == 400 and "does not appear in the description" in p["error"]                  # ProductContextError kept
    live.svc.check_ollama_backends = lambda: dict(READY, ollama_running=False)
    s, p = live.call("/api/evaluation/run", PRODUCT)
    assert s == 503 and "Ollama is not running" in p["error"]
    assert llm.calls == 0 and live.call("/api/evaluation/active")[1]["job"] is None


def test_status_endpoint_errors_are_explicit(live):
    s, p = live.call("/api/evaluation/status/doesnotexist")
    assert s == 404 and "not found" in p["error"]
    s, p = live.call("/api/evaluation/status")
    assert s == 400 and "Missing evaluation job id" in p["error"]


def test_server_defaults_to_port_8765_and_advertises_the_job_routes(live):
    src = (ROOT / "dashboard/server.py").read_text(encoding="utf-8")
    assert 'ap.add_argument("--port", type=int, default=8765)' in src
    h = live.call("/api/health")[1]
    assert {"/api/evaluation/status", "/api/evaluation/active"} <= set(h["get_routes"]) and "/api/evaluation/run" in h["post_routes"]


# ---------- frontend: submit once, poll, survive a network blip, render the result
def _js_fn(js, name):
    start = js.index(f"function {name}(")
    start = js.rfind("async ", 0, start) if js[max(0, start - 6):start] == "async " else start
    depth, i = 0, js.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(js[i], 0)
        i += 1
        if depth == 0:
            return js[start:i]


def test_frontend_polls_until_completed_and_survives_a_network_interruption():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    program = "\n".join([
        "const EVAL_POLL_MS = 1;",
        _js_fn(js, "apiCall"), _js_fn(js, "applyEvaluationResult"), _js_fn(js, "pollEvaluationJob"), _js_fn(js, "handleRunEvaluation"),
        "let renders = 0; function render() { renders++; }",
        "const window = { location: 'http://127.0.0.1:8765/' , history: { replaceState() {} } };",
        "const state = { isRunning: false, activeJobId: null, errorMessage: null, activeEvaluation: null, activeModal: 'new_eval',",
        "  currentView: 'product', runStatus: '', options: { saved_evaluations: [] }, form: { name: 'Nova 5G' } };",
        "const replies = [",
        "  { status: 202, body: { ok: true, job: { job_id: 'j1', status: 'queued' } } },",
        "  'NETWORK',",
        "  { status: 200, body: { job_id: 'j1', status: 'running', progress: { message: 'Generating answers for P2 (2 of 4 personas) with Qwen' } } },",
        "  { status: 200, body: { job_id: 'j1', status: 'completed', eval_id: 'nova_1', result: { eval_id: 'nova_1', concept: { name: 'Nova 5G' } } } },",
        "];",
        "const posts = []; const statuses = [];",
        "async function fetch(path, opts) {",
        "  if (opts && opts.method === 'POST') posts.push(path); else statuses.push(path);",
        "  const r = replies.shift();",
        "  if (r === 'NETWORK') throw new TypeError('Failed to fetch');",
        "  return { ok: r.status < 400, status: r.status, json: async () => r.body };",
        "}",
        "(async () => {",
        "  const first = handleRunEvaluation(null);",
        "  const second = handleRunEvaluation(null);           // double click while running",
        "  await Promise.all([first, second]);",
        "  console.log(JSON.stringify({ posts, statuses, isRunning: state.isRunning, err: state.errorMessage,",
        "    shown: state.activeEvaluation && state.activeEvaluation.eval_id, saved: state.options.saved_evaluations.length }));",
        "})();",
    ])
    out = subprocess.run(["node", "-e", program], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["posts"] == ["/api/evaluation/run"]                                   # submitted exactly once
    assert r["statuses"] == ["/api/evaluation/status/j1"] * 3                       # kept polling through the network error
    assert r["isRunning"] is False and r["err"] is None and r["shown"] == "nova_1" and r["saved"] == 1


def test_frontend_shows_the_real_failure_reason():
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    js = (ROOT / "dashboard/static/app.js").read_text(encoding="utf-8")
    program = "\n".join([
        "const EVAL_POLL_MS = 1;", _js_fn(js, "apiCall"), _js_fn(js, "applyEvaluationResult"), _js_fn(js, "pollEvaluationJob"),
        "function render() {}",
        "const window = { location: 'http://x/', history: { replaceState() {} } };",
        "const state = { isRunning: true, activeJobId: null, errorMessage: null, activeEvaluation: null, options: {} };",
        "async function fetch() { return { ok: true, status: 200, json: async () => ({ job_id: 'j9', status: 'failed',",
        "  error: 'Ollama request timed out after 360s (RuntimeError)' }) }; }",
        "(async () => { await pollEvaluationJob('j9'); console.log(JSON.stringify({ err: state.errorMessage, running: state.isRunning })); })();",
    ])
    out = subprocess.run(["node", "-e", program], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["err"] == "Ollama request timed out after 360s (RuntimeError)" and r["running"] is False


class UnscoredBILLM(SlowLLM):
    """Every BI answer (and its corrections) stays invalid, so BI is flagged and left unscored."""

    def __call__(self, model, prompt, timeout=None, **kw):
        self.calls += 1
        if '{"answer": "..."}' in prompt["user"]:
            return {"response_text": json.dumps({"answer": "Blue."})}, None
        from tests.test_product_context_validation import _per_persona
        return {"response_text": json.dumps(dict(_per_persona(prompt["user"]), BI="Blue."))}, None


def test_saved_run_with_unscored_bi_reloads(live):
    live.svc.generate_fn = UnscoredBILLM()
    job_id = live.call("/api/evaluation/run", PRODUCT)[1]["job"]["job_id"]
    job, _ = live.wait(job_id)
    assert job["status"] == "completed"
    assert all(ps["primary_adoption_rating"] is None for ps in job["result"]["persona_summaries"])
    s, loaded = live.call(f"/api/evaluation/saved?id={job['eval_id']}")                    # used to fail with HTTP 500
    assert s == 200 and all(ps.get("primary_adoption_rating") is None for ps in loaded["persona_summaries"])
