import hashlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from canonical_ssr import expected_rating, similarities_to_pmf
from dashboard import product_research as pr
from dashboard import workflow as wf
from scripts import ollama_utils as ou

ROOT = Path(__file__).resolve().parents[1]
ANSWER = "The camera sounds useful for my classes, but the price is a bit high for me, so I would wait for a discount."
Q26 = "Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?"
CONCEPT = {"concept": "A smartphone with an AI camera, a 6500 mAh battery and fast charging for students in India.", "price_inr": "25000", "confirm_generation": True}


class FakeOllama:
    """Minimal stand-in for the Ollama HTTP API (/api/tags, /api/chat). Records every chat request it receives."""
    def __init__(self, models=("qwen2.5:0.5b",), delay=0.0, answer=ANSWER, done_reason="stop"):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/api/tags":
                    return self._send(200, {"models": [{"name": m} for m in models]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                time.sleep(delay)
                if body["model"] not in models:
                    return self._send(404, {"error": f"model '{body['model']}' not found"})
                self._send(200, {"model": body["model"], "message": {"role": "assistant", "content": answer}, "done": True, "done_reason": done_reason,
                                 "prompt_eval_count": 50, "eval_count": 40})

            def log_message(self, *a):
                pass
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def fake():
    f = FakeOllama()
    yield f
    f.close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def fake_factory(cfg):
    return lambda xs: np.array([np.random.default_rng(int(hashlib.sha256((cfg + t).encode()).hexdigest()[:8], 16)).normal(size=16) for t in xs])


# ---------- configuration and probe
def test_defaults():
    assert ou.DEFAULT_MODEL == "qwen2.5:0.5b" and ou.DEFAULT_HOST == "http://localhost:11434"
    assert ou.model({}) == "qwen2.5:0.5b" and ou.model({"OLLAMA_MODEL": "llama3.2:1b"}) == "llama3.2:1b"
    assert ou.host({"OLLAMA_HOST": "http://127.0.0.1:9999/"}) == "http://127.0.0.1:9999"


def test_probe_running_model_present_missing_and_not_running(fake):
    ok = ou.probe({"OLLAMA_HOST": fake.url})
    assert ok["running"] and ok["model_available"] and ok["error"] is None
    miss = ou.probe({"OLLAMA_HOST": fake.url, "OLLAMA_MODEL": "llama9:70b"})
    assert miss["running"] and not miss["model_available"] and "not installed" in miss["error"] and "qwen2.5:0.5b" in miss["error"]
    down = ou.probe({"OLLAMA_HOST": f"http://127.0.0.1:{free_port()}"}, timeout=1)
    assert not down["running"] and "not reachable" in down["error"] and "ollama serve" in down["error"]


# ---------- the real request shape and error handling
def test_chat_request_payload_and_record(fake):
    rec, err = ou.generate("qwen2.5:0.5b", {"system": "sys text", "user": "user text"}, env={"OLLAMA_HOST": fake.url})
    assert err is None and rec["response_text"] == ANSWER and rec["finish_reason"] == "STOP" and rec["model_returned"] == "qwen2.5:0.5b"
    body = fake.requests[0]
    assert body["model"] == "qwen2.5:0.5b" and body["stream"] is False
    assert body["messages"] == [{"role": "system", "content": "sys text"}, {"role": "user", "content": "user text"}]
    assert body["options"]["num_predict"] == 512 and "format" not in body


def test_length_cutoff_is_reported_as_max_tokens():
    f = FakeOllama(done_reason="length")
    try:
        rec, _ = ou.generate("qwen2.5:0.5b", {"system": "s", "user": "u"}, env={"OLLAMA_HOST": f.url})
        assert rec["finish_reason"] == "MAX_TOKENS"
    finally:
        f.close()


def test_not_running_unknown_model_and_timeout_are_errors_not_responses(fake):
    rec, err = ou.generate("qwen2.5:0.5b", {"system": "s", "user": "u"}, env={"OLLAMA_HOST": f"http://127.0.0.1:{free_port()}"}, timeout=15)
    assert rec is None and err["kind"] == "ollama_not_running"
    rec, err = ou.generate("llama9:70b", {"system": "s", "user": "u"}, env={"OLLAMA_HOST": fake.url})
    assert rec is None and err["kind"] == "model_not_found_or_no_access" and err["provider"]["http_status"] == 404
    slow = FakeOllama(delay=3)
    try:
        rec, err = ou.generate("qwen2.5:0.5b", {"system": "s", "user": "u"}, env={"OLLAMA_HOST": slow.url}, timeout=1)
        assert rec is None and err["kind"] == "network_or_timeout"
    finally:
        slow.close()
    assert "not running" in wf.explain_provider_error({"kind": "ollama_not_running", "provider": {"message": "x"}}, "ollama")


# ---------- provider selection
def test_ollama_needs_no_key_and_is_preferred_when_running(fake):
    env = {"OLLAMA_HOST": fake.url}
    bs = wf.backends_status(env, lambda m: object(), ou.probe)
    assert bs[0]["provider"] == "ollama" and bs[0]["available"] and bs[0]["model"] == "qwen2.5:0.5b"
    assert not any(b["available"] for b in bs[1:])                          # no Gemini/OpenAI key needed or used


def test_llm_provider_pins_without_silent_fallback(fake):
    env = {"OLLAMA_HOST": fake.url, "GEMINI_API_KEY": "x", "LLM_PROVIDER": "gemini"}
    bs = {b["provider"]: b for b in wf.backends_status(env, lambda m: object(), ou.probe)}
    assert bs["gemini"]["available"] and not bs["ollama"]["available"] and "not selected" in bs["ollama"]["unavailable_reason"]
    down = {"OLLAMA_HOST": f"http://127.0.0.1:{free_port()}", "GEMINI_API_KEY": "x", "LLM_PROVIDER": "ollama"}
    bs = wf.backends_status(down, lambda m: object(), lambda e: ou.probe(e, timeout=1))
    assert not any(b["available"] for b in bs)                             # Ollama pinned and down: Gemini is NOT used instead
    assert not any(b["available"] for b in wf.backends_status({"LLM_PROVIDER": "nonsense", "GEMINI_API_KEY": "x"}, lambda m: object(), ou.probe))


def test_tests_never_probe_the_real_local_server_unless_asked():
    called = []
    wf.backends_status({}, lambda m: object())                              # env dict, no probe injected
    bs = {b["provider"]: b for b in wf.backends_status({}, lambda m: object(), lambda e: called.append(e) or {"running": False, "model_available": False, "error": "x"})}
    assert called and bs["ollama"]["available"] is False


# ---------- end to end through the dashboard workflow (fake Ollama server, fake embeddings, canonical SSR)
def research(fake_url):
    env = {"OLLAMA_HOST": fake_url}
    w = wf.Workflow(ROOT, embedder_factory=fake_factory, env=env, find_spec=lambda m: object(), ollama_probe=ou.probe,
                    generator=lambda provider, model, prompt: ou.generate(model, prompt, env=env))
    return pr.ProductResearch(w)


@pytest.mark.parametrize("pid", ["P1_premium_tier_owner", "P2_midrange_heavy_user", "P3_midrange_moderate_user", "P4_budget_or_undisclosed_price"])
def test_each_persona_via_ollama_uses_real_profile_and_concept(fake, pid):
    r = research(fake.url).respond({**CONCEPT, "persona_id": pid})
    assert r["ok"] and r["generation"]["provider"] == "ollama" and r["generation"]["model"] == "qwen2.5:0.5b" and r["response_text"] == ANSWER
    sent = next(r for r in fake.requests if "format" not in r)["messages"]          # the scored response (the summary uses JSON mode)
    spec = next(json.loads(l) for l in (ROOT / "data/persona_generator/respondent_specs_extended.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(l)["persona_id"] == pid)
    assert spec["profile_text_v2"] in sent[1]["content"] and "AI camera" in sent[1]["content"] and "25,000 INR" in sent[1]["content"]
    assert "no ratings" in sent[0]["content"] and r["ssr"]["available"] is False                      # product question: no approved anchors


def test_ollama_text_scored_by_canonical_ssr_when_anchors_exist(fake):
    r = research(fake.url).respond({**CONCEPT, "persona_id": "P1_premium_tier_owner", "question": Q26})
    s = r["ssr"]
    emb = fake_factory(wf.DEFAULT_ENCODER)
    from canonical_ssr import cosine_similarities
    sims = cosine_similarities(emb([ANSWER])[0], emb([a["statement"] for a in s["anchors"]]))
    pmf = similarities_to_pmf(sims)
    assert s["available"] and s["probabilities"] == pytest.approx(list(pmf)) and s["expected_rating"] == pytest.approx(expected_rating(pmf))


def test_ollama_down_gives_clear_error_and_no_fake_response():
    m = research(f"http://127.0.0.1:{free_port()}")
    r = m.respond({**CONCEPT, "persona_id": "P1_premium_tier_owner"})
    assert r["ok"] is False and r["error_code"] == "no_backend" and "Ollama" in r["error"] and "response_text" not in r
    assert "not reachable" in m.options()["generation_unavailable_reason"]
