"""Local Ollama provider (no API key). Standard library only.

Configuration (environment of the process that starts the dashboard; all optional):
  OLLAMA_HOST   default http://localhost:11434
  OLLAMA_MODEL  default qwen2.5:0.5b
  LLM_PROVIDER  ollama | gemini | openai  - pins the provider; when unset the dashboard prefers a running Ollama
The model writes text only; every rating is computed afterwards by canonical SSR.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import socket
import urllib.error
import urllib.request

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:0.5b"
PROBE_TIMEOUT_S = 2
CHAT_TIMEOUT_S = 120
SETTINGS = {"temperature": 1.0, "num_predict": 512}


def host(env=None) -> str:
    env = os.environ if env is None else env
    return (env.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")


def model(env=None) -> str:
    env = os.environ if env is None else env
    return (env.get("OLLAMA_MODEL") or DEFAULT_MODEL).strip()


def _get_json(url: str, timeout: float):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def probe(env=None, timeout: float = PROBE_TIMEOUT_S) -> dict:
    """Is the Ollama server running, and is the configured model installed? Never raises."""
    h, m = host(env), model(env)
    try:
        tags = _get_json(h + "/api/tags", timeout)
    except Exception as e:
        return {"running": False, "model_available": False, "models": [], "host": h, "model": m,
                "error": f"Ollama is not reachable at {h} ({type(e).__name__}). Start it with 'ollama serve' (or open the Ollama app)."}
    names = [x.get("name") or x.get("model") for x in tags.get("models", [])]
    ok = m in names or (":" not in m and f"{m}:latest" in names)
    return {"running": True, "model_available": ok, "models": names, "host": h, "model": m,
            "error": None if ok else f"Ollama is running but the model '{m}' is not installed (installed: {', '.join(names) or 'none'})."}


def classify(exc: BaseException, body: str = "") -> str:
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code == 404 or "not found" in body.lower():
            return "model_not_found_or_no_access"
        return "bad_request" if 400 <= exc.code < 500 else "server_error"
    reason = getattr(exc, "reason", exc)
    if isinstance(exc, (socket.timeout, TimeoutError)) or isinstance(reason, (socket.timeout, TimeoutError)):
        return "network_or_timeout"
    return "ollama_not_running"


def generate(model_name: str, prompt: dict, env=None, timeout: float = CHAT_TIMEOUT_S):
    """POST /api/chat (non-streaming). Returns (record, None) or (None, error) in the same shape as the other providers."""
    payload = {"model": model_name, "stream": False,
               "messages": [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}],
               "options": {"temperature": SETTINGS["temperature"], "num_predict": int(min(prompt.get("max_output_tokens") or SETTINGS["num_predict"], 4096))}}
    timeout = float(prompt.get("timeout_s") or timeout)      # e.g. the long analyst report on a CPU-only machine
    if prompt.get("json"):
        payload["format"] = "json"          # Ollama JSON mode for structured outputs (summary, report); never used for the scored response
    req = urllib.request.Request(host(env) + "/api/chat", data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        try:
            msg = json.loads(body).get("error", body)
        except Exception:
            msg = body
        return None, {"kind": classify(e, msg), "provider": {"http_status": e.code, "provider_status": None, "reason": None, "message": str(msg)[:300]}}
    except Exception as e:
        kind = classify(e)
        text = (f"Ollama did not answer within {timeout:.0f} s." if kind == "network_or_timeout"
                else f"Could not connect to Ollama at {host(env)} ({type(e).__name__}). Is it running?")
        return None, {"kind": kind, "provider": {"http_status": None, "provider_status": None, "reason": None, "message": text}}
    text = ((data.get("message") or {}).get("content") or "").strip()
    if not text:
        return None, {"kind": "empty_response", "provider": {"http_status": 200, "provider_status": None, "reason": None, "message": "Ollama returned an empty answer."}}
    finish = {"stop": "STOP", "length": "MAX_TOKENS"}.get(str(data.get("done_reason") or "stop").lower(), str(data.get("done_reason")).upper())
    return {"response_text": text, "model_requested": model_name, "model_returned": data.get("model") or model_name, "finish_reason": finish,
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "attempts": 1,
            "usage": {"prompt_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")}}, None
