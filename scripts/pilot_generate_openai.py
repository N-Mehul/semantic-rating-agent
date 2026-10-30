"""Generate synthetic Q26-style answers with an OpenAI chat model (separate from the Gemini and earlier OpenAI pilots).

Windows CMD:   set OPENAI_API_KEY=your-key        (current window only; never write it into a file)
               python scripts/pilot_generate_openai.py --model <model-id> [--limit 40]
Optional:      set OPENAI_GEN_MODEL=<model-id>      (used when --model is not given)

Safety properties
  * The model id is never assumed to be valid: before any generation request the script calls models.retrieve(model)
    (a metadata lookup, no generation) and stops if the model is unknown or not accessible to the key.
  * Prompts are the same persona prompts as the other pilots (build_prompts: leak-checked, words only, no ratings).
  * Output goes to data/pilot_openai_v1/ ONLY; the Gemini folders and data/pilot_api (earlier, rate-limited attempt)
    are protected and refused. Gemini and OpenAI results are different experimental conditions and are kept apart.
  * Each record stores finish reason, model returned, response id, token usage, settings and quality flags
    (ssr_eval_lib.response_quality). Flagged responses are kept and marked, never dropped.
  * Resume: finished synthetic_ids in the responses file are reused; only missing prompts are attempted.
  * Permanent errors (auth, permission, unknown model, bad request, insufficient quota) stop the run; rate limits and
    server/network errors are retried with bounded backoff.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gemini_utils import append_jsonl, done_ids  # noqa: E402  (provider-neutral JSONL checkpoint helpers)
from pilot_generate_api import QUESTION, SYSTEM, USER_TEMPLATE, build_prompts, load_specs  # noqa: E402
from ssr_eval_lib import response_quality  # noqa: E402
from dp_constructs import load_prompts_file  # noqa: E402

DEFAULT_OUT = ROOT / "data/pilot_openai_v1"
PROTECTED = [ROOT / "data/pilot_gemini", ROOT / "data/pilot_gemini_v2", ROOT / "data/pilot_api", ROOT / "data/pilot"]
# max_completion_tokens (not max_tokens) so reasoning-style models also work; generous because hidden reasoning, if
# the chosen model uses it, counts against this limit. temperature is sent only if set (some models reject it).
SETTINGS = {"temperature": 1.0, "max_completion_tokens": 1024, "timeout_s": 60}

KEY_HELP = ("OPENAI_API_KEY is not set in this process.\n"
            "Windows CMD (current window only):   set OPENAI_API_KEY=your-key-here\n"
            "then run the script again from the SAME CMD window. Do not put the key in any repository file.")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def classify_openai_error(exc: BaseException) -> tuple[str, bool]:
    """Return (kind, retryable). Works from status_code / error code attributes; imports nothing from openai."""
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").lower()
    body = getattr(exc, "body", None)
    if not code and isinstance(body, dict):
        code = str((body.get("error") or body).get("code") or "").lower()
    name = type(exc).__name__
    if code == "insufficient_quota" or "credit" in code:
        return "insufficient_quota", False
    if status == 401 or name == "AuthenticationError":
        return "authentication", False
    if status == 403 or name == "PermissionDeniedError":
        return "permission_or_model_access", False
    if status == 404 or name == "NotFoundError":
        return "model_not_found", False
    if status in (400, 422) or name in ("BadRequestError", "UnprocessableEntityError"):
        return "bad_request", False
    if status == 429 or name == "RateLimitError":
        return "rate_limit", True
    if (isinstance(status, int) and status >= 500) or name == "InternalServerError":
        return "server_error", True
    if isinstance(exc, (TimeoutError, ConnectionError)) or any(k in name for k in ("Timeout", "Connection")):
        return "network_or_timeout", True
    return "unknown", False


def parse_completion(resp) -> dict:
    """Pull text, finish reason, ids and usage out of a chat-completion response object (tolerates missing fields)."""
    choices = getattr(resp, "choices", None) or []
    ch = choices[0] if choices else None
    msg = getattr(ch, "message", None)
    text = getattr(msg, "content", None)
    u = getattr(resp, "usage", None)
    det = getattr(u, "completion_tokens_details", None) if u else None
    return {"text": (text or "").strip() if isinstance(text, str) else "",
            "refusal": getattr(msg, "refusal", None),
            "finish_reason": getattr(ch, "finish_reason", None),
            "model_returned": getattr(resp, "model", None), "response_id": getattr(resp, "id", None),
            "system_fingerprint": getattr(resp, "system_fingerprint", None),
            "usage": ({"prompt_tokens": getattr(u, "prompt_tokens", None),
                       "completion_tokens": getattr(u, "completion_tokens", None),
                       "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None} if u else None)}


def generate_one(client, model, p, max_retries, state, sleep=time.sleep, settings=None):
    """Return (record, failure); exactly one is None."""
    settings = settings or SETTINGS
    for attempt in range(1, max_retries + 1):
        try:
            kw = {"model": model, "max_completion_tokens": settings["max_completion_tokens"],
                  "messages": [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]}
            if settings.get("temperature") is not None:
                kw["temperature"] = settings["temperature"]
            parsed = parse_completion(client.chat.completions.create(**kw))
            if not parsed["text"]:
                why = "refusal" if parsed["refusal"] else "empty_response"
                raise ValueError(f"{why}: no text (finish_reason={parsed['finish_reason']})")
            q = response_quality(parsed["text"], parsed["finish_reason"])
            return ({"synthetic_id": p["synthetic_id"], "persona_id": p["persona_id"], "response_text": parsed["text"],
                     "source": "openai_generated", "model_requested": model, "model_returned": parsed["model_returned"],
                     "response_id": parsed["response_id"], "system_fingerprint": parsed["system_fingerprint"],
                     "finish_reason": parsed["finish_reason"], "usage": parsed["usage"], "attempts": attempt,
                     "settings": settings, "quality": q, "flag_contains_rating": q["contains_rating"], "task_id": p.get("task_id"),
                     "construct": p.get("construct"), "question_id": p.get("question_id"),
                     "timestamp_utc": now()}, None)
        except Exception as exc:
            kind, retryable = classify_openai_error(exc)
            if isinstance(exc, ValueError) and ("empty_response" in str(exc) or "refusal" in str(exc)):
                kind, retryable = ("refusal" if "refusal" in str(exc) else "empty_response"), True
            if kind == "rate_limit":
                state["rate_limit_hits"] = state.get("rate_limit_hits", 0) + 1
            if not retryable or attempt == max_retries:
                permanent = not retryable and kind != "unknown"
                from gemini_utils import provider_error_detail
                return None, {"synthetic_id": p["synthetic_id"], "kind": kind, "type": type(exc).__name__, "provider": provider_error_detail(exc),
                              "status": getattr(exc, "status_code", None), "attempts": attempt,
                              "stop_run": bool(permanent or (kind == "rate_limit" and state["rate_limit_hits"] >= state["max_rate_limit_hits"])),
                              "timestamp_utc": now()}
            sleep(min(60, 2 ** attempt))
    return None, {"synthetic_id": p["synthetic_id"], "kind": "no_attempts", "stop_run": True, "timestamp_utc": now()}


def quality_summary(records) -> dict:
    keys = ["empty", "finish_reason_not_stop", "no_terminal_punctuation", "too_short_lt3_words", "contains_rating",
            "is_complete", "safe_for_ssr"]
    return {"n": len(records), **{k: sum(bool(r["quality"][k]) for r in records) for k in keys}}


def check_out_dir(out: Path) -> str | None:
    """Return an error message if the directory is protected (other pilots), else None."""
    out = Path(out).resolve()
    for p in PROTECTED:
        if out == p.resolve() or p.resolve() in out.parents:
            return f"Refusing to write into protected folder {p.relative_to(ROOT).as_posix()}; use data/pilot_openai_v1 or a new folder."
    return None


def verify_model(client, model):
    """Metadata lookup only (no generation). Returns (ok, message)."""
    try:
        client.models.retrieve(model)
        return True, "model available to this key"
    except Exception as exc:
        kind, _ = classify_openai_error(exc)
        return False, f"model check failed: kind={kind} type={type(exc).__name__} status={getattr(exc, 'status_code', None)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_GEN_MODEL"), help="OpenAI model id (required unless OPENAI_GEN_MODEL is set)")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--max-rate-limit-hits", type=int, default=5, help="stop the run after this many 429 responses")
    ap.add_argument("--max-completion-tokens", type=int, default=SETTINGS["max_completion_tokens"])
    ap.add_argument("--no-temperature", action="store_true", help="omit temperature (for models that reject it)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true", help="discard existing responses IN --out-dir and start over")
    ap.add_argument("--prompts-file", default=None, help="construct-specific prompts JSONL (see scripts/dp_constructs.py); requires an explicit new --out-dir")
    ap.add_argument("--dry-run", action="store_true", help="build and leak-check prompts only; no API call")
    a = ap.parse_args()

    out = Path(a.out_dir).resolve()
    err = check_out_dir(out)
    if err:
        print(err)
        return 2
    settings = {**SETTINGS, "max_completion_tokens": a.max_completion_tokens,
                "temperature": None if a.no_temperature else SETTINGS["temperature"]}
    out.mkdir(parents=True, exist_ok=True)
    if a.prompts_file:
        if out == DEFAULT_OUT.resolve():
            print("--prompts-file needs its own --out-dir (e.g. data/pilot_openai_FT_PE01) so different questions are never mixed in one folder.")
            return 2
        prompts = load_prompts_file(Path(a.prompts_file), a.limit)  # raises on missing keys, duplicate ids or leaked content
    else:
        prompts = build_prompts(load_specs(a.limit))  # raises if any prompt fails the leak check
    (out / "prompts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8")
    if a.dry_run:
        print(f"dry run ok: {len(prompts)} prompts passed leak check; out_dir={out.name}; model={a.model or '(not set)'}")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        print(KEY_HELP)
        return 2
    if not a.model:
        print("No model chosen. Pass --model <model-id> (or set OPENAI_GEN_MODEL). The script checks it with the API before generating.")
        return 2
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=settings["timeout_s"])  # we control retries; key is read from the environment
    ok, msg = verify_model(client, a.model)
    print(f"model={a.model}: {msg}", flush=True)
    if not ok:
        return 3

    resp, fail = out / "openai_generated_responses.jsonl", out / "failures.jsonl"
    if a.overwrite:
        for f in (resp, fail):
            f.unlink(missing_ok=True)
    finished = done_ids(resp)
    (out / "generation_config.json").write_text(json.dumps({
        "provider": "openai", "model_requested": a.model, "settings": settings, "system_prompt": SYSTEM,
        "user_template": USER_TEMPLATE, "question": QUESTION if not a.prompts_file else "see prompts.jsonl (per-prompt task_id / question_id)", "label": "openai_generated", "prompts_file": a.prompts_file,
        "note": "No Q21, Q26 or evaluation labels are in any prompt (leak-checked). Text only; no ratings requested. API key never stored."},
        indent=2), encoding="utf-8")
    state = {"max_rate_limit_hits": a.max_rate_limit_hits}
    todo = [p for p in prompts if p["synthetic_id"] not in finished]
    ok_n, failed, stopped = 0, 0, False
    print(f"out={out.name} total={len(prompts)} already_done={len(finished)} to_do={len(todo)}", flush=True)
    for i, p in enumerate(todo, 1):
        rec, e = generate_one(client, a.model, p, a.max_retries, state, settings=settings)
        if rec:
            append_jsonl(resp, rec)
            ok_n += 1
            print(f"[{i}/{len(todo)}] {p['synthetic_id']} ok finish={rec['finish_reason']} complete={rec['quality']['is_complete']}", flush=True)
        else:
            append_jsonl(fail, e)
            failed += 1
            print(f"[{i}/{len(todo)}] {p['synthetic_id']} FAILED kind={e['kind']}", flush=True)
            if e.get("stop_run"):
                stopped = True
                print("Stopping: permanent error or rate limit persisted; finished responses are kept. Rerun to resume.", flush=True)
                break
    on_disk = [json.loads(l) for l in resp.open(encoding="utf-8") if l.strip()] if resp.exists() else []
    summary = {"model": a.model, "timestamp_utc": now(), "requested": len(prompts), "newly_succeeded": ok_n,
               "newly_failed": failed, "total_successful_on_disk": len(done_ids(resp)), "stopped_early": stopped,
               "quality": quality_summary(on_disk) if on_disk else None}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if len(done_ids(resp)) == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
