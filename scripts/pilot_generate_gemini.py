"""Generate synthetic Q26-style answers with Google Gemini (separate from the manual and OpenAI pilots).

Windows CMD:  set GEMINI_API_KEY=...   then   python scripts/pilot_generate_gemini.py [--limit 40] [--overwrite]
Reuses the leak-checked prompts of pilot_generate_api.py (non-outcome persona attributes only; words only, no ratings).

Default output directory is the NEW versioned folder data/pilot_gemini_v2/ so the original pilot in
data/pilot_gemini/ (20 responses, 512-token limit) stays reproducible; writing into the original folder is refused.
Files: gemini_generated_responses.jsonl, failures.jsonl, prompts.jsonl, generation_config.json, run_summary.json.
Each record stores the API finish reason, token usage and quality flags (ssr_eval_lib.response_quality). Responses
are checkpointed one by one; reruns skip finished IDs. Flagged responses are kept and marked, never discarded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gemini_utils import KEY_HELP, append_jsonl, classify_error, done_ids, get_model, has_key, provider_error_detail  # noqa: E402
from pilot_generate_api import QUESTION, SYSTEM, USER_TEMPLATE, build_prompts, load_specs  # noqa: E402
from ssr_eval_lib import response_quality  # noqa: E402
from dp_constructs import load_prompts_file  # noqa: E402

ORIGINAL_OUT = ROOT / "data/pilot_gemini"          # frozen original pilot (max_output_tokens=512)
DEFAULT_OUT = ROOT / "data/pilot_gemini_v2"
# 512 was too small: 19 of 20 original answers were cut off (gemini-2.5 models spend output tokens on hidden
# reasoning before the answer). 2048 leaves room for that reasoning plus a 1-4 sentence answer.
SETTINGS = {"temperature": 1.0, "max_output_tokens": 2048, "timeout_s": 60}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def extract_finish_reason(r):
    cands = getattr(r, "candidates", None)
    if not cands:
        return None
    fr = getattr(cands[0], "finish_reason", None)
    return None if fr is None else getattr(fr, "name", str(fr))


def extract_usage(r):
    u = getattr(r, "usage_metadata", None)
    return {"prompt_tokens": getattr(u, "prompt_token_count", None),
            "output_tokens": getattr(u, "candidates_token_count", None),
            "thought_tokens": getattr(u, "thoughts_token_count", None)} if u else None


def generate_one(client, types, model, p, max_retries, max_quota_hits, state, sleep=time.sleep, settings=None):
    """Return (record, failure); exactly one is None. Permanent errors are not retried; quota hits are bounded.
    Truncated/incomplete answers are returned as records with quality flags, not dropped."""
    settings = settings or SETTINGS
    for attempt in range(1, max_retries + 1):
        try:
            r = client.models.generate_content(
                model=model, contents=p["user"],
                config=types.GenerateContentConfig(
                    system_instruction=p["system"], temperature=settings["temperature"],
                    max_output_tokens=settings["max_output_tokens"],
                    http_options=types.HttpOptions(timeout=settings["timeout_s"] * 1000)))
            text = (r.text or "").strip()
            finish = extract_finish_reason(r)
            if not text:
                raise ValueError("empty response text (finish_reason=%s)" % finish)
            q = response_quality(text, finish)
            return ({"synthetic_id": p["synthetic_id"], "persona_id": p["persona_id"], "response_text": text,
                     "source": "gemini_generated", "model_requested": model,
                     "model_returned": getattr(r, "model_version", None), "attempts": attempt,
                     "finish_reason": finish, "usage": extract_usage(r),
                     "max_output_tokens": settings["max_output_tokens"], "quality": q,
                     "flag_contains_rating": q["contains_rating"], "task_id": p.get("task_id"), "construct": p.get("construct"),
                     "question_id": p.get("question_id"), "timestamp_utc": now()}, None)
        except Exception as exc:
            kind, retryable = classify_error(exc)
            if isinstance(exc, ValueError) and "empty response" in str(exc):
                kind, retryable = "empty_response", True
            quota_exhausted = False
            if kind == "rate_limit_or_quota":
                state["quota_hits"] += 1
                quota_exhausted = state["quota_hits"] >= max_quota_hits
            permanent = not retryable and kind != "unknown"
            if quota_exhausted or not retryable or attempt == max_retries:
                return None, {"synthetic_id": p["synthetic_id"], "kind": kind, "type": type(exc).__name__,
                              "code": getattr(exc, "code", None), "attempts": attempt, "provider": provider_error_detail(exc),
                              "stop_run": bool(quota_exhausted or permanent), "timestamp_utc": now()}
            sleep(min(60, 2 ** attempt))
    return None, {"synthetic_id": p["synthetic_id"], "kind": "no_attempts", "stop_run": True, "timestamp_utc": now()}


def quality_summary(records) -> dict:
    n = len(records)
    keys = ["empty", "finish_reason_not_stop", "no_terminal_punctuation", "too_short_lt3_words", "contains_rating",
            "is_complete", "safe_for_ssr"]
    return {"n": n, **{k: sum(bool(r["quality"][k]) for r in records) for k in keys}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--max-quota-hits", type=int, default=5, help="stop the whole run after this many 429 responses")
    ap.add_argument("--max-output-tokens", type=int, default=SETTINGS["max_output_tokens"])
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="must not be the original data/pilot_gemini folder")
    ap.add_argument("--overwrite", action="store_true", help="discard existing responses IN --out-dir and start over")
    ap.add_argument("--prompts-file", default=None, help="construct-specific prompts JSONL (see scripts/dp_constructs.py); requires an explicit new --out-dir")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir).resolve()
    if out == ORIGINAL_OUT.resolve():
        print("Refusing to write into data/pilot_gemini (the frozen original pilot). Use a new --out-dir.")
        return 2
    settings = {**SETTINGS, "max_output_tokens": a.max_output_tokens}
    out.mkdir(parents=True, exist_ok=True)
    if a.prompts_file:
        if out == DEFAULT_OUT.resolve():
            print("--prompts-file needs its own --out-dir (e.g. data/pilot_gemini_FT_PE01) so different questions are never mixed in one folder.")
            return 2
        prompts = load_prompts_file(Path(a.prompts_file), a.limit)  # raises on missing keys, duplicate ids or leaked content
    else:
        prompts = build_prompts(load_specs(a.limit))  # raises if any prompt fails the leak check
    (out / "prompts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8")
    if a.dry_run:
        print(f"dry run ok: {len(prompts)} prompts passed leak check; out_dir={out.name}; max_output_tokens={settings['max_output_tokens']}")
        return 0
    if not has_key():
        print(KEY_HELP)
        return 2
    from google import genai
    from google.genai import types
    model = get_model()
    resp, fail = out / "gemini_generated_responses.jsonl", out / "failures.jsonl"
    if a.overwrite:
        for f in (resp, fail):
            f.unlink(missing_ok=True)
    finished = done_ids(resp)
    (out / "generation_config.json").write_text(json.dumps({
        "provider": "google-genai", "model_requested": model, "settings": settings, "system_prompt": SYSTEM,
        "user_template": USER_TEMPLATE, "question": QUESTION if not a.prompts_file else "see prompts.jsonl (per-prompt task_id / question_id)", "label": "gemini_generated",
        "supersedes": "none; data/pilot_gemini (512-token pilot) is kept unchanged for reproducibility",
        "prompts_file": a.prompts_file,
        "note": "No Q21, Q26 or evaluation labels are in any prompt (leak-checked). Text only; no ratings requested. API key never stored."},
        indent=2), encoding="utf-8")
    client = genai.Client()
    state, ok, failed, stopped = {"quota_hits": 0}, 0, 0, False
    todo = [p for p in prompts if p["synthetic_id"] not in finished]
    print(f"model={model} out={out.name} max_output_tokens={settings['max_output_tokens']} total={len(prompts)} "
          f"already_done={len(finished)} to_do={len(todo)}", flush=True)
    for i, p in enumerate(todo, 1):
        rec, err = generate_one(client, types, model, p, a.max_retries, a.max_quota_hits, state, settings=settings)
        if rec:
            append_jsonl(resp, rec)
            ok += 1
            q = rec["quality"]
            print(f"[{i}/{len(todo)}] {p['synthetic_id']} ok finish={rec['finish_reason']} complete={q['is_complete']}", flush=True)
        else:
            append_jsonl(fail, err)
            failed += 1
            print(f"[{i}/{len(todo)}] {p['synthetic_id']} FAILED kind={err['kind']}", flush=True)
            if err.get("stop_run"):
                stopped = True
                print("Stopping: permanent error or quota exhausted; completed responses are kept. Rerun to resume.", flush=True)
                break
    on_disk = [json.loads(l) for l in resp.open(encoding="utf-8")] if resp.exists() else []
    summary = {"model": model, "timestamp_utc": now(), "requested": len(prompts), "newly_succeeded": ok,
               "newly_failed": failed, "total_successful_on_disk": len(done_ids(resp)), "stopped_early": stopped,
               "quality": quality_summary(on_disk) if on_disk else None}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0 if len(done_ids(resp)) == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
