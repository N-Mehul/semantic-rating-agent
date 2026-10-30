"""Gemini helpers shared by test_gemini_api.py and pilot_generate_gemini.py (key is only read from the environment)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# gemini-2.5-flash was withdrawn for new users (provider answered HTTP 404 NOT_FOUND and recommended gemini-3.8-flash).
# Earlier saved pilots really used gemini-2.5-flash; their recorded provenance is unchanged.
DEFAULT_MODEL = "gemini-3.8-flash"
KEY_HELP = (
    "GEMINI_API_KEY is not set in this process.\n"
    "Windows CMD (current window only):   set GEMINI_API_KEY=your-key-here\n"
    "then run the script again from the SAME CMD window.\n"
    "Optional model:                      set GEMINI_MODEL=gemini-3.8-flash\n"
    "Do not put the key in any file in the repository."
)


def normalize_model(name: str | None) -> str | None:
    """The google-genai SDK adds the "models/" resource prefix itself; strip a user-supplied one so it is never duplicated."""
    if name is None:
        return None
    name = name.strip()
    while name.startswith("models/"):
        name = name[len("models/"):]
    return name or None


def get_model(env=None) -> str:
    """Explicit GEMINI_MODEL wins; otherwise the project default."""
    env = os.environ if env is None else env
    return normalize_model(env.get("GEMINI_MODEL")) or DEFAULT_MODEL


def has_key() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


KEY_LIKE = re.compile(r"(AIza[0-9A-Za-z_\-]{10,}|sk-[0-9A-Za-z_\-]{10,}|key=[^&\s]+)")


def sanitize(text, limit: int = 300) -> str:
    """Provider messages can echo request details; strip anything that looks like a key and truncate."""
    return KEY_LIKE.sub("[redacted]", str(text or ""))[:limit]


def provider_error_detail(exc: BaseException) -> dict:
    """HTTP status, provider status, machine-readable reason (e.g. API_KEY_INVALID) and a sanitized message from an SDK error."""
    details = getattr(exc, "details", None)
    err = details.get("error", {}) if isinstance(details, dict) else {}
    reason = None
    for d in err.get("details", []) if isinstance(err.get("details"), list) else []:
        if isinstance(d, dict) and d.get("reason"):
            reason = d["reason"]
            break
    code = getattr(exc, "code", None)
    if not isinstance(code, int):
        code = getattr(exc, "status_code", None)
    return {"http_status": code if isinstance(code, int) else None, "provider_status": getattr(exc, "status", None) or err.get("status"),
            "reason": reason, "message": sanitize(getattr(exc, "message", None) or err.get("message") or exc)}


def classify_error(exc: BaseException) -> tuple[str, bool]:
    """Return (kind, retryable). Permanent errors (auth, model access, bad request) are never retried."""
    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "").upper()
    reason = provider_error_detail(exc)["reason"] if getattr(exc, "details", None) else None
    if reason == "API_KEY_INVALID":                 # Gemini reports an invalid key as HTTP 400 INVALID_ARGUMENT
        return "invalid_api_key", False
    if status == "FAILED_PRECONDITION" and "location" in str(getattr(exc, "message", "")).lower():
        return "location_not_supported", False
    if code in (401, 403) or status in ("UNAUTHENTICATED", "PERMISSION_DENIED"):
        return "authentication_or_permission", False
    if code == 404 or status == "NOT_FOUND":
        return "model_not_found_or_no_access", False
    if code == 400 or status in ("INVALID_ARGUMENT", "FAILED_PRECONDITION"):
        return "bad_request", False
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return "rate_limit_or_quota", True
    if isinstance(code, int) and code >= 500:
        return "server_error", True
    name = type(exc).__name__
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) or any(
            k in name for k in ("Timeout", "Connect", "Network", "Read", "Remote")):
        return "network_or_timeout", True
    return "unknown", False


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["synthetic_id"] for l in path.open(encoding="utf-8") if l.strip()}


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
