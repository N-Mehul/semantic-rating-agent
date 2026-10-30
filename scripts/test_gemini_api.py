"""One small real Gemini request to verify key, model access and connectivity. Never prints the key.

Windows CMD:  set GEMINI_API_KEY=...   (optional: set GEMINI_MODEL=gemini-3.8-flash)
Run:          python scripts/test_gemini_api.py          (exit 0 = request succeeded)
Not collected by pytest (lives in scripts/, not tests/).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gemini_utils import KEY_HELP, classify_error, get_model, has_key  # noqa: E402


def main() -> int:
    if not has_key():
        print(KEY_HELP)
        return 2
    from google import genai
    from google.genai import types
    model = get_model()
    print(f"Testing one request with model '{model}' ...", flush=True)
    try:
        client = genai.Client(http_options=types.HttpOptions(timeout=30_000))
        r = client.models.generate_content(
            model=model, contents="Reply with one short sentence about smartphones. Use words only.",
            config=types.GenerateContentConfig(temperature=1.0, max_output_tokens=256))
        text = (r.text or "").strip()
        print(f"OK model_version={getattr(r, 'model_version', model)} chars={len(text)}")
        print("sample:", text[:120].replace("\n", " "))
        return 0 if text else 3
    except Exception as exc:  # report class/code only, never request headers
        kind, retryable = classify_error(exc)
        print(f"FAILED kind={kind} retryable={retryable} type={type(exc).__name__} code={getattr(exc, 'code', None)}")
        print("Transient: wait and rerun." if retryable else "Permanent error: fix key/model/request before rerunning.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
