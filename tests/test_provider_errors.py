import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dashboard import workflow as wf
from scripts import gemini_utils as gu
from scripts import pilot_generate_gemini as gg
from scripts import pilot_generate_openai as og

ROOT = Path(__file__).resolve().parents[1]
FAKE_KEY = "AIzaSyFAKE0123456789abcdefghijklmnop"


class ClientError(Exception):
    """Same shape as google.genai.errors.ClientError (captured from a real HTTP 400 response)."""
    def __init__(self, code, status, message, reason=None):
        super().__init__(message)
        self.code, self.status, self.message = code, status, message
        self.details = {"error": {"code": code, "message": message, "status": status,
                                  "details": ([{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason}] if reason else [])}}


INVALID_KEY = ClientError(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", "API_KEY_INVALID")


def test_invalid_key_400_is_classified_specifically_not_as_generic_bad_request():
    assert gu.classify_error(INVALID_KEY) == ("invalid_api_key", False)
    other = ClientError(400, "INVALID_ARGUMENT", "Invalid value at 'generation_config.temperature'")
    assert gu.classify_error(other) == ("bad_request", False)
    loc = ClientError(400, "FAILED_PRECONDITION", "User location is not supported for the API use.")
    assert gu.classify_error(loc) == ("location_not_supported", False)


def test_provider_detail_is_captured_and_sanitized():
    d = gu.provider_error_detail(INVALID_KEY)
    assert d == {"http_status": 400, "provider_status": "INVALID_ARGUMENT", "reason": "API_KEY_INVALID", "message": "API key not valid. Please pass a valid API key."}
    leaky = ClientError(400, "INVALID_ARGUMENT", f"bad request for key={FAKE_KEY} and {FAKE_KEY} sk-abcdefghijklmnop123")
    msg = gu.provider_error_detail(leaky)["message"]
    assert FAKE_KEY not in msg and "sk-abcdefghijklmnop123" not in msg and "[redacted]" in msg
    assert len(gu.sanitize("x" * 1000)) == 300


def test_gemini_generate_one_records_the_provider_detail_and_does_not_retry():
    calls = []

    def boom(**kw):
        calls.append(kw)
        raise INVALID_KEY
    client = SimpleNamespace(models=SimpleNamespace(generate_content=boom))
    types = SimpleNamespace(GenerateContentConfig=lambda **k: k, HttpOptions=lambda **k: k)
    rec, err = gg.generate_one(client, types, "gemini-2.5-flash", {"synthetic_id": "P1_S01", "persona_id": "P1", "system": "s", "user": "u"},
                               3, 5, {"quota_hits": 0}, sleep=lambda s: None)
    assert rec is None and len(calls) == 1 and err["kind"] == "invalid_api_key" and err["stop_run"]
    assert err["provider"]["reason"] == "API_KEY_INVALID" and err["provider"]["http_status"] == 400


def test_openai_generate_one_records_the_provider_detail():
    class APIStatusError(Exception):
        def __init__(self):
            super().__init__("Incorrect API key provided")
            self.status_code, self.message, self.code = 401, "Incorrect API key provided: sk-abcd************wxyz", "invalid_api_key"
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **k: (_ for _ in ()).throw(APIStatusError()))))
    rec, err = og.generate_one(client, "m", {"synthetic_id": "a", "persona_id": "P1", "system": "s", "user": "u"}, 2, {"max_rate_limit_hits": 5}, sleep=lambda s: None)
    assert rec is None and err["provider"]["http_status"] == 401 and "Incorrect API key" in err["provider"]["message"]


@pytest.mark.parametrize("value,issue", [('"AIzaabc"', "quote"), ("'AIzaabc'", "quote"), (" AIzaabc", "spaces"), ("AIzaabc\n", "spaces"), ("AIza abc", "spaces"),
                                         ("AIzaabc", None), ("", None), (None, None)])
def test_key_format_issues_detected_without_revealing_the_key(value, issue):
    got = wf.key_format_issue(value)
    assert (got is None) if issue is None else (issue in got)
    if got:
        assert "AIzaabc" not in got


def test_quoted_key_makes_the_backend_unavailable_with_an_explanation():
    b = {x["provider"]: x for x in wf.backends_status({"GEMINI_API_KEY": '"AIzaSyFAKE"'}, lambda m: object())}["gemini"]
    assert b["available"] is False and "quote" in b["unavailable_reason"] and "AIzaSyFAKE" not in json.dumps(b)


def test_dashboard_shows_the_confirmed_cause_and_no_fake_result(tmp_path):
    w = wf.Workflow(ROOT, generator=lambda *a: (None, {"kind": "invalid_api_key", "provider": gu.provider_error_detail(INVALID_KEY)}),
                    env={"GEMINI_API_KEY": "AIzaSyFAKE"}, find_spec=lambda m: object(), log_dir=tmp_path / "r")
    r = w.simple_run({"persona_id": "P1_premium_tier_owner", "confirm_generation": True,
                      "question": "Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?"})
    assert r["ok"] is False and r["error_code"] == "invalid_api_key" and "API_KEY_INVALID" in r["error"] and "without quotes" in r["error"]
    assert "response_text" not in r and "expected_rating" not in r and "AIzaSyFAKE" not in json.dumps(r)


def test_other_provider_errors_show_status_and_sanitized_message():
    err = {"kind": "bad_request", "provider": gu.provider_error_detail(ClientError(400, "INVALID_ARGUMENT", "Invalid value at 'generation_config'"))}
    msg = wf.explain_provider_error(err, "gemini")
    assert "HTTP 400" in msg and "INVALID_ARGUMENT" in msg and "generation_config" in msg and "No response was generated" in msg
