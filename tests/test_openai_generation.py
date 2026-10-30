import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import gemini_utils as gu
from scripts import pilot_generate_openai as og
from scripts import ssr_eval_lib as lib

ROOT = Path(__file__).resolve().parents[1]
P = {"synthetic_id": "P1_S01", "persona_id": "P1_premium_tier_owner", "system": "s", "user": "u"}


class Err(Exception):
    def __init__(self, status=None, code=None, name=None):
        self.status_code, self.code = status, code
        if name:
            self.__class__ = type(name, (Err,), {})


def completion(text, finish="stop", refusal=None):
    return SimpleNamespace(
        id="chatcmpl-1", model="fake-model-2025", system_fingerprint="fp",
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=text, refusal=refusal))],
        usage=SimpleNamespace(prompt_tokens=60, completion_tokens=40,
                              completion_tokens_details=SimpleNamespace(reasoning_tokens=0)))


class FakeClient:
    def __init__(self, plan):
        self.plan, self.calls, self.kwargs = list(plan), 0, []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(retrieve=self._retrieve)

    def _create(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        item = self.plan.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def _retrieve(self, model):
        if model == "bad-model":
            raise Err(404)
        return SimpleNamespace(id=model)


def run(plan, retries=3, **kw):
    c = FakeClient(plan)
    state = {"max_rate_limit_hits": kw.pop("max_hits", 5)}
    return c, og.generate_one(c, "fake-model", P, retries, state, sleep=lambda s: None, **kw)


# ---- parsing
def test_parse_completion_extracts_all_fields():
    d = og.parse_completion(completion("  Worth it, honestly.  "))
    assert d["text"] == "Worth it, honestly." and d["finish_reason"] == "stop" and d["model_returned"] == "fake-model-2025"
    assert d["response_id"] == "chatcmpl-1" and d["usage"] == {"prompt_tokens": 60, "completion_tokens": 40, "reasoning_tokens": 0}


def test_parse_completion_tolerates_missing_pieces():
    assert og.parse_completion(SimpleNamespace(choices=[]))["text"] == ""
    assert og.parse_completion(completion(None, refusal="cannot help"))["refusal"] == "cannot help"
    assert og.parse_completion(SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=None))]))["text"] == ""


# ---- quality
def test_complete_answer_record_has_quality_and_finish_reason():
    c, (rec, err) = run([completion("Worth it, the camera is great and I would buy it again.")])
    assert err is None and rec["source"] == "openai_generated"
    assert rec["finish_reason"] == "stop" and rec["quality"]["is_complete"] and rec["quality"]["safe_for_ssr"]
    assert c.kwargs[0]["max_completion_tokens"] == 1024 and c.kwargs[0]["temperature"] == 1.0
    assert c.kwargs[0]["messages"][0]["role"] == "system" and "ratings" not in json.dumps(rec["quality"])


def test_truncated_answer_kept_and_flagged():
    _, (rec, err) = run([completion("It is worth it and the battery is", finish="length")])
    assert err is None and rec is not None
    q = rec["quality"]
    assert q["finish_reason_not_stop"] and q["no_terminal_punctuation"] and not q["safe_for_ssr"]


def test_numeric_rating_in_text_flagged_unsafe():
    _, (rec, _) = run([completion("Great phone, I would say 4/5 overall.")])
    assert rec["flag_contains_rating"] and not rec["quality"]["safe_for_ssr"]


def test_empty_and_refusal_retried_then_logged_not_silent():
    c, (rec, err) = run([completion("  ")] * 3)
    assert rec is None and err["kind"] == "empty_response" and c.calls == 3 and not err["stop_run"]
    _, (rec, err) = run([completion(None, finish="stop", refusal="no")] * 3)
    assert rec is None and err["kind"] == "refusal"


def test_no_temperature_option_omits_parameter():
    c, _ = run([completion("Fine for the price.")], settings={**og.SETTINGS, "temperature": None})
    assert "temperature" not in c.kwargs[0]


# ---- errors
def test_error_classification():
    assert og.classify_openai_error(Err(401)) == ("authentication", False)
    assert og.classify_openai_error(Err(403)) == ("permission_or_model_access", False)
    assert og.classify_openai_error(Err(404)) == ("model_not_found", False)
    assert og.classify_openai_error(Err(400)) == ("bad_request", False)
    assert og.classify_openai_error(Err(429)) == ("rate_limit", True)
    assert og.classify_openai_error(Err(429, code="insufficient_quota")) == ("insufficient_quota", False)
    assert og.classify_openai_error(Err(503)) == ("server_error", True)
    assert og.classify_openai_error(TimeoutError()) == ("network_or_timeout", True)


def test_permanent_error_not_retried_and_stops_run():
    c, (rec, err) = run([Err(401)])
    assert rec is None and c.calls == 1 and err["stop_run"]
    c, (rec, err) = run([Err(429, code="insufficient_quota")])
    assert c.calls == 1 and err["kind"] == "insufficient_quota" and err["stop_run"]


def test_rate_limit_retried_then_succeeds_and_bounded_otherwise():
    c, (rec, err) = run([Err(429), Err(503), completion("Worth it, good value.")])
    assert err is None and rec["attempts"] == 3
    c, (rec, err) = run([Err(429)] * 10, retries=3, max_hits=3)
    assert rec is None and c.calls == 3 and err["stop_run"]


# ---- model check
def test_model_check_uses_metadata_lookup_only():
    c = FakeClient([])
    assert og.verify_model(c, "good-model")[0] is True and c.calls == 0     # no generation request
    ok, msg = og.verify_model(c, "bad-model")
    assert ok is False and "model_not_found" in msg


# ---- resume
def test_resume_reuses_finished_ids_and_only_missing_are_attempted(tmp_path):
    f = tmp_path / "openai_generated_responses.jsonl"
    prompts = [{**P, "synthetic_id": f"P1_S0{i}"} for i in range(1, 5)]
    for pid in ("P1_S01", "P1_S02"):
        gu.append_jsonl(f, {"synthetic_id": pid, "response_text": "kept"})
    todo = [p for p in prompts if p["synthetic_id"] not in gu.done_ids(f)]
    assert [p["synthetic_id"] for p in todo] == ["P1_S03", "P1_S04"]
    c = FakeClient([completion("New answer one."), completion("New answer two.")])
    for p in todo:
        rec, err = og.generate_one(c, "m", p, 2, {"max_rate_limit_hits": 5}, sleep=lambda s: None)
        gu.append_jsonl(f, rec)
    rows = [json.loads(l) for l in f.open(encoding="utf-8")]
    assert [r["synthetic_id"] for r in rows] == ["P1_S01", "P1_S02", "P1_S03", "P1_S04"] and c.calls == 2
    assert rows[0]["response_text"] == "kept"                                  # earlier responses untouched


# ---- isolation and prompts
def test_protected_folders_refused_and_default_is_new_folder():
    assert og.DEFAULT_OUT.name == "pilot_openai_v1"
    for d in ("data/pilot_gemini", "data/pilot_gemini_v2", "data/pilot_api", "data/pilot"):
        assert og.check_out_dir(ROOT / d) is not None
    assert og.check_out_dir(ROOT / "data/pilot_openai_v1") is None
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_generate_openai.py"), "--out-dir", str(ROOT / "data/pilot_gemini_v2")],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "Refusing" in r.stdout


def test_prompts_unchanged_text_only_and_leak_free():
    from scripts import pilot_generate_gemini as gg
    a, b = og.build_prompts(og.load_specs(40)), gg.build_prompts(gg.load_specs(40))
    assert a == b                                                              # same persona prompts as the Gemini pilot
    assert all(not lib.prompt_leak_findings(p["system"] + p["user"]) for p in a)


def test_stops_cleanly_without_key_and_without_model():
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_generate_openai.py"), "--model", "x"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "set OPENAI_API_KEY" in r.stdout
    env["OPENAI_API_KEY"] = "dummy-not-used"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_generate_openai.py")], capture_output=True, text=True,
                       env={k: v for k, v in env.items() if k != "OPENAI_GEN_MODEL"})
    assert r.returncode == 2 and "No model chosen" in r.stdout                 # stops before any network call
