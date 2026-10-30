import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import gemini_utils as gu
from scripts import pilot_generate_gemini as gg
from scripts import ssr_eval_lib as lib

ROOT = Path(__file__).resolve().parents[1]


class Err(Exception):
    def __init__(self, code, status=""):
        self.code, self.status = code, status


class FakeClient:
    """Mimics client.models.generate_content; `plan` is a list of exceptions or text strings."""
    def __init__(self, plan):
        self.plan, self.calls = list(plan), 0
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, **kw):
        self.calls += 1
        item = self.plan.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, model_version="fake-1")


FakeTypes = SimpleNamespace(
    GenerateContentConfig=lambda **k: k, HttpOptions=lambda **k: k)
P = {"synthetic_id": "P1_S01", "persona_id": "P1", "system": "s", "user": "u"}


def run(plan, **kw):
    c = FakeClient(plan)
    out = gg.generate_one(c, FakeTypes, "m", P, kw.get("retries", 4), kw.get("quota", 5), {"quota_hits": 0}, sleep=lambda s: None)
    return c, out


def test_classify_permanent_vs_transient():
    assert gu.classify_error(Err(401)) == ("authentication_or_permission", False)
    assert gu.classify_error(Err(403, "PERMISSION_DENIED"))[1] is False
    assert gu.classify_error(Err(404)) == ("model_not_found_or_no_access", False)
    assert gu.classify_error(Err(429, "RESOURCE_EXHAUSTED")) == ("rate_limit_or_quota", True)
    assert gu.classify_error(Err(503)) == ("server_error", True)
    assert gu.classify_error(TimeoutError())[1] is True


def test_permanent_error_not_retried_and_stops_run():
    c, (rec, err) = run([Err(401)])
    assert rec is None and c.calls == 1 and err["stop_run"] and err["kind"] == "authentication_or_permission"


def test_transient_error_retried_then_succeeds():
    c, (rec, err) = run([Err(503), Err(429, "RESOURCE_EXHAUSTED"), "Worth it, good battery."])
    assert err is None and c.calls == 3 and rec["attempts"] == 3 and rec["source"] == "gemini_generated"


def test_retries_bounded_and_quota_stops_run():
    c, (rec, err) = run([Err(503)] * 10, retries=3)
    assert rec is None and c.calls == 3 and not err["stop_run"]
    c, (rec, err) = run([Err(429, "RESOURCE_EXHAUSTED")] * 10, quota=2)
    assert c.calls == 2 and err["stop_run"]


def test_empty_response_is_a_failure_not_a_record():
    c, (rec, err) = run(["   "] * 4)
    assert rec is None and err is not None


def test_rating_in_text_is_flagged_not_hidden():
    _, (rec, _) = run(["I would give it 4/5, great phone."])
    assert rec["flag_contains_rating"] is True


def test_checkpoint_resume_skips_done_and_no_duplicates(tmp_path):
    f = tmp_path / "r.jsonl"
    gu.append_jsonl(f, {"synthetic_id": "A", "response_text": "x"})
    gu.append_jsonl(f, {"synthetic_id": "B", "response_text": "y"})
    assert gu.done_ids(f) == {"A", "B"}
    todo = [i for i in ["A", "B", "C"] if i not in gu.done_ids(f)]
    assert todo == ["C"]
    assert gu.done_ids(tmp_path / "missing.jsonl") == set()


def test_prompts_leak_free_and_ids_stable():
    prompts = gg.build_prompts(gg.load_specs(40))
    assert [p["synthetic_id"] for p in prompts][:2] == ["P1_S01", "P1_S02"] and len(prompts) == 40
    assert not any(lib.prompt_leak_findings(p["system"] + p["user"]) for p in prompts)


def test_generator_stops_cleanly_without_key():
    env = {k: v for k, v in __import__("os").environ.items() if k != "GEMINI_API_KEY"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_generate_gemini.py")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "GEMINI_API_KEY" in r.stdout and "set GEMINI_API_KEY" in r.stdout
