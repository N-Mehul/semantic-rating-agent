import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import pilot_generate_gemini as gg
from scripts import ssr_eval_lib as lib

ROOT = Path(__file__).resolve().parents[1]
FakeTypes = SimpleNamespace(GenerateContentConfig=lambda **k: k, HttpOptions=lambda **k: k)
P = {"synthetic_id": "P1_S01", "persona_id": "P1_premium_tier_owner", "system": "s", "user": "u"}


class FakeClient:
    """plan items: (text, finish_reason_name) tuples or exceptions. Records the config it was called with."""
    def __init__(self, plan):
        self.plan, self.calls, self.configs = list(plan), 0, []
        self.models = SimpleNamespace(generate_content=self._gen)

    def _gen(self, **kw):
        self.calls += 1
        self.configs.append(kw["config"])
        item = self.plan.pop(0)
        if isinstance(item, Exception):
            raise item
        text, fr = item
        return SimpleNamespace(
            text=text, model_version="fake-2",
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=fr))],
            usage_metadata=SimpleNamespace(prompt_token_count=50, candidates_token_count=30, thoughts_token_count=900))


def run(plan, **kw):
    c = FakeClient(plan)
    out = gg.generate_one(c, FakeTypes, "m", P, 3, 5, {"quota_hits": 0}, sleep=lambda s: None, **kw)
    return c, out


def test_default_token_limit_raised_and_passed_to_api():
    assert gg.SETTINGS["max_output_tokens"] == 2048 > 512
    c, (rec, err) = run([("Worth it, the camera is great and I would buy it again.", "STOP")])
    assert err is None and c.configs[0]["max_output_tokens"] == 2048
    assert rec["max_output_tokens"] == 2048


def test_finish_reason_usage_and_quality_recorded_for_complete_answer():
    _, (rec, _) = run([("Worth it, the camera is great and I would buy it again.", "STOP")])
    assert rec["finish_reason"] == "STOP" and rec["quality"]["is_complete"] and rec["quality"]["safe_for_ssr"]
    assert rec["usage"] == {"prompt_tokens": 50, "output_tokens": 30, "thought_tokens": 900}


def test_truncated_answer_is_kept_and_flagged_not_discarded():
    _, (rec, err) = run([("Yeah, it is worth it. The battery is", "MAX_TOKENS")])
    assert err is None and rec is not None                      # still saved
    q = rec["quality"]
    assert q["finish_reason_not_stop"] and q["no_terminal_punctuation"] and not q["is_complete"] and not q["safe_for_ssr"]


def test_missing_terminal_punctuation_flagged_even_if_stop():
    q = lib.response_quality("Yes it is worth it and the", "STOP")
    assert q["no_terminal_punctuation"] and not q["is_complete"]
    assert lib.response_quality("Yes, worth it.", "STOP")["is_complete"]
    assert lib.response_quality("Yes, worth it.", None)["is_complete"]   # unknown reason: heuristic only


def test_empty_response_retried_then_logged_as_failure_not_silent():
    c, (rec, err) = run([("   ", "MAX_TOKENS")] * 3)
    assert rec is None and err["kind"] == "empty_response" and c.calls == 3 and not err["stop_run"]


def test_rating_text_flagged_unsafe_for_ssr():
    q = lib.response_quality("Honestly I would give it 4/5, great phone.", "STOP")
    assert q["contains_rating"] and q["is_complete"] and not q["safe_for_ssr"]


def test_quality_summary_counts():
    recs = [run([("Good phone, worth it.", "STOP")])[1][0], run([("Good phone, but the", "MAX_TOKENS")])[1][0]]
    s = gg.quality_summary(recs)
    assert (s["n"], s["is_complete"], s["finish_reason_not_stop"], s["safe_for_ssr"]) == (2, 1, 1, 1)


def test_original_pilot_is_flagged_by_the_validator_read_only():
    rows = [json.loads(l) for l in open(ROOT / "data/pilot_gemini/gemini_generated_responses.jsonl", encoding="utf-8")]
    assert len(rows) == 20
    flagged = [r for r in rows if not lib.response_quality(r["response_text"], None)["is_complete"]]
    assert len(flagged) == 19                                    # matches the manual finding of 19/20 cut-off answers


def test_generator_refuses_original_folder_and_defaults_to_versioned_folder():
    assert gg.DEFAULT_OUT.name == "pilot_gemini_v2" and gg.DEFAULT_OUT != gg.ORIGINAL_OUT
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_generate_gemini.py"), "--out-dir",
                        str(ROOT / "data/pilot_gemini")], capture_output=True, text=True)
    assert r.returncode == 2 and "Refusing" in r.stdout


def test_prompts_still_text_only_and_leak_free():
    prompts = gg.build_prompts(gg.load_specs(40))
    assert all(not lib.prompt_leak_findings(p["system"] + p["user"]) for p in prompts)
    assert all("no ratings" in p["system"] for p in prompts)
