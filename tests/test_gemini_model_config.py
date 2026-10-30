from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dashboard import workflow as wf
from scripts import gemini_utils as gu
from scripts import pilot_generate_gemini as gg

ROOT = Path(__file__).resolve().parents[1]


def test_default_model_is_the_provider_recommended_replacement():
    assert gu.DEFAULT_MODEL == "gemini-3.8-flash"
    assert gu.get_model({}) == "gemini-3.8-flash"
    assert "gemini-3.8-flash" in gu.KEY_HELP and "gemini-2.5-flash" not in gu.KEY_HELP


@pytest.mark.parametrize("value,expected", [("gemini-custom-x", "gemini-custom-x"), ("models/gemini-custom-x", "gemini-custom-x"),
                                            ("models/models/gemini-custom-x", "gemini-custom-x"), ("  gemini-custom-x  ", "gemini-custom-x"),
                                            ("", "gemini-3.8-flash"), ("   ", "gemini-3.8-flash"), ("models/", "gemini-3.8-flash")])
def test_explicit_gemini_model_wins_and_is_normalised(value, expected):
    assert gu.get_model({"GEMINI_MODEL": value}) == expected


def test_dashboard_and_batches_use_the_same_default_and_override():
    b = {x["provider"]: x for x in wf.backends_status({"GEMINI_API_KEY": "x"}, lambda m: object())}
    assert b["gemini"]["model"] == gu.DEFAULT_MODEL
    b = {x["provider"]: x for x in wf.backends_status({"GEMINI_API_KEY": "x", "GEMINI_MODEL": "models/gemini-custom-x"}, lambda m: object())}
    assert b["gemini"]["model"] == "gemini-custom-x"                    # override kept, prefix not duplicated
    src = (ROOT / "dashboard/workflow.py").read_text(encoding="utf-8") + (ROOT / "dashboard/survey_batches.py").read_text(encoding="utf-8")
    assert "gemini-2.5-flash" not in src and "gemini-3.8-flash" not in src   # no second hardcoded default outside gemini_utils


def test_generator_sends_the_bare_model_name_to_the_sdk():
    seen = {}

    def gen(**kw):
        seen.update(kw)
        return SimpleNamespace(text="It is useful for my studies.", model_version="gemini-3.8-flash",
                               candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))], usage_metadata=None)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=gen))
    types = SimpleNamespace(GenerateContentConfig=lambda **k: k, HttpOptions=lambda **k: k)
    rec, err = gg.generate_one(client, types, gu.get_model({}), {"synthetic_id": "a", "persona_id": "P1", "system": "s", "user": "u"}, 1, 1, {"quota_hits": 0})
    assert err is None and seen["model"] == "gemini-3.8-flash" and not seen["model"].startswith("models/")
    assert {"system_instruction", "temperature", "max_output_tokens"} <= set(seen["config"])


def test_sdk_builds_exactly_one_models_prefix_offline():
    genai = pytest.importorskip("google.genai")
    from google.genai import _transformers as t
    c = genai.Client(api_key="offline-inspection-only")                  # constructing a client sends no request
    for name in (gu.get_model({}), "models/" + gu.get_model({}), gu.get_model({"GEMINI_MODEL": "models/models/gemini-3.8-flash"})):
        assert t.t_model(c._api_client, name) == "models/gemini-3.8-flash"


def test_model_not_found_404_is_reported_with_the_provider_message():
    class ClientError(Exception):
        def __init__(self):
            msg = "models/gemini-2.5-flash is no longer available to new users. Please use models/gemini-3.8-flash."
            super().__init__(msg)
            self.code, self.status, self.message = 404, "NOT_FOUND", msg
            self.details = {"error": {"code": 404, "status": "NOT_FOUND", "message": msg}}
    e = ClientError()
    assert gu.classify_error(e) == ("model_not_found_or_no_access", False)
    text = wf.explain_provider_error({"kind": "model_not_found_or_no_access", "provider": gu.provider_error_detail(e)}, "gemini")
    assert "HTTP 404" in text and "NOT_FOUND" in text and "gemini-3.8-flash" in text and "No response was generated" in text


def test_saved_pilot_provenance_is_not_rewritten():
    df = pd.read_csv(ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv")
    assert set(df.model_requested) == {"gemini-2.5-flash"}                # the model those saved responses really came from
