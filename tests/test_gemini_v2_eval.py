import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import ssr_eval_lib as lib
from scripts import ssr_persona_matched_evaluation as pm
from scripts import ssr_score_gemini_v2 as sv2

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / "data/pilot_embedding_comparison"
CONFIGS = ["local_mpnet_baseline", "local_minilm", "local_bge_base"]


def rec(i, text, fr="STOP", persona="P1_premium_tier_owner"):
    return {"synthetic_id": i, "persona_id": persona, "response_text": text, "finish_reason": fr}


def test_split_by_quality_keeps_good_and_reports_every_exclusion():
    rows = [rec("A", "Worth it, good battery."),
            rec("B", "It is worth it and the", "MAX_TOKENS"),
            rec("C", "Worth it but the camera is meh"),             # no terminal punctuation
            rec("D", "I give it 4/5, solid phone."),                # numeric rating in text
            rec("E", "   "),
            rec("A", "Duplicate id, ignored."),
            rec("F", "Fine for the price!")]
    passed, excluded = lib.split_by_quality(rows)
    assert [r["synthetic_id"] for r in passed] == ["A", "F"]
    reasons = {e["synthetic_id"] + str(i): e["exclusion_reasons"] for i, e in enumerate(excluded)}
    assert len(passed) + len(excluded) == len(rows)                 # nothing silently dropped
    flat = [r for e in excluded for r in e["exclusion_reasons"]]
    assert {"finish_reason_not_stop", "no_terminal_punctuation", "contains_rating", "empty", "duplicate_id"} <= set(flat)


def test_quality_recomputed_when_stored_flags_missing_or_wrong():
    r = rec("A", "Worth it and the", "MAX_TOKENS")
    r["quality"] = {"is_complete": True, "safe_for_ssr": True}      # stale/wrong stored flags must not be trusted
    passed, excluded = lib.split_by_quality([r])
    assert not passed and excluded[0]["exclusion_reasons"]


def test_real_v2_folder_loads_and_all_saved_responses_pass_quality():
    rows, fails, prompt_ids = sv2.load_v2()
    assert len(rows) == 20 and len({r["synthetic_id"] for r in rows}) == 20 and len(prompt_ids) == 40
    assert {r["persona_id"][:2] for r in rows} == {"P1", "P2"}
    assert all(r["max_output_tokens"] == 2048 and r["finish_reason"] == "STOP" for r in rows)
    passed, excluded = lib.split_by_quality(rows)
    assert len(passed) == 20 and not excluded
    assert all(f["kind"] == "rate_limit_or_quota" for f in fails)
    assert set(r["synthetic_id"] for r in rows) <= set(prompt_ids)


def test_v1_pilot_still_intact_and_separate():
    v1 = [json.loads(l) for l in open(ROOT / "data/pilot_gemini/gemini_generated_responses.jsonl", encoding="utf-8")]
    assert len(v1) == 20 and all("max_output_tokens" not in r for r in v1)
    assert (ROOT / "data/pilot_gemini_v2").resolve() != (ROOT / "data/pilot_gemini").resolve()


def test_v2_score_exports_match_quality_passing_records_and_sum_to_one():
    rows, _, _ = sv2.load_v2()
    passed, _ = lib.split_by_quality(rows)
    ids = [r["synthetic_id"] for r in passed]
    for cfg in CONFIGS:
        d = pd.read_csv(CMP / f"synthetic_scores_{cfg}_gemini_v2.csv", keep_default_na=False)
        assert list(d.synthetic_id) == ids and (d.embedding_config == cfg).all() and (d.source == "gemini_v2").all()
        P = d[[f"p{k}" for k in range(1, 6)]].values
        assert np.allclose(P.sum(axis=1), 1.0, atol=1e-12) and (P >= 0).all()
        assert np.allclose(d.expected_rating, P @ np.arange(1, 6))
        assert list(d.response_text) == [r["response_text"] for r in passed]    # text preserved verbatim


def test_v2_persona_matched_outputs_have_expected_counts():
    long = pd.read_csv(CMP / "persona_matched_gemini_v2_per_response_scores.csv")
    summ = pd.read_csv(CMP / "persona_matched_gemini_v2_summary.csv")
    g = long[long.group == "gemini_generated"]
    assert len(g) == 3 * 20 and set(g.persona_id.str[:2]) == {"P1", "P2"}
    assert g.groupby(["embedding_config", "persona_id"]).size().eq(10).all()
    h = long[long.group == "human_q26"]
    assert h.groupby("persona_id").record_id.nunique().to_dict() == {"P1_premium_tier_owner": 58, "P2_midrange_heavy_user": 44}
    assert (summ.n_gemini_scored == 10).all() and set(summ.n_human_scored) == {58, 44}
    assert np.allclose(long[pm.PMF].sum(axis=1), 1.0, atol=1e-12)


def test_v1_and_v2_outputs_are_separate_files_and_versions_config_is_distinct():
    assert pm.VERSIONS["v1"]["report"] != pm.VERSIONS["v2"]["report"]
    assert pm.VERSIONS["v2"]["dir"].endswith("pilot_gemini_v2") and pm.VERSIONS["v2"]["suffix"] == "gemini_v2"
    assert (CMP / "persona_matched_per_response_scores.csv").exists()          # v1 output untouched
    assert (CMP / "persona_matched_gemini_v2_per_response_scores.csv").exists()
