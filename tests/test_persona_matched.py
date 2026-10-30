import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import ssr_persona_matched_evaluation as pm

ROOT = Path(__file__).resolve().parents[1]
P1, P2 = "P1_premium_tier_owner", "P2_midrange_heavy_user"


def pmf_frame(n, rng):
    m = rng.random((n, 5))
    m = m / m.sum(1, keepdims=True)
    return pd.DataFrame(m, columns=pm.PMF)


def fixtures(n_h=12, n_g=9, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"HS{i:03d}" for i in range(n_h + 3)]
    assign = pd.DataFrame({"respondent_id": ids[: n_h + 2], "persona_id": [P1 if i % 2 else P2 for i in range(n_h + 2)]})
    human = pd.DataFrame({"respondent_id": ids[:n_h] + [ids[n_h + 2]],  # last id has NO assignment
                          "q21": rng.integers(1, 6, n_h + 1), "expected_rating": rng.uniform(1, 5, n_h + 1)})
    human = pd.concat([human, pmf_frame(n_h + 1, rng)], axis=1)
    gem = pd.DataFrame({"synthetic_id": [f"G{i}" for i in range(n_g)], "persona_id": [P1] * n_g,
                        "expected_rating": rng.uniform(1, 5, n_g)})
    gem = pd.concat([gem, pmf_frame(n_g, rng)], axis=1)
    return assign, human, gem


def test_matching_uses_assignment_file_and_drops_unassigned():
    assign, human, gem = fixtures()
    long = pm.build_long(human, assign, gem, "cfg")
    h = long[long.group == "human_q26"]
    assert len(h) == 12                                  # the unassigned human is dropped, not guessed
    assert set(h.persona_id) == {P1, P2}
    expected = assign.set_index("respondent_id").persona_id
    assert all(expected[r.record_id] == r.persona_id for r in h.itertuples())
    assert (long[long.group == "gemini_generated"].persona_id == P1).all()


def test_pmf_rows_sum_to_one_and_bad_rows_rejected():
    assign, human, gem = fixtures()
    long = pm.build_long(human, assign, gem, "cfg")
    assert np.allclose(long[pm.PMF].sum(axis=1), 1.0, atol=1e-12)
    gem.loc[0, "p1"] += 0.2
    with pytest.raises(AssertionError):
        pm.build_long(human, assign, gem, "cfg")


def test_duplicate_gemini_ids_rejected():
    assign, human, gem = fixtures()
    gem.loc[1, "synthetic_id"] = gem.loc[0, "synthetic_id"]
    with pytest.raises(AssertionError):
        pm.build_long(human, assign, gem, "cfg")


def test_summary_counts_ci_threshold_and_missing_group():
    assign, human, gem = fixtures(n_h=20, n_g=9)
    long = pm.build_long(human, assign, gem, "cfg")
    s = pm.summarize(long, [P1, P2], n_boot=200).set_index("persona_id")
    assert s.loc[P1, "n_human_scored"] == 10 and s.loc[P1, "n_gemini_scored"] == 9
    assert s.loc[P1, "diff_gemini_minus_human"] == pytest.approx(s.loc[P1, "gemini_mean_E"] - s.loc[P1, "human_mean_E"])
    assert not np.isnan(s.loc[P1, "diff_ci95_lo"])
    assert s.loc[P2, "n_gemini_scored"] == 0 and np.isnan(s.loc[P2, "gemini_mean_E"])  # nothing invented


def test_no_ci_when_sample_too_small():
    assign, human, gem = fixtures(n_h=14, n_g=3)
    s = pm.summarize(pm.build_long(human, assign, gem, "cfg"), [P1], n_boot=100)
    assert np.isnan(s.loc[0, "diff_ci95_lo"])


def test_count_table_reports_exclusions_and_truncation():
    cleaned = pd.DataFrame({"respondent_id": ["A", "B", "C"], "in_primary_cohort": ["True"] * 3,
                            "text_value_rebuy_usable": ["True", "False", "True"], "rebuy_brand_1to5": ["4", "3", "3"]})
    assign = pd.DataFrame({"respondent_id": ["A", "B", "C"], "persona_id": [P1, P1, P1]})
    human = pd.DataFrame({"respondent_id": ["A", "C"]})
    gem = pd.DataFrame({"synthetic_id": ["G0", "G1"], "persona_id": [P1, P1],
                        "response_text": ["Worth it, honestly.", "It is worth it and the"]})
    c = pm.count_table(cleaned, assign, human, gem, [P1]).iloc[0]
    assert (c.primary_cohort_humans, c.human_q26_usable, c.human_q26_unusable_excluded, c.human_scored_records) == (3, 2, 1, 2)
    assert (c.gemini_responses, c.gemini_possibly_truncated, c.gemini_duplicate_ids) == (2, 1, 0)


def test_safe_write_never_overwrites(tmp_path):
    f = tmp_path / "out.csv"
    a, b = pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})
    assert pm.safe_write_csv(a, f) == f
    assert pm.safe_write_csv(a, f) == f                       # identical content: kept, no sibling
    g = pm.safe_write_csv(b, f)
    h = pm.safe_write_csv(pd.DataFrame({"x": [3]}), f)
    assert (g.name, h.name) == ("out_v2.csv", "out_v3.csv")
    assert f.read_text().split() == ["x", "1"]               # original untouched


def test_real_assignments_match_rederivation_and_real_exports_are_consistent():
    cleaned = pd.read_csv(ROOT / "data/human_survey_cleaned.csv", keep_default_na=False, dtype=str)
    assign = pd.read_csv(ROOT / "data/personas/persona_assignments.csv", keep_default_na=False, dtype=str)
    chk = pm.verify_assignments(assign, cleaned)
    assert chk["same_id_set"] and chk["n_mismatch"] == 0 and chk["ids_unique"]
    ids = [json.loads(l)["synthetic_id"] for l in open(ROOT / "data/pilot_gemini/gemini_generated_responses.jsonl", encoding="utf-8")]
    out = pd.read_csv(ROOT / "data/pilot_embedding_comparison/persona_matched_per_response_scores.csv")
    assert np.allclose(out[pm.PMF].sum(axis=1), 1.0, atol=1e-12)
    assert (out.group == "gemini_generated").sum() == 3 * len(ids)   # 3 encoders x saved Gemini responses
