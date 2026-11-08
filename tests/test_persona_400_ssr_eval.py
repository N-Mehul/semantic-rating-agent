"""
Unit and integration tests for the full 400-response SSR evaluation outputs.
"""

import csv
import json
from pathlib import Path
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "experiments" / "persona_validation_400" / "ssr_evaluation"


@pytest.fixture(scope="module")
def eval_files():
    csv_file = EVAL_DIR / "ssr_scored_responses.csv"
    json_file = EVAL_DIR / "ssr_scored_responses.json"
    summary_file = EVAL_DIR / "execution_summary.json"
    persona_summary_file = EVAL_DIR / "summary_by_persona.json"
    construct_summary_file = EVAL_DIR / "summary_by_construct.json"

    assert csv_file.exists(), f"Missing {csv_file}"
    assert json_file.exists(), f"Missing {json_file}"
    assert summary_file.exists(), f"Missing {summary_file}"
    assert persona_summary_file.exists(), f"Missing {persona_summary_file}"
    assert construct_summary_file.exists(), f"Missing {construct_summary_file}"

    with open(csv_file, "r", encoding="utf-8") as f:
        csv_records = list(csv.DictReader(f))

    with open(json_file, "r", encoding="utf-8") as f:
        json_records = json.load(f)

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    return {
        "csv": csv_records,
        "json": json_records,
        "summary": summary_data,
        "persona_summary": json.loads(persona_summary_file.read_text(encoding="utf-8")),
        "construct_summary": json.loads(construct_summary_file.read_text(encoding="utf-8")),
    }


def test_record_count_is_400(eval_files):
    assert len(eval_files["csv"]) == 400
    assert len(eval_files["json"]) == 400
    assert eval_files["summary"]["valid_scored_responses"] == 400


def test_pmf_sums_to_one_for_all(eval_files):
    for r in eval_files["csv"]:
        pmf_sum = float(r["PMF_Sum"])
        assert abs(pmf_sum - 1.0) < 1e-4, f"Sample {r['Sample_ID']} sum {pmf_sum} != 1.0"
        p_vals = [float(r[f"P{i}"]) for i in range(1, 6)]
        for p in p_vals:
            assert 0.0 <= p <= 1.0, f"Sample {r['Sample_ID']} probability {p} out of bounds"
        assert abs(sum(p_vals) - 1.0) < 1e-3


def test_expected_rating_range(eval_files):
    for r in eval_files["csv"]:
        exp_r = float(r["Expected_Rating"])
        assert 1.0 <= exp_r <= 5.0, f"Expected rating {exp_r} out of range [1, 5]"
        argmax_r = int(r["Predicted_Rating_Argmax"])
        assert 1 <= argmax_r <= 5
        rounded_r = int(r["Predicted_Rating_Rounded"])
        assert 1 <= rounded_r <= 5


def test_metadata_preservation(eval_files):
    required_keys = {
        "Sample_ID",
        "Persona_ID",
        "Persona_Name",
        "Construct",
        "Response_Text",
        "Expected_Rating",
        "Predicted_Rating_Argmax",
        "Predicted_Rating_Rounded",
        "P1",
        "P2",
        "P3",
        "P4",
        "P5",
        "PMF_Sum",
        "Sim_Anchor_1",
        "Sim_Anchor_2",
        "Sim_Anchor_3",
        "Sim_Anchor_4",
        "Sim_Anchor_5",
    }
    sample = eval_files["csv"][0]
    for key in required_keys:
        assert key in sample, f"Missing key {key}"


def test_execution_summary_zero_errors(eval_files):
    assert eval_files["summary"]["status"] == "COMPLETED"
    assert eval_files["summary"]["errors"] == []
    assert eval_files["summary"]["all_pmf_sums_valid"] is True

