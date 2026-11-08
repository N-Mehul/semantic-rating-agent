"""
Unit and integration tests for the 400-response synthetic persona dataset.
"""

import csv
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data" / "experiments" / "persona_validation_400"
CSV_PATH = DATASET_DIR / "synthetic_responses.csv"
METADATA_PATH = DATASET_DIR / "generation_metadata.json"
LOG_PATH = DATASET_DIR / "generation_log.csv"
README_PATH = DATASET_DIR / "README.md"

EXPECTED_PERSONAS = {"P1", "P2", "P3", "P4"}
EXPECTED_CONSTRUCTS = {"PE", "EE", "HM", "SI", "PP", "BI"}


def test_dataset_files_exist():
    assert CSV_PATH.exists(), f"Missing dataset CSV at {CSV_PATH}"
    assert METADATA_PATH.exists(), f"Missing metadata at {METADATA_PATH}"
    assert LOG_PATH.exists(), f"Missing log at {LOG_PATH}"
    assert README_PATH.exists(), f"Missing README at {README_PATH}"


def test_dataset_schema_and_counts():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 400, f"Expected 400 responses, got {len(reader)}"

    expected_cols = ["Sample_ID", "Persona_ID", "Persona_Name", "Construct", "Response_Text"]
    assert list(reader[0].keys()) == expected_cols, f"Invalid columns: {list(reader[0].keys())}"

    persona_counts = {}
    construct_counts = {}
    sample_ids = set()
    response_texts = set()

    rating_pattern = re.compile(
        r"\b([1-5]\s*/\s*5|[1-5]\s*stars?|rating\s*:\s*[1-5]|\b[1-5]\s*out\s*of\s*5)\b",
        re.IGNORECASE,
    )

    for r in reader:
        sid = r["Sample_ID"]
        pid = r["Persona_ID"]
        c = r["Construct"]
        txt = r["Response_Text"].strip()

        assert sid not in sample_ids, f"Duplicate Sample_ID found: {sid}"
        sample_ids.add(sid)

        assert txt not in response_texts, f"Duplicate Response_Text found for {sid}"
        response_texts.add(txt)

        assert len(txt) >= 40, f"Response text too short ({len(txt)} chars) for {sid}"
        assert pid in EXPECTED_PERSONAS, f"Invalid Persona_ID: {pid}"
        assert c in EXPECTED_CONSTRUCTS, f"Invalid Construct: {c}"

        matches = rating_pattern.findall(txt)
        assert len(matches) == 0, f"Found numeric rating pattern {matches} in {sid}"

        persona_counts[pid] = persona_counts.get(pid, 0) + 1
        construct_counts[c] = construct_counts.get(c, 0) + 1

    for p in EXPECTED_PERSONAS:
        assert persona_counts[p] == 100, f"Persona {p} count is {persona_counts.get(p)}, expected 100"

    for c in EXPECTED_CONSTRUCTS:
        assert construct_counts[c] > 0, f"Construct {c} has zero responses"


def test_metadata_consistency():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_responses_generated"] == 400
    assert meta["personas_count"] == 4
    for p, count in meta["responses_per_persona"].items():
        assert count == 100

