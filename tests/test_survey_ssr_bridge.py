"""Deterministic tests for survey_ssr_bridge.py."""

from __future__ import annotations

import csv
from pathlib import Path
import pytest

from survey_ssr_bridge import (
    CONSTRUCT_CODE_MAP,
    DEFAULT_ANCHOR_CSV_PATH,
    ConstructAnchors,
    get_construct_anchors,
    load_all_survey_anchors,
)


def test_default_csv_exists_and_loads():
    """Verify that the official anchor statements CSV exists and contains 6 constructs."""
    assert DEFAULT_ANCHOR_CSV_PATH.is_file()
    anchors_by_construct = load_all_survey_anchors()
    assert len(anchors_by_construct) == 6
    expected_constructs = {
        "Performance Expectancy",
        "Effort Expectancy",
        "Hedonic Motivation",
        "Social Influence",
        "Perceived Protection",
        "Behavioral Intention",
    }
    assert set(anchors_by_construct.keys()) == expected_constructs


@pytest.mark.parametrize(
    "code,expected_name",
    [
        ("PE", "Performance Expectancy"),
        ("EE", "Effort Expectancy"),
        ("HM", "Hedonic Motivation"),
        ("SI", "Social Influence"),
        ("PP", "Perceived Protection"),
        ("BI", "Behavioral Intention"),
        ("Performance Expectancy", "Performance Expectancy"),
        ("behavioral intention", "Behavioral Intention"),
    ],
)
def test_get_construct_anchors_valid_codes(code: str, expected_name: str):
    """Verify loading anchors by 2-letter codes and full names."""
    anchors = get_construct_anchors(code)
    assert isinstance(anchors, ConstructAnchors)
    assert anchors.construct_name == expected_name
    assert len(anchors.statements) == 5
    assert anchors.ratings == (1, 2, 3, 4, 5)
    assert anchors.labels == (
        "Strongly disagree",
        "Disagree",
        "Neither agree nor disagree",
        "Agree",
        "Strongly agree",
    )
    assert anchors.is_domain_adaptation is True
    assert all(isinstance(s, str) and len(s) > 10 for s in anchors.statements)


def test_missing_construct_raises_key_error():
    """Verify KeyError when requesting non-existent construct."""
    with pytest.raises(KeyError, match="Construct 'NON_EXISTENT' not found"):
        get_construct_anchors("NON_EXISTENT")


def test_nonexistent_csv_path_raises_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError when CSV path does not exist."""
    fake_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        get_construct_anchors("PE", csv_path=fake_path)


def test_missing_columns_in_csv_raises_value_error(tmp_path: Path):
    """Verify ValueError when CSV has corrupt headers."""
    bad_csv = tmp_path / "bad_headers.csv"
    bad_csv.write_text("wrong_col,another_col\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        get_construct_anchors("PE", csv_path=bad_csv)


def test_unordered_ratings_raises_value_error(tmp_path: Path):
    """Verify that silently out-of-order ratings (e.g. 1, 3, 2, 4, 5) are rejected."""
    bad_csv = tmp_path / "unordered.csv"
    rows = [
        ["construct", "anchor_rating", "anchor_label", "anchor_statement", "anchor_status"],
        ["Test Construct", "1", "SD", "Statement 1", "STATUS"],
        ["Test Construct", "3", "N", "Statement 3", "STATUS"],
        ["Test Construct", "2", "D", "Statement 2", "STATUS"],
        ["Test Construct", "4", "A", "Statement 4", "STATUS"],
        ["Test Construct", "5", "SA", "Statement 5", "STATUS"],
    ]
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    with pytest.raises(ValueError, match="must be strictly ordered"):
        get_construct_anchors("Test Construct", csv_path=bad_csv)


def test_missing_rating_level_raises_value_error(tmp_path: Path):
    """Verify that an anchor set with fewer than 5 statements is rejected."""
    bad_csv = tmp_path / "incomplete.csv"
    rows = [
        ["construct", "anchor_rating", "anchor_label", "anchor_statement", "anchor_status"],
        ["Test Construct", "1", "SD", "Statement 1", "STATUS"],
        ["Test Construct", "2", "D", "Statement 2", "STATUS"],
        ["Test Construct", "3", "N", "Statement 3", "STATUS"],
        ["Test Construct", "4", "A", "Statement 4", "STATUS"],
    ]
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    with pytest.raises(ValueError, match="must have exactly 5 anchors"):
        get_construct_anchors("Test Construct", csv_path=bad_csv)


def test_duplicate_statements_raises_value_error(tmp_path: Path):
    """Verify rejection when statements are duplicated across levels."""
    bad_csv = tmp_path / "duplicate_stmts.csv"
    rows = [
        ["construct", "anchor_rating", "anchor_label", "anchor_statement", "anchor_status"],
        ["Test Construct", "1", "SD", "Statement Identical", "STATUS"],
        ["Test Construct", "2", "D", "Statement 2", "STATUS"],
        ["Test Construct", "3", "N", "Statement 3", "STATUS"],
        ["Test Construct", "4", "A", "Statement 4", "STATUS"],
        ["Test Construct", "5", "SA", "Statement Identical", "STATUS"],
    ]
    with open(bad_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Duplicate anchor statements detected"):
        get_construct_anchors("Test Construct", csv_path=bad_csv)


def test_facilitating_conditions_is_unsupported_not_aliased_to_perceived_protection():
    """FC has no items or approved anchors in this instrument; it must not silently receive another construct's anchors."""
    from survey_ssr_bridge import CONSTRUCT_CODE_MAP
    assert "FC" not in CONSTRUCT_CODE_MAP
    for name in ("FC", "fc", "Facilitating Conditions"):
        with pytest.raises(KeyError):
            get_construct_anchors(name)
