"""Structural checks for the draft smartphone survey instrument files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CODEBOOK = HERE / "survey_codebook.csv"
ANCHORS = HERE / "smartphone_construct_anchor_statements.csv"
LIKERT_SCALE = (
    "1=Strongly disagree; 2=Disagree; 3=Neither agree nor disagree; "
    "4=Agree; 5=Strongly agree"
)
CONSTRUCTS = (
    "Performance Expectancy",
    "Effort Expectancy",
    "Hedonic Motivation",
    "Social Influence",
    "Perceived Protection",
    "Behavioral Intention",
)


def validate():
    codebook = pd.read_csv(CODEBOOK, keep_default_na=False)
    anchors = pd.read_csv(ANCHORS, keep_default_na=False)
    assert codebook.question_id.notna().all(), "Question ID missing"
    assert codebook.question_text.str.strip().ne("").all(), "Question text missing"
    assert codebook.question_id.is_unique, "Duplicate question IDs"
    assert codebook.export_column_name.is_unique, "Duplicate export column names"
    assert set(codebook.columns) == {
        "question_id", "question_text", "export_column_name", "question_type",
        "scale", "construct", "reverse_scored",
    }, "Codebook schema mismatch"

    likert = codebook[codebook.question_type == "Likert (agreement 1-5)"]
    free_text = codebook[codebook.question_type == "Free text (paragraph optional)"]
    assert len(likert) == 30, f"Expected 30 Likert items; found {len(likert)}"
    assert len(free_text) == 3, f"Expected 3 free-text items; found {len(free_text)}"
    assert len(codebook) == 33, f"Expected 33 respondent items; found {len(codebook)}"
    assert (likert.scale == LIKERT_SCALE).all(), "Invalid or inconsistent Likert scale"
    assert (free_text.scale == "Not applicable").all(), "Free-text scale should be N/A"
    assert (codebook.reverse_scored == "No").all(), "Unexpected reverse-scored item"

    construct_likert = likert[likert.construct.isin(CONSTRUCTS)]
    counts = construct_likert.groupby("construct").size().to_dict()
    assert counts == {name: 4 for name in CONSTRUCTS}, f"Construct item counts incorrect: {counts}"

    aspect = codebook[codebook.construct.str.startswith("Aspect Satisfaction")]
    overall = codebook[codebook.construct == "Overall Smartphone Satisfaction"]
    assert len(aspect) == 5, f"Expected 5 smartphone aspects; found {len(aspect)}"
    assert len(overall) == 1, f"Expected one overall satisfaction item; found {len(overall)}"
    assert set(free_text.construct) == {
        "Performance Expectancy", "Price Value (open-ended only)", "Behavioral Intention"
    }, "Free-text constructs mismatch"
    order = {question_id: index for index, question_id in enumerate(codebook.question_id)}
    assert order["FT_PE01"] < order["PE01"], "PE free text must precede PE Likert items"
    assert order["FT_BI01"] < order["BI01"], "BI free text must precede BI Likert items"

    assert len(anchors) == 30, f"Expected 30 construct anchors; found {len(anchors)}"
    assert set(anchors.construct) == set(CONSTRUCTS), "Anchor constructs mismatch"
    assert (anchors.groupby("construct").size() == 5).all(), "Each construct must have five anchors"
    assert (anchors.groupby("construct").anchor_rating.apply(list).apply(lambda values: values == [1, 2, 3, 4, 5])).all(), "Anchor order must be ratings 1 through 5"
    assert anchors.anchor_statement.str.strip().ne("").all(), "Missing anchor statement"
    assert anchors.anchor_statement.is_unique, "Duplicate anchor statement"
    assert anchors.anchor_status.str.fullmatch(
        "PROPOSED_DOMAIN_ADAPTATION_NOT_ORIGINAL_MAIER_ANCHOR"
    ).all(), "Anchor provenance label missing or invalid"

    print("PASS: 33 questions, 24 construct Likert items, 5 aspects, 1 overall item, 3 free-text items")
    print("PASS: unique question IDs/export columns; consistent 1-5 scale; reverse-scoring flags valid")
    print("PASS: PE and BI free-text prompts precede their related Likert items")
    print("PASS: 30 anchors; 5 per construct; ratings ordered 1-5; provenance clearly non-original")


if __name__ == "__main__":
    validate()
