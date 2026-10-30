"""Bridge module for loading technology-adoption survey anchor statements.

This module provides deterministic loading and validation of domain-adapted Likert
anchor statements from ``docs/survey_instrument/smartphone_construct_anchor_statements.csv``.

CRITICAL METHODOLOGICAL DISTINCTION:
These anchor statements are proposed domain-specific adaptations for smartphone
technology adoption research (Performance Expectancy, Effort Expectancy, Hedonic
Motivation, Social Influence, Perceived Protection, Behavioral Intention).
They are NOT the original six purchase-intent anchor sets from Maier et al.
(arXiv:2510.08338v3), which were not published by the authors.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

DEFAULT_ANCHOR_CSV_PATH: Final[Path] = (
    Path(__file__).resolve().parent
    / "docs"
    / "survey_instrument"
    / "smartphone_construct_anchor_statements.csv"
)

# Standard construct code to canonical construct name mapping
CONSTRUCT_CODE_MAP: Final[dict[str, str]] = {
    "PE": "Performance Expectancy",
    "EE": "Effort Expectancy",
    "HM": "Hedonic Motivation",
    "SI": "Social Influence",
    "PP": "Perceived Protection",
    # No "FC" entry: Facilitating Conditions is not a construct of this instrument (no items in survey_codebook.csv, no
    # anchors in the approved CSV) and is unrelated to Perceived Protection (privacy/security of personal data, PP01-PP04).
    # An earlier alias FC -> Perceived Protection was removed so FC is reported as unsupported instead of receiving PP anchors.
    "BI": "Behavioral Intention",
}

EXPECTED_RATINGS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class ConstructAnchors:
    """Represents a verified, ordered five-point Likert anchor set for a construct."""

    construct_code: str
    construct_name: str
    statements: tuple[str, str, str, str, str]
    ratings: tuple[int, int, int, int, int]
    labels: tuple[str, str, str, str, str]
    is_domain_adaptation: bool = True

    def to_anchor_list(self) -> list[str]:
        """Return the statements as a list ordered from rating 1 to 5."""
        return list(self.statements)


def load_all_survey_anchors(
    csv_path: str | Path = DEFAULT_ANCHOR_CSV_PATH,
) -> dict[str, list[dict[str, str]]]:
    """Read the raw anchor CSV and group rows by construct name.
    
    Validates file existence, expected CSV columns, and non-empty rows.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Anchor statement CSV not found at: {path}")

    expected_columns = {
        "construct",
        "anchor_rating",
        "anchor_label",
        "anchor_statement",
        "anchor_status",
    }

    records_by_construct: dict[str, list[dict[str, str]]] = {}
    with open(path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or corrupted CSV header at: {path}")

        fieldnames_set = set(reader.fieldnames)
        missing_fields = expected_columns - fieldnames_set
        if missing_fields:
            raise ValueError(
                f"Anchor CSV missing required columns: {sorted(missing_fields)}"
            )

        for row_idx, row in enumerate(reader, start=2):
            construct = row["construct"].strip()
            if not construct:
                raise ValueError(f"Missing construct name on line {row_idx}")
            records_by_construct.setdefault(construct, []).append(row)

    return records_by_construct


def get_construct_anchors(
    construct: str,
    csv_path: str | Path = DEFAULT_ANCHOR_CSV_PATH,
) -> ConstructAnchors:
    """Retrieve verified, strictly ordered 1-to-5 anchor statements for a construct.

    Parameters
    ----------
    construct : str
        Construct identifier or code (e.g., 'PE', 'Performance Expectancy', 'EE', 'BI').
    csv_path : str | Path, optional
        Path to the anchor statements CSV.

    Returns
    -------
    ConstructAnchors
        Validated anchor object containing the 5 statements in exact 1..5 order.

    Raises
    ------
    KeyError
        If the construct is unknown or missing from the CSV.
    ValueError
        If the anchor levels are incomplete, unordered, duplicated, or invalid.
    """
    clean_construct = construct.strip()
    target_name = CONSTRUCT_CODE_MAP.get(clean_construct.upper(), clean_construct)

    all_anchors = load_all_survey_anchors(csv_path)

    # Check if construct exists in CSV
    matching_constructs = [
        name for name in all_anchors.keys() if name.lower() == target_name.lower()
    ]
    if not matching_constructs:
        available = sorted(list(all_anchors.keys()) + list(CONSTRUCT_CODE_MAP.keys()))
        raise KeyError(
            f"Construct '{construct}' not found. Available constructs: {available}"
        )

    matched_name = matching_constructs[0]
    rows = all_anchors[matched_name]

    if len(rows) != len(EXPECTED_RATINGS):
        raise ValueError(
            f"Construct '{matched_name}' must have exactly 5 anchors, found {len(rows)}"
        )

    # Validate ratings and statements without silent sorting/reordering
    ratings: list[int] = []
    statements: list[str] = []
    labels: list[str] = []

    for item in rows:
        try:
            r = int(item["anchor_rating"].strip())
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid rating value '{item.get('anchor_rating')}' for construct '{matched_name}'"
            ) from exc

        stmt = item["anchor_statement"].strip()
        if not stmt:
            raise ValueError(
                f"Empty anchor statement found for construct '{matched_name}', rating {r}"
            )

        label = item.get("anchor_label", "").strip()

        ratings.append(r)
        statements.append(stmt)
        labels.append(label)

    # Assert strictly ordered 1 through 5
    if tuple(ratings) != EXPECTED_RATINGS:
        raise ValueError(
            f"Anchor ratings for '{matched_name}' must be strictly ordered [1, 2, 3, 4, 5], "
            f"found {ratings}. Silent reordering is prohibited."
        )

    # Assert unique statements
    if len(set(statements)) != len(EXPECTED_RATINGS):
        raise ValueError(
            f"Duplicate anchor statements detected for construct '{matched_name}'"
        )

    # Reverse lookup for code
    canonical_code = clean_construct.upper()
    for code, name in CONSTRUCT_CODE_MAP.items():
        if name.lower() == matched_name.lower():
            canonical_code = code
            break

    return ConstructAnchors(
        construct_code=canonical_code,
        construct_name=matched_name,
        statements=(statements[0], statements[1], statements[2], statements[3], statements[4]),
        ratings=(ratings[0], ratings[1], ratings[2], ratings[3], ratings[4]),
        labels=(labels[0], labels[1], labels[2], labels[3], labels[4]),
        is_domain_adaptation=True,
    )
