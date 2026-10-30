"""Clean and validate the human smartphone survey export (Step 3).

Reads data/Capstone_Reviews(Responses).xlsx (never modified) and writes:
  data/human_survey_cleaned.csv
  data/human_survey_inclusion_exclusion_log.csv
  docs/project_status/human_survey_data_quality_report.md

Run from the repository root:  python scripts/clean_human_survey.py
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "Capstone_Reviews(Responses)(1).xlsx" if (ROOT / "data" / "Capstone_Reviews(Responses)(1).xlsx").exists() else ROOT / "data" / "Capstone_Reviews(Responses).xlsx"
CLEAN_PATH = ROOT / "data" / "human_survey_cleaned.csv"
LOG_PATH = ROOT / "data" / "human_survey_inclusion_exclusion_log.csv"
REPORT_PATH = ROOT / "docs" / "project_status" / "human_survey_data_quality_report.md"

# Raw question prefix -> clean column name (matched by "Qn." prefix; header text has newlines).
COLUMN_MAP = {
    "Q1.": "consent",
    "Q2.": "age_18_plus_self_report",
    "Q3.": "age",
    "Q4.": "owns_smartphone",
    "Q5.": "usage_duration",
    "Q6.": "gender",
    "Q7.": "education",
    "Q8.": "current_status",
    "Q10.": "monthly_spending",
    "Q11.": "area_type",
    "Q12.": "brand",
    "Q13.": "price_paid",
    "Q14.": "operating_system",
    "Q15.": "phones_owned_5y",
    "Q16.": "replacement_frequency",
    "Q17.": "upgrade_reason",
    "Q18.": "daily_usage",
    "Q19.": "main_use",
    "Q20.": "tech_comfort_label",
    "Q21.": "rebuy_brand_label",
    "Q22.": "info_sources",
    "Q23.": "text_overall_experience",
    "Q24.": "text_battery_camera",
    "Q25.": "text_performance_display",
    "Q26.": "text_value_rebuy",
}
TEXT_COLS = [
    "text_overall_experience",
    "text_battery_camera",
    "text_performance_display",
    "text_value_rebuy",
]
LIKERT_COLS = {  # label column -> score column
    "tech_comfort_label": "tech_comfort_1to5",
    "rebuy_brand_label": "rebuy_brand_1to5",
}
LIKERT_LABELS = {
    "tech_comfort_label": {
        "1": "Not at all comfortable", "2": "Slightly comfortable", "3": "Moderately comfortable",
        "4": "Very comfortable", "5": "Extremely comfortable",
    },
    "rebuy_brand_label": {
        "1": "Very unlikely", "2": "Unlikely", "3": "Not sure", "4": "Likely", "5": "Very likely",
    },
}
CONSENT_YES = "Yes, I agree to participate."
NONRESPONSE_TOKENS = {"na", "n/a", "nil", "none", "null", "nan", ".", "..", "-", "--", "?"}
NUMERIC_ONLY = re.compile(r"^\d+(\.\d+)?(\s*/\s*\d+(\.\d+)?)?$")
REPEAT_PATTERN = re.compile(r"^(\w{2,3})\1+\w?$", re.IGNORECASE)

EXCLUSION_ORDER = ["no_consent", "age_under_18", "not_smartphone_owner", "unresolved_age_eligibility"]


# ---------------------------------------------------------------- loading
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read every cell as a string, keeping "NA" and blanks distinct (no NaN coercion)."""
    df = pd.read_excel(path, dtype=str, keep_default_na=False)
    rename = {}
    for col in df.columns:
        for prefix, new in COLUMN_MAP.items():
            if col.startswith(prefix):
                rename[col] = new
                break
    missing = set(COLUMN_MAP.values()) - set(rename.values())
    if missing:
        raise ValueError(f"Raw export is missing expected questions: {sorted(missing)}")
    ts_col = df.columns[0]
    df = df.rename(columns=rename).rename(columns={ts_col: "_timestamp"})
    return df


# ---------------------------------------------------------------- eligibility
def parse_age(value: str) -> float:
    s = str(value).strip()
    m = re.match(r"^(\d+)", s)
    if m:
        return float(m.group(1))
    try:
        return float(s)
    except ValueError:
        return float("nan")


def age_status(q2: str, age: float) -> str:
    """'under_18' | 'ok' | 'unresolved'. Q2 blank or Q2/Q3 contradiction => unresolved."""
    q2 = str(q2).strip()
    if pd.notna(age) and age < 18:
        return "under_18"
    if q2 == "Yes" and pd.notna(age) and age >= 18:
        return "ok"
    return "unresolved"


def build_log(df: pd.DataFrame) -> pd.DataFrame:
    age = df["age"].map(parse_age)
    status = [age_status(q, a) for q, a in zip(df["age_18_plus_self_report"], age)]
    flags = pd.DataFrame(index=df.index)
    flags["no_consent"] = df["consent"].str.strip() != CONSENT_YES
    flags["age_under_18"] = pd.Series([s == "under_18" for s in status], index=df.index)
    flags["not_smartphone_owner"] = df["owns_smartphone"].str.strip() != "Yes"
    flags["unresolved_age_eligibility"] = pd.Series([s == "unresolved" for s in status], index=df.index)

    log = pd.DataFrame({
        "respondent_id": df["respondent_id"],
        "source_excel_row": df.index + 2,  # header is Excel row 1
        "source_df_index_0based": df.index,
        "consent": df["consent"],
        "age_18_plus_self_report": df["age_18_plus_self_report"],
        "age": df["age"],
        "owns_smartphone": df["owns_smartphone"],
    })
    for name in EXCLUSION_ORDER:
        log[f"flag_{name}"] = flags[name]
    log["all_reasons"] = flags.apply(lambda r: ";".join(n for n in EXCLUSION_ORDER if r[n]), axis=1)

    # Hierarchical attribution: first failing stage in EXCLUSION_ORDER.
    first = flags.apply(lambda r: next((n for n in EXCLUSION_ORDER if r[n]), ""), axis=1)
    log["first_exclusion_stage"] = first
    log["in_initial_eligible_cohort"] = ~flags[EXCLUSION_ORDER[:3]].any(axis=1)
    log["in_primary_cohort"] = ~flags.any(axis=1)
    log["age_flag_note"] = [
        "Q2 blank; recorded age >= 18 but self-attestation missing" if (not q2.strip() and s == "unresolved")
        else "Q2 contradicts recorded age" if s == "unresolved"
        else ""
        for q2, s in zip(df["age_18_plus_self_report"], status)
    ]
    return log


# ---------------------------------------------------------------- Likert
def parse_likert(label: str, column: str) -> float:
    """Map '4 — Very comfortable' to 4. Returns NaN (never guesses) for blank/unrecognised labels."""
    m = re.match(r"^\s*([1-5])\s*[—–-]\s*(.+?)\s*$", str(label))
    if not m:
        return float("nan")
    num, text = m.groups()
    if LIKERT_LABELS[column][num].lower() != text.lower():
        return float("nan")
    return float(num)


# ---------------------------------------------------------------- text quality
def text_flags(value: str) -> dict:
    raw = str(value)
    s = raw.strip()
    low = s.lower()
    missing = s == ""
    token = (not missing) and low in NONRESPONSE_TOKENS
    numeric = (not missing) and bool(NUMERIC_ONLY.match(low))
    alpha = re.sub(r"[^A-Za-z]", "", s)
    single_word = (not missing) and len(s.split()) == 1 and alpha == s
    no_vowel = not re.search(r"[aeiouy]", low)
    gibberish = bool(single_word and (
        (len(s) >= 3 and no_vowel and low not in {"hmm", "brr"})
        or (len(s) >= 4 and REPEAT_PATTERN.match(low))
    ))
    words = len(s.split())
    return {
        "missing": missing,
        "nonresponse_token": token,
        "numeric_only": numeric,
        "suspected_gibberish": gibberish,
        "short_lt3_words": (not missing) and words < 3,
        "usable": not (missing or token or numeric or gibberish),
    }


# ---------------------------------------------------------------- pipeline
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw.drop(columns=["_timestamp"]).copy()
    df.insert(0, "respondent_id", [f"HS{i + 1:03d}" for i in range(len(df))])
    log = build_log(df)

    for lab, score in LIKERT_COLS.items():
        df[score] = df[lab].map(lambda v, c=lab: parse_likert(v, c))
        df[f"{score}_missing"] = df[score].isna()

    for col in TEXT_COLS:
        for k, v in zip(["missing", "nonresponse_token", "numeric_only", "suspected_gibberish",
                         "short_lt3_words", "usable"], zip(*[text_flags(x).values() for x in df[col]])):
            df[f"{col}_{k}"] = list(v)
    usable_cols = [f"{c}_usable" for c in TEXT_COLS]
    df["n_text_usable"] = df[usable_cols].sum(axis=1)
    df["text_usable_any"] = df["n_text_usable"] > 0
    df["text_usable_all4"] = df["n_text_usable"] == len(TEXT_COLS)

    cohort = log.set_index("respondent_id")
    df["in_initial_eligible_cohort"] = df["respondent_id"].map(cohort["in_initial_eligible_cohort"])
    df["in_primary_cohort"] = df["respondent_id"].map(cohort["in_primary_cohort"])
    df["age_flag_note"] = df["respondent_id"].map(cohort["age_flag_note"])
    df["eligible_demographic"] = df["in_primary_cohort"]
    df["eligible_likert"] = df["in_primary_cohort"] & ~df[[f"{s}_missing" for s in LIKERT_COLS.values()]].any(axis=1)
    df["eligible_text_any"] = df["in_primary_cohort"] & df["text_usable_any"]
    df["eligible_text_all4"] = df["in_primary_cohort"] & df["text_usable_all4"]

    log["eligible_demographic"] = df["eligible_demographic"].values
    log["eligible_likert"] = df["eligible_likert"].values
    log["eligible_text_any"] = df["eligible_text_any"].values
    log["eligible_text_all4"] = df["eligible_text_all4"].values
    return df, log


# ---------------------------------------------------------------- validation
def validate(raw: pd.DataFrame, df: pd.DataFrame, log: pd.DataFrame, raw_hash_before: str) -> list[tuple[str, bool, str]]:
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    add("row count preserved", len(raw) == len(df) == len(log), f"raw={len(raw)} cleaned={len(df)} log={len(log)}")
    add("respondent IDs unique", df["respondent_id"].is_unique and log["respondent_id"].is_unique,
        f"duplicates={int(df['respondent_id'].duplicated().sum())}")
    add("IDs identical in cleaned and log", list(df["respondent_id"]) == list(log["respondent_id"]), "")
    mism = sum(int((raw[c].values != df[c].values).sum()) for c in COLUMN_MAP.values())
    add("original answers byte-identical (all 25 survey columns)", mism == 0, f"mismatched cells={mism}")
    add("raw workbook unchanged (sha256)", sha256(RAW_PATH) == raw_hash_before, raw_hash_before[:16] + "…")
    counts = log["all_reasons"].map(lambda s: len([x for x in s.split(";") if x]))
    first_sum = int((log["first_exclusion_stage"] != "").sum())
    add("hierarchy has no double counting",
        first_sum + int(log["in_primary_cohort"].sum()) == len(log),
        f"excluded(first reason)={first_sum} + primary={int(log['in_primary_cohort'].sum())} = {len(log)}")
    stage_in_reasons = all(f in r.split(";") for f, r in zip(log["first_exclusion_stage"], log["all_reasons"]) if f)
    add("first-stage reason is always among recorded reasons (overlaps kept)", stage_in_reasons,
        f"respondents with >1 reason={int((counts > 1).sum())}")
    add("NA preserved verbatim, not blanked",
        all(((raw[c] == "NA") == (df[c] == "NA")).all() for c in TEXT_COLS), "")
    for lab, score in LIKERT_COLS.items():
        valid = df[score].dropna()
        ok = valid.between(1, 5).all() and (df[score].isna() == df[f"{score}_missing"]).all()
        blank_guess = ((raw[lab].str.strip() == "") & df[score].notna()).sum()
        add(f"{score} in 1..5, no guessed values", ok and blank_guess == 0,
            f"valid={len(valid)} missing={int(df[score].isna().sum())}")
    unresolved_ids = log.loc[log["flag_unresolved_age_eligibility"], "source_df_index_0based"].tolist()
    add("unresolved age cases flagged and excluded from primary",
        not log.loc[log["flag_unresolved_age_eligibility"], "in_primary_cohort"].any(),
        f"flagged count={len(unresolved_ids)} indices={sorted(unresolved_ids)}")
    add("primary cohort has no excluded-flag respondent",
        not log.loc[log["in_primary_cohort"], [f"flag_{n}" for n in EXCLUSION_ORDER]].any().any(), "")
    add("text-unusable respondents retained for demographic/Likert",
        int(df["eligible_demographic"].sum()) == int(df["in_primary_cohort"].sum()), "")
    return checks


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- report
def stage_counts(log: pd.DataFrame) -> list[tuple[str, int, int, int]]:
    rows, remaining = [], len(log)
    for name in EXCLUSION_ORDER:
        newly = int((log["first_exclusion_stage"] == name).sum())
        any_flag = int(log[f"flag_{name}"].sum())
        remaining -= newly
        rows.append((name, newly, any_flag, remaining))
    return rows


def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_report(raw, df, log, checks, raw_hash) -> None:
    n = len(raw)
    primary = df[df["in_primary_cohort"]]
    all_pass = all(ok for _, ok, _ in checks)
    stages = stage_counts(log)
    stage_rows = [("Raw submissions", "-", "-", n)] + [(s, new, anyf, rem) for s, new, anyf, rem in stages]
    overlap = log[log["all_reasons"].str.contains(";")][["respondent_id", "source_df_index_0based", "all_reasons"]]
    unresolved = log[log["flag_unresolved_age_eligibility"]]
    initial = int(log["in_initial_eligible_cohort"].sum())
    final = int(log["in_primary_cohort"].sum())

    text_rows = []
    for c in TEXT_COLS:
        p = primary
        text_rows.append((c, int(p[f"{c}_missing"].sum()), int(p[f"{c}_nonresponse_token"].sum()),
                          int(p[f"{c}_numeric_only"].sum()), int(p[f"{c}_suspected_gibberish"].sum()),
                          int(p[f"{c}_short_lt3_words"].sum()), int(p[f"{c}_usable"].sum())))
    gib = []
    for c in TEXT_COLS:
        for rid, v in primary.loc[primary[f"{c}_suspected_gibberish"], ["respondent_id", c]].itertuples(index=False):
            gib.append((rid, c, repr(v)))

    miss_rows = []
    for c in COLUMN_MAP.values():
        blanks = int((primary[c].str.strip() == "").sum())
        if blanks:
            miss_rows.append((c, blanks))
    likert_rows = []
    for lab, score in LIKERT_COLS.items():
        dist = primary[score].value_counts().sort_index()
        likert_rows.append((score, int(primary[score].notna().sum()), int(primary[score].isna().sum()),
                            ", ".join(f"{int(k)}:{v}" for k, v in dist.items())))
    check_rows = [(name, "PASS" if ok else "FAIL", detail) for name, ok, detail in checks]

    md = f"""# Human Survey Data Quality Report (Step 3)

Generated by `scripts/clean_human_survey.py`. Source: `data/Capstone_Reviews(Responses).xlsx` (unmodified, sha256 `{raw_hash}`).

**Validation status: {"all automated checks passed" if all_pass else "ONE OR MORE CHECKS FAILED - do not treat the data as validated"}.** Passing these checks shows the cleaning logic behaved as specified; it does not establish the survey's construct validity.

## 1. Sample counts

Exclusions are applied hierarchically in the order below. "Newly excluded" counts each respondent once at the first stage they fail; "Flagged at stage" counts every respondent with that reason, including those already excluded earlier (overlap).

{md_table(["Stage", "Newly excluded", "Flagged at stage (incl. overlap)", "Remaining"], stage_rows)}

- **Initial eligible cohort** (consent, recorded age >= 18, smartphone owner): **{initial}**.
- **Final primary analysis cohort**: **{final}** = {initial} minus {initial - final} respondents with unresolved age eligibility.
- The difference arises because Q2 (age self-attestation) is blank for 3 respondents and contradicts the recorded age for 1; no approved protocol in the repository establishes eligibility in those cases, so they are held out of the primary cohort (conservative choice).

Respondents with more than one exclusion reason:

{md_table(["respondent_id", "df index (0-based)", "all reasons"], overlap.itertuples(index=False)) if len(overlap) else "None."}

### Unresolved age-eligibility cases

The reference numbers 117, 34, 153, 161 are 0-based DataFrame indices (Excel rows = index + 2).

{md_table(["respondent_id", "df index", "Excel row", "Q2", "Q3 age", "note"], [(r.respondent_id, r.source_df_index_0based, r.source_excel_row, repr(r.age_18_plus_self_report), r.age, r.age_flag_note) for r in unresolved.itertuples()])}

If the study protocol later accepts recorded age (Q3 >= 18) as sufficient for the three blank-Q2 cases, they can be reinstated by changing `age_status` in the script; the primary cohort would become {final + 3}. Row 117 (Q2 = "No" but age 18) would stay excluded absent a documented correction.

### Analysis-specific eligible counts (within primary cohort)

{md_table(["Analysis", "N"], [
    ("Demographic (no dependence on text)", int(df["eligible_demographic"].sum())),
    ("Likert (both 1-5 items present)", int(df["eligible_likert"].sum())),
    ("Text, at least one usable answer", int(df["eligible_text_any"].sum())),
    ("Text, all four answers usable", int(df["eligible_text_all4"].sum())),
])}

Respondents are never dropped from demographic or Likert analyses for text problems.

## 2. Text quality (primary cohort, n={final})

Original text is kept verbatim in the cleaned CSV (including `NA`, blanks, "." and keyboard-mash strings). Flags are separate columns; "usable" = not missing, not a non-response token, not numeric-only, not suspected gibberish. `short_lt3_words` is informational and does **not** make an answer unusable.

{md_table(["Field", "Missing", "Non-response token", "Numeric only", "Suspected gibberish", "<3 words", "Usable"], text_rows)}

Suspected-gibberish heuristic (single alphabetic word with no vowel, or a repeated 2-3 letter block such as "okok") is crude and can produce false positives/negatives. Items flagged for manual review:

{md_table(["respondent_id", "field", "value"], gib) if gib else "None."}

Many "usable" answers are one-word ("Good"). Whether they are rich enough for SSR-style scoring is an analysis-stage decision, not a cleaning one.

## 3. Missing values (primary cohort)

Blank cells per source column (columns not listed have none):

{md_table(["Column", "Blank count"], miss_rows) if miss_rows else "None."}

Blank text answers are not imputed; indicators are `<field>_missing` columns. "NA" is a non-blank answer and is flagged via `<field>_nonresponse_token`. Note: reading the CSV with default pandas settings converts the literal string "NA" to NaN; use `keep_default_na=False` to preserve it.

## 4. Likert-style items

Q20 (technology comfort) and Q21 (likelihood of rebuying the same brand) are the only Likert-type items in this export. Labels such as "4 - Very comfortable" are parsed to integers 1-5, cross-checking number against label text; unmatched or blank labels would become missing (never guessed).

{md_table(["Score column", "Valid", "Missing", "Distribution"], likert_rows)}

## 5. Validation checks

{md_table(["Check", "Result", "Detail"], check_rows)}

## 6. Limitations and unresolved issues

1. **Instrument mismatch.** The collected export (Q1-Q26: demographics, 2 Likert items, 4 free-text items) is not the PE/EE/HM/SI/PP/BI item battery in `docs/survey_instrument/survey_codebook.csv`. The existing construct mapping is preserved unchanged; this survey does **not** measure or validate all UTAUT2 constructs, and Q20/Q21 should not be treated as validated construct scales.
2. **Eligibility protocol.** No approved protocol text was found in the repository, so the 4 unresolved cases are excluded conservatively. Confirm with the study owner.
3. **Consent text, ethics approval and Q9** (absent from the export; numbering jumps from Q8 to Q10) were not verifiable from the data.
4. **Sample composition.** The sample is overwhelmingly young students (see `current_status`, `age`); results will not generalise to the wider population.
5. The `Timestamp` column is intentionally not copied to the cleaned CSV; the mapping from `respondent_id` to source row is in the inclusion/exclusion log (`source_excel_row`). IDs are sequential by submission order (`HS001` = Excel row 2), and no direct identifiers exist in the export.
6. `canonical_ssr.py`, anchor statements, frozen baselines and model artifacts were not touched.

## 7. Outputs

- `data/human_survey_cleaned.csv` - all {n} submissions with eligibility, Likert-score and text-quality flags.
- `data/human_survey_inclusion_exclusion_log.csv` - respondent-level log.
- `scripts/clean_human_survey.py`, `tests/test_clean_human_survey.py`.
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md, encoding="utf-8")


def main() -> int:
    raw_hash = sha256(RAW_PATH)
    raw = load_raw()
    df, log = clean(raw)
    checks = validate(raw, df, log, raw_hash)
    df.to_csv(CLEAN_PATH, index=False, encoding="utf-8")
    log.to_csv(LOG_PATH, index=False, encoding="utf-8")
    write_report(raw, df, log, checks, raw_hash)
    for name, ok, detail in checks:
        print(("PASS " if ok else "FAIL ") + name + (f" [{detail}]" if detail else ""))
    print(f"raw={len(raw)} initial_eligible={int(log['in_initial_eligible_cohort'].sum())} "
          f"primary={int(log['in_primary_cohort'].sum())}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
