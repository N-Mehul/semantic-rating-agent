# Step 3 Final Verification Audit

Audit run: 2026-10-28, about 11:13 local time. Scope: human survey cleaning outputs. No survey answers, SSR code, anchors, baselines or model artifacts were modified by this audit; the only file written is this document. Persona generation has not been started.

## Results

| # | Check | Result | Evidence |
|---|---|---|---|
| 1a | 6 new cleaning tests | PASS | `python -m pytest tests/test_clean_human_survey.py -q`: 6 passed |
| 1b | Full suite with the documented command | PASS | `python -m pytest --confcutdir=tests tests/` (from `PROJECT_HANDOVER_QUICK_START.md`): 198 passed (includes the 6 new tests). Unscoped `pytest` was deliberately not used, per that document. |
| 2 | Protected files unmodified | PASS with limitation | See section 2 |
| 3 | Script rerun reproduces counts | PASS | Rerun gave 215 raw / 204 initial eligible / 200 primary. Cleaned CSV and log are byte-identical to the previous run (`cmp`), and the report is identical apart from its 3 header lines. 13/13 internal checks passed. |
| 4 | Raw workbook unchanged | PASS with limitation | See section 4 |
| 5 | 215 rows kept; primary flagged; log maps to Excel rows | PASS | Cleaned CSV = 215 rows, log = 215 rows, `in_primary_cohort` True for 200 in both. `source_excel_row` runs 2..216 contiguously and equals `source_df_index_0based + 2`. `consent` and `age` in the log match the workbook at the mapped row for all 215. |
| 6 | Literal `NA` preserved | PASS | Read with `keep_default_na=False`: HS101 has `NA` in all four text fields and HS164 has `NA` in three and `NA ` (trailing space, kept verbatim) in `text_value_rebuy`. Default pandas read turns them into NaN, as documented in the report. |
| 7 | Gibberish flags reviewed | PASS (nothing altered) | See section 7 |
| 8 | Short meaningful answers not auto-rejected | PASS | See section 8 |
| 9 | Report separates eligibility concepts | PASS | See section 9 |
| 10 | No edits to SSR math, anchors, baselines, responses | PASS with limitation | See sections 2 and 4 |

## 2. Protected files and git state

- `git status` is identical to the start-of-session snapshot except for the new untracked `scripts/` (this work). The `git diff` shows only the pre-existing modifications to `agent.py`, `batch_predict.py` and `requirements.txt` and the pre-existing deletions of `cleanup_candidates.md`, `diagnose.py`, `run_analysis.py` and `verify.py`. None of these were touched in Step 3. They are unrelated to Step 3.
- **Limitation:** `canonical_ssr.py`, `survey_ssr_bridge.py`, the anchor statements, `data/*` experiment outputs, `results/` and `analysis/` are untracked, so git cannot show a content diff for them. Evidence is by modification time instead: `canonical_ssr.py` was last modified 2026-10-11 14:04 and `smartphone_construct_anchor_statements.csv` 2026-10-11 22:32. A search of the whole repo for files modified in the last 120 minutes found no baseline, model artifact, `results/` or `analysis/` file and neither of those two. This is good evidence but not a cryptographic proof.
- **Other changes, not caused by Step 3 outputs:** files modified earlier in the same session window (10:22-10:44) are `survey_ssr_bridge.py`, `tests/test_survey_ssr_bridge.py`, `tests/test_ssr_reference_parity.py`, `scratch/dataset_stats.py`, and four files under `docs/project_status/` (handover report, quick start, complete status audit, evidence matrix). `.claude/settings.local.json` also changed. These predate the cleaning script (created 11:11) and were not written by the cleaning workflow; the audit did not review their content.

## 4. Raw workbook integrity

- Current SHA-256: `10d4bc65c74bbfdb5dbd06ae169719a2576a4f0508695724f4706eb618f7f0f7`.
- The script hashes the workbook before reading and re-hashes after writing; the hash matched, and the same value is quoted in the quality report.
- **Limitation:** the workbook is untracked in git and no baseline hash was recorded independently before Step 3. Nothing outside the script's own before/after comparison proves it was never altered. Its modification time (10:22) is earlier than all Step 3 outputs, and the script has no code path that writes to it. Recommend recording this hash (e.g. in the project notes) as the reference going forward.

## 7. Suspected-gibberish flags (7 flags, 4 respondents, all in the primary cohort)

All cell values below match the raw workbook exactly; nothing was removed or rewritten.

| Respondent | Excel row | Field | Raw value |
|---|---|---|---|
| HS002 | 3 | text_overall_experience (Q23) | `Gdhdb` |
| HS002 | 3 | text_battery_camera (Q24) | `Hshdbdbd` |
| HS002 | 3 | text_performance_display (Q25) | `Hdhdhdh` |
| HS002 | 3 | text_value_rebuy (Q26) | `Hdhdhd` |
| HS004 | 5 | text_performance_display (Q25) | `Okokk ` (trailing space) |
| HS156 | 157 | text_performance_display (Q25) | `okok` |
| HS158 | 159 | text_performance_display (Q25) | `okok` |

HS002 (all four answers keyboard-mash) is a confident flag. `okok`/`Okokk` are lower confidence: they may be a genuine "ok ok" answer, so they are only flagged, not deleted. A reviewer should decide whether to keep them as usable.

## 8. Text-usability rules

- Unusable means: blank, non-response token (`NA`, `.` and similar), numeric-only (e.g. `10/10`, `9`), or suspected gibberish. Word count is not a criterion.
- Short answers (< 3 words) per field, and how many remain usable: overall 160 → 150 usable, battery/camera 130 → 119, performance/display 154 → 138, value/rebuy 107 → 101. The short answers that were rejected are all explained by the rules above (listed values: `10/10`, `.`, `NA`, `9`, `3/5`, `okok`, gibberish strings, `15000`). Examples kept usable: `Good`, `No`, `Idk`, `Not bad`, `Yes`.
- All rejected non-short answers are missing values (the single blank in Q23).
- **Judgement call to confirm:** numeric-only ratings such as `10/10` are classed as unusable because they are not natural language. They may still be informative as stated ratings; the flag `*_numeric_only` lets later steps reuse them. Whether one-word answers carry enough signal for SSR scoring remains a Step 4 decision.

## 9. Report clarity

The report keeps three notions apart: (a) demographic and Likert eligibility (n = 200, never reduced by text problems), (b) text-analysis eligibility (at least one usable answer = 195; all four usable = 181), and (c) text completeness per field (missing, non-response token, numeric-only, gibberish, short, usable counts). The cleaned CSV has separate columns for each: `eligible_demographic`, `eligible_likert`, `eligible_text_any`, `eligible_text_all4` and the per-field flags.

## Unresolved issues

1. The four age cases (indices 34, 117, 153, 161) remain excluded pending a protocol decision; reinstating the three blank-Q2 cases would give 203.
2. No independently recorded baseline hash for the raw workbook (section 4).
3. Protected untracked files verified by modification time only (section 2).
4. `okok`-type flags need human review (section 7); numeric-only policy needs confirmation (section 8).
5. The survey does not measure the PE/EE/HM/SI/PP/BI item battery in `survey_codebook.csv`; no UTAUT2 construct validation is claimed.
6. The "data cleaned and checked" status applies to the logic checks above, not to the survey's construct validity.
