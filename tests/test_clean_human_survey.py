import math

import pandas as pd

from scripts import clean_human_survey as c


def make_raw(rows):
    base = {col: "x" for col in c.COLUMN_MAP.values()}
    base.update(consent=c.CONSENT_YES, age_18_plus_self_report="Yes", age="20", owns_smartphone="Yes",
                tech_comfort_label="4 — Very comfortable", rebuy_brand_label="3 — Not sure",
                text_overall_experience="Works well for me", text_battery_camera="Good battery",
                text_performance_display="Fast screen", text_value_rebuy="Yes worth it")
    df = pd.DataFrame([{**base, **r} for r in rows])
    df.insert(0, "_timestamp", "t")
    return df


def run(rows):
    return c.clean(make_raw(rows))


def test_hierarchy_no_double_counting():
    df, log = run([
        {},
        {"consent": "No, I do not agree.", "owns_smartphone": "No"},
        {"age": "17", "age_18_plus_self_report": "No"},
        {"owns_smartphone": "No"},
        {"age_18_plus_self_report": ""},
        {"age_18_plus_self_report": "No", "age": "18"},
    ])
    assert list(log["first_exclusion_stage"]) == [
        "", "no_consent", "age_under_18", "not_smartphone_owner",
        "unresolved_age_eligibility", "unresolved_age_eligibility"]
    assert "not_smartphone_owner" in log.loc[1, "all_reasons"]
    assert log["in_initial_eligible_cohort"].sum() == 3
    assert log["in_primary_cohort"].sum() == 1


def test_unresolved_age_not_in_primary_but_in_initial():
    _, log = run([{"age_18_plus_self_report": ""}])
    assert log.loc[0, "in_initial_eligible_cohort"] and not log.loc[0, "in_primary_cohort"]


def test_text_preserved_and_flags_separate():
    df, _ = run([{"text_overall_experience": "NA", "text_battery_camera": "", "text_performance_display": "Hdhdhd"}])
    assert df.loc[0, "text_overall_experience"] == "NA"
    assert df.loc[0, "text_battery_camera"] == ""
    assert df.loc[0, "text_overall_experience_nonresponse_token"]
    assert df.loc[0, "text_battery_camera_missing"]
    assert df.loc[0, "text_performance_display_suspected_gibberish"]
    assert df.loc[0, "eligible_demographic"] and df.loc[0, "eligible_likert"]
    assert df.loc[0, "eligible_text_any"] and not df.loc[0, "eligible_text_all4"]


def test_text_flags_cases():
    assert c.text_flags("Good")["usable"]
    assert c.text_flags("No")["usable"]
    assert c.text_flags("Idk")["usable"]
    assert not c.text_flags("10/10")["usable"]
    assert not c.text_flags(".")["usable"]
    assert c.text_flags("okokk")["suspected_gibberish"]
    assert not c.text_flags("Omg")["suspected_gibberish"]


def test_likert_parsing_no_guessing():
    assert c.parse_likert("5 — Very likely", "rebuy_brand_label") == 5
    assert c.parse_likert("1 — Not at all comfortable", "tech_comfort_label") == 1
    assert math.isnan(c.parse_likert("", "tech_comfort_label"))
    assert math.isnan(c.parse_likert("4 — Unlikely", "rebuy_brand_label"))  # number/label mismatch
    df, _ = run([{"tech_comfort_label": ""}])
    assert math.isnan(df.loc[0, "tech_comfort_1to5"])
    assert not df.loc[0, "eligible_likert"] and df.loc[0, "eligible_demographic"]


def test_ids_unique_and_stable():
    df, log = run([{}, {}, {}])
    assert list(df["respondent_id"]) == ["HS001", "HS002", "HS003"]
    assert list(log["source_excel_row"]) == [2, 3, 4]
