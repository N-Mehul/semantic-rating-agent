"""Step 4: survey summaries and evidence-based personas from the 200-person primary cohort.

Inputs : data/human_survey_cleaned.csv  (read with keep_default_na=False)
Outputs: data/personas/*.csv|json and docs/project_status/step4_personas_report.md

Personas are rule-based segments over NON-OUTCOME variables (price paid, usage hours).
Rebuy intention (Q21) and value/rebuy text (Q26) are never part of a persona profile; they are
reported in a separate outcome table kept for Step 7 evaluation.
Run from repo root:  python scripts/build_personas.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "human_survey_cleaned.csv"
OUT = ROOT / "data" / "personas"
REPORT = ROOT / "docs" / "project_status" / "step4_personas_report.md"

DEMO_VARS = ["age", "gender", "education", "current_status", "monthly_spending", "area_type", "brand",
             "operating_system", "price_paid", "usage_duration", "phones_owned_5y", "replacement_frequency",
             "upgrade_reason", "daily_usage", "info_sources"]
TEXT = {"text_overall_experience": "Q23", "text_battery_camera": "Q24",
        "text_performance_display": "Q25", "text_value_rebuy": "Q26"}

# Rule-based keyword themes (case-insensitive substring/regex). Interpretive coding, not validated.
THEMES = {
    "battery": r"batter|charg",
    "camera": r"camera|photo|pic",
    "performance_speed": r"perform|speed|fast|smooth|lag|slow|hang|snapdragon|processor|fluid",
    "display": r"display|screen|amoled|refresh|brightness",
    "design_build": r"design|look|build|premium|weight|slim",
    "value_price": r"price|worth|value|money|cost|budget|afford|expens",
    "brand_ecosystem": r"brand|ecosystem|apple|ios|samsung|iphone|service|trust",
    "storage": r"storage|memory|ram",
    "negative_issue": r"slow|lag|heat|hang|problem|issue|poor|bad|drain|not good|low |disappoint|worse",
    "positive_general": r"good|great|excellent|amazing|love|satisf|best|nice|superb|perfect",
}


def read_primary() -> pd.DataFrame:
    df = pd.read_csv(CLEAN, keep_default_na=False, dtype=str)
    df = df[df["in_primary_cohort"] == "True"].copy()
    df["tech_comfort_1to5"] = pd.to_numeric(df["tech_comfort_1to5"])
    df["rebuy_brand_1to5"] = pd.to_numeric(df["rebuy_brand_1to5"])
    df["age"] = pd.to_numeric(df["age"].astype(str).str.extract(r"(\d+)", expand=False))
    return df


def freq(series: pd.Series, var: str, denom: int | None = None) -> pd.DataFrame:
    vc = series.value_counts()
    d = denom if denom is not None else int(vc.sum())
    return pd.DataFrame({"variable": var, "category": vc.index, "n": vc.values,
                         "pct": (vc.values / d * 100).round(1), "denominator": d})


def assign_persona(r: pd.Series) -> str:
    """Hierarchical, mutually exclusive; uses only price paid and daily usage."""
    if r["price_paid"] in ("₹35,001–₹60,000", "More than ₹60,000"):
        return "P1_premium_tier_owner"
    if r["price_paid"] in ("Less than ₹10,000", "Prefer not to say"):
        return "P4_budget_or_undisclosed_price"
    if r["daily_usage"] in ("6–8 hours", "More than 8 hours"):
        return "P2_midrange_heavy_user"
    return "P3_midrange_moderate_user"


def multi_select(series: pd.Series) -> pd.Series:
    return series.str.split(r",\s*").explode().str.strip().replace("", pd.NA).dropna()


def theme_counts(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        usable = df[df[f"{col}_usable"] == "True"][col].str.lower()
        for theme, pat in THEMES.items():
            n = int(usable.str.contains(pat, regex=True).sum())
            rows.append({"field": col, "question": TEXT[col], "theme": theme, "n_mentions": n,
                         "pct_of_usable": round(n / len(usable) * 100, 1), "denominator_usable": len(usable)})
    return pd.DataFrame(rows)


def top(series: pd.Series, k: int = 3) -> list[dict]:
    vc = series.value_counts().head(k)
    return [{"category": c, "n": int(n), "pct": round(n / len(series) * 100, 1)} for c, n in vc.items()]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = read_primary()
    N = len(p)
    p["persona_id"] = p.apply(assign_persona, axis=1)

    # ---- aggregate tables
    tables = [freq(p[v], v) for v in DEMO_VARS if v != "info_sources"]
    tables.append(freq(p["info_sources"], "info_sources"))
    ms = multi_select(p["main_use"])
    tables.append(freq(ms, "main_use (multi-select; pct of respondents)", denom=N))
    tables.append(freq(p["tech_comfort_1to5"].astype(int).astype(str), "tech_comfort_1to5"))
    tables.append(freq(p["rebuy_brand_1to5"].astype(int).astype(str), "rebuy_brand_1to5"))
    demo = pd.concat(tables, ignore_index=True)
    demo.to_csv(OUT / "aggregate_survey_summary.csv", index=False, encoding="utf-8")

    theme_exp = theme_counts(p, ["text_overall_experience", "text_battery_camera", "text_performance_display"])
    theme_val = theme_counts(p, ["text_value_rebuy"])
    pd.concat([theme_exp, theme_val]).to_csv(OUT / "theme_counts.csv", index=False, encoding="utf-8")

    # ---- Q26 stance (explicit yes/no cue at start of answer), for Step 7 context
    q26 = p[p["text_value_rebuy_usable"] == "True"]["text_value_rebuy"].str.strip().str.lower()

    # ---- personas
    profiles, outcome_rows = [], []
    for pid, g in p.groupby("persona_id"):
        usable_exp = {c: g[g[f"{c}_usable"] == "True"] for c in list(TEXT)[:3]}
        th = theme_counts(g, list(TEXT)[:3])
        th = th.groupby("theme").agg(n_mentions=("n_mentions", "sum"), denom=("denominator_usable", "sum"))
        th["pct_of_usable_answers"] = (th["n_mentions"] / th["denom"] * 100).round(1)
        top_themes = th.sort_values("pct_of_usable_answers", ascending=False).head(5)
        profiles.append({
            "persona_id": pid, "n": len(g), "pct_of_primary": round(len(g) / N * 100, 1),
            "defining_rule": {"P1_premium_tier_owner": "price_paid above ₹35,000",
                              "P2_midrange_heavy_user": "price_paid ₹10,000–₹35,000 and daily use ≥ 6 hours",
                              "P3_midrange_moderate_user": "price_paid ₹10,000–₹35,000 and daily use < 6 hours",
                              "P4_budget_or_undisclosed_price": "price_paid below ₹10,000 or 'Prefer not to say'"}[pid],
            "age_range": [int(g.age.min()), int(g.age.max())], "median_age": float(g.age.median()),
            "gender": top(g.gender, 3), "current_status": top(g.current_status, 2),
            "area_type": top(g.area_type, 4), "monthly_spending": top(g.monthly_spending, 3),
            "operating_system": top(g.operating_system, 2), "brand": top(g.brand, 4),
            "usage_duration": top(g.usage_duration, 3), "replacement_frequency": top(g.replacement_frequency, 3),
            "upgrade_reason": top(g.upgrade_reason, 3), "daily_usage": top(g.daily_usage, 3),
            "info_sources": top(g.info_sources, 3), "main_use_top": [
                {"category": c, "n": int(n), "pct_of_respondents": round(n / len(g) * 100, 1)}
                for c, n in multi_select(g.main_use).value_counts().head(5).items()],
            "tech_comfort_mean_1to5": round(float(g.tech_comfort_1to5.mean()), 2),
            "top_experience_themes_from_Q23_Q25": [
                {"theme": t, "pct_of_usable_answers": float(r.pct_of_usable_answers), "mentions": int(r.n_mentions)}
                for t, r in top_themes.iterrows()],
            "text_denominators_usable_Q23_Q25": {k: len(v) for k, v in usable_exp.items()},
            "excluded_from_profile": "Q21 rebuy intention and Q26 text (held out for Step 7 evaluation)",
        })
        dist = g.rebuy_brand_1to5.astype(int).value_counts().reindex(range(1, 6), fill_value=0)
        outcome_rows.append({"persona_id": pid, "n": len(g),
                             **{f"q21_{k}": int(v) for k, v in dist.items()},
                             "q21_mean": round(float(g.rebuy_brand_1to5.mean()), 3),
                             "q26_usable_n": int((g.text_value_rebuy_usable == "True").sum())})
    profiles.sort(key=lambda x: x["persona_id"])
    (OUT / "persona_profiles.json").write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(outcome_rows).sort_values("persona_id").to_csv(OUT / "persona_q21_outcomes_for_step7.csv", index=False)
    p[["respondent_id", "persona_id"]].to_csv(OUT / "persona_assignments.csv", index=False)

    write_report(p, N, demo, theme_exp, theme_val, profiles, outcome_rows)
    print(f"primary N={N}", {x['persona_id']: x['n'] for x in profiles})


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def write_report(p, N, demo, theme_exp, theme_val, profiles, outcome_rows) -> None:
    def block(var, title=None):
        d = demo[demo.variable == var][["category", "n", "pct"]]
        return f"**{title or var}** (n={N})\n\n{md(d)}\n"

    th_piv = pd.concat([theme_exp, theme_val]).pivot(index="theme", columns="question", values="pct_of_usable").reset_index()
    dens = pd.concat([theme_exp, theme_val]).drop_duplicates("question")[["question", "denominator_usable"]]
    pers = ""
    for pr in profiles:
        fmt = lambda k: "; ".join(f"{x['category']} ({x['n']})" for x in pr[k])
        pers += f"""### {pr['persona_id']} (n={pr['n']}, {pr['pct_of_primary']}% of primary cohort)

- **Rule:** {pr['defining_rule']}.
- **Observed:** ages {pr['age_range'][0]}–{pr['age_range'][1]} (median {pr['median_age']}); gender: {fmt('gender')}; OS: {fmt('operating_system')}; brands: {fmt('brand')}.
- **Usage:** daily use: {fmt('daily_usage')}; ownership: {fmt('usage_duration')}; replacement: {fmt('replacement_frequency')}; upgrade reason: {fmt('upgrade_reason')}.
- **Context:** area: {fmt('area_type')}; spending capacity: {fmt('monthly_spending')}; info sources: {fmt('info_sources')}.
- **Tech comfort (Q20, mean 1–5):** {pr['tech_comfort_mean_1to5']}.
- **Most frequent keyword themes in Q23–Q25 answers:** {", ".join(f"{t['theme']} ({t['pct_of_usable_answers']}%)" for t in pr['top_experience_themes_from_Q23_Q25'])} (usable-answer denominators {pr['text_denominators_usable_Q23_Q25']}).

"""
    oc = pd.DataFrame(outcome_rows).sort_values("persona_id")
    REPORT.write_text(f"""# Step 4: Survey Summary and Evidence-Based Personas

Generated by `scripts/build_personas.py`. Primary cohort only (n={N}); source `data/human_survey_cleaned.csv`.
Outputs in `data/personas/`: `aggregate_survey_summary.csv`, `theme_counts.csv`, `persona_profiles.json`, `persona_assignments.csv`, `persona_q21_outcomes_for_step7.csv`.

> Scope warning: the sample is {demo[(demo.variable=='current_status')&(demo.category=='Student')].pct.iloc[0]}% students aged {int(p.age.min())}–{int(p.age.max())}; findings describe this convenience sample, not smartphone owners generally. The questionnaire has demographics, two Likert-type items (Q20 technology comfort, Q21 rebuy-brand likelihood) and four free-text items. It does **not** measure or validate all UTAUT2 constructs.

## 1. Demographics and usage (denominator = {N} unless stated)

{block('gender')}
{block('education')}
{block('current_status')}
{block('area_type')}
{block('monthly_spending')}
{block('operating_system')}
{block('brand')}
{block('price_paid')}
{block('usage_duration', 'Time using current phone')}
{block('replacement_frequency')}
{block('upgrade_reason')}
{block('daily_usage')}
{block('info_sources')}
{block('main_use (multi-select; pct of respondents)', 'Main uses (multi-select; % of respondents)')}
Age: min {int(p.age.min())}, median {p.age.median()}, max {int(p.age.max())}.

## 2. Likert-type items

{block('tech_comfort_1to5', 'Q20 technology comfort (1 = not at all, 5 = extremely)')}
{block('rebuy_brand_1to5', 'Q21 likelihood of buying the same brand again (1 = very unlikely, 5 = very likely)')}
Q21 mean {p.rebuy_brand_1to5.mean():.2f}, SD {p.rebuy_brand_1to5.std():.2f}. Q21 measures brand repurchase, not continued use of the same phone, so it is only a proxy for the Behavioral Intention anchors.

## 3. Recurring themes in usable text answers

Keyword-based coding (regex dictionary in the script); one answer can match several themes; "% of usable answers" uses the usable-answer denominator per question: {", ".join(f"{r.question} n={r.denominator_usable}" for r in dens.itertuples())}. Many answers are one word, so theme percentages are lower bounds on what respondents think. This is interpretive coding, not a validated codebook.

{md(th_piv)}

## 4. Personas

Interpretation (not a survey finding): segments are defined by what people paid and how heavily they use the phone. These two non-outcome variables were chosen by judgement (price and usage intensity differ visibly across the sample and give groups of workable size); I did not test that they separate respondents' language, and Q21 was not used to define them. They are descriptive segments, not statistically derived clusters, and **P4 is small and mixes budget buyers with respondents who declined to give a price**.

{pers}
### Held-out outcome table (not in persona profiles; used only in Step 7)

{md(oc)}

## 5. Limitations
- Personas use survey aggregates; no individual-level persona was copied from a respondent.
- Theme coding comes from my regex dictionary and was not independently coded.
- Because the author of persona profiles had seen overall Q21 results while writing, persona-to-outcome comparisons in Step 7 are not fully blind; this is stated there.
- Segments of very different size (see n above) give noisy per-persona estimates.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
