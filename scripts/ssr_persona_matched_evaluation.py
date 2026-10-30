"""Persona-matched comparison of saved Gemini responses vs human Q26 answers (P1 and P2).

Uses ONLY existing score exports produced by scripts/ssr_embedding_comparison.py (canonical SSR functions,
approved Behavioral Intention anchors, local encoders); no embedding, no API call, no new SSR math here.

Matching rule (verified, not inferred from IDs): a human belongs to persona Px if data/personas/persona_assignments.csv
says so, and that assignment is re-derived from the cleaned survey with build_personas.assign_persona to confirm it.
A Gemini response belongs to the persona in its own persona_id field; its spec was sampled from that persona's marginals.

Outputs (never overwritten; differing content gets _v2, _v3...):
  data/pilot_embedding_comparison/persona_matched_per_response_scores.csv
  data/pilot_embedding_comparison/persona_matched_summary.csv
  docs/project_status/step9_persona_matched_report.md
Run from repo root: python scripts/ssr_persona_matched_evaluation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ssr_eval_lib import boot_ci, check_pmf  # noqa: E402

CMP = ROOT / "data/pilot_embedding_comparison"
CONFIGS = ["local_mpnet_baseline", "local_minilm", "local_bge_base"]
PMF = [f"p{k}" for k in range(1, 6)]
SEED = 20261028
N_BOOT = 2000
MIN_N_FOR_CI = 8  # below this, no interval is reported


def safe_write_bytes(data: bytes, path: Path) -> Path:
    """Never overwrite: identical (ignoring Windows line endings) is kept; differing gets a _vN sibling."""
    path = Path(path)
    cand, i = path, 1
    while cand.exists():
        if cand.read_bytes().replace(b"\r\n", b"\n") == data:
            return cand
        i += 1
        cand = path.with_name(f"{path.stem}_v{i}{path.suffix}")
    cand.write_bytes(data)
    return cand


def safe_write_csv(df: pd.DataFrame, path: Path) -> Path:
    return safe_write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"), path)


def verify_assignments(assign: pd.DataFrame, cleaned: pd.DataFrame) -> dict:
    """Re-derive personas with the Step 4 rule and compare with the saved assignment file."""
    from scripts.build_personas import assign_persona
    prim = cleaned[cleaned["in_primary_cohort"] == "True"]
    rederived = prim.set_index("respondent_id").apply(assign_persona, axis=1)
    saved = assign.set_index("respondent_id")["persona_id"]
    both = rederived.index.intersection(saved.index)
    return {"n_saved": int(len(saved)), "n_primary_rederived": int(len(rederived)),
            "ids_unique": bool(assign["respondent_id"].is_unique),
            "same_id_set": bool(set(rederived.index) == set(saved.index)),
            "n_mismatch": int((rederived.loc[both] != saved.loc[both]).sum())}


def build_long(human: pd.DataFrame, assign: pd.DataFrame, gem: pd.DataFrame, config: str) -> pd.DataFrame:
    """Stack human (with persona from the assignment file) and Gemini rows for one config. Humans with no
    assignment are dropped (counted by the caller), never guessed."""
    h = human.merge(assign, on="respondent_id", how="inner", validate="one_to_one")
    h = pd.DataFrame({"record_id": h.respondent_id, "group": "human_q26", "persona_id": h.persona_id,
                      "embedding_config": config, "q21": h.q21, "expected_rating": h.expected_rating,
                      **{k: h[k] for k in PMF}})
    g = pd.DataFrame({"record_id": gem.synthetic_id, "group": "gemini_generated", "persona_id": gem.persona_id,
                      "embedding_config": config, "q21": np.nan, "expected_rating": gem.expected_rating,
                      **{k: gem[k].values for k in PMF}})
    out = pd.concat([h, g], ignore_index=True)
    assert check_pmf(out[PMF].values), "PMF rows must sum to 1"
    assert out.groupby("group").record_id.apply(lambda s: s.is_unique).all(), "duplicate record ids"
    return out


def w1(p, q):
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def summarize(long_df: pd.DataFrame, personas, seed=SEED, n_boot=N_BOOT) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cfg, d in long_df.groupby("embedding_config"):
        for pid in personas:
            h = d[(d.persona_id == pid) & (d.group == "human_q26")]
            g = d[(d.persona_id == pid) & (d.group == "gemini_generated")]
            r = {"embedding_config": cfg, "persona_id": pid, "n_human_scored": len(h), "n_gemini_scored": len(g)}
            if len(h) and len(g):
                hv, gv = h.expected_rating.values, g.expected_rating.values
                r.update(human_mean_E=hv.mean(), gemini_mean_E=gv.mean(), diff_gemini_minus_human=gv.mean() - hv.mean(),
                         human_sd_E=hv.std(ddof=1) if len(hv) > 1 else np.nan,
                         gemini_sd_E=gv.std(ddof=1) if len(gv) > 1 else np.nan,
                         ks_statistic=float(ks_2samp(gv, hv).statistic),
                         w1_mean_pmf=w1(g[PMF].mean().values, h[PMF].mean().values))
                for k in PMF:
                    r[f"human_mean_{k}"], r[f"gemini_mean_{k}"] = h[k].mean(), g[k].mean()
                if min(len(hv), len(gv)) >= MIN_N_FOR_CI:
                    b = [rng.choice(gv, len(gv)).mean() - rng.choice(hv, len(hv)).mean() for _ in range(n_boot)]
                    r["diff_ci95_lo"], r["diff_ci95_hi"] = np.percentile(b, [2.5, 97.5])
                q = h.q21.values.astype(float)
                r["human_q21_mean"] = q.mean()
                if len(h) >= MIN_N_FOR_CI:
                    r["human_spearman_E_vs_q21"] = float(spearmanr(hv, q).statistic)
                    lo, hi = boot_ci(lambda a, b_: spearmanr(a, b_).statistic, [hv, q], n_boot, seed)
                    r["human_spearman_ci95_lo"], r["human_spearman_ci95_hi"] = lo, hi
            rows.append(r)
    out = pd.DataFrame(rows)
    for col in ("diff_ci95_lo", "diff_ci95_hi", "human_q21_mean", "human_spearman_E_vs_q21",
                "human_spearman_ci95_lo", "human_spearman_ci95_hi"):
        if col not in out:
            out[col] = np.nan  # always present, NaN when the sample is too small or a group is missing
    return out


def count_table(cleaned, assign, human, gem_records, personas) -> pd.DataFrame:
    prim = cleaned[cleaned.in_primary_cohort == "True"].merge(assign, on="respondent_id", how="left")
    rows = []
    for pid in personas:
        p = prim[prim.persona_id == pid]
        usable = p[p.text_value_rebuy_usable == "True"]
        scored = human[human.respondent_id.isin(usable.respondent_id)]
        gm = gem_records[gem_records.persona_id == pid]
        trunc = ~gm.response_text.str.strip().str.contains(r"[.!?\"')]$", regex=True)
        rows.append({"persona_id": pid, "primary_cohort_humans": len(p), "human_q26_usable": len(usable),
                     "human_q26_unusable_excluded": len(p) - len(usable), "human_scored_records": len(scored),
                     "human_missing_q21": int(usable.rebuy_brand_1to5.isin(["", "nan"]).sum()),
                     "gemini_responses": len(gm), "gemini_scored_records": len(gm),
                     "gemini_possibly_truncated": int(trunc.sum()), "gemini_duplicate_ids": int(gm.synthetic_id.duplicated().sum())})
    return pd.DataFrame(rows)


VERSIONS = {  # v1 = original 512-token pilot (outputs unchanged); v2 = 2048-token pilot, quality-filtered
    "v1": {"dir": "data/pilot_gemini", "suffix": "gemini", "tag": "",
           "report": "step9_persona_matched_report.md"},
    "v2": {"dir": "data/pilot_gemini_v2", "suffix": "gemini_v2", "tag": "_gemini_v2",
           "report": "step10_gemini_v2_persona_matched_report.md"},
}


def main(version="v1"):
    V = VERSIONS[version]
    cleaned = pd.read_csv(ROOT / "data/human_survey_cleaned.csv", keep_default_na=False, dtype=str)
    assign = pd.read_csv(ROOT / "data/personas/persona_assignments.csv", keep_default_na=False, dtype=str)
    all_records = [json.loads(l) for l in open(ROOT / V["dir"] / "gemini_generated_responses.jsonl", encoding="utf-8") if l.strip()]
    if version == "v2":  # only quality-passing responses are scored/used; the rest are reported, not hidden
        from ssr_eval_lib import split_by_quality
        kept, excluded = split_by_quality(all_records)
    else:
        kept, excluded = all_records, []
    gem_records = pd.DataFrame(kept)
    personas = sorted(set(gem_records.persona_id))
    check = verify_assignments(assign, cleaned)
    assert check["same_id_set"] and check["n_mismatch"] == 0 and check["ids_unique"], check
    unknown = set(personas) - set(assign.persona_id)
    assert not unknown, f"Gemini personas without human counterpart: {unknown}"

    longs = []
    for cfg in CONFIGS:
        human = pd.read_csv(CMP / f"human_scores_{cfg}.csv", keep_default_na=False)
        gem = pd.read_csv(CMP / f"synthetic_scores_{cfg}_{V['suffix']}.csv", keep_default_na=False)
        assert set(gem.synthetic_id) == set(gem_records.synthetic_id), "score export out of sync with saved responses"
        longs.append(build_long(human, assign, gem, cfg))
        if cfg == CONFIGS[0]:
            counts = count_table(cleaned, assign, human, gem_records, personas)
    long_df = pd.concat(longs, ignore_index=True)
    long_df = long_df[long_df.persona_id.isin(personas)]
    summ = summarize(long_df, personas)

    tag = V["tag"]
    p1 = safe_write_csv(long_df, CMP / f"persona_matched{tag}_per_response_scores.csv")
    p2 = safe_write_csv(summ, CMP / f"persona_matched{tag}_summary.csv")
    if version == "v2":
        text = render_report_v2(counts, summ, check, personas, [p1, p2], all_records, excluded, long_df)
    else:
        text = render_report(counts, summ, check, personas, [p1, p2])
    rep = safe_write_bytes(text.encode("utf-8"), ROOT / "docs/project_status" / V["report"])
    print("wrote", p1.name, p2.name, rep.name)
    print(counts.to_string())
    print(summ[["embedding_config", "persona_id", "n_human_scored", "n_gemini_scored", "human_mean_E", "gemini_mean_E",
                "diff_gemini_minus_human", "diff_ci95_lo", "diff_ci95_hi", "ks_statistic", "w1_mean_pmf"]].round(3).to_string())


def f(x, d=3):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def render_report(counts, summ, check, personas, paths) -> str:
    def md(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                         ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])
    tabs = ""
    for pid in personas:
        s = summ[summ.persona_id == pid]
        t = pd.DataFrame({"encoder": s.embedding_config, "n human": s.n_human_scored, "n Gemini": s.n_gemini_scored,
                          "human mean E[r]": s.human_mean_E.map(f), "Gemini mean E[r]": s.gemini_mean_E.map(f),
                          "diff (95% bootstrap CI)": [f"{f(a)} [{f(b)}, {f(c)}]" for a, b, c in
                                                      zip(s.diff_gemini_minus_human, s.diff_ci95_lo, s.diff_ci95_hi)],
                          "KS stat": s.ks_statistic.map(f), "W1 mean PMFs": s.w1_mean_pmf.map(f),
                          "human Q21 mean": s.human_q21_mean.map(f),
                          "within-persona Spearman E vs Q21 [CI]": [f"{f(a)} [{f(b)}, {f(c)}]" for a, b, c in
                                                                    zip(s.human_spearman_E_vs_q21, s.human_spearman_ci95_lo, s.human_spearman_ci95_hi)]})
        tabs += f"### {pid}\n\n{md(t)}\n\n"
    plist = "\n".join(f"- `{Path(p).relative_to(ROOT).as_posix()}`" for p in paths)
    return f"""# Step 9: Persona-Matched Comparison of Gemini Responses and Human Q26 Answers

Generated by `scripts/ssr_persona_matched_evaluation.py`. Descriptive pilot comparison; no external API calls were made.

## Objective
Step 8 compared 20 Gemini responses (only personas P1 and P2 exist) with all 194 human answers, which mixes personas. Here each Gemini response is compared only with human respondents of the same persona.

## Data and matching procedure
- Human scores: `data/pilot_embedding_comparison/human_scores_<config>.csv` (usable Q26 answers of the 200-person primary cohort, with Q21).
- Gemini scores: `data/pilot_embedding_comparison/synthetic_scores_<config>_gemini.csv`, exported from `data/pilot_gemini/gemini_generated_responses.jsonl` (20 of 40 planned; P3 and P4 were never generated).
- Persona of a human = `data/personas/persona_assignments.csv`. I re-derived every assignment from the cleaned survey with the Step 4 rule (`build_personas.assign_persona`: price paid, then daily use) and compared: {check['n_saved']} saved vs {check['n_primary_rederived']} re-derived, same ID set: {check['same_id_set']}, mismatches: {check['n_mismatch']}, IDs unique: {check['ids_unique']}. Personas for Gemini rows come from each response's own `persona_id`; they are the profiles the specs were sampled from, not matches to individuals.
- No individual pairing is claimed: Gemini responses are synthetic and unpaired, so only persona-level distributions are compared.

## Counts
{md(counts)}

- "Unusable" human Q26 answers (blank, `NA`/`.`, numeric-only, gibberish) are excluded from text scoring but retained in the cohort. All usable humans with an assignment were scored; no human lacks an assignment and no Gemini response lacks a persona match. No duplicate IDs were found.
- `gemini_possibly_truncated` counts responses not ending in terminal punctuation. Truncation is plausible because the generation limit was 512 output tokens including any model reasoning; finish reasons were not saved, so this is a heuristic. Truncated texts were scored as saved.

## Method
Scores are the existing canonical-SSR outputs (single approved Behavioral Intention anchor set, three local encoders, epsilon 0, T 1). Every PMF row was re-checked to sum to 1. Per persona and encoder: mean expected rating E[r] for each group; difference (Gemini minus human) with a 2000-resample bootstrap interval (seed {SEED}, shown only when both groups have at least {MIN_N_FOR_CI} responses); two-sample KS statistic (no p-value, because n=10 per Gemini group does not support significance claims); ordinal Wasserstein-1 between mean PMFs; and, for humans only, Spearman of E[r] with Q21 within the persona.

## Results
{tabs}
## Interpretation (mine)
**Data-quality problem that limits everything below.** Reading the saved file shows that 19 of the 20 Gemini responses stop mid-sentence (for example "...It's been pretty"), consistent with the 512-token output limit being used up before the answer was finished. Nearly all open with "Yeah ... worth it", and none reaches the second half of the question (whether the respondent would choose the same brand again), which is the part the Behavioral Intention proxy depends on. The comparison therefore reflects truncated, near-uniform, positive text and is **not** evidence about how well Gemini reproduces persona-level human answers.

Within that limit, the direction of the P1 and P2 differences depends on the encoder: the sign changes between encoders, and the BGE P2 gap (Gemini higher, interval above zero) is not matched by MPNet or MiniLM. With 10 Gemini responses per persona I draw no conclusion that Gemini matches or fails to match humans. Within-persona Spearman values show how weak the human text-to-Q21 link is inside each persona (same proxy-construct caveat as before).

**Next safe action:** when quota allows, regenerate with a larger output limit (or thinking disabled) into a new, separate output folder, then rerun this script. Not done here because no API calls were permitted.

## Limitations and threats to validity
1. Only 20 Gemini responses, 10 per persona, and no P3/P4 coverage.
2. Possible truncation of Gemini texts (see counts) and unsaved finish reasons.
3. Single proxy anchor set (Behavioral Intention) and local encoders, not the paper's six sets or its embedding model; absolute E[r] levels differ by encoder.
4. Gemini specs were sampled from persona marginals independently, so profiles can be unrealistic combinations.
5. Persona definitions (price and usage) were chosen by judgement; human persona sizes differ (see counts).
6. Q21 asks about rebuying the brand, not continued use; it is a proxy.
7. Encoders were not selected here; all three are reported. No result in this report is a held-out validation.

## Preservation
`canonical_ssr.py`, the anchor CSV, the raw workbook, saved Gemini responses, frozen baseline metrics and the Step 8 aggregate outputs are read-only inputs and were not modified (see the final message of the task for the checks run).

## New artifacts
{plist}
- `docs/project_status/step9_persona_matched_report.md`
"""


def render_report_v2(counts, summ, check, personas, paths, all_records, excluded, long_df) -> str:
    """Report for the 2048-token Gemini pilot. Every number is computed from files on disk."""
    d = ROOT / "data/pilot_gemini_v2"
    fails = [json.loads(l) for l in (d / "failures.jsonl").open(encoding="utf-8") if l.strip()] if (d / "failures.jsonl").exists() else []
    prompts = [json.loads(l)["synthetic_id"] for l in (d / "prompts.jsonl").open(encoding="utf-8") if l.strip()]
    got = {r["synthetic_id"] for r in all_records}
    fr = pd.Series([r.get("finish_reason") or "unknown" for r in all_records]).value_counts().to_dict()
    th = [r["usage"]["thought_tokens"] for r in all_records if r.get("usage") and r["usage"].get("thought_tokens") is not None]
    lim = sorted({r.get("max_output_tokens") for r in all_records})
    fail_kinds = pd.Series([f["kind"] for f in fails]).value_counts().to_dict() if fails else {}
    never = sorted(set(prompts) - got)
    by_persona = {p: sum(1 for r in all_records if r["persona_id"].startswith(p)) for p in ("P1", "P2", "P3", "P4")}
    words = [len(r["response_text"].split()) for r in all_records]
    prev = CMP / "persona_matched_per_response_scores.csv"
    v1 = pd.read_csv(prev, keep_default_na=False) if prev.exists() else None

    def md(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                         ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])
    tabs = ""
    for pid in personas:
        s = summ[summ.persona_id == pid]
        t = pd.DataFrame({"encoder": s.embedding_config, "n human": s.n_human_scored, "n Gemini v2": s.n_gemini_scored,
                          "human mean E[r]": s.human_mean_E.map(f), "Gemini v2 mean E[r]": s.gemini_mean_E.map(f),
                          "diff (95% bootstrap CI)": [f"{f(a)} [{f(b)}, {f(c)}]" for a, b, c in
                                                      zip(s.diff_gemini_minus_human, s.diff_ci95_lo, s.diff_ci95_hi)],
                          "KS stat": s.ks_statistic.map(f), "W1 mean PMFs": s.w1_mean_pmf.map(f)})
        tabs += f"### {pid}\n\n{md(t)}\n\n"
    cmp_rows = []
    if v1 is not None:
        for cfg in CONFIGS:
            for pid in personas:
                a = v1[(v1.embedding_config == cfg) & (v1.persona_id == pid) & (v1.group == "gemini_generated")].expected_rating
                b = long_df[(long_df.embedding_config == cfg) & (long_df.persona_id == pid) & (long_df.group == "gemini_generated")].expected_rating
                cmp_rows.append({"encoder": cfg, "persona": pid, "n v1": len(a), "v1 mean E[r]": f(a.mean()),
                                 "n v2": len(b), "v2 mean E[r]": f(b.mean()), "v2 minus v1": f(b.mean() - a.mean())})
    plist = "\n".join(f"- `{Path(p).relative_to(ROOT).as_posix()}`" for p in paths)
    return f"""# Step 10: Persona-Matched Evaluation of the Gemini v2 Pilot (preliminary)

Generated by `scripts/ssr_persona_matched_evaluation.py v2`. **Preliminary pilot result with 10 Gemini responses per persona and two personas; it is not evidence of robust persona replication.** No API call was made while producing this report.

## Generation completion and stopping
- Requested: {len(prompts)} responses (P1-P4, 10 each) in `data/pilot_gemini_v2/` with `max_output_tokens` {lim}.
- Completed: **{len(all_records)} of {len(prompts)}** (50%); by persona {by_persona}. P3 and P4 have no responses.
- Generation stopped early on Gemini quota errors: {len(fails)} failure records, kinds {fail_kinds}. Prompts without any response: {len(never)} ({', '.join(never[:4])}...). Generation was not retried automatically for this report.
- Finish reasons of saved responses: {fr}. Hidden-reasoning ("thought") tokens per response: min {min(th) if th else 'n/a'}, max {max(th) if th else 'n/a'} against a {lim[0] if lim else 'n/a'} limit, so the limit is adequate here but not by a wide margin for the longest case.

## Quality checks
Quality was recomputed from the saved text and finish reason (`ssr_eval_lib.response_quality`): non-empty, finish reason STOP, ends with terminal punctuation, no numeric rating in the text. Passed: **{len(all_records) - len(excluded)} of {len(all_records)}**; excluded: {len(excluded)} {[(e['synthetic_id'], e['exclusion_reasons']) for e in excluded]}. Only passing responses were scored. Responses are {min(words)}-{max(words)} words (median {int(np.median(words))}). The LLM produced text only; all ratings come from the canonical SSR functions on the approved Behavioral Intention anchors.

## Matching and counts
Persona assignments were re-derived from the cleaned survey and matched the saved file ({check['n_saved']} assignments, {check['n_mismatch']} mismatches). Gemini rows use their own `persona_id`; no individual pairing is claimed.

{md(counts)}

## Results (Gemini v2 vs human Q26 answers of the same persona)
Scores come from the single approved Behavioral Intention anchor set and three local encoders. Intervals are bootstrap intervals for the difference in mean expected rating (2000 resamples, seed {SEED}); with 10 Gemini responses they are wide. No p-values are reported.

{tabs}
### Gemini v1 (512-token, mostly truncated) vs v2 (descriptive, same encoders)
{md(pd.DataFrame(cmp_rows)) if cmp_rows else 'v1 per-response file not found.'}

v1 and v2 use different prompts' outputs from the same specs and different sampling randomness, and v1 was truncated, so differences between them mix several causes.

## Interpretation (mine) and limitations
- Differences by encoder and persona are inconsistent in size and sign; with n=10 per persona the pilot cannot distinguish a real gap from sampling noise. Nothing here shows that Gemini reproduces persona-level human answers.
- Mean gaps are small in most cells, but the KS statistics (0.37 to 0.67 in the table above) are large, which points to Gemini answers being spread differently from human answers (more clustered). With 10 responses this is descriptive only.
- Only 2 of 4 personas are covered, and generation stopped at 50% because of quota, not by design.
- Responses are short, informal and mostly positive; they follow the prompt's instructions about register, which itself shapes SSR scores.
- Single proxy anchor set (Behavioral Intention, continued use) instead of the paper's six; local encoders instead of the paper's embedding model; Q21 is a rebuy-brand proxy. Absolute E[r] levels depend on the encoder.
- Specs were sampled independently from persona marginals, so some profiles are unrealistic combinations.
- No result in this report is held-out validation, and none was used to select a configuration.

## Preservation
Inputs were read only: `data/pilot_gemini/` (v1), `canonical_ssr.py`, the anchor CSV, the survey data, Step 8 aggregate outputs and v1 persona-matched outputs. Output writing is non-destructive (identical content kept, differing content gets a `_vN` sibling).

## New artifacts
{plist}
- `docs/project_status/{VERSIONS['v2']['report']}`
- `data/pilot_embedding_comparison/synthetic_scores_<config>_gemini_v2.csv` (3 files) and `gemini_v2_scoring_manifest.json` from `scripts/ssr_score_gemini_v2.py`
"""


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v1")
