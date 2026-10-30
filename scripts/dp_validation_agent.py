"""Validation Agent (stage 5 of the Digital Persona pipeline): compare the synthetic dataset with human answers.

Checks (each reports its sample sizes and says when the data cannot support it):
  1. coverage and integrity of the synthetic dataset (missing personas are reported, never filled),
  2. persona-matched distribution comparison per source x persona x encoder (mean difference with bootstrap CI,
     KS statistic, ordinal Wasserstein-1 between mean PMFs; no equivalence claim and no p-values),
  3. persona consistency of the generated text with the persona's sampled attributes (brand / operating-system mentions),
  4. repeated-generation stability (only if the same spec was generated more than once by the same model),
  5. encoder agreement on the synthetic responses (measurement stability),
  6. human reference: SSR on human Q26 text vs Q21 with mean / majority baselines (Step 8 metrics, read-only).

Reads files only; no API calls.  Run:  python scripts/dp_validation_agent.py
Outputs (non-destructive): data/validation/*.csv|json and docs/project_status/validation_agent_report.md
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ssr_persona_matched_evaluation import safe_write_bytes  # noqa: E402

CMP = ROOT / "data/pilot_embedding_comparison"
DATASET = ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv"
COVERAGE = ROOT / "data/synthetic_dataset/coverage_by_persona_source.csv"
VAL = ROOT / "data/validation"
REPORT = ROOT / "docs/project_status/validation_agent_report.md"
ENCODERS = ["local_mpnet_baseline", "local_minilm", "local_bge_base"]
PMF = [f"p{k}" for k in range(1, 6)]
SEED = 20261028
N_BOOT = 2000
MIN_N = 8  # smallest group size for which an interval is reported

BRANDS = {"Apple": r"\b(apple|iphone|ios)\b", "Samsung": r"\bsamsung\b", "Vivo": r"\bvivo\b", "OnePlus": r"\boneplus\b|\bone plus\b",
          "Realme": r"\brealme\b", "Oppo": r"\boppo\b", "Xiaomi": r"\b(xiaomi|redmi|mi)\b", "Motorola": r"\bmotorola\b|\bmoto\b",
          "Google": r"\b(google|pixel)\b", "Nothing": r"\bnothing phone\b"}
ANDROID = re.compile(r"\bandroid\b", re.I)
IOS = re.compile(r"\b(ios|iphone)\b", re.I)


def w1(p, q) -> float:
    return float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())


def boot_diff(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ---- 1. coverage / integrity
def integrity_checks(df: pd.DataFrame, cov: pd.DataFrame) -> list[dict]:
    out = []
    for key, g in df.groupby("source_key"):
        s = g[g.status == "scored"]
        pmf_ok = True
        for cfg in ENCODERS:
            cols = [f"{cfg}__{k}" for k in PMF]
            if s.empty or cols[0] not in s:
                continue
            P = s[cols].values
            pmf_ok &= bool(np.allclose(P.sum(1), 1.0, atol=1e-12) and (P >= -1e-12).all()
                           and np.allclose(s[f"{cfg}__E"], P @ np.arange(1, 6)))
        out.append({"source_key": key, "rows": len(g), "scored": len(s), "excluded_quality": int((g.status == "excluded_quality").sum()),
                    "duplicate_ids": int(g.synthetic_id.duplicated().sum()), "rows_without_spec": int((~g.spec_found.astype(bool)).sum()),
                    "pmf_valid_and_E_consistent": pmf_ok})
    return out


# ---- 2. persona-matched distribution comparison
def distribution_comparison(df: pd.DataFrame, human: dict[str, pd.DataFrame], seed=SEED, n_boot=N_BOOT) -> pd.DataFrame:
    """human[cfg] has columns persona_id, expected_rating, p1..p5, q21."""
    rows = []
    comparable = df[df.status == "scored"]
    if "human_reference" in comparable:   # only sources whose question has a comparable human answer (Q26 text) are compared
        comparable = comparable[comparable.human_reference == "Q26_Q21_proxy"]
    for key, g in comparable.groupby("source_key"):
        for pid, gp in g.groupby("persona_id"):
            for cfg in ENCODERS:
                h = human[cfg][human[cfg].persona_id == pid]
                sv, hv = gp[f"{cfg}__E"].values, h.expected_rating.values
                r = {"source_key": key, "persona_id": pid, "encoder": cfg, "n_synthetic": len(sv), "n_human": len(hv)}
                if len(sv) and len(hv):
                    r.update(synthetic_mean_E=sv.mean(), human_mean_E=hv.mean(), diff_synth_minus_human=sv.mean() - hv.mean(),
                             synthetic_sd_E=sv.std(ddof=1) if len(sv) > 1 else np.nan, human_sd_E=hv.std(ddof=1) if len(hv) > 1 else np.nan,
                             ks_statistic=float(ks_2samp(sv, hv).statistic),
                             w1_mean_pmf=w1(gp[[f"{cfg}__{k}" for k in PMF]].mean().values, h[PMF].mean().values),
                             mean_pmf_sd=float(gp[f"{cfg}__sd"].mean()), mean_pmf_entropy=float(gp[f"{cfg}__entropy"].mean()))
                    if min(len(sv), len(hv)) >= MIN_N:
                        r["diff_ci95_lo"], r["diff_ci95_hi"] = boot_diff(sv, hv, n_boot, seed)
                rows.append(r)
    out = pd.DataFrame(rows)
    for c in ("diff_ci95_lo", "diff_ci95_hi"):
        if c not in out:
            out[c] = np.nan
    out["interval_includes_zero"] = ~((out.diff_ci95_lo > 0) | (out.diff_ci95_hi < 0)) & out.diff_ci95_lo.notna()
    out["equivalence_claimed"] = False  # an interval containing zero is not evidence of equivalence
    return out


def not_comparable_sources(df: pd.DataFrame) -> list[dict]:
    """Sources with scored rows but no comparable human answer in the collected survey (reported, never compared)."""
    if "human_reference" not in df:
        return []
    g = df[(df.status == "scored") & (df.human_reference != "Q26_Q21_proxy")]
    return [{"source_key": k, "task_id": x.task_id.iloc[0], "scored_rows": len(x),
             "reason": "the collected survey has no human answers to this question; distribution comparison skipped"} for k, x in g.groupby("source_key")]


def persona_ordering(df: pd.DataFrame, human: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Does the ordering of personas by mean synthetic E[r] match the ordering by mean human-text E[r] and by human Q21?
    Descriptive only: with k personas there are only k points (k=2 gives a single direction)."""
    rows = []
    comparable = df[df.status == "scored"]
    if "human_reference" in comparable:
        comparable = comparable[comparable.human_reference == "Q26_Q21_proxy"]
    for key, g in comparable.groupby("source_key"):
        for cfg in ENCODERS:
            sm = g.groupby("persona_id")[f"{cfg}__E"].mean()
            hm = human[cfg].groupby("persona_id").expected_rating.mean()
            hq = human[cfg].groupby("persona_id").q21.mean()
            pids = sorted(sm.index)
            if len(pids) < 2:
                continue
            a, b = pids[0], pids[1]
            sgn = lambda x: int(np.sign(x))
            rows.append({"source_key": key, "encoder": cfg, "n_personas": len(pids), "persona_a": a, "persona_b": b,
                         "synthetic_E_a_minus_b": float(sm[a] - sm[b]), "human_text_E_a_minus_b": float(hm[a] - hm[b]),
                         "human_q21_a_minus_b": float(hq[a] - hq[b]),
                         "same_direction_as_human_text": sgn(sm[a] - sm[b]) == sgn(hm[a] - hm[b]),
                         "same_direction_as_human_q21": sgn(sm[a] - sm[b]) == sgn(hq[a] - hq[b]),
                         "note": "two personas only: one direction, descriptive"})
    return pd.DataFrame(rows)


# ---- 3. persona consistency
def persona_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in df.itertuples():
        t = str(r.response_text)
        spec_brand = str(r.brand)
        key = "Apple" if spec_brand.startswith("Apple") else spec_brand
        named = [b for b, pat in BRANDS.items() if re.search(pat, t, re.I)]
        spec_apple = str(r.operating_system) == "iOS"
        rows.append({"source_key": r.source_key, "synthetic_id": r.synthetic_id, "persona_id": r.persona_id, "spec_brand": spec_brand,
                     "brands_named_in_text": ";".join(named),
                     "mentions_spec_brand": (key in named) if key in BRANDS else None,
                     "mentions_other_brand": any(b != key for b in named),
                     "os_contradiction": bool((spec_apple and ANDROID.search(t)) or ((not spec_apple) and IOS.search(t))),
                     "informational_note": "mentioning another brand can be a legitimate switching intention; only os_contradiction is a hard inconsistency"})
    return pd.DataFrame(rows)


# ---- 4. repeated-generation stability
def generation_stability(df: pd.DataFrame, cfg: str = "local_bge_base") -> dict:
    """A spec counts as repeated only if the same synthetic_id was scored in >= 2 sources with the SAME model."""
    s = df[df.status == "scored"]
    g = s.groupby(["synthetic_id", "model_requested"]).source_key.nunique()
    rep = g[g >= 2]
    if rep.empty:
        return {"status": "not_supported_by_data", "reason": "each spec was generated once per model; no repeated generations exist",
                "how_to_enable": "generate the same specs again into a new output folder with the same model and register it as a source"}
    keep = set(rep.index)
    sub = s[[(a, b) in keep for a, b in zip(s.synthetic_id, s.model_requested)]]
    within = sub.groupby(["synthetic_id", "model_requested"])[f"{cfg}__E"].std(ddof=1).mean()
    return {"status": "computed", "n_repeated_specs": len(rep), "mean_within_spec_sd_E": float(within), "encoder": cfg}


# ---- 5. encoder agreement
def encoder_agreement(df: pd.DataFrame) -> list[dict]:
    out = []
    for key, g in df[df.status == "scored"].groupby("source_key"):
        for i, a in enumerate(ENCODERS):
            for b in ENCODERS[i + 1:]:
                if len(g) >= MIN_N:
                    out.append({"source_key": key, "encoder_a": a, "encoder_b": b, "n": len(g),
                                "spearman_E": float(spearmanr(g[f"{a}__E"], g[f"{b}__E"]).statistic)})
    return out


def load_human(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    a = pd.read_csv(root / "data/personas/persona_assignments.csv")
    out = {}
    for cfg in ENCODERS:
        h = pd.read_csv(root / f"data/pilot_embedding_comparison/human_scores_{cfg}.csv", keep_default_na=False)
        out[cfg] = h.merge(a, on="respondent_id", how="inner", validate="one_to_one")
    return out


def human_reference(root: Path = ROOT) -> list[dict]:
    m = json.load(open(root / "data/pilot_embedding_comparison/comparison_metrics.json"))
    rows = []
    for cfg in ENCODERS:
        for split in ("full", "test"):
            x = m["results"][cfg][split]
            rows.append({"encoder": cfg, "split": split, "n": x["n"], "spearman": x["spearman"], "ci_lo": x["spearman_ci95"][0],
                         "ci_hi": x["spearman_ci95"][1], "mae": x["mae"], "mae_mean_baseline": x["mae_baseline_mean"],
                         "accuracy": x["accuracy_exact"], "accuracy_majority_baseline": x["accuracy_baseline_majority"]})
    return rows


def run(root: Path = ROOT, write: bool = True) -> dict:
    if not DATASET.exists() and root == ROOT:
        raise FileNotFoundError("Synthetic dataset missing: run  python scripts/dp_synthetic_dataset.py  first.")
    df = pd.read_csv(root / "data/synthetic_dataset/synthetic_survey_dataset.csv", keep_default_na=False)
    for c in df.columns:
        if c.endswith(tuple(f"__{s}" for s in ["E", "sd", "entropy", "argmax"] + PMF)) or c == "encoder_spread_sd":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    cov = pd.read_csv(root / "data/synthetic_dataset/coverage_by_persona_source.csv")
    human = load_human(root)
    res = {"integrity": integrity_checks(df, cov), "coverage": cov, "distribution": distribution_comparison(df, human),
           "consistency": persona_consistency(df), "not_comparable": not_comparable_sources(df),
           "ordering": persona_ordering(df, human), "stability": generation_stability(df), "encoder_agreement": encoder_agreement(df),
           "human_reference": human_reference(root)}
    if write:
        VAL.mkdir(parents=True, exist_ok=True)
        res["written"] = [
            safe_write_bytes(res["distribution"].to_csv(index=False, lineterminator="\n").encode("utf-8"), VAL / "distribution_comparison.csv").name,
            safe_write_bytes(res["consistency"].to_csv(index=False, lineterminator="\n").encode("utf-8"), VAL / "persona_consistency.csv").name,
            safe_write_bytes(res["ordering"].to_csv(index=False, lineterminator="\n").encode("utf-8"), VAL / "persona_ordering.csv").name,
            safe_write_bytes(json.dumps({"integrity": res["integrity"], "stability": res["stability"], "not_comparable": res["not_comparable"],
                                         "encoder_agreement": res["encoder_agreement"], "human_reference": res["human_reference"]},
                                        indent=2, default=str).encode("utf-8"), VAL / "validation_summary.json").name]
        res["report"] = safe_write_bytes(render_report(res, df).encode("utf-8"), REPORT).name
    return res


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] +
                     ["| " + " | ".join("n/a" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |"
                      for r in df.itertuples(index=False)])


def render_report(res: dict, df: pd.DataFrame) -> str:
    d = res["distribution"]
    cons = res["consistency"]
    cons_s = cons.groupby(["source_key", "persona_id"]).agg(n=("synthetic_id", "size"), os_contradictions=("os_contradiction", "sum"),
                                                          mentions_spec_brand=("mentions_spec_brand", lambda s: int(s.eq(True).sum())),
                                                          mentions_other_brand=("mentions_other_brand", "sum")).reset_index()
    ex0 = int((~d.interval_includes_zero).sum()) if len(d) else 0
    st = res["stability"]
    dist_t = d[["source_key", "persona_id", "encoder", "n_synthetic", "n_human", "synthetic_mean_E", "human_mean_E",
                "diff_synth_minus_human", "diff_ci95_lo", "diff_ci95_hi", "ks_statistic", "w1_mean_pmf", "mean_pmf_sd", "mean_pmf_entropy"]]
    href = pd.DataFrame(res["human_reference"])
    ea = pd.DataFrame(res["encoder_agreement"])
    od = res["ordering"]
    ord_note = ""
    if len(od):
        ord_note = (f"Interpretation: the synthetic ordering matches the human-text ordering for {int(od.same_direction_as_human_text.sum())} of {len(od)} encoders "
                    f"and the human Q21 ordering for {int(od.same_direction_as_human_q21.sum())} of {len(od)}; but the synthetic gaps between personas "
                    f"(|difference| at most {od.synthetic_E_a_minus_b.abs().max():.2f}) are much smaller than the human Q21 gap "
                    f"({od.human_q21_a_minus_b.abs().max():.2f}), so agreement in direction is weak evidence. Two personas give one direction only.")
    ea_note = ""
    if len(ea):
        ea_note = (f"Interpretation: pairwise Spearman between encoders ranges {ea.spearman_E.min():.2f} to {ea.spearman_E.max():.2f}. "
                   + ("This is low: the ordering of individual synthetic responses depends strongly on which encoder is used, so per-response scores "
                      "should not be interpreted individually; only group-level summaries are discussed, and even those differ by encoder."
                      if ea.spearman_E.max() < 0.5 else "Encoders broadly agree on the ordering of responses."))
    return f"""# Validation Agent Report (preliminary)

Generated by `scripts/dp_validation_agent.py` from `data/synthetic_dataset/` and the human score files. No API call was made. **Results are preliminary: they do not establish robust persona replication or statistical equivalence between synthetic and human answers.**

## 1. Coverage and integrity
Coverage (personas without responses are `not_generated`; nothing was imputed):

{md(res['coverage'])}

Integrity of scored rows:

{md(pd.DataFrame(res['integrity']))}

## 2. Persona-matched distribution comparison (synthetic vs human Q26 scores, same persona)
Difference = synthetic minus human mean expected rating E[r]. Bootstrap intervals (2000 resamples, seed {SEED}) appear only when both groups have at least {MIN_N} responses. No p-values and no equivalence tests are reported: an interval that includes zero does not show equivalence. {ex0} of {len(d)} comparisons have an interval that includes zero; the rest have no interval or exclude zero.

{md(dist_t)}

Sources without a comparable human answer: {json.dumps(res['not_comparable']) if res['not_comparable'] else 'none'}.

### Persona ordering (do synthetic persona means order like human ones?)
{md(res['ordering']) if len(res['ordering']) else 'Fewer than two personas have synthetic responses.'}

{ord_note}

## 3. Persona consistency of generated text
Rule-based checks of each response against the sampled spec (brand and operating-system mentions). `os_contradiction` is a hard inconsistency (spec iOS but text says Android, or spec Android but text says iPhone/iOS). Naming another brand can be a legitimate switching intention and is informational only.

{md(cons_s)}

## 4. Repeated-generation stability
Status: **{st['status']}**. {st.get('reason', '')} {('To enable: ' + st['how_to_enable']) if 'how_to_enable' in st else ''}

## 5. Encoder agreement on the synthetic responses (measurement stability)
{md(ea) if len(ea) else 'Fewer than %d scored responses per source.' % MIN_N}

{ea_note}

## 6. Human reference: SSR on human Q26 text vs the same respondent's Q21
Read-only copy of the Step 8 results with the mean-predictor and majority-class baselines (Q21 is a proxy for the Behavioral Intention anchors).

{md(href)}

## 7. Limitations
- Only sources registered in the dataset are evaluated; synthetic responses currently exist only for personas P1 and P2 (Gemini v2: 20 of 40 planned). P3 and P4 are untested. OpenAI has zero responses.
- Single adapted Behavioral Intention anchor set and three local encoders instead of the paper's six sets and embedding model.
- Small samples (10 per persona), one generation per spec, independently sampled profiles.
- The consistency checks are keyword rules, not semantic judgement.
- Frozen baselines, canonical SSR code, anchors and survey data are read-only inputs.
"""


def main() -> int:
    try:
        res = run()
    except FileNotFoundError as e:
        print(e)
        return 2
    print("written", res["written"], res["report"])
    print("stability:", res["stability"]["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
