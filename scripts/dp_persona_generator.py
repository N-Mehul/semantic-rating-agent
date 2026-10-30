"""Digital Persona Generator (stage 1 of the Digital Persona pipeline).

Turns the approved persona assignments + the cleaned human survey into
  * persona cards   (evidence summary per persona, with provenance hashes), and
  * respondent specs (synthetic respondents sampled from each persona's observed marginals, seeded).

Sampling reproduces data/pilot/synthetic_respondent_specs.jsonl exactly (same seed, order and draws), so responses already
generated for those specs stay linked to them. Outcome variables (Q21, Q26 text) are never used. Outputs go to
data/persona_generator/ with non-destructive writes. No API calls.

Run:  python scripts/dp_persona_generator.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pilot_generate_api import profile_text  # noqa: E402  (the prompt-facing profile wording, shared with all providers)
from ssr_persona_matched_evaluation import safe_write_bytes  # noqa: E402

OUT = ROOT / "data/persona_generator"
SEED = 20261028
N_PER_PERSONA = 10
SPEC_VARS = ["brand", "price_paid", "usage_duration", "replacement_frequency", "upgrade_reason", "daily_usage",
             "area_type", "gender", "tech_comfort_1to5"]
EXCLUDED_FROM_PROFILES = ["Q21 rebuy intention", "Q26 free text", "any evaluation label"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_primary_with_personas(root: Path = ROOT, cohort: str = "pilot") -> pd.DataFrame:
    c = pd.read_csv(root / "data/human_survey_cleaned.csv", keep_default_na=False, dtype=str)
    a = pd.read_csv(root / "data/personas/persona_assignments.csv")
    merged = c[c.in_primary_cohort == "True"].merge(a, on="respondent_id")
    if cohort == "pilot":
        # First 215 submissions (HS001-HS215) comprise the historical 200-respondent pilot cohort
        return merged[merged.respondent_id.apply(lambda x: int(x[2:]) <= 215)]
    return merged


def generate_specs(primary: pd.DataFrame, n_per_persona: int = N_PER_PERSONA, seed: int = SEED) -> list[dict]:
    """Seeded independent sampling from each persona's marginals (attributes are NOT jointly sampled)."""
    rng = np.random.default_rng(seed)
    specs = []
    for pid, g in primary.groupby("persona_id"):
        for i in range(n_per_persona):
            s = {"synthetic_id": f"{pid.split('_')[0]}_S{i + 1:02d}", "persona_id": pid}
            for v in SPEC_VARS:
                s[v] = str(rng.choice(g[v].values))
            s["operating_system"] = "iOS" if s["brand"].startswith("Apple") else "Android"
            s["tech_comfort_1to5"] = int(float(s["tech_comfort_1to5"]))
            specs.append(s)
    return specs


EXT_VARS = ["age", "education", "current_status", "monthly_spending", "info_sources"]
# Where each required persona attribute lives. v1 = attributes in the prompts of every response generated so far.
ATTRIBUTE_COVERAGE = {
    "demographic": {"gender": "v1+v2", "area_type": "v1+v2", "age": "v2 only", "education": "v2 only",
                    "current_status": "v2 only", "monthly_spending": "v2 only"},
    "behavioural": {"brand": "v1+v2", "operating_system": "v1+v2", "price_paid": "v1+v2", "usage_duration": "v1+v2",
                    "replacement_frequency": "v1+v2", "upgrade_reason": "v1+v2", "daily_usage": "v1+v2",
                    "tech_comfort_1to5": "v1+v2", "main_uses": "v2 only", "info_sources": "v2 only"},
    "note": ("v1 = data/persona_generator/respondent_specs.jsonl (used in the prompts of all existing Gemini/OpenAI runs; unchanged). "
             "v2 = respondent_specs_extended.jsonl: same 40 specs plus the extra attributes, drawn from an independent random stream "
             "so v1 values are untouched. v2 attributes are NOT in the prompts of responses generated before v2 existed."),
}


def generate_extended(primary: pd.DataFrame, specs: list[dict], seed: int = SEED + 1) -> list[dict]:
    """Add demographic/behavioural attributes to each spec (independent draws, separate RNG stream)."""
    rng = np.random.default_rng(seed)
    by = {pid: g for pid, g in primary.groupby("persona_id")}
    tokens = {pid: np.array(g.main_use.str.split(r",\s*").explode().str.strip().replace("", pd.NA).dropna().values) for pid, g in by.items()}
    out = []
    for s in specs:
        g = by[s["persona_id"]]
        e = dict(s)
        e["age"] = int(float(rng.choice(g["age"].values)))
        for v in ("education", "current_status", "monthly_spending", "info_sources"):
            e[v] = str(rng.choice(g[v].values))
        uses = []
        while len(uses) < 2:
            u = str(rng.choice(tokens[s["persona_id"]]))
            if u not in uses:
                uses.append(u)
        e["main_uses"] = "; ".join(sorted(uses))
        out.append(e)
    return out


def profile_text_v2(s: dict) -> str:
    """Fuller persona wording for FUTURE generations (includes age, education, status, spending, uses, info source)."""
    nd = lambda v: "not disclosed" if str(v).lower().startswith("prefer not") else str(v)
    return (f"{s['age']}-year-old {str(s['gender']).lower()}, {str(s['current_status']).lower()} (education: {str(s['education']).lower()}), "
            f"living in a {str(s['area_type']).lower()}; monthly personal spending capacity {nd(s['monthly_spending'])}; "
            f"phone brand {s['brand']} ({s['operating_system']}), paid {nd(s['price_paid'])}; has used it for {s['usage_duration']}; "
            f"replaces phones {str(s['replacement_frequency']).lower()}; upgraded because: {str(s['upgrade_reason']).lower().rstrip('.')}; "
            f"uses it {s['daily_usage']} per day, mainly for {s['main_uses'].lower()}; chose it using {str(s['info_sources']).lower()}; "
            f"comfort with new technology {s['tech_comfort_1to5']} on a 1-5 scale")


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def persona_cards(profiles: list[dict], specs: list[dict]) -> list[dict]:
    """One evidence card per persona; every statistic is copied from persona_profiles.json (Step 4)."""
    by = {}
    for s in specs:
        by.setdefault(s["persona_id"], []).append(s["synthetic_id"])
    cards = []
    for p in profiles:
        top = lambda k: [f"{x['category']} ({x['n']})" for x in p[k]]
        cards.append({
            "persona_id": p["persona_id"], "n_human_respondents": p["n"], "pct_of_primary_cohort": p["pct_of_primary"],
            "defining_rule": p["defining_rule"], "median_age": p["median_age"], "age_range": p["age_range"],
            "top_operating_system": top("operating_system"), "top_brands": top("brand"), "top_daily_usage": top("daily_usage"),
            "top_gender": top("gender"), "top_current_status": top("current_status"), "top_area_type": top("area_type"),
            "top_monthly_spending": top("monthly_spending"), "top_replacement_frequency": top("replacement_frequency"),
            "top_upgrade_reason": top("upgrade_reason"), "top_info_sources": top("info_sources"),
            "top_main_uses": [f"{x['category']} ({x['pct_of_respondents']}% of respondents)" for x in p["main_use_top"]],
            "tech_comfort_mean_1to5": p["tech_comfort_mean_1to5"],
            "experience_themes_Q23_Q25": p["top_experience_themes_from_Q23_Q25"],
            "synthetic_respondent_ids": by.get(p["persona_id"], []),
            "interpretation_note": "Descriptive segment defined by price paid and daily use; not a statistically derived cluster.",
            "excluded_from_profile": EXCLUDED_FROM_PROFILES})
    return cards


def build(root: Path = ROOT, n_per_persona: int = N_PER_PERSONA, seed: int = SEED, write: bool = True,
          cohort: str = "pilot", out_suffix: str = "") -> dict:
    primary = load_primary_with_personas(root, cohort=cohort)
    specs = generate_specs(primary, n_per_persona, seed)
    for s in specs:
        s["profile_text"] = profile_text(s)
        s["spec_sha"] = spec_hash({k: v for k, v in s.items() if k not in ("profile_text", "spec_sha")})
    profiles = json.load(open(root / "data/personas/persona_profiles.json", encoding="utf-8"))
    cards = persona_cards(profiles, specs)
    ext = generate_extended(primary, [{k: v for k, v in s.items() if k not in ("profile_text", "spec_sha")} for s in specs], seed + 1)
    for e, s in zip(ext, specs):
        e["profile_text_v1"] = s["profile_text"]
        e["profile_text_v2"] = profile_text_v2(e)
        e["spec_sha_v1"] = s["spec_sha"]
    manifest = {"seed": seed, "extended_seed": seed + 1, "n_per_persona": n_per_persona, "n_specs": len(specs), "spec_variables": SPEC_VARS,
                "extended_variables": EXT_VARS + ["main_uses"], "attribute_coverage": ATTRIBUTE_COVERAGE,
                "primary_cohort_n": len(primary), "cohort": cohort,
                "inputs_sha256": {k: sha(root / k) for k in ("data/human_survey_cleaned.csv", "data/personas/persona_assignments.csv",
                                                           "data/personas/persona_profiles.json")},
                "note": "Specs sample each attribute independently from the persona marginals; combinations can be unrealistic."}
    result = {"specs": specs, "cards": cards, "manifest": manifest, "extended_specs": ext}
    if write:
        OUT.mkdir(parents=True, exist_ok=True)
        files = {f"respondent_specs{out_suffix}.jsonl": "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in specs),
                 f"respondent_specs_extended{out_suffix}.jsonl": "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in ext),
                 f"persona_cards{out_suffix}.json": json.dumps(cards, indent=2, ensure_ascii=False),
                 f"persona_generator_manifest{out_suffix}.json": json.dumps(manifest, indent=2)}
        result["written"] = [safe_write_bytes(t.encode("utf-8"), OUT / n).name for n, t in files.items()]
    return result


def main() -> int:
    use_latest = any(arg in ("--latest", "--all", "-v2") for arg in sys.argv[1:])
    if use_latest:
        r = build(cohort="all", out_suffix="_v2")
        print(f"[LATEST COHORT N={r['manifest']['primary_cohort_n']}] specs={len(r['specs'])} cards={len(r['cards'])} written={r['written']}")
        return 0
    r = build(cohort="pilot", out_suffix="")
    existing = [json.loads(l) for l in (ROOT / "data/pilot/synthetic_respondent_specs.jsonl").open(encoding="utf-8")]
    same = [{k: v for k, v in s.items() if k not in ("profile_text", "spec_sha")} for s in r["specs"]] == existing
    print(f"[PILOT COHORT N={r['manifest']['primary_cohort_n']}] specs={len(r['specs'])} cards={len(r['cards'])} reproduces_pilot_specs={same} written={r['written']}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
