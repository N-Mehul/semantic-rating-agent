"""End-to-end consistency verifier for the Digital Persona pipeline outputs (read-only; no API calls, no generation).

Checks that expected files exist, that the synthetic dataset is exactly the saved quality-approved responses (no imputed
rows), that missing persona/source combinations are marked not_generated, that numbers agree across files, and that the
validation report states encoder disagreement and the unsupported stability analysis.

Run:  python scripts/verify_pipeline_outputs.py        (exit code 0 = all checks passed)
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
from dp_synthetic_dataset import ENCODERS, PERSONAS, SOURCES, read_jsonl  # noqa: E402
from ssr_eval_lib import split_by_quality  # noqa: E402

EXPECTED_FILES = [
    "data/persona_generator/persona_cards.json", "data/persona_generator/respondent_specs.jsonl",
    "data/persona_generator/persona_generator_manifest.json", "data/persona_generator/respondent_specs_extended.jsonl",
    "data/construct_prompts/construct_registry.json", "data/construct_prompts/Q26_BI_proxy_prompts.jsonl",
    "data/construct_prompts/FT_PE01_prompts.jsonl", "data/construct_prompts/FT_BI01_prompts.jsonl", "data/validation/persona_ordering.csv", "data/synthetic_dataset/synthetic_survey_dataset.csv",
    "data/synthetic_dataset/coverage_by_persona_source.csv", "data/synthetic_dataset/dataset_manifest.json",
    "data/validation/distribution_comparison.csv", "data/validation/persona_consistency.csv",
    "data/validation/validation_summary.json", "docs/project_status/validation_agent_report.md",
    "docs/project_status/pipeline_component_checklist.md", "README.md"]
PMF = [f"p{k}" for k in range(1, 6)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_checks(root: Path = ROOT) -> list[tuple[str, bool, str]]:
    out = []

    def add(name, ok, detail=""):
        out.append((name, bool(ok), str(detail)))

    missing = [f for f in EXPECTED_FILES if not (root / f).exists()]
    add("all expected pipeline output files exist", not missing, f"missing={missing}")
    if missing:
        return out
    df = pd.read_csv(root / "data/synthetic_dataset/synthetic_survey_dataset.csv", keep_default_na=False)
    cov = pd.read_csv(root / "data/synthetic_dataset/coverage_by_persona_source.csv")
    man = json.loads((root / "data/synthetic_dataset/dataset_manifest.json").read_text(encoding="utf-8"))
    report = (root / "docs/project_status/validation_agent_report.md").read_text(encoding="utf-8")

    # --- dataset == saved, quality-approved responses of each enabled source (no fabrication / imputation)
    for key, s in SOURCES.items():
        if key not in man["sources"]:
            continue
        saved = read_jsonl(root / s["dir"] / s["responses"])
        passed, excluded = split_by_quality(saved)
        rows = df[df.source_key == key]
        add(f"{key}: dataset rows equal saved responses (count {len(saved)})", len(rows) == len(saved), f"rows={len(rows)}")
        add(f"{key}: row IDs are exactly the saved IDs, no duplicates",
            sorted(rows.synthetic_id) == sorted(r["synthetic_id"] for r in saved) and rows.synthetic_id.is_unique)
        text = {r["synthetic_id"]: r["response_text"] for r in saved}
        add(f"{key}: response texts are verbatim copies", all(text[i] == t for i, t in zip(rows.synthetic_id, rows.response_text)))
        add(f"{key}: scored rows are exactly the quality-passing responses",
            set(rows[rows.status == "scored"].synthetic_id) == {r["synthetic_id"] for r in passed}
            and set(rows[rows.status == "excluded_quality"].synthetic_id) == {e["synthetic_id"] for e in excluded},
            f"scored={int((rows.status == 'scored').sum())} excluded={len(excluded)}")
    add("dataset contains no rows from unregistered or disabled sources", set(df.source_key) <= set(man["sources"]), sorted(set(df.source_key)))
    v2 = df[df.source_key == "gemini_v2"]
    add("Gemini v2: 20 scored rows, personas P1 and P2 only (10 each)",
        len(v2) == 20 and (v2.status == "scored").all() and v2.persona_id.str[:2].value_counts().to_dict() == {"P1": 10, "P2": 10})
    add("OpenAI v1 contributes zero rows", not (df.source_key == "openai_v1").any())

    # --- numeric consistency of scores
    s = df[df.status == "scored"]
    ok_pmf = True
    for cfg in ENCODERS:
        P = s[[f"{cfg}__{k}" for k in PMF]].astype(float).values
        ok_pmf &= bool(np.allclose(P.sum(1), 1.0, atol=1e-12) and (P >= -1e-12).all()
                       and np.allclose(s[f"{cfg}__E"].astype(float), P @ np.arange(1, 6)) and s[f"{cfg}__E"].astype(float).between(1, 5).all())
    add("scored rows: PMFs sum to 1, are non-negative, E[r] = sum r*p(r) within [1,5], for all encoders", ok_pmf)
    for cfg in ENCODERS:
        ref = pd.read_csv(root / f"data/pilot_embedding_comparison/synthetic_scores_{cfg}_gemini_v2.csv").set_index("synthetic_id")
        g = v2.set_index("synthetic_id")
        add(f"dataset scores equal the independent Step 8 export ({cfg})",
            np.allclose(g[f"{cfg}__E"].astype(float), ref.loc[g.index, "expected_rating"], atol=1e-9))

    # --- coverage table: missing combinations explicitly not_generated
    keys = list(man["sources"])
    add("coverage has one row per enabled source x persona", len(cov) == len(keys) * len(PERSONAS) and
        set(zip(cov.source_key, cov.persona_id)) == {(k, p) for k in keys for p in PERSONAS}, f"rows={len(cov)}")
    for r in cov.itertuples():
        n_rows = int(((df.source_key == r.source_key) & (df.persona_id == r.persona_id)).sum())
        if r.responses_on_disk == 0:
            add(f"coverage {r.source_key}/{r.persona_id[:2]}: marked 'none', not_generated={r.not_generated}, no dataset rows",
                r.coverage_status == "none" and r.not_generated == r.planned > 0 and n_rows == 0)
        else:
            add(f"coverage {r.source_key}/{r.persona_id[:2]}: counts agree with dataset",
                n_rows == r.responses_on_disk and r.passed_quality_and_scored + r.not_generated <= r.planned)
    add("manifest counts agree with dataset", man["n_rows"] == len(df) and man["n_scored"] == len(s))

    # --- specs / cards
    specs = read_jsonl(root / "data/persona_generator/respondent_specs.jsonl")
    pilot = read_jsonl(root / "data/pilot/synthetic_respondent_specs.jsonl")
    add("40 respondent specs equal the original pilot specs",
        [{k: v for k, v in x.items() if k not in ("profile_text", "spec_sha")} for x in specs] == pilot and len(specs) == 40)
    add("every dataset row links to a spec", bool(v2.spec_found.astype(bool).all()) and v2.spec_sha.astype(str).str.len().gt(0).all())
    cards = json.loads((root / "data/persona_generator/persona_cards.json").read_text(encoding="utf-8"))
    add("4 persona cards, 10 synthetic ids each", len(cards) == 4 and all(len(c["synthetic_respondent_ids"]) == 10 for c in cards))

    # --- validation outputs and report wording
    d = pd.read_csv(root / "data/validation/distribution_comparison.csv")
    add("validation distribution table: 6 rows (P1, P2 x 3 encoders), no equivalence claimed",
        len(d) == 6 and set(d.persona_id.str[:2]) == {"P1", "P2"} and (d.equivalence_claimed.astype(str) == "False").all())
    summ = json.loads((root / "data/validation/validation_summary.json").read_text(encoding="utf-8"))
    add("validation summary: stability status is not_supported_by_data", summ["stability"]["status"] == "not_supported_by_data")
    add("validation report states encoder disagreement", "Interpretation: pairwise Spearman between encoders" in report and "depends strongly on which encoder" in report)
    add("validation report states repeated-generation stability is not supported", "not_supported_by_data" in report and "generated once per model" in report)
    add("validation report states results are preliminary / no equivalence", "preliminary" in report.lower() and "equivalence" in report.lower())
    ea = summ["encoder_agreement"]
    add("encoder agreement values in the summary are all low (< 0.5), matching the report wording", ea and max(x["spearman_E"] for x in ea) < 0.5,
        [round(x["spearman_E"], 3) for x in ea])

    # --- protected inputs unchanged vs hashes recorded by earlier steps
    rec = json.loads((root / "data/pilot/ssr_config.json").read_text(encoding="utf-8"))["sha256"]
    add("canonical_ssr.py unchanged since Step 6", sha(root / "canonical_ssr.py") == rec["canonical_ssr.py"])
    add("approved anchor CSV unchanged since Step 6", sha(root / "docs/survey_instrument/smartphone_construct_anchor_statements.csv") == rec["anchors_csv"])
    add("dataset manifest hashes equal current canonical_ssr.py / anchors", man["sha256"]["canonical_ssr.py"] == sha(root / "canonical_ssr.py")
        and man["sha256"]["anchors_csv"] == rec["anchors_csv"])
    q = (root / "docs/project_status/human_survey_data_quality_report.md").read_text(encoding="utf-8")
    add("raw survey workbook unchanged since Step 3", sha(root / "data/Capstone_Reviews(Responses).xlsx") in q)
    return out


def main() -> int:
    res = run_checks()
    for n, ok, d in res:
        print(("PASS " if ok else "FAIL ") + n + (f"  [{d}]" if d and not ok else ""))
    bad = [n for n, ok, _ in res if not ok]
    print(f"\n{len(res) - len(bad)}/{len(res)} checks passed")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
