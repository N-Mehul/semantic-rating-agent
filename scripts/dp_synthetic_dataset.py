"""Synthetic Survey dataset builder: ingest saved LLM responses, score them with the canonical SSR functions, export one
provenance-rich table.

Workflow stages covered here: synthetic text response -> sentence embedding -> comparison with the five Likert anchor statements
(raw cosine similarities s1..s5) -> similarity-to-probability conversion (PMF p1..p5) -> expected Likert score E[r], plus uncertainty.

Reads already-saved responses only; it never calls an LLM or an embedding API. Each registered source is a separate experiment
(provider / model / run folder / question task) and is never merged into another. Responses failing the quality checks are kept
with status `excluded_quality` and are not scored. Persona/source combinations without responses appear only in the coverage
table as `not_generated`; nothing is imputed.

SSR: the approved anchor set of the source's construct (Behavioral Intention for the Q26 pilots; a proposed domain adaptation, not
the paper's six sets), the unmodified canonical_ssr functions (cosine_similarities -> similarities_to_pmf -> expected_rating) and
three local encoders. Uncertainty: PMF standard deviation, PMF entropy, spread of E[r] across encoders.

Run:  python scripts/dp_synthetic_dataset.py [--sources gemini_v2,openai_v1]
Outputs (data/synthetic_dataset/, non-destructive): synthetic_survey_dataset.csv, coverage_by_persona_source.csv, dataset_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf  # noqa: E402
from dp_constructs import TASKS, anchor_info  # noqa: E402
from ssr_eval_lib import split_by_quality  # noqa: E402
from ssr_persona_matched_evaluation import safe_write_bytes  # noqa: E402

OUT = ROOT / "data/synthetic_dataset"
PERSONAS = ["P1_premium_tier_owner", "P2_midrange_heavy_user", "P3_midrange_moderate_user", "P4_budget_or_undisclosed_price"]
ENCODERS = {"local_mpnet_baseline": "sentence-transformers/all-mpnet-base-v2",
            "local_minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "local_bge_base": "BAAI/bge-base-en-v1.5"}
SPEC_FIELDS = ["brand", "operating_system", "price_paid", "usage_duration", "replacement_frequency", "upgrade_reason",
               "daily_usage", "area_type", "gender", "tech_comfort_1to5", "age", "education", "current_status",
               "monthly_spending", "main_uses", "info_sources"]
PMF = [f"p{k}" for k in range(1, 6)]
SIM = [f"s{k}" for k in range(1, 6)]


def _src(label, provider, dir_, responses, enabled, note, task="Q26_BI_proxy"):
    t = TASKS[task]
    return {"label": label, "provider": provider, "dir": dir_, "responses": responses, "enabled": enabled, "note": note,
            "task_id": task, "construct": t["construct"], "question_id": t["question_id"],
            "profile_version": t["profile_version"], "human_reference": t["human_reference"]}


# Registry of experiments. `enabled` controls the default run; every source keeps its own provider/model/task label.
# Sources for FT_* tasks are placeholders: they have no responses until you run the generators with --prompts-file.
SOURCES = {
    "gemini_v2": _src("Gemini 2.5 Flash, 2048-token run", "google-genai", "data/pilot_gemini_v2", "gemini_generated_responses.jsonl", True,
                      "20 of 40 responses (P1 and P2 only); stopped on quota errors."),
    "openai_v1": _src("OpenAI chat model", "openai", "data/pilot_openai_v1", "openai_generated_responses.jsonl", True,
                      "Zero responses so far (insufficient quota)."),
    "gemini_v1": _src("Gemini 2.5 Flash, 512-token run", "google-genai", "data/pilot_gemini", "gemini_generated_responses.jsonl", False,
                      "Frozen original pilot; 19 of 20 answers truncated. Kept for reproducibility, off by default."),
    "gemini_FT_BI01": _src("Gemini, codebook FT_BI01 prompt", "google-genai", "data/pilot_gemini_FT_BI01", "gemini_generated_responses.jsonl", False,
                           "Not generated yet; no human FT_BI01 answers exist in the collected survey.", "FT_BI01"),
    "gemini_FT_PE01": _src("Gemini, codebook FT_PE01 prompt", "google-genai", "data/pilot_gemini_FT_PE01", "gemini_generated_responses.jsonl", False,
                           "Not generated yet; no human FT_PE01 answers exist in the collected survey.", "FT_PE01"),
    "openai_FT_BI01": _src("OpenAI, codebook FT_BI01 prompt", "openai", "data/pilot_openai_FT_BI01", "openai_generated_responses.jsonl", False,
                           "Not generated yet.", "FT_BI01"),
    "openai_FT_PE01": _src("OpenAI, codebook FT_PE01 prompt", "openai", "data/pilot_openai_FT_PE01", "openai_generated_responses.jsonl", False,
                           "Not generated yet.", "FT_PE01"),
}


def sha_text(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def load_source(key: str, root: Path = ROOT, registry: dict | None = None) -> dict:
    src = {"task_id": "Q26_BI_proxy", "construct": "BI", "question_id": "Q26", "profile_version": "v1",
           "human_reference": "Q26_Q21_proxy", **(registry or SOURCES)[key]}   # defaults keep older/custom registries working
    d = root / src["dir"]
    cfg = json.loads((d / "generation_config.json").read_text(encoding="utf-8")) if (d / "generation_config.json").exists() else {}
    return {"key": key, **src, "records": read_jsonl(d / src["responses"]), "prompts": {p["synthetic_id"]: p for p in read_jsonl(d / "prompts.jsonl")},
            "config": cfg, "failures": read_jsonl(d / "failures.jsonl")}


def local_embed_factories():
    from ssr_embedding_comparison import local_embedder
    return {cfg: (lambda m=m: local_embedder(m)) for cfg, m in ENCODERS.items()}


def score_detailed(texts: list[str], embed, anchors: list[str]):
    """Canonical SSR step by step: similarities s1..s5 -> PMF -> E[r]. Returns (S[n,5], P[n,5], E[n])."""
    A = np.asarray(embed(anchors), dtype=float)
    E = np.asarray(embed(texts), dtype=float)
    S, P, er = [], [], []
    for e in E:
        s = cosine_similarities(e, A)
        p = similarities_to_pmf(s)
        S.append(s)
        P.append(p)
        er.append(expected_rating(p))
    P = np.stack(P)
    assert np.allclose(P.sum(1), 1.0, atol=1e-12), "PMF normalisation failed"
    return np.stack(S), P, np.array(er)


def score_with_encoders(texts: list[str], anchors: list[str], factories: dict) -> dict:
    return {cfg: score_detailed(texts, make(), anchors) for cfg, make in factories.items()}


def build_dataset(source_keys: list[str], specs, factories: dict | None = None, root: Path = ROOT,
                  registry: dict | None = None, anchors: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """specs: list of v1 specs (applies to every source) or {"v1": [...], "v2": [...]} keyed by the source's profile_version."""
    reg = registry or SOURCES
    factories = factories if factories is not None else local_embed_factories()
    spec_sets = specs if isinstance(specs, dict) else {"v1": specs, "v2": specs}
    rows, cover, src_meta = [], [], {}
    for key in source_keys:
        S = load_source(key, root, reg)
        spec_by_id = {s["synthetic_id"]: s for s in spec_sets[S["profile_version"]]}
        ainfo = anchor_info(S["construct"]) if anchors is None else {"statements": anchors, "anchor_sha": sha_text(*anchors),
                                                                     "construct_name": "test anchors", "status": "TEST"}
        passed, excluded = split_by_quality(S["records"])
        reasons = {e["synthetic_id"]: e["exclusion_reasons"] for e in excluded}
        scored = score_with_encoders([r["response_text"] for r in passed], ainfo["statements"], factories) if passed else {}
        score_idx = {r["synthetic_id"]: i for i, r in enumerate(passed)}
        for r in S["records"]:
            sid = r["synthetic_id"]
            spec = spec_by_id.get(sid, {})
            pr = S["prompts"].get(sid)
            row = {"source_key": key, "provider": S["provider"], "model_requested": r.get("model_requested") or S["config"].get("model_requested"),
                   "model_returned": r.get("model_returned"), "task_id": r.get("task_id") or S["task_id"],
                   "question_id": r.get("question_id") or S["question_id"], "construct": S["construct"],
                   "anchor_construct": ainfo["construct_name"], "anchor_sha": ainfo["anchor_sha"], "profile_version": S["profile_version"],
                   "human_reference": S["human_reference"] or "none", "synthetic_id": sid, "persona_id": r["persona_id"],
                   **{k: spec.get(k) for k in SPEC_FIELDS}, "spec_sha": spec.get("spec_sha") or spec.get("spec_sha_v1"),
                   "prompt_sha": sha_text(pr["system"], pr["user"]) if pr else None,
                   "response_text": r["response_text"], "finish_reason": r.get("finish_reason"),
                   "generation_timestamp_utc": r.get("timestamp_utc"), "max_output_tokens": r.get("max_output_tokens"),
                   "spec_found": bool(spec)}
            if sid in score_idx:
                i = score_idx[sid]
                row.update(status="scored", exclusion_reasons="")
                Es = []
                for cfg, (Sm, P, er) in scored.items():
                    p = P[i]
                    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                    row.update({f"{cfg}__E": float(er[i]), f"{cfg}__sd": float(np.sqrt(p @ (np.arange(1, 6) - er[i]) ** 2)),
                                f"{cfg}__entropy": ent, f"{cfg}__argmax": int(p.argmax()) + 1,
                                **{f"{cfg}__{k}": float(Sm[i][j]) for j, k in enumerate(SIM)},
                                **{f"{cfg}__{k}": float(p[j]) for j, k in enumerate(PMF)}})
                    Es.append(float(er[i]))
                row["encoder_spread_sd"] = float(np.std(Es))
            else:
                row.update(status="excluded_quality", exclusion_reasons=";".join(reasons.get(sid, ["unknown"])))
            rows.append(row)
        planned = set(S["prompts"]) or {s["synthetic_id"] for s in spec_by_id.values()}
        for pid in PERSONAS:
            plan = {i for i in planned if i in spec_by_id and spec_by_id[i]["persona_id"] == pid}
            got = [r for r in S["records"] if r["persona_id"] == pid]
            ok = [r for r in passed if r["persona_id"] == pid]
            cover.append({"source_key": key, "persona_id": pid, "planned": len(plan), "responses_on_disk": len(got),
                          "passed_quality_and_scored": len(ok), "not_generated": len(plan - {r["synthetic_id"] for r in got}),
                          "coverage_status": "complete" if plan and len(ok) == len(plan) else ("none" if not got else "partial")})
        src_meta[key] = {"label": S["label"], "dir": S["dir"], "note": S["note"], "task_id": S["task_id"], "construct": S["construct"],
                         "anchor_sha": ainfo["anchor_sha"], "anchors": ainfo["statements"], "profile_version": S["profile_version"],
                         "human_reference": S["human_reference"] or "none"}
    df = pd.DataFrame(rows)
    cov = pd.DataFrame(cover)
    from survey_ssr_bridge import DEFAULT_ANCHOR_CSV_PATH
    manifest = {"sources": src_meta, "n_rows": len(df), "n_scored": int((df.status == "scored").sum()) if len(df) else 0,
                "encoders": ENCODERS if set(factories) == set(ENCODERS) else {k: "injected" for k in factories},
                "ssr": {"functions": "canonical_ssr.cosine_similarities -> similarities_to_pmf -> expected_rating", "epsilon": 0.0, "temperature": 1.0,
                        "anchor_status": "PROPOSED_DOMAIN_ADAPTATION_NOT_ORIGINAL_MAIER_ANCHOR (single set per construct; paper's six sets unavailable)"},
                "columns_note": "{encoder}__s1..s5 = cosine similarity to the 5 anchors; __p1..p5 = PMF; __E expected rating; __sd, __entropy = PMF uncertainty",
                "sha256": {"canonical_ssr.py": sha_file(root / "canonical_ssr.py"), "anchors_csv": sha_file(DEFAULT_ANCHOR_CSV_PATH)},
                "limitations": ["Personas without responses are listed as not_generated in the coverage table; nothing is imputed.",
                                "Single anchor set per construct and local encoders, not the paper's six sets or embedding model.",
                                "v1 specs sample attributes independently; v2-only attributes (age, education, ...) are absent from v1-prompt sources.",
                                "Sources with human_reference 'none' have no comparable human answers in the collected survey."]}
    return df, cov, manifest


def export(df, cov, manifest, out: Path = OUT) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    w = []
    w.append(safe_write_bytes(df.to_csv(index=False, lineterminator="\n").encode("utf-8"), out / "synthetic_survey_dataset.csv").name)
    w.append(safe_write_bytes(cov.to_csv(index=False, lineterminator="\n").encode("utf-8"), out / "coverage_by_persona_source.csv").name)
    w.append(safe_write_bytes(json.dumps(manifest, indent=2).encode("utf-8"), out / "dataset_manifest.json").name)
    return w


def load_spec_sets(root: Path = ROOT) -> dict | None:
    v1, v2 = root / "data/persona_generator/respondent_specs.jsonl", root / "data/persona_generator/respondent_specs_extended.jsonl"
    if not v1.exists() or not v2.exists():
        return None
    return {"v1": read_jsonl(v1), "v2": read_jsonl(v2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(k for k, v in SOURCES.items() if v["enabled"]))
    a = ap.parse_args()
    keys = [k.strip() for k in a.sources.split(",") if k.strip()]
    bad = [k for k in keys if k not in SOURCES]
    if bad:
        print(f"Unknown source(s) {bad}; choose from {list(SOURCES)}")
        return 2
    specs = load_spec_sets()
    if specs is None:
        print("Run  python scripts/dp_persona_generator.py  first (respondent specs missing).")
        return 2
    df, cov, manifest = build_dataset(keys, specs)
    written = export(df, cov, manifest)
    print(f"rows={len(df)} scored={manifest['n_scored']} written={written}")
    print(cov.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
