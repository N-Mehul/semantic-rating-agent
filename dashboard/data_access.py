"""Read-only data layer for the Digital Persona dashboard.

Reads the saved pipeline outputs (persona generator, synthetic dataset, validation, construct prompts) relative to the repository
root and returns JSON-ready structures. It never generates responses, runs SSR, calls an API, or writes files. Missing, empty or
malformed files produce `{"available": False, "error": ...}` for that component; missing numbers become `None` (never 0).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "persona_cards": "data/persona_generator/persona_cards.json",
    "respondent_specs": "data/persona_generator/respondent_specs.jsonl",
    "persona_manifest": "data/persona_generator/persona_generator_manifest.json",
    "dataset": "data/synthetic_dataset/synthetic_survey_dataset.csv",
    "coverage": "data/synthetic_dataset/coverage_by_persona_source.csv",
    "dataset_manifest": "data/synthetic_dataset/dataset_manifest.json",
    "distribution": "data/validation/distribution_comparison.csv",
    "consistency": "data/validation/persona_consistency.csv",
    "ordering": "data/validation/persona_ordering.csv",
    "validation_summary": "data/validation/validation_summary.json",
    "construct_registry": "data/construct_prompts/construct_registry.json",
}
ENCODERS = ["local_mpnet_baseline", "local_minilm", "local_bge_base"]
SCORE_SUFFIXES = ["E", "sd", "entropy", "argmax"] + [f"p{k}" for k in range(1, 6)] + [f"s{k}" for k in range(1, 6)]
SPEC_ATTRS = ["brand", "operating_system", "price_paid", "usage_duration", "replacement_frequency", "upgrade_reason", "daily_usage",
              "area_type", "gender", "tech_comfort_1to5", "age", "education", "current_status", "monthly_spending", "main_uses", "info_sources"]
LOW_AGREEMENT = 0.5


def clean(v):
    """JSON-safe scalar: NaN/inf -> None, numpy -> python."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


class Dashboard:
    def __init__(self, root: Path | str = REPO_ROOT):
        self.root = Path(root)

    # ---- low-level loading (never raises)
    def _path(self, key: str) -> Path:
        return self.root / FILES[key]

    def _json(self, key: str):
        p = self._path(key)
        if not p.exists():
            return None, f"file not found: {FILES[key]}"
        try:
            text = p.read_text(encoding="utf-8")
            if not text.strip():
                return None, f"file is empty: {FILES[key]}"
            return json.loads(text), None
        except Exception as e:
            return None, f"could not read {FILES[key]}: {type(e).__name__}"

    def _jsonl(self, key: str):
        p = self._path(key)
        if not p.exists():
            return None, f"file not found: {FILES[key]}"
        try:
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            return rows, None
        except Exception as e:
            return None, f"could not read {FILES[key]}: {type(e).__name__}"

    def _csv(self, key: str, text_cols: tuple = ()):
        p = self._path(key)
        if not p.exists():
            return None, f"file not found: {FILES[key]}"
        try:
            df = pd.read_csv(p, keep_default_na=False, dtype={c: str for c in text_cols})
        except pd.errors.EmptyDataError:
            return None, f"file is empty: {FILES[key]}"
        except Exception as e:
            return None, f"could not read {FILES[key]}: {type(e).__name__}"
        return df, None

    def _dataset(self):
        df, err = self._csv("dataset", text_cols=("response_text",))
        if err:
            return None, err
        need = {"source_key", "synthetic_id", "persona_id", "status", "response_text"}
        if not need <= set(df.columns):
            return None, f"dataset is missing columns {sorted(need - set(df.columns))}"
        for c in df.columns:
            if any(c.endswith(f"__{s}") for s in SCORE_SUFFIXES) or c in ("encoder_spread_sd", "max_output_tokens", "tech_comfort_1to5", "age"):
                df[c] = pd.to_numeric(df[c], errors="coerce")   # blanks become NaN -> None, never 0
        return df, None

    def _numeric_csv(self, key: str):
        """Plain CSV load for the validation tables: blanks become None (never 0)."""
        p = self._path(key)
        if not p.exists():
            return None, f"file not found: {FILES[key]}"
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return None, f"file is empty: {FILES[key]}"
        except Exception as e:
            return None, f"could not read {FILES[key]}: {type(e).__name__}"
        return df.astype(object).where(df.notna(), None), None

    # ---- overview
    def overview(self) -> dict:
        files = []
        for key, rel in FILES.items():
            p = self.root / rel
            entry = {"key": key, "path": rel, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else None, "ok": False, "error": None}
            if p.exists():
                loader = {"persona_cards": lambda: self._json(key), "dataset": self._dataset, "respondent_specs": lambda: self._jsonl(key)}.get(
                    key, (lambda: self._json(key)) if rel.endswith(".json") else (lambda: self._csv(key)))
                obj, err = loader()
                entry["ok"], entry["error"] = err is None, err
            else:
                entry["error"] = "file not found"
            files.append(entry)
        ds, ds_err = self._dataset()
        cov = self.coverage()
        scored = int((ds.status == "scored").sum()) if ds is not None else None
        return {"files": files, "n_files_ok": sum(f["ok"] for f in files), "n_files": len(files),
                "dataset_rows": None if ds is None else len(ds), "scored_responses": scored,
                "excluded_responses": None if ds is None else int((ds.status != "scored").sum()),
                "sources_with_responses": None if ds is None else sorted(ds.source_key.unique().tolist()),
                "models": None if ds is None or "model_requested" not in ds else sorted({str(m) for m in ds.model_requested.dropna().unique() if str(m)}),
                "n_warnings": len(cov.get("warnings", [])) if cov.get("available") else None,
                "dataset_error": ds_err}

    # ---- personas
    def personas(self) -> dict:
        cards, err = self._json("persona_cards")
        if err or not isinstance(cards, list) or not cards:
            return {"available": False, "error": err or "persona_cards.json has no personas", "personas": []}
        specs, _ = self._jsonl("respondent_specs")
        n_specs: dict = {}
        for s in specs or []:
            n_specs[s.get("persona_id")] = n_specs.get(s.get("persona_id"), 0) + 1
        cov = self.coverage()
        by_p: dict = {}
        if cov.get("available"):
            for r in cov["rows"]:
                by_p.setdefault(r["persona_id"], []).append(r)
        ds, _ = self._dataset()
        out = []
        for c in cards:
            pid = c.get("persona_id")
            srcs = by_p.get(pid, [])
            scored = sum((r.get("passed_quality_and_scored") or 0) for r in srcs)
            models = {}
            if ds is not None and "model_requested" in ds:
                for k, g in ds[(ds.persona_id == pid) & (ds.status == "scored")].groupby("source_key"):
                    models[k] = sorted({str(m) for m in g.model_requested.dropna().unique() if str(m)})
            out.append({"card": c, "n_planned_specs": n_specs.get(pid), "coverage": srcs, "scored_responses_total": scored if cov.get("available") else None,
                        "has_generated_responses": (scored > 0) if cov.get("available") else None, "models_by_source": models})
        return {"available": True, "error": None, "personas": out,
                "personas_with_responses": [p["card"]["persona_id"] for p in out if p["has_generated_responses"]],
                "personas_without_responses": [p["card"]["persona_id"] for p in out if p["has_generated_responses"] is False]}

    # ---- responses
    def responses(self) -> dict:
        ds, err = self._dataset()
        if err:
            return {"available": False, "error": err, "rows": []}
        rows = []
        for r in ds.to_dict(orient="records"):
            row = {k: clean(r.get(k)) for k in ["source_key", "synthetic_id", "persona_id", "provider", "model_requested", "model_returned", "task_id",
                                               "question_id", "construct", "status", "exclusion_reasons", "finish_reason", "profile_version", "human_reference"]}
            row["expected_rating"] = {c: clean(r.get(f"{c}__E")) for c in ENCODERS}
            row["preview"] = str(r.get("response_text", ""))[:110]
            rows.append(row)
        return {"available": True, "error": None, "rows": rows, "n": len(rows)}

    def response_detail(self, source_key: str, synthetic_id: str) -> dict | None:
        ds, err = self._dataset()
        if err:
            return {"available": False, "error": err}
        m = ds[(ds.source_key == source_key) & (ds.synthetic_id == synthetic_id)]
        if m.empty:
            return None
        r = m.iloc[0].to_dict()
        manifest, _ = self._json("dataset_manifest")
        src_meta = ((manifest or {}).get("sources") or {}).get(source_key, {})
        scored = r.get("status") == "scored"
        encs = {}
        for c in ENCODERS:
            if f"{c}__E" not in ds.columns:
                continue
            if not scored:
                encs[c] = None
                continue
            p = [clean(r.get(f"{c}__p{k}")) for k in range(1, 6)]
            s = [clean(r.get(f"{c}__s{k}")) for k in range(1, 6)]
            encs[c] = {"probabilities": p, "similarities": s if all(v is not None for v in s) else None,
                       "expected_rating": clean(r.get(f"{c}__E")), "pmf_sd": clean(r.get(f"{c}__sd")), "entropy": clean(r.get(f"{c}__entropy")),
                       "argmax": clean(r.get(f"{c}__argmax")),
                       "probability_sum": None if any(v is None for v in p) else sum(p)}
        spec = {k: (None if clean(r.get(k)) == "" else clean(r.get(k))) for k in SPEC_ATTRS}   # "" (attribute absent from this prompt version) -> None
        prov = {k: clean(r.get(k)) for k in ["source_key", "provider", "model_requested", "model_returned", "task_id", "question_id", "construct",
                                            "anchor_construct", "anchor_sha", "profile_version", "human_reference", "prompt_sha", "spec_sha",
                                            "finish_reason", "generation_timestamp_utc", "max_output_tokens", "status", "exclusion_reasons"]}
        return {"available": True, "synthetic_id": synthetic_id, "persona_id": clean(r.get("persona_id")), "response_text": r.get("response_text"),
                "provenance": prov, "persona_attributes": spec, "encoders": encs, "encoder_spread_sd": clean(r.get("encoder_spread_sd")),
                "anchors": src_meta.get("anchors"), "anchor_labels": [1, 2, 3, 4, 5] if src_meta.get("anchors") else None,
                "scored": scored}

    # ---- coverage and warnings
    def coverage(self) -> dict:
        df, err = self._csv("coverage")
        if err or df is None:
            return {"available": False, "error": err, "rows": [], "warnings": [{"level": "error", "message": f"Coverage table unavailable ({err})."}]}
        rows = records(df)
        warnings = []
        for r in rows:
            if r.get("coverage_status") == "none":
                warnings.append({"level": "warning", "source_key": r["source_key"], "persona_id": r["persona_id"],
                                 "message": f"No generated responses for {r['persona_id']} in source {r['source_key']} ({r.get('not_generated')} of {r.get('planned')} planned prompts not generated)."})
            elif r.get("coverage_status") == "partial":
                warnings.append({"level": "warning", "source_key": r["source_key"], "persona_id": r["persona_id"],
                                 "message": f"Partial coverage for {r['persona_id']} in {r['source_key']}: {r.get('passed_quality_and_scored')} scored of {r.get('planned')} planned."})
        for k in sorted({r["source_key"] for r in rows}):
            if sum((r.get("responses_on_disk") or 0) for r in rows if r["source_key"] == k) == 0:
                warnings.append({"level": "error", "source_key": k, "message": f"Source {k} has zero saved responses; nothing can be scored or compared for it."})
        for pid in sorted({r["persona_id"] for r in rows}):
            if sum((r.get("passed_quality_and_scored") or 0) for r in rows if r["persona_id"] == pid) == 0:
                warnings.append({"level": "error", "persona_id": pid, "message": f"{pid} has no generated responses in any source."})
        v = self.validation()
        if v.get("available"):
            if (v.get("stability") or {}).get("status") != "computed":
                warnings.append({"level": "info", "message": "Repeated-generation stability is not supported by the current data: " + str((v.get("stability") or {}).get("reason", "no repeated generations")) + "."})
            ea = v.get("encoder_agreement") or []
            if ea and max(x["spearman_E"] for x in ea if x.get("spearman_E") is not None) < LOW_AGREEMENT:
                warnings.append({"level": "warning", "message": "Encoders disagree on the ordering of individual responses (all pairwise Spearman values below %.1f), so per-response scores depend on the encoder." % LOW_AGREEMENT})
        return {"available": True, "error": None, "rows": rows, "warnings": warnings}

    # ---- validation
    def validation(self) -> dict:
        summ, serr = self._json("validation_summary")
        dist, derr = self._numeric_csv("distribution")
        order, oerr = self._numeric_csv("ordering")
        cons, cerr = self._numeric_csv("consistency")
        errors = {k: e for k, e in (("validation_summary", serr), ("distribution", derr), ("ordering", oerr), ("consistency", cerr)) if e}
        if serr and derr:
            return {"available": False, "error": "; ".join(errors.values()), "errors": errors}
        drows = records(dist) if dist is not None else []
        for r in drows:
            r["interval_available"] = r.get("diff_ci95_lo") is not None and r.get("diff_ci95_hi") is not None
        crows = records(cons) if cons is not None else []
        csum: dict = {}
        for r in crows:
            k = (r.get("source_key"), r.get("persona_id"))
            d = csum.setdefault(k, {"source_key": k[0], "persona_id": k[1], "n": 0, "os_contradictions": 0, "mentions_spec_brand": 0, "mentions_other_brand": 0})
            d["n"] += 1
            d["os_contradictions"] += int(bool(r.get("os_contradiction")))
            d["mentions_spec_brand"] += int(r.get("mentions_spec_brand") is True)
            d["mentions_other_brand"] += int(bool(r.get("mentions_other_brand")))
        n_with = sum(1 for r in drows if r["interval_available"])
        return {"available": True, "error": None, "errors": errors, "distribution": drows,
                "distribution_headline": {"n_comparisons": len(drows), "n_with_interval": n_with,
                                          "n_interval_includes_zero": sum(1 for r in drows if r["interval_available"] and r.get("interval_includes_zero") is True),
                                          "equivalence_claimed_any": any(r.get("equivalence_claimed") is True for r in drows)},
                "ordering": records(order) if order is not None else [], "consistency_summary": list(csum.values()),
                "stability": (summ or {}).get("stability"), "encoder_agreement": (summ or {}).get("encoder_agreement") or [],
                "human_reference": (summ or {}).get("human_reference") or [], "integrity": (summ or {}).get("integrity") or [],
                "not_comparable": (summ or {}).get("not_comparable") or []}

    # ---- limitations
    def limitations(self) -> dict:
        manifest, merr = self._json("dataset_manifest")
        val = self.validation()
        items = []
        for t in (manifest or {}).get("limitations", []):
            items.append({"source": "dataset_manifest.json", "text": t})
        if val.get("available") and val.get("stability"):
            st = val["stability"]
            items.append({"source": "validation_summary.json", "text": f"Repeated-generation stability: {st.get('status')}. {st.get('reason', '')}".strip()})
        ssr = (manifest or {}).get("ssr") or {}
        if ssr.get("anchor_status"):
            items.append({"source": "dataset_manifest.json", "text": "Anchor status: " + ssr["anchor_status"]})
        items.append({"source": "dashboard", "text": "This dashboard displays saved outputs only. It does not generate responses, run SSR or call any API. "
                                                      "The results are preliminary and do not establish robust persona replication or equivalence between synthetic and human answers."})
        return {"available": True, "error": merr, "items": items}
