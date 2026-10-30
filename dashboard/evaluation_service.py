"""Evaluation Service implementing Independent Single-Set SSR for Six Constructs (Option A).

Domain-Adapted Semantic Similarity Rating (SSR) based on:
"LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings"
by Maier et al. (arXiv:2510.08338v3).

Methodological Principles:
1. Six Separate Constructs: Performance Expectancy (PE), Effort Expectancy (EE),
   Hedonic Motivation (HM), Social Influence (SI), Perceived Protection (PP),
   and Behavioural Intention (BI).
2. Five Likert Anchors per Construct: Each construct is evaluated independently
   against its own 5-statement anchor set.
3. No Cross-Construct Averaging: In psychometric theory (UTAUT2), antecedents
   (PE, EE, HM, SI, PP) cannot be averaged with the outcome criterion (BI).
4. Primary Outcome vs. Supporting Antecedents: BI represents primary adoption intent;
   PE, EE, HM, SI, and PP provide diagnostic explanations of adoption drivers/barriers.
5. Canonical SSR Mathematics: Minimum-subtraction normalisation (Eq. 7 & 8) with
   epsilon=0.0, T=1.0. No top-k nearest-neighbor truncation or Softmax.
6. Local AI Execution: Local Qwen 2.5 7B (text generation only, zero rating leaks)
   and nomic-embed-text (768-dim embeddings). Zero OpenAI API dependencies.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any, Final

import numpy as np

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from canonical_ssr import (
    PAPER_EPSILON,
    PAPER_TEMPERATURE,
    cosine_similarities,
    expected_rating,
    similarities_to_pmf,
)
from scripts import ollama_utils
from scripts.ollama_embedding_adapter import get_ollama_embedding_adapter
from scripts.product_evaluation_prompts import (
    ProductConcept,
    ProductFeature,
    SUPPORTED_CONSTRUCTS,
    UNAVAILABLE_CONSTRUCTS,
    build_product_evaluation_prompt,
    validate_prompt_for_ssr,
    validate_requested_constructs,
)
from survey_ssr_bridge import CONSTRUCT_CODE_MAP, get_construct_anchors

EVAL_BASE_DIR: Final[Path] = ROOT / "data" / "product_evaluation"
APPROVED_ANCHOR_SETS: Final[list[str]] = ["PE", "EE", "HM", "SI", "PP", "BI"]


def slugify(text: str) -> str:
    """Convert text to URL/directory friendly slug."""
    s = re.sub(r"[^\w\s-]", "", text.strip().lower())
    s = re.sub(r"[-\s]+", "_", s)
    return s[:40] or "product_eval"


class ProductEvaluationService:
    """Backend service orchestrating the exact 8-step Maier et al. SSR workflow."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self._lock = threading.Lock()

    def get_personas_meta(self) -> list[dict[str, Any]]:
        """Step 1: Load empirical synthetic consumer profiles."""
        profiles_path = self.root / "data" / "personas" / "persona_profiles.json"
        if not profiles_path.exists():
            return []
        profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
        out = []
        for p in profiles:
            pid = p["persona_id"]
            def top_cat(k, default=""):
                v = p.get(k)
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v[0].get("category", default)
                return str(v) if v else default

            out.append({
                "persona_id": pid,
                "label": pid.split("_")[0] + " — " + pid.split("_", 1)[1].replace("_", " ").title(),
                "defining_rule": p.get("defining_rule", ""),
                "n_respondents": p.get("n", 0),
                "pct_cohort": p.get("pct_of_primary", 0.0),
                "median_age": p.get("median_age", 20.0),
                "primary_brand": top_cat("brand", "Samsung"),
                "operating_system": top_cat("operating_system", "Android"),
                "daily_usage": top_cat("daily_usage", "4-6 hours"),
                "monthly_spending": top_cat("monthly_spending", "Moderate"),
                "tech_comfort": p.get("tech_comfort_mean_1to5", 4.0),
            })
        return out

    def get_survey_meta(self) -> dict[str, Any]:
        """Metadata documenting human survey origin vs synthetic consumers."""
        summary_csv = self.root / "data" / "personas" / "aggregate_survey_summary.csv"
        demographics: dict[str, list[dict[str, Any]]] = {}
        if summary_csv.exists():
            with open(summary_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    var = row.get("variable", "")
                    demographics.setdefault(var, []).append({
                        "category": row.get("category", ""),
                        "n": int(row.get("n", 0)),
                        "pct": float(row.get("pct", 0.0)),
                    })

        return {
            "dataset_filename": "Capstone_Reviews(Responses)(1).xlsx",
            "total_raw_submissions": 275,
            "eligible_respondents": 258,
            "excluded_submissions": 17,
            "exclusion_reasons": "Hierarchy cleaning: 1 missing consent, 10 incomplete records, 6 out-of-scope age",
            "empirical_personas_count": 4,
            "traceability_file": "data/personas/persona_assignments.csv",
            "demographics": demographics,
        }

    def get_constructs_meta(self) -> dict[str, Any]:
        """Return the six approved SSR anchor sets and PV qualitative status."""
        ssr_constructs = []
        for code in APPROVED_ANCHOR_SETS:
            meta = SUPPORTED_CONSTRUCTS[code]
            ssr_constructs.append({
                "code": code,
                "name": meta["canonical_name"],
                "focus": meta["prompt_focus"],
                "has_ssr_rating": True,
            })
        qualitative = [{
            "code": "PV",
            "name": "Price Value",
            "focus": SUPPORTED_CONSTRUCTS["PV"]["prompt_focus"],
            "has_ssr_rating": False,
            "note": "Qualitative feedback only. No approved 5-anchor Likert set exists in survey instrument.",
        }]
        return {
            "ssr_constructs": ssr_constructs,
            "qualitative_constructs": qualitative,
            "unavailable": UNAVAILABLE_CONSTRUCTS,
        }

    def check_ollama_backends(self) -> dict[str, Any]:
        """Verify presence of Qwen 2.5 7B and nomic-embed-text in local Ollama."""
        gen_probe = ollama_utils.probe(timeout=3.0)
        embed_adapter = get_ollama_embedding_adapter()
        embed_probe = embed_adapter.probe()

        installed = gen_probe.get("models", [])
        qwen_ready = "qwen2.5:7b-instruct" in installed or any(m.startswith("qwen2.5:7b") for m in installed)
        embed_ready = embed_probe.get("model_available", False)

        return {
            "ollama_running": gen_probe.get("running", False),
            "host": gen_probe.get("host", "http://localhost:11434"),
            "generator_model": "qwen2.5:7b-instruct",
            "generator_ready": qwen_ready,
            "embedding_model": "nomic-embed-text",
            "embedding_ready": embed_ready,
            "all_ready": gen_probe.get("running", False) and qwen_ready and embed_ready,
            "installed_models": installed,
            "error": gen_probe.get("error") or embed_probe.get("error"),
        }

    def list_saved_evaluations(self) -> list[dict[str, Any]]:
        """List previously saved product evaluation runs."""
        if not EVAL_BASE_DIR.exists():
            return []
        evals = []
        for d in EVAL_BASE_DIR.iterdir():
            if not d.is_dir():
                continue
            concept_file = d / "product_concept.json"
            ssr_file = d / "ssr_results" / "ssr_results.json"
            if concept_file.exists():
                try:
                    c = json.loads(concept_file.read_text(encoding="utf-8"))
                    has_ssr = ssr_file.exists()
                    evals.append({
                        "eval_id": d.name,
                        "product_name": c.get("name", d.name),
                        "category": c.get("category", "Smartphone"),
                        "price_inr": c.get("price_inr"),
                        "has_ssr_results": has_ssr,
                        "timestamp": c.get("timestamp_created"),
                    })
                except Exception:
                    pass
        return sorted(evals, key=lambda x: str(x.get("timestamp") or ""), reverse=True)

    def options(self) -> dict[str, Any]:
        """Initialization payload for the 8-step dashboard."""
        example_product = {
            "name": "Nova AI Smartphone",
            "category": "Mid-Range Smartphone",
            "description": "A hypothetical mid-range smartphone designed for students and young professionals, with AI-powered features.",
            "price_inr": 24999.0,
            "target_group": "College students and young professionals",
            "target_age_group": "18-25 years",
            "market": "India",
            "brand_or_ecosystem": "Android ecosystem",
            "features": [
                {"name": "AI Personal Assistant", "explanation": "Daily tasks, reminders, smart notifications, and study support."},
                {"name": "AI Camera Enhancement", "explanation": "Automated low-light and portrait enhancement for photos and reels."},
                {"name": "AI Battery Optimisation", "explanation": "Smart power management adapting to daily app usage routines."},
                {"name": "On-Device AI Processing", "explanation": "Selected tasks processed locally to protect personal data from cloud exposure."},
                {"name": "Smooth Multitasking Performance", "explanation": "Optimized chip and high refresh rate display for smooth daily use."},
                {"name": "Simple Interface", "explanation": "Clean UI with optional contextual AI shortcuts."},
            ],
            "constructs": ["PE", "EE", "HM", "SI", "PP", "BI", "PV"],
            "personas": ["P1_premium_tier_owner", "P2_midrange_heavy_user", "P3_midrange_moderate_user", "P4_budget_or_undisclosed_price"],
        }
        return {
            "title": "Digital Persona — Independent 6-Construct SSR Platform",
            "subtitle": "Independent Single-Set SSR for 6 Constructs (Maier et al. adaptation): 4 Personas, 6 Independent Constructs (PE, EE, HM, SI, PP, BI), Zero Cross-Construct Averaging.",
            "methodology": "Independent Single-Set SSR for Six Constructs (Option A)",
            "primary_adoption_construct": "BI",
            "supporting_constructs": ["PE", "EE", "HM", "SI", "PP"],
            "cross_construct_averaging": False,
            "embedding_difference_note": "Paper uses OpenAI text-embedding-3-small (1536 dims). This implementation uses local Ollama nomic-embed-text (768 dims) for zero API key dependency and offline reproduction.",
            "personas": self.get_personas_meta(),
            "survey_meta": self.get_survey_meta(),
            "constructs": self.get_constructs_meta(),
            "backends": self.check_ollama_backends(),
            "example_product": example_product,
            "saved_evaluations": self.list_saved_evaluations(),
        }

    def load_evaluation(self, eval_id: str) -> dict[str, Any] | None:
        """Load a saved evaluation directory containing 8-step calculations."""
        target_dir = EVAL_BASE_DIR / eval_id
        if not target_dir.is_dir():
            return None
        concept_file = target_dir / "product_concept.json"
        responses_file = target_dir / "persona_responses.json"
        ssr_file = target_dir / "ssr_results" / "ssr_results.json"

        if not concept_file.exists():
            return None

        concept = json.loads(concept_file.read_text(encoding="utf-8"))
        responses = json.loads(responses_file.read_text(encoding="utf-8")) if responses_file.exists() else []
        ssr_results = json.loads(ssr_file.read_text(encoding="utf-8")) if ssr_file.exists() else None

        # Extract persona summaries and comparison table without cross-construct averaging
        persona_summaries = []
        comparison = []
        if ssr_results:
            persona_summaries = ssr_results.get("persona_summaries", [])
            # Fallback or reconstruct if needed
            if not persona_summaries and "records" in ssr_results:
                by_persona = {}
                for r in ssr_results["records"]:
                    by_persona.setdefault(r["persona_id"], []).append(r)
                for pid, recs in by_persona.items():
                    scores = {r["construct_code"]: r["expected_rating"] for r in recs}
                    bi_rec = next((r for r in recs if r["construct_code"] == "BI"), None)
                    bi_rating = bi_rec["expected_rating"] if bi_rec else scores.get("BI", 0.0)
                    bi_pmf = bi_rec["pmf_probabilities"] if bi_rec else [0.0]*5
                    persona_summaries.append({
                        "persona_id": pid,
                        "product_name": concept.get("name", "Smartphone"),
                        "primary_adoption_rating": round(bi_rating, 4),
                        "primary_adoption_pmf": bi_pmf,
                        "construct_scores": scores,
                        "anchor_sets": recs,
                    })
                    comparison.append({
                        "persona_id": pid,
                        "scores": scores,
                        "primary_adoption_rating": round(bi_rating, 4),
                        "primary_adoption_pmf": bi_pmf,
                    })
            else:
                for ps in persona_summaries:
                    scores = ps.get("construct_scores", {})
                    if not scores:
                        for a_set in ps.get("anchor_sets", []):
                            scores[a_set["construct_code"]] = a_set["expected_rating"]
                    bi_rating = ps.get("primary_adoption_rating", scores.get("BI", 0.0))
                    bi_pmf = ps.get("primary_adoption_pmf", [])
                    if not bi_pmf:
                        bi_rec = next((a for a in ps.get("anchor_sets", []) if a["construct_code"] == "BI"), None)
                        if bi_rec:
                            bi_pmf = bi_rec.get("pmf_probabilities", [])
                    comparison.append({
                        "persona_id": ps["persona_id"],
                        "scores": scores,
                        "primary_adoption_rating": round(bi_rating, 4),
                        "primary_adoption_pmf": bi_pmf,
                    })

        return {
            "eval_id": eval_id,
            "concept": concept,
            "responses": responses,
            "ssr_results": ssr_results,
            "persona_summaries": persona_summaries,
            "comparison": comparison,
        }

    def run_evaluation(self, body: dict[str, Any]) -> dict[str, Any]:
        """Execute the exact 8-step methodology: prompts -> Qwen generation -> local nomic SSR -> 6-set averaged PMF -> Expected Likert Rating."""
        with self._lock:
            # Step 2: Validate Product Concept inputs
            name = str(body.get("name", "")).strip()
            if len(name) < 2:
                raise ValueError("Product name must be at least 2 characters.")
            description = str(body.get("description", "")).strip()
            if len(description) < 10:
                raise ValueError("Product description must be at least 10 characters.")
            category = str(body.get("category", "Smartphone")).strip()
            target_group = str(body.get("target_group", "")).strip()
            price_raw = body.get("price_inr")
            try:
                price_inr = float(price_raw) if price_raw not in (None, "") else None
            except (ValueError, TypeError):
                raise ValueError("Proposed price must be a valid number.")

            raw_features = body.get("features", [])
            if not isinstance(raw_features, list) or len(raw_features) == 0:
                raise ValueError("Please provide at least one product feature.")

            features = []
            for f in raw_features:
                fn = str(f.get("name", "")).strip()
                fe = str(f.get("explanation", "")).strip()
                if fn and fe:
                    features.append(ProductFeature(name=fn, explanation=fe, category=f.get("category")))
            if not features:
                raise ValueError("Each feature must have both a name and an explanation.")

            product = ProductConcept(
                name=name,
                description=description,
                price_inr=price_inr,
                target_group=target_group,
                brand_or_ecosystem=body.get("brand_or_ecosystem"),
                features=features,
            )

            # Step 1: Select Synthetic Consumers (Personas)
            req_personas = body.get("personas", [
                "P1_premium_tier_owner",
                "P2_midrange_heavy_user",
                "P3_midrange_moderate_user",
                "P4_budget_or_undisclosed_price",
            ])
            all_personas = [p["persona_id"] for p in self.get_personas_meta()]
            personas = [p for p in req_personas if p in all_personas]
            if not personas:
                raise ValueError("Please select at least one valid persona.")

            # Validate constructs (the six approved anchor sets + qualitative PV)
            req_constructs = body.get("constructs", ["PE", "EE", "HM", "SI", "PP", "BI", "PV"])
            valid_constructs = validate_requested_constructs(req_constructs)

            # Verify local Ollama backends
            backends = self.check_ollama_backends()
            if not backends["ollama_running"]:
                raise RuntimeError("Local Ollama is not running. Please start Ollama before evaluating.")
            if not backends["generator_ready"]:
                raise RuntimeError("Ollama model 'qwen2.5:7b-instruct' is not installed.")
            if not backends["embedding_ready"]:
                raise RuntimeError("Ollama model 'nomic-embed-text' is not installed.")

            # Destination directory
            eval_slug = slugify(name)
            timestamp_str = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_id = f"{eval_slug}_{timestamp_str}"
            eval_dir = EVAL_BASE_DIR / eval_id
            eval_dir.mkdir(parents=True, exist_ok=True)
            ssr_dir = eval_dir / "ssr_results"
            ssr_dir.mkdir(parents=True, exist_ok=True)

            # Save product concept
            concept_dict = {
                "name": product.name,
                "category": category,
                "description": product.description,
                "price_inr": product.price_inr,
                "target_group": product.target_group,
                "target_age_group": body.get("target_age_group", "18-25 years"),
                "market": body.get("market", "India"),
                "brand_or_ecosystem": product.brand_or_ecosystem,
                "features": [{"name": f.name, "explanation": f.explanation, "category": f.category} for f in product.features],
                "evaluated_constructs": valid_constructs,
                "timestamp_created": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            (eval_dir / "product_concept.json").write_text(json.dumps(concept_dict, indent=2, ensure_ascii=False), encoding="utf-8")

            # Step 3: Generate Natural-Language Responses (TEXT ONLY, ZERO NUMERICAL RATINGS)
            responses = []
            for pid in personas:
                prompt = build_product_evaluation_prompt(
                    product=product,
                    persona_id_or_dict=pid,
                    constructs=valid_constructs,
                    root=self.root,
                )
                issues = validate_prompt_for_ssr(prompt)
                if issues:
                    raise RuntimeError(f"Prompt validation failed for {pid}: {issues}")

                gen_prompt = {
                    "system": prompt["system"],
                    "user": prompt["user"],
                    "timeout_s": 180.0,
                }
                ollama_utils.SETTINGS["temperature"] = 0.7
                rec, err = ollama_utils.generate("qwen2.5:7b-instruct", gen_prompt, timeout=180.0)
                if err is not None:
                    raise RuntimeError(f"Generation failed for {pid}: {err}")

                resp_text = rec["response_text"].strip()
                responses.append({
                    "persona_id": pid,
                    "product_name": product.name,
                    "response_text": resp_text,
                    "model_returned": rec.get("model_returned", "qwen2.5:7b-instruct"),
                    "finish_reason": rec.get("finish_reason", "STOP"),
                    "timestamp_utc": rec.get("timestamp_utc"),
                })

            (eval_dir / "persona_responses.json").write_text(json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8")

            # Step 4: Sentence Embeddings via local nomic-embed-text (same model for all texts)
            embed_adapter = get_ollama_embedding_adapter("nomic-embed-text")

            # Load the six approved anchor sets
            anchor_data = {}
            anchor_embs = {}
            for code in APPROVED_ANCHOR_SETS:
                anchors = get_construct_anchors(code)
                stmts = list(anchors.statements)
                anchor_data[code] = {
                    "code": code,
                    "name": anchors.construct_name,
                    "statements": stmts,
                    "labels": list(anchors.labels),
                }
                anchor_embs[code] = embed_adapter.embed_texts(stmts)

            # Step 5, 6, 7, 8: For each synthetic consumer response:
            persona_summaries = []
            intermediate_records = []
            flat_records = []
            summary_csv_rows = []
            inter_csv_rows = []

            for r in responses:
                pid = r["persona_id"]
                resp_text = r["response_text"]
                resp_emb = embed_adapter.embed_text(resp_text)

                set_pmfs = []
                set_details = []

                for code in APPROVED_ANCHOR_SETS:
                    c_meta = anchor_data[code]
                    c_embs = anchor_embs[code]

                    # Step 5: Cosine similarities to the 5 anchors
                    cos_sims = cosine_similarities(resp_emb, c_embs)

                    # Step 6: Minimum subtraction and normalization -> individual PMF
                    pmf = similarities_to_pmf(cos_sims, epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE)
                    exp_set = expected_rating(pmf)

                    set_pmfs.append(pmf)
                    min_sim = float(np.min(cos_sims))

                    record_detail = {
                        "construct_code": code,
                        "construct_name": c_meta["name"],
                        "cosine_similarities": [round(float(s), 6) for s in cos_sims],
                        "min_similarity": round(min_sim, 6),
                        "pmf_probabilities": [round(float(p), 6) for p in pmf],
                        "expected_rating": round(float(exp_set), 4),
                        "anchor_statements": c_meta["statements"],
                        "anchor_labels": c_meta["labels"],
                    }
                    set_details.append(record_detail)

                    inter_row = {
                        "persona_id": pid,
                        "product_name": product.name,
                        "anchor_set_code": code,
                        "anchor_set_name": c_meta["name"],
                        "cos_sim_1": round(float(cos_sims[0]), 6),
                        "cos_sim_2": round(float(cos_sims[1]), 6),
                        "cos_sim_3": round(float(cos_sims[2]), 6),
                        "cos_sim_4": round(float(cos_sims[3]), 6),
                        "cos_sim_5": round(float(cos_sims[4]), 6),
                        "min_similarity": round(min_sim, 6),
                        "p1": round(float(pmf[0]), 6),
                        "p2": round(float(pmf[1]), 6),
                        "p3": round(float(pmf[2]), 6),
                        "p4": round(float(pmf[3]), 6),
                        "p5": round(float(pmf[4]), 6),
                        "set_expected_rating": round(float(exp_set), 4),
                        "embedding_model": "nomic-embed-text",
                    }
                    intermediate_records.append(inter_row)
                    inter_csv_rows.append(inter_row)

                    # Record for each independent construct
                    flat_records.append({
                        "persona_id": pid,
                        "product_name": product.name,
                        "construct_code": code,
                        "construct_name": c_meta["name"],
                        "is_primary_adoption_outcome": (code == "BI"),
                        "cosine_similarities": [round(float(s), 6) for s in cos_sims],
                        "pmf_probabilities": [round(float(p), 6) for p in pmf],
                        "expected_rating": round(float(exp_set), 4),
                        "embedding_model": "nomic-embed-text",
                    })

                # Construct scores mapping for this persona
                c_scores = {s["construct_code"]: s["expected_rating"] for s in set_details}
                bi_detail = next((s for s in set_details if s["construct_code"] == "BI"), set_details[-1])
                bi_rating = bi_detail["expected_rating"]
                bi_pmf = bi_detail["pmf_probabilities"]

                # NO cross-construct averaging! Each construct is preserved independently.
                ps_summary = {
                    "persona_id": pid,
                    "product_name": product.name,
                    "primary_adoption_rating": round(bi_rating, 4),
                    "primary_adoption_pmf": bi_pmf,
                    "construct_scores": c_scores,
                    "anchor_sets": set_details,
                }
                persona_summaries.append(ps_summary)

                for code_item in set_details:
                    summary_csv_rows.append({
                        "persona_id": pid,
                        "product_name": product.name,
                        "construct_code": code_item["construct_code"],
                        "construct_name": code_item["construct_name"],
                        "is_primary_outcome": (code_item["construct_code"] == "BI"),
                        "expected_rating": code_item["expected_rating"],
                        "p1": code_item["pmf_probabilities"][0],
                        "p2": code_item["pmf_probabilities"][1],
                        "p3": code_item["pmf_probabilities"][2],
                        "p4": code_item["pmf_probabilities"][3],
                        "p5": code_item["pmf_probabilities"][4],
                        "embedding_model": "nomic-embed-text",
                    })

            # Save full SSR json
            ssr_payload = {
                "evaluation_id": eval_id,
                "product_name": product.name,
                "methodology": "Option A: Independent Single-Set SSR for Six Constructs (Maier et al. adaptation)",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "embedding_model": "nomic-embed-text",
                "embedding_dimensions": 768,
                "embedding_notes": "Local Ollama nomic-embed-text (768 dims) used instead of paper's OpenAI text-embedding-3-small (1536 dims) for 100% offline, local execution.",
                "n_personas": len(persona_summaries),
                "n_constructs": 6,
                "anchors_per_construct": 5,
                "total_anchors": 30,
                "approved_constructs": ["PE", "BI", "EE", "HM", "SI", "PP"],
                "primary_adoption_construct": "BI",
                "supporting_diagnostic_constructs": ["PE", "EE", "HM", "SI", "PP"],
                "cross_construct_averaging": False,
                "n_evaluations": len(flat_records),
                "qualitative_constructs": [c for c in valid_constructs if c not in CONSTRUCT_CODE_MAP],
                "persona_summaries": persona_summaries,
                "records": flat_records,
                "intermediate_records": intermediate_records,
            }
            (ssr_dir / "ssr_results.json").write_text(json.dumps(ssr_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            # Save Summary CSV with per-construct rows
            with open(ssr_dir / "ssr_results.csv", mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "persona_id", "product_name", "construct_code", "construct_name",
                    "is_primary_outcome", "expected_rating", "p1", "p2", "p3", "p4", "p5", "embedding_model"
                ])
                writer.writeheader()
                writer.writerows(summary_csv_rows)

            # Save Intermediate Calculations CSV
            with open(ssr_dir / "ssr_intermediate_calculations.csv", mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "persona_id", "product_name", "anchor_set_code", "anchor_set_name",
                    "cos_sim_1", "cos_sim_2", "cos_sim_3", "cos_sim_4", "cos_sim_5", "min_similarity",
                    "p1", "p2", "p3", "p4", "p5", "set_expected_rating", "embedding_model"
                ])
                writer.writeheader()
                writer.writerows(inter_csv_rows)

            # Build comparison table without cross-construct averaging
            comparison = []
            for ps in persona_summaries:
                scores = ps.get("construct_scores", {})
                comparison.append({
                    "persona_id": ps["persona_id"],
                    "scores": scores,
                    "primary_adoption_rating": ps["primary_adoption_rating"],
                    "primary_adoption_pmf": ps["primary_adoption_pmf"],
                })

            eval_result = {
                "eval_id": eval_id,
                "concept": concept_dict,
                "responses": responses,
                "ssr_results": ssr_payload,
                "persona_summaries": persona_summaries,
                "comparison": comparison,
            }

            return {
                "ok": True,
                "eval_id": eval_id,
                "concept": concept_dict,
                "responses": responses,
                "ssr_results": ssr_payload,
                "persona_summaries": persona_summaries,
                "comparison": comparison,
                "evaluation": eval_result,
            }

    def export_data(self, eval_id: str, fmt: str = "json") -> tuple[bytes, str, str]:
        """Export evaluation data as JSON or CSV."""
        data = self.load_evaluation(eval_id)
        if not data:
            raise ValueError(f"Evaluation '{eval_id}' not found.")

        pname = slugify(data["concept"].get("name", "evaluation"))

        if fmt == "csv":
            ssr_csv = EVAL_BASE_DIR / eval_id / "ssr_results" / "ssr_results.csv"
            if ssr_csv.exists():
                return ssr_csv.read_bytes(), "text/csv; charset=utf-8", f"{pname}_ssr_summary.csv"
            # Fallback
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["persona_id", "expected_rating"])
            for ps in data.get("persona_summaries", []):
                w.writerow([ps.get("persona_id"), ps.get("expected_rating")])
            return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"{pname}_ssr_summary.csv"

        if fmt == "intermediate_csv":
            inter_csv = EVAL_BASE_DIR / eval_id / "ssr_results" / "ssr_intermediate_calculations.csv"
            if inter_csv.exists():
                return inter_csv.read_bytes(), "text/csv; charset=utf-8", f"{pname}_ssr_intermediate_calculations.csv"

        if fmt == "responses_csv":
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["persona_id", "product_name", "response_text", "model", "timestamp_utc"])
            for r in data.get("responses", []):
                w.writerow([
                    r.get("persona_id"),
                    r.get("product_name"),
                    r.get("response_text"),
                    r.get("model_returned"),
                    r.get("timestamp_utc"),
                ])
            return buf.getvalue().encode("utf-8"), "text/csv; charset=utf-8", f"{pname}_verbatim_responses.csv"

        # Default JSON
        return (
            json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            f"{pname}_evaluation_full.json",
        )
