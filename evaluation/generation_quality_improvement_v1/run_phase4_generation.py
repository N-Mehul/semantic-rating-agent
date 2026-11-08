"""Phase 4: generate the final 24 improved responses (4 personas x 6 constructs), Nova AI Smartphone, Qwen 2.5 7B.
One construct per call, rating-leak guard (max one corrective retry), metadata + quality flags. NO SSR scoring.
Output: a new versioned directory under data/product_evaluation/. Resumable: re-running continues the same directory.
"""
from __future__ import annotations
import csv, datetime as dt, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts.run_product_evaluation import NOVA_AI_SMARTPHONE as PRODUCT, PERSONA_IDS  # noqa: E402
from scripts import ollama_utils  # noqa: E402
import product_evaluation_prompts_v2 as v2  # noqa: E402

MODEL, TEMPERATURE = "qwen2.5:7b-instruct", 0.7
STATE = Path(__file__).resolve().parent / "phase4_output_dir.txt"


def out_dir() -> Path:
    if STATE.exists():
        return ROOT / STATE.read_text(encoding="utf-8").strip()
    d = ROOT / "data/product_evaluation" / f"nova_ai_smartphone_v3_improved_{dt.datetime.now():%Y%m%d_%H%M%S}"
    d.mkdir(parents=True, exist_ok=False)
    STATE.write_text(str(d.relative_to(ROOT)).replace("\\", "/"), encoding="utf-8")
    return d


def main() -> int:
    probe = ollama_utils.probe(env={"OLLAMA_MODEL": MODEL})
    if not probe["running"] or not probe["model_available"]:
        print("Ollama not ready:", probe["error"]); return 1
    d = out_dir(); rec_path = d / "generation_records.jsonl"
    done = {}
    if rec_path.exists():
        for line in rec_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); done[(r["persona_id"], r["construct"])] = r
    specs = {pid: v2.load_spec(pid) for pid in PERSONA_IDS}
    t_start = time.perf_counter()
    for pid in PERSONA_IDS:                       # existing persona order
        for code in v2.CONSTRUCT_ORDER:           # existing construct order PE, EE, HM, SI, PP, BI
            if (pid, code) in done:
                continue
            avoid = [v2.opening(done[(p, code)]["text"]) for p in PERSONA_IDS if (p, code) in done and done[(p, code)].get("text")]
            t0 = time.perf_counter()
            r = v2.generate_construct_response(ollama_utils.generate, MODEL, PRODUCT, pid, code, spec=specs[pid],
                                               avoid_openings=avoid, temperature=TEMPERATURE, settings=ollama_utils.SETTINGS)
            r["elapsed_total_s"] = round(time.perf_counter() - t0, 1)
            r["timestamp_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
            with rec_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[(pid, code)] = r
            print(f"{pid} {code}: {r['status']} calls={r['n_calls']} {r['elapsed_total_s']}s | {(r['text'] or '')[:90]}", flush=True)

    # ---- quality flags (after all 24 exist, so cross-persona repetition can be checked) ----
    rows = []
    for pid in PERSONA_IDS:
        for code in v2.CONSTRUCT_ORDER:
            r = done[(pid, code)]
            others = [done[(p, code)]["text"] for p in PERSONA_IDS if p != pid and done[(p, code)].get("text")]
            q = v2.quality_flags(r["text"], code, PRODUCT, specs[pid], other_texts=others)
            ctx = r["attempts"][-1]["context"]
            rows.append({"persona_id": pid, "construct": code, "respondent_spec_id": r["respondent_spec_id"], "status": r["status"],
                         "n_calls": r["n_calls"], "corrective_retry": r["n_calls"] == 2, "elapsed_total_s": r["elapsed_total_s"],
                         "prompt_tokens_actual": ctx.get("prompt_tokens_actual"), "output_tokens_actual": ctx.get("output_tokens_actual"),
                         "num_ctx": ctx["num_ctx"], "truncation_likely": ctx["truncation_likely"],
                         "first_attempt_leak": "|".join(r["attempts"][0]["rating_leak"]), "final_rating_leak": "|".join(q["rating_leak"]),
                         "issues": "|".join(q["issues"]), "n_issues": len(q["issues"]), "words": len((r["text"] or "").split()),
                         "response_text": r["text"] or "", "quality_flags_json": json.dumps(q, ensure_ascii=False)})
    with (d / "quality_flags.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # dashboard-compatible responses file (text only; no SSR)
    responses = [{"persona_id": pid, "product_name": PRODUCT.name, "respondent_spec_id": specs[pid].get("synthetic_id"),
                  "construct_responses": {c: done[(pid, c)]["text"] for c in v2.CONSTRUCT_ORDER},
                  "construct_status": {c: done[(pid, c)]["status"] for c in v2.CONSTRUCT_ORDER},
                  "model_returned": MODEL, "prompt_version": v2.PROMPT_VERSION_V3} for pid in PERSONA_IDS]
    (d / "persona_responses.json").write_text(json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "product_concept.json").write_text(json.dumps({"name": PRODUCT.name, "category": "Smartphone", "description": PRODUCT.description,
        "price_inr": PRODUCT.price_inr, "target_group": PRODUCT.target_group, "brand_or_ecosystem": PRODUCT.brand_or_ecosystem,
        "features": [{"name": f.name, "explanation": f.explanation, "category": f.category} for f in PRODUCT.features],
        "source": "scripts/run_product_evaluation.py::NOVA_AI_SMARTPHONE"}, indent=2, ensure_ascii=False), encoding="utf-8")
    recs = [done[(p, c)] for p in PERSONA_IDS for c in v2.CONSTRUCT_ORDER]
    src = (ROOT / "scripts/product_evaluation_prompts_v2.py").read_bytes()
    manifest = {
        "label": "Phase 4 improved generation (24 responses). TEXT ONLY - SSR not computed.",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "model": MODEL, "temperature": TEMPERATURE,
        "prompt_version": v2.PROMPT_VERSION_V3, "prompt_module_sha256": hashlib.sha256(src).hexdigest(),
        "design": "4 personas x 6 constructs, one construct per call, persona order then construct order",
        "personas": PERSONA_IDS, "constructs": list(v2.CONSTRUCT_ORDER),
        "respondent_specs": {p: specs[p].get("synthetic_id") for p in PERSONA_IDS}, "respondent_specs_source": str(v2.SPECS.relative_to(ROOT)),
        "excluded_generation_inputs": ["Q21 rebuy rating", "Q26 free text", "SSR anchors", "answer-scale labels"],
        "rating_leak_policy": "max one corrective regeneration; unresolved leaks flagged, not retried further",
        "ssr_settings_for_later_scoring_unchanged": {"method": "Improved SSR", "k": 3, "T": 0.75, "embedding_model": "nomic-embed-text",
                                                      "anchors": "docs/survey_instrument/smartphone_construct_anchor_statements.csv"},
        "summary": {"responses": len(recs), "with_text": sum(1 for r in recs if r["text"]),
                    "status_counts": {s: sum(1 for r in recs if r["status"] == s) for s in sorted({r["status"] for r in recs})},
                    "corrective_retries": sum(1 for r in recs if r["n_calls"] == 2), "llm_calls": sum(r["n_calls"] for r in recs),
                    "total_generation_s": round(sum(r["elapsed_total_s"] for r in recs), 1),
                    "mean_generation_s_per_response": round(sum(r["elapsed_total_s"] for r in recs) / len(recs), 1),
                    "wall_clock_this_run_s": round(time.perf_counter() - t_start, 1),
                    "responses_with_any_issue": sum(1 for x in rows if x["n_issues"]),
                    "issue_counts": {}},
        "files": ["generation_records.jsonl", "persona_responses.json", "quality_flags.csv", "product_concept.json", "generation_manifest.json"],
    }
    for x in rows:
        for i in filter(None, x["issues"].split("|")):
            manifest["summary"]["issue_counts"][i] = manifest["summary"]["issue_counts"].get(i, 0) + 1
    (d / "generation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=1)); print("OUTPUT", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
