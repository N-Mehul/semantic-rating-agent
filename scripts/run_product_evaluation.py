"""Product Evaluation Response Generator (Step 4 of Digital Persona Pipeline).

Evaluates the hypothetical 'Nova AI Smartphone' concept across the four empirical personas
(P1, P2, P3, P4) using local Qwen 2.5 7B Instruct through Ollama.

METHODOLOGICAL PROTECTIONS:
- Textual consumer critiques only.
- Strict prohibition of numerical ratings, Likert scores, percentages, and probabilities.
- Rejection of synthetic fabrication: if generation fails, reports the actual error.
- Canonical SSR calculations are NOT run in this step.

Outputs written to: data/product_evaluation/nova_ai_smartphone/
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Final

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import ollama_utils
from scripts.product_evaluation_prompts import (
    ProductConcept,
    ProductFeature,
    build_product_evaluation_prompt,
    validate_prompt_for_ssr,
)

OUT_DIR: Final[Path] = ROOT / "data" / "product_evaluation" / "nova_ai_smartphone"
MODEL_NAME: Final[str] = "qwen2.5:7b-instruct"
TEMPERATURE: Final[float] = 0.7  # Moderate temperature for creative yet grounded persona voice
TIMEOUT_S: Final[float] = 180.0

PERSONA_IDS: Final[list[str]] = [
    "P1_premium_tier_owner",
    "P2_midrange_heavy_user",
    "P3_midrange_moderate_user",
    "P4_budget_or_undisclosed_price",
]

EVAL_CONSTRUCTS: Final[list[str]] = ["PE", "BI", "EE", "HM", "SI", "PP", "PV"]

# Clearly labelled hypothetical product concept
NOVA_AI_SMARTPHONE: Final[ProductConcept] = ProductConcept(
    name="Nova AI Smartphone",
    description="A hypothetical mid-range smartphone designed for students and young professionals, with AI-powered features. (Hypothetical test concept; not an existing market product).",
    price_inr=24999.0,
    target_group="Students and young professionals",
    brand_or_ecosystem="Android ecosystem",
    features=[
        ProductFeature(
            name="AI Personal Assistant",
            explanation="Daily tasks, schedule reminders, smart notifications, and study support.",
            category="Software & Productivity",
        ),
        ProductFeature(
            name="AI Camera Enhancement",
            explanation="Automated low-light and portrait enhancement for photos and short-form videos.",
            category="Camera & Media",
        ),
        ProductFeature(
            name="AI Battery Optimisation",
            explanation="Smart power management that adapts background consumption to daily usage patterns.",
            category="Battery & Power",
        ),
        ProductFeature(
            name="On-Device AI Processing",
            explanation="Selected AI tasks processed locally on device to reduce the need to transmit personal data to cloud services.",
            category="Privacy & Security",
        ),
        ProductFeature(
            name="Smooth Performance",
            explanation="Optimized processor and high-refresh display for multitasking, casual gaming, and entertainment.",
            category="Hardware & Performance",
        ),
        ProductFeature(
            name="Simple Interface with AI Suggestions",
            explanation="Clean UI with contextual shortcuts and suggestions based on time and location.",
            category="User Interface",
        ),
    ],
)

RATING_LEAK_CHECK: Final[re.Pattern] = re.compile(
    r"\b[1-5]\s*/\s*5\b|\bout of 5\b|\bout of five\b|\brating:\s*\d|\bscore:\s*\d|\b\d{1,3}%\b|\bprobability:\s*\d",
    re.IGNORECASE,
)


def run_evaluation(
    model: str = MODEL_NAME,
    out_dir: Path = OUT_DIR,
    product: ProductConcept = NOVA_AI_SMARTPHONE,
    personas: list[str] = PERSONA_IDS,
    constructs: list[str] = EVAL_CONSTRUCTS,
    timeout: float = TIMEOUT_S,
) -> dict[str, Any]:
    """Execute persona-based product evaluation using local Ollama model."""
    print(f"=== Running Product Evaluation: {product.name} with {model} ===")

    # 1. Probe Ollama server
    probe_result = ollama_utils.probe(timeout=5.0)
    if not probe_result.get("running"):
        raise RuntimeError(f"Ollama server is not running: {probe_result.get('error')}")
    installed_models = probe_result.get("models", [])
    if model not in installed_models and f"{model}:latest" not in installed_models:
        raise RuntimeError(
            f"Requested model '{model}' is not available in Ollama. "
            f"Installed models: {installed_models}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save product concept details
    product_json_path = out_dir / "product_concept.json"
    product_dict = {
        "name": product.name,
        "description": product.description,
        "price_inr": product.price_inr,
        "target_group": product.target_group,
        "brand_or_ecosystem": product.brand_or_ecosystem,
        "features": [
            {"name": f.name, "explanation": f.explanation, "category": f.category}
            for f in product.features
        ],
        "is_hypothetical_test_concept": True,
        "timestamp_created": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    product_json_path.write_text(json.dumps(product_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved product concept to: {product_json_path}")

    # 3. Generate responses persona by persona
    responses = []
    errors = []

    for pid in personas:
        print(f"\n[Generating for Persona: {pid}]...")
        prompt = build_product_evaluation_prompt(
            product=product,
            persona_id_or_dict=pid,
            constructs=constructs,
            root=ROOT,
        )

        prompt_issues = validate_prompt_for_ssr(prompt)
        if prompt_issues:
            err = f"Prompt validation failed for {pid}: {prompt_issues}"
            print(f"ERROR: {err}")
            errors.append({"persona_id": pid, "error": err, "stage": "prompt_validation"})
            continue

        # Invoke Ollama
        gen_prompt = {
            "system": prompt["system"],
            "user": prompt["user"],
            "timeout_s": timeout,
        }

        # Override options in ollama_utils via env/settings if needed
        ollama_utils.SETTINGS["temperature"] = TEMPERATURE

        rec, err = ollama_utils.generate(model, gen_prompt, timeout=timeout)

        if err is not None:
            err_msg = f"Generation failed for {pid}: {err.get('kind')} - {err.get('provider', {}).get('message')}"
            print(f"ERROR: {err_msg}")
            errors.append({"persona_id": pid, "error": err, "stage": "ollama_generation"})
            continue

        resp_text = rec["response_text"].strip()

        # Check for rating leakage
        has_rating_leak = bool(RATING_LEAK_CHECK.search(resp_text))

        resp_record = {
            "persona_id": pid,
            "product_name": product.name,
            "response_text": resp_text,
            "model_requested": rec.get("model_requested"),
            "model_returned": rec.get("model_returned"),
            "finish_reason": rec.get("finish_reason"),
            "timestamp_utc": rec.get("timestamp_utc"),
            "usage": rec.get("usage"),
            "contains_numeric_rating": has_rating_leak,
            "evaluated_constructs": constructs,
            "prompt_metadata": prompt["metadata"],
        }
        responses.append(resp_record)
        print(f"Success for {pid} ({len(resp_text)} characters, {len(resp_text.split())} words, finish={rec.get('finish_reason')})")

    # 4. Save results
    responses_json_path = out_dir / "persona_responses.json"
    responses_json_path.write_text(json.dumps(responses, indent=2, ensure_ascii=False), encoding="utf-8")

    responses_jsonl_path = out_dir / "persona_responses.jsonl"
    with open(responses_jsonl_path, "w", encoding="utf-8") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "evaluation_id": "nova_ai_smartphone_eval_01",
        "product_name": product.name,
        "is_hypothetical_test_concept": True,
        "model": model,
        "temperature": TEMPERATURE,
        "timeout_s": timeout,
        "personas_requested": personas,
        "n_personas_requested": len(personas),
        "n_responses_generated": len(responses),
        "n_errors": len(errors),
        "errors": errors,
        "constructs": constructs,
        "ssr_protection_verified": all(not r["contains_numeric_rating"] for r in responses),
        "timestamp_completed": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files_saved": [
            str(product_json_path.relative_to(ROOT)),
            str(responses_json_path.relative_to(ROOT)),
            str(responses_jsonl_path.relative_to(ROOT)),
        ],
    }
    manifest_path = out_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nManifest saved to: {manifest_path}")

    return {
        "manifest": manifest,
        "responses": responses,
        "errors": errors,
    }


def main() -> int:
    try:
        results = run_evaluation()
        print(f"\nCompleted: {results['manifest']['n_responses_generated']}/{results['manifest']['n_personas_requested']} responses generated.")
        if results["errors"]:
            print(f"Encountered {len(results['errors'])} errors.")
            return 1
        return 0
    except Exception as e:
        print(f"FATAL ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

