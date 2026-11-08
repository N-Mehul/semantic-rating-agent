"""
Quick test of the canonical SSR pipeline on 10 randomly selected responses.
"""

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "data" / "experiments" / "persona_validation_400" / "synthetic_responses.csv"
OUT_DIR = ROOT / "data" / "experiments" / "persona_validation_400" / "ssr_evaluation" / "test_10_responses"

from canonical_ssr import (
    cosine_similarities,
    similarities_to_pmf,
    expected_rating,
    PAPER_EPSILON,
    PAPER_TEMPERATURE,
    RATINGS,
)
from scripts.ollama_embedding_adapter import get_ollama_embedding_adapter
from survey_ssr_bridge import get_construct_anchors


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # Load responses
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        all_responses = list(csv.DictReader(f))

    assert len(all_responses) == 400, f"Expected 400 responses, got {len(all_responses)}"

    # Randomly select 10 responses with fixed seed for reproducibility
    random.seed(20261102)
    selected_indices = sorted(random.sample(range(len(all_responses)), 10))
    selected_samples = [all_responses[i] for i in selected_indices]

    # Initialize Ollama embedding adapter
    embed_init_start = time.perf_counter()
    adapter = get_ollama_embedding_adapter("nomic-embed-text")
    probe = adapter.probe()
    if not probe.get("model_available"):
        raise RuntimeError(f"Ollama nomic-embed-text not available: {probe}")
    embed_init_time = time.perf_counter() - embed_init_start

    # Pre-embed canonical anchors for all 6 constructs
    anchor_embed_start = time.perf_counter()
    construct_anchors = {}
    construct_anchor_embs = {}
    for c_code in ["PE", "EE", "HM", "SI", "PP", "BI"]:
        a = get_construct_anchors(c_code)
        construct_anchors[c_code] = a
        construct_anchor_embs[c_code] = adapter.embed_texts(list(a.statements))
    anchor_embed_time = time.perf_counter() - anchor_embed_start

    # Evaluate the 10 selected samples
    eval_records = []
    inference_times = []

    for idx, s in enumerate(selected_samples, 1):
        t0 = time.perf_counter()
        sid = s["Sample_ID"]
        pid = s["Persona_ID"]
        pname = s["Persona_Name"]
        construct = s["Construct"]
        text = s["Response_Text"]

        # Embed response text
        resp_emb = adapter.embed_text(text)

        # Get construct anchors
        a_embs = construct_anchor_embs[construct]

        # Canonical cosine similarities
        sims = cosine_similarities(resp_emb, a_embs)

        # Canonical min-subtraction & level-3 tie breaking
        pmf = similarities_to_pmf(sims, epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE)

        # Verification of PMF sum = 1.0
        pmf_sum = float(np.sum(pmf))
        assert np.isclose(pmf_sum, 1.0, atol=1e-5), f"PMF sum {pmf_sum} is not 1.0 for {sid}"

        # Continuous expected rating
        exp_r = expected_rating(pmf)

        # Discrete predicted rating (argmax and rounded expected)
        argmax_r = int(np.argmax(pmf) + 1)
        rounded_r = int(np.round(exp_r))

        sample_time = time.perf_counter() - t0
        inference_times.append(sample_time)

        eval_records.append({
            "Test_Index": idx,
            "Sample_ID": sid,
            "Persona_ID": pid,
            "Persona_Name": pname,
            "Construct": construct,
            "Expected_Rating": round(float(exp_r), 4),
            "Predicted_Rating_Argmax": argmax_r,
            "Predicted_Rating_Rounded": rounded_r,
            "P1": round(float(pmf[0]), 4),
            "P2": round(float(pmf[1]), 4),
            "P3": round(float(pmf[2]), 4),
            "P4": round(float(pmf[3]), 4),
            "P5": round(float(pmf[4]), 4),
            "PMF_Sum": round(pmf_sum, 6),
            "Latency_Seconds": round(sample_time, 4),
            "Response_Excerpt": text[:100] + "...",
            "Full_Text": text
        })

    total_duration = time.perf_counter() - t_start

    # Save test results CSV
    csv_path = OUT_DIR / "test_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Test_Index", "Sample_ID", "Persona_ID", "Persona_Name", "Construct",
            "Expected_Rating", "Predicted_Rating_Argmax", "Predicted_Rating_Rounded",
            "P1", "P2", "P3", "P4", "P5", "PMF_Sum", "Latency_Seconds", "Response_Excerpt"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(eval_records)

    # Save detailed JSON
    json_path = OUT_DIR / "test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_records, f, indent=2)

    # Save summary metadata
    summary = {
        "test_name": "ssr_pipeline_speed_and_functionality_test",
        "sample_size": 10,
        "seed": 20261102,
        "selected_indices": selected_indices,
        "total_execution_time_seconds": round(total_duration, 4),
        "adapter_initialization_seconds": round(embed_init_time, 4),
        "anchor_embedding_seconds": round(anchor_embed_time, 4),
        "average_inference_time_per_sample_seconds": round(float(np.mean(inference_times)), 4),
        "min_inference_time_seconds": round(float(np.min(inference_times)), 4),
        "max_inference_time_seconds": round(float(np.max(inference_times)), 4),
        "all_pmf_sums_valid": all(np.isclose(r["PMF_Sum"], 1.0, atol=1e-5) for r in eval_records),
        "errors": []
    }
    with open(OUT_DIR / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("SUCCESS: Evaluated 10 responses successfully.")
    print(f"Total time: {total_duration:.2f}s (Avg per response: {np.mean(inference_times):.4f}s)")
    print(f"Outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
