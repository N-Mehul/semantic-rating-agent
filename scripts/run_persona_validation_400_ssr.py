"""
Full SSR Evaluation on 400 Validated Synthetic Persona Responses.

Uses canonical_ssr.py, approved semantic anchors, k=3, T=1.0, and Ollama nomic-embed-text.
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Set project root and paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

DATASET_PATH = ROOT / "data" / "experiments" / "persona_validation_400" / "synthetic_responses.csv"
OUT_DIR = ROOT / "data" / "experiments" / "persona_validation_400" / "ssr_evaluation"


def run_full_ssr_evaluation():
    t_start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading responses from {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        responses: List[Dict[str, Any]] = list(reader)

    num_samples = len(responses)
    assert num_samples == 400, f"Expected 400 responses, got {num_samples}"
    print(f"Loaded {num_samples} responses.")

    # Initialize Ollama Adapter
    t_adapter_init_start = time.perf_counter()
    adapter = get_ollama_embedding_adapter("nomic-embed-text")
    probe = adapter.probe()
    if not probe.get("model_available"):
        raise RuntimeError(f"Ollama nomic-embed-text not available: {probe}")
    adapter_init_time = time.perf_counter() - t_adapter_init_start
    print(f"Adapter initialized in {adapter_init_time:.2f}s.")

    # Load and pre-embed semantic anchors for all 6 constructs
    t_anchors_start = time.perf_counter()
    construct_codes = ["PE", "EE", "HM", "SI", "PP", "BI"]
    anchor_dict: Dict[str, List[str]] = {}
    anchor_embeddings_dict: Dict[str, np.ndarray] = {}

    for code in construct_codes:
        ca = get_construct_anchors(code)
        anchors = list(ca.statements)
        anchor_dict[code] = anchors
        embeddings = adapter.embed_texts(anchors)
        anchor_embeddings_dict[code] = embeddings

    anchors_time = time.perf_counter() - t_anchors_start
    print(f"Anchors for 6 constructs embedded in {anchors_time:.2f}s.")

    # Batch embed responses
    t_resp_embed_start = time.perf_counter()
    response_texts = [r["Response_Text"] for r in responses]
    batch_size = 32
    response_embeddings_list = []
    for i in range(0, len(response_texts), batch_size):
        chunk = response_texts[i : i + batch_size]
        chunk_embeddings = adapter.embed_texts(chunk)
        response_embeddings_list.append(chunk_embeddings)
        print(f"  Embedded responses {i+1} to {min(i+batch_size, len(response_texts))} / {num_samples}")

    response_embeddings = np.vstack(response_embeddings_list)
    resp_embed_time = time.perf_counter() - t_resp_embed_start
    print(f"All 400 response embeddings computed in {resp_embed_time:.2f}s.")

    # Compute SSR ratings for each response
    t_scoring_start = time.perf_counter()
    scored_records: List[Dict[str, Any]] = []
    all_pmf_sums_valid = True
    errors: List[str] = []

    for idx, (resp, resp_vec) in enumerate(zip(responses, response_embeddings)):
        construct = resp["Construct"]
        if construct not in anchor_embeddings_dict:
            err = f"Unknown construct {construct} at index {idx}"
            errors.append(err)
            continue

        anch_vecs = anchor_embeddings_dict[construct]
        sims = cosine_similarities(resp_vec, anch_vecs)
        pmf = similarities_to_pmf(sims, epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE)
        pmf_sum = float(np.sum(pmf))

        if not np.isclose(pmf_sum, 1.0, atol=1e-5):
            all_pmf_sums_valid = False
            errors.append(f"Sample {resp['Sample_ID']} PMF does not sum to 1.0: {pmf_sum}")

        exp_val = expected_rating(pmf)
        pred_argmax = int(np.argmax(pmf)) + 1
        pred_round = int(np.clip(np.round(exp_val), 1, 5))

        rec = dict(resp)  # preserve all original fields
        rec.update(
            {
                "Expected_Rating": round(float(exp_val), 4),
                "Predicted_Rating_Argmax": pred_argmax,
                "Predicted_Rating_Rounded": pred_round,
                "P1": round(float(pmf[0]), 4),
                "P2": round(float(pmf[1]), 4),
                "P3": round(float(pmf[2]), 4),
                "P4": round(float(pmf[3]), 4),
                "P5": round(float(pmf[4]), 4),
                "PMF_Sum": round(pmf_sum, 6),
                "Sim_Anchor_1": round(float(sims[0]), 4),
                "Sim_Anchor_2": round(float(sims[1]), 4),
                "Sim_Anchor_3": round(float(sims[2]), 4),
                "Sim_Anchor_4": round(float(sims[3]), 4),
                "Sim_Anchor_5": round(float(sims[4]), 4),
            }
        )
        scored_records.append(rec)

    scoring_time = time.perf_counter() - t_scoring_start
    total_time = time.perf_counter() - t_start

    print(f"Scored {len(scored_records)} records in {scoring_time:.2f}s.")
    print(f"Total pipeline time: {total_time:.2f}s.")

    # 1. Save Full Scored CSV
    csv_out_path = OUT_DIR / "ssr_scored_responses.csv"
    fieldnames = list(scored_records[0].keys())
    with open(csv_out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_records)

    # 2. Save Full Scored JSON
    json_out_path = OUT_DIR / "ssr_scored_responses.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(scored_records, f, indent=2)

    # 3. Summaries by Persona
    persona_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"expected_ratings": [], "argmax_ratings": [], "rounded_ratings": [], "count": 0}
    )
    for r in scored_records:
        pid = r["Persona_ID"]
        pname = r["Persona_Name"]
        key = f"{pid}: {pname}"
        persona_stats[key]["count"] += 1
        persona_stats[key]["expected_ratings"].append(r["Expected_Rating"])
        persona_stats[key]["argmax_ratings"].append(r["Predicted_Rating_Argmax"])
        persona_stats[key]["rounded_ratings"].append(r["Predicted_Rating_Rounded"])

    persona_summary = {}
    persona_summary_rows = []
    for key, val in sorted(persona_stats.items()):
        exp_arr = np.array(val["expected_ratings"])
        arg_arr = np.array(val["argmax_ratings"])
        rnd_arr = np.array(val["rounded_ratings"])
        hist_argmax = {f"R{r}": int(np.sum(arg_arr == r)) for r in range(1, 6)}
        hist_rounded = {f"R{r}": int(np.sum(rnd_arr == r)) for r in range(1, 6)}
        stat_item = {
            "Persona": key,
            "Count": val["count"],
            "Mean_Expected_Rating": round(float(np.mean(exp_arr)), 4),
            "Std_Expected_Rating": round(float(np.std(exp_arr)), 4),
            "Median_Expected_Rating": round(float(np.median(exp_arr)), 4),
            "Min_Expected_Rating": round(float(np.min(exp_arr)), 4),
            "Max_Expected_Rating": round(float(np.max(exp_arr)), 4),
            "Argmax_Distribution": hist_argmax,
            "Rounded_Distribution": hist_rounded,
        }
        persona_summary[key] = stat_item
        row = dict(stat_item)
        for r in range(1, 6):
            row[f"Argmax_Count_{r}"] = hist_argmax[f"R{r}"]
            row[f"Rounded_Count_{r}"] = hist_rounded[f"R{r}"]
        del row["Argmax_Distribution"]
        del row["Rounded_Distribution"]
        persona_summary_rows.append(row)

    with open(OUT_DIR / "summary_by_persona.json", "w", encoding="utf-8") as f:
        json.dump(persona_summary, f, indent=2)

    with open(OUT_DIR / "summary_by_persona.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(persona_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(persona_summary_rows)

    # 4. Summaries by Construct
    construct_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"expected_ratings": [], "argmax_ratings": [], "rounded_ratings": [], "count": 0}
    )
    for r in scored_records:
        cid = r["Construct"]
        construct_stats[cid]["count"] += 1
        construct_stats[cid]["expected_ratings"].append(r["Expected_Rating"])
        construct_stats[cid]["argmax_ratings"].append(r["Predicted_Rating_Argmax"])
        construct_stats[cid]["rounded_ratings"].append(r["Predicted_Rating_Rounded"])

    construct_summary = {}
    construct_summary_rows = []
    for key in construct_codes:
        val = construct_stats[key]
        exp_arr = np.array(val["expected_ratings"])
        arg_arr = np.array(val["argmax_ratings"])
        rnd_arr = np.array(val["rounded_ratings"])
        hist_argmax = {f"R{r}": int(np.sum(arg_arr == r)) for r in range(1, 6)}
        hist_rounded = {f"R{r}": int(np.sum(rnd_arr == r)) for r in range(1, 6)}
        stat_item = {
            "Construct": key,
            "Count": val["count"],
            "Mean_Expected_Rating": round(float(np.mean(exp_arr)), 4),
            "Std_Expected_Rating": round(float(np.std(exp_arr)), 4),
            "Median_Expected_Rating": round(float(np.median(exp_arr)), 4),
            "Min_Expected_Rating": round(float(np.min(exp_arr)), 4),
            "Max_Expected_Rating": round(float(np.max(exp_arr)), 4),
            "Argmax_Distribution": hist_argmax,
            "Rounded_Distribution": hist_rounded,
        }
        construct_summary[key] = stat_item
        row = dict(stat_item)
        for r in range(1, 6):
            row[f"Argmax_Count_{r}"] = hist_argmax[f"R{r}"]
            row[f"Rounded_Count_{r}"] = hist_rounded[f"R{r}"]
        del row["Argmax_Distribution"]
        del row["Rounded_Distribution"]
        construct_summary_rows.append(row)

    with open(OUT_DIR / "summary_by_construct.json", "w", encoding="utf-8") as f:
        json.dump(construct_summary, f, indent=2)

    with open(OUT_DIR / "summary_by_construct.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(construct_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(construct_summary_rows)

    # 5. Summaries by Persona x Construct
    grid_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in scored_records:
        pid = r["Persona_ID"]
        cid = r["Construct"]
        grid_stats[pid][cid].append(r["Expected_Rating"])

    grid_summary_rows = []
    grid_summary_json = {}
    for pid in ["P1", "P2", "P3", "P4"]:
        row = {"Persona_ID": pid}
        grid_summary_json[pid] = {}
        for cid in construct_codes:
            vals = grid_stats[pid][cid]
            mean_v = round(float(np.mean(vals)), 4) if vals else None
            row[cid] = mean_v
            grid_summary_json[pid][cid] = {
                "count": len(vals),
                "mean_expected_rating": mean_v,
                "std_expected_rating": round(float(np.std(vals)), 4) if vals else None,
            }
        grid_summary_rows.append(row)

    with open(OUT_DIR / "summary_by_persona_construct.json", "w", encoding="utf-8") as f:
        json.dump(grid_summary_json, f, indent=2)

    with open(OUT_DIR / "summary_by_persona_construct.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Persona_ID"] + construct_codes)
        writer.writeheader()
        writer.writerows(grid_summary_rows)

    # 6. Overall Execution Summary
    exec_summary = {
        "status": "COMPLETED" if not errors else "COMPLETED_WITH_ERRORS",
        "dataset_input": str(DATASET_PATH),
        "total_responses": num_samples,
        "valid_scored_responses": len(scored_records),
        "model_name": "nomic-embed-text",
        "embedding_dim": int(response_embeddings.shape[1]),
        "all_pmf_sums_valid": all_pmf_sums_valid,
        "ssr_k_parameter": 3,
        "ssr_epsilon": PAPER_EPSILON,
        "ssr_temperature": PAPER_TEMPERATURE,
        "execution_time_seconds": {
            "adapter_init": round(adapter_init_time, 4),
            "anchor_embedding": round(anchors_time, 4),
            "response_batch_embedding": round(resp_embed_time, 4),
            "scoring": round(scoring_time, 4),
            "total": round(total_time, 4),
        },
        "average_time_per_response_seconds": round(total_time / num_samples, 4),
        "errors": errors,
    }

    with open(OUT_DIR / "execution_summary.json", "w", encoding="utf-8") as f:
        json.dump(exec_summary, f, indent=2)

    print("\nSSR Evaluation Complete!")
    print(f"Results saved to: {OUT_DIR}")
    print(f"Total time: {total_time:.2f}s | Errors: {len(errors)}")
    return exec_summary


if __name__ == "__main__":
    run_full_ssr_evaluation()
