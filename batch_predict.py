"""
batch_predict.py — Batch Prediction on Separate Unseen Reviews CSV.

Requirements:
  1. Load the already-trained knowledge from memory.json (no retraining, no training CSV access).
  2. Use the SAME prediction method currently implemented in agent.py.
  3. Predict for every review:
     - predicted_sentiment
     - predicted_likert_rating
     - expected_rating
  4. Terminal shows only:
     Processing 1 / N
     Processing 2 / N
     ...
     Processing complete.
  5. Save output to data/predictions.csv with columns:
     review_text, predicted_sentiment, predicted_likert_rating, expected_rating
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr

warnings.filterwarnings("ignore")

# ── UTF-8 encoding on Windows ────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from agent import SemanticRatingAgent, _cosine_sim, _softmax

MEMORY_FILE    = "memory.json"
DEFAULT_OUTPUT = os.path.join("data", "predictions.csv")


def predict_single(agent: SemanticRatingAgent, text: str) -> dict:
    """
    Predict sentiment and Likert rating for one review text.
    Exact implementation matching agent.py's prediction logic.
    """
    patterns        = agent.memory.get("text_rating_patterns", {})
    example_bank    = patterns.get("example_bank", [])
    rating_profiles = patterns.get("rating_profiles", {})
    sent_profiles   = patterns.get("sentiment_profiles", {})
    target_info     = agent.memory.get("target_analysis", {})

    # Embed review text using the sentence-transformers model
    vec = agent._embed([text])[0]

    # ── 1. Sentiment Prediction ──────────────────────────────────────────────
    top_sent = "Unknown"
    if sent_profiles:
        sent_sims = {
            sv: _cosine_sim(vec, np.array(prof["centroid"]))
            for sv, prof in sent_profiles.items()
        }
        sorted_sents = sorted(sent_sims.items(), key=lambda x: -x[1])
        top_sent = sorted_sents[0][0] if sorted_sents else "Unknown"

    # ── 2. Likert Rating Prediction ──────────────────────────────────────────
    rating_vals = target_info.get("rating", {}).get("unique_values", [1.0, 2.0, 3.0, 4.0, 5.0])
    rating_keys = [
        str(int(v)) if float(v).is_integer() else str(v)
        for v in sorted(rating_vals)
    ]
    pred_dist: dict[str, float] = {k: 0.0 for k in rating_keys}

    if example_bank:
        sims = np.array([_cosine_sim(vec, np.array(e["embedding"])) for e in example_bank])
        top_k = min(5, len(example_bank))
        top_idx = np.argsort(-sims)[:top_k]
        weights = _softmax(sims[top_idx], temp=0.1)

        for w, idx in zip(weights, top_idx):
            e_dist = example_bank[idx].get("rating_distribution", {})
            for rk in rating_keys:
                p = float(e_dist.get(rk, e_dist.get(str(float(rk)), 0.0)))
                pred_dist[rk] += float(w * p)

        total = sum(pred_dist.values())
        if total > 0:
            pred_dist = {k: v / total for k, v in pred_dist.items()}
        else:
            pred_dist = {k: 1.0 / len(pred_dist) for k in pred_dist}

    elif rating_profiles:
        sims = {
            rk: _cosine_sim(vec, np.array(rating_profiles[rk]["centroid"]))
            for rk in rating_keys if rk in rating_profiles
        }
        w = _softmax(np.array(list(sims.values())), temp=0.1)
        for k, val in zip(sims.keys(), w):
            pred_dist[k] = float(val)

    exp_rating = sum(float(k) * pred_dist[k] for k in rating_keys)
    best_rating_k = max(pred_dist.items(), key=lambda x: x[1])[0]
    best_rating_v = float(best_rating_k)
    pred_likert = int(best_rating_v) if best_rating_v.is_integer() else best_rating_v

    return {
        "predicted_sentiment":     top_sent,
        "predicted_likert_rating": pred_likert,
        "expected_rating":         round(exp_rating, 4),
    }


def run_batch(input_csv: str, output_csv: str = DEFAULT_OUTPUT) -> None:
    if not os.path.exists(MEMORY_FILE):
        print(f"ERROR: {MEMORY_FILE} not found. Please ensure memory.json exists.")
        sys.exit(1)

    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        sys.exit(1)

    # Read input CSV
    df_in = pd.read_csv(input_csv, encoding="utf-8")

    # Locate review text column (accept 'review_text' or first column)
    if "review_text" in df_in.columns:
        text_col = "review_text"
    else:
        text_col = df_in.columns[0]

    reviews = df_in[text_col].fillna("").astype(str).tolist()
    total = len(reviews)

    # Load agent memory strictly from memory.json without reading or retraining
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        loaded_memory = json.load(f)

    agent = SemanticRatingAgent()
    agent.memory = loaded_memory

    # Initialize the embedding model quietly
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        agent._get_model()

    results = []

    for i, review_text in enumerate(reviews, start=1):
        print(f"Processing {i} / {total}", flush=True)

        if not review_text.strip():
            results.append({
                "review_text":             review_text,
                "predicted_sentiment":     "Unknown",
                "predicted_likert_rating": None,
                "expected_rating":         None,
            })
            continue

        try:
            pred = predict_single(agent, review_text)
            results.append({
                "review_text":             review_text,
                "predicted_sentiment":     pred["predicted_sentiment"],
                "predicted_likert_rating": pred["predicted_likert_rating"],
                "expected_rating":         pred["expected_rating"],
            })
        except Exception:
            results.append({
                "review_text":             review_text,
                "predicted_sentiment":     "Unknown",
                "predicted_likert_rating": None,
                "expected_rating":         None,
            })

    print("Processing complete.", flush=True)

    # Save output predictions CSV
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(results, columns=[
        "review_text",
        "predicted_sentiment",
        "predicted_likert_rating",
        "expected_rating",
    ])
    df_out.to_csv(output_csv, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch prediction on a CSV of review texts using memory.json knowledge."
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_path",
        default=None,
        help="Path to the input CSV containing 'review_text'.",
    )
    parser.add_argument(
        "positional_input",
        nargs="?",
        default=None,
        help="Path to the input CSV (positional argument alternative).",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_path",
        default=DEFAULT_OUTPUT,
        help=f"Path to save output CSV (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_path or args.positional_input

    if not input_file:
        print("Usage: python batch_predict.py --input <path_to_reviews.csv>")
        sys.exit(1)

    run_batch(input_csv=input_file, output_csv=args.output_path)
