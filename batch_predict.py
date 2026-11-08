"""
LEGACY (mobile-phone review rating; NOT the canonical SSR pipeline). This module's kNN example-bank scoring
and its direct-rating LLM mode are earlier-stage work kept for provenance of the baselines in data/. The canonical
SSR pipeline is canonical_ssr.py + scripts/ + dashboard/. See docs/project_status/legacy_direct_rating_paths.md.

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

Modes (--mode flag):
  deterministic : (default) existing embedding-based model
  vader         : VADER lexicon baseline (zero API cost)
  agentic       : LEGACY direct-rating mode: the LLM is asked for an integer 1-5 (NOT SSR despite the name;
                  runs VADER first as sanity check)
  all           : run all three sequentially
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

from agent import (
    SemanticRatingAgent,
    _cosine_sim,
    _softmax,
    _build_rating_uncertainty,
)

MEMORY_FILE    = "memory.json"
DEFAULT_OUTPUT = os.path.join("data", "predictions.csv")
DEFAULT_CSV    = os.path.join("data", "Mobile Reviews Sentiment.csv")


def predict_single(agent: SemanticRatingAgent, text: str, top_k: int = 3) -> dict:
    """
    Predict sentiment, Likert rating, and calibrated uncertainty metrics for one review text.
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
        bank_sims = np.array([_cosine_sim(vec, np.array(e["embedding"])) for e in example_bank])
        k_val = min(top_k, len(example_bank))
        top_idx = np.argsort(-bank_sims)[:k_val]
        weights = _softmax(bank_sims[top_idx], temp=0.1)

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

    # ── 3. Calibrated Uncertainty Metrics ─────────────────────────────────────
    unc_info = _build_rating_uncertainty(pred_dist, level=0.80, margin_threshold=0.10)
    p_int = unc_info["prediction_interval"]

    return {
        "predicted_sentiment":       top_sent,
        "predicted_likert_rating":   pred_likert,
        "expected_rating":           round(exp_rating, 4),
        "confidence":                unc_info["confidence"],
        "prediction_margin":         unc_info["prediction_margin"],
        "uncertainty_status":        unc_info["uncertainty_status"],
        "uncertainty_explanation":   unc_info["uncertainty_explanation"],
        "prediction_interval_lower": p_int["lower"],
        "prediction_interval_upper": p_int["upper"],
        "rating_distribution":       json.dumps(unc_info["rating_distribution"]),
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
        print(f"Note: Using column '{text_col}' as review_text.")

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
                "review_text":               review_text,
                "predicted_sentiment":       "Unknown",
                "predicted_likert_rating":   None,
                "expected_rating":           None,
                "confidence":                None,
                "prediction_margin":         None,
                "uncertainty_status":        "ambiguous",
                "uncertainty_explanation":   "Empty review text.",
                "prediction_interval_lower": 1,
                "prediction_interval_upper": 5,
                "rating_distribution":       "{}",
            })
            continue

        try:
            pred = predict_single(agent, review_text)
            results.append({
                "review_text":               review_text,
                "predicted_sentiment":       pred["predicted_sentiment"],
                "predicted_likert_rating":   pred["predicted_likert_rating"],
                "expected_rating":           pred["expected_rating"],
                "confidence":                pred["confidence"],
                "prediction_margin":         pred["prediction_margin"],
                "uncertainty_status":        pred["uncertainty_status"],
                "uncertainty_explanation":   pred["uncertainty_explanation"],
                "prediction_interval_lower": pred["prediction_interval_lower"],
                "prediction_interval_upper": pred["prediction_interval_upper"],
                "rating_distribution":       pred["rating_distribution"],
            })
        except Exception:
            results.append({
                "review_text":               review_text,
                "predicted_sentiment":       "Unknown",
                "predicted_likert_rating":   None,
                "expected_rating":           None,
                "confidence":                None,
                "prediction_margin":         None,
                "uncertainty_status":        "ambiguous",
                "uncertainty_explanation":   "Prediction failed.",
                "prediction_interval_lower": 1,
                "prediction_interval_upper": 5,
                "rating_distribution":       "{}",
            })

    print("Processing complete.", flush=True)

    # Save output predictions CSV
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    output_columns = [
        "review_text",
        "predicted_sentiment",
        "predicted_likert_rating",
        "expected_rating",
        "confidence",
        "prediction_margin",
        "uncertainty_status",
        "uncertainty_explanation",
        "prediction_interval_lower",
        "prediction_interval_upper",
        "rating_distribution",
    ]

    df_out = pd.DataFrame(results, columns=output_columns)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# VADER Lexicon Baseline Batch Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_vader_batch(
    input_csv: str,
    actual_csv: str = DEFAULT_CSV,
    output_csv: str = os.path.join("data", "vader_predictions.csv"),
    sample_n: int = 0,
    seed: int = 42,
) -> None:
    """Run VADER baseline over the eval set. Zero API calls."""
    from agent import predict_rating_vader

    print("=" * 60)
    print("  VADER LEXICON BASELINE")
    print("=" * 60)

    # Load eval set
    df_eval = _load_eval_set(input_csv, actual_csv)
    if sample_n > 0 and sample_n < len(df_eval):
        df_eval = df_eval.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {sample_n} rows (seed={seed})")

    total = len(df_eval)
    results = []

    for i, row in df_eval.iterrows():
        review_text = str(row["review_text"])
        actual_rating = row.get("actual_rating", None)

        pred_rating = predict_rating_vader(review_text)
        results.append({
            "review_text": review_text,
            "actual_rating": actual_rating,
            "pred_vader": pred_rating,
        })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [VADER] Processing {i + 1} / {total}", flush=True)

    print("  [VADER] Processing complete.", flush=True)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"  [VADER] Saved to {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Agentic LLM Batch Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_agentic_batch(
    input_csv: str,
    actual_csv: str = DEFAULT_CSV,
    output_csv: str = os.path.join("data", "agentic_predictions.csv"),
    provider: str = "groq",
    model: Optional[str] = None,
    host: str = "http://localhost:11434",
    n_samples: int = 5,
    temperature: float = 0.8,
    sample_n: int = 0,
    seed: int = 42,
) -> None:
    """LEGACY direct-rating baseline (NOT canonical SSR): asks the LLM for an integer rating per review.

    Despite the earlier "SSR sampling" wording, no embedding/anchor scoring is used here. Behaviour and outputs unchanged.
    """
    print("[LEGACY] agentic mode asks the LLM for a numeric rating; it is not the canonical SSR pipeline.", file=sys.stderr)
    from agent import predict_rating_agentic, select_anchor_examples, DEFAULT_GROQ_MODEL, DEFAULT_OLLAMA_MODEL

    target_model = model or (DEFAULT_GROQ_MODEL if provider.lower() == "groq" else DEFAULT_OLLAMA_MODEL)

    print("=" * 60)
    print("  AGENTIC LLM RATING PREDICTION")
    print(f"  Provider: {provider.upper()}  |  Model: {target_model}")
    print(f"  Samples/review: {n_samples}  |  Temperature: {temperature}")
    print("=" * 60)

    # Load eval set
    df_eval = _load_eval_set(input_csv, actual_csv)
    if sample_n > 0 and sample_n < len(df_eval):
        df_eval = df_eval.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        print(f"  Subsampled to {sample_n} rows (seed={seed})")

    total = len(df_eval)

    # Select few-shot anchors from the training pool (held-out from eval set)
    eval_ids = set(range(39998, 50001))
    anchors = select_anchor_examples(
        csv_path=actual_csv,
        eval_review_ids=eval_ids,
        n_per_rating=5,
        seed=seed,
    )
    anchor_ids = {a["review_id"] for a in anchors}
    print(f"  Selected {len(anchors)} anchor examples ({len(anchor_ids)} unique IDs, excluded from eval)")
    print(f"  Anchor IDs: {sorted(anchor_ids)[:10]}...")

    # Check for existing checkpoint
    checkpoint_csv = output_csv + ".checkpoint.csv"
    start_idx = 0
    results = []
    total_fallbacks = 0

    if os.path.exists(checkpoint_csv):
        df_ckpt = pd.read_csv(checkpoint_csv, encoding="utf-8")
        start_idx = len(df_ckpt)
        results = df_ckpt.to_dict("records")
        print(f"  Resuming from checkpoint at row {start_idx}")

    for i in range(start_idx, total):
        row = df_eval.iloc[i]
        review_text = str(row["review_text"])
        actual_rating = row.get("actual_rating", None)

        if not review_text.strip():
            results.append({
                "review_text": review_text,
                "actual_rating": actual_rating,
                "pred_mean": None,
                "pred_mode": None,
                "pred_median": None,
                "pred_samples": "[]",
            })
        else:
            try:
                pred = predict_rating_agentic(
                    review_text=review_text,
                    anchor_examples=anchors,
                    n_samples=n_samples,
                    temperature=temperature,
                    provider=provider,
                    model=target_model,
                    host=host,
                )
                total_fallbacks += pred.get("fallback_count", 0)
                results.append({
                    "review_text": review_text,
                    "actual_rating": actual_rating,
                    "pred_mean": pred["mean"],
                    "pred_mode": pred["mode"],
                    "pred_median": pred["median"],
                    "pred_samples": json.dumps(pred["samples"]),
                })
            except Exception as e:
                print(f"    [ERROR] Row {i+1}: {e}", flush=True)
                results.append({
                    "review_text": review_text,
                    "actual_rating": actual_rating,
                    "pred_mean": None,
                    "pred_mode": None,
                    "pred_median": None,
                    "pred_samples": "[]",
                })

        if (i + 1) % 20 == 0 or (i + 1) == total:
            print(f"  [Agentic] Processing {i + 1} / {total}", flush=True)

        # Checkpoint every 50 rows
        if (i + 1) % 50 == 0:
            _save_csv(results, checkpoint_csv)

    print("  [Agentic] Processing complete.", flush=True)

    # Calculate and report retry-fallback rate
    total_samples_collected = total * n_samples
    fallback_rate = (total_fallbacks / total_samples_collected * 100.0) if total_samples_collected > 0 else 0.0
    print(f"\n  [Reliability Check] Fallback extraction count: {total_fallbacks} / {total_samples_collected} samples ({fallback_rate:.2f}%)")
    if fallback_rate > 5.0:
        print(f"  ⚠ WARNING: Fallback parsing rate is {fallback_rate:.2f}% (>5.0%). Review raw LLM responses before trusting results.")
    else:
        print(f"  ✓ Fallback parsing rate {fallback_rate:.2f}% is within acceptable tolerance (<=5.0%).")

    # Save final output and remove checkpoint
    _save_csv(results, output_csv)
    if os.path.exists(checkpoint_csv):
        os.remove(checkpoint_csv)
    print(f"  [Agentic] Saved to {output_csv}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_eval_set(input_csv: str, actual_csv: str = DEFAULT_CSV) -> pd.DataFrame:
    """
    Load the 10,003-row eval set.

    The eval set is the same one used for the deterministic model:
    - reviews_only.csv (review_text only) matched against actual_reviews.xlsx
      for ground-truth ratings.
    - If the input CSV has 'rating' column, use it directly.
    - Otherwise, merge with the actual_csv via review_text.
    """
    df_in = pd.read_csv(input_csv, encoding="utf-8")

    if "review_text" not in df_in.columns:
        df_in = df_in.rename(columns={df_in.columns[0]: "review_text"})

    # If actual ratings already present, return as-is
    if "rating" in df_in.columns:
        df_in = df_in.rename(columns={"rating": "actual_rating"})
        return df_in

    # Try to load actual ratings from actual_reviews.xlsx
    actual_path = os.path.join(os.path.dirname(actual_csv), "actual_reviews.xlsx")
    if os.path.exists(actual_path):
        cols_25 = [
            "review_id", "name", "age", "brand", "model", "price_usd",
            "price_local", "currency", "exchange_rate_to_usd", "rating",
            "review_text", "sentiment", "country", "language", "date",
            "verified_purchase", "battery_life_rating", "camera_rating",
            "performance_rating", "design_rating", "display_rating",
            "review_length", "word_count", "helpful_votes", "source"
        ]
        df_actual = pd.read_excel(actual_path, header=None)
        df_actual.columns = cols_25

        # Row-by-row alignment (reviews_only.csv is identical ordering to actual_reviews.xlsx)
        if len(df_in) == len(df_actual):
            df_in["actual_rating"] = df_actual["rating"].values
        else:
            # Fallback: merge on review_text
            df_in["_row_idx"] = range(len(df_in))
            df_actual["_row_idx"] = range(len(df_actual))
            merged = pd.merge(df_in, df_actual[["_row_idx", "rating"]], on="_row_idx", how="left")
            merged = merged.rename(columns={"rating": "actual_rating"})
            merged = merged.drop(columns=["_row_idx"])
            return merged
    else:
        print(f"  Warning: {actual_path} not found. Actual ratings will be missing.")
        df_in["actual_rating"] = None

    return df_in


def _save_csv(results: list, path: str) -> None:
    """Save results list to CSV."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(path, index=False, encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

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
    # ── New args for agentic/vader modes ──────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["deterministic", "vader", "agentic", "all"],
        default="deterministic",
        help="Prediction mode (default: deterministic).",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "ollama"],
        default="groq",
        help="LLM provider for agentic mode (default: groq).",
    )
    parser.add_argument(
        "--agentic-model",
        default=None,
        help="Model identifier for agentic prediction (default: llama-3.1-8b-instant for Groq, llama3.2:3b for Ollama).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of LLM samples per review for agentic mode (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="LLM temperature for agentic mode (default: 0.8).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Subsample N rows from eval set for cost control (default: 0 = all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling and anchor selection (default: 42).",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host URL (only used when provider=ollama, default: http://localhost:11434).",
    )
    parser.add_argument(
        "--actual-csv",
        default=DEFAULT_CSV,
        help=f"Path to full CSV for anchor selection (default: {DEFAULT_CSV}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_file = args.input_path or args.positional_input

    if not input_file:
        # Default to reviews_only.csv for vader/agentic modes
        if args.mode in ("vader", "agentic", "all"):
            input_file = os.path.join("data", "reviews_only.csv")
            if not os.path.exists(input_file):
                print(f"Usage: python batch_predict.py --input <path_to_reviews.csv> --mode {args.mode}")
                sys.exit(1)
        else:
            print("Usage: python batch_predict.py --input <path_to_reviews.csv>")
            sys.exit(1)

    mode = args.mode

    if mode == "deterministic":
        run_batch(input_csv=input_file, output_csv=args.output_path)

    elif mode == "vader":
        run_vader_batch(
            input_csv=input_file,
            actual_csv=args.actual_csv,
            sample_n=args.sample,
            seed=args.seed,
        )

    elif mode == "agentic":
        # Always run VADER first as a free sanity check
        print("\n── Running VADER baseline first (sanity check) ──\n")
        run_vader_batch(
            input_csv=input_file,
            actual_csv=args.actual_csv,
            sample_n=args.sample,
            seed=args.seed,
        )
        print(f"\n── Now running Agentic LLM prediction via {args.provider.upper()} ──\n")
        run_agentic_batch(
            input_csv=input_file,
            actual_csv=args.actual_csv,
            provider=args.provider,
            model=args.agentic_model,
            host=args.host,
            n_samples=args.n_samples,
            temperature=args.temperature,
            sample_n=args.sample,
            seed=args.seed,
        )

    elif mode == "all":
        print("\n── Running Deterministic model ──\n")
        run_batch(input_csv=input_file, output_csv=args.output_path)
        print("\n── Running VADER baseline ──\n")
        run_vader_batch(
            input_csv=input_file,
            actual_csv=args.actual_csv,
            sample_n=args.sample,
            seed=args.seed,
        )
        print(f"\n── Running Agentic LLM prediction via {args.provider.upper()} ──\n")
        run_agentic_batch(
            input_csv=input_file,
            actual_csv=args.actual_csv,
            provider=args.provider,
            model=args.agentic_model,
            host=args.host,
            n_samples=args.n_samples,
            temperature=args.temperature,
            sample_n=args.sample,
            seed=args.seed,
        )



