"""
evaluate_predictions.py — Evaluation script for batch predictions.

Usage:
  python evaluate_predictions.py --predictions data/predictions.csv --actual data/actual_reviews.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ── UTF-8 encoding on Windows ────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats


def run_evaluation(predictions_path: str, actual_path: str, output_results_path: str) -> None:
    if not os.path.exists(predictions_path):
        print(f"ERROR: Predictions file not found: {predictions_path}")
        sys.exit(1)

    if not os.path.exists(actual_path):
        print(f"ERROR: Actual labels file not found: {actual_path}")
        sys.exit(1)

    def load_file(path: str, is_actual: bool = False) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        if ext in (".xlsx", ".xls"):
            if is_actual:
                cols_25 = [
                    "review_id",
                    "name",
                    "age",
                    "brand",
                    "model",
                    "price_usd",
                    "price_local",
                    "currency",
                    "exchange_rate_to_usd",
                    "rating",
                    "review_text",
                    "sentiment",
                    "country",
                    "language",
                    "date",
                    "verified_purchase",
                    "battery_life_rating",
                    "camera_rating",
                    "performance_rating",
                    "design_rating",
                    "display_rating",
                    "review_length",
                    "word_count",
                    "helpful_votes",
                    "source"
                ]
                df = pd.read_excel(path, header=None)
                if df.shape[1] == len(cols_25):
                    df.columns = cols_25
                else:
                    df.columns = [cols_25[i] if i < len(cols_25) else f"col_{i}" for i in range(df.shape[1])]
                return df
            else:
                return pd.read_excel(path)
        else:
            return pd.read_csv(path, encoding="utf-8")

    df_pred = load_file(predictions_path, is_actual=False)
    df_act = load_file(actual_path, is_actual=True)

    # Check required columns
    pred_cols = ["review_text", "predicted_sentiment", "predicted_likert_rating", "expected_rating"]
    for col in pred_cols:
        if col not in df_pred.columns:
            print(f"ERROR: Predictions file is missing required column: {col}")
            sys.exit(1)

    # Print detected columns from the actual file
    detected_cols = [str(c) for c in df_act.columns]
    print(f"Detected columns in actual file: {detected_cols}")

    act_text_col = None
    act_rating_col = None
    act_sent_col = None

    # Try exact matches first (case-insensitive)
    for col in df_act.columns:
        col_str = str(col).strip().lower()
        if col_str == "review_text":
            act_text_col = col
        elif col_str == "rating":
            act_rating_col = col
        elif col_str == "sentiment":
            act_sent_col = col

    # Fallback matches if not found exactly
    for col in df_act.columns:
        col_str = str(col).strip().lower()
        if not act_text_col:
            if "review" in col_str or "text" in col_str or "comment" in col_str:
                act_text_col = col
        if not act_rating_col:
            if "rating" in col_str or "score" in col_str or "likert" in col_str:
                act_rating_col = col
        if not act_sent_col:
            if "sentiment" in col_str or "label" in col_str or "class" in col_str:
                act_sent_col = col

    # If text column is still missing, fallback to the first column
    if not act_text_col and len(df_act.columns) > 0:
        act_text_col = df_act.columns[0]

    # Validate that all required columns are found
    missing_cols = []
    if act_text_col is None:
        missing_cols.append("review_text (text)")
    if act_rating_col is None:
        missing_cols.append("rating (numeric)")
    if act_sent_col is None:
        missing_cols.append("sentiment (class)")

    if missing_cols:
        print(f"ERROR: Could not map required columns: {', '.join(missing_cols)}")
        print(f"Detected columns in file: {detected_cols}")
        sys.exit(1)

    # ── 2. Alignment & Matching ───────────────────────────────────────────────
    # To handle duplicate review_text values cleanly without cartesian product,
    # we rank occurrences within each review_text group.
    df_pred_clean = df_pred.copy()
    df_act_clean = df_act.copy()

    df_pred_clean["occurrence_idx"] = df_pred_clean.groupby("review_text").cumcount()
    df_act_clean["occurrence_idx"] = df_act_clean.groupby(act_text_col).cumcount()

    # Rename actual columns for clarity before merge
    df_act_clean = df_act_clean.rename(columns={
        act_text_col: "review_text",
        act_rating_col: "actual_rating",
        act_sent_col: "actual_sentiment"
    })

    # Perform outer merge to find matched and unmatched rows
    df_merged = pd.merge(
        df_pred_clean,
        df_act_clean[["review_text", "occurrence_idx", "actual_rating", "actual_sentiment"]],
        on=["review_text", "occurrence_idx"],
        how="outer",
        indicator=True
    )

    matched = df_merged[df_merged["_merge"] == "both"]
    unmatched_pred = df_merged[df_merged["_merge"] == "left_only"]
    unmatched_act = df_merged[df_merged["_merge"] == "right_only"]

    n_pred_rows = len(df_pred)
    n_act_rows = len(df_act)
    n_matched = len(matched)
    n_unmatched_pred = len(unmatched_pred)
    n_unmatched_act = len(unmatched_act)
    n_dup_pred = df_pred.duplicated(subset=["review_text"]).sum()

    # ── 3. Calculate Metrics ──────────────────────────────────────────────────
    if n_matched == 0:
        print("ERROR: No matched reviews found between predictions and actual records.")
        sys.exit(1)

    # Convert columns to correct types
    matched_df = matched.copy()
    matched_df["actual_rating"] = matched_df["actual_rating"].astype(float)
    matched_df["predicted_likert_rating"] = matched_df["predicted_likert_rating"].astype(float)
    matched_df["expected_rating"] = matched_df["expected_rating"].astype(float)

    # Sentiment Metrics
    actual_sents = matched_df["actual_sentiment"].astype(str).str.strip()
    pred_sents = matched_df["predicted_sentiment"].astype(str).str.strip()
    sent_correct_mask = (actual_sents.str.lower() == pred_sents.str.lower())
    sent_correct = sent_correct_mask.sum()
    sent_incorrect = n_matched - sent_correct
    sent_accuracy = (sent_correct / n_matched) * 100.0

    # Rating Metrics
    actual_ratings = matched_df["actual_rating"].values
    pred_likert = matched_df["predicted_likert_rating"].values
    expected_ratings = matched_df["expected_rating"].values

    rating_correct_mask = (actual_ratings == pred_likert)
    rating_correct = rating_correct_mask.sum()
    rating_incorrect = n_matched - rating_correct
    rating_accuracy = (rating_correct / n_matched) * 100.0

    within_1_mask = (np.abs(pred_likert - actual_ratings) <= 1.0)
    within_1_correct = within_1_mask.sum()
    within_1_accuracy = (within_1_correct / n_matched) * 100.0

    mae = np.mean(np.abs(expected_ratings - actual_ratings))
    rmse = np.sqrt(np.mean((expected_ratings - actual_ratings) ** 2))

    if len(actual_ratings) >= 2 and np.std(expected_ratings) > 1e-9 and np.std(actual_ratings) > 1e-9:
        spearman_corr, _ = stats.spearmanr(actual_ratings, expected_ratings)
    else:
        spearman_corr = 0.0

    # Rating-Wise Accuracy (for actual ratings 1 to 5)
    rating_wise_acc = {}
    for r in range(1, 6):
        r_mask = (actual_ratings == float(r))
        r_total = r_mask.sum()
        if r_total > 0:
            r_correct = ((pred_likert == float(r)) & r_mask).sum()
            rating_wise_acc[r] = (r_correct / r_total) * 100.0
        else:
            rating_wise_acc[r] = 0.0

    # Confusion Matrix (5x5)
    conf_matrix = np.zeros((5, 5), dtype=int)
    for a, p in zip(actual_ratings, pred_likert):
        a_idx = int(a) - 1
        p_idx = int(p) - 1
        if 0 <= a_idx < 5 and 0 <= p_idx < 5:
            conf_matrix[p_idx, a_idx] += 1

    # ── 4. Print Concise Report ───────────────────────────────────────────────
    print("========================================")
    print("BATCH MODEL EVALUATION")
    print("========================================")
    print(f"Reviews predicted       : {n_pred_rows}")
    print(f"Successfully matched    : {n_matched}")
    print(f"Unmatched               : {n_unmatched_pred}")
    print()
    print("SENTIMENT")
    print(f"Accuracy                 : {sent_accuracy:.2f}%")
    print(f"Correct                  : {sent_correct}")
    print(f"Incorrect                : {sent_incorrect}")
    print()
    print("LIKERT RATING")
    print(f"Exact Accuracy           : {rating_accuracy:.2f}%")
    print(f"Within ±1 Accuracy       : {within_1_accuracy:.2f}%")
    print(f"MAE                      : {mae:.2f}")
    print(f"RMSE                     : {rmse:.2f}")
    print(f"Spearman Correlation     : {spearman_corr:.2f}")
    print()
    print("RATING-WISE ACCURACY")
    for r in range(1, 6):
        print(f"Rating {r}                 : {rating_wise_acc[r]:.2f}%")
    print()
    print("CONFUSION MATRIX")
    print()
    print("             Actual")
    print("Predicted    1    2    3    4    5")
    for p_val in range(1, 6):
        row_str = "  ".join(f"{conf_matrix[p_val-1, a_val-1]:>3}" for a_val in range(1, 6))
        print(f"{p_val:<12}{row_str}")
    print("========================================")

    # ── 5. Generate detailed evaluation CSV ───────────────────────────────────
    matched_df["rating_correct"] = rating_correct_mask
    matched_df["sentiment_correct"] = sent_correct_mask
    matched_df["rating_error"] = matched_df["predicted_likert_rating"] - matched_df["actual_rating"]

    df_eval_out = matched_df[[
        "review_text",
        "actual_rating",
        "predicted_likert_rating",
        "actual_sentiment",
        "predicted_sentiment",
        "rating_correct",
        "sentiment_correct",
        "rating_error"
    ]].rename(columns={
        "predicted_likert_rating": "predicted_rating"
    })

    out_dir = os.path.dirname(output_results_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_eval_out.to_csv(output_results_path, index=False, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate batch predictions against actual labels."
    )
    parser.add_argument(
        "--predictions", "-p",
        default="data/predictions.csv",
        help="Path to the predictions CSV generated by batch_predict.py",
    )
    parser.add_argument(
        "--actual", "-a",
        required=True,
        help="Path to the CSV containing actual rating and sentiment labels.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/evaluation_results.csv",
        help="Path to write the detailed evaluation results CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        predictions_path=args.predictions,
        actual_path=args.actual,
        output_results_path=args.output_csv
    )
