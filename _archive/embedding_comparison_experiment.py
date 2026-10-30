"""
embedding_comparison_experiment.py
===============================
A standalone script that compares three sentence-embedding models for Likert rating prediction on the
Mobile Reviews Sentiment dataset:
  1. all-MiniLM-L6-v2 (MiniLM)
  2. all-mpnet-base-v2 (MPNet)
  3. BAAI/bge-base-en-v1.5 (BGE)

For each embedding model:
- The exact stratified 80/20 train/test split (random_state=42) is used.
- A downstream Logistic Regression model (class_weight="balanced", max_iter=1000, random_state=42) is trained.
- Evaluates:
    - Exact Accuracy (%)
    - Within +/-1 Accuracy (%)
    - MAE
    - RMSE
    - Spearman Correlation
    - Rating-wise Accuracy (1 to 5)
    - 5x5 Confusion Matrix

Outputs:
- Saves comparison table to `data/embedding_comparison_results.csv`.
- Prints comprehensive summary tables and full 5x5 confusion matrices.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List

# Suppress noisy warnings
warnings.filterwarnings("ignore")

# UTF-8 Output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

# Path constants
CSV_PATH = os.path.join("data", "Mobile Reviews Sentiment.csv")
RESULTS_CSV = os.path.join("data", "embedding_comparison_results.csv")


def evaluate_model(y_true: np.ndarray, pred_likert: np.ndarray, expected_rating: np.ndarray) -> Dict:
    """Compute all evaluation metrics for rating prediction."""
    n_samples = len(y_true)
    y_true = np.array(y_true, dtype=float)
    pred_likert = np.array(pred_likert, dtype=float)
    expected_rating = np.array(expected_rating, dtype=float)

    # Core metrics
    correct = np.sum(y_true == pred_likert)
    exact_acc = (correct / n_samples) * 100.0
    within_1_acc = (np.sum(np.abs(y_true - pred_likert) <= 1.0) / n_samples) * 100.0
    mae = np.mean(np.abs(expected_rating - y_true))
    rmse = np.sqrt(np.mean((expected_rating - y_true) ** 2))

    # Spearman correlation
    if n_samples >= 2 and np.std(expected_rating) > 1e-9 and np.std(y_true) > 1e-9:
        spearman_corr, _ = stats.spearmanr(y_true, expected_rating)
    else:
        spearman_corr = 0.0

    # Per-rating accuracy (1 to 5)
    rating_accs = {}
    for r in range(1, 6):
        mask = (y_true == float(r))
        total = mask.sum()
        if total > 0:
            correct_r = ((pred_likert == float(r)) & mask).sum()
            rating_accs[f"rating_{r}_acc"] = (correct_r / total) * 100.0
        else:
            rating_accs[f"rating_{r}_acc"] = 0.0

    # 5x5 Confusion Matrix (Row: Predicted, Column: Actual)
    conf_matrix = np.zeros((5, 5), dtype=int)
    for a, p in zip(y_true, pred_likert):
        a_idx = int(a) - 1
        p_idx = int(p) - 1
        if 0 <= a_idx < 5 and 0 <= p_idx < 5:
            conf_matrix[p_idx, a_idx] += 1

    return {
        "exact_accuracy": exact_acc,
        "within_1_accuracy": within_1_acc,
        "mae": mae,
        "rmse": rmse,
        "spearman_correlation": spearman_corr,
        "rating_wise_accuracy": rating_accs,
        "confusion_matrix": conf_matrix,
    }


def main() -> None:
    print("=" * 80)
    print("  EMBEDDING MODEL COMPARISON EXPERIMENT — SEMANTIC RATING AGENT")
    print("=" * 80)

    # 1. Load dataset
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at {CSV_PATH}")
        sys.exit(1)

    print("\n[Step 1/4] Loading and cleaning dataset...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    df_clean = df.dropna(how="all").reset_index(drop=True)
    text_col = "review_text"
    rating_col = "rating"
    pair = df_clean[[text_col, rating_col]].dropna(subset=[text_col, rating_col]).reset_index(drop=True)
    pair[rating_col] = pair[rating_col].astype(int)

    n_total = len(pair)
    unique_texts = pair[text_col].astype(str).unique().tolist()
    n_unique = len(unique_texts)
    print(f"  Informative rows containing text + rating: {n_total:,}")
    print(f"  Unique review texts                      : {n_unique:,}")

    # 2. Define embedding models
    embedding_models = {
        "MiniLM (all-MiniLM-L6-v2)": "sentence-transformers/all-MiniLM-L6-v2",
        "MPNet (all-mpnet-base-v2)": "sentence-transformers/all-mpnet-base-v2",
        "BGE (bge-base-en-v1.5)": "BAAI/bge-base-en-v1.5",
    }

    results: List[Dict] = []
    confusion_matrices: Dict[str, np.ndarray] = {}

    # 3. Iterate over models
    print("\n[Step 2/4] Generating embeddings, training models, and evaluating...")
    for short_name, model_id in embedding_models.items():
        print(f"\n--- Processing: {short_name} ---")
        print(f"  Loading sentence transformer: {model_id}...")
        embedder = SentenceTransformer(model_id)

        print(f"  Embedding {n_unique:,} unique review texts...")
        uniq_emb = embedder.encode(unique_texts, batch_size=64, show_progress_bar=False)
        text_to_emb = {txt: uniq_emb[i] for i, txt in enumerate(unique_texts)}

        print("  Mapping embeddings to full dataset...")
        X_full = np.stack(pair[text_col].astype(str).map(text_to_emb).values)
        y_full = pair[rating_col].values

        print(f"  Feature matrix shape: {X_full.shape}")

        # Stratified 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.20, stratify=y_full, random_state=42
        )
        print(f"  Train samples: {X_train.shape[0]:,} | Test samples: {X_test.shape[0]:,}")

        # Train downstream Logistic Regression
        print("  Fitting Logistic Regression classifier...")
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Predict
        probs = clf.predict_proba(X_test)
        pred_likert = clf.predict(X_test)
        expected_rating = np.dot(probs, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # Evaluate
        eval_metrics = evaluate_model(y_test, pred_likert, expected_rating)
        confusion_matrices[short_name] = eval_metrics["confusion_matrix"]

        result_row = {
            "Embedding Model": short_name,
            "Exact Accuracy (%)": round(eval_metrics["exact_accuracy"], 2),
            "Within ±1 Accuracy (%)": round(eval_metrics["within_1_accuracy"], 2),
            "MAE": round(eval_metrics["mae"], 4),
            "RMSE": round(eval_metrics["rmse"], 4),
            "Spearman Correlation": round(eval_metrics["spearman_correlation"], 4),
            "Rating 1 Acc (%)": round(eval_metrics["rating_wise_accuracy"]["rating_1_acc"], 2),
            "Rating 2 Acc (%)": round(eval_metrics["rating_wise_accuracy"]["rating_2_acc"], 2),
            "Rating 3 Acc (%)": round(eval_metrics["rating_wise_accuracy"]["rating_3_acc"], 2),
            "Rating 4 Acc (%)": round(eval_metrics["rating_wise_accuracy"]["rating_4_acc"], 2),
            "Rating 5 Acc (%)": round(eval_metrics["rating_wise_accuracy"]["rating_5_acc"], 2),
        }
        results.append(result_row)
        print(f"  Exact Acc: {result_row['Exact Accuracy (%)']}% | Within-1: {result_row['Within ±1 Accuracy (%)']}% | MAE: {result_row['MAE']:.4f} | Spearman: {result_row['Spearman Correlation']:+.4f}")

    # 4. Save results to CSV
    print("\n[Step 3/4] Saving results to CSV...")
    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"  Saved to {RESULTS_CSV}")

    # 5. Display comparison report
    print("\n[Step 4/4] Final Results Summary:")
    print("=" * 110)
    print(f"{'Embedding Model':<28} | {'Exact Acc':<9} | {'Within-1':<8} | {'MAE':<6} | {'RMSE':<6} | {'Spearman':<8} | {'R1 Acc':<6} | {'R2 Acc':<6} | {'R3 Acc':<6} | {'R4 Acc':<6} | {'R5 Acc':<6}")
    print("-" * 110)
    for res in results:
        print(f"{res['Embedding Model']:<28} | {res['Exact Accuracy (%)']:>8}% | {res['Within ±1 Accuracy (%)']:>7}% | {res['MAE']:>.4f} | {res['RMSE']:>.4f} | {res['Spearman Correlation']:>+.4f} | {res['Rating 1 Acc (%)']:>5}% | {res['Rating 2 Acc (%)']:>5}% | {res['Rating 3 Acc (%)']:>5}% | {res['Rating 4 Acc (%)']:>5}% | {res['Rating 5 Acc (%)']:>5}%")
    print("=" * 110)

    # Print Confusion Matrices
    print("\n" + "=" * 80)
    print("  CONFUSION MATRICES (Row: Predicted Rating, Column: Actual Ground Truth)")
    print("=" * 80)
    for name, matrix in confusion_matrices.items():
        print(f"\n--- {name} ---")
        print("             Actual")
        print("Predicted    1       2       3       4       5")
        for p_val in range(1, 6):
            row_str = "  ".join(f"{matrix[p_val-1, a_val-1]:>6}" for a_val in range(1, 6))
            print(f"{p_val:<12}{row_str}")

    print("\nExperiment successfully completed.")


if __name__ == "__main__":
    main()
