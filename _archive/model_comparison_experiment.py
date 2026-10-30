"""
model_comparison_experiment.py
==============================
A completely separate model-comparison experiment using the full dataset
and sentence embeddings. 

Models compared:
  1. Logistic Regression
  2. Ridge Regression
  3. Support Vector Machine (calibrated LinearSVC)
  4. k-NN
  5. MLP Neural Network
  6. XGBoost Classifier
  7. LightGBM Classifier
  8. CatBoost Classifier
  9. Ordinal Regression (Frank & Hall ordinal approach using Logistic Regression)

No modifications to the existing agent, training pipeline, or memory.json.
"""

from __future__ import annotations

import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ── UTF-8 Output on Windows ──────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

# Model imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Path Constants
CSV_PATH = os.path.join("data", "Mobile Reviews Sentiment.csv")
RESULTS_OUTPUT_CSV = os.path.join("data", "model_comparison_results.csv")


# ── Custom Ordinal Classifier (Frank & Hall Method) ──────────────────────────
class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        self.clfs = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)
        for i in range(n_classes - 1):
            # Binary label: Y > class_i
            binary_y = (y > self.classes_[i]).astype(int)
            clf = clone(self.base_estimator)
            clf.fit(X, binary_y)
            self.clfs[i] = clf
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # S[i] = P(Y > classes_[i])
        S = np.zeros((n_samples, n_classes - 1))
        for i in range(n_classes - 1):
            # probability of class 1
            S[:, i] = self.clfs[i].predict_proba(X)[:, 1]

        probs = np.zeros((n_samples, n_classes))
        # P(Y = classes_[0]) = 1 - P(Y > classes_[0])
        probs[:, 0] = 1.0 - S[:, 0]
        # P(Y = classes_[k]) = P(Y > classes_[k-1]) - P(Y > classes_[k])
        for k in range(1, n_classes - 1):
            probs[:, k] = S[:, k - 1] - S[:, k]
        # P(Y = classes_[K-1]) = P(Y > classes_[K-2])
        probs[:, n_classes - 1] = S[:, n_classes - 2]

        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.where(row_sums > 0, probs / row_sums, 1.0 / n_classes)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array([self.classes_[idx] for idx in np.argmax(probs, axis=1)])


def evaluate_model(y_true, pred_likert, expected_rating) -> dict:
    """
    Computes all standard capstone evaluation metrics.
    """
    n_samples = len(y_true)
    y_true = np.array(y_true, dtype=float)
    pred_likert = np.array(pred_likert, dtype=float)
    expected_rating = np.array(expected_rating, dtype=float)

    # Core Metrics
    correct = np.sum(y_true == pred_likert)
    exact_acc = (correct / n_samples) * 100.0
    within_1_acc = (np.sum(np.abs(y_true - pred_likert) <= 1.0) / n_samples) * 100.0
    mae = np.mean(np.abs(expected_rating - y_true))
    rmse = np.sqrt(np.mean((expected_rating - y_true) ** 2))

    if n_samples >= 2 and np.std(expected_rating) > 1e-9 and np.std(y_true) > 1e-9:
        spearman_corr, _ = stats.spearmanr(y_true, expected_rating)
    else:
        spearman_corr = 0.0

    # Rating-wise Accuracy (1 to 5)
    rating_accs = {}
    for r in range(1, 6):
        r_mask = (y_true == float(r))
        r_total = r_mask.sum()
        if r_total > 0:
            r_correct = ((pred_likert == float(r)) & r_mask).sum()
            rating_accs[f"rating_{r}_acc"] = (r_correct / r_total) * 100.0
        else:
            rating_accs[f"rating_{r}_acc"] = 0.0

    # 5x5 Confusion Matrix
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


def main():
    print("=" * 70)
    print("  MODEL COMPARISON EXPERIMENT — SEMANTIC RATING AGENT")
    print("=" * 70)

    # ── 1. Load and Clean Dataset ─────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at: {CSV_PATH}")
        sys.exit(1)

    print("\n[Step 1/5] Loading and cleaning dataset...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    
    # Exclude completely empty rows
    df_clean = df.dropna(how="all").reset_index(drop=True)
    
    # Align text and rating columns
    text_col = "review_text"
    rating_col = "rating"
    pair = df_clean[[text_col, rating_col]].dropna(subset=[text_col, rating_col])
    
    n_total = len(pair)
    unique_texts = pair[text_col].astype(str).unique().tolist()
    n_unique = len(unique_texts)
    
    print(f"  Informative rows containing text + rating: {n_total:,}")
    print(f"  Unique review texts                      : {n_unique}")

    # ── 2. Sentence Embeddings ────────────────────────────────────────────────
    print("\n[Step 2/5] Initializing sentence-transformer model & embedding text...")
    from sentence_transformers import SentenceTransformer
    model_st = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Embed unique texts and map them back to all rows
    print(f"  Embedding {n_unique} unique reviews...")
    unique_embeddings = model_st.encode(unique_texts, batch_size=64, show_progress_bar=False)
    text_to_emb = {t: unique_embeddings[i] for i, t in enumerate(unique_texts)}
    
    print("  Mapping embeddings back to the full dataset...")
    X = np.stack(pair[text_col].map(text_to_emb).values)
    y = pair[rating_col].astype(int).values
    
    print(f"  Feature matrix X shape: {X.shape}")
    print(f"  Target vector y shape : {y.shape}")

    # ── 3. Stratified Train/Test Split (80/20) ────────────────────────────────
    print("\n[Step 3/5] Performing stratified 80/20 split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Training set size: {X_train.shape[0]:,}")
    print(f"  Testing set size : {X_test.shape[0]:,}")

    # ── 4. Train and Evaluate Models ──────────────────────────────────────────
    print("\n[Step 4/5] Training and evaluating all 9 models...")
    
    # Models dictionary
    models = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "SVM (Calibrated Linear)": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", dual=False, random_state=42)
        ),
        "k-NN (distance-weighted)": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric="mlogloss"),
        "LightGBM": lgb.LGBMClassifier(random_state=42, verbose=-1),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
        "Ordinal Regression": OrdinalClassifier(),
    }

    results_summary = []
    confusion_matrices = {}

    for name, clf in models.items():
        print(f"  Training {name}...")
        
        # Train
        if name in ["XGBoost", "LightGBM", "CatBoost"]:
            # Standard GBDTs require 0-based classification labels
            clf.fit(X_train, y_train - 1)
        else:
            clf.fit(X_train, y_train)

        # Predict
        if name == "Ridge Regression":
            # Regression model outputs continuous expected ratings directly
            pred_continuous = clf.predict(X_test)
            expected_rating = np.clip(pred_continuous, 1.0, 5.0)
            pred_likert = np.round(expected_rating).astype(int)
        elif name in ["XGBoost", "LightGBM", "CatBoost"]:
            probs = clf.predict_proba(X_test)
            pred_likert = np.argmax(probs, axis=1) + 1
            expected_rating = np.dot(probs, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        else:
            probs = clf.predict_proba(X_test)
            pred_likert = clf.predict(X_test)
            expected_rating = np.dot(probs, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # Evaluate
        eval_metrics = evaluate_model(y_test, pred_likert, expected_rating)
        
        # Save evaluation data
        metrics_dict = {
            "Model": name,
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
        results_summary.append(metrics_dict)
        confusion_matrices[name] = eval_metrics["confusion_matrix"]

    # Convert results summary to a DataFrame
    df_results = pd.DataFrame(results_summary)

    # ── 5. Save Results and Report ────────────────────────────────────────────
    print("\n[Step 5/5] Saving metrics and printing final comparison report...")
    df_results.to_csv(RESULTS_OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"  Results saved to: {RESULTS_OUTPUT_CSV}")

    # Print nicely formatted terminal table
    print("\n" + "=" * 105)
    print(f"{'Model Name':<25} | {'Exact Acc':<9} | {'Within-1':<8} | {'MAE':<6} | {'RMSE':<6} | {'Spearman':<8} | {'R1 Acc':<6} | {'R2 Acc':<6} | {'R3 Acc':<6} | {'R4 Acc':<6} | {'R5 Acc':<6}")
    print("-" * 105)
    for res in results_summary:
        print(f"{res['Model']:<25} | {res['Exact Accuracy (%)']:>8}% | {res['Within ±1 Accuracy (%)']:>7}% | {res['MAE']:>.4f} | {res['RMSE']:>.4f} | {res['Spearman Correlation']:>+.4f} | {res['Rating 1 Acc (%)']:>5}% | {res['Rating 2 Acc (%)']:>5}% | {res['Rating 3 Acc (%)']:>5}% | {res['Rating 4 Acc (%)']:>5}% | {res['Rating 5 Acc (%)']:>5}%")
    print("=" * 105)

    # Print Confusion Matrices
    print("\nCONFUSION MATRICES:")
    for name, matrix in confusion_matrices.items():
        print(f"\n--- {name} ---")
        print("             Actual")
        print("Predicted    1    2    3    4    5")
        for p_val in range(1, 6):
            row_str = "  ".join(f"{matrix[p_val-1, a_val-1]:>4}" for a_val in range(1, 6))
            print(f"{p_val:<12}{row_str}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
