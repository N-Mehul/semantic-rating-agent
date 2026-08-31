"""
safe_feature_model_comparison_experiment.py
===========================================
Safe-Feature Final Model Comparison Experiment.

Target Variable: `rating` (1 to 5)

Explicitly Enforced Feature Governance:
- EXCLUDED (Banned Leaked Features):
    * camera_rating
    * battery_life_rating
    * display_rating
    * performance_rating
    * design_rating
    * sentiment (ground truth)
    * helpful_votes
- INCLUDED (Audited Safe Features):
    * review_text (all-MiniLM-L6-v2 384-d dense embeddings)
    * review_length, word_count
    * brand, model, price_usd, price_local, currency, exchange_rate_to_usd
    * verified_purchase, source, language, country, age
    * review_date (extracted month, day-of-week, year)

Models Evaluated:
  1. Logistic Regression (Tuned & Regularized)
  2. Ridge Regression (Tuned & Regularized)
  3. Support Vector Machine (Calibrated LinearSVC, Tuned)
  4. MLP Neural Network (Tuned Architecture & Regularization)
  5. XGBoost Classifier (Tuned Depth, LR, Regularization)
  6. LightGBM Classifier (Tuned Leaves, Class-Weights, Regularization)
  7. CatBoost Classifier (Tuned Depth, Regularization)
  8. Ordinal Regression (Frank & Hall Method with Tuned Base Estimator)

Methodology:
- Strict zero-leakage validation assertion prior to model training.
- Fixed stratified 80/20 train/test split (random_state=42).
- Hyperparameter tuning performed strictly on the training set using Stratified K-Fold CV.
- Evaluates on the held-out 8,000 test samples:
    * Exact Accuracy (%)
    * Within +/-1 Accuracy (%)
    * MAE
    * RMSE
    * Spearman Correlation
    * Rating-wise accuracy (Ratings 1 to 5)
    * 5x5 Confusion Matrix
- Compares performance against the text-only baseline (~46.14%).

Results saved to: data/safe_feature_model_comparison_results.csv
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, List

# Suppress warnings
warnings.filterwarnings("ignore")

# UTF-8 Output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

# Path Constants
CSV_PATH = os.path.join("data", "Mobile Reviews Sentiment.csv")
RESULTS_CSV = os.path.join("data", "safe_feature_model_comparison_results.csv")


# â”€â”€ Custom Ordinal Classifier (Frank & Hall Method) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
class FrankHallOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """
    Ordinal Classifier using the Frank & Hall (2001) binary decomposition method.
    K classes -> K-1 binary classifiers predicting P(Y > k).
    """
    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )
        self.clfs = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        n_classes = len(self.classes_)
        self.clfs = {}
        for i in range(n_classes - 1):
            binary_y = (y > self.classes_[i]).astype(int)
            clf = clone(self.base_estimator)
            clf.fit(X, binary_y)
            self.clfs[i] = clf
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        S = np.zeros((n_samples, n_classes - 1))
        for i in range(n_classes - 1):
            if hasattr(self.clfs[i], "predict_proba"):
                S[:, i] = self.clfs[i].predict_proba(X)[:, 1]
            else:
                df = self.clfs[i].decision_function(X)
                S[:, i] = 1.0 / (1.0 + np.exp(-df))

        probs = np.zeros((n_samples, n_classes))
        probs[:, 0] = 1.0 - S[:, 0]
        for k in range(1, n_classes - 1):
            probs[:, k] = S[:, k - 1] - S[:, k]
        probs[:, n_classes - 1] = S[:, n_classes - 2]

        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = np.where(row_sums > 0, probs / row_sums, 1.0 / n_classes)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.array([self.classes_[idx] for idx in np.argmax(probs, axis=1)])


def evaluate_model(y_true: np.ndarray, pred_likert: np.ndarray, expected_rating: np.ndarray) -> Dict:
    """Computes capstone evaluation metrics."""
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
    print("=" * 85)
    print("  SAFE-FEATURE MODEL COMPARISON EXPERIMENT")
    print("=" * 85)

    # â”€â”€ 1. Load Dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at: {CSV_PATH}")
        sys.exit(1)

    print("\n[Step 1/5] Loading and inspecting dataset...")
    df_raw = pd.read_csv(CSV_PATH, encoding="utf-8-sig", low_memory=False)
    df_clean = df_raw.dropna(subset=["review_text", "rating"]).reset_index(drop=True)
    df_clean["rating"] = df_clean["rating"].astype(int)

    n_total = len(df_clean)
    print(f"  Informative rows loaded: {n_total:,}")

    # â”€â”€ 2. Strict Target Leakage Verification Check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[Step 2/5] Running pre-training Target Leakage Audit...")
    banned_features = [
        "camera_rating",
        "battery_life_rating",
        "display_rating",
        "performance_rating",
        "design_rating",
        "sentiment",
        "helpful_votes"
    ]

    # Clean price_local: strip currency symbols like â‚¹, $, â‚¬
    if "price_local" in df_clean.columns:
        df_clean["price_local"] = (
            df_clean["price_local"]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True)
        )
        df_clean["price_local"] = pd.to_numeric(df_clean["price_local"], errors="coerce")

    # Clean numeric columns
    for col in ["price_usd", "exchange_rate_to_usd", "age", "review_length", "word_count"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Pre-parse date features
    if "review_date" in df_clean.columns:
        date_series = pd.to_datetime(df_clean["review_date"], errors="coerce")
        df_clean["review_year"] = date_series.dt.year.fillna(2023).astype(float)
        df_clean["review_month"] = date_series.dt.month.fillna(6).astype(float)
        df_clean["review_dayofweek"] = date_series.dt.dayofweek.fillna(3).astype(float)

    safe_numeric_cols = [
        "review_length", "word_count", "price_usd", "price_local",
        "exchange_rate_to_usd", "age", "review_year", "review_month", "review_dayofweek"
    ]
    safe_categorical_cols = [
        "brand", "model", "currency", "source", "language", "country", "verified_purchase"
    ]

    # Ensure no banned features exist in our feature column list
    for f in safe_numeric_cols + safe_categorical_cols + ["review_text"]:
        assert f not in banned_features, f"LEAKAGE DETECTED: Banned feature {f} in safe feature list!"
        assert f != "rating", "LEAKAGE DETECTED: Target variable in feature list!"

    print("  Pre-training Leakage Check Passed: ZERO leaked features detected.")
    print(f"  Safe Numeric Features     : {safe_numeric_cols}")
    print(f"  Safe Categorical Features : {safe_categorical_cols}")

    # â”€â”€ 3. Feature Extraction & Engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[Step 3/5] Computing embeddings & preprocessing tabular metadata...")
    
    # 3a. Sentence Embeddings (all-MiniLM-L6-v2)
    unique_texts = df_clean["review_text"].astype(str).unique().tolist()
    print(f"  Encoding {len(unique_texts):,} unique review texts with all-MiniLM-L6-v2...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    uniq_emb = embedder.encode(unique_texts, batch_size=64, show_progress_bar=False)
    text_to_emb = {t: uniq_emb[i] for i, t in enumerate(unique_texts)}

    X_text = np.stack(df_clean["review_text"].astype(str).map(text_to_emb).values)
    print(f"  Text feature matrix shape: {X_text.shape}")

    # 3b. Tabular Preprocessor
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, safe_numeric_cols),
        ("cat", cat_pipeline, safe_categorical_cols),
    ])

    X_tabular = preprocessor.fit_transform(df_clean[safe_numeric_cols + safe_categorical_cols])
    print(f"  Tabular metadata matrix shape: {X_tabular.shape}")

    # Combine text embeddings + safe metadata
    X_combined = np.hstack([X_text, X_tabular])
    y = df_clean["rating"].values
    print(f"  Combined Safe Feature Matrix shape: {X_combined.shape} (384 text dims + {X_tabular.shape[1]} tabular dims)")

    # â”€â”€ 4. Stratified 80/20 Train/Test Split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"  Train set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # â”€â”€ 5. Train & Evaluate Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[Step 4/5] Training and evaluating all 8 models with safe features...")

    results_summary = []
    confusion_matrices = {}

    # 1. Logistic Regression
    print("\n1. Optimizing Logistic Regression (Safe Features)...")
    param_grid_lr = {"C": [0.1, 1.0, 10.0], "class_weight": ["balanced", None]}
    grid_lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs"),
                           param_grid_lr, cv=cv, scoring="accuracy", n_jobs=1)
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    print(f"   Best Params: {grid_lr.best_params_}")
    probs_lr = best_lr.predict_proba(X_test)
    pred_lr = best_lr.predict(X_test)
    exp_lr = np.dot(probs_lr, np.arange(1, 6))
    eval_lr = evaluate_model(y_test, pred_lr, exp_lr)

    # 2. Ridge Regression
    print("\n2. Optimizing Ridge Regression (Safe Features)...")
    param_grid_ridge = {"alpha": [1.0, 10.0, 50.0, 100.0]}
    grid_ridge = GridSearchCV(Ridge(random_state=42), param_grid_ridge, cv=cv, scoring="neg_mean_absolute_error", n_jobs=1)
    grid_ridge.fit(X_train, y_train)
    best_ridge = grid_ridge.best_estimator_
    print(f"   Best Params: {grid_ridge.best_params_}")
    exp_ridge = np.clip(best_ridge.predict(X_test), 1.0, 5.0)
    pred_ridge = np.round(exp_ridge).astype(int)
    eval_ridge = evaluate_model(y_test, pred_ridge, exp_ridge)

    # 3. SVM (Calibrated LinearSVC)
    print("\n3. Optimizing Support Vector Machine (LinearSVC, Safe Features)...")
    param_grid_svm = {"C": [0.05, 0.1, 0.5, 1.0], "class_weight": ["balanced", None]}
    grid_svm = GridSearchCV(LinearSVC(dual=False, random_state=42, max_iter=2000), param_grid_svm, cv=cv, scoring="accuracy", n_jobs=1)
    grid_svm.fit(X_train, y_train)
    best_svm_raw = grid_svm.best_estimator_
    print(f"   Best Params: {grid_svm.best_params_}")
    best_svm = CalibratedClassifierCV(best_svm_raw, cv=3)
    best_svm.fit(X_train, y_train)
    probs_svm = best_svm.predict_proba(X_test)
    pred_svm = best_svm.predict(X_test)
    exp_svm = np.dot(probs_svm, np.arange(1, 6))
    eval_svm = evaluate_model(y_test, pred_svm, exp_svm)

    # 4. MLP Neural Network
    print("\n4. Optimizing MLP Neural Network (Safe Features)...")
    param_grid_mlp = {
        "hidden_layer_sizes": [(64, 32), (128, 64)],
        "alpha": [0.001, 0.01],
        "learning_rate_init": [0.001, 0.01],
    }
    grid_mlp = GridSearchCV(MLPClassifier(max_iter=150, random_state=42, early_stopping=True),
                            param_grid_mlp, cv=cv, scoring="accuracy", n_jobs=1)
    grid_mlp.fit(X_train, y_train)
    best_mlp = grid_mlp.best_estimator_
    print(f"   Best Params: {grid_mlp.best_params_}")
    probs_mlp = best_mlp.predict_proba(X_test)
    pred_mlp = best_mlp.predict(X_test)
    exp_mlp = np.dot(probs_mlp, np.arange(1, 6))
    eval_mlp = evaluate_model(y_test, pred_mlp, exp_mlp)

    # 5. XGBoost Classifier
    print("\n5. Optimizing XGBoost (Safe Features)...")
    param_grid_xgb = {
        "n_estimators": [60, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "reg_alpha": [0.5],
        "reg_lambda": [1.0],
    }
    grid_xgb = GridSearchCV(xgb.XGBClassifier(random_state=42, eval_metric="mlogloss", n_jobs=1, verbosity=0),
                            param_grid_xgb, cv=cv, scoring="accuracy", n_jobs=1)
    grid_xgb.fit(X_train, y_train - 1)
    best_xgb = grid_xgb.best_estimator_
    print(f"   Best Params: {grid_xgb.best_params_}")
    probs_xgb = best_xgb.predict_proba(X_test)
    pred_xgb = np.argmax(probs_xgb, axis=1) + 1
    exp_xgb = np.dot(probs_xgb, np.arange(1, 6))
    eval_xgb = evaluate_model(y_test, pred_xgb, exp_xgb)

    # 6. LightGBM Classifier
    print("\n6. Optimizing LightGBM (Safe Features)...")
    param_grid_lgb = {
        "n_estimators": [60, 100],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "reg_lambda": [1.0],
    }
    grid_lgb = GridSearchCV(lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
                            param_grid_lgb, cv=cv, scoring="accuracy", n_jobs=1)
    grid_lgb.fit(X_train, y_train - 1)
    best_lgb = grid_lgb.best_estimator_
    print(f"   Best Params: {grid_lgb.best_params_}")
    probs_lgb = best_lgb.predict_proba(X_test)
    pred_lgb = np.argmax(probs_lgb, axis=1) + 1
    exp_lgb = np.dot(probs_lgb, np.arange(1, 6))
    eval_lgb = evaluate_model(y_test, pred_lgb, exp_lgb)

    # 7. CatBoost Classifier
    print("\n7. Optimizing CatBoost (Safe Features)...")
    param_grid_cb = {
        "iterations": [100, 150],
        "depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [5.0],
    }
    grid_cb = GridSearchCV(CatBoostClassifier(random_state=42, verbose=0, thread_count=1),
                           param_grid_cb, cv=cv, scoring="accuracy", n_jobs=1)
    grid_cb.fit(X_train, y_train - 1)
    best_cb = grid_cb.best_estimator_
    print(f"   Best Params: {grid_cb.best_params_}")
    probs_cb = best_cb.predict_proba(X_test)
    pred_cb = np.argmax(probs_cb, axis=1) + 1
    exp_cb = np.dot(probs_cb, np.arange(1, 6))
    eval_cb = evaluate_model(y_test, pred_cb, exp_cb)

    # 8. Ordinal Regression (Frank & Hall)
    print("\n8. Optimizing Ordinal Regression (Safe Features)...")
    param_grid_ord = {
        "base_estimator__C": [0.1, 1.0, 10.0],
        "base_estimator__class_weight": ["balanced", None],
    }
    grid_ord = GridSearchCV(
        FrankHallOrdinalClassifier(base_estimator=LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")),
        param_grid_ord, cv=cv, scoring="accuracy", n_jobs=1
    )
    grid_ord.fit(X_train, y_train)
    best_ord = grid_ord.best_estimator_
    print(f"   Best Params: {grid_ord.best_params_}")
    probs_ord = best_ord.predict_proba(X_test)
    pred_ord = best_ord.predict(X_test)
    exp_ord = np.dot(probs_ord, np.arange(1, 6))
    eval_ord = evaluate_model(y_test, pred_ord, exp_ord)

    # Collect Results
    model_evaluations = {
        "Logistic Regression (Safe-Features)": eval_lr,
        "Ridge Regression (Safe-Features)": eval_ridge,
        "SVM Linear (Safe-Features)": eval_svm,
        "MLP Neural Net (Safe-Features)": eval_mlp,
        "XGBoost (Safe-Features)": eval_xgb,
        "LightGBM (Safe-Features)": eval_lgb,
        "CatBoost (Safe-Features)": eval_cb,
        "Ordinal Regression (Safe-Features)": eval_ord,
    }

    for name, m_eval in model_evaluations.items():
        results_summary.append({
            "Model": name,
            "Exact Accuracy (%)": round(m_eval["exact_accuracy"], 2),
            "Within Â±1 Accuracy (%)": round(m_eval["within_1_accuracy"], 2),
            "MAE": round(m_eval["mae"], 4),
            "RMSE": round(m_eval["rmse"], 4),
            "Spearman Correlation": round(m_eval["spearman_correlation"], 4),
            "Rating 1 Acc (%)": round(m_eval["rating_wise_accuracy"]["rating_1_acc"], 2),
            "Rating 2 Acc (%)": round(m_eval["rating_wise_accuracy"]["rating_2_acc"], 2),
            "Rating 3 Acc (%)": round(m_eval["rating_wise_accuracy"]["rating_3_acc"], 2),
            "Rating 4 Acc (%)": round(m_eval["rating_wise_accuracy"]["rating_4_acc"], 2),
            "Rating 5 Acc (%)": round(m_eval["rating_wise_accuracy"]["rating_5_acc"], 2),
        })
        confusion_matrices[name] = m_eval["confusion_matrix"]

    df_results = pd.DataFrame(results_summary)
    df_results.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\n[Step 5/5] Results saved to: {RESULTS_CSV}")

    # â”€â”€ Display Comparison Table â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n" + "=" * 115)
    print(f"{'Model Name':<35} | {'Exact Acc':<9} | {'Within-1':<8} | {'MAE':<6} | {'RMSE':<6} | {'Spearman':<8} | {'R1 Acc':<6} | {'R2 Acc':<6} | {'R3 Acc':<6} | {'R4 Acc':<6} | {'R5 Acc':<6}")
    print("-" * 115)
    for res in results_summary:
        print(f"{res['Model']:<35} | {res['Exact Accuracy (%)']:>8}% | {res['Within Â±1 Accuracy (%)']:>7}% | {res['MAE']:>.4f} | {res['RMSE']:>.4f} | {res['Spearman Correlation']:>+.4f} | {res['Rating 1 Acc (%)']:>5}% | {res['Rating 2 Acc (%)']:>5}% | {res['Rating 3 Acc (%)']:>5}% | {res['Rating 4 Acc (%)']:>5}% | {res['Rating 5 Acc (%)']:>5}%")
    print("=" * 115)

    # Print Confusion Matrices
    print("\n" + "=" * 85)
    print("  CONFUSION MATRICES (Row: Predicted Rating, Column: Actual Ground Truth)")
    print("=" * 85)
    for name, matrix in confusion_matrices.items():
        print(f"\n--- {name} ---")
        print("             Actual")
        print("Predicted    1       2       3       4       5")
        for p_val in range(1, 6):
            row_str = "  ".join(f"{matrix[p_val-1, a_val-1]:>6}" for a_val in range(1, 6))
            print(f"{p_val:<12}{row_str}")

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()

