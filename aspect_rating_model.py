"""
aspect_rating_model.py
========================
IMPLEMENT D1 ASPECT-BASED RANDOM FOREST MODEL

Diagnostic model that reproduces Experiment D1:
  Random Forest Classifier on exactly the 5 aspect ratings.

IMPORTANT:
  - This is an ASPECT-BASED DIAGNOSTIC MODEL.
  - The 5 aspect ratings are co-assigned sub-target variables and contain potential target leakage.
  - The 73.34% result must NOT be presented as a leakage-free improvement over the 46.14% text baseline.
  - The production text baseline remains completely unchanged.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "data", "Mobile Reviews Sentiment.csv")
EVAL_EXCEL = os.path.join(BASE_DIR, "data", "actual_reviews.xlsx")
OUT_DIR = os.path.join(BASE_DIR, "data", "aspect_model")

ASPECT_COLS = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
]

COLS_25 = [
    "review_id", "name", "age", "brand", "model", "price_usd", "price_local",
    "currency", "exchange_rate_to_usd", "rating", "review_text", "sentiment",
    "country", "language", "date", "verified_purchase", "battery_life_rating",
    "camera_rating", "performance_rating", "design_rating", "display_rating",
    "review_length", "word_count", "helpful_votes", "source"
]

CLASSES = [1, 2, 3, 4, 5]


def load_datasets():
    df_train = pd.read_csv(TRAIN_CSV, low_memory=False)
    df_train = df_train.dropna(subset=["review_text", "rating"] + ASPECT_COLS).copy()
    df_train["rating"] = df_train["rating"].astype(int)

    df_eval = pd.read_excel(EVAL_EXCEL, header=None)
    df_eval.columns = COLS_25
    df_eval["rating"] = df_eval["rating"].astype(int)

    return df_train, df_eval


def plot_confusion_matrix(cm, labels, title, filepath, normalize=False):
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    if normalize:
        # Normalize by row (actual rating support)
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100.0
        fmt = "{:.1f}%"
        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    else:
        cm_display = cm
        fmt = "{:,d}"
        im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if normalize:
        cbar.set_label("Percentage (%)", rotation=270, labelpad=15, fontsize=11)
    else:
        cbar.set_label("Number of Reviews", rotation=270, labelpad=15, fontsize=11)

    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([f"{c}★" for c in labels], fontsize=11, fontweight="bold")
    ax.set_yticklabels([f"{c}★" for c in labels], fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Rating", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Actual Rating", fontsize=12, fontweight="bold", labelpad=10)

    # Text annotations in each cell
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_display[i, j]
            color = "white" if val > thresh else "black"
            text_str = fmt.format(int(val) if not normalize else val)
            ax.text(j, i, text_str, ha="center", va="center", color=color,
                    fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df_train, df_eval = load_datasets()

    X_train_raw = df_train[ASPECT_COLS].values
    y_train = df_train["rating"].values

    X_eval_raw = df_eval[ASPECT_COLS].values
    y_eval = df_eval["rating"].values

    # Preprocessing: StandardScaler fitted strictly on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_eval = scaler.transform(X_eval_raw)

    # Model: Exactly as in Experiment D1
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    t0_fit = time.time()
    rf.fit(X_train, y_train)
    fit_time = time.time() - t0_fit

    # Train predictions (for overfitting check)
    preds_train = rf.predict(X_train)
    train_acc = accuracy_score(y_train, preds_train) * 100.0

    # Eval predictions
    t0_pred = time.time()
    preds_eval = rf.predict(X_eval)
    pred_time = time.time() - t0_pred

    eval_acc = accuracy_score(y_eval, preds_eval) * 100.0
    macro_f1 = f1_score(y_eval, preds_eval, average="macro", zero_division=0) * 100.0
    weighted_f1 = f1_score(y_eval, preds_eval, average="weighted", zero_division=0) * 100.0
    mae = mean_absolute_error(y_eval, preds_eval)
    rmse = np.sqrt(mean_squared_error(y_eval, preds_eval))
    spearman, _ = stats.spearmanr(y_eval, preds_eval)
    pearson, _ = stats.pearsonr(y_eval, preds_eval)

    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
        y_eval, preds_eval, labels=CLASSES, zero_division=0
    )

    # 1. Save predictions.csv
    pred_df = pd.DataFrame({
        "actual_rating": y_eval,
        "predicted_rating": preds_eval,
        "battery_life_rating": df_eval["battery_life_rating"].values,
        "camera_rating": df_eval["camera_rating"].values,
        "performance_rating": df_eval["performance_rating"].values,
        "design_rating": df_eval["design_rating"].values,
        "display_rating": df_eval["display_rating"].values,
    })
    pred_path = os.path.join(OUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # 2. Confusion matrix numerical CSV
    cm = confusion_matrix(y_eval, preds_eval, labels=CLASSES)
    cm_df = pd.DataFrame(
        cm,
        index=[f"{c}★" for c in CLASSES],
        columns=[f"{c}★" for c in CLASSES]
    )
    cm_df.index.name = "Actual/Predicted"
    cm_path = os.path.join(OUT_DIR, "confusion_matrix.csv")
    cm_df.to_csv(cm_path)

    # 3. Confusion matrix plots
    cm_img_path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(
        cm,
        labels=CLASSES,
        title="D1 Aspect-Based Random Forest Confusion Matrix",
        filepath=cm_img_path,
        normalize=False,
    )

    cm_norm_img_path = os.path.join(OUT_DIR, "confusion_matrix_normalized.png")
    plot_confusion_matrix(
        cm,
        labels=CLASSES,
        title="D1 Aspect-Based Random Forest — Normalized Confusion Matrix",
        filepath=cm_norm_img_path,
        normalize=True,
    )

    # 4. Classification report CSV
    clf_dict = {
        "class": CLASSES,
        "precision": [float(p) * 100.0 for p in prec_arr],
        "recall": [float(r) * 100.0 for r in rec_arr],
        "f1_score": [float(f) * 100.0 for f in f1_arr],
        "support": [int(s) for s in supp_arr],
    }
    clf_df = pd.DataFrame(clf_dict)
    clf_path = os.path.join(OUT_DIR, "classification_report.csv")
    clf_df.to_csv(clf_path, index=False)

    # 5. Metrics JSON
    metrics_json_data = {
        "accuracy": float(eval_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "training_accuracy": float(train_acc),
        "evaluation_accuracy": float(eval_acc),
        "train_eval_gap": float(train_acc - eval_acc),
        "recall_1": float(rec_arr[0]) * 100.0,
        "recall_2": float(rec_arr[1]) * 100.0,
        "recall_3": float(rec_arr[2]) * 100.0,
        "recall_4": float(rec_arr[3]) * 100.0,
        "recall_5": float(rec_arr[4]) * 100.0,
        "precision_1": float(prec_arr[0]) * 100.0,
        "precision_2": float(prec_arr[1]) * 100.0,
        "precision_3": float(prec_arr[2]) * 100.0,
        "precision_4": float(prec_arr[3]) * 100.0,
        "precision_5": float(prec_arr[4]) * 100.0,
        "f1_1": float(f1_arr[0]) * 100.0,
        "f1_2": float(f1_arr[1]) * 100.0,
        "f1_3": float(f1_arr[2]) * 100.0,
        "f1_4": float(f1_arr[3]) * 100.0,
        "f1_5": float(f1_arr[4]) * 100.0,
        "leakage_status": "POTENTIAL TARGET LEAKAGE — DIAGNOSTIC ONLY",
    }
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json_data, f, indent=2)

    # ── Terminal Output ───────────────────────────────────────────────────────
    print("==================================================")
    print("D1 ASPECT-BASED RANDOM FOREST")
    print("==================================================")
    print("")
    print("Features:")
    print("5 aspect ratings")
    print("")
    print("Model:")
    print("Random Forest")
    print("")
    print("Evaluation samples:")
    print(f"{len(df_eval):,}")
    print("")
    print(f"Accuracy:\n{eval_acc:.2f}%")
    print("")
    print(f"Macro F1:\n{macro_f1:.2f}%")
    print("")
    print(f"Weighted F1:\n{weighted_f1:.2f}%")
    print("")
    print(f"MAE:\n{mae:.4f}")
    print("")
    print(f"RMSE:\n{rmse:.4f}")
    print("")
    print(f"Pearson:\n{pearson:.4f}")
    print("")
    print(f"Spearman:\n{spearman:.4f}")
    print("")
    print("==================================================")
    print("PER-CLASS RESULTS")
    print("==================================================")
    for c in CLASSES:
        idx = c - 1
        print(f"\n{c}★:")
        print(f"Precision = {prec_arr[idx]*100:.2f}%")
        print(f"Recall = {rec_arr[idx]*100:.2f}%")
        print(f"F1 = {f1_arr[idx]*100:.2f}%")
    print("")
    print("==================================================")
    print("CLASSIFICATION REPORT")
    print("==================================================")
    print(classification_report(y_eval, preds_eval, labels=CLASSES, digits=4, zero_division=0))
    print("==================================================")
    print("CONFUSION MATRIX")
    print("==================================================")
    print("                PREDICTED")
    print("             1★    2★    3★    4★    5★")
    for i, c in enumerate(CLASSES):
        row = cm[i]
        print(f"ACTUAL {c}★   {row[0]:5d} {row[1]:5d} {row[2]:5d} {row[3]:5d} {row[4]:5d}")
    print("")
    print("==================================================")
    print("OVERFITTING CHECK")
    print("==================================================")
    print(f"Training accuracy   : {train_acc:.2f}%")
    print(f"Evaluation accuracy : {eval_acc:.2f}%")
    print(f"Train-Eval Gap      : {train_acc - eval_acc:+.2f} percentage points")
    print("")
    print("==================================================")
    print("GENERATED FILES")
    print("==================================================")
    print(f"predictions.csv             : {pred_path}")
    print(f"metrics.json                : {metrics_path}")
    print(f"classification_report.csv   : {clf_path}")
    print(f"confusion_matrix.csv        : {cm_path}")
    print(f"confusion_matrix.png        : {cm_img_path}")
    print(f"confusion_matrix_normalized.png : {cm_norm_img_path}")
    print("")
    print("==================================================")
    print("LEAKAGE WARNING")
    print("==================================================")
    print("WARNING:")
    print("This is an aspect-based diagnostic experiment.")
    print("The five aspect ratings are co-assigned sub-target variables and may introduce target leakage.")
    print("The 73.34% result must NOT be presented as a leakage-free improvement over the 46.14% text-based baseline.")
    print("==================================================")


if __name__ == "__main__":
    main()
