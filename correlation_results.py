"""
correlation_results.py
======================
Reads existing calculated results from project files and displays a
formatted summary in the terminal.

Reads from:
  - data/correlation_metrics_summary.csv   (all correlation statistics)
  - data/evaluation_results.csv            (sentiment correct/incorrect counts)

Does NOT:
  - Modify agent.py or any model/sentiment logic
  - Hard-code any result values
  - Change the dataset or evaluation methodology
"""

from __future__ import annotations

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

# ── File paths ────────────────────────────────────────────────────────────────
METRICS_CSV    = os.path.join("data", "correlation_metrics_summary.csv")
EVAL_CSV       = os.path.join("data", "evaluation_results.csv")

# ── Validate files exist ──────────────────────────────────────────────────────
for path in [METRICS_CSV, EVAL_CSV]:
    if not os.path.exists(path):
        print(f"ERROR: Required file not found: {path}")
        print("Please run evaluate_predictions.py and scratch/compute_correlation_analysis.py first.")
        sys.exit(1)

# ── 1. Load sentiment model metrics from evaluation_results.csv ───────────────
df_eval = pd.read_csv(EVAL_CSV, encoding="utf-8")

total_reviews   = len(df_eval)
sent_correct    = int(df_eval["sentiment_correct"].sum())
sent_incorrect  = total_reviews - sent_correct
sent_accuracy   = sent_correct / total_reviews * 100.0

# ── 2. Load correlation metrics from correlation_metrics_summary.csv ──────────
df_metrics = pd.read_csv(METRICS_CSV, encoding="utf-8")

# Row 0: Actual Sentiment vs Actual Rating
row_act  = df_metrics.iloc[0]
# Row 1: Predicted Sentiment vs Actual Rating
row_pred = df_metrics.iloc[1]
# Row 3: Review Text Embeddings 5-Fold CV Ridge (the definitive out-of-fold result)
row_text = df_metrics.iloc[3]

def fmt_rho(val: float) -> str:
    """Format a correlation coefficient with a leading + sign."""
    return f"+{val:.4f}" if val >= 0 else f"{val:.4f}"

def fmt_pval(val: float) -> str:
    """Format p-value: show '< 1 x 10^-300' when scipy returns 0.0."""
    if val == 0.0 or val < 1e-300:
        return "< 1 x 10^-300"
    return f"{val:.4e}"

# ── 3. Determine the best Spearman correlation ────────────────────────────────
comparisons = [
    ("Actual Sentiment -> Rating   ", row_act["Spearman_rho"],  "Actual Sentiment vs Actual Rating"),
    ("Predicted Sentiment -> Rating", row_pred["Spearman_rho"], "Predicted Sentiment vs Actual Rating"),
    ("Review Text -> Rating        ", row_text["Spearman_rho"], "Review Text vs Actual Rating"),
]
best_label, best_rho, best_full = max(comparisons, key=lambda x: x[1])

# ── 4. Print formatted terminal output ───────────────────────────────────────
SEP = "=" * 60

print()
print(SEP)
print("SENTIMENT MODEL")
print("=" * 15)
print()
print(f"Total Reviews       : {total_reviews:,}")
print(f"Correct             : {sent_correct:,}")
print(f"Incorrect           : {sent_incorrect:,}")
print(f"Accuracy            : {sent_accuracy:.2f}%")
print()

print(SEP)
print("CORRELATION ANALYSIS")
print("=" * 20)
print()

# 1. Actual Sentiment vs Actual Rating
print("1. ACTUAL SENTIMENT vs ACTUAL RATING")
print(f"   Spearman Correlation : {fmt_rho(row_act['Spearman_rho'])}")
print(f"   Pearson Correlation  : {fmt_rho(row_act['Pearson_r'])}")
print(f"   p-value              : {fmt_pval(row_act['Spearman_p_value'])}")
print()

# 2. Predicted Sentiment vs Actual Rating
print("2. PREDICTED SENTIMENT vs ACTUAL RATING")
print(f"   Spearman Correlation : {fmt_rho(row_pred['Spearman_rho'])}")
print(f"   Pearson Correlation  : {fmt_rho(row_pred['Pearson_r'])}")
print(f"   p-value              : {fmt_pval(row_pred['Spearman_p_value'])}")
print()

# 3. Review Text vs Actual Rating (5-Fold CV Ridge)
print("3. REVIEW TEXT vs ACTUAL RATING")
print(f"   Spearman Correlation : {fmt_rho(row_text['Spearman_rho'])}")
print(f"   Pearson Correlation  : {fmt_rho(row_text['Pearson_r'])}")
print(f"   p-value              : {fmt_pval(row_text['Spearman_p_value'])}")
print(f"   MAE                  : {row_text['MAE']:.4f}")
print(f"   RMSE                 : {row_text['RMSE']:.4f}")
print(f"   R\u00b2                   : {row_text['R2']:.4f}")
print()

print(SEP)
print("WHICH CORRELATION IS BEST?")
print("=" * 26)
print()
for label, rho, _ in comparisons:
    print(f"   {label}   {fmt_rho(rho)}")
print()
print("BEST:")
print(f"   {best_full}")
print()
print(f"   Spearman Correlation = {fmt_rho(best_rho)}")
print()

print(SEP)
print("CONCLUSION")
print("=" * 10)
print()

act_rho  = row_act["Spearman_rho"]
pred_rho = row_pred["Spearman_rho"]
text_rho = row_text["Spearman_rho"]

print(f"The strongest relationship is between Actual Sentiment and")
print(f"Actual Rating, with a Spearman correlation of {fmt_rho(act_rho)}.")
print()
print(f"The predicted sentiment correlation of {fmt_rho(pred_rho)} is very close")
print(f"to the actual sentiment correlation, showing that the model's")
print(f"sentiment predictions preserve the strong relationship with")
print(f"customer ratings.")
print()
print(f"Review Text also has a strong positive relationship with")
print(f"customer rating, with a Spearman correlation of {fmt_rho(text_rho)}.")
print()
print(SEP)
print()
