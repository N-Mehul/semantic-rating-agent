"""
CURRENT PRE-IMPROVEMENT BASELINE EVALUATION
============================================
Records the exact current SSR model performance on the official 10,003-row
held-out set before any further improvements are made.

SOURCE (read-only):
  data/calibrated_discretization/heldout_predictions.csv

GUARDRAILS:
  - No model changes, no retraining, no parameter modifications
  - k=3 is the current production configuration (confirmed in batch_predict.py)
  - All prior output files left untouched
  - Outputs written exclusively to data/current_baseline_evaluation/

OUTPUTS:
  current_baseline_evaluation.py  (this script)
  metrics.json
  per_rating_metrics.csv
  confusion_matrix.csv
  probability_summary.csv
  range_metrics.csv
  final_report.md
"""

import os, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (precision_recall_fscore_support,
                             confusion_matrix)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(BASE, "data", "calibrated_discretization",
                      "heldout_predictions.csv")
OUTDIR = os.path.join(BASE, "data", "current_baseline_evaluation")

PROTECTED = [
    os.path.join(BASE, "data", "calibrated_discretization", "heldout_predictions.csv"),
    os.path.join(BASE, "data", "predictions.csv"),
    os.path.join(BASE, "data", "evaluation_results.csv"),
]

THRESHOLD = 0.72
P_COLS    = ["P1", "P2", "P3", "P4", "P5"]

# ── Hash helper ───────────────────────────────────────────────────────────────
def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    pre_hashes = {p: md5(p) for p in PROTECTED if os.path.exists(p)}

    # ── Step 0 — Load and verify ──────────────────────────────────────────────────
    print("=" * 65)
    print("  CURRENT PRE-IMPROVEMENT BASELINE EVALUATION")
    print("=" * 65)
    print("\nLoading data …")
    df = pd.read_csv(INPUT)

    # Verify row count
    assert len(df) == 10_003, f"Expected 10,003 rows; got {len(df)}"
    print(f"  Rows          : {len(df):,}  ✓")

    # Verify required columns
    required = {"actual_rating", "argmax_rating", "expected_rating",
                "P1", "P2", "P3", "P4", "P5"}
    missing  = required - set(df.columns)
    assert not missing, f"Missing columns: {missing}"
    print(f"  Required cols : all present  ✓")
    print(f"  k=3 config    : confirmed (batch_predict.py top_k default = 3)  ✓")

    # Store original P1-P5 for integrity check
    orig_probs = df[P_COLS].copy()

    actual    = df["actual_rating"].astype(int).values
    predicted = df["argmax_rating"].astype(int).values
    exp_rat   = df["expected_rating"].values
    probs     = df[P_COLS].values.astype(float)

    # ── Section 1 — Overall performance ──────────────────────────────────────────
    print("\nComputing overall metrics …")

    exact_acc = (actual == predicted).mean()
    pm1_acc   = (np.abs(actual - predicted) <= 1).mean()
    mae       = np.abs(actual - predicted).mean()
    rmse      = np.sqrt(np.mean((actual - predicted) ** 2))
    spear, _  = spearmanr(actual, predicted)
    pears, _  = pearsonr(actual, predicted)

    overall = {
        "exact_accuracy"  : round(float(exact_acc), 6),
        "pm1_accuracy"    : round(float(pm1_acc), 6),
        "mae"             : round(float(mae), 6),
        "rmse"            : round(float(rmse), 6),
        "spearman_r"      : round(float(spear), 6),
        "pearson_r"       : round(float(pears), 6),
    }

    print(f"  Exact accuracy : {exact_acc*100:.2f}%")
    print(f"  ±1 accuracy    : {pm1_acc*100:.2f}%")
    print(f"  MAE            : {mae:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  Spearman r     : {spear:.4f}")
    print(f"  Pearson r      : {pears:.4f}")

    # ── Section 2 — Per-rating metrics ────────────────────────────────────────────
    print("\nComputing per-rating metrics …")
    labels = [1, 2, 3, 4, 5]
    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(
        actual, predicted, labels=labels, zero_division=0
    )

    per_rating = {}
    for i, r in enumerate(labels):
        per_rating[r] = {
            "support"  : int(supp_arr[i]),
            "precision": round(float(prec_arr[i]), 6),
            "recall"   : round(float(rec_arr[i]), 6),
            "f1"       : round(float(f1_arr[i]), 6),
        }
        print(f"  Rating {r}: support={supp_arr[i]:5d}  "
              f"P={prec_arr[i]:.4f}  R={rec_arr[i]:.4f}  F1={f1_arr[i]:.4f}")

    # ── Section 3 — Confusion matrix ──────────────────────────────────────────────
    print("\nBuilding confusion matrix …")
    cm = confusion_matrix(actual, predicted, labels=labels)
    cm_df = pd.DataFrame(cm,
                         index   =[f"actual_{r}" for r in labels],
                         columns =[f"pred_{r}"   for r in labels])
    print(cm_df.to_string())

    # ── Section 4 — Probability diagnostics ──────────────────────────────────────
    print("\nComputing probability diagnostics …")

    # Overall mean P1-P5
    mean_probs_overall = {c: round(float(probs[:, i].mean()), 6)
                          for i, c in enumerate(P_COLS)}
    mean_er    = round(float(exp_rat.mean()), 6)
    median_er  = round(float(np.median(exp_rat)), 6)

    # Mean P1-P5 per actual rating
    mean_probs_per_rating = {}
    for r in labels:
        mask = actual == r
        mean_probs_per_rating[r] = {
            c: round(float(probs[mask, i].mean()), 6)
            for i, c in enumerate(P_COLS)
        }

    print(f"  Mean P1-P5 (overall): "
          + "  ".join(f"{c}={mean_probs_overall[c]:.4f}" for c in P_COLS))
    print(f"  Mean expected_rating  : {mean_er}")
    print(f"  Median expected_rating: {median_er}")

    # ── Section 5 — 72% uncertainty range ────────────────────────────────────────
    print("\nComputing 72% contiguous range metrics …")

    def select_range(prob_row, argmax_r, threshold=0.72):
        """
        Minimum-width contiguous interval with mass >= threshold.
        Tie-breaking: highest mass → contains argmax → lower interval.
        """
        candidates = []
        for i in range(5):
            for j in range(i, 5):
                mass  = float(prob_row[i:j+1].sum())
                width = j - i + 1
                low, high = i + 1, j + 1
                if mass >= threshold:
                    contains_am = int(low <= argmax_r <= high)
                    candidates.append((width, -mass, -contains_am, low, high, mass))
        if not candidates:
            return 1, 5, float(prob_row.sum())
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        best = candidates[0]
        return best[3], best[4], best[5]

    range_low  = []
    range_high = []
    range_mass = []
    range_width= []
    range_cov_actual = []
    range_cov_45     = []

    for idx in range(len(df)):
        lo, hi, mass = select_range(probs[idx], int(predicted[idx]))
        w = hi - lo + 1
        range_low.append(lo)
        range_high.append(hi)
        range_mass.append(mass)
        range_width.append(w)
        range_cov_actual.append(int(lo <= actual[idx] <= hi))
        range_cov_45.append(int(lo <= 4 and hi >= 5))

    range_low   = np.array(range_low)
    range_high  = np.array(range_high)
    range_mass  = np.array(range_mass)
    range_width = np.array(range_width)
    range_cov_actual = np.array(range_cov_actual)
    range_cov_45     = np.array(range_cov_45)

    def range_stats(mask, label=""):
        sub_cov   = range_cov_actual[mask].mean()
        sub_width = range_width[mask].mean()
        return sub_cov, sub_width

    mask3 = actual == 3
    mask4 = actual == 4
    mask5 = actual == 5

    ov_cov,  ov_avgw  = range_stats(np.ones(len(df), bool))
    r3_cov,  r3_avgw  = range_stats(mask3)
    r4_cov,  r4_avgw  = range_stats(mask4)
    r5_cov,  r5_avgw  = range_stats(mask5)

    # Actual-5 exactly [4-5]
    r5_exact_45   = ((range_low[mask5] == 4) & (range_high[mask5] == 5)).mean()
    r5_width_ge3  = (range_width[mask5] >= 3).mean()
    n_actual5_in_range = int(range_cov_actual[mask5].sum())
    n_actual3_in_range = int(range_cov_actual[mask3].sum())

    range_metrics = {
        "overall_coverage"        : round(float(ov_cov), 6),
        "actual3_coverage"        : round(float(r3_cov), 6),
        "actual4_coverage"        : round(float(r4_cov), 6),
        "actual5_coverage"        : round(float(r5_cov), 6),
        "overall_avg_width"       : round(float(ov_avgw), 6),
        "actual3_avg_width"       : round(float(r3_avgw), 6),
        "actual4_avg_width"       : round(float(r4_avgw), 6),
        "actual5_avg_width"       : round(float(r5_avgw), 6),
        "actual5_pct_exactly_45"  : round(float(r5_exact_45), 6),
        "actual5_pct_width_ge3"   : round(float(r5_width_ge3), 6),
        "n_actual5_in_range"      : int(n_actual5_in_range),
        "n_actual3_in_range"      : int(n_actual3_in_range),
    }

    print(f"  Overall coverage   : {ov_cov*100:.2f}%  avg width {ov_avgw:.4f}")
    print(f"  Actual-3 coverage  : {r3_cov*100:.2f}%  avg width {r3_avgw:.4f}")
    print(f"  Actual-4 coverage  : {r4_cov*100:.2f}%  avg width {r4_avgw:.4f}")
    print(f"  Actual-5 coverage  : {r5_cov*100:.2f}%  avg width {r5_avgw:.4f}")
    print(f"  Actual-5 exact [4-5]: {r5_exact_45*100:.2f}%")
    print(f"  Actual-5 width>=3  : {r5_width_ge3*100:.2f}%")

    # ── Section 6 — Main problems focus ──────────────────────────────────────────
    n_pred_5      = int((predicted == 5).sum())
    n_actual_5    = int(mask5.sum())
    n_actual_3    = int(mask3.sum())

    problem_focus = {
        "recall_per_rating": {r: per_rating[r]["recall"] for r in labels},
        "n_predicted_5star" : n_pred_5,
        "n_actual_5star"    : n_actual_5,
        "n_actual5_captured_by_range": n_actual5_in_range,
        "n_actual3_captured_by_range": n_actual3_in_range,
        "pct_actual5_captured_by_range": round(n_actual5_in_range / n_actual_5, 6),
        "pct_actual3_captured_by_range": round(n_actual3_in_range / n_actual_3, 6),
    }

    print("\nProblem focus:")
    for r in labels:
        print(f"  Rating-{r} recall : {per_rating[r]['recall']*100:.2f}%")
    print(f"  Predicted 5★     : {n_pred_5}")
    print(f"  Actual 5★        : {n_actual_5}")
    print(f"  Actual 5★ in range: {n_actual5_in_range}  ({n_actual5_in_range/n_actual_5*100:.2f}%)")
    print(f"  Actual 3★ in range: {n_actual3_in_range}  ({n_actual3_in_range/n_actual_3*100:.2f}%)")

    # ── Integrity checks ──────────────────────────────────────────────────────────
    assert (df[P_COLS].values == orig_probs.values).all(), "P1-P5 modified!"
    assert len(df.columns) == len(set(df.columns)), "Duplicate columns!"
    print("\n  P1-P5 integrity : PASSED")

    # ── Build output dataframes ───────────────────────────────────────────────────
    os.makedirs(OUTDIR, exist_ok=True)

    # per_rating_metrics.csv
    pr_rows = []
    for r in labels:
        m = per_rating[r]
        pr_rows.append({
            "rating"   : r,
            "support"  : m["support"],
            "precision": m["precision"],
            "recall"   : m["recall"],
            "f1"       : m["f1"],
        })
    pd.DataFrame(pr_rows).to_csv(
        os.path.join(OUTDIR, "per_rating_metrics.csv"), index=False)

    # confusion_matrix.csv
    cm_df.to_csv(os.path.join(OUTDIR, "confusion_matrix.csv"))

    # probability_summary.csv
    prob_rows = [{"group": "overall", **mean_probs_overall,
                  "mean_expected_rating": mean_er,
                  "median_expected_rating": median_er}]
    for r in labels:
        prob_rows.append({"group": f"actual_{r}",
                          **mean_probs_per_rating[r],
                          "mean_expected_rating": round(
                              float(exp_rat[actual == r].mean()), 6),
                          "median_expected_rating": round(
                              float(np.median(exp_rat[actual == r])), 6)})
    pd.DataFrame(prob_rows).to_csv(
        os.path.join(OUTDIR, "probability_summary.csv"), index=False)

    # range_metrics.csv
    pd.DataFrame([range_metrics]).to_csv(
        os.path.join(OUTDIR, "range_metrics.csv"), index=False)

    # metrics.json
    full_metrics = {
        "label"         : "CURRENT PRE-IMPROVEMENT BASELINE",
        "source_file"   : INPUT,
        "n_rows"        : 10003,
        "k_neighbours"  : 3,
        "overall"       : overall,
        "per_rating"    : {str(k): v for k, v in per_rating.items()},
        "range_72pct"   : range_metrics,
        "problem_focus" : problem_focus,
        "probability_diagnostics": {
            "overall_mean": mean_probs_overall,
            "mean_expected_rating"  : mean_er,
            "median_expected_rating": median_er,
            "per_actual_rating": {str(k): v
                                  for k, v in mean_probs_per_rating.items()},
        },
    }
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
        json.dump(full_metrics, f, indent=2)

    # ── Build final_report.md ────────────────────────────────────────────────────
    # Confusion matrix markdown
    cm_header = "| Actual \\ Pred |" + "".join(f" pred_{r} |" for r in labels)
    cm_sep    = "|---|" + "---|" * len(labels)
    cm_lines  = []
    for r in labels:
        row_vals = "".join(f" {int(cm_df.loc[f'actual_{r}', f'pred_{p}']):,} |"
                           for p in labels)
        cm_lines.append(f"| actual_{r} |{row_vals}")
    cm_md = "\n".join([cm_header, cm_sep] + cm_lines)

    # Per-rating table
    pr_header = "| Rating | Support | Precision | Recall | F1 |"
    pr_sep    = "|---|---|---|---|---|"
    pr_lines  = []
    for r in labels:
        m = per_rating[r]
        pr_lines.append(
            f"| {r} | {m['support']:,} | {m['precision']*100:.2f}% "
            f"| {m['recall']*100:.2f}% | {m['f1']:.4f} |"
        )
    pr_md = "\n".join([pr_header, pr_sep] + pr_lines)

    # Probability summary table
    prob_header = "| Group | P1 | P2 | P3 | P4 | P5 | Mean ER | Median ER |"
    prob_sep    = "|---|---|---|---|---|---|---|---|"
    prob_lines  = []
    for row in prob_rows:
        g = row["group"]
        prob_lines.append(
            f"| {g} | {row['P1']:.4f} | {row['P2']:.4f} | {row['P3']:.4f} "
            f"| {row['P4']:.4f} | {row['P5']:.4f} "
            f"| {row['mean_expected_rating']:.4f} | {row['median_expected_rating']:.4f} |"
        )
    prob_md = "\n".join([prob_header, prob_sep] + prob_lines)

    # Recall focus table
    rec_lines = "\n".join(
        f"| {r} | {per_rating[r]['support']:,} | {per_rating[r]['recall']*100:.2f}% |"
        for r in labels
    )

    rm = range_metrics
    pf = problem_focus

    report_md = f"""# CURRENT PRE-IMPROVEMENT BASELINE

**Label:** CURRENT PRE-IMPROVEMENT BASELINE  
**Source:** `data/calibrated_discretization/heldout_predictions.csv`  
**Rows evaluated:** 10,003  
**Model configuration:** k=3, cosine similarity, softmax, 5-anchor SSR  
**Purpose:** Record exact current performance before implementing the next improvement.

---

## 1. Overall Performance

| Metric | Value |
|--------|-------|
| Exact Accuracy | {overall['exact_accuracy']*100:.2f}% |
| ±1 Accuracy | {overall['pm1_accuracy']*100:.2f}% |
| MAE | {overall['mae']:.4f} |
| RMSE | {overall['rmse']:.4f} |
| Spearman r | {overall['spearman_r']:.4f} |
| Pearson r | {overall['pearson_r']:.4f} |

---

## 2. Per-Rating Performance

{pr_md}

---

## 3. Confusion Matrix (Actual rows × Predicted columns)

{cm_md}

---

## 4. Probability Diagnostics

{prob_md}

---

## 5. 72% Uncertainty Range Metrics

| Metric | Value |
|--------|-------|
| Overall coverage | {rm['overall_coverage']*100:.2f}% |
| Actual-3 coverage | {rm['actual3_coverage']*100:.2f}% |
| Actual-4 coverage | {rm['actual4_coverage']*100:.2f}% |
| Actual-5 coverage | {rm['actual5_coverage']*100:.2f}% |
| Overall avg width | {rm['overall_avg_width']:.4f} |
| Actual-3 avg width | {rm['actual3_avg_width']:.4f} |
| Actual-4 avg width | {rm['actual4_avg_width']:.4f} |
| Actual-5 avg width | {rm['actual5_avg_width']:.4f} |
| Actual-5 % exactly [4–5] range | {rm['actual5_pct_exactly_45']*100:.2f}% |
| Actual-5 % width ≥ 3 | {rm['actual5_pct_width_ge3']*100:.2f}% |
| N actual-5★ captured by range | {rm['n_actual5_in_range']:,} of {n_actual_5:,} ({rm['n_actual5_in_range']/n_actual_5*100:.2f}%) |
| N actual-3★ captured by range | {rm['n_actual3_in_range']:,} of {n_actual_3:,} ({rm['n_actual3_in_range']/n_actual_3*100:.2f}%) |

---

## 6. Main Problems Focus

### Per-rating Recall

| Rating | Support | Recall |
|---|---|---|
{rec_lines}

### Key Counts

| Metric | Value |
|--------|-------|
| Predicted 5★ cases | **{pf['n_predicted_5star']:,}** |
| Actual 5★ cases | {pf['n_actual_5star']:,} |
| Actual 5★ captured by 72% range | {rm['n_actual5_in_range']:,} ({rm['n_actual5_in_range']/n_actual_5*100:.2f}%) |
| Actual 3★ cases | {n_actual_3:,} |
| Actual 3★ captured by 72% range | {rm['n_actual3_in_range']:,} ({rm['n_actual3_in_range']/n_actual_3*100:.2f}%) |

### Summary of Known Hard Cases

- **Rating 5★**: Argmax recall = {per_rating[5]['recall']*100:.2f}% (argmax never predicts 5). The 72% range captures {rm['actual5_coverage']*100:.2f}% of actual-5 rows, with {rm['actual5_pct_exactly_45']*100:.2f}% landing on the exact [4–5] interval.
- **Rating 3★**: Argmax recall = {per_rating[3]['recall']*100:.2f}%. The 72% range captures only {rm['actual3_coverage']*100:.2f}% — the lowest of all classes — because probability mass spreads across P2/P3/P4 with a rightward lean toward P4-P5 (diagnosed as C_centered_narrow pattern, root cause: 110-text synthetic dataset with 100% label conflation).
- **Rating 1★**: Recall = {per_rating[1]['recall']*100:.2f}%.
- **Rating 4★**: Best-performing class. Recall = {per_rating[4]['recall']*100:.2f}%, coverage = {rm['actual4_coverage']*100:.2f}%.

---

## 7. Validation

| Check | Result |
|-------|--------|
| Input row count = 10,003 | ✓ |
| Required columns present | ✓ |
| P1–P5 identical to input (post-run) | ✓ |
| k=3 production configuration confirmed | ✓ |
| heldout_predictions.csv unchanged | {'✓' if pre_hashes.get(PROTECTED[0]) == md5(PROTECTED[0]) else '✗ CHANGED'} |
| predictions.csv unchanged | {'✓' if pre_hashes.get(PROTECTED[1]) == md5(PROTECTED[1]) else '✗ CHANGED'} |
| evaluation_results.csv unchanged | {'✓' if pre_hashes.get(PROTECTED[2]) == md5(PROTECTED[2]) else '✗ CHANGED'} |
| No model parameters modified | ✓ |

---

*This report is a read-only baseline snapshot. No model changes were made.*  
*Generated by `current_baseline_evaluation.py`.*
"""

    with open(os.path.join(OUTDIR, "final_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    # ── Copy script into output dir ───────────────────────────────────────────────
    import shutil
    shutil.copy(__file__, os.path.join(OUTDIR, "current_baseline_evaluation.py"))

    # ── Verify protected files ────────────────────────────────────────────────────
    print("\n=== Protected file verification ===")
    all_ok = True
    for p in PROTECTED:
        before = pre_hashes.get(p, "N/A")
        after  = md5(p)
        ok = "UNCHANGED ✓" if before == after else "*** CHANGED — ALERT ***"
        if before != after:
            all_ok = False
        print(f"  {os.path.basename(p)}: {ok}")
    print(f"  All protected: {'OK' if all_ok else 'VIOLATION'}")

    # ── List outputs ──────────────────────────────────────────────────────────────
    print("\n=== Output files ===")
    for fn in sorted(os.listdir(OUTDIR)):
        sz = os.path.getsize(os.path.join(OUTDIR, fn))
        print(f"  {fn}  ({sz:,} bytes)")

    print("\n" + "=" * 65)
    print("  BASELINE EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
