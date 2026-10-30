"""
analysis/distribution_eval.py
-------------------------------
Compares the *predicted rating distribution* (obtained by two methods) against
the *real empirical test-set histogram* using three divergence measures plus
bootstrapped 95% CIs.

Two prediction strategies
--------------------------
  point   : use argmax(p_k) per sample, build the predicted histogram from those
            discrete point-predictions.
  sample  : draw one rating from p_k per sample (stochastic), average 100 such
            draws to get a smooth predicted histogram.

Divergence measures (all computed against the empirical test histogram q)
--------------------------------------------------------------------------
  Wasserstein-1 : Earth Mover's Distance on the ordinal scale {1,…,5}.
                  W1 = sum_{k=1}^{4} |CDF_pred(k) - CDF_true(k)|
                       (equivalent to L1 distance between CDFs)
  chi-square    : Pearson chi-squared statistic (pred vs true cell counts).
                  chi2 = sum_k (O_k - E_k)^2 / E_k  where O=pred*N, E=true*N
  KL divergence : KL(pred || true) = sum_k p_k * log(p_k / (q_k + eps))
                  with eps=1e-6 added to both distributions before dividing.

Bootstrap CIs
-------------
  1 000 resamples (with replacement) of the test rows.
  Each resample: rebuild the predicted histogram and recompute the three metrics.
  CI = [2.5th, 97.5th] percentile across resamples.

Outputs
-------
  results/distribution_eval_regime{A,B}.json   — full numeric results
  results/distribution_eval_summary.csv        — one row per (regime, strategy)
  results/dist_histogram_regime{A,B}.png       — side-by-side bar chart
"""

from __future__ import annotations

import io
import json
import os
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from analysis.splits         import DATA_PATH, RESULTS_DIR, hash_text, normalize_text
from analysis.label_structure import get_frozen_agent, get_baseline_predictions

# ── constants ─────────────────────────────────────────────────────────────────
RATINGS      = [1, 2, 3, 4, 5]
N_RATINGS    = 5
KL_EPS       = 1e-6
N_BOOT       = 1000
SEED         = 42
N_SAMPLE_AVG = 100          # stochastic samples to average for "sample" strategy


# ─────────────────────────────────────────────────────────────────────────────
# Distribution helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dist_to_array(dist: dict) -> np.ndarray:
    """Convert {rating: prob} dict → length-5 array (index 0 = rating 1)."""
    arr = np.array([dist.get(r, 0.0) for r in RATINGS], dtype=np.float64)
    s = arr.sum()
    return arr / s if s > 0 else np.ones(5) / 5


def _empirical_hist(y: np.ndarray) -> np.ndarray:
    """Normalised empirical histogram over {1,…,5}."""
    counts = np.bincount(y, minlength=6)[1:6].astype(np.float64)
    return counts / counts.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Point vs Sample prediction histograms
# ─────────────────────────────────────────────────────────────────────────────

def _point_histogram(probs: np.ndarray) -> np.ndarray:
    """argmax prediction → normalised histogram."""
    preds  = probs.argmax(axis=1) + 1          # 1-indexed
    counts = np.bincount(preds, minlength=6)[1:6].astype(np.float64)
    return counts / counts.sum()


def _sample_histogram(probs: np.ndarray, n_avg: int = N_SAMPLE_AVG,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Vectorised: draw n_avg ratings per row from p_k, build one combined histogram.
    probs shape: (N, 5).  Uses CDF inverse-CDF trick.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    N   = probs.shape[0]
    cdf = np.cumsum(probs, axis=1)             # (N, 5)
    u   = rng.random((N, n_avg))               # (N, n_avg)
    # drawn[i, t] = number of CDF steps < u[i,t], i.e. 0-indexed class
    drawn = (cdf[:, :, None] < u[:, None, :]).sum(axis=1) + 1  # (N, n_avg), 1-indexed
    flat  = drawn.ravel()                      # N * n_avg values in {1..5}
    counts = np.bincount(flat, minlength=6)[1:6].astype(np.float64)
    return counts / counts.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Divergence measures
# ─────────────────────────────────────────────────────────────────────────────

def wasserstein1(p: np.ndarray, q: np.ndarray) -> float:
    """W1 = sum_{k=1}^{K-1} |CDF_p(k) - CDF_q(k)|  (ordinal W1)."""
    cdf_p = np.cumsum(p)[:-1]
    cdf_q = np.cumsum(q)[:-1]
    return float(np.sum(np.abs(cdf_p - cdf_q)))


def chi_square(p_hist: np.ndarray, q_hist: np.ndarray, N: int) -> float:
    """
    Pearson chi-squared comparing predicted cell counts (p_hist*N) to
    expected (q_hist*N).  Returns the statistic (not p-value).
    Zero expected cells are skipped.
    """
    O = p_hist * N
    E = q_hist * N
    mask = E > 0
    return float(np.sum((O[mask] - E[mask]) ** 2 / E[mask]))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = KL_EPS) -> float:
    """KL(p || q) with epsilon floor on both distributions."""
    p_s = p + eps;  p_s /= p_s.sum()
    q_s = q + eps;  q_s /= q_s.sum()
    return float(np.sum(p_s * np.log(p_s / q_s)))


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_metrics(probs: np.ndarray, y_true: np.ndarray,
                        strategy: str, n_boot: int = N_BOOT,
                        seed: int = SEED):
    """
    Return bootstrap 95% CIs for (W1, chi2, KL) under the given strategy.
    Fully vectorised — no Python loops over rows inside the boot loop.

    Returns
    -------
    dict with keys "W1", "chi2", "KL", each mapping to (lo, hi).
    """
    rng = np.random.default_rng(seed)
    N   = len(y_true)
    W1s, chi2s, KLs = [], [], []

    # Pre-compute CDF for sample strategy (used in fast sampling)
    cdf = np.cumsum(probs, axis=1)          # (N, 5)

    for _ in range(n_boot):
        idx   = rng.integers(0, N, size=N)
        p_b   = probs[idx]
        y_b   = y_true[idx]
        q_emp = _empirical_hist(y_b)

        if strategy == "point":
            pred_hist = _point_histogram(p_b)
        else:
            # Vectorised sample: draw n_avg_b trials via CDF trick
            cdf_b   = np.cumsum(p_b, axis=1)   # (N, 5)
            n_avg_b = 10
            u       = rng.random((N, n_avg_b))
            drawn = (cdf_b[:, :, None] < u[:, None, :]).sum(axis=1) + 1  # (N, n_avg_b)
            flat  = drawn.ravel()
            counts = np.bincount(flat, minlength=6)[1:6].astype(np.float64)
            pred_hist = counts / counts.sum()

        W1s.append(wasserstein1(pred_hist, q_emp))
        chi2s.append(chi_square(pred_hist, q_emp, N))
        KLs.append(kl_divergence(pred_hist, q_emp))

    def ci(vals):
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {"W1": ci(W1s), "chi2": ci(chi2s), "KL": ci(KLs)}


# ─────────────────────────────────────────────────────────────────────────────
# Histogram plot
# ─────────────────────────────────────────────────────────────────────────────

def _histogram_plot(q_emp: np.ndarray,
                    point_hist: np.ndarray,
                    sample_hist: np.ndarray,
                    regime: str, out_path: str):
    x    = np.arange(1, 6)
    w    = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(x - w,  q_emp,       width=w, label="True (test)",    color="steelblue",  alpha=0.8)
    ax.bar(x,      point_hist,  width=w, label="Point (argmax)", color="darkorange", alpha=0.8)
    ax.bar(x + w,  sample_hist, width=w, label="Sample (avg)",   color="forestgreen",alpha=0.8)

    ax.set_xticks(x)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Predicted vs True Rating Distribution — Regime {regime}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-regime logic
# ─────────────────────────────────────────────────────────────────────────────

def _load_dataframe() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, header=None)
    cols_25 = [
        "review_id", "name", "age", "brand", "model", "price_usd",
        "price_local", "currency", "exchange_rate_to_usd", "rating",
        "review_text", "sentiment", "country", "language", "date",
        "verified_purchase", "battery_life_rating", "camera_rating",
        "performance_rating", "design_rating", "display_rating",
        "review_length", "word_count", "helpful_votes", "source",
    ]
    if len(df.columns) == 25:
        df.columns = cols_25
    else:
        if "review_text" not in df.columns:
            df = df.rename(columns={10: "review_text", 9: "rating"})
    df["review_text"] = df["review_text"].astype(str)
    df["rating"]      = pd.to_numeric(df["rating"], errors="coerce")
    df["text_hash"]   = df["review_text"].apply(
        lambda t: hash_text(normalize_text(t)))
    return df


def run_regime(regime: str, df: pd.DataFrame, agent) -> list[dict]:
    print(f"\n{'=' * 64}")
    print(f"  DISTRIBUTION EVAL — REGIME {regime}")
    print(f"{'=' * 64}")

    with open(os.path.join(RESULTS_DIR, f"splits_regime{regime}.json")) as f:
        splits = json.load(f)

    df_test = df.iloc[splits["test"]].copy()
    if "text_hash" not in df_test.columns:
        df_test["text_hash"] = df_test["review_text"].apply(
            lambda t: hash_text(normalize_text(t)))

    y_true = df_test["rating"].values.astype(int)
    N      = len(y_true)

    print(f"  Test set size: {N}")

    print("  Computing baseline distributions...")
    test_dists = get_baseline_predictions(agent, df_test)
    probs = np.array([_dist_to_array(test_dists[h]) for h in df_test["text_hash"]])

    # ── empirical histogram ──────────────────────────────────────────────────
    q_emp = _empirical_hist(y_true)
    print(f"  True histogram:   {dict(zip(RATINGS, q_emp.round(4)))}")

    # ── point histogram ──────────────────────────────────────────────────────
    point_hist = _point_histogram(probs)
    print(f"  Point histogram:  {dict(zip(RATINGS, point_hist.round(4)))}")

    # ── sample histogram ─────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    sample_hist = _sample_histogram(probs, n_avg=N_SAMPLE_AVG, rng=rng)
    print(f"  Sample histogram: {dict(zip(RATINGS, sample_hist.round(4)))}")

    # ── divergence metrics (point) ────────────────────────────────────────────
    def compute_metrics(pred_hist):
        return {
            "W1":   wasserstein1(pred_hist, q_emp),
            "chi2": chi_square(pred_hist, q_emp, N),
            "KL":   kl_divergence(pred_hist, q_emp),
        }

    m_point  = compute_metrics(point_hist)
    m_sample = compute_metrics(sample_hist)

    print(f"\n  Point  metrics  → W1={m_point['W1']:.4f}  "
          f"chi2={m_point['chi2']:.4f}  KL={m_point['KL']:.6f}")
    print(f"  Sample metrics  → W1={m_sample['W1']:.4f}  "
          f"chi2={m_sample['chi2']:.4f}  KL={m_sample['KL']:.6f}")

    # ── bootstrap CIs ────────────────────────────────────────────────────────
    print(f"  Running {N_BOOT} bootstrap resamples (point)...")
    ci_point  = _bootstrap_metrics(probs, y_true, strategy="point",  n_boot=N_BOOT, seed=SEED)
    print(f"  Running {N_BOOT} bootstrap resamples (sample)...")
    ci_sample = _bootstrap_metrics(probs, y_true, strategy="sample", n_boot=N_BOOT, seed=SEED)

    def fmt_ci(val, lo, hi):
        return f"{val:.4f} [{lo:.4f}, {hi:.4f}]"

    print(f"\n  Bootstrap 95% CIs:")
    print(f"  {'Metric':<10} {'Point':<30} {'Sample'}")
    print(f"  {'-'*70}")
    for k in ["W1", "chi2", "KL"]:
        p_str = fmt_ci(m_point[k],  *ci_point[k])
        s_str = fmt_ci(m_sample[k], *ci_sample[k])
        print(f"  {k:<10} {p_str:<30} {s_str}")

    # ── Save histogram plot ───────────────────────────────────────────────────
    png_path = os.path.join(RESULTS_DIR, f"dist_histogram_regime{regime}.png")
    _histogram_plot(q_emp, point_hist, sample_hist, regime, png_path)
    print(f"\n  Saved PNG → {png_path}")

    # ── Build result rows ─────────────────────────────────────────────────────
    rows = []
    for strategy, metrics, cis, hist in [
        ("point",  m_point,  ci_point,  point_hist),
        ("sample", m_sample, ci_sample, sample_hist),
    ]:
        rows.append({
            "regime":   regime,
            "strategy": strategy,
            "N":        N,
            "W1":       round(metrics["W1"],   4),
            "W1_lo":    round(cis["W1"][0],    4),
            "W1_hi":    round(cis["W1"][1],    4),
            "chi2":     round(metrics["chi2"], 4),
            "chi2_lo":  round(cis["chi2"][0],  4),
            "chi2_hi":  round(cis["chi2"][1],  4),
            "KL":       round(metrics["KL"],   6),
            "KL_lo":    round(cis["KL"][0],    6),
            "KL_hi":    round(cis["KL"][1],    6),
            "true_hist":    json.dumps(dict(zip(RATINGS, q_emp.round(6).tolist()))),
            "pred_hist":    json.dumps(dict(zip(RATINGS, hist.round(6).tolist()))),
        })

    # ── Save JSON ─────────────────────────────────────────────────────────────
    json_out = {
        "regime":       regime,
        "N":            N,
        "true_hist":    dict(zip(RATINGS, q_emp.tolist())),
        "point_hist":   dict(zip(RATINGS, point_hist.tolist())),
        "sample_hist":  dict(zip(RATINGS, sample_hist.tolist())),
        "point":  {**m_point,  "CI": ci_point},
        "sample": {**m_sample, "CI": ci_sample},
    }
    json_path = os.path.join(RESULTS_DIR, f"distribution_eval_regime{regime}.json")
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved JSON → {json_path}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def task4():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading dataset...")
    df = _load_dataframe()

    print("Loading frozen agent...")
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        agent = get_frozen_agent()

    all_rows = []
    for regime in ["A", "B"]:
        rows = run_regime(regime, df, agent)
        all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "distribution_eval_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary CSV saved → {csv_path}")

    print("\n\nFULL DISTRIBUTION EVAL SUMMARY")
    display_cols = ["regime", "strategy", "N", "W1", "W1_lo", "W1_hi",
                    "chi2", "chi2_lo", "chi2_hi", "KL", "KL_lo", "KL_hi"]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    task4()
