"""
analysis/calibration.py
------------------------
Temperature calibration for the frozen kNN semantic-rating agent.

What this does
--------------
1.  Loads the frozen agent + val/test splits for Regimes A and B.
2.  Collects the raw (T=1.0) softmax distributions from predict_single()
    for every unique review text in val and test.
3.  Sweeps ~40 log-spaced temperatures in [0.02, 2.0].
4.  Selects T* = argmin_{T} mean-NLL on the validation set.
5.  Reports a two-row table for each regime:
        Frozen T=0.1  vs  T=T*
    with NLL, RPS, ECE (15 bins), sharpness.
6.  Saves:
        results/calibration_regimeA.json
        results/calibration_regimeB.json
        results/reliability_diagram_regime{A,B}.png
        results/nll_vs_T_regime{A,B}.png

Temperature re-scaling
----------------------
predict_single() already applies T=0.1 internally (hard-coded in batch_predict.py
via _softmax(bank_sims[top_idx], temp=0.1)).  To sweep temperatures we need
the *pre-softmax logits* (the raw cosine-similarity scores).

We reconstruct them by inverting the T=0.1 softmax:
    logits_k = log(p_k) * T_frozen     (up to a constant shift)
then re-apply softmax at temperature T:
    p_k(T) = exp(logits_k / T) / sum_j exp(logits_j / T)

This is exact because log(softmax(x/T0)) = x/T0 - log(Z), so
    x_k = T0 * log(p_k) + T0 * log(Z)
and the constant T0*log(Z) cancels in the softmax re-application.
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

# ── project imports ──────────────────────────────────────────────────────────
from analysis.splits    import DATA_PATH, RESULTS_DIR, hash_text, normalize_text
from analysis.label_structure import get_frozen_agent, get_baseline_predictions
from analysis.metrics   import nll as metric_nll, rps as metric_rps
from analysis.metrics   import ece as metric_ece, sharpness as metric_sharpness

# ── constants ────────────────────────────────────────────────────────────────
T_FROZEN   = 0.1          # temperature baked into predict_single()
N_TEMPS    = 40           # number of sweep points
T_MIN, T_MAX = 0.02, 2.0  # sweep range
N_BINS_ECE = 15
RATINGS    = [1, 2, 3, 4, 5]
SEED       = 42


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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


def _load_splits(regime: str):
    path = os.path.join(RESULTS_DIR, f"splits_regime{regime}.json")
    with open(path) as f:
        return json.load(f)


def _dist_to_array(dist: dict) -> np.ndarray:
    """Convert {rating: prob} dict to length-5 np.ndarray, index 0 = rating 1."""
    arr = np.array([dist.get(r, 0.0) for r in RATINGS], dtype=np.float64)
    s = arr.sum()
    return arr / s if s > 0 else np.ones(5) / 5


def _rescale_temperature(probs_frozen: np.ndarray, T_new: float) -> np.ndarray:
    """
    Given probabilities produced at T_frozen, return probabilities at T_new.
    Uses the logit-inversion trick described in the module docstring.

    probs_frozen : (N, 5)
    returns      : (N, 5)
    """
    eps = 1e-30
    logits = T_FROZEN * np.log(np.clip(probs_frozen, eps, 1.0))  # (N, 5)
    # softmax at T_new
    scaled = logits / T_new
    scaled -= scaled.max(axis=1, keepdims=True)   # numerical stability
    exp_s  = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram
# ─────────────────────────────────────────────────────────────────────────────

def _reliability_diagram(probs: np.ndarray, y_true: np.ndarray,
                          title: str, out_path: str, n_bins: int = N_BINS_ECE):
    """Plot and save a reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1) + 1
    correct     = (predictions == y_true).astype(float)

    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    bin_acc    = []
    bin_conf   = []
    bin_counts = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_acc.append(np.nan)
            bin_conf.append((lo + hi) / 2)
            bin_counts.append(0)
        else:
            bin_acc.append(correct[mask].mean())
            bin_conf.append(confidences[mask].mean())
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    # Bar chart (gap = miscalibration)
    widths = np.diff(bin_edges)
    centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_acc)
    ax.bar(
        centres[valid], np.array(bin_acc)[valid],
        width=widths[valid], alpha=0.5, color="steelblue",
        label="Accuracy", align="center",
    )
    ax.bar(
        centres[valid], np.array(bin_conf)[valid],
        width=widths[valid], alpha=0.3, color="orange",
        label="Confidence", align="center",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (max predicted probability)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ece_val = metric_ece(probs, y_true, n_bins)
    ax.text(0.05, 0.92, f"ECE = {ece_val:.4f}", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# NLL-vs-T plot
# ─────────────────────────────────────────────────────────────────────────────

def _nll_vs_T_plot(temps: np.ndarray, nll_val: np.ndarray, nll_test: np.ndarray,
                   T_star: float, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temps, nll_val,  label="Validation NLL", color="steelblue", lw=2)
    ax.plot(temps, nll_test, label="Test NLL",       color="darkorange", lw=2, linestyle="--")
    ax.axvline(T_star,  color="red",   lw=1.5, linestyle=":",  label=f"T* = {T_star:.3f}")
    ax.axvline(T_FROZEN, color="grey", lw=1.5, linestyle="-.", label=f"T_frozen = {T_FROZEN}")
    ax.set_xscale("log")
    ax.set_xlabel("Temperature (log scale)")
    ax.set_ylabel("Mean NLL (nats)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Per-regime logic
# ─────────────────────────────────────────────────────────────────────────────

def run_regime(regime: str, df: pd.DataFrame, agent):
    print(f"\n{'=' * 64}")
    print(f"  CALIBRATION — REGIME {regime}")
    print(f"{'=' * 64}")

    splits   = _load_splits(regime)
    df_val   = df.iloc[splits["val"]].copy()
    df_test  = df.iloc[splits["test"]].copy()

    # Ensure text_hash present
    for d in [df_val, df_test]:
        if "text_hash" not in d.columns:
            d["text_hash"] = d["review_text"].apply(
                lambda t: hash_text(normalize_text(t)))

    y_val  = df_val["rating"].values.astype(int)
    y_test = df_test["rating"].values.astype(int)

    # ── Get raw T=0.1 distributions ─────────────────────────────────────────
    print("  Computing baseline distributions (val + test)...")
    val_dists  = get_baseline_predictions(agent, df_val)
    test_dists = get_baseline_predictions(agent, df_test)

    # Build (N, 5) arrays
    P_val  = np.array([_dist_to_array(val_dists[h])  for h in df_val["text_hash"]])
    P_test = np.array([_dist_to_array(test_dists[h]) for h in df_test["text_hash"]])

    # ── Temperature sweep ────────────────────────────────────────────────────
    temps = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_TEMPS)
    nll_val_sweep  = np.zeros(N_TEMPS)
    nll_test_sweep = np.zeros(N_TEMPS)

    for i, T in enumerate(temps):
        P_val_T  = _rescale_temperature(P_val,  T)
        P_test_T = _rescale_temperature(P_test, T)
        nll_val_sweep[i]  = metric_nll(P_val_T,  y_val)
        nll_test_sweep[i] = metric_nll(P_test_T, y_test)

    T_star_idx = int(np.argmin(nll_val_sweep))
    T_star     = float(temps[T_star_idx])
    print(f"  T* (val-NLL optimal) = {T_star:.4f}   "
          f"(val NLL = {nll_val_sweep[T_star_idx]:.4f})")

    # ── Compute metrics at T_frozen and T* ───────────────────────────────────
    rows = []
    for label, T in [("frozen_T0.1", T_FROZEN), (f"T*={T_star:.4f}", T_star)]:
        P_t_val  = _rescale_temperature(P_val,  T)
        P_t_test = _rescale_temperature(P_test, T)
        row = {
            "regime":    regime,
            "setting":   label,
            "T":         round(T, 6),
            "val_NLL":   round(metric_nll(P_t_val,  y_val),        4),
            "val_RPS":   round(metric_rps(P_t_val,  y_val),        4),
            "val_ECE":   round(metric_ece(P_t_val,  y_val),        4),
            "val_sharp": round(metric_sharpness(P_t_val),          4),
            "test_NLL":  round(metric_nll(P_t_test, y_test),       4),
            "test_RPS":  round(metric_rps(P_t_test, y_test),       4),
            "test_ECE":  round(metric_ece(P_t_test, y_test),       4),
            "test_sharp":round(metric_sharpness(P_t_test),         4),
        }
        rows.append(row)

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"\n  {'Setting':<20} {'T':>8} | "
          f"{'valNLL':>8} {'valRPS':>8} {'valECE':>8} {'valSharp':>9} | "
          f"{'tstNLL':>8} {'tstRPS':>8} {'tstECE':>8} {'tstSharp':>9}")
    print("  " + "-" * 100)
    for r in rows:
        print(f"  {r['setting']:<20} {r['T']:>8.4f} | "
              f"{r['val_NLL']:>8.4f} {r['val_RPS']:>8.4f} "
              f"{r['val_ECE']:>8.4f} {r['val_sharp']:>9.4f} | "
              f"{r['test_NLL']:>8.4f} {r['test_RPS']:>8.4f} "
              f"{r['test_ECE']:>8.4f} {r['test_sharp']:>9.4f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    result = {
        "regime":   regime,
        "T_frozen": T_FROZEN,
        "T_star":   T_star,
        "T_sweep":  temps.tolist(),
        "nll_val_sweep":  nll_val_sweep.tolist(),
        "nll_test_sweep": nll_test_sweep.tolist(),
        "table":    rows,
    }
    json_path = os.path.join(RESULTS_DIR, f"calibration_regime{regime}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved JSON  → {json_path}")

    # ── Reliability diagram at T* (test set) ─────────────────────────────────
    P_star_test = _rescale_temperature(P_test, T_star)
    rel_path = os.path.join(RESULTS_DIR, f"reliability_diagram_regime{regime}.png")
    _reliability_diagram(
        P_star_test, y_test,
        title=f"Reliability Diagram — Regime {regime} (T*={T_star:.3f}, test set)",
        out_path=rel_path,
    )
    print(f"  Saved PNG   → {rel_path}")

    # ── NLL-vs-T plot ────────────────────────────────────────────────────────
    nll_path = os.path.join(RESULTS_DIR, f"nll_vs_T_regime{regime}.png")
    _nll_vs_T_plot(
        temps, nll_val_sweep, nll_test_sweep, T_star,
        title=f"NLL vs Temperature — Regime {regime}",
        out_path=nll_path,
    )
    print(f"  Saved PNG   → {nll_path}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def task3():
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

    # Combined summary CSV
    summary = pd.DataFrame(all_rows)
    csv_path = os.path.join(RESULTS_DIR, "calibration_summary.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nCombined summary saved → {csv_path}")

    print("\n\nFULL CALIBRATION SUMMARY (both regimes)")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    task3()
