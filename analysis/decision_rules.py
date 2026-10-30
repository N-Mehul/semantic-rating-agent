import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from analysis.splits import DATA_PATH, RESULTS_DIR, hash_text, normalize_text
from analysis.label_structure import (
    get_frozen_agent,
    get_baseline_predictions,
    build_smoothed_distributions,
    bootstrap_metrics,
    format_cell,
)

# Helper metric functions (same as in label_structure)
def exact_acc(y_pred, y_true):
    return np.mean(y_pred == y_true)

def within_1(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true) <= 1)

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def rmse(y_pred, y_true):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def mode_from_dist(dist: Dict[int, float]) -> int:
    """Return rating with highest probability; break ties by lower rating."""
    return max(dist.items(), key=lambda x: (x[1], -x[0]))[0]

def median_from_dist(dist: Dict[int, float]) -> int:
    """Return the smallest rating where cumulative probability >= 0.5."""
    cum = 0.0
    for r in range(1, 6):
        cum += dist.get(r, 0.0)
        if cum >= 0.5:
            return r
    return 5

def expected_from_dist(dist: Dict[int, float]) -> float:
    return sum(r * p for r, p in dist.items())

def round_expected(val: float) -> int:
    return int(round(val))

def evaluate_rule(rule_name: str, preds: np.ndarray, y_true: np.ndarray) -> Tuple[float, Tuple[float, float], float, Tuple[float, float], float, Tuple[float, float], float, Tuple[float, float]]:
    """Compute metrics and 95% bootstrap CIs for a single rule.
    Returns a tuple with (exact, exact_ci, within1, within1_ci, mae, mae_ci, rmse, rmse_ci).
    """
    exact = exact_acc(preds, y_true)
    w1 = within_1(preds, y_true)
    m = mae(preds, y_true)
    r = rmse(preds, y_true)
    exact_ci = bootstrap_metrics(preds, y_true, exact_acc)
    w1_ci = bootstrap_metrics(preds, y_true, within_1)
    mae_ci = bootstrap_metrics(preds, y_true, mae)
    rmse_ci = bootstrap_metrics(preds, y_true, rmse)
    return (exact, exact_ci, w1, w1_ci, m, mae_ci, r, rmse_ci)

def confusion_matrix_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Return a confusion matrix as a pandas DataFrame (rows = true, cols = pred)."""
    return pd.crosstab(pd.Series(y_true, name='True'), pd.Series(y_pred, name='Pred'))


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           rule_name: str, regime: str) -> str:
    """Save a full 5×5 confusion matrix (rows=true, cols=pred) to results/cm_<regime>_<rule>.csv.

    Guarantees all rating labels 1-5 appear as both rows and columns even if
    absent in this split, so downstream readers always get a 5×5 matrix.
    Returns the path of the written file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    labels = list(range(1, 6))
    cm = pd.crosstab(
        pd.Series(y_true, name='True'),
        pd.Series(y_pred, name='Pred'),
    ).reindex(index=labels, columns=labels, fill_value=0)
    path = os.path.join(RESULTS_DIR, f"cm_regime{regime}_{rule_name}.csv")
    cm.to_csv(path)
    return path


def task2():
    # Load data
    df = pd.read_excel(DATA_PATH, header=None)
    # column handling (same as in splits)
    cols_25 = [
        "review_id", "name", "age", "brand", "model", "price_usd",
        "price_local", "currency", "exchange_rate_to_usd", "rating",
        "review_text", "sentiment", "country", "language", "date",
        "verified_purchase", "battery_life_rating", "camera_rating",
        "performance_rating", "design_rating", "display_rating",
        "review_length", "word_count", "helpful_votes", "source"
    ]
    if len(df.columns) == 25:
        df.columns = cols_25
    else:
        if 'review_text' not in df.columns:
            df = df.rename(columns={10: 'review_text', 9: 'rating'})
    df['review_text'] = df['review_text'].astype(str)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Load splits
    splits = {}
    for regime in ['A', 'B']:
        with open(os.path.join(RESULTS_DIR, f'splits_regime{regime}.json'), 'r') as f:
            splits[regime] = json.load(f)

    agent = get_frozen_agent()
    alphas = [0, 0.5, 1, 2, 5]

    for regime in ['A', 'B']:
        print(f"\n================ REGIME {regime} ================\n")
        train_idx = splits[regime]['train']
        val_idx = splits[regime]['val']
        test_idx = splits[regime]['test']
        df_train = df.iloc[train_idx].copy()
        # Ensure the hash column is present in the split
        if 'text_hash' not in df_train.columns:
            df_train['text_hash'] = df_train['review_text'].apply(lambda t: hash_text(normalize_text(t)))

        df_val = df.iloc[val_idx].copy()
        if 'text_hash' not in df_val.columns:
            df_val['text_hash'] = df_val['review_text'].apply(lambda t: hash_text(normalize_text(t)))

        df_test = df.iloc[test_idx].copy()
        if 'text_hash' not in df_test.columns:
            df_test['text_hash'] = df_test['review_text'].apply(lambda t: hash_text(normalize_text(t)))
        # Ensure we are scoring exactly the frozen test split size
        assert len(df_test) == len(test_idx), f"Test split size mismatch for Regime {regime}: expected {len(test_idx)} rows, got {len(df_test)}"
        # Build per‑text counts for smoothing (oracle) – needed for prior‑corrected mode
        counts_dict = {}
        for th, group in df_train.groupby('text_hash'):
            ratings = group['rating'].dropna().astype(int)
            counts_dict[th] = ratings.value_counts().to_dict()
        global_counts = df_train['rating'].value_counts().to_dict()

        # Determine NLL‑optimal α for this regime (same logic as in label_structure)
        best_alpha_nll = None
        best_nll = float('inf')
        for alpha in alphas:
            smoothed = build_smoothed_distributions(counts_dict, global_counts, alpha)
            nll_vals = []
            for _, row in df_val.iterrows():
                th = row['text_hash']
                true_r = int(row['rating'])
                prob = smoothed.get(th)
                if prob is None:
                    n_global = sum(global_counts.values())
                    prob = {r: (global_counts.get(r, 0) + alpha) / (n_global + 5 * alpha) for r in range(1, 6)}
                p = prob.get(true_r, 1e-12)
                nll_vals.append(-np.log(p))
            avg_nll = np.mean(nll_vals)
            if avg_nll < best_nll:
                best_nll = avg_nll
                best_alpha_nll = alpha
        print(f"NLL‑optimal α for Regime {regime}: {best_alpha_nll} (NLL = {best_nll:.4f})")

        # Baseline distributions from kNN
        baseline_dists = get_baseline_predictions(agent, df_test)
        y_true = df_test['rating'].values.astype(int)
        # Compute fallback count: rows where raw kNN distribution equals the global prior (within tolerance)
        # This helps verify that the prior fallback is not overly common.
        n_global = sum(global_counts.values())
        prior_fallback = {r: (global_counts.get(r, 0) + best_alpha_nll) / (n_global + 5 * best_alpha_nll) for r in range(1, 6)}
        fallback_count = 0
        for th_debug in df_test['text_hash']:
            raw_dist_debug = baseline_dists.get(th_debug, {})
            # Compare each rating probability
            max_diff = max(abs(raw_dist_debug.get(r, 0.0) - prior_fallback[r]) for r in range(1, 6))
            if max_diff < 1e-9:
                fallback_count += 1
        print(f"Regime {regime}: rows where P(r|x) matches prior fallback = {fallback_count}")

        # Containers for rule predictions
        preds = {
            'mode': [],
            'median': [],
            'expected': [],
            'rounded_expected': [],
            'prior_corrected_mode': []
        }
        # Smoothed distributions for test (for prior‑corrected mode)
        smoothed_test = build_smoothed_distributions(counts_dict, global_counts, best_alpha_nll)
        # Diagnostic print for the exact hash from the original collapse warning
        target_hash = "84b7b4050dd6edbe1c8514e8986724454a7c090044986dd45edc501adcbe1c9"
        if target_hash in df_test['text_hash'].values:
            th_debug = target_hash
            raw_dist = baseline_dists.get(th_debug, {})
            # Compute global prior P_train(r)
            n_global = sum(global_counts.values())
            prior = {r: (global_counts.get(r, 0) + best_alpha_nll) / (n_global + 5 * best_alpha_nll) for r in range(1, 6)}
            # Apply prior correction (current code divides by prior)
            corrected_raw = {r: raw_dist.get(r, 0.0) / prior[r] if prior[r] != 0 else 0.0 for r in range(1, 6)}
            sum_corr = sum(corrected_raw.values())
            corrected_norm = {r: (corrected_raw[r] / sum_corr) if sum_corr != 0 else 0.0 for r in range(1, 6)}
            print("Diagnostic for row with hash", th_debug)
            print("P(r|x) raw distribution:", raw_dist)
            print("P_train(r) prior:", prior)
            print("P'(r|x) before renorm (division):", corrected_raw)
            print("P'(r|x) after renorm:", corrected_norm)
        else:
            print("Exact hash", target_hash, "not found in test split.")
        for th in df_test['text_hash']:
            # Base predictions
            dist = baseline_dists[th]
            preds['mode'].append(mode_from_dist(dist))
            preds['median'].append(median_from_dist(dist))
            exp_val = expected_from_dist(dist)
            preds['expected'].append(exp_val)
            preds['rounded_expected'].append(round_expected(exp_val))
            # Prior‑corrected mode: divide raw kNN distribution by the global prior and renormalize
            raw_dist = baseline_dists.get(th, {})
            # Compute global prior P_train(r) (same α as best_alpha_nll)
            n_global = sum(global_counts.values())
            prior = {r: (global_counts.get(r, 0) + best_alpha_nll) / (n_global + 5 * best_alpha_nll) for r in range(1, 6)}
            corrected_raw = {r: raw_dist.get(r, 0.0) / prior[r] if prior[r] != 0 else 0.0 for r in range(1, 6)}
            sum_corr = sum(corrected_raw.values())
            corrected_norm = {r: (corrected_raw[r] / sum_corr) if sum_corr != 0 else 0.0 for r in range(1, 6)}
            preds['prior_corrected_mode'].append(mode_from_dist(corrected_norm))

        # Sanity check: ensure prior_corrected_mode does not collapse to a single class
        unique_preds = np.unique(preds['prior_corrected_mode'])
        if unique_preds.size <= 1:
            print(f"WARNING: prior_corrected_mode collapsed to a single class: {unique_preds}")
            # Show a few example corrected probability vectors for debugging
            sample_idxs = np.random.choice(len(df_test), size=min(5, len(df_test)), replace=False)
            for i in sample_idxs:
                th = df_test['text_hash'].iloc[i]
                probs = smoothed_test.get(th)
                if probs is None:
                    n_global = sum(global_counts.values())
                    probs = {r: (global_counts.get(r, 0) + best_alpha_nll) / (n_global + 5 * best_alpha_nll) for r in range(1, 6)}
                print(f"Row {i} text_hash {th}: probs {probs}")

        for k in preds:
            preds[k] = np.array(preds[k])
        # Sanity check: prior_corrected_mode should produce more than one unique class
        unique_pc = np.unique(preds['prior_corrected_mode'])
        if unique_pc.size <= 1:
            raise AssertionError(f"prior_corrected_mode collapsed to a single class after fix: {unique_pc}")

        print("--- Metrics per decoding rule (with 95% CI) ---")
        header = f"{'Rule':<25} | {'Exact (95% CI)':<25} | {'Within ±1 (95% CI)':<22} | {'MAE (95% CI)':<25} | {'RMSE (95% CI)'}"
        print(header)
        print('-' * len(header))
        for rule_name, y_pred in preds.items():
            if rule_name == 'expected':
                mae_val = mae(y_pred, y_true)
                mae_ci = bootstrap_metrics(y_pred, y_true, mae)
                rmse_val = rmse(y_pred, y_true)
                rmse_ci = bootstrap_metrics(y_pred, y_true, rmse)
                mae_str = format_cell(mae_val, *mae_ci)
                rmse_str = format_cell(rmse_val, *rmse_ci)
                print(f"{rule_name:<25} | {'N/A':<25} | {'N/A':<22} | {mae_str:<25} | {rmse_str}")
                continue
            exact, exact_ci, w1, w1_ci, mae_val, mae_ci, rmse_val, rmse_ci = evaluate_rule(rule_name, y_pred, y_true)
            exact_str = format_cell(exact, *exact_ci)
            w1_str = format_cell(w1, *w1_ci)
            mae_str = format_cell(mae_val, *mae_ci)
            rmse_str = format_cell(rmse_val, *rmse_ci)
            print(f"{rule_name:<25} | {exact_str:<25} | {w1_str:<22} | {mae_str:<25} | {rmse_str}")

        print("\n--- Confusion matrices (True vs Pred) ---")
        for rule_name in ['mode', 'median', 'rounded_expected', 'prior_corrected_mode']:
            cm = confusion_matrix_df(y_true, preds[rule_name])
            print(f"\nConfusion matrix for rule: {rule_name}")
            print(cm)
            saved_path = save_confusion_matrix(y_true, preds[rule_name], rule_name, regime)
            print(f"  -> Saved to {saved_path}")

if __name__ == '__main__':
    task2()
