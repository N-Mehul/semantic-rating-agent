"""Comprehensive Multi-Dataset Evaluation for Phase 6 Improved SSR vs Canonical SSR.

Evaluates both methods strictly on actual available data in the repository:
1. Original Phase 6 Evaluation (24 synthetic responses on Nova AI Smartphone v3, evaluated against Qwen 2.5 7B blind judge).
2. Phase 6 Baseline Evaluation (24 baseline synthetic responses on Nova AI Smartphone, evaluated against Qwen 2.5 7B blind judge).
3. Human Dataset Evaluation (Cleaned empirical survey responses, Q26 open-ended text vs Q21 ground truth rebuy intention).
4. Persona Validation 400 Synthetic Dataset (400 responses across 4 personas & 6 constructs).

Outputs:
- evaluation/phase6_improved_integration/metrics_comparison.json
- evaluation/phase6_improved_integration/metrics_summary_table.csv
- evaluation/phase6_improved_integration/per_rating_comparison.csv
- evaluation/phase6_improved_integration/per_persona_comparison.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from canonical_ssr import expected_rating, similarities_to_pmf
from ssr_phase6_improved import SSR_Phase6_Improved, phase6_similarities_to_pmf

OUT_DIR = Path(__file__).parent


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_cont: np.ndarray, labels: list[int] = [1, 2, 3, 4, 5]) -> Dict[str, Any]:
    """Compute exact acc, within-1 acc, MAE, RMSE, Spearman, QWK, Linear Kappa, Macro F1, and distributions."""
    y = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    e = np.asarray(y_cont, dtype=float)
    n = len(y)

    exact_acc = float((yp == y).mean())
    within_1_acc = float((np.abs(yp - y) <= 1).mean())
    mae_cont = float(np.mean(np.abs(e - y)))
    rmse_cont = float(np.sqrt(np.mean((e - y) ** 2)))
    mae_disc = float(np.mean(np.abs(yp - y)))
    rmse_disc = float(np.sqrt(np.mean((yp - y) ** 2)))

    # Spearman & Pearson
    if np.ptp(y) > 0 and np.ptp(e) > 0:
        sp_r, sp_p = stats.spearmanr(e, y)
        pe_r, pe_p = stats.pearsonr(e, y)
    else:
        sp_r, sp_p = None, None
        pe_r, pe_p = None, None

    # Cohen's Kappa
    labs_present = sorted(list(set(y) | set(yp)))
    if len(labs_present) > 1:
        qwk = float(cohen_kappa_score(y, yp, weights="quadratic", labels=labels))
        lwk = float(cohen_kappa_score(y, yp, weights="linear", labels=labels))
    else:
        qwk, lwk = None, None

    # F1 and per-rating
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=labels, zero_division=0)
    per_rating = {}
    for i, r in enumerate(labels):
        per_rating[str(r)] = {
            "precision": float(pr[i]),
            "recall": float(rc[i]),
            "f1": float(f1[i]),
            "support": int(sup[i]),
            "n_pred": int((yp == r).sum()),
        }

    macro_f1 = float(f1_score(y, yp, labels=labs_present, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y, yp, labels=labs_present, average="weighted", zero_division=0))

    # Confusion matrix
    conf = confusion_matrix(y, yp, labels=labels).tolist()

    dist_ref = {str(r): int((y == r).sum()) for r in labels}
    dist_pred = {str(r): int((yp == r).sum()) for r in labels}

    return {
        "n": n,
        "exact_accuracy": exact_acc,
        "within_1_accuracy": within_1_acc,
        "mae_continuous": mae_cont,
        "rmse_continuous": rmse_cont,
        "mae_discrete": mae_disc,
        "rmse_discrete": rmse_disc,
        "spearman_r": float(sp_r) if sp_r is not None else None,
        "spearman_p": float(sp_p) if sp_p is not None else None,
        "pearson_r": float(pe_r) if pe_r is not None else None,
        "pearson_p": float(pe_p) if pe_p is not None else None,
        "qwk": qwk,
        "linear_kappa": lwk,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "dist_ref": dist_ref,
        "dist_pred": dist_pred,
        "confusion_matrix": conf,
        "per_rating": per_rating,
    }


def evaluate_dataset_A_original_phase6():
    """Evaluate on the 24 synthetic responses from Phase 6 (Nova AI Smartphone v3 improved)."""
    csv_path = ROOT / "evaluation/generation_quality_improvement_v1/phase6_ssr/ssr_scores_with_reference.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df_imp = df[df["set"] == "improved"].copy()
    L = [1, 2, 3, 4, 5]
    cos_cols = [f"cos_{k}" for k in L]

    # Re-score using exact methods
    can_E, can_pred = [], []
    imp_E, imp_pred = [], []
    for _, row in df_imp.iterrows():
        sims = row[cos_cols].values.astype(float)
        p_c = similarities_to_pmf(sims)
        p_i = phase6_similarities_to_pmf(sims)
        e_c = expected_rating(p_c)
        e_i = expected_rating(p_i)
        can_E.append(e_c)
        can_pred.append(int(np.clip(np.round(e_c), 1, 5)))
        imp_E.append(e_i)
        imp_pred.append(int(np.clip(np.round(e_i), 1, 5)))

    y_ref = df_imp["ref"].values.astype(int)
    met_can = compute_metrics(y_ref, np.array(can_pred), np.array(can_E))
    met_imp = compute_metrics(y_ref, np.array(imp_pred), np.array(imp_E))

    # Persona-wise
    personas = df_imp["persona_id"].unique()
    per_persona = {}
    for p in personas:
        idx = (df_imp["persona_id"] == p).values
        per_persona[p] = {
            "n": int(idx.sum()),
            "canonical_exact": float((np.array(can_pred)[idx] == y_ref[idx]).mean()),
            "canonical_mae": float(np.mean(np.abs(np.array(can_E)[idx] - y_ref[idx]))),
            "phase6_exact": float((np.array(imp_pred)[idx] == y_ref[idx]).mean()),
            "phase6_mae": float(np.mean(np.abs(np.array(imp_E)[idx] - y_ref[idx]))),
        }

    return {
        "dataset_name": "Phase 6 Improved Synthetic (Nova AI v3, n=24, Qwen Blind Judge)",
        "n": len(df_imp),
        "canonical": met_can,
        "phase6_improved": met_imp,
        "per_persona": per_persona,
    }


def evaluate_dataset_B_baseline_synthetic():
    """Evaluate on the 24 baseline synthetic responses from Phase 6 (Nova AI baseline)."""
    csv_path = ROOT / "evaluation/generation_quality_improvement_v1/phase6_ssr/ssr_scores_with_reference.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df_base = df[df["set"] == "baseline"].copy()
    L = [1, 2, 3, 4, 5]
    cos_cols = [f"cos_{k}" for k in L]

    can_E, can_pred = [], []
    imp_E, imp_pred = [], []
    for _, row in df_base.iterrows():
        sims = row[cos_cols].values.astype(float)
        p_c = similarities_to_pmf(sims)
        p_i = phase6_similarities_to_pmf(sims)
        e_c = expected_rating(p_c)
        e_i = expected_rating(p_i)
        can_E.append(e_c)
        can_pred.append(int(np.clip(np.round(e_c), 1, 5)))
        imp_E.append(e_i)
        imp_pred.append(int(np.clip(np.round(e_i), 1, 5)))

    y_ref = df_base["ref"].values.astype(int)
    met_can = compute_metrics(y_ref, np.array(can_pred), np.array(can_E))
    met_imp = compute_metrics(y_ref, np.array(imp_pred), np.array(imp_E))

    per_persona = {}
    for p in df_base["persona_id"].unique():
        idx = (df_base["persona_id"] == p).values
        per_persona[p] = {
            "n": int(idx.sum()),
            "canonical_exact": float((np.array(can_pred)[idx] == y_ref[idx]).mean()),
            "canonical_mae": float(np.mean(np.abs(np.array(can_E)[idx] - y_ref[idx]))),
            "phase6_exact": float((np.array(imp_pred)[idx] == y_ref[idx]).mean()),
            "phase6_mae": float(np.mean(np.abs(np.array(imp_E)[idx] - y_ref[idx]))),
        }

    return {
        "dataset_name": "Phase 6 Baseline Synthetic (Nova AI Baseline, n=24, Qwen Blind Judge)",
        "n": len(df_base),
        "canonical": met_can,
        "phase6_improved": met_imp,
        "per_persona": per_persona,
    }


def evaluate_dataset_C_human_survey():
    """Evaluate on the empirical human survey dataset (Q26 response text scored vs Q21 rebuy Likert rating)."""
    csv_path = ROOT / "evaluation/phase13_human_ssr/phase13_scored_human_q26.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, encoding="utf-8")
    # Columns: q21 (target 1-5), cos_1..cos_5
    cos_cols = [f"cos_{k}" for k in range(1, 6)]

    can_E, can_pred = [], []
    imp_E, imp_pred = [], []
    for _, row in df.iterrows():
        sims = row[cos_cols].values.astype(float)
        p_c = similarities_to_pmf(sims)
        p_i = phase6_similarities_to_pmf(sims)
        e_c = expected_rating(p_c)
        e_i = expected_rating(p_i)
        can_E.append(e_c)
        can_pred.append(int(np.clip(np.round(e_c), 1, 5)))
        imp_E.append(e_i)
        imp_pred.append(int(np.clip(np.round(e_i), 1, 5)))

    y_ref = df["q21"].values.astype(int)
    met_can = compute_metrics(y_ref, np.array(can_pred), np.array(can_E))
    met_imp = compute_metrics(y_ref, np.array(imp_pred), np.array(imp_E))

    per_persona = {}
    if "persona" in df.columns:
        for p in df["persona"].dropna().unique():
            idx = (df["persona"] == p).values
            per_persona[p] = {
                "n": int(idx.sum()),
                "canonical_exact": float((np.array(can_pred)[idx] == y_ref[idx]).mean()),
                "canonical_mae": float(np.mean(np.abs(np.array(can_E)[idx] - y_ref[idx]))),
                "phase6_exact": float((np.array(imp_pred)[idx] == y_ref[idx]).mean()),
                "phase6_mae": float(np.mean(np.abs(np.array(imp_E)[idx] - y_ref[idx]))),
            }

    return {
        "dataset_name": "Empirical Human Survey (Q26 text vs Q21 Ground Truth Likert, n=251)",
        "n": len(df),
        "canonical": met_can,
        "phase6_improved": met_imp,
        "per_persona": per_persona,
    }


def evaluate_dataset_D_synthetic_400():
    """Evaluate on the 400 Synthetic Responses across 4 personas & 6 constructs (canonical vs improved distribution comparison)."""
    csv_path = ROOT / "data/experiments/persona_validation_400/canonical_vs_improved_comparison/canonical_vs_improved_responses.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    # Check if legitimate reference labels exist
    # Note: These synthetic responses do not have independent human ratings or judge ratings in this file.
    # We evaluate distributional metrics, entropy, and persona differentiation.
    can_E = df["Canonical_Expected_Rating"].values
    can_pred = df["Canonical_Rounded"].values
    imp_E = df["Improved_Expected_Rating"].values
    imp_pred = df["Improved_Rounded"].values

    per_persona = {}
    for p in df["Persona_ID"].unique():
        idx = (df["Persona_ID"] == p).values
        per_persona[p] = {
            "n": int(idx.sum()),
            "canonical_mean_E": float(np.mean(can_E[idx])),
            "canonical_std_E": float(np.std(can_E[idx], ddof=1)),
            "phase6_mean_E": float(np.mean(imp_E[idx])),
            "phase6_std_E": float(np.std(imp_E[idx], ddof=1)),
        }

    return {
        "dataset_name": "Synthetic Persona Validation 400 (400 responses, 4 personas, 6 constructs)",
        "n": len(df),
        "ground_truth_available": False,
        "limitation_note": "Synthetic 400 dataset contains no external ground-truth ratings (neither human nor blind LLM judge). Metric evaluation is distributional only.",
        "canonical_summary": {
            "mean_E": float(np.mean(can_E)),
            "std_E": float(np.std(can_E, ddof=1)),
            "dist_pred": {str(r): int((can_pred == r).sum()) for r in range(1, 6)},
        },
        "phase6_summary": {
            "mean_E": float(np.mean(imp_E)),
            "std_E": float(np.std(imp_E, ddof=1)),
            "dist_pred": {str(r): int((imp_pred == r).sum()) for r in range(1, 6)},
        },
        "per_persona": per_persona,
    }


def main():
    print("=" * 80)
    print("RUNNING MULTI-DATASET EVALUATION: SSR_Phase6_Improved vs Canonical SSR")
    print("=" * 80)

    results = {}

    # A. Original Phase 6
    res_A = evaluate_dataset_A_original_phase6()
    if res_A:
        results["original_phase6_improved"] = res_A
        print(f"[A] Evaluated {res_A['dataset_name']}")
        print(f"    Canonical Exact Acc:       {res_A['canonical']['exact_accuracy']*100:.2f}% | MAE: {res_A['canonical']['mae_continuous']:.3f} | QWK: {res_A['canonical']['qwk']:.3f}")
        print(f"    Phase 6 Improved Acc:      {res_A['phase6_improved']['exact_accuracy']*100:.2f}% | MAE: {res_A['phase6_improved']['mae_continuous']:.3f} | QWK: {res_A['phase6_improved']['qwk']:.3f}")

    # B. Phase 6 Baseline Synthetic
    res_B = evaluate_dataset_B_baseline_synthetic()
    if res_B:
        results["original_phase6_baseline"] = res_B
        print(f"[B] Evaluated {res_B['dataset_name']}")
        print(f"    Canonical Exact Acc:       {res_B['canonical']['exact_accuracy']*100:.2f}% | MAE: {res_B['canonical']['mae_continuous']:.3f} | QWK: {res_B['canonical']['qwk']:.3f}")
        print(f"    Phase 6 Improved Acc:      {res_B['phase6_improved']['exact_accuracy']*100:.2f}% | MAE: {res_B['phase6_improved']['mae_continuous']:.3f} | QWK: {res_B['phase6_improved']['qwk']:.3f}")

    # C. Human Dataset
    res_C = evaluate_dataset_C_human_survey()
    if res_C:
        results["human_survey_dataset"] = res_C
        print(f"[C] Evaluated {res_C['dataset_name']}")
        print(f"    Canonical Exact Acc:       {res_C['canonical']['exact_accuracy']*100:.2f}% | MAE: {res_C['canonical']['mae_continuous']:.3f} | QWK: {res_C['canonical']['qwk']:.3f}")
        print(f"    Phase 6 Improved Acc:      {res_C['phase6_improved']['exact_accuracy']*100:.2f}% | MAE: {res_C['phase6_improved']['mae_continuous']:.3f} | QWK: {res_C['phase6_improved']['qwk']:.3f}")

    # D. Synthetic 400
    res_D = evaluate_dataset_D_synthetic_400()
    if res_D:
        results["synthetic_400_dataset"] = res_D
        print(f"[D] Evaluated {res_D['dataset_name']}")
        print(f"    Note: {res_D['limitation_note']}")

    # Save complete JSON
    json_path = OUT_DIR / "metrics_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}")

    # Build summary CSV
    summary_rows = []
    for k, v in results.items():
        if "canonical" in v:
            summary_rows.append({
                "Dataset": v["dataset_name"],
                "Method": "Canonical SSR",
                "N": v["n"],
                "Exact_Acc": v["canonical"]["exact_accuracy"],
                "Within1_Acc": v["canonical"]["within_1_accuracy"],
                "MAE_Continuous": v["canonical"]["mae_continuous"],
                "RMSE_Continuous": v["canonical"]["rmse_continuous"],
                "MAE_Discrete": v["canonical"]["mae_discrete"],
                "RMSE_Discrete": v["canonical"]["rmse_discrete"],
                "Spearman_r": v["canonical"]["spearman_r"],
                "QWK": v["canonical"]["qwk"],
                "Linear_Kappa": v["canonical"]["linear_kappa"],
                "Macro_F1": v["canonical"]["macro_f1"],
            })
            summary_rows.append({
                "Dataset": v["dataset_name"],
                "Method": "SSR_Phase6_Improved",
                "N": v["n"],
                "Exact_Acc": v["phase6_improved"]["exact_accuracy"],
                "Within1_Acc": v["phase6_improved"]["within_1_accuracy"],
                "MAE_Continuous": v["phase6_improved"]["mae_continuous"],
                "RMSE_Continuous": v["phase6_improved"]["rmse_continuous"],
                "MAE_Discrete": v["phase6_improved"]["mae_discrete"],
                "RMSE_Discrete": v["phase6_improved"]["rmse_discrete"],
                "Spearman_r": v["phase6_improved"]["spearman_r"],
                "QWK": v["phase6_improved"]["qwk"],
                "Linear_Kappa": v["phase6_improved"]["linear_kappa"],
                "Macro_F1": v["phase6_improved"]["macro_f1"],
            })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = OUT_DIR / "metrics_summary_table.csv"
    df_summary.to_csv(csv_path, index=False)
    print(f"Saved summary CSV to: {csv_path}")

    # Build per-rating CSV
    rating_rows = []
    for k in ["original_phase6_improved", "original_phase6_baseline", "human_survey_dataset"]:
        if k in results:
            v = results[k]
            for m_name, m_key in [("Canonical SSR", "canonical"), ("SSR_Phase6_Improved", "phase6_improved")]:
                for r_str, r_data in v[m_key]["per_rating"].items():
                    rating_rows.append({
                        "Dataset": v["dataset_name"],
                        "Method": m_name,
                        "Rating": int(r_str),
                        "Support": r_data["support"],
                        "N_Pred": r_data["n_pred"],
                        "Precision": r_data["precision"],
                        "Recall": r_data["recall"],
                        "F1": r_data["f1"],
                    })
    df_ratings = pd.DataFrame(rating_rows)
    ratings_csv = OUT_DIR / "per_rating_comparison.csv"
    df_ratings.to_csv(ratings_csv, index=False)
    print(f"Saved per-rating CSV to: {ratings_csv}")

    # Build per-persona CSV
    persona_rows = []
    for k in ["original_phase6_improved", "original_phase6_baseline", "human_survey_dataset"]:
        if k in results and "per_persona" in results[k]:
            v = results[k]
            for p_id, p_data in v["per_persona"].items():
                persona_rows.append({
                    "Dataset": v["dataset_name"],
                    "Persona": p_id,
                    "N": p_data["n"],
                    "Canonical_Exact_Acc": p_data.get("canonical_exact"),
                    "Canonical_MAE": p_data.get("canonical_mae"),
                    "Phase6_Exact_Acc": p_data.get("phase6_exact"),
                    "Phase6_MAE": p_data.get("phase6_mae"),
                })
    df_persona = pd.DataFrame(persona_rows)
    persona_csv = OUT_DIR / "per_persona_comparison.csv"
    df_persona.to_csv(persona_csv, index=False)
    print(f"Saved per-persona CSV to: {persona_csv}")


if __name__ == "__main__":
    main()

