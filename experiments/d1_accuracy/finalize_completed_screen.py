"""Consolidate a completed grouped-CV screen and evaluate CV champions.

Run from repository root:
    python -m experiments.d1_accuracy.finalize_completed_screen

This consumes the already completed screen's OOF confusion matrices and saved
fold summaries, tunes only selected finalists on grouped training folds, and
uses the fixed workbook only once for final selected-model evaluation.
"""

from __future__ import annotations

import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, GroupKFold

from experiments.d1_accuracy.run_experiments import (
    ASPECTS,
    CLASSES,
    CONFIG_PATH,
    FEATURE_GROUPS,
    OUT,
    ROOT,
    build_pipeline,
    exact_d1_reproduction,
    read_inputs,
    rows_for_group,
    safe_slug,
)

SCREEN_RUN = "run_20261011_214543"
SCREEN_DIR = OUT / "confusion_matrices" / SCREEN_RUN
SCREEN_SUMMARY = OUT / "grouped_cv_screen_results.csv"

# Small, bounded grids. The text-only Random Forest was the best accuracy
# screen, but its prior grouped search was interrupted; it remains untuned.
TUNING_GRIDS = {
    "A_d1_aspects_diagnostic": {"model__C": [0.1, 1.0, 10.0]},
    "B_eligible_numeric": {"model__C": [0.1, 1.0, 10.0]},
    "C_eligible_structured": {"model__C": [0.1, 1.0, 10.0]},
    "D_product_metadata": {"model__max_iter": [70, 120], "model__max_leaf_nodes": [15, 25]},
    "F_text_plus_structured": {"model__C": [0.1, 1.0, 10.0]},
    "G_train_selected_structured": {"model__C": [0.1, 1.0, 10.0]},
    "H_aspects_engineered_diagnostic": {"model__n_estimators": [70, 120], "model__max_depth": [3, 5]},
    "I_no_aspects_with_sentiment": {"model__max_depth": [8, 14], "model__min_samples_leaf": [1, 3]},
}


def metrics_from_predictions(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def predictions_from_confusion(matrix):
    true_values = []
    predicted_values = []
    for true_idx, true_rating in enumerate(CLASSES):
        for pred_idx, pred_rating in enumerate(CLASSES):
            count = int(matrix[true_idx, pred_idx])
            if count:
                true_values.extend([int(true_rating)] * count)
                predicted_values.extend([int(pred_rating)] * count)
    return np.asarray(true_values), np.asarray(predicted_values)


def class_rows(y_true, y_pred, group, model, protocol):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    return [
        {"feature_group": group, "model": model, "protocol": protocol,
         "class": int(label), "precision": float(pp), "recall": float(rr),
         "f1": float(ff), "support": int(ss)}
        for label, pp, rr, ff, ss in zip(CLASSES, p, r, f, s)
    ]


def write_confusion(path, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    pd.DataFrame(matrix, index=[f"actual_{c}" for c in CLASSES],
                 columns=[f"predicted_{c}" for c in CLASSES]).to_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="run selected bounded grouped-CV tuning grids")
    parser.add_argument(
        "--tune-groups",
        nargs="+",
        default=["A_d1_aspects_diagnostic", "B_eligible_numeric", "C_eligible_structured", "F_text_plus_structured", "G_train_selected_structured"],
        help="feature groups to tune; always uses grouped training folds only",
    )
    args = parser.parse_args()
    tune_groups = set(args.tune_groups) if args.tune else set()
    if not SCREEN_DIR.exists() or not SCREEN_SUMMARY.exists():
        raise FileNotFoundError("Completed grouped-CV screen files are missing")
    timestamp = datetime.now().strftime("finalized_%Y%m%d_%H%M%S")
    confusion_dir = OUT / "confusion_matrices" / timestamp
    models_dir = OUT / "models" / timestamp
    confusion_dir.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir(parents=True, exist_ok=False)

    screen = pd.read_csv(SCREEN_SUMMARY, skipinitialspace=True)
    comparison_rows = []
    per_class = []
    for record in screen.to_dict("records"):
        group = record["feature_group"]
        model = record["model"]
        row = dict(record)
        row.update({
            "grouped_oof_accuracy": np.nan,
            "grouped_oof_macro_f1": np.nan,
            "grouped_oof_weighted_f1": np.nan,
            "grouped_oof_mae": np.nan,
            "grouped_oof_rmse": np.nan,
            "grouped_oof_spearman": np.nan,
            "cv_fit_seconds": np.nan,
            "cv_predict_seconds": np.nan,
            "final_test_evaluated": False,
            "selected_by_cv": False,
        })
        if row["status"] == "ok":
            matrix_path = SCREEN_DIR / f"{safe_slug(group)}__{safe_slug(model)}__grouped_oof.csv"
            if not matrix_path.exists():
                row["status"] = "missing_oof_confusion"
                row["notes"] = "No OOF confusion matrix found for this screen row"
            else:
                matrix = pd.read_csv(matrix_path, index_col=0).to_numpy(dtype=int)
                y_true, y_pred = predictions_from_confusion(matrix)
                oof = metrics_from_predictions(y_true, y_pred)
                row.update({
                    "grouped_oof_accuracy": oof["accuracy"],
                    "grouped_oof_macro_f1": oof["macro_f1"],
                    "grouped_oof_weighted_f1": oof["weighted_f1"],
                    "grouped_oof_mae": oof["mae"],
                    "grouped_oof_rmse": oof["rmse"],
                    "grouped_oof_spearman": oof["spearman"],
                })
                per_class.extend(class_rows(y_true, y_pred, group, model, "grouped_text_oof"))
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows)
    valid_screen = comparison.loc[comparison.status == "ok"].copy()
    champions = {}
    for group, group_rows in valid_screen.groupby("feature_group"):
        winner = group_rows.sort_values(
            ["cv_accuracy_mean", "cv_macro_f1_mean"], ascending=False
        ).iloc[0]
        champions[group] = str(winner.model)
        comparison.loc[
            (comparison.feature_group == group) & (comparison.model == winner.model),
            "selected_by_cv",
        ] = True

    train, eval_frame = read_inputs()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    folds = int(config["group_cv_folds"])
    seed = int(config["seed"])
    threads = int(config["threads"])
    feature_rows = []
    tuned = []
    final_metrics = {}

    for group, screen_model in champions.items():
        selected_model = screen_model
        parameters = {}
        tune_cv_accuracy = np.nan
        tuning_status = "not_tuned"
        grid = TUNING_GRIDS.get(group) if group in tune_groups else None
        # Avoid repeating the expensive, previously interrupted text-RF tuning.
        if group == "E_text_only":
            tuning_status = "skipped_resource_cost"
        elif not args.tune or group not in tune_groups:
            tuning_status = "not_requested"
        elif grid:
            frame = rows_for_group(train, group).reset_index(drop=True)
            features = FEATURE_GROUPS[group]
            X = frame[features]
            y = frame.rating.to_numpy(dtype=int) - 1
            groups = frame.review_text.astype(str).to_numpy()
            splits = list(GroupKFold(n_splits=folds).split(X, y, groups))
            search = GridSearchCV(
                build_pipeline(group, screen_model, seed, threads),
                param_grid=grid,
                scoring="accuracy",
                cv=splits,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search_start = time.perf_counter()
            search.fit(X, y)
            tune_seconds = time.perf_counter() - search_start
            parameters = search.best_params_
            tune_cv_accuracy = float(search.best_score_)
            tuning_status = "completed"
            tuned.append({
                "feature_group": group,
                "screen_model": screen_model,
                "best_parameters_json": json.dumps(parameters, default=str),
                "grouped_cv_best_accuracy": tune_cv_accuracy,
                "grouped_cv_accuracy_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "tuning_seconds": tune_seconds,
                "test_used_for_selection": False,
            })

        final_train = rows_for_group(train, group).reset_index(drop=True)
        X_train = final_train[FEATURE_GROUPS[group]]
        y_train = final_train.rating.to_numpy(dtype=int) - 1
        groups = final_train.review_text.astype(str).to_numpy()
        X_test = eval_frame[FEATURE_GROUPS[group]]
        y_test = eval_frame.rating.to_numpy(dtype=int)
        model = build_pipeline(group, screen_model, seed, threads)
        if parameters:
            model.set_params(**parameters)
        if screen_model == "KNN":
            from experiments.d1_accuracy.run_experiments import cap_training_indices, KNN_TRAIN_CAP
            fit_indices = cap_training_indices(np.arange(len(final_train)), y_train,
                                               KNN_TRAIN_CAP, seed + 1000)
        else:
            fit_indices = np.arange(len(final_train))
        fit_start = time.perf_counter()
        model.fit(X_train.iloc[fit_indices], y_train[fit_indices])
        fit_seconds = time.perf_counter() - fit_start
        predict_start = time.perf_counter()
        prediction = model.predict(X_test).astype(int) + 1
        predict_seconds = time.perf_counter() - predict_start
        result = metrics_from_predictions(y_test, prediction)
        final_metrics[group] = result
        write_confusion(confusion_dir / f"{safe_slug(group)}__fixed_id_holdout.csv", y_test, prediction)
        per_class.extend(class_rows(y_test, prediction, group, screen_model, "fixed_id_holdout_champion"))
        model_path = models_dir / f"{safe_slug(group)}__champion.joblib"
        joblib.dump(model, model_path)
        screen_row = valid_screen.loc[
            (valid_screen.feature_group == group) & (valid_screen.model == screen_model)
        ].iloc[0]
        feature_rows.append({
            "feature_group": group,
            "included_features": json.dumps(FEATURE_GROUPS[group]),
            "screen_champion": screen_model,
            "best_parameters": json.dumps(parameters, default=str),
            "tuning_status": tuning_status,
            "grouped_cv_accuracy_mean": float(screen_row.cv_accuracy_mean),
            "grouped_cv_accuracy_std": float(screen_row.cv_accuracy_std),
            "grouped_cv_macro_f1_mean": float(screen_row.cv_macro_f1_mean),
            "grouped_cv_macro_f1_std": float(screen_row.cv_macro_f1_std),
            "grouped_oof_accuracy": float(screen_row.grouped_oof_accuracy),
            "grouped_oof_macro_f1": float(screen_row.grouped_oof_macro_f1),
            "fixed_id_holdout_accuracy": result["accuracy"],
            "fixed_id_holdout_macro_f1": result["macro_f1"],
            "fixed_id_holdout_weighted_f1": result["weighted_f1"],
            "fixed_id_holdout_mae": result["mae"],
            "fixed_id_holdout_rmse": result["rmse"],
            "fixed_id_holdout_spearman": result["spearman"],
            "fixed_test_rows": len(eval_frame),
            "final_train_rows": len(final_train),
            "effective_fit_rows": len(fit_indices),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "model_file": str(model_path.relative_to(OUT)),
            "leakage_status": (
                "DIAGNOSTIC_ONLY: simultaneous aspect sub-target inputs"
                if group in ("A_d1_aspects_diagnostic", "H_aspects_engineered_diagnostic")
                else "SENTIMENT_PROVENANCE_CHECK"
                if group == "I_no_aspects_with_sentiment"
                else "No aspect inputs; review feature timing/provenance"
            ),
        })

    # Keep the D1 replay as a distinct legacy metric row, not as a newly selected model.
    d1 = json.loads((ROOT / "data" / "aspect_model" / "metrics.json").read_text(encoding="utf-8"))
    d1_result = exact_d1_reproduction(train, eval_frame, confusion_dir, per_class, threads)
    d1_row = {
        "feature_group": "A_d1_aspects_diagnostic",
        "model": "D1_RF_exact_saved_reproduction",
        "status": "verified" if d1_result["saved_d1_prediction_match"] else "mismatch",
        "n_rows": d1_result["n_rows"],
        "cv_accuracy_mean": np.nan,
        "cv_accuracy_std": np.nan,
        "cv_macro_f1_mean": np.nan,
        "cv_macro_f1_std": np.nan,
        "grouped_oof_accuracy": np.nan,
        "grouped_oof_macro_f1": np.nan,
        "grouped_oof_weighted_f1": np.nan,
        "grouped_oof_mae": np.nan,
        "grouped_oof_rmse": np.nan,
        "grouped_oof_spearman": np.nan,
        "external_accuracy": d1_result["external_accuracy"],
        "external_macro_f1": d1_result["external_macro_f1"],
        "external_weighted_f1": d1_result["external_weighted_f1"],
        "external_mae": d1_result["external_mae"],
        "external_rmse": d1_result["external_rmse"],
        "external_spearman": d1_result["external_spearman"],
        "saved_d1_prediction_match": d1_result["saved_d1_prediction_match"],
        "parameters_json": json.dumps({"n_estimators": 100, "max_depth": 10, "random_state": 42}),
        "error": "",
    }
    comparison = pd.concat([comparison, pd.DataFrame([d1_row])], ignore_index=True, sort=False)
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(OUT / "feature_comparison.csv", index=False)
    pd.DataFrame(per_class).to_csv(OUT / "cv_per_class_metrics.csv", index=False)
    pd.DataFrame(tuned).to_csv(OUT / "tuning_results.csv", index=False)

    metadata = {
        "screen_source_run": SCREEN_RUN,
        "finalized_at": timestamp,
        "fixed_test_metrics_not_used_for_selection": True,
        "d1_reproduction_saved_predictions_exact_match": d1_result["saved_d1_prediction_match"],
        "d1_reproduction_metrics": {key: d1_result[key] for key in ["external_accuracy", "external_macro_f1", "external_weighted_f1", "external_mae", "external_rmse", "external_spearman"]},
        "cv_groups": folds,
        "feature_group_champions": champions,
        "failures": screen.loc[screen.status == "failed", ["feature_group", "model", "notes"]].to_dict("records"),
        "unrun": screen.loc[screen.status == "not_run", ["feature_group", "model", "notes"]].to_dict("records"),
        "limitations": [
            "CV fold mean and standard deviation are imported from the captured completed run log; OOF confusion matrices are preserved in the listed run folder.",
            "CV per-model fit and prediction times were not captured by the interrupted screen, so they are blank for screen rows; final champion and tuning times are measured in this finalization.",
            "Fixed-workbook results are ID-held-out but 90/110 unique review texts overlap source training; interpret as row-level only.",
            "Aspect groups are target-leakage diagnostics; sentiment is derived/annotated and requires provenance before deployment.",
            "CatBoost failed during fit with a shape mismatch and is not included as a result.",
            "No source data or protected baseline/D1 outputs were modified.",
        ],
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    feature_frame = pd.DataFrame(feature_rows).sort_values("grouped_cv_accuracy_mean", ascending=False)
    report = [
        "# D1 Accuracy Experiments — Best Model Report", "",
        "## Evaluation protocol", "",
        "All grouped screening metrics use 3-fold GroupKFold over exact review-text strings; the same exact text never crosses a fold boundary. Hyperparameter searches use those training rows only. The fixed workbook is used only after group-CV model selection. It has 10,003 distinct review IDs but shares 90 of 110 distinct text strings with training; its metrics are ID-held-out, not unseen-text results.",
        "",
        "The source has 110 unique review texts and every text group has multiple target ratings. Grouped CV therefore has only 110 independent text groups and substantial label conflict. All measures must be interpreted accordingly.",
        "",
        "## D1 baseline reproduction", "",
        f"The exact existing D1 Random Forest was re-fit in memory on 39,997 training rows and evaluated on the 10,003-row workbook. Accuracy={d1_result['external_accuracy']:.4%}, macro F1={d1_result['external_macro_f1']:.4%}, weighted F1={d1_result['external_weighted_f1']:.4%}, MAE={d1_result['external_mae']:.4f}, RMSE={d1_result['external_rmse']:.4f}, Spearman={d1_result['external_spearman']:.4f}. Saved prediction vectors match exactly: **{d1_result['saved_d1_prediction_match']}**. The original D1 artifacts were not overwritten.",
        "",
        "## Feature-group champion results", "",
        "Champions are selected by grouped-CV accuracy, with grouped macro F1 as tie-breaker. Fixed workbook metrics below were not used in selection.",
        "",
        "| Feature group | Model | Grouped CV accuracy | CV macro F1 | OOF accuracy | Fixed ID-holdout accuracy | Fixed macro F1 | MAE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in feature_frame.iterrows():
        report.append(
            f"| {r.feature_group} | {r.screen_champion} | {r.grouped_cv_accuracy_mean:.4f} ± {r.grouped_cv_accuracy_std:.4f} | {r.grouped_cv_macro_f1_mean:.4f} ± {r.grouped_cv_macro_f1_std:.4f} | {r.grouped_oof_accuracy:.4f} | {r.fixed_id_holdout_accuracy:.4f} | {r.fixed_id_holdout_macro_f1:.4f} | {r.fixed_id_holdout_mae:.4f} |"
        )
    report += [
        "",
        "## Interpretation", "",
        "No leakage-free result approached 90% on grouped text CV. Text-only RF was the strongest text-only screen at 45.58% ± 0.32% accuracy (macro F1 34.47%). Combining text and structured non-aspect inputs reached 45.41% ± 0.06% with LinearSVC. Adding sentiment yielded a 45.52% ± 0.30% ExtraTrees screen result, but sentiment provenance/availability must be validated. These are consistent with the frozen 46.14% row-level text baseline, but are not directly comparable because this uses grouped unseen-text validation.",
        "",
        "The five aspect-only D1 group reaches 73.66% ± 0.23% grouped-CV accuracy with Logistic Regression; aspect-mean engineered variants reach 73.75% ± 0.27% with XGBoost. Both are explicitly diagnostic-only because the aspects are co-assigned sub-targets. They do not constitute valid text-only or deployment results.",
        "",
        "No 90% claim is supported. The dataset's 110 templated text strings, all with conflicting ratings, and ID-held-out text overlap are the primary limitations. Obtain more independently authored, correctly labeled reviews and evaluate on a genuinely unseen-text test set before raising an accuracy target.",
        "",
        "## Failures and timing gaps", "",
        "CatBoost failed during grouped fit with a shape-mismatch ValueError and is recorded as failed. The full run was interrupted during text-Random-Forest hyperparameter search; that search is marked skipped for resource cost. The finalizer records tuning only for completed bounded finalist grids. Screen-stage per-model fit/predict timings were not captured before the interruption; final champion and tuning timings are in `feature_comparison.csv` and `tuning_results.csv`.",
        "",
        "Files: `model_comparison.csv`, `feature_comparison.csv`, `tuning_results.csv`, `cv_per_class_metrics.csv`, `run_metadata.json`, `confusion_matrices/`, and `models/`. The full screened algorithm metrics are from `grouped_cv_screen_results.csv` plus their grouped OOF confusion matrices under `confusion_matrices/`.",
    ]
    (OUT / "best_model_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Finalized comparison tables, selected-model test evaluations, confusion matrices, and champion models.")


if __name__ == "__main__":
    main()
