"""Grouped-CV model comparison isolated from production and frozen artifacts.

Run from the repository root:
    python -m experiments.d1_accuracy.run_experiments

The fixed workbook is evaluated only after grouped-CV screening/tuning. Exact
review-text groups are kept together during CV. Outputs are never written to
``data/`` and existing experiment outputs are not overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, GridSearchCV

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG_PATH = OUT / "config.json"
KNN_TRAIN_CAP = 2500
CLASSES = np.array([1, 2, 3, 4, 5])
ASPECTS = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
]
SAFE_NUMERIC = ["age", "price_usd", "exchange_rate_to_usd", "review_length", "word_count"]
SAFE_CATEGORICAL = [
    "brand", "model", "currency", "country", "language", "verified_purchase", "source"
]
PRODUCT_METADATA = ["brand", "model", "price_usd", "currency", "source"]
TEXT = ["review_text"]
FEATURE_GROUPS = {
    "A_d1_aspects_diagnostic": ASPECTS,
    "B_eligible_numeric": SAFE_NUMERIC,
    "C_eligible_structured": SAFE_NUMERIC + SAFE_CATEGORICAL,
    "D_product_metadata": PRODUCT_METADATA,
    "E_text_only": TEXT,
    "F_text_plus_structured": TEXT + SAFE_NUMERIC + SAFE_CATEGORICAL,
    "G_train_selected_structured": SAFE_NUMERIC + SAFE_CATEGORICAL,
    "H_aspects_engineered_diagnostic": ASPECTS + [
        "aspect_mean", "aspect_std", "aspect_range", "aspect_min", "aspect_max",
        "battery_minus_camera", "camera_minus_performance",
        "performance_minus_display", "design_minus_display",
    ],
    "I_no_aspects_with_sentiment": TEXT + SAFE_NUMERIC + SAFE_CATEGORICAL + ["sentiment"],
}


class OrdinalThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Cumulative-link approximation using one logistic model per threshold."""

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 5:
            raise ValueError("ordinal model requires all five ordered rating classes")
        self.models_ = []
        for threshold in self.classes_[:-1]:
            binary = (np.asarray(y) > threshold).astype(int)
            model = LogisticRegression(C=self.C, max_iter=self.max_iter)
            model.fit(X, binary)
            self.models_.append(model)
        return self

    def predict_proba(self, X):
        q = np.column_stack([model.predict_proba(X)[:, 1] for model in self.models_])
        q = np.minimum.accumulate(q, axis=1)
        probs = np.column_stack((1.0 - q[:, 0], q[:, :-1] - q[:, 1:], q[:, -1]))
        probs = np.clip(probs, 0.0, 1.0)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RidgeOrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Regularized ordinal regression followed by rounded/clipped star values."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.clip(np.rint(self.model_.predict(X)), self.classes_[0], self.classes_[-1]).astype(int)


def to_dense_float32(matrix):
    if issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def read_inputs():
    train = pd.read_csv(ROOT / "data" / "Mobile Reviews Sentiment.csv", low_memory=False)
    eval_frame = pd.read_excel(ROOT / "data" / "actual_reviews.xlsx", header=None)
    eval_columns = [
        "review_id", "customer_name", "age", "brand", "model", "price_usd", "price_local",
        "currency", "exchange_rate_to_usd", "rating", "review_text", "sentiment", "country",
        "language", "review_date", "verified_purchase", *ASPECTS, "review_length", "word_count",
        "helpful_votes", "source",
    ]
    eval_frame.columns = eval_columns
    train = train.loc[train.rating.notna()].copy()
    eval_frame = eval_frame.loc[eval_frame.rating.notna()].copy()
    train["rating"] = train.rating.astype(int)
    eval_frame["rating"] = eval_frame.rating.astype(int)
    for frame in (train, eval_frame):
        frame["review_text"] = frame.review_text.astype("string")
        for col in ["verified_purchase", "brand", "model", "currency", "country", "language", "source", "sentiment"]:
            frame[col] = frame[col].astype("string")
        aspect_values = frame[ASPECTS].astype(float)
        frame["aspect_mean"] = aspect_values.mean(axis=1)
        frame["aspect_std"] = aspect_values.std(axis=1).fillna(0.0)
        frame["aspect_range"] = aspect_values.max(axis=1) - aspect_values.min(axis=1)
        frame["aspect_min"] = aspect_values.min(axis=1)
        frame["aspect_max"] = aspect_values.max(axis=1)
        frame["battery_minus_camera"] = frame.battery_life_rating - frame.camera_rating
        frame["camera_minus_performance"] = frame.camera_rating - frame.performance_rating
        frame["performance_minus_display"] = frame.performance_rating - frame.display_rating
        frame["design_minus_display"] = frame.design_rating - frame.display_rating
    return train, eval_frame


def rows_for_group(frame: pd.DataFrame, group_name: str) -> pd.DataFrame:
    out = frame.loc[frame.rating.notna()].copy()
    if group_name in ("A_d1_aspects_diagnostic", "H_aspects_engineered_diagnostic"):
        out = out.dropna(subset=["review_text", "rating", *ASPECTS])
    if "review_text" in FEATURE_GROUPS[group_name]:
        out = out.loc[out.review_text.notna()].copy()
        out["review_text"] = out.review_text.astype(str)
    return out


def build_preprocessor(group_name: str):
    features = FEATURE_GROUPS[group_name]
    numeric = [c for c in features if c in SAFE_NUMERIC or c in ASPECTS or c.startswith("aspect_") or "_minus_" in c]
    categorical = [c for c in features if c not in numeric and c != "review_text"]
    transformers = []
    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
        transformers.append(("categorical", cat_pipe, categorical))
    if "review_text" in features:
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        transformers.append(("text", vectorizer, "review_text"))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=1.0)


def available_models(group_name: str):
    names = [
        "Majority", "LogisticRegression", "DecisionTree", "RandomForest", "ExtraTrees",
        "LinearSVC", "GradientBoosting", "HistGradientBoosting", "OrdinalThresholdLogistic",
        "RidgeOrdinal", "XGBoost", "LightGBM", "CatBoost",
    ]
    text_group = "review_text" in FEATURE_GROUPS[group_name]
    if not group_name.startswith(("E_", "F_", "G_", "I_")):
        names.insert(5, "KNN")
    if text_group:
        names = [name for name in names if name not in {
            "GradientBoosting", "HistGradientBoosting", "CatBoost"
        }]
    else:
        # CatBoost was attempted in the first screen and failed during fit due
        # a wrapper/data-shape incompatibility; do not repeatedly rerun it.
        names = [name for name in names if name != "CatBoost"]
    if group_name == "E_text_only":
        names.append("MultinomialNB")
    if group_name == "A_d1_aspects_diagnostic":
        names.append("RandomForest_D1_exact")
    return names


def estimator(name: str, seed: int, threads: int):
    if name == "Majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "LogisticRegression":
        return LogisticRegression(C=1.0, max_iter=1200, random_state=seed)
    if name == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=14, min_samples_leaf=2, random_state=seed)
    if name in ("RandomForest", "RandomForest_D1_exact"):
        if name == "RandomForest_D1_exact":
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=threads)
        return RandomForestClassifier(n_estimators=160, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=threads)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=160, max_depth=14, min_samples_leaf=2, random_state=seed, n_jobs=threads)
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=threads)
    if name == "LinearSVC":
        return LinearSVC(C=1.0, max_iter=4000, random_state=seed)
    if name == "GradientBoosting":
        return GradientBoostingClassifier(n_estimators=70, learning_rate=0.05, max_depth=2, random_state=seed)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=20, l2_regularization=1.0, random_state=seed)
    if name == "OrdinalThresholdLogistic":
        return OrdinalThresholdClassifier(C=1.0, max_iter=1200)
    if name == "RidgeOrdinal":
        return RidgeOrdinalClassifier(alpha=1.0)
    if name == "MultinomialNB":
        return MultinomialNB(alpha=1.0)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             objective="multi:softmax", num_class=5, eval_metric="mlogloss",
                             tree_method="hist", random_state=seed, n_jobs=threads, verbosity=0)
    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=120, learning_rate=0.08, num_leaves=20,
                              verbosity=-1, random_state=seed, n_jobs=threads)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=120, depth=6, learning_rate=0.08,
                                  loss_function="MultiClass", verbose=False,
                                  allow_writing_files=False, thread_count=threads, random_seed=seed)
    raise KeyError(name)


def build_pipeline(group_name: str, model_name: str, seed: int, threads: int):
    steps = [("preprocess", build_preprocessor(group_name))]
    dense_models = {"GradientBoosting", "HistGradientBoosting", "CatBoost"}
    if model_name in dense_models:
        steps.append(("dense", FunctionTransformer(to_dense_float32, validate=False)))
    if group_name == "G_train_selected_structured":
        steps.append(("select", SelectPercentile(score_func=f_classif, percentile=40)))
    steps.append(("model", estimator(model_name, seed, threads)))
    return Pipeline(steps)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def class_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, zero_division=0
    )
    return [
        {"class": int(label), "precision": float(p), "recall": float(r), "f1": float(score), "support": int(n)}
        for label, p, r, score, n in zip(CLASSES, precision, recall, f1, support)
    ]


def write_confusion(path: Path, y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=CLASSES)
    pd.DataFrame(matrix, index=[f"actual_{c}" for c in CLASSES],
                 columns=[f"predicted_{c}" for c in CLASSES]).to_csv(path)


def safe_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def cap_training_indices(indices, y_encoded, cap, seed):
    """Deterministically stratify-cap training rows; never samples validation rows."""
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= cap:
        return indices
    from sklearn.model_selection import train_test_split
    selected, _ = train_test_split(
        indices,
        train_size=cap,
        random_state=seed,
        stratify=y_encoded[indices],
    )
    return np.sort(selected)


def grouped_cv(group_name, model_name, frame, folds, seed, threads, confusion_dir, per_class_rows, model_params=None):
    frame = rows_for_group(frame, group_name).reset_index(drop=True)
    features = FEATURE_GROUPS[group_name]
    X = frame[features].copy()
    y = frame.rating.to_numpy(dtype=int)
    y_encoded = y - 1
    groups = frame.review_text.astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    oof = np.zeros(len(frame), dtype=int)
    fold_rows = []
    train_seconds = 0.0
    predict_seconds = 0.0
    for fold_index, (train_idx, valid_idx) in enumerate(splitter.split(X, y_encoded, groups), start=1):
        fit_idx = train_idx
        if model_name == "KNN":
            fit_idx = cap_training_indices(train_idx, y_encoded, KNN_TRAIN_CAP, seed + fold_index)
        pipeline = build_pipeline(group_name, model_name, seed, threads)
        if model_params:
            pipeline.set_params(**model_params)
        start = time.perf_counter()
        pipeline.fit(X.iloc[fit_idx], y_encoded[fit_idx])
        train_seconds += time.perf_counter() - start
        start = time.perf_counter()
        prediction = pipeline.predict(X.iloc[valid_idx]).astype(int) + 1
        predict_seconds += time.perf_counter() - start
        oof[valid_idx] = prediction
        fold_metrics = metrics(y[valid_idx], prediction)
        fold_metrics.update({"fold": fold_index, "n_train": len(fit_idx), "n_valid": len(valid_idx)})
        fold_rows.append(fold_metrics)
    oof_metrics = metrics(y, oof)
    for row in class_metrics(y, oof):
        per_class_rows.append({"feature_group": group_name, "model": model_name,
                               "protocol": "grouped_text_oof", **row})
    slug = f"{safe_slug(group_name)}__{safe_slug(model_name)}__grouped_oof.csv"
    write_confusion(confusion_dir / slug, y, oof)
    out = {
        "feature_group": group_name,
        "model": model_name,
        "status": "ok",
        "n_rows": len(frame),
        "n_text_groups": int(pd.Series(groups).nunique()),
        "cv_accuracy_mean": float(np.mean([r["accuracy"] for r in fold_rows])),
        "cv_accuracy_std": float(np.std([r["accuracy"] for r in fold_rows], ddof=1)),
        "cv_macro_f1_mean": float(np.mean([r["macro_f1"] for r in fold_rows])),
        "cv_macro_f1_std": float(np.std([r["macro_f1"] for r in fold_rows], ddof=1)),
        "cv_weighted_f1_mean": float(np.mean([r["weighted_f1"] for r in fold_rows])),
        "cv_weighted_f1_std": float(np.std([r["weighted_f1"] for r in fold_rows], ddof=1)),
        "cv_mae_mean": float(np.mean([r["mae"] for r in fold_rows])),
        "cv_mae_std": float(np.std([r["mae"] for r in fold_rows], ddof=1)),
        "cv_rmse_mean": float(np.mean([r["rmse"] for r in fold_rows])),
        "cv_rmse_std": float(np.std([r["rmse"] for r in fold_rows], ddof=1)),
        "grouped_oof_accuracy": oof_metrics["accuracy"],
        "grouped_oof_macro_f1": oof_metrics["macro_f1"],
        "grouped_oof_weighted_f1": oof_metrics["weighted_f1"],
        "grouped_oof_mae": oof_metrics["mae"],
        "grouped_oof_rmse": oof_metrics["rmse"],
        "grouped_oof_spearman": oof_metrics["spearman"],
        "cv_train_seconds_sum": train_seconds,
        "cv_predict_seconds_sum": predict_seconds,
        "fold_results_json": json.dumps(fold_rows),
        "parameters_json": json.dumps(pipeline.get_params(deep=True), default=str),
        "error": "",
        "selected_by_group_cv": False,
        "final_test_evaluated": False,
    }
    return out


def tuning_grid(model_name):
    if model_name in ("RandomForest", "RandomForest_D1_exact"):
        return {"model__max_depth": [8, 10, None], "model__min_samples_leaf": [1, 3]}
    if model_name == "ExtraTrees":
        return {"model__max_depth": [8, 14, None], "model__min_samples_leaf": [1, 3]}
    if model_name == "LogisticRegression":
        return {"model__C": [0.1, 1.0, 10.0]}
    if model_name == "LinearSVC":
        return {"model__C": [0.1, 1.0, 10.0]}
    if model_name == "DecisionTree":
        return {"model__max_depth": [6, 10, 14], "model__min_samples_leaf": [2, 5]}
    if model_name == "KNN":
        return {"model__n_neighbors": [5, 15, 31], "model__weights": ["uniform", "distance"]}
    if model_name == "GradientBoosting":
        return {"model__n_estimators": [50, 90], "model__max_depth": [1, 2, 3]}
    if model_name == "HistGradientBoosting":
        return {"model__max_iter": [70, 120], "model__max_leaf_nodes": [15, 25]}
    if model_name == "OrdinalThresholdLogistic":
        return {"model__C": [0.1, 1.0, 10.0]}
    if model_name == "RidgeOrdinal":
        return {"model__alpha": [0.1, 1.0, 10.0]}
    if model_name == "XGBoost":
        return {"model__n_estimators": [70, 120], "model__max_depth": [3, 5]}
    if model_name == "LightGBM":
        return {"model__n_estimators": [80, 140], "model__num_leaves": [15, 31]}
    if model_name == "CatBoost":
        return {"model__iterations": [80, 140], "model__depth": [4, 6]}
    return {}


def load_saved_d1_metrics():
    path = ROOT / "data" / "aspect_model" / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def exact_d1_reproduction(train, eval_frame, confusion_dir, per_class_rows, threads):
    from sklearn.preprocessing import StandardScaler
    training = train.dropna(subset=["review_text", "rating", *ASPECTS]).copy()
    X_eval = eval_frame[ASPECTS].to_numpy()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(training[ASPECTS].to_numpy())
    X_eval = scaler.transform(X_eval)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=threads)
    start = time.perf_counter(); model.fit(X_train, training.rating.astype(int)); fit_seconds = time.perf_counter()-start
    start = time.perf_counter(); prediction = model.predict(X_eval).astype(int); predict_seconds=time.perf_counter()-start
    y = eval_frame.rating.to_numpy(dtype=int)
    result = metrics(y, prediction)
    saved_predictions_path = ROOT / "data" / "aspect_model" / "predictions.csv"
    saved_match = False
    if saved_predictions_path.exists():
        saved = pd.read_csv(saved_predictions_path)
        saved_match = len(saved) == len(prediction) and np.array_equal(saved.predicted_rating.to_numpy(), prediction) and np.array_equal(saved.actual_rating.to_numpy(), y)
    write_confusion(confusion_dir / "d1_random_forest_exact_saved_eval.csv", y, prediction)
    for row in class_metrics(y, prediction):
        per_class_rows.append({"feature_group": "A_d1_aspects_diagnostic", "model": "D1_RF_exact_reproduction",
                               "protocol": "fixed_id_holdout", **row})
    stored = load_saved_d1_metrics()
    result.update({
        "feature_group": "A_d1_aspects_diagnostic",
        "model": "D1_RF_exact_reproduction",
        "status": "verified" if saved_match else "measured",
        "n_rows": len(training),
        "n_text_groups": int(training.review_text.nunique()),
        "external_accuracy": result["accuracy"],
        "external_macro_f1": result["macro_f1"],
        "external_weighted_f1": result["weighted_f1"],
        "external_mae": result["mae"],
        "external_rmse": result["rmse"],
        "external_spearman": result["spearman"],
        "external_train_seconds": fit_seconds,
        "external_predict_seconds": predict_seconds,
        "saved_d1_prediction_match": saved_match,
        "stored_accuracy": stored.get("accuracy"),
        "stored_macro_f1": stored.get("macro_f1"),
        "stored_weighted_f1": stored.get("weighted_f1"),
        "stored_mae": stored.get("mae"),
        "stored_rmse": stored.get("rmse"),
        "stored_spearman": stored.get("spearman"),
    })
    return result


def smoke_check(train):
    frame = rows_for_group(train, "C_eligible_structured").iloc[:1200].copy()
    X = frame[FEATURE_GROUPS["C_eligible_structured"]]
    y = frame.rating.astype(int).to_numpy() - 1
    model = build_pipeline("C_eligible_structured", "LogisticRegression", 42, 1)
    model.fit(X.iloc[:1000], y[:1000])
    pred = model.predict(X.iloc[1000:])
    print(f"SMOKE OK: {len(pred)} structured predictions; accuracy={accuracy_score(y[1000:], pred):.4f}")


def render_report(path, comparison, features, baseline_result, config):
    ranked = comparison[comparison.status == "ok"].sort_values(
        ["cv_accuracy_mean", "cv_macro_f1_mean"], ascending=False
    )
    best_lines = []
    for _, row in features.sort_values("grouped_cv_accuracy", ascending=False).iterrows():
        best_lines.append(
            f"| {row.feature_group} | {row.best_model} | {row.grouped_cv_accuracy:.4f} ± {row.grouped_cv_accuracy_std:.4f} | "
            f"{row.grouped_oof_macro_f1:.4f} | {row.fixed_id_holdout_accuracy:.4f} | "
            f"{row.fixed_id_holdout_macro_f1:.4f} | {row.fixed_id_holdout_mae:.4f} |"
        )
    top = ranked.head(10)
    lines = [
        "# D1 Accuracy Experiments — Best Model Report", "",
        "## Evaluation and safeguards", "",
        "All model/feature selection and hyperparameter search used 3-fold `GroupKFold` over exact `review_text` groups in the source training CSV. Rows with identical text are never split across a CV train/validation boundary. The existing `actual_reviews.xlsx` workbook was not used for tuning; it was evaluated only after selecting one CV champion per feature group. To bound the pairwise-distance cost, each KNN fold uses a deterministic stratified sample capped at 2,500 training rows; KNN metrics are thus marked as sample-capped and should not be interpreted as using the full fold's training-row count.",
        "",
        "The workbook is an ID-held-out, not unseen-text, test: its 10,003 review IDs do not overlap training IDs, but it contains 110 distinct text strings, 90 of which occur in training. Grouped out-of-fold metrics are the relevant unseen-text estimates. The data contains only 110 distinct texts and every text has conflicting labels, so results have a label-noise ceiling and high group-level uncertainty.",
        "",
        "Aspect-based results are diagnostic only. Five aspect ratings were co-assigned with `rating`; they are not available in a text-only prediction setting. `helpful_votes` was excluded as post-publication information. IDs/names/target and raw `price_local` were excluded. No source data, existing D1 artifacts, or frozen text baseline files were written.",
        "",
        "## D1 exact reproduction", "",
        f"The exact D1 source procedure reproduced {baseline_result.get('external_accuracy', float('nan')):.4%} accuracy on {int(baseline_result.get('n_eval_rows', 10003)):,} rows. Prediction vectors match the saved D1 artifact: **{baseline_result.get('saved_d1_prediction_match', False)}**. Saved reference accuracy is {baseline_result.get('stored_accuracy', float('nan')):.4%}; no output under `data/aspect_model/` was modified.",
        "",
        "## Feature-group champions", "",
        "Grouped CV chooses the champion; fixed-workbook metrics are a separate row-level report and were not used for ranking.",
        "",
        "| Feature group | CV champion | Grouped CV accuracy | OOF macro F1 | Fixed ID-holdout accuracy | Fixed macro F1 | Fixed MAE |",
        "|---|---|---:|---:|---:|---:|---:|",
        *best_lines,
        "",
        "## Top grouped-CV model runs", "",
        "| Feature group | Model | CV accuracy | CV macro F1 | Grouped OOF accuracy | Grouped OOF macro F1 |",
        "|---|---|---:|---:|---:|---:|",
        *[
            f"| {row.feature_group} | {row.model} | {row.cv_accuracy_mean:.4f} ± {row.cv_accuracy_std:.4f} | {row.cv_macro_f1_mean:.4f} ± {row.cv_macro_f1_std:.4f} | {row.grouped_oof_accuracy:.4f} | {row.grouped_oof_macro_f1:.4f} |"
            for _, row in top.iterrows()
        ],
        "",
        "## Limitations and interpretation", "",
        "A score near 90% on five aspect inputs would not be a valid text-only result. Accuracy on the existing workbook is also optimistic for novel text because 90/110 test strings appear in training. Use grouped-CV/OOF figures for unseen-text claims. The dataset is repeated-template data, not 40,000 independent authored reviews: there are 110 unique review texts, 10,001 duplicate full rows, and every exact text appears with multiple ratings.",
        "",
        "For the original smartphone review use case, genuinely independent review texts and labels are needed to establish a reliable 90% generalization target. None of the reported model comparisons changes labels, fabricates rows, applies oversampling, or uses test rows in model selection.",
        "",
        "See `model_comparison.csv`, `feature_comparison.csv`, `cv_per_class_metrics.csv`, and `confusion_matrices/` for result detail. Parameters and data paths are in `config.json`; run from the repository root with `python -m experiments.d1_accuracy.run_experiments`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run one small in-memory pipeline check; write no output")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    train, eval_frame = read_inputs()
    if args.smoke:
        smoke_check(train)
        return

    expected_outputs = [OUT / f for f in config["output_files"]]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("Experiment outputs already exist; refusing to overwrite. Archive them before a new run.")
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    confusion_dir = OUT / "confusion_matrices" / run_id
    models_dir = OUT / "models" / run_id
    confusion_dir.mkdir(parents=True, exist_ok=False)
    models_dir.mkdir(parents=True, exist_ok=False)

    threads = int(config["threads"])
    folds = int(config["group_cv_folds"])
    seed = int(config["seed"])
    per_class_rows = []
    rows = []
    print("Reproducing D1 in memory and comparing to saved predictions...")
    d1_result = exact_d1_reproduction(train, eval_frame, confusion_dir, per_class_rows, threads)
    d1_result["n_eval_rows"] = len(eval_frame)
    print(f"D1 exact accuracy={d1_result['external_accuracy']:.4%}; saved predictions match={d1_result['saved_d1_prediction_match']}")

    failures = []
    for group_name in FEATURE_GROUPS:
        print(f"\n=== Grouped CV: {group_name} ===")
        for model_name in available_models(group_name):
            print(f"  {model_name} ...", flush=True)
            try:
                row = grouped_cv(group_name, model_name, train, folds, seed, threads,
                                 confusion_dir, per_class_rows)
                rows.append(row)
                print(f"    accuracy={row['cv_accuracy_mean']:.4f}±{row['cv_accuracy_std']:.4f}; macroF1={row['cv_macro_f1_mean']:.4f}±{row['cv_macro_f1_std']:.4f}")
            except Exception as exc:
                failures.append({"feature_group": group_name, "model": model_name, "error": f"{type(exc).__name__}: {exc}"})
                rows.append({"feature_group": group_name, "model": model_name, "status": "failed", "error": failures[-1]["error"]})
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)

    comparison = pd.DataFrame(rows)
    if comparison.empty or not (comparison.status == "ok").any():
        raise RuntimeError("No model completed grouped cross-validation")

    winners = {}
    tuned_results = []
    for group_name in FEATURE_GROUPS:
        group_rows = comparison[(comparison.feature_group == group_name) & (comparison.status == "ok")]
        if group_rows.empty:
            continue
        best_row = group_rows.sort_values(["cv_accuracy_mean", "cv_macro_f1_mean"], ascending=False).iloc[0]
        best_model = str(best_row.model)
        train_frame = rows_for_group(train, group_name).reset_index(drop=True)
        X = train_frame[FEATURE_GROUPS[group_name]]
        y = train_frame.rating.to_numpy(dtype=int) - 1
        groups = train_frame.review_text.astype(str).to_numpy()
        grid = tuning_grid(best_model)
        selected_model = best_model
        selected_params = {}
        selected_accuracy = float(best_row.cv_accuracy_mean)
        selected_model_name = best_model
        if grid:
            print(f"\n=== Grouped-CV tuning: {group_name}/{best_model} ===", flush=True)
            cv_splits = []
            for fold_index, (train_indices, valid_indices) in enumerate(
                GroupKFold(n_splits=folds).split(X, y, groups), start=1
            ):
                if best_model == "KNN":
                    train_indices = cap_training_indices(
                        train_indices, y, KNN_TRAIN_CAP, seed + 100 + fold_index
                    )
                cv_splits.append((train_indices, valid_indices))
            search = GridSearchCV(
                build_pipeline(group_name, best_model, seed, threads),
                param_grid=grid,
                scoring="accuracy",
                cv=cv_splits,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            start = time.perf_counter()
            search.fit(X, y, groups=groups)
            tune_seconds = time.perf_counter() - start
            selected_params = search.best_params_
            tuned_accuracy = float(search.best_score_)
            tuned_results.append({
                "feature_group": group_name,
                "model": f"{best_model}_tuned",
                "status": "ok",
                "cv_accuracy_mean": tuned_accuracy,
                "cv_accuracy_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                "cv_train_seconds_sum": tune_seconds,
                "cv_predict_seconds_sum": np.nan,
                "fold_results_json": "",
                "parameters_json": json.dumps(selected_params, default=str),
                "error": "",
                "selected_by_group_cv": tuned_accuracy >= float(best_row.cv_accuracy_mean),
                "final_test_evaluated": False,
            })
            if tuned_accuracy >= float(best_row.cv_accuracy_mean):
                selected_model_name = f"{best_model}_tuned"
                selected_accuracy = tuned_accuracy
                comparison.loc[comparison.feature_group == group_name, "selected_by_group_cv"] = False
                tuned_oof = grouped_cv(
                    group_name, best_model, train, folds, seed, threads, confusion_dir,
                    per_class_rows, model_params=selected_params,
                )
                for key in [
                    "cv_macro_f1_mean", "cv_macro_f1_std", "cv_weighted_f1_mean",
                    "cv_weighted_f1_std", "cv_mae_mean", "cv_mae_std", "cv_rmse_mean",
                    "cv_rmse_std", "grouped_oof_accuracy", "grouped_oof_macro_f1",
                    "grouped_oof_weighted_f1", "grouped_oof_mae", "grouped_oof_rmse",
                    "grouped_oof_spearman", "fold_results_json",
                ]:
                    tuned_results[-1][key] = tuned_oof[key]
                tuned_results[-1]["model"] = selected_model_name
                tuned_results[-1]["cv_train_seconds_sum"] += tuned_oof["cv_train_seconds_sum"]
                tuned_results[-1]["cv_predict_seconds_sum"] = tuned_oof["cv_predict_seconds_sum"]
            else:
                selected_params = {}
                comparison.loc[(comparison.feature_group == group_name) & (comparison.model == best_model), "selected_by_group_cv"] = True
        else:
            comparison.loc[(comparison.feature_group == group_name) & (comparison.model == best_model), "selected_by_group_cv"] = True
        winners[group_name] = (selected_model_name, selected_params, selected_accuracy)

    if tuned_results:
        comparison = pd.concat([comparison, pd.DataFrame(tuned_results)], ignore_index=True, sort=False)

    feature_rows = []
    final_test_metrics = {}
    for group_name, (selected_model_name, params, cv_accuracy) in winners.items():
        model_name = selected_model_name.removesuffix("_tuned")
        training = rows_for_group(train, group_name).reset_index(drop=True)
        X_train = training[FEATURE_GROUPS[group_name]]
        y_train = training.rating.to_numpy(dtype=int) - 1
        X_test = eval_frame[FEATURE_GROUPS[group_name]].copy()
        y_test = eval_frame.rating.to_numpy(dtype=int)
        final_model = build_pipeline(group_name, model_name, seed, threads)
        if params:
            final_model.set_params(**params)
        fit_rows = np.arange(len(training))
        if model_name == "KNN":
            fit_rows = cap_training_indices(fit_rows, y_train, KNN_TRAIN_CAP, seed + 1000)
        start = time.perf_counter(); final_model.fit(X_train.iloc[fit_rows], y_train[fit_rows]); fit_seconds=time.perf_counter()-start
        start = time.perf_counter(); test_prediction=final_model.predict(X_test).astype(int) + 1; predict_seconds=time.perf_counter()-start
        test_metrics = metrics(y_test, test_prediction)
        test_class = class_metrics(y_test, test_prediction)
        final_test_metrics[group_name] = test_metrics
        comparison.loc[(comparison.feature_group == group_name) & (comparison.model == selected_model_name), "final_test_evaluated"] = True
        test_slug = f"{safe_slug(group_name)}__fixed_id_holdout.csv"
        write_confusion(confusion_dir / test_slug, y_test, test_prediction)
        for item in test_class:
            per_class_rows.append({"feature_group": group_name, "model": model_name,
                                   "protocol": "fixed_id_holdout_champion", **item})
        model_path = models_dir / f"{safe_slug(group_name)}__champion.joblib"
        joblib.dump(final_model, model_path)
        feature_rows.append({
            "feature_group": group_name,
            "features_json": json.dumps(FEATURE_GROUPS[group_name]),
            "best_model": selected_model_name,
            "best_parameters_json": json.dumps(params, default=str),
            "grouped_cv_accuracy": cv_accuracy,
            "grouped_cv_accuracy_std": float(comparison.loc[(comparison.feature_group == group_name) & (comparison.model == selected_model_name), "cv_accuracy_std"].iloc[0]) if not math.isnan(float(comparison.loc[(comparison.feature_group == group_name) & (comparison.model == selected_model_name), "cv_accuracy_std"].iloc[0])) else np.nan,
            "grouped_oof_accuracy": float(comparison.loc[(comparison.feature_group == group_name) & (comparison.model == selected_model_name), "grouped_oof_accuracy"].iloc[0]),
            "grouped_oof_macro_f1": float(comparison.loc[(comparison.feature_group == group_name) & (comparison.model == selected_model_name), "grouped_oof_macro_f1"].iloc[0]),
            "fixed_id_holdout_accuracy": test_metrics["accuracy"],
            "fixed_id_holdout_macro_f1": test_metrics["macro_f1"],
            "fixed_id_holdout_weighted_f1": test_metrics["weighted_f1"],
            "fixed_id_holdout_mae": test_metrics["mae"],
            "fixed_id_holdout_rmse": test_metrics["rmse"],
            "fixed_id_holdout_spearman": test_metrics["spearman"],
            "final_train_rows": len(training),
            "effective_fit_rows": len(fit_rows),
            "fixed_test_rows": len(eval_frame),
            "final_fit_seconds": fit_seconds,
            "final_predict_seconds": predict_seconds,
            "model_path": str(model_path.relative_to(OUT)),
            "leakage_status": "DIAGNOSTIC_ONLY" if "aspects" in group_name else "NO_ASPECTS; inspect feature timing/provenance",
        })
        print(f"\nFINAL {group_name}/{model_name}: grouped-CV acc={cv_accuracy:.4f}; fixed-ID-test acc={test_metrics['accuracy']:.4f}; macroF1={test_metrics['macro_f1']:.4f}")

    comparison = pd.concat([comparison, pd.DataFrame([{
        "feature_group": d1_result["feature_group"], "model": d1_result["model"], "status": d1_result["status"],
        "n_rows": d1_result["n_rows"], "n_text_groups": np.nan,
        "cv_accuracy_mean": np.nan, "cv_accuracy_std": np.nan, "cv_macro_f1_mean": np.nan,
        "cv_macro_f1_std": np.nan, "cv_weighted_f1_mean": np.nan, "cv_weighted_f1_std": np.nan,
        "cv_mae_mean": np.nan, "cv_mae_std": np.nan, "cv_rmse_mean": np.nan, "cv_rmse_std": np.nan,
        "grouped_oof_accuracy": np.nan, "grouped_oof_macro_f1": np.nan, "grouped_oof_weighted_f1": np.nan,
        "grouped_oof_mae": np.nan, "grouped_oof_rmse": np.nan, "grouped_oof_spearman": np.nan,
        "cv_train_seconds_sum": d1_result["external_train_seconds"],
        "cv_predict_seconds_sum": d1_result["external_predict_seconds"],
        "fold_results_json": "", "parameters_json": json.dumps({"n_estimators":100,"max_depth":10,"random_state":42}),
        "error": "", "selected_by_group_cv": False, "final_test_evaluated": True,
        "external_accuracy": d1_result["external_accuracy"], "external_macro_f1": d1_result["external_macro_f1"],
        "external_weighted_f1": d1_result["external_weighted_f1"], "external_mae": d1_result["external_mae"],
        "external_rmse": d1_result["external_rmse"], "external_spearman": d1_result["external_spearman"],
        "saved_d1_prediction_match": d1_result["saved_d1_prediction_match"],
    }])], ignore_index=True, sort=False)

    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    feature_table = pd.DataFrame(feature_rows)
    feature_table.to_csv(OUT / "feature_comparison.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT / "cv_per_class_metrics.csv", index=False)
    metadata = {
        "config": config,
        "run_id": run_id,
        "run_confusion_dir": str(confusion_dir.relative_to(OUT)),
        "run_models_dir": str(models_dir.relative_to(OUT)),
        "source_shape": [50000, 25],
        "labeled_rows": 39998,
        "d1_train_rows": 39997,
        "fixed_id_holdout_rows": len(eval_frame),
        "fixed_id_holdout_unique_texts": int(eval_frame.review_text.nunique()),
        "unique_review_text_overlap_train_test": int(len(set(train.review_text.dropna().astype(str)) & set(eval_frame.review_text.dropna().astype(str)))),
        "d1_saved_predictions_exact_match": d1_result["saved_d1_prediction_match"],
        "d1_reproduction": {key: d1_result[key] for key in ["external_accuracy", "external_macro_f1", "external_weighted_f1", "external_mae", "external_rmse", "external_spearman", "saved_d1_prediction_match"]},
        "failures": failures,
        "warnings": [
            "Fixed workbook evaluates unseen review IDs but overlaps 90 of 110 unique review-text strings with training.",
            "Grouped OOF is the primary unseen-text validation estimate; only 110 unique strings exist and all have conflicting labels.",
            "Aspect-rating feature groups are simultaneous sub-target diagnostic results, not valid text-only deployment results.",
            "Test workbook was evaluated only after grouped-CV model selection; no hyperparameter choices used its labels.",
            "Sentiment is a derived/annotated predictor and its provenance must be confirmed for deployment.",
        ],
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    render_report(OUT / "best_model_report.md", comparison, feature_table, d1_result, config)
    print(f"\nExperiment run complete. Results: {OUT}")
    if failures:
        print(f"Model failures: {len(failures)}; see run_metadata.json and model_comparison.csv")


if __name__ == "__main__":
    main()
