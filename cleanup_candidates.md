# Project Cleanup Audit Report: Candidate Categorization

This document categorizes all project files prior to cleanup to ensure that no necessary code, dataset, memory file, evaluation pipeline, or correlation results are modified or deleted.

---

## 1. KEEP

These files are **strictly required** for the operational pipeline, interactive application, sentiment prediction, rating prediction, evaluation suite, and correlation analysis reports.

### Core Application & Pipeline
- `agent.py` — Core `SemanticRatingAgent` implementation with semantic centroid classifier, reasoning engine, and uncertainty modeling.
- `batch_predict.py` — Batch prediction pipeline CLI for unseen reviews dataset.
- `evaluate_predictions.py` — Full evaluation script calculating sentiment accuracy (98.74%), exact Likert accuracy, and error breakdown.
- `correlation_results.py` — Terminal reporting script reading pre-calculated metrics and outputting formatted summary.
- `main.py` — Interactive CLI entry point for conversational querying and unseen review rating.
- `run_analysis.py` — Script to execute full dataset understanding analysis and generate memory.
- `verify.py` — Verification suite testing single unseen review predictions and dataset Q&A.
- `requirements.txt` — Project Python package dependencies.
- `memory.json` — Pre-trained knowledge store containing sentiment centroids, rating centroids, and aspect profiles.
- `.gitignore` — Git configuration.
- `.git/` — Git version control directory.

### Core Datasets & Results (`data/`)
- `data/Mobile Reviews Sentiment.csv` — Full training/ground dataset.
- `data/actual_reviews.xlsx` — Ground truth evaluation dataset with 10,003 records.
- `data/reviews_only.csv` — Unseen review inputs for `batch_predict.py`.
- `data/predictions.csv` — Current predictions output from `batch_predict.py`.
- `data/evaluation_results.csv` — Matched prediction vs ground-truth records used by `correlation_results.py`.
- `data/sentiment_errors.csv` — Error log for the 126 sentiment misclassifications.
- `data/correlation_metrics_summary.csv` — Master table of all correlation statistics and regression metrics.
- `data/sentiment_summary_stats_actual.csv` — Descriptive statistics for actual sentiment classes.
- `data/sentiment_summary_stats_predicted.csv` — Descriptive statistics for predicted sentiment classes.
- `data/sentiment_rating_crosstab_actual.csv` — Crosstab matrix of actual sentiment vs Likert ratings.
- `data/sentiment_rating_crosstab_predicted.csv` — Crosstab matrix of predicted sentiment vs Likert ratings.
- `data/plot1_sentiment_vs_avg_rating.png` — Visual artifact for sentiment vs mean rating.
- `data/plot2_sentiment_rating_distribution.png` — Visual artifact for rating distribution per sentiment.
- `data/plot3_sentiment_rating_relationship.png` — Visual artifact for boxplot/scatter of sentiment vs rating.
- `data/plot4_semantic_score_vs_rating.png` — Visual artifact for text semantic score vs actual rating.

### Key Analysis Scripts & Final Documentation
- `scratch/compute_correlation_analysis.py` — Script that generated correlation metrics and plots from scratch.
- `sentiment_rating_correlation_analysis.md` — Final comprehensive markdown report on all correlation findings.

---

## 2. SAFE TO DELETE

These files are temporary text output dumps, intermediate experiment CSV outputs, old duplicate prediction runs, or auto-generated cache directories that are **no longer referenced or required**.

### Temporary Terminal Output & Log Files
- `diag_out.txt` — Temporary log from past diagnosis execution.
- `qa_test.txt` — Temporary text output from past Q&A verification.
- `run_log.txt` — Temporary log from past pipeline run.
- `temp_out.txt` — Temporary command output log.
- `verify_out.txt` — Temporary output log from verify.py.
- `scratch/model_comparison_out.txt` — Temporary output log from model comparison run.
- `scratch/validation_audit_out.txt` — Temporary output log from validation audit.
- `scratch/sample_reviews.csv` — Temporary 5-line scratch review file.

### Intermediate & Superseded Prediction/Evaluation CSVs
- `data/predictions_new.csv` — Superseded intermediate prediction file from earlier experiment iterations.
- `data/predictions_phase1.csv` — Superseded prediction file from Phase 1 experiment.
- `data/predictions_phase2.csv` — Superseded prediction file from Phase 2 experiment.
- `data/evaluation_results_new.csv` — Superseded intermediate evaluation file.
- `data/evaluation_results_phase1.csv` — Superseded evaluation file from Phase 1 experiment.
- `data/evaluation_results_phase2.csv` — Superseded evaluation file from Phase 2 experiment.

### Cache & Logging Directories
- `catboost_info/` — Auto-generated logging directory from previous CatBoost experimentation.

---

## 3. REVIEW MANUALLY

These files contain historical experiment code, alternative benchmark model comparisons, and in-depth investigation research notes. **They will NOT be deleted automatically during cleanup**, but are retained for research archive purposes.

### Benchmark & Alternative Model Scripts (Root)
- `diagnose.py` — Text quality and label ambiguity diagnostic tool.
- `embedding_comparison_experiment.py` — Benchmark comparing different sentence transformer backbones.
- `model_comparison_experiment.py` — Benchmark comparing Ridge, Random Forest, XGBoost, CatBoost.
- `optimized_model_comparison_experiment.py` — Feature-engineered model comparison experiment.
- `safe_feature_model_comparison_experiment.py` — Leakage-free feature experiment.
- `xgboost_batch_experiment.py` — XGBoost baseline experiment script.

### Benchmark Data Outputs (`data/`)
- `data/embedding_comparison_results.csv` — Metric results for embedding comparison.
- `data/model_comparison_results.csv` — Metric results for ML model comparison.
- `data/optimized_model_comparison_results.csv` — Metric results for optimized ML model comparison.
- `data/xgboost_predictions.csv` — Predictions generated by XGBoost baseline.
- `data/xgboost_evaluation_results.csv` — Evaluation results for XGBoost baseline.

### Research & Investigation Markdown Reports (Root)
- `persona_rating5_investigation.md` — Investigation of persona/author effects on Rating 5.
- `rating5_training_signal_investigation.md` — Investigation on training signal for extreme ratings.
- `rating_calibration_investigation.md` — Investigation on temperature calibration and thresholds.
- `rating_intensity_investigation.md` — Investigation on linguistic intensity modifiers.
- `rating_uncertainty_implementation.md` — Documentation on calibrated uncertainty implementation.
- `sentiment_rating_investigation.md` — Initial investigation on sentiment accuracy behavior.
- `target_leakage_audit_report.md` — Formal audit report documenting zero data leakage.
- `text_to_aspect_rating_investigation.md` — Investigation on aspect-level rating decomposition.

### Scratch Investigation Scripts (`scratch/`)
- `scratch/audit_prediction_files.py`
- `scratch/audit_regression_investigation.py`
- `scratch/bayes_optimal_limit.py`
- `scratch/check_feature_correlations.py`
- `scratch/check_text_rating_variance.py`
- `scratch/compare_all_candidate_models.py`
- `scratch/disk_report.py`
- `scratch/evaluate_final_restored.py`
- `scratch/evaluate_phase2_full.py`
- `scratch/evaluate_rating5_from_baseline.py`
- `scratch/evaluate_stochastic_on_baseline.py`
- `scratch/inspect_example_bank.py`
- `scratch/inspect_memory.py`
- `scratch/inspect_raw_rating_distributions.py`
- `scratch/investigate_persona_signals.py`
- `scratch/investigate_rating5_augmentation.py`
- `scratch/investigate_rating5_training_signal.py`
- `scratch/investigate_rating_5_linguistic_signals.py`
- `scratch/investigate_rating_calibration.py`
- `scratch/investigate_rating_intensity.py`
- `scratch/investigate_sentiment_rating_signal.py`
- `scratch/investigate_sentiment_strength_rating.py`
- `scratch/investigate_text_to_aspect_rating.py`
- `scratch/leakage_audit_calc.py`
- `scratch/model_validation_audit.py`
- `scratch/rating_investigation_phase2.py`
- `scratch/sentiment_rule_analysis.py`
- `scratch/simulate_new_calibration.py`
- `scratch/simulate_rating5_candidates.py`
- `scratch/simulate_rating_calibration.py`
- `scratch/simulate_stochastic_sampling.py`
- `scratch/test_rating_strategies.py`
- `scratch/test_rating_uncertainty_layer.py`
- `scratch/test_retrieval.py`
- `scratch/test_routing.py`
- `scratch/validate_new_results.py`
- `scratch/verify_memory.py`
- `scratch/verify_uncertainty_implementation.py`
