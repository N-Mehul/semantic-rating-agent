# _archive: canonical benchmark experiment code and research notes

Nothing here is imported by the active Digital Persona application, the SSR pipeline, the dashboard, or the tests.

The retained experiment scripts correspond to the benchmark comparisons documented in the repository README (Sections 16 and 23):
- `embedding_comparison_experiment.py`: benchmarks MiniLM, MPNet, and BGE on review text.
- `model_comparison_experiment.py`: classical ML vs. neural baseline benchmark.
- `optimized_model_comparison_experiment.py`: tuned model comparison (Ridge, Logistic Regression, XGBoost, etc.).
- `safe_feature_model_comparison_experiment.py`: leak-free evaluation ensuring text-only representation.
- `xgboost_batch_experiment.py`: standalone XGBoost baseline benchmark.
- `correlation_results.py`: computes cross-variable correlation metrics.

Also retained are the 8 research investigation notes documenting the review-rating empirical findings (README Section 18).
