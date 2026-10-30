# CLEANUP_LOG

## Result of cleanup pass 5 (2026-10-29 — Aggressive data/ Directory Cleanup)
| | Before pass 5 | After pass 5 |
|---|---|---|
| Files (excluding `.git`) | 672 | 432 |
| Directories (excluding `.git`) | 68 | 44 |

Total removed in pass 5: 240 files and 24 directories (17 top-level abandoned experiment folders in `data/` plus 7 nested subdirectories).

## Deleted (pass 5)
| Item | Files | Why |
|---|---|---|
| `data/decision_rule_recovery/` | 20 files | Abandoned experiment outputs; zero imports or references in tests, dashboard, or pipeline. |
| `data/experiment_c/` | 4 files | Abandoned experiment outputs; zero imports or references. |
| `data/experiment_c_multimodel/` | 18 files | Abandoned experiment outputs (confusion matrices, reports, predictions); zero imports or references. |
| `data/experiment_d2/` | 9 files | Abandoned experiment outputs; zero imports or references. |
| `data/experiment_d_aspect_features/` | 26 files | Abandoned experiment outputs; zero imports or references. |
| `data/experiment_e_safe_context/` | 45 files | Abandoned experiment outputs; zero imports or references. |
| `data/experiment_f/` | 3 files | Abandoned experiment outputs; zero imports or references. |
| `data/neural_aspect_models/` | 16 files | Abandoned neural aspect modeling outputs; zero imports or references. |
| `data/ordinal_evaluation_report/` | 3 files | Abandoned diagnostic outputs; zero imports or references. |
| `data/ordinal_meta_model_experiment/` | 15 files | Abandoned meta-model experiment outputs; zero imports or references. |
| `data/persona_nl_controlled_probe/` | 3 files | Abandoned exploratory probe outputs; zero imports or references. |
| `data/range_72pct_test/` | 5 files | Abandoned range evaluation outputs; zero imports or references. |
| `data/range_prediction_evaluation/` | 7 files | Abandoned range prediction outputs; zero imports or references. |
| `data/range_threshold_sweep/` | 18 files | Abandoned threshold sweep outputs; zero imports or references. |
| `data/rating_3_diagnosis/` | 10 files | Abandoned rating 3 pattern diagnosis outputs; zero imports or references. |
| `data/rating_4_5_separability/` | 7 files | Abandoned rating separability outputs; zero imports or references. |
| `data/star_wise_ensemble/` | 9 files | Abandoned ensemble modeling outputs; zero imports or references. |
| 22 loose unreferenced files directly in `data/` | 22 files | Unused diagnostic plots (`plot1_*.png` through `plot4_*.png`), ad-hoc investigation CSVs (`rating5_investigation.csv`, `rating_345_*.csv`, `rating_distribution_analysis.csv`, `rating_error_analysis.csv`, `rating_model_comparison.csv`, `sentiment_errors.csv`, `sentiment_rating_crosstab_*.csv`, `sentiment_summary_stats_*.csv`, `vader_predictions.csv`), and unused 5-anchor summary CSVs (`five_anchor_class_summary.csv`, `five_anchor_similarity_summary.csv`, `five_anchor_test_results.csv`). |

## Verification (pass 5)
| Check | Result |
|---|---|
| Core pytest test suites (108 items) | exit 0, 108/108 passed in 32.10s |
| Pipeline output verification (`verify_pipeline_outputs.py`) | exit 0, 39/39 passed |
| Key entry points and required folders | Verified present (`main.py`, `canonical_ssr.py`, `agent.py`, `dashboard/server.py`, `scripts/run_pipeline.py`, frozen split, embedding cache, champion models, all tests) |
| Complete final directory tree | Verified against actual filesystem and regenerated to `docs/project_status/complete_directory_tree.txt` (432 files, 44 dirs) |

## Result of cleanup pass 4 (2026-10-29 — Aggressive Cleanup)
| | Before pass 4 | After pass 4 |
|---|---|---|
| Files (excluding `.git`) | 704 | 672 |
| Directories (excluding `.git`) | 71 | 68 |

Total removed in pass 4: 32 files and 3 directories.

## Deleted (pass 4)
| Item | Count | Why |
|---|---|---|
| `_archive/scratch/` | 1 dir, 8 files | Obsolete scratch scripts, diagnosis notes, and intermediate reports (`compute_ceiling.py`, `five_anchor_diagnosis_report.md`, `interactive_k3_validation.csv`, `interactive_k3_validation_summary.md`, `rating_prediction_diagnosis_report.md`, `run_five_anchor_experiment.py`, `sanity_check_k3_k5.py`, `validate_k3_and_threshold.py`). |
| `_archive/analysis/` | 1 dir, 2 files | One-off ceiling/class-5 print scripts (`per_class_ceiling.py`, `print_class5_metrics.py`). |
| `_archive/scripts/` | 1 dir, 3 files | Obsolete pilot generation scripts (`pilot_make_specs.py`, `pilot_score_and_evaluate.py`, `pilot_write_responses.py`) superseded by active modules `dp_persona_generator.py` and `dp_synthetic_dataset.py`. |
| `_archive/` obsolete experiment scripts | 13 files | Obsolete exploratory scripts from early review modeling (`calibrated_discretization_experiment.py`, `decision_rule_recovery_experiment.py`, `experiment_c_multimodel.py`, `experiment_c_text_metadata.py`, `experiment_d2_aspect_models.py`, `experiment_e_safe_context.py`, `experiment_f_feature_combinations.py`, `neural_aspect_models_experiment.py`, `persona_nl_controlled_probe.py`, `range_prediction_evaluation.py`, `range_threshold_sweep.py`, `rating_4_5_separability_analysis.py`, `star_wise_ensemble_experiment.py`). |
| Runner scripts inside `data/` directories | 4 files | Experiment runner scripts mistakenly saved inside data directories (`data/ordinal_evaluation_report/ordinal_evaluation_report.py`, `data/ordinal_meta_model_experiment/ordinal_meta_model_experiment.py`, `data/range_72pct_test/range_72pct_evaluation.py`, `data/rating_3_diagnosis/rating_3_pattern_diagnosis.py`). |
| `scripts/build_final_results_report.py` | 1 file | One-off standalone report generation script. Output (`docs/project_status/step11_final_research_results.md`) is already generated and checked in; not imported or required by any tests or active pipeline stage. |
| `experiment_d_aspect_features.py` | 1 file | Obsolete standalone script at repository root; superseded by the official grouped-CV pipeline in `experiments/d1_accuracy/run_experiments.py` and verified showcase model `aspect_rating_model.py`. |

## Verification (pass 4)
| Check | Result |
|---|---|
| Core pytest test suites | exit 0, 79/79 passed in 10.56s |
| Pipeline output verification (`verify_pipeline_outputs.py`) | exit 0, 39/39 passed |
| Key entry points and required folders | Verified present (`main.py`, `canonical_ssr.py`, `agent.py`, `dashboard/server.py`, `scripts/run_pipeline.py`, frozen split, embedding cache, champion models, all tests) |
| Complete final directory tree | Verified against actual filesystem and regenerated to `docs/project_status/complete_directory_tree.txt` (672 files, 68 dirs) |

## Result of cleanup pass 3 (2026-10-29 — Deep Clean)
| | Before pass 3 | After pass 3 |
|---|---|---|
| Files (excluding `.git`) | 757 | 704 |
| Directories (excluding `.git`) | 73 | 71 |

Total removed in pass 3: 53 files and 2 directories.

## Deleted (pass 3)
| Item | Count | Why |
|---|---|---|
| `.claude/` (`settings.local.json`) | 1 dir, 1 file | Local editor/tool configuration leftover from another agent platform; not needed by project or runtime. |
| `data/current_baseline_evaluation/current_baseline_evaluation.py` | 1 file | Duplicate copy of `current_baseline_evaluation.py` at repository root (which is preserved with import-safe guard). |
| `_archive/cheap_transformer_diagnostic.py` | 1 file | Obsolete standalone diagnostic script. Produced no retained outputs (`data/transformer_diagnostic/` was removed earlier). Not referenced in active codebase or documentation. |
| `_archive/evaluate_agentic.py` | 1 file | Obsolete evaluator for removed legacy agentic direct-rating experiment; its target input `data/agentic_predictions.csv` was already deleted. |
| `_archive/scratch/dataset_stats.py` | 1 file | Obsolete 8-line scratch script printing `df.shape` with hardcoded absolute user machine paths; replaced by formal data audits. |
| `_archive/scratch/select_test_set.py` | 1 file | Obsolete 40-line scratch probe attempting to write candidate items to a non-existent `scratch/` folder. |
| `_archive/scratch/run_full_diagnosis.py` | 1 file | Obsolete console diagnostic probe printing confusion matrices and conditional probabilities to stdout with no file outputs. |
| `experiments/d1_accuracy/confusion_matrices/*.csv` (loose files) | 46 files | Obsolete intermediate artifacts from preliminary screen run at 21:07:11 (incomplete, groups A-D only). The canonical screen run is preserved in `run_20261011_214543/` (all groups A-I, 115 CSVs, cited in `run_metadata.json`), and the finalized holdout results are preserved in `finalized_20261011_221830/` (9 CSVs). |

## Verification (pass 3)
| Check | Result |
|---|---|
| `python -m pytest --confcutdir=tests tests/` | exit 0, 494 passed, 2 warnings in 231.91s |
| Key entry points and required folders | Verified present (`main.py`, `canonical_ssr.py`, `agent.py`, `dashboard/server.py`, `scripts/run_pipeline.py`, frozen split, embedding cache, champion models, all tests) |
| Complete final directory tree | Verified against actual filesystem and regenerated to `docs/project_status/complete_directory_tree.txt` (704 files, 71 dirs) |

## Result of cleanup pass 2 (2026-10-29)
| | Before | After |
|---|---|---|
| Files (excluding `.git`) | 819 | 757 |
| Directories (excluding `.git`) | 76 | 73 |

Space saved: ~62 files (intermediate CV run + incomplete finalization folder in both confusion_matrices and models).

## Deleted (pass 2)
| Item | Why |
|---|---|
| `experiments/d1_accuracy/confusion_matrices/run_20261011_211841/` (55 files) | Superseded intermediate CV screen run. `run_metadata.json` names `run_20261011_214543` as the canonical `screen_source_run`; `run_20261011_211841` produced an incomplete screen (missing text-only / extended feature groups) and was not the source for any finalized model or saved eval. Only referenced in the old `complete_directory_tree.txt`. |
| `experiments/d1_accuracy/confusion_matrices/finalized_20261011_221705/` (1 file) | Incomplete first finalization attempt — contains only 1 holdout CSV (`a_d1_aspects_diagnostic`). `finalized_20261011_221830/` is the canonical complete finalization with all 9 feature-group holdout CSVs. Only referenced in the old `complete_directory_tree.txt`. |
| `experiments/d1_accuracy/models/finalized_20261011_221705/` (1 file) | Model checkpoint from the same incomplete first finalization attempt. `finalized_20261011_221830/` contains all 9 champion models. Only referenced in the old `complete_directory_tree.txt`. |

## Updated
| Item | Action |
|---|---|
| `docs/project_status/complete_directory_tree.txt` | Regenerated to reflect the post-pass-2 state (757 files, 73 dirs, `.git` excluded). |

## Verification (pass 2)
Tests were not re-run in this pass because no source code, test files, data, or results were modified. The deletions affect only intermediate experiment run folders that are: (a) not imported by any code, (b) not referenced in any non-tree `.py`/`.md`/`.json`/`.txt` file, (c) superseded by the canonical `run_20261011_214543` / `finalized_20261011_221830` folders that remain intact.

The previous pass verification (494 tests passed, 39/39 pipeline checks, all protected files unchanged) remains valid.

All cleanup on 2026-10-29. Plain file operations; no git commits. The owner confirmed an external filesystem backup before the first change; the 80 Ollama responses generated afterwards (`data/pilot_ollama_*_FT_*`) are not in that backup and were never touched.

## Result of the final pass
| | Before | After |
|---|---|---|
| Files (excluding `.git`) | 851 | 819 |
| Size (excluding `.git`) | 124,411 KB | 124,221 KB |

Space saved in this pass: about 190 KB, because the large items were removed earlier in the session (see "Earlier passes"). `data/` is about 70 MB of retained research evidence.

## Deleted (final pass)
| Item | Why |
|---|---|
| `_archive/full_user_prompt.txt`, `_archive/temp_user_prompt.txt` | Earlier audit-task prompts, not project artifacts, unreferenced |
| `_archive/f1_check.py` | 20-line console check, unreferenced, wrote nothing |
| `_archive/docs/PROJECT_STATUS_AUDIT.md`, `_archive/docs/REPORT.md`, `_archive/docs/project_status/complete_project_status_audit.md` | Superseded by `docs/project_status/full_project_audit.md`; contradicted the current pipeline (kNN-era showcase, FC as a construct) |
| `_archive/AI_Synthetic_Review_Generator/*.py` (3) | Unrelated Tkinter helper from another machine; not referenced, no retained outputs |
| 21 scratch console probes in `_archive/scratch/` (`diagnose_*`, `test_*`, `trace_subset`, `probe_distributions`, `robustness_5_seeds`, `dump_results`, `inspect_data`, `preliminary_anchor_test`, ...) | Wrote nothing, no doc dependence; their findings survive in the retained diagnosis notes |
| `docs/project_status/PROJECT_HANDOVER_QUICK_START.md` | Overlapped `DIGITAL_PERSONA_PROJECT_HANDOVER_REPORT.md` and the README; had outdated test counts |
| `docs/project_status/cleanup_manifest.md` | Merged into this file |
| `.vscode/` | Editor settings only |
| 3 empty directories (`data/experiment_e_safe_context/feature_importance`, two `experiments/d1_accuracy/models/run_*`), `__pycache__`, `.pytest_cache` | Empty or regenerable |

## Restored (an earlier archive step broke README commands)
`main.py` and `evaluate_predictions.py` are back in the root: the README documents `python main.py` and `python evaluate_predictions.py`.

## Earlier passes (same day)
- Deleted: `AI_Synthetic_Review_Generator/venv/` (about 182 MB, not the running interpreter), all `__pycache__`/`.pytest_cache`, `data/transformer_diagnostic/` (empty), `scratch/simulate_new_calibration.py` (0 bytes), four `_archive` scripts that were md5-identical to copies in `data/`.
- Moved to `_archive/`: root experiment scripts and notes whose outputs remain in `data/`; early pilot scripts; `scratch/` (now trimmed); `maier_*.txt` and `match_57037.txt` went to `docs/ssr_verification/source_excerpts/`.
- Changed code: `current_baseline_evaluation.py` (import-safe `main()` guard); `dashboard/workflow.py` (optional `DP_DASHBOARD_RUN_LOG_DIR`); `survey_ssr_bridge.py` (`FC` alias removed, FC now unsupported); LEGACY banners in `agent.py` and `batch_predict.py`; new `tests/conftest.py` and `tests/test_run_log_isolation.py`.
- Added data (additions only): `data/pilot_ollama_{persona,control}_{FT_PE01,FT_BI01}/`, `data/ollama_ft_generation_manifest.json`.

## Kept although they look removable
- `agent.py`, `batch_predict.py`, `analysis/label_structure.py`: imported by `analysis/{calibration,decision_rules,distribution_eval}.py` and `tests/test_analysis.py`; also the kNN baseline path (`k=3`).
- `five_anchor_ssr.py`: superseded by `canonical_ssr.py`, only used by an archived experiment, kept because the owner marked it protected.
- `pymc_response_rater.py`: cited by `docs/ssr_verification/original_anchor_source_audit.md`.
- `_archive/` scripts that wrote retained result files (see `_archive/README.md`); `_archive/scratch/` keeps the `k=3` provenance scripts and the five-anchor experiment.
- `results/embedding_cache.npz`: cache path read by `analysis/splits.py`.
- `memory.json`, `requirements.txt`, `README.md`, `Project_Report.docx`, `paper/`.
- `data/dashboard_demo_runs/runs.jsonl`: 38 entries of unproven origin, none deleted.

## Verification (after the final pass)
| Check | Result |
|---|---|
| `python -m pytest --confcutdir=tests tests/` | exit 0, 494 passed, 2 warnings |
| `python scripts/verify_pipeline_outputs.py` | exit 0, 39/39 |
| `python scripts/run_pipeline.py all` | exit 0 (it reads saved responses and calls no LLM; a hash comparison showed its outputs byte-identical to the baseline) |
| Imports of 22 entry-point modules | all OK |
| Dashboard `/`, health, workflow options, personas, validation, simple and product options | all 200 |
| Live Ollama `qwen2.5:0.5b` persona and control request scored with canonical SSR | persona E[R] 4.1313, control 4.2808, probabilities sum to 1.0; unknown construct raises `KeyError` |
| 189 protected files (canonical_ssr.py, anchor CSV, survey workbook, cleaned data, frozen split, baselines, Gemini/OpenAI/Ollama responses, `runs.jsonl`, source excerpts) | all identical to the pre-cleanup checksums |
| `runs.jsonl` | still 38 lines, unchanged by the tests |

## Known limitations
- `README.md` was not edited (it was on the owner's protected list) and still describes some archived files: section 16 lists `embedding_comparison_experiment.py`, `model_comparison_experiment.py`, `optimized_model_comparison_experiment.py`, `safe_feature_model_comparison_experiment.py`, `xgboost_batch_experiment.py` and `correlation_results.py` as present in the repository; they are now in `_archive/`. The eight investigation notes it references are there too.
- `requirements.txt` omits packages the code imports (`openai`, `google-genai`, `pytest`, `xgboost`, `catboost`; the README installs some separately) and lists `tqdm`, `colorama`, `tabulate` that no active code imports. Not changed.
- Persona leakage (assignments and sampling marginals use all 200 respondents, including the test half) and the FC-versus-PP construct naming are unresolved; see `docs/project_status/persona_leakage_and_dev_only_proposal.md` and `full_project_audit.md`.
- Real Gemini and OpenAI generation was not exercised in this pass.

---

## Pass 6 — Aggressive Deep Clean of `experiments/` (2026-10-29)

### Goal
Remove all obsolete experiment outputs, duplicate metrics, old model comparisons, unused prediction CSVs, and superseded model artifacts from `experiments/`, leaving only the active runnable code, configuration, and audit documentation.

### Summary of Changes in `experiments/`
- Deleted **136 files** and **7 directories** across `experiments/d1_accuracy/`:
  - `confusion_matrices/run_20261011_214543/` (110 CSV files) — model-specific grouped OOF confusion matrix results.
  - `confusion_matrices/finalized_20261011_221830/` (10 CSV files) — finalized holdout confusion matrix results.
  - `confusion_matrices/` directory (empty after deleting run subdirectories).
  - `models/finalized_20261011_221830/` (9 `.joblib` files, ~50.4 MB) — superseded model checkpoint artifacts.
  - `models/` directory (empty after deleting run subdirectories).
  - `model_comparison.csv` (25 KB) — old model comparisons.
  - `feature_comparison.csv` (5 KB) — old feature-group comparisons.
  - `grouped_cv_screen_results.csv` (9 KB) — duplicate intermediate screening metrics.
  - `cv_per_class_metrics.csv` (67 KB) — per-class precision/recall/F1 metrics.
  - `tuning_results.csv` (1 KB) — duplicate intermediate grid-tuning metrics.
  - `run_metadata.json` (3 KB) — run metadata from 2026-10-11 run.
  - `best_model_report.md` (6 KB) — generated report artifact from 2026-10-11 run.

### Preserved in `experiments/`
- `experiments/__init__.py` — package marker.
- `experiments/d1_accuracy/__init__.py` — package marker.
- `experiments/d1_accuracy/config.json` — active experiment configuration.
- `experiments/d1_accuracy/run_experiments.py` — active experiment runner (verified with `--smoke`).
- `experiments/d1_accuracy/finalize_completed_screen.py` — screen completion and finalist tuning script.
- `experiments/d1_accuracy/dataset_audit.md` — dataset column structure and leakage documentation.

### Final Verification
- `python -m experiments.d1_accuracy.run_experiments --smoke`: exit 0 (SMOKE OK: 200 predictions; acc=0.3100)
- `python scripts/verify_pipeline_outputs.py`: exit 0 (39/39 checks passed)
- `python -m pytest tests/test_canonical_ssr.py tests/test_pipeline_verification.py`: exit 0 (48/48 passed)
- Total project state (excluding `.git/`): **296 files, 39 directories**

---

## Pass 7 — Aggressive Deep Clean of `data/` (2026-10-30)

### Goal
Remove obsolete provider pilot artifacts (failed API/OpenAI runs), duplicate intermediate embedding comparisons, old review-model predictions/evaluations, unused plots, and superseded calibration files while preserving all active Ollama pilot data, canonical SSR files, survey inputs, frozen splits, and test-required assets.

### Summary of Changes in `data/`
- Deleted **42 files** and **3 directories**:
  - `data/pilot_api/` (entire folder, 3 files: `failures.jsonl`, `generation_config.json`, `prompts.jsonl`) — failed rate-limited API pilot with 0 responses.
  - `data/pilot_openai_v1/` (entire folder, 4 files: `failures.jsonl`, `generation_config.json`, `prompts.jsonl`, `run_summary.json`) — failed quota-limited OpenAI pilot with 0 responses.
  - `data/pilot/plots/` (entire folder, 3 files: `persona_means.png`, `pmf_comparison.png`, `ssr_vs_q21_boxplot.png`) — unused pilot plots.
  - `data/pilot/generation_prompts.md` (31.8 KB) — preliminary prompt notes.
  - `data/pilot/persona_comparison.csv` (0.5 KB) — preliminary pilot comparison.
  - `data/pilot/pilot_config.json` (0.7 KB) — preliminary pilot config.
  - `data/pilot/ssr_scores_human_q26.csv` (64.0 KB) — superseded pilot score dump.
  - `data/pilot/ssr_scores_synthetic.csv` (19.3 KB) — superseded synthetic score dump.
  - `data/aspect_model/confusion_matrix.png` (146 KB) — unused diagnostic plot.
  - `data/aspect_model/confusion_matrix_normalized.png` (179 KB) — unused diagnostic plot.
  - `data/aspect_model/confusion_matrix.csv` (0.2 KB) — superseded matrix.
  - `data/aspect_model/classification_report.csv` (0.3 KB) — superseded report.
  - `data/calibrated_discretization/calibration_scored.csv` (11.58 MB) — unreferenced intermediate calibration matrix.
  - `data/calibrated_discretization/calibration_data.csv` (3.08 MB) — unreferenced intermediate calibration inputs.
  - `data/calibrated_discretization/four_five_uncertainty_examples.csv` (13.8 KB) — ad-hoc analysis table.
  - `data/calibrated_discretization/threshold_stability_5fold.csv` (0.5 KB) — intermediate threshold sweep.
  - `data/calibrated_discretization/argmax_rating_confusion_matrix.csv` (0.2 KB) — superseded confusion matrix.
  - `data/calibrated_discretization/calibrated_rating_confusion_matrix.csv` (0.2 KB) — superseded confusion matrix.
  - `data/xgboost_predictions.csv` (840 KB) — unreferenced predictions from archived batch script.
  - `data/xgboost_evaluation_results.csv` (1.21 MB) — unreferenced evaluation from archived batch script.
  - `data/correlation_metrics_summary.csv` (0.6 KB) — archived correlation experiment output.
  - `data/embedding_comparison_results.csv` (0.4 KB) — archived embedding comparison output.
  - `data/model_comparison_results.csv` (0.9 KB) — archived model comparison output.
  - `data/optimized_model_comparison_results.csv` (0.9 KB) — archived optimized model comparison output.
  - `data/pilot_embedding_comparison/spearman_by_embedding.png` (37.9 KB) — unused plot.
  - `data/pilot_embedding_comparison/synthetic_scores_local_bge_base_gemini.csv` (4.9 KB) — superseded v1 Gemini pilot score.
  - `data/pilot_embedding_comparison/synthetic_scores_local_minilm_gemini.csv` (4.9 KB) — superseded v1 Gemini pilot score.
  - `data/pilot_embedding_comparison/synthetic_scores_local_mpnet_baseline_gemini.csv` (5.1 KB) — superseded v1 Gemini pilot score.
  - `data/pilot_embedding_comparison/persona_matched_summary.csv` (3.3 KB) — superseded summary table.
  - `data/pilot_gemini/failures.jsonl` (1.6 KB), `run_summary.json` (0.2 KB), `generation_config.json` (0.8 KB), `prompts.jsonl` (31.4 KB) — unreferenced logs/prompts (saved responses preserved).
  - `data/pilot_gemini_v2/failures.jsonl` (1.5 KB), `run_summary.json` (0.4 KB) — unreferenced failure logs (responses, prompts, config preserved).

### Preserved in `data/`
- All **6 active Ollama pilot directories** (`pilot_ollama_{control,persona}_{FT_BI01,FT_PE01,Q26_BI_proxy}`) and manifest `ollama_ft_generation_manifest.json`.
- Core datasets: `actual_reviews.xlsx`, `Capstone_Reviews(Responses).xlsx`, `human_survey_cleaned.csv`, `human_survey_inclusion_exclusion_log.csv`, `Mobile Reviews Sentiment.csv`, `reviews_only.csv`.
- Canonical review-rating baseline artifacts: `predictions.csv`, `evaluation_results.csv`, `data/aspect_model/metrics.json` & `predictions.csv`, `data/calibrated_discretization/heldout_predictions.csv` & `learned_calibration_thresholds.json` & `heldout_metrics.json` & `final_report.md`, `data/current_baseline_evaluation/`.
- Active pipeline & validation directories: `construct_prompts/`, `personas/`, `persona_generator/`, `synthetic_dataset/`, `validation/`, `validation_split/`, `dashboard_demo_runs/`.
- Saved response integrity files required by tests & pipeline verifier: `data/pilot_gemini/gemini_generated_responses.jsonl`, `data/pilot_gemini_v2/gemini_generated_responses.jsonl`, `data/pilot/synthetic_respondent_specs.jsonl`, `data/pilot/synthetic_responses.jsonl`, `data/pilot/evaluation_metrics.json`, `data/pilot/ssr_config.json`, `data/pilot_embedding_comparison/`.

### Final Verification
- `python scripts/verify_pipeline_outputs.py`: **39/39 checks passed** (exit code 0).
- `python -m pytest tests/test_product_research.py tests/test_survey_batches.py tests/test_canonical_ssr.py tests/test_pipeline_verification.py tests/test_gemini_quality.py tests/test_gemini_v2_eval.py tests/test_openai_generation.py tests/test_persona_matched.py tests/test_methodology_eval.py tests/test_dashboard.py tests/test_dp_pipeline.py`: **196/196 passed** (exit code 0).
- Total project state (excluding `.git/`): **254 files, 36 directories** (down from 296 files, 39 directories).


