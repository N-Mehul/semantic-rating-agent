# Phase 18: Hybrid SSR conversion methods (experimental)

**Status:** exploratory. Canonical SSR (M0: rounded E[r]; Improved SSR k=3, T=0.75, nomic-embed-text, original BI anchors) is unchanged and remains the primary method.

## Inputs (read-only)
- `evaluation/phase13_human_ssr/phase13_scored_human_q26.csv`: 251 human respondents. SSR E[r] and PMF from their Q26 text; label = their own Q21.
- `evaluation/phase16_bi_only_generation/phase16_scored_responses.csv`: 200 synthetic BI answers. SSR E[r] and PMF; label = intended stance (**evaluation only**).

## Methods
| ID | Definition | Fitted on | Uses unlabelled target E[r]? |
|---|---|---|---|
| M0 | round(E[r]) | – | no |
| M2 | round(mu_y + (E − mu_E)·sd_y/sd_E) | human train | no |
| M6 | reweight PMF by w so mean reweighted train PMF = train Q21 shares; round(E') | human train | no |
| M7 | thresholds at target E[r] quantiles = train Q21 cumulative shares | human train shares | **yes (transductive)** |
| M8 | z-score E within each source; thresholds at train-z quantiles of Q21 shares | human train | **yes** |
| M9 | M6, then M2 on the reweighted E' | human train | no |
| M10 | M2 then M7: **identical to M7** (an affine map preserves ranks; verified on all predictions) | – | yes |
| M11 | M6, then M7 on E' | human train | yes |
| M12a/b/c | ensemble of members {M2, M6, M7, M8}: (a) mean of predictions, (b) majority vote (ties → lower median), (c) weights picked by inner 3-fold CV on the train fold only | human train | yes |

## Protocol
- Human: RepeatedStratifiedKFold(5 × 4, seed 20261106), 20 held-out folds, identical to Phase 17. All parameters, thresholds and weights are learned on training folds.
- Paired bootstrap on repeat-0 out-of-fold predictions (each respondent predicted once), plus McNemar.
- **Selection (fixed in advance, human only):** max mean human QWK among methods with QWK > M0 in ≥ 15/20 folds, mean exact ≥ M0 − 0.02, and mean 1★ and 5★ recall > 0; ties go to lower MAE.
- Synthetic: every method refitted on all 251 humans and applied once to the 200 responses. Stances are used only afterwards, as a pass/fail non-deterioration check (QWK bootstrap CI lower bound vs M0 > −0.05). They never choose between methods.

## Integrity notes
- No LLM or embedding calls. No changes to SSR, anchors, codebook or earlier phases.
- M7, M8, M11 and M12 use the unlabelled target E[r] distribution. M7 and M11 force predictions to the human Q21 distribution. The Phase 16 stances were *sampled from* that distribution, so their synthetic distribution match is partly circular.
- No multiple-comparison correction across the 10 candidates. Treat CIs as exploratory.

## Outputs
`phase18_metrics.json`, `phase18_comparison.csv`, `phase18_human_cv_per_fold.csv`, `phase18_human_oof_predictions_repeat0.csv`, `phase18_synthetic_predictions.csv`, `phase18_distributions.png`, `phase18_confusion_matrices.png`, `phase18_stdout.txt`.
