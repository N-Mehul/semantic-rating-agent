# Calibrated Discretization Layer Experiment

## Protocol

The official evaluation set contains 10003 rows in a separate workbook. Calibration used 39997 complete labeled rows from the separate source CSV; no official workbook records were used for calibration. The SSR mathematics and protected files were unchanged.

`argmax_rating` is the old probability-mode decision. `calibrated_rating` is assigned by frozen ordered cutpoints over the unchanged `expected_rating`. `expected_rating` remains the continuous SSR score. `likely_rating_range` is the smallest contiguous interval containing the calibrated rating whose probability mass reaches 70%; ties expand to the higher rating.

Frozen cutpoints (expected-rating boundaries): 1.549450, 2.498950, 3.227750, 4.004550.

## Held-Out Results

| Decision | Exact | +/-1 | MAE | RMSE | Spearman | Pearson | Precision | Recall | F1 | R3 recall | R4 recall | R5 recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| argmax_rating | 46.14% | 93.55% | 0.6055 | 0.8626 | 0.7755 | 0.7696 | 35.82% | 43.94% | 36.11% | 9.16% | 90.99% | 0.00% |
| calibrated_rating | 43.28% | 95.22% | 0.6165 | 0.8474 | 0.7511 | 0.7424 | 36.71% | 35.82% | 27.70% | 14.47% | 90.99% | 0.00% |

## Interpretation

Rating 5 prediction changed from 0 argmax predictions to 0 calibrated predictions. This is a decision redistribution, not an artificial 5-star override; all calibrated values come from the frozen calibration objective.

4259 held-out rows have a 4-5 or 3-5 likely range under the documented 70% contiguous-mass rule; the evidence CSV saves the first 100 examples.
