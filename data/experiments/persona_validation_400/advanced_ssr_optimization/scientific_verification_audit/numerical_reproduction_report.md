# Independent Numerical Reproduction Report

## Overview
All reported baseline statistics, parameter grids, and calibration numbers were independently recalculated from raw source data (`ssr_scored_responses.csv` and `human_survey_clean.csv`).

## Summary Table

| Metric                      |   Reported_Value |   Recalculated_Value |   Absolute_Difference | Verification_Status   | Explanation                            |
|:----------------------------|-----------------:|---------------------:|----------------------:|:----------------------|:---------------------------------------|
| Synthetic_N_Total           |       400        |           400        |                     0 | VERIFIED              | Exact match within numerical precision |
| Synthetic_N_Valid           |       343        |           343        |                     0 | VERIFIED              | Exact match within numerical precision |
| Human_N_Pooled              |       516        |           516        |                     0 | VERIFIED              | Exact match within numerical precision |
| Synthetic_Mean              |         3.3424   |             3.3424   |                     0 | VERIFIED              | Exact match within numerical precision |
| Synthetic_Std               |         0.6225   |             0.6225   |                     0 | VERIFIED              | Exact match within numerical precision |
| Synthetic_Median            |         3.5272   |             3.5272   |                     0 | VERIFIED              | Exact match within numerical precision |
| Synthetic_IQR               |         1.054    |             1.054    |                     0 | VERIFIED              | Exact match within numerical precision |
| Human_Mean                  |         3.6124   |             3.6124   |                     0 | VERIFIED              | Exact match within numerical precision |
| Human_Std                   |         1.0762   |             1.0762   |                     0 | VERIFIED              | Exact match within numerical precision |
| Mean_Diff_Synth_Minus_Human |        -0.27     |            -0.27     |                     0 | VERIFIED              | Exact match within numerical precision |
| Wasserstein_Distance        |         0.4881   |             0.4881   |                     0 | VERIFIED              | Exact match within numerical precision |
| KS_Statistic                |         0.4551   |             0.4551   |                     0 | VERIFIED              | Exact match within numerical precision |
| PE_ANOVA_F                  |         4.3475   |             4.3475   |                     0 | VERIFIED              | Exact match within numerical precision |
| PE_ANOVA_p                  |         0.007518 |             0.007518 |                     0 | VERIFIED              | Exact match within numerical precision |
| PE_Eta_Squared              |         0.1693   |             0.1693   |                     0 | VERIFIED              | Exact match within numerical precision |
| PE_P1_vs_P3_Mean_Diff       |         0.5261   |             0.5261   |                     0 | VERIFIED              | Exact match within numerical precision |
| PE_P1_vs_P3_Tukey_p         |         0.003368 |             0.003368 |                     0 | VERIFIED              | Exact match within numerical precision |
| PE_P1_vs_P3_Hedges_g        |         1.0598   |             1.0598   |                     0 | VERIFIED              | Exact match within numerical precision |

## Discrepancies and Inconsistencies Detected

| Metric                            | Reported                                                   | Recalculated                                                  |   Absolute_Difference | Impact                                                                                                                                  |
|:----------------------------------|:-----------------------------------------------------------|:--------------------------------------------------------------|----------------------:|:----------------------------------------------------------------------------------------------------------------------------------------|
| Anchor_Monotonicity_PE            | rho <= -0.92 (in report text line 79)                      | rho = -0.4005 (in anchor_quality_analysis.csv)                |                0.5195 | CRITICAL: Text claim in report drastically overstated anchor monotonicity relative to underlying CSV!                                   |
| Anchor_Monotonicity_EE            | rho <= -0.92 (in report text line 79)                      | rho = -0.4322 (in anchor_quality_analysis.csv)                |                0.4878 | CRITICAL: Text claim in report drastically overstated anchor monotonicity relative to underlying CSV!                                   |
| Anchor_Monotonicity_HM            | rho <= -0.92 (in report text line 79)                      | rho = -0.6229 (in anchor_quality_analysis.csv)                |                0.2971 | CRITICAL: Text claim in report drastically overstated anchor monotonicity relative to underlying CSV!                                   |
| Anchor_Monotonicity_SI            | rho <= -0.92 (in report text line 79)                      | rho = -0.1653 (in anchor_quality_analysis.csv)                |                0.7547 | CRITICAL: Text claim in report drastically overstated anchor monotonicity relative to underlying CSV!                                   |
| Anchor_Monotonicity_PP            | rho <= -0.92 (in report text line 79)                      | rho = 0.1335 (in anchor_quality_analysis.csv)                 |                1.0535 | CRITICAL: Text claim in report drastically overstated anchor monotonicity relative to underlying CSV!                                   |
| Anchor_Monotonicity_BI            | rho <= -0.92 (in report text line 79)                      | rho = 0.0381 (in anchor_quality_analysis.csv)                 |                0.9581 | CRITICAL: Text claim in report drastically overstated anchor monotonicity relative to underlying CSV!                                   |
| Baseline_Mean_Definition_Conflict | Mean=3.4057, SD=0.7623 (in ablation_study_results.csv)     | Mean=3.3424, SD=0.6225 (in baseline_reproduction_metrics.csv) |                0.0633 | CRITICAL: Ablation script recomputed k=3 top-filtering, whereas baseline reproduction used raw canonical ratings (k=4/5 unconstrained)! |
| Linear_Calibration_Test_W1_Claim  | W1 = 0.291 (40.5% error reduction, in report text line 97) | W1 = 0.3377 (31.0% reduction, in calibration_experiments.csv) |                0.0467 | HIGH: Report text quoted an outdated or miscalculated figure of 0.291 instead of actual CSV result 0.3377.                              |

### Critical Discrepancy Findings:
1. **Anchor Monotonicity Overstatement:** In `advanced_ssr_optimization_report.md` (line 79), it was claimed that all 6 constructs had Spearman $\rho \le -0.92$. In reality, the recalculated values in `anchor_quality_analysis.csv` show PE $\rho = -0.40$, EE $\rho = -0.43$, HM $\rho = -0.62$, SI $\rho = -0.17$, PP $\rho = +0.13$, and BI $\rho = +0.04$.
2. **Canonical Baseline Definition Conflict:** `ablation_study_results.csv` recomputed 'Canonical Baseline' with explicit top-3 truncation ($N=343$, Mean=3.4057, SD=0.7623), whereas `baseline_reproduction_metrics.csv` and the original paper baseline used unconstrained canonical minimum-subtraction ($N=343$, Mean=3.3424, SD=0.6225). This subtle distinction must be transparently clarified.
3. **W1 Error Reduction Discrepancy:** Report text cited 0.291 (40.5% reduction), but the actual CSV value was 0.3377 (31.0% reduction).
