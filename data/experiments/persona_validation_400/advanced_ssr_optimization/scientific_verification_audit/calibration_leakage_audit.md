# Calibration Leakage and Data Flow Audit

## Executive Summary
An end-to-end trace of data structures from raw human benchmarks to fitted transformations was conducted.

## Leakage Audit Checklist

| Audit Question | Classification | Detailed Evidence |
|---|---|---|
| Human dataset split before any parameter fitting? | **No leakage detected** | Data was partitioned via rng.permutation(n_human) before computing mu_train and sigma_train. |
| Were calibration parameters fitted using ONLY training data? | **No leakage detected** | Parameters mu_train and sigma_train were calculated exclusively from h_train. |
| Was the test set used to select the best calibration method? | **Confirmed leakage** | All 4 calibration methods were evaluated on h_test and the best method (linear rescaling) was chosen post-hoc based on test-set W1. |
| Were multiple calibration methods compared on the same test set without validation split? | **Confirmed leakage** | No 3-way train/validation/test or nested cross-validation was used; model ranking occurred directly on test data. |
| Were synthetic observations or personas used to fit human parameters? | **No leakage detected** | Human distribution parameters used only human train observations. |
| Was any human test information indirectly incorporated into synthetic transformations? | **Possible leakage** | Subject clustering was ignored: pooling tech_comfort and rebuy_brand means the same human participant could appear in both train and test splits (subject-level contamination). |
| Were transformation boundaries fitted on complete human dataset? | **No leakage detected** | Only training statistics were used for linear affine coefficients. |
| Was the random split reproducible and appropriately stratified? | **Possible leakage** | Seed 42 was fixed, but the permutation was completely unstratified across constructs and respondent demographics. |
| Is the single 50/50 split sufficient? | **Insufficient evidence** | A single arbitrary split of 258 human items is vulnerable to partition variance; repeated cross-validation is mandatory. |


## Key Architectural Findings
- **Parameter Fitting:** Human training parameters ($\mu_{train}, \sigma_{train}$) were fitted cleanly on the 50% training partition.
- **Model Selection Leakage:** Multiple calibration models were evaluated on the held-out test partition and ranked. While acceptable for exploratory comparison, the best model was selected based on test performance without a separate validation split.
- **Subject Contamination:** By unnesting `tech_comfort` and `rebuy_brand` without respondent-level grouping, responses from the same human respondent were present across train and test partitions in the original run. Our Phase 4 robustness audit rectified this with respondent-stratified splits.
