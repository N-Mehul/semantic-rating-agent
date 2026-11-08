# Model Performance and Accuracy Audit — Digital Persona SSR

**Date:** 2026-11-04 · **Type:** evaluation-only (no project file, model, dataset or result was modified)
**Reproduce:** `python evaluation/model_performance_audit/run_audit.py` (set `HF_HUB_OFFLINE=1`) · all numbers below are read from `metrics/` · figures (300 dpi PNG + vector PDF) in `figures/`

---

## 1. Executive summary

The project contains **three different evidence layers**, and they must not be merged:

| Layer | What is evaluated | Ground truth? | Headline result |
|---|---|---|---|
| **A. Review-rating SSR** | 5-anchor SSR on 10,003 held-out mobile reviews | **Yes**: review star rating | Exact 46.1%, ±1 93.6%, MAE 0.61 (integer) / 0.63 (E[r]), Spearman 0.70, R² 0.61. **Good ordinal agreement, weak exact classification; never predicts 5★.** It sits at the dataset's text-only ceiling (47.0%) because the dataset has only 110 distinct, templated texts. |
| **B. Digital Persona SSR on human survey text** | Canonical and Improved SSR (k=3, T=0.75) on real respondents' Q26 free text vs their own Q21 rebuy rating | **Yes**: human Q21 (n=251; frozen test n=97) | Exact 31%, MAE 0.92, Spearman 0.22 [0.09, 0.32], R² ≈ 0. **Poor. It does not beat a majority-class or mean predictor in MAE or accuracy.** |
| **C. Synthetic persona responses (P1–P4)** | 400 Qwen-2.5-7B responses scored by SSR | **No**: no per-response or per-persona human labels | Accuracy **cannot be calculated**. Personas are distinguishable in *text* (63% vs 25% chance) but **not in SSR ratings** (η² = 0.005, p = 0.59). PE is the only construct with a persona effect (p = 0.037). |

**Bottom line:** SSR is a good *ordinal sentiment scorer* on clean review text (Layer A). As a predictor of *individual human survey intentions* it is currently not better than trivial baselines (Layer B). The Digital Persona layer produces valid, diverse, rating-free text, but persona identity does not carry through into the ratings (Layer C). The paper can report SSR as a validated scoring mechanism (A), a negative/preliminary result for individual prediction (B), and distributional plus differentiation diagnostics (C). It cannot claim persona prediction accuracy.

---

## 2. Datasets and evaluation methodology

| Dataset | File | n | Use |
|---|---|---|---|
| Review held-out set | `data/calibrated_discretization/heldout_predictions.csv` + metadata from `data/actual_reviews.xlsx` (header-less; columns from the CSV) | 10,003 rows, **110 unique texts** | Supervised metrics (A) |
| Review training source | `data/Mobile Reviews Sentiment.csv` minus held-out IDs | 39,996 | Baselines (majority, prior, TF-IDF, sentiment lookup) |
| Human survey | `data/human_survey_cleaned.csv`, primary cohort with usable Q26 text | **251** (frozen split manifest covers 194: dev 97 / test 97) | Supervised metrics (B) |
| Synthetic personas | `data/experiments/persona_validation_400/canonical_vs_improved_comparison/canonical_vs_improved_responses.csv` | 400 (100 per persona) | Unsupervised diagnostics (C) |
| Product-evaluation runs | `data/product_evaluation/*/persona_responses.json` | 57 runs, 1,131 texts | Relevance check (C/§5) |

**Method notes**
- SSR code was imported unmodified (`canonical_ssr.similarities_to_pmf`, `improved_ssr.improved_similarities_to_pmf`). Saved Improved/Canonical E[r] were reproduced to within 5e-5 (rounding).
- Encoder for Layer B and the text analyses: `all-mpnet-base-v2` (the project's baseline encoder); anchors: approved BI set (`get_construct_anchors("BI")`).
- **MAE/MSE/RMSE/Pearson/Spearman/R²** use the continuous E[r]. **Exact / ±1 / precision / recall / F1 / κ** use the integer decision (argmax unless stated). Integer MAE/RMSE are also saved (`mae_int`, `rmse_int`). They reproduce the project's stored baseline exactly (0.6055 / 0.8626).
- CIs: 95% percentile bootstrap. In Layer A they are a **cluster bootstrap over the 110 texts**, because rows sharing a text are not independent. Paired differences use the same resampling.
- Persona tests: one-way ANOVA, Kruskal–Wallis, Levene, η² with bootstrap CI, Welch t / Mann–Whitney pairwise with Holm correction, Hedges g with CI.

---

## 3. Performance metrics

### 3.1 Layer A: review-rating SSR (human ground truth, n = 10,003)

| Model | Exact | ±1 | MAE | MSE | RMSE | Pearson | Spearman | R² | QW-κ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **SSR argmax (PMF mode)** | **46.1%** | **93.6%** | **0.629** | 0.619 | **0.787** | **0.782** | **0.700** | **0.611** | 0.763 |
| SSR rounded E[r] | 43.1% | 95.3% | 0.629 | 0.619 | 0.787 | 0.782 | 0.700 | 0.611 | 0.710 |
| SSR calibrated cutpoints | 43.3% | 95.2% | 0.629 | 0.619 | 0.787 | 0.782 | 0.700 | 0.611 | 0.712 |
| 95% cluster CI (argmax) | 44.9–47.4 | 92.7–94.3 | 0.616–0.643 | | 0.774–0.799 | 0.761–0.800 | 0.638–0.740 | 0.578–0.639 | |

| Metric | Meaning | Value | Grade |
|---|---|---|---|
| Exact accuracy | share of rows where predicted star = true star | 46.1% | moderate (5-class chance ≈ 21%, but ceiling is 47%) |
| MAE | average distance in stars | 0.63 | good–moderate |
| MSE / RMSE | penalise large misses; RMSE in stars | 0.62 / 0.79 | moderate |
| ±1 accuracy | prediction within one star | 93.6% | good |
| Pearson | linear agreement of E[r] with truth | 0.78 | good |
| Spearman | rank agreement | 0.70 | good |
| R² | share of rating variance explained | 0.61 | good (appropriate here because the target is near-interval and n is large) |

**Per-rating (argmax):**

| Rating | Support | Precision | Recall | F1 | Mean signed error of E[r] |
|---|---:|---:|---:|---:|---:|
| 1★ | 1,320 | 0.52 | **0.77** | 0.62 | +0.81 (over-predicts) |
| 2★ | 1,989 | 0.42 | 0.42 | 0.42 | +0.29 |
| 3★ | 2,412 | 0.39 | **0.09** | 0.15 | +0.19 |
| 4★ | 2,785 | 0.46 | **0.91** | 0.61 | −0.19 |
| 5★ | 1,497 | **0.00** | **0.00** | 0.00 | **−1.07** (under-predicts) |

Confusion matrix: `metrics/A_confusion_argmax.csv`, `figures/fig_A_confusion_matrix_review_ssr.png`. The model **never predicts 5★** (1,485 of 1,497 true 5★ rows go to 4★). It also absorbs 3★ into 2★ and 4★. Overall bias is ≈ 0 (−0.001), but it is extreme-compressing: prediction SD 0.96 vs truth 1.26.

**Key context: label noise ceiling.** Every one of the 110 texts carries several different star ratings (100% of texts). The best possible text-only predictor (the per-text majority star, an oracle) reaches only **47.0% exact / 93.5% ±1**. SSR's 46.1% / 93.6% is therefore **at the ceiling of what any text model can achieve on this dataset**. The remaining error comes from label noise, not from SSR.

**Generalisation across brands:** exact accuracy is 44.8–48.2% for all seven brands (Spearman 0.68–0.72), so performance is stable (`A_per_brand.csv`).

### 3.2 Layer B: Digital-Persona SSR vs human Q21 (human ground truth)

Q21 asks how likely the respondent is to rebuy the same brand. SSR scores the same respondent's Q26 free text against the adapted BI anchors. This is a **proxy construct match**.

| Split | Model | n | Exact | ±1 | MAE | RMSE | Pearson | Spearman | R² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full | Canonical SSR | 251 | 31.1% | 68.9% | 0.938 | 1.151 | 0.196 | 0.147 | −0.019 |
| full | **Improved SSR k=3, T=0.75** | 251 | 31.1% | 68.9% (80.1% rounded) | **0.919** | 1.139 | 0.230 | **0.216** [0.093, 0.323] | 0.003 |
| full | Majority class (3) | 251 | **33.9%** | 69.3% | 0.968 | 1.258 | — | — | −0.216 |
| full | Mean constant (optimistic) | 251 | 31.9% | 87.3% | 0.932 | 1.141 | — | — | 0.000 |
| frozen test | Canonical SSR | 97 | 32.0% | 72.2% | 0.891 | 1.109 | 0.214 | 0.170 [−0.04, 0.37] | −0.009 |
| frozen test | Improved SSR | 97 | 32.0% | 72.2% | 0.907 | 1.139 | 0.167 | 0.165 [−0.03, 0.35] | −0.064 |
| frozen test | Majority class | 97 | 35.1% | 72.2% | 0.928 | 1.218 | — | — | −0.218 |

Grades: exact accuracy **poor**, MAE **poor–moderate**, correlation **poor** (ρ ≈ 0.2, CI excludes 0 only on the full set), R² **poor** (≈ 0, so the model explains no variance).

**Per-rating (Improved, full):** 3★ recall 0.80 (F1 0.47); 4★ recall 0.09; 5★ recall 0.06; **1★ and 2★ recall 0.00**. 68 of the 85 true 3★, 66 of the 80 true 4★ and 49 of the 54 true 5★ respondents are predicted 3★. In practice the model predicts "3" for most respondents (`B_confusion_improved_argmax_full.csv`).

**Paired significance (full set, bootstrap of |error| difference):**
- Improved vs majority: −0.049 MAE [−0.117, +0.017], p = 0.15 (**not significant**)
- Improved vs a-priori lexicon: −0.041 [−0.133, +0.052], p = 0.38 (**n.s.**)
- Improved vs Canonical: −0.019 [−0.044, +0.005], p = 0.12 (**n.s.**)

**Robustness:** over 50 random stratified half-splits, Improved SSR exact = 30.8 ± 1.9%, MAE = 0.917 ± 0.033, Spearman = 0.22 ± 0.07. SSR is deterministic, so the variation comes from the sample alone. The result is stable, but it is stable at a weak level.

---

## 4. Improved SSR (k = 3, T = 0.75): behaviour analysis

| Property (synthetic n = 400) | Canonical | Improved | Human pooled Q20/Q21 |
|---|---:|---:|---:|
| PMF valid (non-negative, sums to 1) | ✔ | ✔ | — |
| Mean E[r] | 3.32 | 3.41 | 3.63 |
| SD of E[r] | 0.648 | **0.820** | 1.071 |
| SD ratio vs human | 0.60 | **0.77** | 1 |
| E[r] range | 1.58–4.32 | 1.46–4.56 | 1–5 |
| E[r] ≥ 4.5 or ≤ 1.5 | 0% | 0.5% | 28.4% at endpoints |
| Mean PMF entropy (bits; max 2.32) | 1.78 | 1.44 | — |
| Zero-probability PMF cells | 20% | 40% | — |

- **Variance preservation:** Improved SSR recovers about 27% more spread than Canonical, but it still compresses human variance by 23%.
- **Extreme-rating compression:** it remains severe. Only 0.5% of synthetic E[r] reach the outer half-stars, while 28% of human answers are 1 or 5. The argmax distribution is identical for both variants, because truncation and temperature don't move the mode.
- **Calibration and uncertainty (against human Q21, n = 251):**

| | ECE (top-1) | Brier | NLL (ε=1e-3) | 80% set coverage | 90% set coverage |
|---|---:|---:|---:|---:|---:|
| Canonical | 0.160 | 0.817 | 1.93 | 76.5% | 84.1% |
| Improved | **0.270** | 0.882 | 2.44 | **59.8%** | 76.9% |
| Constant human-prior PMF | 0.000 | **0.728** | **1.40** | 87.3% | 96.4% |

  Both SSR PMFs are **over-confident and worse calibrated than simply predicting the human prior**. Improved is worse than Canonical here: truncation puts 40% of cells at exactly zero, so a respondent whose true answer lands there gets zero probability. PMF confidence does not track correctness (Spearman ≈ 0.02). **The PMF should not be presented as a calibrated uncertainty estimate.**

- **Sensitivity to k and T** (separate experiment; the model was not changed; `B_k_T_sensitivity_human_q21.csv`, `C_k_T_sensitivity_synthetic.csv`, `fig_B_k_T_sensitivity_heatmaps.png`):
  - Against human Q21, MAE varies only between 0.90 and 1.02, and Spearman between 0.01 (k=1) and 0.26 (k=2, T≥1). No setting reaches a useful level. **k = 3, T = 0.75 is not uniquely optimal; k = 2–3 with T = 1–2 is equal or slightly better.** This is a post-hoc observation on the evaluation data and must not be used to re-tune.
  - On the synthetic data, lower T and k raise the spread (k=1: SD 1.21, entropy 0) but never create persona differentiation (η² ≤ 0.007 for every k and T). The rank order of responses is stable (ρ ≥ 0.90 vs default for k ≥ 2).

---

## 5. Persona and construct analysis (synthetic; no ground truth)

**No persona-level ground-truth labels exist. Nothing in this section is accuracy.**

**5.1 Do personas produce different ratings?** (Improved SSR; `C_persona_omnibus_tests.csv`)

| Construct | n | ANOVA p | Kruskal p | η² [95% CI] | Verdict |
|---|---:|---:|---:|---:|---|
| All pooled | 400 | 0.588 | 0.520 | 0.005 [0.001, 0.035] | no difference |
| **PE** | 68 | **0.037** | **0.019** | **0.123** [0.026, 0.348] | medium–large effect (P1 3.70 vs P3 3.05) |
| EE | 68 | 0.988 | 0.962 | 0.002 | none |
| HM | 65 | 0.953 | 0.473 | 0.006 | none |
| SI | 75 | 0.999 | 0.996 | 0.000 | none |
| PP | 57 | 0.991 | 0.910 | 0.002 | none (construct invalid, see 5.3) |
| **BI** | 67 | 0.986 | 0.992 | 0.002 | **none** |

After Holm correction, **no pairwise persona comparison is significant** in any construct (all pooled pairs |g| ≤ 0.19). PE is significant in the omnibus test only, with about 17 responses per cell. Canonical SSR gives the same pattern (PE p = 0.0075, η² = 0.17).

**5.2 Do personas produce different text?** Yes. A logistic classifier on mpnet embeddings identifies the persona from the response text with **62.9% ± 0.8% accuracy** (5-fold CV, chance 25%; permutation null 25.3%, p = 0.0099). Budget and price vocabulary rises monotonically from P1 to P4 (0.30 → 0.89 mentions per response, Kruskal p = 0.0006), and premium vocabulary peaks in P1 (0.47). **Persona conditioning works in the language, but SSR discards it**: the BI/construct anchors measure positive vs negative intention, and every persona writes mostly positive answers.

**5.3 Construct alignment.** For each response, I checked which construct's anchor set it is closest to:

| PE | EE | HM | SI | BI | **PP** | Overall |
|---:|---:|---:|---:|---:|---:|---:|
| 84% | 84% | 89% | 84% | 79% | **0%** | 72% (chance 17%) |

PP responses are never closest to the PP anchors. This confirms the documented construct mismatch: PP was prompted as Price/Value but scored against Perceived-Protection anchors. **PP results are invalid and must be excluded.**

**5.4 BI prediction performance.** Individual-level performance is only measurable on human text (Layer B: ρ = 0.22, not better than baselines). For synthetic BI, only distributional comparison with human Q21 is possible (`C_synthetic_vs_human_distribution_NOT_ACCURACY.csv`): mean 3.41 vs 3.51 (d = −0.09), W₁ = 0.46, SD ratio 0.72, and KS p < 0.001. The distributions differ significantly, mainly because the synthetic ones are compressed. EE vs Q20: d = −0.15, W₁ = 0.36, KS p = 0.004.

Figure: `fig_C_persona_x_construct_heatmap_and_effect_sizes.png` shows mean E[r] per persona × construct, plus η² with CIs.

---

## 6. Synthetic response quality

| Check | Result |
|---|---|
| Unique texts | 400 / 400 |
| Length | 44.8 words, 3.0 sentences on average |
| Lexical diversity | distinct-1 0.20, distinct-2 0.66, distinct-3 0.85; 2.3% of sentences repeated verbatim |
| Semantic redundancy | mean pairwise cosine 0.40; 26 pairs > 0.90 and 5 pairs > 0.95 |
| Template reuse | near-duplicates across personas with only persona nouns swapped (e.g. `SYN_P2_BI_087` vs `SYN_P4_BI_087`, cos 0.95: "…during long workdays…" ↔ "…during busy semester exam periods…") |
| Repeated openings | "I plan to" ×11, "I will continue" ×8 (mostly BI) |
| Same-construct vs cross-construct similarity | 0.54 vs 0.37 (responses cluster by construct, as intended) |
| **Direct numerical ratings** | **0 regex hits**. 10 responses contain digits, all spec numbers ("120Hz", "4K", "5G"), and 3 use rating-like words in non-rating senses |
| Contradiction screen (lexical) | 5.8% contain both positive and negative cue words; manual inspection shows these are nuance (e.g. "without any lag") rather than contradictions |
| Convergent validity | SSR E[r] vs an independent lexicon score: Spearman 0.36 (agreement between two automatic scorers, not accuracy) |

**Product-evaluation runs (57 runs, 1,131 texts):** responses are more similar to their own product description than to a random one (cos 0.44 vs 0.34, Wilcoxon p ≈ 4e-75), so they are relevant on average. However, only 9% name the product, and 14% are exact duplicates. Some responses for non-phone products describe a smartphone, e.g. Echo Vision Smart Glasses: *"The smartphone is a solid choice for everyday use…"*. The automated flag rate (16%) is an upper bound, because it also counts legitimate phrases like "paired with my phone". These are cases of **product-relevance drift**.

---

## 7. Baseline comparisons

### 7.1 Layer A (same 10,003 rows)

| Model | Exact | ±1 | MAE | RMSE | Spearman | Paired MAE diff vs SSR [95% CI] |
|---|---:|---:|---:|---:|---:|---|
| **SSR argmax** | 46.1% | 93.6% | 0.629 | 0.787 | 0.700 | — |
| Random (train prior; 30 seeds 21.7 ± 0.4%) | 21.3% | 58.2% | 1.415 | 1.781 | −0.01 | SSR better by 0.79 [0.75, 0.82] |
| Majority class (4★) | 27.8% | 66.9% | 1.184 | 1.541 | — | SSR better by 0.56 [0.44, 0.70] |
| Train-mean constant | 24.1% | 71.8% | 1.058 | 1.261 | — | SSR better by 0.43 [0.39, 0.48] |
| Sentiment-label lookup ⚠ | 46.3% | 93.6% | 0.627 | 0.786 | 0.781 | equal (Δ 0.003) |
| TF-IDF + LogReg (supervised) | 45.4% | 93.8% | 0.629 | 0.788 | 0.700 | equal (Δ 0.001 [−0.001, 0.003]) |
| **Aspect RF D1 ⚠ TARGET LEAKAGE** | 73.3% | 99.9% | 0.268 | 0.520 | 0.917 | not comparable |

- SSR is **significantly better than all trivial baselines** and **equal to a supervised text classifier**. That is a strong result for a zero-training method, though it is partly forced by the 47% ceiling.
- ⚠ The sentiment label is near-deterministic in rating (e.g. 0 Negative labels at 5★, 21 Positive labels at 1★), so it is a leaky reference and not a fair baseline.
- ⚠ **Aspect RF:** the five aspect sub-ratings correlate with the target at Spearman 0.94 and are co-assigned with it. Its 73% is target leakage, not text understanding, and **it must not be presented as a competing model**.
- **Leakage between train and eval in Layer A:** 82% of the held-out unique texts (84% of rows) also appear in the training source. That matters for the supervised TF-IDF baseline, but not for SSR, which is never trained on this data. Leave-one-brand-out gives the same picture (TF-IDF 44–47%, SSR 45–48%).

### 7.2 Layer B (same 251 respondents)

| Model | Exact | ±1 | MAE | Spearman |
|---|---:|---:|---:|---:|
| Improved SSR | 31.1% | 68.9% | 0.919 | 0.216 |
| Canonical SSR | 31.1% | 68.9% | 0.938 | 0.147 |
| Majority class | **33.9%** | 69.3% | 0.968 | — |
| Random prior (100 seeds) | 26.9 ± 2.6% | — | 1.232 | −0.01 |
| A-priori sentiment lexicon | 34.3% | 76.5% | 0.960 | 0.024 |
| TF-IDF + LogReg, 2-fold CV ×10 | 31.6 ± 1.5% | 77.2% | **0.891** | **0.284** |
| TF-IDF + LogReg, frozen dev → test (n=97) | 32.0% | 80.4% | 0.871 | 0.351 |

On human survey text, **no method does well**. SSR beats random but not majority-class or a tiny supervised classifier. The ceiling problem is different here: brand-rebuy intention is weakly expressed in short Q26 answers.

---

## 8. Reliability and robustness

| Check | Finding |
|---|---|
| Train/test leakage (SSR) | None. SSR has no trained parameters on either dataset; the frozen dev/test split is verified. |
| Train/test leakage (baselines) | Layer A TF-IDF sees 82% of eval texts in training (inflates it). Aspect RF uses target-coupled features. The sentiment label derives from the rating. |
| Calibration leakage | Layer A calibrated cutpoints were fitted on a separate 39,997-row file; the held-out workbook was not used. |
| Duplicates | Layer A: 10,003 rows but only **110 unique texts**. The effective n for text-level inference is 110, so cluster CIs are used. Synthetic: 0 exact duplicates, 26 near-duplicate pairs. Product runs: 14% duplicate texts. |
| Sample size | Layer B n = 251 (test 97): a CI half-width of about ±0.12 on ρ, so it is under-powered for small effects. Layer C: about 17 per persona × construct cell, which detects only η² ≳ 0.15. |
| Class imbalance | Q21: 1★ = 23, 2★ = 9 (13% combined). Per-class metrics for 1–2★ are unreliable. Layer A is reasonably balanced (13–28% per class). |
| Bootstrap CIs | Reported throughout (`*_bootstrap_ci*.json`). |
| Repeated evaluation | 50 stratified half-splits (Layer B); 10 × 2-fold CV for TF-IDF; 5 × 5-fold CV for persona identifiability. |
| Significance of differences | Layer A: SSR vs trivial baselines p < 0.05 (CI excludes 0); SSR ≈ TF-IDF. Layer B: no SSR comparison significant. |
| Seed sensitivity | SSR is deterministic. Random-baseline seeds: SD 0.4% (A), 2.6% (B). Split seeds: SD 1.9% in exact accuracy (B). |
| Generalisation | Brands (A): stable, 44.8–48.2%. Unseen product concepts: relevance is above chance, with category drift as described in §6. |
| Encoder dependence | Stored project result: encoder agreement on Gemini v2 E[r] is Spearman 0.01–0.24 between three local encoders, so the scores are encoder-sensitive. |
| Human cohort drift | The frozen split manifest lists 194 respondents; the current cleaned file has 251 usable rows. The full-set numbers here use 251; the frozen-test numbers use the 97 manifest IDs. |

---

## 9. Limitations and sources of bias

1. **Proxy ground truth.** Q21 (rebuy *brand*) is scored against anchors about continuing to use the *phone* (BI adaptation). Q20 tech comfort is a loose proxy for EE. No human ground truth exists for PE, HM, SI or PP.
2. **No persona ground truth.** Synthetic personas cannot be scored for accuracy. The P1–P4 labels of the 400-response set ("Premium Tier", "Mid-Range Heavy", "Budget Practical", "Student Value") differ from the human-cohort definitions in `data/personas/` (e.g. `P3_midrange_moderate_user`, `P4_budget_or_undisclosed_price`), so persona-level human comparison would require a mapping that hasn't been validated.
3. **Templated review data.** Layer A is a 110-template dataset with noisy labels. Strong correlations there show that SSR orders sentiment correctly, not that it predicts real-world reviews.
4. **Single LLM, single generation per spec** (Qwen-2.5-7B) for the 400 set; stability across repeated generations is untested.
5. **Anchor-set and encoder dependence.** There is one adapted anchor set per construct (not the six of the original method), and local encoders were used instead of `text-embedding-3-small`.
6. **Positivity bias** in synthetic text: 56% of Improved E[r] fall in [3.5, 4.5), which limits rating variance.
7. **Analyst degrees of freedom.** The k/T grid here is exploratory. The project's choice of k=3, T=0.75 was made on synthetic distributional criteria, not on human accuracy.

---

## 10. Strengths and weaknesses

**Strengths**
- The mathematics is correct: PMFs are valid, E[r] is consistent, and saved results reproduce to 5e-5.
- On review text, a training-free SSR matches a supervised classifier and reaches the label-noise ceiling (ρ = 0.70, ±1 = 94%).
- Clean separation of the LLM from rating: 0 numeric ratings in 400 responses.
- Persona conditioning produces measurably different language (63% identifiable), and responses align with their intended construct (79–89%, PP excepted).
- The evaluation practice is good (frozen splits, leakage audits, labelled diagnostic-only models).

**Weaknesses**
- Individual-level prediction of human survey intention is not better than majority-class (Spearman 0.22, R² ≈ 0).
- Systematic extreme compression: 5★ is never predicted (A), and 1–2★ are never predicted (B).
- PMF uncertainty is miscalibrated (worse than a constant prior), and Improved SSR makes calibration worse.
- Persona differences vanish in the ratings (η² ≈ 0.005), except PE.
- The PP construct is invalid, and product-relevance drift appears in the product-evaluation runs.

---

## 11. Conclusion: is the model performing well?

- **As a semantic rating mechanism on review-style text: yes, moderately well**, at the data's ceiling. Rating agreement is moderate (46% exact), ordinal correlation is good (0.70–0.78), and error magnitude is good (MAE ≈ 0.6). Its weakness is the extreme categories.
- **As a predictor of individual human survey ratings: no.** Agreement and correlation are poor, and it does not significantly beat trivial baselines.
- **As a persona simulator:** it produces *textual* differentiation but *not rating* differentiation. Distributional alignment with humans is partial: mean difference < 0.15 points, variance about 25% too low, and endpoints almost absent.

These are four different properties: **rating agreement** (A good, B poor), **correlation** (A good, B weak), **distributional alignment** (partial), and **persona differentiation** (text yes, ratings no). They must be reported separately.

---

## 12. Recommendations

1. Report Layer B honestly as a negative result, with the baseline table, in the paper. Do not merge it with Layer A.
2. Fix the PP prompt/anchor mismatch, then regenerate PP. Until then, exclude PP everywhere.
3. Add construct-specific *intensity* anchors that encode persona-relevant attributes (price sensitivity, value) if persona differences should appear in ratings. Test this on dev data only.
4. Do not market the PMF as calibrated uncertainty. If uncertainty is needed, fit a post-hoc calibrator (e.g. temperature or isotonic) on the human dev split and evaluate it on test. Consider k ≥ 4 to avoid zero-probability cells.
5. Collect human answers to the construct-specific questions (FT_PE01, FT_BI01, …) so that per-construct accuracy becomes measurable. Collect a few hundred more respondents with 1–2★ answers.
6. Repeat generation (≥ 3 seeds per spec, ≥ 2 LLMs) to measure stability. Deduplicate template-like near-duplicates (cos > 0.93).
7. Add a product-category guard to product-evaluation prompts, and reject responses that name the wrong category.
8. Drop the Aspect RF from model comparisons, or present it only as a leakage illustration.

---

## 13. Validated vs experimental results

| Result | Status | Basis |
|---|---|---|
| Layer A metrics (exact 46.1%, MAE 0.61/0.63, ρ 0.70, R² 0.61), confusion matrix, per-rating, baselines | **VALIDATED against human ratings** (held-out, cluster-bootstrap CIs). Caveat: templated, noisy dataset | `metrics/A_*` |
| Layer B metrics (Spearman 0.22, MAE 0.92, exact 31%), calibration, baselines | **VALIDATED against human ratings** (proxy construct; frozen split) | `metrics/B_*` |
| k/T sensitivity grids | **EXPERIMENTAL / exploratory**; not used for selection | `B_k_T_*`, `C_k_T_*` |
| Persona differentiation (ANOVA, η², pairwise, text identifiability) | **Internal synthetic statistics; no ground truth** | `metrics/C_persona_*` |
| Synthetic-vs-human distribution (W₁, KS, d, SD ratio) | **Distributional alignment only; not accuracy** | `C_synthetic_vs_human_distribution_NOT_ACCURACY.csv` |
| Response quality, construct alignment, product relevance | **Automated diagnostics + qualitative examples** | `C_synthetic_scored_with_flags.csv`, `D_*` |
| Aspect RF 73.3% | **DIAGNOSTIC ONLY: target leakage** | `data/aspect_model/metrics.json` |
| Persona accuracy, PE/EE/HM/SI/PP construct accuracy | **CANNOT BE CALCULATED**: no ground truth exists | — |

---

### Output inventory
- `metrics/`: 24 CSV/JSON files plus `ALL_RESULTS.json` (every number in this report) and `synthetic_response_embeddings_mpnet.npy`
- `figures/`: actual vs predicted (A, B), confusion matrices (A, B canonical, B improved), rating distributions (A, B), MAE/RMSE model comparison, error distributions (A, B), k×T sensitivity heatmaps, mean PMF + reliability diagram, example PMFs, persona × construct heatmap + η², variance compression. All are 300-dpi PNG + PDF.
