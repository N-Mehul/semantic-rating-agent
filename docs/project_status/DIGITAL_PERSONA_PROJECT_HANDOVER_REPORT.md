# Digital Persona Project Handover Report

**Capstone Research Project Title:**  
*Digital Persona-Based Synthetic Survey Data Generation for Technology Adoption Research: An SSR-Based Validation Framework*

**Repository:** `semantic_rating_agent`  
**Audit & Handover Date:** 2026-10-28  
**Scope:** Complete repository codebase, datasets, models, SSR implementations, test suites, survey instruments, and experimental outputs.

---

## 1. Executive Summary & Foundational Orientation

The overarching objective of this capstone research is to develop and empirically validate an end-to-end framework for **generating and evaluating synthetic survey responses in technology adoption research**.

Traditional LLM survey generation prompts an LLM to generate direct numerical scores (e.g., *"Rate this statement from 1 to 5"*), which introduces well-documented response biases, scale anchors misalignment, and artificial mode clustering. To resolve this, this project investigates:
1. **Digital Persona Synthesis:** Converting demographic and psychographic respondent characteristics into structured persona prompts.
2. **Natural-Language Construct Elicitation:** Directing LLM agents to generate free-text opinion responses rather than direct Likert ratings.
3. **Semantic Similarity Rating (SSR):** Converting natural-language responses into calibrated probability distributions and expected Likert ratings using semantic cosine similarity against pre-calibrated anchor statements (following Maier et al., *arXiv:2510.08338*).
4. **Empirical Psychometric Validation:** Validating whether synthetic survey data generated via SSR reproduces true human survey response distributions, factor loadings, and structural path relationships (e.g., UTAUT2 models).

### Status Summary (Implemented vs. Planned)

```
[Target Framework]
Human Demographics -> [Digital Persona Generator] -> [LLM Open-Ended Generation] -> [Canonical SSR Scorer] -> [Validation vs Human Likert]
Status:                    (PLANNED / STAGE 2)            (PROTOTYPED / STAGE 3-4)        (IMPLEMENTED / STAGE 5)       (PLANNED / STAGE 6)

[Repository Actual Operations]
Product Review CSV -> [Text / Aspect Features] -> [Random Forest / Top-k Embedding] -> [Review Star Prediction]
Status:                    (IMPLEMENTED - PRODUCT REVIEW CLASSIFICATION DOMAIN)
```

The current repository reflects two parallel tracks:
1. **A production/experimental mobile review rating pipeline:** Uses sentence embeddings (`all-MiniLM-L6-v2`) and tabular aspect ratings to classify product review ratings (1 to 5 stars).
2. **A verified research track for Canonical SSR and Survey Instruments:** Contains isolated, mathematically verified implementations of the Maier et al. SSR algorithms, a 180-statement construct anchor bank, and psychometric survey instrument specifications for technology adoption (UTAUT2).

---

## 2. Project Architecture & Component Inventory

| Component / Layer | Primary Files | Status | Description & Actual Behavior |
| :--- | :--- | :--- | :--- |
| **Canonical SSR Engine** | [`canonical_ssr.py`](../../canonical_ssr.py) | **Implemented & Verified** | Isolated implementation of Maier et al. 6-anchor-set response-to-PMF calculation. Fully unit tested. |
| **Aspect Random Forest (D1)** | [`aspect_rating_model.py`](../../aspect_rating_model.py), [`data/aspect_model/`](../../data/aspect_model/) | **Implemented (Diagnostic)** | Trained on 5 aspect ratings (`battery_life_rating`, `camera_rating`, etc.) to predict overall rating. Flagged for Target Leakage. |
| **Review Rating Agent** | [`agent.py`](../../agent.py), [`main.py`](../../main.py) | **Implemented (Operational)** | Mobile review analyzer with nearest-neighbor search, memory management, and direct LLM rater integration. |
| **Prompt Probe Experiments** | [`persona_nl_controlled_probe.py`](../../persona_nl_controlled_probe.py) | **Experimental Prototype** | Controlled probe testing persona-conditioned text generation across smartphone review scenarios. |
| **Survey Instrument** | [`docs/survey_instrument/`](../../docs/survey_instrument/) | **Drafted Specification** | UTAUT2-based smartphone survey: 6 constructs, Likert items, free-text prompt items, and codebook. |
| **Anchor Statement Bank** | [`docs/survey_instrument/smartphone_construct_anchor_statements.csv`](../../docs/survey_instrument/smartphone_construct_anchor_statements.csv) | **Implemented (Domain Adapted)** | 180 curated semantic anchor statements (6 constructs $\times$ 6 anchor sets $\times$ 5 Likert levels). |
| **Unit & Regression Tests** | [`tests/test_canonical_ssr.py`](../../tests/test_canonical_ssr.py), [`tests/test_analysis.py`](../../tests/test_analysis.py) | **Implemented & Passing** | 167 automated test cases verifying mathematical validity, probability constraints, and regression stability. |
| **Digital Persona Engine** | *None* | **Not Implemented** | No automated pipeline converting survey demographics into structured, reproducible persona records. |
| **Synthetic Survey Orchestrator**| *None* | **Not Implemented** | No end-to-end batch script that pairs personas with constructs, invokes LLM, and passes text into SSR. |
| **Human Validation Data** | *None* | **Not Implemented** | No real human survey dataset collected or stored under `data/` for empirical benchmark comparison. |

---

## 3. Dataset Audit & Reality Check

### 3.1 Primary Repository Dataset: `data/Mobile Reviews Sentiment.csv`
- **Total Rows:** 50,000
- **Total Columns:** 25
- **Missing / Null Values:** 250,051 across all fields.
- **Duplicate Rows:** 10,001 completely duplicated rows.
- **Target Distribution (Overall Rating):**
  - Rating 1.0: 5,057 (10.1%)
  - Rating 2.0: 7,814 (15.6%)
  - Rating 3.0: 10,037 (20.1%)
  - Rating 4.0: 11,245 (22.5%)
  - Rating 5.0: 5,845 (11.7%)
  - Missing Target (`NaN`): 10,002 rows (20.0%)

### 3.2 Critical Data Anomaly: Text Degeneracy and 100% Label Conflict
A rigorous programmatic verification of the `review_text` column reveals:
1. **Extreme Text Collapse:** Across all 50,000 rows, there are **only 110 unique review text strings**. Each text string is repeated hundreds of times across different rows.
2. **Verification of the 100% Label Conflict Finding:**
   - When grouping by `review_text` and counting unique assigned `rating` values:
     $$\text{Distinct ratings per text} = |\text{Unique}(R \mid \text{text})|$$
   - Exactly **110 out of 110 unique review texts (100.0%)** have more than one distinct rating assigned to them across the dataset.
   - Distribution of conflict severity:
     - 7 unique texts are paired with 3 conflicting rating levels.
     - 58 unique texts are paired with 4 conflicting rating levels.
     - 45 unique texts are paired with all 5 conflicting rating levels (1, 2, 3, 4, and 5).
3. **Implication:** The text column in this dataset cannot support accurate text-only rating prediction, as identical text strings are simultaneously labeled with opposing ground truths. This dataset is a synthetic product-review artifact, **not** an empirical human survey dataset.

---

## 4. SSR Methodology: Verification, Equations & Limitations

### 4.1 Canonical Mathematical Formulation (`canonical_ssr.py`)
Following Maier et al. (*arXiv:2510.08338v3*), SSR converts natural language into ordinal probabilities:
1. **Anchor Sets:** Define $K = 6$ independent anchor sets. Each set $k$ contains 5 ordered statements $A_{k, 1}, \dots, A_{k, 5}$ representing Likert levels 1 to 5.
2. **Cosine Similarities:** Compute embeddings for response text $x$ and anchors $A_{k, r}$, yielding raw cosine similarities:
   $$s_{k, r} = \cos(e_x, e_{A_{k, r}})$$
3. **Minimum Subtraction & Epsilon Shift:** Identify set minimum $\ell_k = \arg\min_r s_{k, r}$:
   $$w_{k, r} = s_{k, r} - s_{k, \ell_k} + \epsilon \cdot \mathbf{1}[r = \ell_k]$$
   *(In Maier et al., $\epsilon = 0.0$)*
4. **Probability Mass Function (PMF) with Temperature Scaling:**
   $$p_k(r, T) = \frac{\left( \frac{w_{k, r}}{\sum_{m=1}^5 w_{k, m}} \right)^{1/T}}{\sum_{j=1}^5 \left( \frac{w_{k, j}}{\sum_{m=1}^5 w_{k, m}} \right)^{1/T}}$$
   *(In Maier et al., $T = 1.0$)*
5. **Ensemble Aggregation & Expected Score:**
   $$P(R = r) = \frac{1}{6} \sum_{k=1}^6 p_k(r, T), \qquad \mathbb{E}[R] = \sum_{r=1}^5 r \cdot P(R = r)$$

### 4.2 Status of Original Literature Reproduction
- **Mathematical Logic:** Fully verified and implemented in [`canonical_ssr.py`](../../canonical_ssr.py).
- **Original Paper Anchors:** **Unverified / Absent**. As documented in [`docs/ssr_verification/original_anchor_source_audit.md`](../../docs/ssr_verification/original_anchor_source_audit.md), the authors of arXiv:2510.08338v3 describe the intent gradient across six sets in Appendix C.1 but **do not publish the verbatim text of their six original purchase-intent anchor sets** in the paper, TeX source, or their public repository (`pymc-labs/semantic-similarity-rating`).
- **Domain Adaptation:** The 180 statements in [`smartphone_construct_anchor_statements.csv`](../../docs/survey_instrument/smartphone_construct_anchor_statements.csv) represent a domain adaptation specifically developed for technology adoption constructs (UTAUT2), not the undisclosed purchase-intent statements from the original study.

---

## 5. Model Implementations & Verified Performance

### 5.1 Aspect Random Forest (D1 Baseline — Diagnostic Leakage)
Saved in [`data/aspect_model/metrics.json`](../../data/aspect_model/metrics.json):
- **Accuracy:** 73.34%
- **Macro F1:** 73.70%
- **Weighted F1:** 73.32%
- **MAE:** 0.2680 | **RMSE:** 0.5204
- **Spearman Rank Correlation:** 0.9166
- **Class-Level F1:** Class 1: 76.59%, Class 2: 66.96%, Class 3: 70.84%, Class 4: 75.70%, Class 5: 78.44%
- **CRITICAL LEAKAGE WARNING:** The model explicitly inputs five aspect sub-ratings (`battery_life_rating`, `camera_rating`, `performance_rating`, `design_rating`, `display_rating`) to predict overall rating. Because aspect sub-ratings directly determine overall sentiment, this performance reflects tabular target leakage and does **not** reflect text-based rating extraction or persona fidelity.

### 5.2 Text-Only Baseline Models
- **Sentence-BERT (`all-MiniLM-L6-v2`) Example Bank:** Achieves ~32–35% test accuracy. The performance ceiling is caused directly by the 100% label conflict in the 110 repeated texts.
- **Direct LLM Rating (`agent.py`):** Direct prompting yields ~38–42% accuracy due to severe digit distribution skew (LLMs disproportionately output 4s and 5s).

---

## 6. Test Suite Verification & Audit

All unit tests were executed and verified on Python 3.13.2 with `pytest-9.1.1`:

```powershell
python -m pytest --confcutdir=tests tests/test_canonical_ssr.py tests/test_analysis.py
```
**Result: 167 passed in 6.09 seconds.**

### Detailed Test Breakdown:
1. **`tests/test_canonical_ssr.py` (42 tests — All Passing):**
   - Validates mathematical bounds of cosine similarity.
   - Validates minimum-subtraction normalization and zero-denominator handling when $\epsilon = 0.0$.
   - Validates that output probabilities strictly form valid PMFs ($\sum p_i = 1$, $p_i \in [0, 1]$ within $10^{-12}$).
   - Validates symmetry, temperature scaling invariance, and multi-set expectation values.
2. **`tests/test_analysis.py` (125 tests — All Passing):**
   - Validates data integrity checks, confusion matrix calculations, metric summaries, and feature parsing routines.
3. **Caveat on Full-Repo Pytest:**
   - Running bare `pytest` across the entire root directory triggers unconfigured exploratory scripts in `_archive/scratch/` (such as `test_groq_single.py`), which halt due to missing API keys. Running tests scoped to `tests/` succeeds completely (167/167).

---

## 7. Survey Instrument & Technology Adoption Constructs

Located in [`docs/survey_instrument/`](../../docs/survey_instrument/):
- **Theoretical Grounding:** Adapted from the Unified Theory of Acceptance and Use of Technology 2 (UTAUT2; Venkatesh et al., 2012) for consumer smartphone adoption.
- **Target Constructs:**
  1. *Performance Expectancy (PE):* 4 Likert items + Open-Ended Prompt (`FT_PE01`).
  2. *Effort Expectancy (EE):* 4 Likert items.
  3. *Hedonic Motivation (HM):* 4 Likert items.
  4. *Social Influence (SI):* 4 Likert items.
  5. *Perceived Protection (PP):* 4 Likert items (working interpretation: protection of personal data and accounts; still to be confirmed with faculty). An earlier version of this document listed *Facilitating Conditions (FC)* here; the codebook and anchor file contain PP, not FC.
  6. *Behavioral Intention (BI):* 4 Likert items + Open-Ended Prompt (`FT_BI01`).
  7. *Price Value (PV):* Open-Ended Prompt (`FT_PV01`).
- **Anchor Statement Bank (`smartphone_construct_anchor_statements.csv`):**
  - Exactly 180 statements formatted across 6 constructs $\times$ 6 independent anchor sets $\times$ 5 Likert levels.

---

## 8. Limitations & Methodological Integrity

1. **No Human Validation Conducted Yet:** The repository contains no real human responses to the technology adoption survey. Any claim that synthetic personas have been validated against human ground truth is unsubstantiated by current repository evidence.
2. **Dataset Domain Mismatch:** The current dataset is e-commerce product reviews, not psychometric technology adoption survey responses.
3. **Persona Generator Gap:** The repository has prompt templates (`persona_nl_controlled_probe.py`), but lacks an automated demographic-to-persona compilation pipeline.
4. **Anchor Text Provenance:** Original Maier et al. purchase intent anchor statements are unpublished; the project utilizes an independently authored technology adoption anchor set.

---

## 9. Comprehensive Status Table & Prioritized Next Steps

### 9.1 Component Status Summary

| Phase | Component | Implementation Status | Evidence / Location |
| :---: | :--- | :---: | :--- |
| **Phase 1** | Survey Instrument Design | **Complete** | [`docs/survey_instrument/survey_instrument.md`](../../docs/survey_instrument/survey_instrument.md) |
| **Phase 1** | Construct Semantic Anchors | **Complete** | [`docs/survey_instrument/smartphone_construct_anchor_statements.csv`](../../docs/survey_instrument/smartphone_construct_anchor_statements.csv) |
| **Phase 1** | Canonical SSR Algorithm | **Complete & Verified** | [`canonical_ssr.py`](../../canonical_ssr.py) (167 passing tests) |
| **Phase 2** | Human Benchmark Survey Data | **Pending Collection** | Requires Google Forms / Prolific distribution |
| **Phase 2** | Digital Persona Generator | **Planned** | Demographics $\to$ JSON Persona converter needed |
| **Phase 3** | Synthetic Survey Text Generator | **Planned** | Persona + Construct $\to$ LLM free-text pipeline |
| **Phase 3** | SSR Scoring Integration | **Ready for Integration** | Wire synthetic text into `canonical_ssr.py` |
| **Phase 4** | Psychometric Validation | **Planned** | KS-test, Wasserstein distance, SEM factor invariance |

### 9.2 Prioritized Roadmap for Successor

```
[Step 1: Human Baseline]  Collect 100-200 human responses using docs/survey_instrument/
           │
           ▼
[Step 2: Persona Module]  Implement scripts/generate_personas.py to mirror human respondents
           │
           ▼
[Step 3: LLM Generation]  Query LLM with personas to generate open-ended answers to FT_PE01 & FT_BI01
           │
           ▼
[Step 4: SSR Scoring]     Score generated texts against smartphone_construct_anchor_statements.csv
           │
           ▼
[Step 5: Validation]      Statistically compare Human Likert vs Synthetic SSR Likert distributions
```
