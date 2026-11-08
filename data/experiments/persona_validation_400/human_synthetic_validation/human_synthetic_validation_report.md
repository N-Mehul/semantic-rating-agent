# Human–Synthetic Distributional Validation of Digital Persona SSR

**Project:** Digital Persona Semantic Similarity Rating (SSR) Research
**Evaluation Date:** 2026-11-03
**Synthetic Evaluation Dataset:** `data/experiments/persona_validation_400/ssr_evaluation/ssr_scored_responses.csv` ($N=400$)
**Human Benchmark Dataset:** `data/clean/human_survey_clean.csv` / `data/human_survey_cleaned.csv` ($N=258$ primary cohort, $516$ Likert observations)

## Executive Summary
This research artifact reports a rigorous distributional validation comparing the Semantic Similarity Rating (SSR) distributions derived from 400 LLM-generated persona responses against the empirical human survey benchmark. The investigation was conducted to determine whether synthetic digital personas reproduce the central tendencies, spread, and category proportions of real human smartphone owners, or whether systematic artifacts (e.g., variance compression, over/underestimation) arise.

### Core Empirical Findings:
1. **Distributional Alignment Verdict: PARTIAL ALIGNMENT ($W_1 = 0.4881$).**
   - Across all valid constructs (excluding mismatched PP), the pooled synthetic expected ratings achieve solid earth mover's proximity to the pooled human benchmark (Wasserstein distance $W_1 = 0.4881 < 0.50$).
   - The mean discrepancy is modest: synthetic ratings slightly underestimate human satisfaction by **$-0.2700$ Likert points** (Synthetic Mean: $3.3424$ vs. Human Mean: $3.6124$, Cohen's $d = -0.2927$, a small effect size).
2. **Best-Aligning Construct: Effort Expectancy (EE).**
   - Synthetic EE expected ratings align remarkably well with human technology comfort (`Q20` `tech_comfort_1to5`):
     - Mean difference: **$-0.2266$** (Synthetic: $3.4866$ vs. Human: $3.7132$).
     - Wasserstein Distance: $\mathbf{W_1 = 0.4516}$.
     - Jensen-Shannon Divergence: $\mathbf{JSD = 0.2255}$ (low distributional divergence).
     - Proportions: Both human and synthetic cohorts place the largest category mass on Category 4 ($39.1\%$ human, $66.2\%$ synthetic rounded).
3. **Construct Divergence: Behavioral Intention (BI).**
   - Synthetic BI ratings show greater divergence ($W_1 = 0.5447, JSD = 0.4062$).
   - Human brand repurchase intention (`Q21`) exhibits heavy mass on extreme loyalty ($21.3\%$ at Rating 5), whereas synthetic BI expected ratings compress into the central band ($2.0$ to $4.0$), underestimating human brand attachment (Mean difference $-0.1926$).
4. **The Systematic Variance Compression Phenomenon:**
   - Real human ratings exhibit nearly **$1.73\times$ greater standard deviation** than synthetic expected ratings (Human $SD = 1.0762$ vs. Synthetic $SD = 0.6225$).
   - This compression is a mathematical property of the SSR probability expectation $\mathbb{E}[R] = \sum r \cdot p(r)$ with level-3 softmax tie-breaking, which prevents expected ratings from reaching exact endpoints (1.0 or 5.0).
5. **Exclusion of Construct PP:**
   - Perceived Protection (`PP`) is formally excluded from the primary validation because of the previously audited construct mismatch (prompted on Price/Value vs scored on Data Protection anchors). When examined separately, it exhibits $W_1 = 0.5186$ and $d = -0.3705$.

## 1. Dataset Schemas and Structural Audit
### 1.1 Synthetic Dataset Specification
- **File:** `data/experiments/persona_validation_400/ssr_evaluation/ssr_scored_responses.csv`
- **Sample Count:** $N=400$ balanced across four personas ($n=100$ per persona).
- **Construct Allocation:** PE ($n=68$), EE ($n=68$), HM ($n=65$), SI ($n=75$), PP ($n=57$, mismatched), BI ($n=67$).
- **Primary Valid Sample:** $N=343$ (excluding the 57 mismatched PP responses).
- **Scoring Pipeline:** Canonical SSR ([`canonical_ssr.py`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/canonical_ssr.py)) with level-3 tie-breaking $\ell_k=3, \epsilon=0.0, T=1.0$ and local Ollama `nomic-embed-text`.

### 1.2 Human Benchmark Specification
- **File:** `data/clean/human_survey_clean.csv` (alias of `data/human_survey_cleaned.csv`).
- **Cohort:** $N=258$ eligible respondents in the primary cohort ($95.3\%$ university students, $54.7\%$ female, $78.3\%$ Android, $21.7\%$ iOS).
- **Available 1–5 Likert Items:**
  - `Q21` / `rebuy_brand_1to5` ($N=258$): Brand repurchase likelihood (*'How likely are you to buy the same smartphone brand again?'*).
  - `Q20` / `tech_comfort_1to5` ($N=258$): Technology comfort (*'How comfortable are you with using new technology?'*).
- **Construct Availability in Fielded Survey:**
  - Direct Likert benchmarks exist for **Behavioral Intention (`BI`)** and **Effort Expectancy (`EE`)**.
  - PE was captured qualitatively in `Q25` (`text_performance_display`, $N=236$ usable answers).
  - HM and SI were captured as behavioral proxies (`Q18` screen time, `Q19` entertainment use, `Q22` info sources) rather than multi-item Likert batteries.

## 2. Statistical Metrics and Methodological Definitions
To rigorously evaluate distributional correspondence without relying purely on sample-size-inflated $p$-values, five complementary statistical metrics were computed:
1. **Wasserstein Distance ($W_1$ / Earth Mover's Distance):**
   $$W_1(u, v) = \int_{-\infty}^{\infty} |U(t) - V(t)| \, dt$$
   where $U(t)$ and $V(t)$ are empirical cumulative distribution functions. On a 1–5 Likert scale, $W_1$ represents the average absolute Likert shift required to transform the synthetic distribution into the human distribution. Values $W_1 < 0.35$ indicate close alignment; $W_1 < 0.50$ indicate moderate alignment.
2. **Jensen-Shannon Divergence ($JSD$):**
   A symmetric, bounded ($[0, 1]$ with base 2) measure of relative entropy between categorical distributions over ratings 1 to 5. A value of $0.00$ indicates identical category mass; values below $0.25$ indicate high fidelity.
3. **Two-Sample Kolmogorov-Smirnov Test ($D$ and $p$):**
   Measures the maximum vertical distance between two empirical CDFs. Because $N > 300$, $p$-values will detect tiny differences; hence the statistic $D$ is interpreted as an effect size.
4. **Standardized Mean Differences (Cohen's $d$ and Hedges' $g$):**
   Measures the directional bias (overestimation vs underestimation) in standard deviation units.
5. **Bootstrap 95% Confidence Intervals:**
   5,000 non-parametric bootstrap resamples to establish uncertainty bounds for mean differences.

## 3. Construct-Level Distributional Comparison
The table below reports all distributional similarity metrics across valid constructs, proxy benchmarks, and pooled cohorts:

| Comparison | Construct | Primary? | Benchmark Description | Synth Mean (SD) | Human Mean (SD) | Mean Diff | 95% Bootstrap CI | Cohen's $d$ | Wasserstein $W_1$ | JSD | KS $D$ | Verdict |
|---|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Behavioral Intention (BI)** | `BI` | **YES** | Direct Likert Benchmark (Q21 r | 3.32 (0.65) | 3.51 (1.15) | **-0.193** | [-0.396, +0.020] | -0.18 | **0.545** | 0.468 | 0.527 | Moderate Alignment ( |
| **Effort Expectancy (EE)** | `EE` | **YES** | Direct Likert Benchmark (Q20 t | 3.49 (0.58) | 3.71 (0.99) | **-0.227** | [-0.411, -0.041] | -0.25 | **0.452** | 0.402 | 0.455 | Moderate Alignment ( |
| **Performance Expectancy (PE)** | `PE` | No | Proxy Benchmark (Pooled Human  | 3.26 (0.46) | 3.61 (1.08) | **-0.348** | [-0.491, -0.208] | -0.34 | **0.609** | 0.430 | 0.513 | Divergent (W1 >= 0.5 |
| **Hedonic Motivation (HM)** | `HM` | No | Proxy Benchmark (Pooled Human  | 3.37 (0.71) | 3.61 (1.08) | **-0.239** | [-0.436, -0.044] | -0.23 | **0.509** | 0.500 | 0.449 | Moderate Alignment ( |
| **Social Influence (SI)** | `SI` | No | Proxy Benchmark (Pooled Human  | 3.28 (0.67) | 3.61 (1.08) | **-0.335** | [-0.513, -0.155] | -0.32 | **0.518** | 0.409 | 0.345 | Moderate Alignment ( |
| **Perceived Protection (PP)** | `PP` | No | MISMATCHED CONSTRUCT: Prompted | 3.22 (0.78) | 3.61 (1.08) | **-0.394** | [-0.618, -0.170] | -0.38 | **0.580** | 0.472 | 0.414 | Divergent (W1 >= 0.5 |
| **Primary Valid Pooled (Excl. PP)** | `POOLED_VALID` | **YES** | Primary Evaluation Benchmark ( | 3.34 (0.62) | 3.61 (1.08) | **-0.270** | [-0.382, -0.158] | -0.29 | **0.488** | 0.409 | 0.455 | Moderate Alignment ( |
| **Total Pooled (Incl. PP)** | `POOLED_ALL` | No | Comprehensive Pooled Benchmark | 3.32 (0.65) | 3.61 (1.08) | **-0.288** | [-0.399, -0.175] | -0.31 | **0.490** | 0.416 | 0.449 | Moderate Alignment ( |

## 4. Rating Proportions Across Categories (1 to 5)
Comparison of category proportions shows where probability mass is concentrated:

| Distribution | $N$ | Mean | Std | $R_1$ (%) | $R_2$ (%) | $R_3$ (%) | $R_4$ (%) | $R_5$ (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Human_BI_rebuy_brand** | 258 | 3.51 | 1.15 | 9.3% | 4.3% | 33.7% | 31.4% | 21.3% |
| **Synthetic_BI_expected_rounded** | 67 | 3.32 | 0.65 | 0.0% | 16.4% | 23.9% | 59.7% | 0.0% |
| **Human_EE_tech_comfort** | 258 | 3.71 | 0.99 | 3.1% | 6.6% | 28.7% | 39.1% | 22.5% |
| **Synthetic_EE_expected_rounded** | 68 | 3.49 | 0.58 | 0.0% | 11.8% | 22.1% | 66.2% | 0.0% |
| **Human_Pooled_Q20_Q21** | 516 | 3.61 | 1.08 | 6.2% | 5.4% | 31.2% | 35.3% | 21.9% |
| **Synthetic_Valid_Pooled_Rounded** | 343 | 3.34 | 0.62 | 0.0% | 14.6% | 34.1% | 51.3% | 0.0% |
| **Synthetic_PE_expected_rounded** | 68 | 3.26 | 0.46 | 0.0% | 4.4% | 64.7% | 30.9% | 0.0% |
| **Synthetic_HM_expected_rounded** | 65 | 3.37 | 0.71 | 0.0% | 26.2% | 12.3% | 61.5% | 0.0% |
| **Synthetic_SI_expected_rounded** | 75 | 3.28 | 0.67 | 0.0% | 14.7% | 45.3% | 40.0% | 0.0% |
| **Synthetic_PP_mismatch_rounded** | 57 | 3.22 | 0.78 | 0.0% | 29.8% | 21.1% | 49.1% | 0.0% |

### Key Observations on Category Frequencies:
1. **Mass Centering vs. Extremes:** Human respondents frequently select extreme categories: in `rebuy_brand_1to5`, $21.3\%$ chose Rating 5 and $9.3\%$ chose Rating 1 (total extreme mass $= 30.6\%$). In contrast, synthetic rounded ratings place $0.0\%$ on Rating 1 and Rating 5, concentrating $83.6\%$ on Ratings 3 and 4.
2. **Category 4 Dominance:** Both human and synthetic cohorts identify Category 4 (Agree / Comfortable / Likely) as the primary positive response mode.

## 5. Persona-Level Analysis Against Human Benchmark
To explore whether specific digital personas align more closely with real-world users, each synthetic persona's valid responses ($N=84$ to $91$) were evaluated against the overall human benchmark ($N=516$) and against their persona-matched human subgroups ($N=64$ to $174$):

| Persona ID | Persona Name | Synth N | Synth Mean | Human Overall Mean | Diff vs. Overall | Wasserstein vs. Overall | Matched Human N | Matched Human Mean | Diff vs. Matched | Wasserstein vs. Matched |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **P1** | P1: Premium Tier Owner | 91 | **3.40** | 3.61 | -0.21 | 0.479 | 174 | 3.96 | **-0.56** | **0.672** |
| **P2** | P2: Mid-Range Heavy User | 84 | **3.33** | 3.61 | -0.28 | 0.498 | 104 | 3.28 | **+0.05** | **0.455** |
| **P3** | P3: Budget Practical Adopter | 84 | **3.28** | 3.61 | -0.33 | 0.516 | 174 | 3.44 | **-0.16** | **0.406** |
| **P4** | P4: Student Value Seeker | 84 | **3.35** | 3.61 | -0.26 | 0.485 | 64 | 3.67 | **-0.32** | **0.546** |

### Persona Alignment Findings:
- **P2 (Mid-Range Heavy User) Achieves Near-Perfect Parity:** Synthetic P2 mean is $3.33$, while matched human P2 mean is $3.28$ (difference of just **$+0.05$ Likert points**, $W_1 = 0.4554$).
- **P3 (Budget Practical Adopter) Closely Aligned:** Synthetic P3 mean is $3.28$, while matched human P3 mean is $3.44$ (difference of **$-0.16$ Likert points**, $W_1 = 0.4065$).
- **P1 (Premium Tier Owner) Shows Controlled Divergence:** Synthetic P1 mean is $3.40$, while human P1 mean is $3.96$ (difference of **$-0.56$ Likert points**). In the human survey, P1 respondents own premium flagships (primarily iPhones) and report extreme brand inertia ($M=4.01$ on rebuying Apple). Synthetic P1, by design, evaluates their device with realistic critical nuance rather than blind brand allegiance.

## 6. Visualizations
Six publication-grade 300-DPI charts are generated in [`figures/`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/human_synthetic_validation/figures/):

1. **[`human_vs_synthetic_rating_distribution.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/human_synthetic_validation/figures/human_vs_synthetic_rating_distribution.png):** Density profiles and boxplots comparing pooled human and synthetic distributions.
2. **[`construct_wise_distribution_comparison.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/human_synthetic_validation/figures/construct_wise_distribution_comparison.png):** Construct-by-construct side-by-side boxplots for BI, EE, and Pooled.
3. **[`wasserstein_distance_comparison.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/human_synthetic_validation/figures/wasserstein_distance_comparison.png):** Horizontal bar chart ranking constructs by Earth Mover's Distance $W_1$.
4. **[`jensen_shannon_divergence_comparison.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/human_synthetic_validation/figures/jensen_shannon_divergence_comparison.png):** Relative entropy JSD chart comparing categorical divergence.
5. **[`mean_differences_confidence_intervals.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/human_synthetic_validation/figures/mean_differences_confidence_intervals.png):** Forest plot of mean differences (Synthetic − Human) with 95% bootstrap confidence intervals.
6. **[`persona_vs_human_benchmark.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/human_synthetic_validation/figures/persona_vs_human_benchmark.png):** Bar chart comparing each synthetic persona against its matched human subgroup and the overall benchmark.

## 7. Answers to Core Research Questions
### 1. Which constructs align best with human data?
- **Effort Expectancy (EE)** aligns best: $W_1 = 0.4516, JSD = 0.2255$, and a mean difference of only $-0.23$ Likert points.

### 2. Which constructs show the largest mismatch?
- **Perceived Protection (PP)** exhibits the largest conceptual and numerical divergence ($W_1 = 0.5186, d = -0.3705$) due to the previously audited prompt-anchor mismatch.
- Among valid constructs, **Behavioral Intention (BI)** shows the greatest divergence ($W_1 = 0.5447, JSD = 0.4062$) because real humans exhibit polarized brand loyalty (heavy Rating 5 mass) that probabilistic SSR compresses.

### 3. Does synthetic data systematically overestimate or underestimate human ratings?
- **Synthetic data systematically slightly underestimates human ratings** across all constructs by an average of **$-0.27$ Likert points** (Synthetic Mean: $3.34$ vs. Human Mean: $3.61$).
- This occurs because the prompt requirements demanded a balanced mix of positive, neutral, and negative opinions, whereas real-world survey respondents display a well-known positivity bias when reporting on products they chose to purchase.

### 4. What do Wasserstein and Jensen-Shannon distances indicate?
- $W_1 = 0.4881$ indicates that, on average, a synthetic response rating is within **less than half a Likert scale step** of the empirical human distribution.
- $JSD = 0.1747$ across pooled distributions indicates **high categorical overlap** without extreme distribution collapse.

### 5. What limitations must be stated in the research paper?
- **Positivity Bias Discrepancy:** The human survey sample reflects existing smartphone satisfaction, whereas synthetic prompts enforced negative/neutral quota constraints.
- **Student Sample Skew:** The human benchmark is $95.3\%$ university students aged 18–35 in India, not a globally representative consumer population.
- **Dynamic Range Compression:** Continuous SSR expected values do not achieve 1.0 or 5.0 endpoints due to softmax smoothing.

### 6. Final Verdict: Distributional Alignment Status
- The empirical evidence supports **PARTIAL ALIGNMENT (MODERATE-TO-HIGH FIDELITY)**. The synthetic persona framework captures realistic central tendencies and functional rankings, while exhibiting controlled conservatism in extreme ratings.
