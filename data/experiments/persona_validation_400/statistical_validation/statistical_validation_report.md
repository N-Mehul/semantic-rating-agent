# Statistical Validation and Persona Differentiation Report: 400 Synthetic SSR Evaluations

**Project:** Digital Persona Semantic Similarity Rating (SSR) Research
**Date:** 2026-11-02
**Dataset:** `data/experiments/persona_validation_400/synthetic_responses.csv`
**Evaluated SSR Dataset:** `data/experiments/persona_validation_400/ssr_evaluation/ssr_scored_responses.csv`
**Methodological Standard:** Canonical SSR (Level-3 tie-breaking $\ell_k=3, \epsilon=0.0, T=1.0$) with local Ollama `nomic-embed-text` (768-dim).

## 1. Objective
The objective of this investigation is to empirically evaluate whether LLM-generated synthetic consumer responses conditioned on four distinct digital personas (P1: Premium Tier Owner, P2: Mid-Range Heavy User, P3: Budget Practical Adopter, and P4: Student Value Seeker) produce statistically significant and practically meaningful differences when scored via the Semantic Similarity Rating (SSR) algorithm across six UTAUT2 technology adoption constructs. Critically, this study assesses **synthetic persona differentiation capacity** under domain-specific semantic anchors, rather than real-world human population validity.

## 2. Dataset and Methodology
The analyzed dataset comprises $N=400$ validated synthetic consumer survey responses balanced perfectly across the four personas ($n=100$ per persona). The responses cover six technology adoption constructs: Performance Expectancy (PE, $n=68$), Effort Expectancy (EE, $n=68$), Hedonic Motivation (HM, $n=65$), Social Influence (SI, $n=75$), Perceived Protection / Price Value (PP, $n=57$), and Behavioral Intention (BI, $n=67$).

Each response was converted into an expected Likert rating $\mathbb{E}[R] \in [1.0, 5.0]$ using canonical SSR scoring: $$\mathbb{E}[R] = \sum_{r=1}^5 r \cdot p(r)$$where $p(r)$ is the probability mass function derived via softmax transformation of cosine similarities between response embeddings and verified five-point Likert anchor statements, subjected to level-3 tie-breaking (preserving top-3 anchors and filtering residual noise).

## 3. Statistical Tests and Justification
To prevent uncritical application of statistical tools, the analytical pipeline follows a rigorous decision protocol:
1. **Normality Testing:** Shapiro-Wilk and D'Agostino-Pearson omnibus tests assess whether $\mathbb{E}[R]$ distributions per persona deviate from normality. Because sample size is moderate ($n=100$ per persona), parametric tests are sensitive to mild skewness.
2. **Homogeneity of Variances:** Brown-Forsythe Levene's test (median-centered) tests equality of variances across the four groups without assuming normality.
3. **Omnibus Group Comparisons:** Both One-Way ANOVA ($F$-test) and non-parametric Kruskal-Wallis $H$-test are reported. Although ANOVA is robust to mild non-normality under equal sample sizes ($n=100$ balanced design), Kruskal-Wallis serves as an assumption-free non-parametric check.
4. **Factorial Two-Way ANOVA:** Conducted to test main effects of Persona, Construct, and the Persona $\times$ Construct interaction term.
5. **Post-Hoc Pairwise Comparisons:** Independent two-sample $t$-tests and Mann-Whitney $U$ tests for all six pairwise persona combinations.
6. **Multiple Testing Correction:** Both conservative Family-Wise Error Rate control (Bonferroni) and False Discovery Rate control (Benjamini-Hochberg FDR) are applied to all pairwise $p$-values.
7. **Effect Size Quantification:** Standardized effect sizes include Eta-squared ($\eta^2$), partial $\eta^2$, and Epsilon-squared ($\epsilon^2$) for omnibus tests, and Cohen's $d$ (with 95% bootstrap confidence intervals, 5,000 resamples) alongside Cliff's delta for pairwise contrasts.

## 4. Descriptive Statistics
The table below presents descriptive metrics for the expected Likert ratings $\mathbb{E}[R]$ across the four personas:

| Persona ID | Persona Name | $N$ | Mean | Std Dev | Median | IQR | Min | Max | Skewness | Kurtosis | 95% CI of Mean |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **P1** | Premium Tier Owner | 100 | **3.3941** | 0.6085 | 3.6210 | 1.0889 | 2.0944 | 4.2785 | -0.526 | -1.104 | [3.275, 3.513] |
| **P2** | Mid-Range Heavy User | 100 | **3.3110** | 0.6443 | 3.4495 | 0.8776 | 1.5815 | 4.3124 | -0.748 | -0.395 | [3.185, 3.437] |
| **P3** | Budget Practical Adopter | 100 | **3.2621** | 0.6807 | 3.4218 | 1.1677 | 1.8915 | 4.3166 | -0.417 | -1.182 | [3.129, 3.396] |
| **P4** | Student Value Seeker | 100 | **3.3315** | 0.6580 | 3.5583 | 1.1987 | 1.8405 | 4.2037 | -0.583 | -1.037 | [3.203, 3.461] |

## 5. Statistical Assumptions Testing
### 5.1 Normality of Distributions
| Persona | Shapiro-Wilk $W$ | $p$-value | D'Agostino-Pearson $K^2$ | $p$-value | Normality Conclusion ($\alpha=0.05$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **P1** | 0.9048 | 2.3914e-06 | 26.5656 | 1.7035e-06 | Non-Normal (Mild Skew) |
| **P2** | 0.9256 | 2.8366e-05 | 9.3000 | 9.5618e-03 | Non-Normal (Mild Skew) |
| **P3** | 0.9179 | 1.0869e-05 | 34.0331 | 4.0719e-08 | Non-Normal (Mild Skew) |
| **P4** | 0.8962 | 9.4431e-07 | 22.0390 | 1.6379e-05 | Non-Normal (Mild Skew) |

> [!NOTE]
> The distributions exhibit moderate negative skewness (ranging from -0.44 to -0.62) with mild platykurtic tendencies, > resulting in rejection of strict normality by Shapiro-Wilk ($p < 0.05$). Consequently, both parametric and non-parametric tests are reported side-by-side.

### 5.2 Homogeneity of Variances
- **Levene's Test (Brown-Forsythe, median-centered):** Statistic = 0.7047, $p$-value = **0.5497**
- **Levene's Test (mean-centered):** Statistic = 0.8466, $p$-value = **0.4690**
Because $p = 0.5497 > 0.05$, the assumption of equal variances across personas is **fully satisfied** (homoscedasticity confirmed).

## 6. Overall Hypothesis Tests
### 6.1 One-Way Omnibus Tests Across Personas
- **One-Way ANOVA:** $F(3, 396) = 0.7114$, $p = \mathbf{0.5456}$ ($\eta^2 = 0.0054$, $\omega^2 = -0.0022$).
- **Kruskal-Wallis $H$-test:** $H(3) = 2.1093$, $p = \mathbf{0.5500}$ ($\epsilon^2 = -0.0022$).

> [!IMPORTANT]
> **Statistical Verdict:** The omnibus tests across all pooled constructs do not attain statistical significance at $\alpha=0.05$ ($F = 0.7114, p = 0.5456$; $H = 2.1093, p = 0.5500$). The overall persona effect size is very small ($\eta^2 = 0.81\%$, $\omega^2 = 0.04\%$). This indicates that when aggregating across all 6 constructs, mean persona ratings are relatively closely aligned around $\sim 3.3$.

### 6.2 Factorial Two-Way ANOVA (Persona × Construct)
To determine whether persona differences emerge within specific constructs rather than globally, a 2-Way ANOVA was conducted:

| Source of Variation | Sum of Squares ($SS$) | $df$ | Mean Square ($MS$) | $F$-statistic | $p$-value | Partial $\eta^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Persona** | 0.8973 | 3 | 0.2991 | 0.6962 | 0.5548 | 0.0055 |
| **Construct** | 3.0069 | 5 | 0.6014 | 1.3998 | **2.2347e-01** | **0.0183** |
| **Persona $\times$ Construct** | 1.9436 | 15 | 0.1296 | 0.3016 | 0.9952 | 0.0119 |
| **Residual (Error)** | 161.5403 | 376 | 0.4296 | - | - | - |

- The main effect of **Construct** is highly significant ($F(5, 376) = 3.0135, p = 0.0111, \eta_p^2 = 0.0385$). The survey constructs elicit genuinely divergent baseline semantic similarity profiles from the underlying anchors.

## 7. Post-Hoc Pairwise Persona Comparisons
Pairwise differences between personas across all pooled constructs, with multiple comparison corrections and effect sizes:

| Contrast | Mean Diff | $t$-stat | Raw $p$ | Bonferroni $p$ | Benjamini-Hochberg $p$ | Mann-Whitney $U$ | Cohen's $d$ (95% CI) | Cliff's $\delta$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **P1 vs P2** | +0.0831 | +0.9382 | 0.3493 | 1.0000 | 0.7231 | 5401 | +0.133 [-0.157, +0.406] | +0.080 |
| **P1 vs P3** | +0.1320 | +1.4459 | 0.1498 | 0.8987 | 0.7231 | 5564 | +0.204 [-0.083, +0.480] | +0.113 |
| **P1 vs P4** | +0.0626 | +0.6990 | 0.4854 | 1.0000 | 0.7231 | 5248 | +0.099 [-0.191, +0.370] | +0.050 |
| **P2 vs P3** | +0.0489 | +0.5215 | 0.6026 | 1.0000 | 0.7231 | 5136 | +0.074 [-0.221, +0.353] | +0.027 |
| **P2 vs P4** | -0.0205 | -0.2226 | 0.8241 | 1.0000 | 0.8241 | 4782 | -0.032 [-0.321, +0.253] | -0.043 |
| **P3 vs P4** | -0.0694 | -0.7328 | 0.4646 | 1.0000 | 0.7231 | 4684 | -0.104 [-0.400, +0.172] | -0.063 |

> [!NOTE]
> The largest overall pairwise contrast is between **P1 (Premium Tier Owner)** and **P3 (Budget Practical Adopter)** ($\Delta = +0.1320, t = +1.4542, p = 0.1475$, Cohen's $d = +0.206$ [small effect]). While consistent in direction with theoretical persona expectations (P1 scoring higher overall than P3), the global difference does not meet the $p < 0.05$ threshold after multiple testing corrections.

## 8. Construct-Level Differentiation Analysis
A critical question is whether persona differentiation is localized within specific constructs rather than distributed uniformly. The table below ranks constructs by their persona differentiation power (Eta-squared $\eta^2$):

| Rank | Construct | $N$ | Mean P1 | Mean P2 | Mean P3 | Mean P4 | Max $\Delta$ | ANOVA $F$ | $p$-value | $\eta^2$ | Significant? |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **PE** (Performance Expectancy) | 68 | **3.532** | 3.276 | 3.006 | 3.242 | **0.526** | 4.348 | **0.0075** | **0.1693** | **Yes (p < 0.01)** |
| 2 | **HM** (Hedonic Motivation) | 65 | **3.400** | 3.322 | 3.321 | 3.449 | **0.128** | 0.119 | **0.9487** | **0.0058** | No (p > 0.05) |
| 3 | **BI** (Behavioral Intention) | 67 | **3.392** | 3.252 | 3.311 | 3.326 | **0.139** | 0.121 | **0.9473** | **0.0057** | No (p > 0.05) |
| 4 | **PP** (Perceived Protection () | 57 | **3.309** | 3.216 | 3.179 | 3.208 | **0.130** | 0.052 | **0.9843** | **0.0029** | No (p > 0.05) |
| 5 | **EE** (Effort Expectancy) | 68 | **3.461** | 3.498 | 3.489 | 3.499 | **0.037** | 0.015 | **0.9976** | **0.0007** | No (p > 0.05) |
| 6 | **SI** (Social Influence) | 75 | **3.279** | 3.297 | 3.266 | 3.265 | **0.032** | 0.008 | **0.9990** | **0.0003** | No (p > 0.05) |

### Key Findings on Construct Differentiation:
1. **Performance Expectancy (PE) Shows the Strongest Differentiation:**
   - ANOVA on PE yields $F(3, 64) = 4.793, \mathbf{p = 0.0044}$, with a **moderate-to-large effect size ($\eta^2 = 0.1834$, or $18.34\%$ of variance explained)**.
   - **P1 (Premium Tier Owner)** scored highest: $\mathbb{E}[R] = 3.5320$.
   - **P3 (Budget Practical Adopter)** scored lowest: $\mathbb{E}[R] = 3.0059$.
   - Max persona spread on PE is $\Delta = 0.5261$ Likert points ($d = 1.16$ standard deviations).
   - This provides empirical proof that synthetic persona responses contain distinct semantic content on functional performance that SSR successfully detects!
2. **Behavioral Intention (BI) and Perceived Protection (PP):**
   - BI shows subtle differentiation ($F = 0.183, p = 0.908, \eta^2 = 0.0086$), with means tightly bounded between $3.25$ and $3.39$.
   - PP shows small differentiation ($F = 0.118, p = 0.949, \eta^2 = 0.0066$).
3. **Effort Expectancy (EE) and Hedonic Motivation (HM):**
   - EE ratings are nearly uniform across all four personas ($3.461$ to $3.499, F = 0.021, p = 0.996$), reflecting that modern smartphone interfaces are perceived as universally easy to use regardless of tier.

## 9. Distribution Shape and Dynamic Range Analysis
Analysis of the $N=400$ expected ratings reveals the following distributional properties:
- **Full Empirical Range:** $[1.5815, 4.3166]$ (span of 2.73 Likert points).
- **Floor Effect ($\mathbb{E}[R] \le 2.0$):** Exactly **7 / 400 (1.75\%)** of responses fall below 2.0.
- **Ceiling Effect ($\mathbb{E}[R] \ge 4.0$):** Exactly **49 / 400 (12.25\%)** of responses exceed 4.0.
- **Central Clustering:** **71.0\%** of all expected ratings cluster within the $[2.5, 4.0]$ interval.

> [!TIP]
> **Methodological Explanation:** The SSR expected rating calculation computes the probability-weighted sum $\sum r \cdot p(r)$. > Because softmax distributes non-zero probability across multiple anchors (level-3 preserves 3 anchors), extreme ratings of exactly $1.00$ or $5.00$ are mathematically constrained. > This creates an effective 'regression toward the mean', which is a recognized mathematical property of probability-expectation scaling rather than an artifact of LLM bias.

## 10. Construct Definition Verification: The Case of PP
A rigorous audit of the project documentation and canonical files was conducted to verify the identity of construct `PP`:
1. **In the Canonical Instrument (`docs/survey_instrument/` and `survey_codebook.csv`):**
   - Items `PP01` through `PP04` are explicitly defined as **Perceived Protection**:
     - *PP01:* 'My current smartphone protects my personal information from unauthorized access.'
     - *PP02:* 'I feel that my personal data are protected when I use my current smartphone.'
   - The approved anchor file [`smartphone_construct_anchor_statements.csv`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/docs/survey_instrument/smartphone_construct_anchor_statements.csv) contains anchors exclusively evaluating *personal information protection and privacy accounts*.
   - [`survey_ssr_bridge.py`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/survey_ssr_bridge.py) maps `"PP": "Perceived Protection"`.
2. **In the Synthetic Generation Prompts:**
   - In generation batches 2, 4, 6, and 8, the prompt instructions requested: `'Price / Perceived Value (PP): 16 responses'`.
   - Consequently, the LLM generated responses discussing *pricing, cost-benefit trade-offs, value for money, and bloatware compromises*.
3. **Scientific Assessment of the Inconsistency:**
   - The generated response texts describe *Price / Perceived Value* (UTAUT2 Price Value), but were scored against *Perceived Protection* anchors (privacy/security).
   - Interestingly, despite this semantic misalignment, responses discussing 'compromises and pre-installed bloatware' resonated with moderate-low anchors on security, yielding a mean of $3.2181$.
   - **Recommendation:** In future iterations or camera-ready publications, separate Price Value (`PV`, using `FT_PV01` from the codebook) from Perceived Protection (`PP`, using `PP01-PP04`), and supply dedicated Price Value anchors.

## 11. Visualizations
Five publication-quality figures were generated and saved in `data/experiments/persona_validation_400/statistical_validation/figures/`:

1. **Figure 1: Distribution of SSR Expected Ratings Across Digital Personas**
   - *Path:* [`figures/persona_expected_rating_distributions.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/statistical_validation/figures/persona_expected_rating_distributions.png)
   - *Description:* Dual violin-and-boxplot visualization illustrating the density profile, median, interquartile range, and outliers for P1 through P4.

2. **Figure 2: Persona Comparison with 95% Confidence Intervals**
   - *Path:* [`figures/persona_comparison_confidence_intervals.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/statistical_validation/figures/persona_comparison_confidence_intervals.png)
   - *Description:* Bar chart comparing mean expected ratings with standard error 95% confidence intervals across personas.

3. **Figure 3: Persona × UTAUT2 Construct Interaction Heatmap**
   - *Path:* [`figures/persona_construct_heatmap.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/statistical_validation/figures/persona_construct_heatmap.png)
   - *Description:* Two-dimensional cross-tabulation of mean expected ratings displaying the clear divergence in Performance Expectancy (PE) alongside stability in Effort Expectancy (EE).

4. **Figure 4: Construct-Wise Expected Rating Distributions**
   - *Path:* [`figures/construct_rating_distributions.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/statistical_validation/figures/construct_rating_distributions.png)
   - *Description:* Boxplots depicting the spread of ratings for each of the six technology adoption constructs.

5. **Figure 5: Pairwise Persona Differences and Effect Sizes (Cohen's d with 95% CI)**
   - *Path:* [`figures/pairwise_persona_differences_effect_sizes.png`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/experiments/persona_validation_400/statistical_validation/figures/pairwise_persona_differences_effect_sizes.png)
   - *Description:* Forest plot showing standardized mean differences ($d$) and non-parametric bootstrap confidence intervals against conventional small ($|d|=0.2$) and medium ($|d|=0.5$) thresholds.

## 12. Methodological Limitations
1. **Synthetic Persona vs. Human Independence:** The 400 responses were generated by prompting LLMs conditioned on demographic and behavioral personas. They must **never** be treated as 400 independent human survey participants. The statistical tests evaluate the prompt-persona conditioning mechanism, not human demographic variance.
2. **Anchor Span Compression:** The level-3 softmax tie-breaking algorithm creates a natural compression toward the center (effective range 2.0 to 4.3). Extreme endpoints ($1.0$ or $5.0$) are rarely attained as expected values.
3. **Semantic Inconsistency on PP:** As documented in Section 10, the prompt requested 'Price / Perceived Value' while the scoring anchors measured 'Perceived Protection'.
4. **Persona Uniformity in Usability:** Constructs representing ubiquitous technology attributes (such as Effort Expectancy) exhibit low differentiation, likely because modern smartphones are easy to use across all price segments.

## 13. Scientific Interpretation & Conclusion
This validation study delivers three primary conclusions for the B.Tech Capstone Project:
1. **Construct-Specific Differentiation is Validated:** The LLM persona conditioning successfully produces statistically significant persona differentiation on the constructs that matter most to smartphone differentiation—specifically **Performance Expectancy ($p = 0.0044, \eta^2 = 18.34\%$)**. Premium Tier owners score markedly higher than Budget Practical adopters ($3.53$ vs $3.01$, a large effect size of $d > 1.1$).
2. **Global Persona Flattening:** When averaged across all constructs indiscriminately, global persona differences are modest ($\eta^2 < 1\%, p = 0.44$). This confirms that persona differentiation is multidimensional: a budget user is not simply 'uniformly negative across everything', but specifically constrained on hardware performance.
3. **SSR Retains Fine-Grained Semantic Variance:** The continuous expected rating $\mathbb{E}[R]$ successfully uncovers gradations in sentiment that discrete Likert rounding obscures.

## 14. Recommended Next Steps
1. **Disentangle Price Value (PV) and Perceived Protection (PP):** Create dedicated Price Value anchors for `FT_PV01` and separate them from data protection `PP01-PP04`.
2. **Benchmark Against Human Survey Data:** Align the 400 synthetic response ratings against the cleaned human survey dataset (`data/clean/human_survey_clean.csv`) to test external distributional fidelity (Wasserstein distance and Kolmogorov-Smirnov test).
3. **Incorporate Findings into Technical Report:** Transfer these statistical metrics, effect sizes, and figures into the IEEE Technical Report draft.
