# Digital Persona-Based Synthetic Survey Data Generation for Technology Adoption Research: An SSR-Based Validation Framework
## Executive Research & Technical Summary

**Lead Researcher:** Mehul  
**Institution:** Department of Computer Engineering, SVKM's NMIMS MPSTME  
**Project Repository:** `semantic_rating_agent` (Primary Source of Truth)  
**Execution Environment:** Python 3.13.2 | Windows NT 10.0 | Local Ollama Runtime (Qwen 2.5 7B Instruct & `nomic-embed-text`)  
**Primary Research Benchmark:** 79.17% (19/24) Qwen 2.5 7B Blind-Judge Agreement  
**Target IEEE Venues:** *IEEE Transactions on Computational Social Systems* / *IEEE Access*  

---

## 1. Executive Problem Statement & Motivation

Traditional pre-launch technology adoption research relies heavily on human focus groups and cross-sectional Likert surveys operationalized under psychometric models such as the Technology Acceptance Model (TAM) and the Unified Theory of Acceptance and Use of Technology (UTAUT/UTAUT2). However, manual consumer surveying suffers from severe structural bottlenecks:
1. **Prohibitive Financial Costs:** Commercial consumer panels frequently cost \$10,000 to \$50,000 per fielding.
2. **Protracted Turnaround Timelines:** Standard fielding campaigns require 3 to 8 weeks from instrument design to statistical reporting.
3. **Sampling Inflexibility:** Capturing representative feedback across heterogeneous socio-economic strata is logistically challenging.
4. **Failure of Direct Likert Elicitation (DLE):** Prompting Large Language Models (LLMs) directly for numeric ratings (e.g., *"Rate this phone from 1 to 5"*) fails catastrophically due to **central tendency bias** (rigid clustering at 3 and 4 stars), **distributional variance collapse**, **digit hallucination**, and **mode collapse** across diverse consumer personas.

To solve this foundational problem, this project introduces a fully offline, privacy-preserving, reproducible end-to-end framework combining **empirical digital personas**, a **Zero Numerical Leakage** generation protocol, **dual-gate response validation**, and an **Enhanced Semantic Similarity Rating (E-SSR)** mathematical engine.

---

## 2. Digital Persona Architecture Grounded in Empirical Survey Data

Rather than utilizing hand-crafted or stereotypical agent prompts, the framework grounds its personas in an empirical human survey dataset of **258 verified respondents** (`data/clean/human_survey_clean.csv`) with demographic and behavioral clustering:
- **Demographics:** 54.7% Female (n = 141), 45.3% Male (n = 117); Median Age = 20.0 years (57.0% aged 18–20, 37.6% aged 21–22); 45.3% Town/Rural, 25.6% Metro, 22.1% Tier-2, 7.0% Tier-3.
- **P1 – Premium Consumer (`P1_premium_tier_owner`):** Historical device investment > ₹35,000 (Category zone: ₹50,000–₹1,00,000+). Values aesthetic elegance, build finish, tactile materials, and top-tier reliability. Highly skeptical of cheap plastic finishes.
- **P2 – Tech Enthusiast (`P2_midrange_heavy_user`):** Historical investment ₹10,000–₹35,000 with intensive usage (>6 hours daily). Values processor benchmarks, RAM, multitasking, thermal throttling stability, and 120Hz display responsiveness.
- **P3 – Practical Everyday User (`P3_midrange_moderate_user`):** Historical investment ₹10,000–₹35,000 with balanced daily usage. Values everyday dependability, battery longevity, intuitive UI navigation, and practical utility.
- **P4 – Budget-Conscious User (`P4_budget_or_undisclosed`):** Historical investment < ₹10,000 or undisclosed allowance. Extremely price-sensitive, evaluates essential utility per rupee, and treats prices above ₹15,000 as prohibitive.

---

## 3. Technology Adoption Constructs & Approved Anchor Codebook

The framework operationalizes six independent constructs grounded in UTAUT2 psychometric theory:
1. **Performance Expectancy (PE):** Everyday task productivity, speed, and multitasking efficiency.
2. **Effort Expectancy (EE):** Ease of learning, user interface clarity, and intuitive onboarding.
3. **Hedonic Motivation (HM):** Multimedia entertainment, display vibrancy, and tactile pleasure.
4. **Social Influence (SI):** Peer recommendations, social perception, and brand prestige.
5. **Perceived Protection (PP):** Data privacy, encryption, and account security (**never conflated with physical build quality**).
6. **Behavioral Intention (BI):** Prospective purchase, adoption, or switching intention given product capabilities and price.

Crucially, **antecedents (PE, EE, HM, SI, PP) are strictly separated from outcome criteria (BI)**. Retail price is quarantined exclusively within BI, preventing economic bias from corrupting functional perceptions. The system embeds responses against **30 domain-adapted 5-point Likert anchor statements** (`docs/survey_instrument/smartphone_construct_anchor_statements.csv`).

---

## 4. The E-SSR Mathematical Formulation

Semantic Scale Rating (SSR) projects unconstrained qualitative text into a continuous Likert distribution:

### Step 1: Raw Cosine Similarity (Maier et al., Eq. 7)
$$\gamma(\sigma_{r,i}, t) = \frac{\mathbf{v}_{r,i} \cdot \mathbf{v}_t}{\|\mathbf{v}_{r,i}\| \, \|\mathbf{v}_t\|}$$

### Step 2: Minimum-Subtracted Similarity Weights (Maier et al., Eq. 8)
$$w_i(r) = \gamma(\sigma_{r,i}, t) - \gamma(\sigma_{\ell,i}, t) + \epsilon \cdot \delta_{\ell,r}, \quad \text{where } \ell = \arg\min_j \gamma(\sigma_{j,i}, t)$$

### Step 3: Top-k Truncation ($k = 3$) & Normalization (E-SSR Enhancement)
To eliminate distant semantic noise, E-SSR retains only the $k = 3$ largest weights and sets the remaining two to zero:
$$w_{\text{topk}}(r) = \begin{cases} w(r) & \text{if } r \in \text{TopK}(w, k=3) \\ 0 & \text{otherwise} \end{cases}$$
$$p_{\text{raw}}(r) = \frac{w_{\text{topk}}(r)}{\sum_{j=1}^5 w_{\text{topk}}(j)}$$

### Step 4: Mild Temperature Sharpening ($T = 0.75$) (E-SSR Enhancement)
To counteract scale edge contraction without inducing artificial argmax collapse:
$$p_{\text{E-SSR}}(r) = \frac{p_{\text{raw}}(r)^{1/T}}{\sum_{j=1}^5 p_{\text{raw}}(j)^{1/T}} \quad \text{for } p_{\text{raw}}(r) > 0$$

### Step 5: Continuous Expected Likert Rating & Uncertainty
$$\mathbb{E}[r] = \sum_{r=1}^5 r \cdot p_{\text{E-SSR}}(r)$$
$$H(p) = -\frac{\sum_{r=1}^5 p(r) \log_2 p(r)}{\log_2(5)}$$

---

## 5. Dual-Gate Validation & Quality Assurance Pipeline

1. **Gate 1 (Pre-Generation Context Gate):** Automatically detects product family across 12 technology categories, enforces usage terminology, and validates that product descriptions do not contradict product categories or reference foreign brands.
2. **Gate 2 (Post-Generation Construct Gate):** Automatically inspects generated text prior to embedding:
   - Prohibits numerical leaks (`NUMERIC_RATING`: digits, fractions, stars, percentages).
   - Prohibits breaking character (`PERSONA_LABEL`: e.g., *"as a P1 user"*).
   - Enforces price quarantine (`PRICE_WORDS`: price strictly forbidden in PE, EE, HM, SI, PP).
   - Prohibits purchase verbs outside BI (`PURCHASE_OUTSIDE_BI`).
   - Requires explicit adoption intention in BI (`BI_INTENT`).
   - Filters robotic/formulaic prose (`HARD_WORDS`).
   - Enforces length bounds (30–130 words; BI floor 20 words).
3. **Targeted Correction Retry Loop:** Failing answers trigger concise targeted rewrite prompts (maximum 2 retries). Only failing answers are re-prompted; valid answers are preserved. Answers that still fail are saved with full diagnostic issue logs and left unscored.

---

## 6. Key Experimental Benchmarks & Product Results

### 1. Primary Research Benchmark — 79.17% (19/24) Qwen Blind-Judge Agreement
- **Exact Rating Agreement:** 79.17% (19 out of 24 persona-construct evaluations).
- **Within-1-Point Agreement:** 100.00% (24 out of 24 evaluations).
- **Discrepant (>1 Point) Errors:** 0.00% (0 out of 24 evaluations).
- **Evaluation Context:** Engineering agreement between mathematical E-SSR projection of qualitative text and an independent blind Qwen 2.5 7B judge evaluating the same text.

### 2. Multi-Product Commercial Case Studies
| Product Evaluated | Retail Price | PE | EE | HM | SI | PP | Overall BI | Total Runtime | Qwen Calls |
|---|---|---|---|---|---|---|---|---|---|
| **Google Pixel 8a** | ₹39,999 | 3.31 | 4.24 | 3.89 | 3.30 | 2.59 | **3.17 / 5.0** | 15m 33s (933s) | 9 (4 main + 5 corr) |
| **OnePlus Nord CE4 5G** | ₹24,999 | 3.60 | 3.88 | 3.70 | 2.97 | 2.32 | **3.24 / 5.0** | 11m 35s (695s) | 7 (4 main + 3 corr) |
| **Samsung Galaxy M35 5G**| ₹20,000 | 3.36 | 4.08 | 3.89 | 2.97 | 1.94 | **2.88 / 5.0** | 21m 31s (1291s)| 7 (4 main + 3 corr) |

### 3. Google Pixel 8a Persona Divergence Breakdown
- **P1 (Premium):** BI = 3.20 (Hesitant regarding plastic/chassis finish vs. ₹40k flagship expectation).
- **P2 (Tech Enthusiast):** BI = 3.09 (Requires sustained gaming and thermal throttling benchmarks).
- **P3 (Practical Everyday):** BI = 3.41 (Highest adoption intent; values clean Pixel UI and camera reliability).
- **P4 (Budget-Conscious):** BI = 2.98 (Lowest adoption intent; ₹39,999 is on the higher side for essential needs).

---

## 7. Full-Stack Research Dashboard Architecture

- **Backend (`dashboard/server.py`):** Multi-threaded REST API server (`/api/health`, `/api/evaluation/options`, `/api/evaluation/run`, `/api/evaluation/status/<id>`, `/api/evaluation/saved`, `/api/evaluation/export`).
- **Frontend (`dashboard/static/`):** High-performance Single Page Application in vanilla ES6 JavaScript, HTML5, and CSS3 with zero npm build dependencies. Features live progress monitoring, Persona × Construct heatmap matrix, interactive SVG PMF distribution bar charts, and one-click CSV/JSON export.

---

## 8. Research Contributions & Scientific Significance

1. **Empirical Digital Persona Parameterization:** Demonstrates that clustering human survey responses generates realistic consumer simulation archetypes without mode collapse.
2. **Zero Numerical Leakage Elicitation:** Eliminates the central tendency bias and digit hallucination endemic to Direct Likert Elicitation.
3. **E-SSR Mathematical Optimization:** Proves that top-k truncation ($k=3$) and temperature scaling ($T=0.75$) eliminate distant semantic noise while counteracting scale edge contraction.
4. **Dual-Gate Architectural Validation:** Proves that lexical constraints and targeted retries maintain strict psychometric separation between antecedents and outcomes.
5. **Privacy-Preserving Offline Laboratory:** Provides a full-stack, local research environment eliminating cloud API costs and corporate data leakage risks.

