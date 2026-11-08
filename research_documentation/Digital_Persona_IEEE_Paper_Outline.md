# IEEE Paper Preparation Outline & Submission Blueprint

**Paper Title:** Digital Persona-Based Synthetic Survey Data Generation for Technology Adoption Research: An SSR-Based Validation Framework  
**Target IEEE Transactions:** *IEEE Transactions on Computational Social Systems (TCSS)*  
**Alternative Venues:** *IEEE Access*, *IEEE Transactions on Emerging Topics in Computational Intelligence*, *IEEE ICWS 2026*  
**Authors:** Mehul et al.  
**Affiliation:** Department of Computer Engineering, SVKM's NMIMS MPSTME  

---

## Abstract Structure (200–250 Words)
- **Background & Context:** Importance of technology acceptance research (UTAUT2) and the logistical/economic cost barriers of empirical human focus groups and Likert surveys.
- **Problem Statement:** Catastrophic failure of Direct Likert Elicitation (DLE) in LLMs due to central tendency bias, variance truncation, and mode collapse across heterogeneous personas.
- **Proposed Methodology:** An offline, privacy-preserving framework integrating empirical digital personas (P1–P4), the Zero Numerical Leakage elicitation principle, dual-gate lexical validation, and Enhanced Semantic Similarity Rating (E-SSR, $k=3$, $T=0.75$).
- **Key Empirical Results:** 79.17% exact agreement with Qwen blind-judge labels (100% within-1-point); demonstrated behavioral divergence across commercial smartphones (Pixel 8a, Nord CE4, M35); full psychometric separation between antecedents and primary adoption intention (BI).
- **Impact & Significance:** Provides an auditable, privacy-preserving computational laboratory for automated pre-launch technology adoption research.
- **Index Terms (Keywords):** Large Language Models (LLMs), Synthetic Survey Generation, Semantic Similarity Rating (SSR), Digital Personas, Technology Adoption, UTAUT2, Likert Scale Recovery, Natural Language Processing.

---

## 1. Introduction
- **1.1 The Imperative of Pre-Launch Technology Adoption Assessment:** Theoretical primacy of UTAUT/UTAUT2; high monetary cost (\$10k–\$50k) and slow turnaround (4–8 weeks) of traditional human surveys.
- **1.2 Generative AI in Computational Social Science:** Shift toward generative agent simulations (Argyle et al., 2023; Park et al., 2023); potential for synthetic respondents.
- **1.3 Pitfalls of Direct Likert Elicitation (DLE):** Systematic analysis of why autoregressive token generation fails when asked for scalar digits (central tendency bias, digit hallucination, cross-persona collapse).
- **1.4 Paradigm Shift: Semantic Scale Ratings (SSR):** Introduction of Maier et al. (2025) SSR approach; qualitative prose elicitation paired with geometric projection.
- **1.5 Summary of Key Contributions:**
  1. Empirical persona parameterization grounded in 258 human survey respondents.
  2. The Zero Numerical Leakage elicitation protocol for local LLMs (Qwen 2.5 7B).
  3. Formulation and verification of E-SSR ($k=3$, $T=0.75$).
  4. Dual-gate validation framework enforcing construct boundary integrity and price quarantine.
  5. Empirical multi-product validation and full-stack research dashboard.

---

## 2. Related Work & Theoretical Grounding
- **2.1 Theoretical Frameworks in Technology Adoption:** TAM (Davis, 1989), UTAUT (Venkatesh et al., 2003), and UTAUT2 (Venkatesh et al., 2012). Focus on the causal distinction between functional antecedents (PE, EE, HM, SI, PP) and behavioral outcome criteria (BI).
- **2.2 LLMs as Synthetic Human Proxies:** Review of recent studies simulating human demographics and decision-making; critical evaluation of persona stability and prompt sensitivity.
- **2.3 Psychometric Scale Recovery & SSR:** Deep dive into Maier et al. (arXiv:2510.08338v3); comparative analysis of prompting vs. embedding projection; limitations of canonical SSR in domain-specific technology adoption.

---

## 3. The Digital Persona-Based Generation Framework
- **3.1 Empirical Human Survey Cohort (n = 258):** Survey methodology, sampling characteristics (54.7% female, median age 20.0, geographic tier distributions).
- **3.2 Four Empirical Digital Personas:**
  - *P1 (Premium Consumer):* High historical investment (> ₹35,000), luxury aesthetics, build finish priority.
  - *P2 (Tech Enthusiast):* Mid-range intensive user, chipset speed, RAM, 120Hz display, multitasking focus.
  - *P3 (Practical Everyday User):* Mainstream balanced user, reliability, battery endurance, straightforward usability.
  - *P4 (Budget-Conscious User):* Entry-level value user (< ₹10,000), value per rupee, price-sensitive hesitation.
- **3.3 Anti-Stereotyping Controls & Prompt Architecture:** Structured system prompts, prompt prefix caching across personas, and elimination of ungrounded personal backstories.

---

## 4. Construct Operationalization & Dual-Gate Validation
- **4.1 Technology Adoption Construct Battery:** Detailed definitions of PE, EE, HM, SI, PP, BI, and PV (qualitative only); strict isolation of Perceived Protection (PP) as data/account privacy rather than hardware build durability.
- **4.2 The 30-Item Domain-Adapted Anchor Codebook:** 5-point Likert anchor statements across all six constructs; mathematical validation of monotonicity and semantic ordering.
- **4.3 Gate 1: Pre-Generation Context Gate:** Lexical category detection across 12 product families; terminology and price-tier enforcement.
- **4.4 Gate 2: Post-Generation Construct Gate:** Lexical checks prohibiting numeric leaks, persona label breaking, price contamination outside BI, and unsupported claims.
- **4.5 Targeted Correction Loop:** Algorithmic details of the maximum 2-round targeted retry mechanism.

---

## 5. Enhanced Semantic Similarity Rating (E-SSR) Mathematical Engine
- **5.1 Mathematical Formulations:**
  - Raw cosine similarity: $\gamma(\sigma_{r,i}, t) = (\mathbf{v}_{r,i} \cdot \mathbf{v}_t) / (\|\mathbf{v}_{r,i}\| \|\mathbf{v}_t\|)$.
  - Minimum subtraction: $w_i(r) = \gamma(\sigma_{r,i}, t) - \gamma(\sigma_{\ell,i}, t) + \epsilon \delta_{\ell,r}$.
  - Top-k truncation ($k = 3$): zeroing out the two most distant anchors.
  - Mild temperature sharpening ($T = 0.75$): counteracting scale edge contraction.
  - Probability Mass Function (PMF) and continuous Expected Likert Rating: $\mathbb{E}[r] = \sum r \cdot p(r)$.
  - Distributional uncertainty via normalized Shannon entropy: $H(p)$.
- **5.2 Local Semantic Embedding Space:** Mathematical properties and advantages of `nomic-embed-text` (768 dimensions) over cloud embedding APIs.
- **5.3 Optional Variance Rescaling:** Mathematical formulation of `LinearVarianceRescaler` for human distribution alignment.

---

## 6. Experimental Evaluation & Results
- **6.1 Experimental Protocol:** 4 × 6 balanced factorial design (24 evaluations per product); offline inference execution on local hardware.
- **6.2 Primary Research Benchmark (79.17% Blind-Judge Agreement):**
  - Presentation of 19/24 exact agreement and 100% within-1-point agreement with independent Qwen blind-judge labels.
  - Confusion matrix and error distribution analysis.
- **6.3 Commercial Product Case Studies:**
  - *Google Pixel 8a (₹39,999):* Full 24-point evaluation matrix; validation gate statistics (16 passed first time, 8 corrected); runtime economics (15m 33s, 9 calls).
  - *OnePlus Nord CE4 5G (₹24,999):* Construct profile (PE = 3.60, BI = 3.24, runtime = 11m 35s).
  - *Samsung Galaxy M35 5G (₹20,000):* Construct profile (PE = 3.36, BI = 2.88, runtime = 21m 31s).
- **6.4 Cross-Product Comparative Analysis:** Behavioral divergence, price elasticity across personas, and construct radar visualizations.

---

## 7. System Implementation & Research Dashboard
- **7.1 Architecture of `dashboard/server.py`:** REST API endpoints, asynchronous daemon job management, file streaming.
- **7.2 Architecture of `dashboard/static/`:** Zero-dependency Single Page Application (vanilla ES6, HTML5, CSS3); live execution monitoring, PMF bar chart rendering, and CSV/JSON export.

---

## 8. Discussion & Implications
- **8.1 Psychometric Validity:** Preservation of variance and elimination of central tendency bias.
- **8.2 Practical Implications for Industry:** Rapid, zero-cost pre-launch concept screening for product managers and hardware designers.
- **8.3 Methodological Robustness:** Importance of lexical validation gates and prompt prefix caching for local edge computing.

---

## 9. Limitations & Future Directions
- **9.1 Limitations:** Single anchor set per construct (vs. 6 sets in original paper); youth/student demographic focus; single local LLM architecture.
- **9.2 Future Work:** Multi-set construct anchor averaging; large-scale human panel co-validation; parameter-efficient fine-tuning via demographic LoRA adapters.

---

## 10. Conclusion
- Synthesis of findings; final verdict on the viability of Digital Persona E-SSR frameworks for computational survey science.

---

## References (IEEE Format)
- Complete bibliographic references citing foundational psychometrics (Venkatesh, Davis, Likert), generative agents (Argyle, Park), SSR (Maier et al., 2025), and model architectures (Qwen, Nomic).

