# Product-Evaluation Prompt Design & Construct Mapping Specification

**Project:** Digital Persona-Based Synthetic Consumer Research  
**Workflow Stage:** Step 3 — Design Construct-Specific Product-Evaluation Prompts  
**Date:** October 2026  
**Status:** Completed & Verified  

---

## 1. Executive Summary

This document specifies the prompt architecture and theoretical construct mapping designed to evaluate new mobile product concepts using local open-weight language models (specifically **Qwen 2.5 7B Instruct** via Ollama) conditioned on empirical digital personas.

In strict adherence to the project's methodology, **large language models are never asked for numerical ratings, Likert scores, probabilities, or adoption percentages**. Instead, Qwen generates rich, realistic qualitative consumer critiques. The project's canonical Semantic Similarity Rating (SSR) pipeline converts these textual responses into continuous expected ratings $E[R] \in [1.0, 5.0]$ by computing cosine similarities against approved 5-point Likert anchor statements.

---

## 2. Survey Instrument & Construct Audit

A rigorous audit of the project's fielded questionnaire ([`data/Capstone_Reviews(Responses)(1).xlsx`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/Capstone_Reviews(Responses)(1).xlsx)), cleaned dataset ([`data/human_survey_cleaned.csv`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/human_survey_cleaned.csv)), and survey instrument codebook ([`docs/survey_instrument/survey_codebook.csv`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/docs/survey_instrument/survey_codebook.csv)) establishes the empirical boundaries of measurable constructs.

### 2.1 The Two Levels of Construct Evidence

The project contains two distinct levels of construct definition that must not be conflated:
1. **Empirically Fielded Survey Items ($N=258$ eligible respondents)**:
   - Demographics & behavioral habits: Age (`Q3`), gender (`Q6`), education (`Q7`), status (`Q8`), area tier (`Q11`), spending capacity (`Q10`), brand (`Q12`), price paid (`Q13`), OS (`Q14`), replacement cycle (`Q16`), upgrade trigger (`Q17`), screen time (`Q18`), use cases (`Q19`), information sources (`Q22`).
   - Likert items (only two): Technology Comfort (`Q20`) and Brand Rebuy Intention (`Q21`).
   - Free-text qualitative questions (four): Overall experience (`Q23`), battery & camera (`Q24`), performance & display (`Q25`), and price worthiness / repurchase rationale (`Q26`).
2. **Theoretical UTAUT Codebook & Approved Domain Anchors**:
   - The theoretical framework models technology adoption via domain-adapted UTAUT constructs, with approved 5-point Likert anchor sets stored in [`docs/survey_instrument/smartphone_construct_anchor_statements.csv`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/docs/survey_instrument/smartphone_construct_anchor_statements.csv).

---

### 2.2 Supported Measurable Constructs

| Construct Code | Canonical Name | Theoretical Basis | Fielded Survey Evidence | Approved SSR Anchors | Primary Evaluation Focus in Product Prompts |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **`PE`** | **Performance Expectancy** | UTAUT | `Q19` (main uses), `Q25` (speed, performance), `FT_PE01` | **Yes** (5 anchors) | How effectively the product helps accomplish everyday tasks, productivity, and academic/professional work. |
| **`BI`** | **Behavioral Intention** | UTAUT | `Q21` (rebuy likelihood), `Q26` (repurchase reasoning), `Q16` (replacement cycle), `FT_BI01` | **Yes** (5 anchors) | Realism of adoption: whether the persona would actually consider purchasing or switching to this smartphone. |
| **`EE`** | **Effort Expectancy** | UTAUT | `Q20` (tech comfort Likert 1–5), `Q23` (overall usability) | **Yes** (5 anchors) | Ease of learning, clarity of interface, and transition friction from their current OS. |
| **`HM`** | **Hedonic Motivation** | UTAUT | `Q18` (daily screen time), `Q19` (entertainment, gaming, social media), `Q25` (display quality) | **Yes** (5 anchors) | Fun, enjoyment, entertainment, and multimedia experience offered by the product. |
| **`SI`** | **Social Influence** | UTAUT | `Q22` (word-of-mouth from friends/family, tech reviews, social media influencers) | **Yes** (5 anchors) | Peer perception, social circle norms, and brand prestige among student/professional peers. |
| **`PP`** | **Perceived Protection** | UTAUT Extension | Working interpretation: on-device privacy, software updates, and personal data protection | **Yes** (5 anchors) | Security controls, biometric convenience, and personal privacy assurances. |
| **`PV`** | **Price Value** | UTAUT2 (Open-ended) | `Q10` (monthly spending capacity), `Q13` (price paid), `Q26` (perceived worth vs cost), `FT_PV01` | *No* (Qualitative text only; unanchored for SSR) | Value-for-money: whether the proposed price feels justified relative to the persona's spending power. |

---

### 2.3 Explicitly Unavailable / Unmeasured Constructs

To maintain scientific integrity, the prompt generator explicitly **rejects** requests to evaluate constructs not supported by the survey:

- **Trust**: No survey items or anchors measure vendor, institutional, or AI system trust.
- **Perceived Risk**: Financial risk, security threat perception, and physical device hazard are not operationalised.
- **Perceived Usefulness (PU) & Perceived Ease of Use (PEOU)**: These TAM-specific constructs are superseded by UTAUT's Performance Expectancy (`PE`) and Effort Expectancy (`EE`).
- **Facilitating Conditions (`FC`)**: The legacy alias mapping `FC` to Perceived Protection was deliberately removed on 2026-10-29; no `FC` items exist in the codebook or approved anchor CSV.
- **Willingness to Pay (WTP Curve)**: Econometric willingness-to-pay elasticity is not measured; the survey captures only historical price paid brackets (`Q13`) and monthly spending capacity (`Q10`).
- **Affective Brand Loyalty**: The survey measures single-item brand rebuy intention (`Q21`), not a multidimensional psychological loyalty scale.

---

## 3. Questionnaire Mapping Matrix

The complete mapping of all 26 survey questions to constructs, data types, and SSR relevance:

| Field Index | Question ID | Exact Questionnaire Question Text | Data Type | Construct / Role | SSR Scoring Relevance |
| :---: | :--- | :--- | :--- | :--- | :--- |
| `0` | `Timestamp` | Submission timestamp | Timestamp | Audit / Deduplication | None |
| `1` | `Q1` | *Do you agree to participate in this survey?* | Binary | Informed consent screener | Inclusion filter |
| `2` | `Q2` | *Are you 18 years old or above?* | Binary | Age eligibility screener | Inclusion filter |
| `3` | `Q3` | *What is your age?* | Numeric | Demographic: Age | Persona demographic conditioning |
| `4` | `Q4` | *Do you personally own and use a smartphone?* | Binary | Ownership screener | Inclusion filter |
| `5` | `Q5` | *How long have you been using your current smartphone?* | Categorical | Usage duration | Persona conditioning |
| `6` | `Q6` | *What is your gender?* | Categorical | Demographic: Gender | Persona conditioning |
| `7` | `Q7` | *What is your highest level of education?* | Categorical | Demographic: Education | Persona conditioning |
| `8` | `Q8` | *What is your current status?* | Categorical | Demographic: Status | Persona conditioning |
| `9` | `Q10` | *What is your approximate monthly personal spending capacity?* | Ordinal bracket | Socioeconomic capacity | Price sensitivity conditioning |
| `10` | `Q11` | *What type of area do you live in?* | Categorical | Geographic tier | Persona conditioning |
| `11` | `Q12` | *Which brand is your current smartphone?* | Categorical | Device ecosystem / Brand | Brand switching friction |
| `12` | `Q13` | *Approximately how much did you pay for your current smartphone?* | Ordinal bracket | Price tier paid | Primary Persona Segmentation Rule |
| `13` | `Q14` | *Which operating system does your smartphone use?* | Categorical | OS ecosystem | OS switching barrier |
| `14` | `Q15` | *How many smartphones have you owned in the last five years?* | Numeric / Ordinal | Tech churn rate | Adoption history |
| `15` | `Q16` | *How often do you usually replace your smartphone?* | Categorical | Replacement cycle | Behavioral intention timing |
| `16` | `Q17` | *What was the main reason for upgrading to your current smartphone?* | Categorical | Upgrade trigger | Feature attraction priorities |
| `17` | `Q18` | *How much time do you spend using your smartphone each day?* | Ordinal bracket | Usage intensity | Persona Segmentation Rule (P2 vs P3) |
| `18` | `Q19` | *What do you mainly use your smartphone for?* | Multi-select | Primary use cases | Task / Performance Expectancy |
| `19` | `Q20` | *How comfortable are you with using new technology?* | Likert (1–5) | Tech comfort / EE proxy | Effort Expectancy proxy |
| `20` | `Q21` | *How likely are you to buy the same smartphone brand again?* | Likert (1–5) | Rebuy intention / BI proxy | Primary empirical baseline for BI |
| `21` | `Q22` | *What sources did you use when choosing your current smartphone?* | Multi-select | Information channels | Social Influence proxy |
| `22` | `Q23` | *How would you describe your overall experience with your current smartphone?* | Free-text | Overall experience theme | Grounding text |
| `23` | `Q24` | *How would you describe your smartphone's battery life and camera quality?* | Free-text | Battery & Camera themes | Feature expectation grounding |
| `24` | `Q25` | *How would you describe your smartphone's performance, speed, and display quality?* | Free-text | Performance & Display themes | Feature expectation grounding |
| `25` | `Q26` | *Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?* | Free-text | Price-value & Rebuy rationale | Behavioral Intention empirical proxy |

---

## 4. Product-Evaluation Prompt Architecture

The reusable prompt system is implemented in [`scripts/product_evaluation_prompts.py`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/scripts/product_evaluation_prompts.py).

### 4.1 System Prompt Design

The system prompt sets the role, instructions, and hard methodological constraints for **Qwen 2.5 7B Instruct**:

```text
You role-play one real consumer from an empirical market-research panel reacting to a proposed mobile product concept. 
Respond in your own authentic voice from the perspective of your assigned persona profile.

EVALUATION INSTRUCTIONS:
1. Honestly evaluate the proposed product against your current smartphone, daily routine, and personal budget.
2. State clearly:
   - Which specific features attract you and why.
   - Which features feel unappealing, gimmicky, or unnecessary for your needs.
   - Reasons why you would consider adopting or purchasing this product.
   - Reasons for hesitation, skepticism, or rejection.
   - Any concerns or barriers (e.g. price vs. spending capacity, operating system transition, battery life, display).
   - Concrete improvements that would make the product substantially more appealing to you.
   - Whether the product feels genuinely relevant to your daily lifestyle.
3. Maintain a natural, conversational, critical tone. Do not default to polite praise; give mixed, critical, or skeptical feedback if the product does not fit your profile.

STRICT METHODOLOGICAL RESTRICTIONS (MANDATORY):
- Output TEXT ONLY: 4 to 8 sentences in continuous prose.
- NEVER provide numerical ratings, Likert scores, points, stars, percentages, or probabilities (e.g., do NOT say '4/5', '8/10', '70% chance').
- NEVER state an adoption percentage or conversion score.
- Do NOT mention being an AI, an LLM, or a language model.
```

### 4.2 User Prompt Structure

The user prompt combines four modular components:
1. **Persona Context Block**: Grounded in empirical respondent characteristics:
   - Segment definition and cohort percentage.
   - Median age, gender, education, employment status, tier city.
   - Current phone brand and operating system.
   - Daily screen time and primary use cases.
   - Monthly personal spending capacity.
   - Self-reported technology comfort (1–5 scale).
   - Top qualitative priorities from `Q23`–`Q25` free text.
2. **Product Concept Specification**:
   - Product name and high-level concept description.
   - Proposed retail price in INR.
   - Target consumer group (if defined by the product team).
   - Itemized list of features, specifications, and explanatory context.
3. **Construct-Specific Evaluation Dimensions**:
   - Explicit prompt instructions for each selected construct (e.g., Performance Expectancy, Behavioral Intention, Effort Expectancy).
4. **Execution Directive**:
   - Instructions to write 4–8 sentences of natural prose addressing features, barriers, and purchase consideration without numeric scores.

---

## 5. Protecting Canonical SSR Methodology

> [!IMPORTANT]
> **Why Numerical Outputs are Strictly Prohibited:**
> In direct prompting, LLMs suffer from severe calibration artifacts, extreme positive skew (clustering on 4 and 5), and prompt-sensitivity when asked for numeric Likert ratings.
> 
> Under the Semantic Similarity Rating (SSR) methodology:
> 1. The LLM acts purely as a semantic generator of natural language consumer opinion.
> 2. The text is projected into a vector space using a frozen sentence encoder (e.g., `BAAI/bge-base-en-v1.5`, `all-mpnet-base-v2`, `all-MiniLM-L6-v2`).
> 3. Cosine similarities are computed against the 5 approved anchor statements for the construct.
> 4. Canonical normalisation (minimum-subtraction, temperature $T=1.0$) yields a Probability Mass Function $P = [p_1, p_2, p_3, p_4, p_5]$.
> 5. Expected rating $E[R] = \sum_{i=1}^5 i \cdot p_i$.
> 
> If the LLM generates numbers or stars in the text, it corrupts the semantic embedding and creates artificial similarity to high-rating anchors.

The prompt design enforces this protection through:
- Hard negative system prompt constraints.
- Automated validation via `validate_prompt_for_ssr()`.
- Quality filters in `dp_validation_agent.py` and `product_research.py` that reject text containing rating strings (`X/5`, `rating:`, `stars`).

---

## 6. Implementation & Verification

### 6.1 Created Files

1. [`scripts/product_evaluation_prompts.py`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/scripts/product_evaluation_prompts.py):
   - Reusable prompt template module.
   - Supported construct registry (`SUPPORTED_CONSTRUCTS`).
   - Unavailable construct registry (`UNAVAILABLE_CONSTRUCTS`).
   - `ProductFeature` and `ProductConcept` dataclasses.
   - `build_persona_context()` extracting empirical data from `persona_profiles.json`.
   - `build_product_evaluation_prompt()` builder function.
   - `validate_requested_constructs()` validation function.
   - `validate_prompt_for_ssr()` leakage checker.

2. [`tests/test_product_evaluation_prompts.py`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/tests/test_product_evaluation_prompts.py):
   - 13 focused test cases covering construct validation, persona context generation, text-only mandates, critical consumer feedback requirements, SSR anchor alignment, and metadata tracking.

3. [`docs/product_evaluation_prompt_design.md`](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/docs/product_evaluation_prompt_design.md):
   - This comprehensive specification document.

### 6.2 Test Suite Execution

All 13 new unit tests pass cleanly:
```bash
python -m pytest tests/test_product_evaluation_prompts.py
# 13 passed in 0.54s
```

All 38 critical pipeline and verification tests pass:
```bash
python -m pytest tests/test_clean_human_survey.py tests/test_dp_pipeline.py tests/test_pipeline_verification.py tests/test_product_evaluation_prompts.py
# 38 passed in 12.90s
```

---

## 7. Known Methodological Limitations

1. **Student Sample Skew**: The underlying survey sample ($N=258$, median age 20) is predominantly composed of undergraduate and postgraduate students. Product evaluations will naturally reflect the perspectives, financial constraints, and media habits of young Indian consumers.
2. **Proxy vs Direct Ground Truth**: The fielded survey measures brand repurchase intention (`Q21`), not willingness to adopt an entirely unreleased third-party product concept. Results represent synthetic simulated consumer reactions, not binding purchase commitments.
3. **Aspect Coverage**: Aspect satisfactions (battery, camera, performance) are captured through free-text responses rather than separate multi-item Likert batteries.

