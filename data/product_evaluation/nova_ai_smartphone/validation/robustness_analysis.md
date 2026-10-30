# Sensitivity and Robustness Analysis

**Evaluation:** Nova AI Smartphone SSR Ratings  
**Embedding Model:** `nomic-embed-text` (Local Ollama, 768 dimensions)  
**Mathematical Pipeline:** Canonical minimum-subtraction normalisation ($T=1.0, \epsilon=0.0$)  

---

## 1. Summary of Distribution Spreads and Entropy

In Semantic Similarity Rating (SSR), the **cosine similarity spread** $\Delta s = \max(s) - \min(s)$ measures how sharply the embedding model distinguishes between the five Likert anchor statements.
- When $\Delta s$ is large ($> 0.08$), the response is clearly closer to certain anchors, producing a decisive PMF.
- When $\Delta s$ is small ($< 0.04$), all five anchors are nearly equidistant in embedding space, indicating higher rating uncertainty.

| Persona | Construct | Expected Rating ($E[R]$) | Cosine Spread ($\Delta s$) | Shannon Entropy ($H$) | Dominant Anchor Category (and Mass) | Uncertainty Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **P1** | PE | 3.14 | 0.0622 | 1.82 | Cat 3: Neither agree nor disagree (41.8%) | Moderate certainty |
| **P1** | BI | 3.26 | **0.0931** | 1.86 | Cat 3: Might consider buying (43.4%) | High signal / Decisive |
| **P1** | EE | 3.38 | 0.0677 | 1.84 | Cat 3: Neither easy nor difficult (45.3%) | Moderate certainty |
| **P1** | HM | 2.55 | **0.0403** | 1.96 | Cat 3: Moderately entertaining (34.9%) | **High uncertainty (narrow spread)** |
| **P1** | SI | 3.28 | 0.0615 | 1.40 | Cat 3: Neutral social influence (63.4%) | Highly concentrated |
| **P1** | PP | 2.94 | 0.0865 | 1.96 | Cat 2: Disagree privacy is protected (32.3%) | Moderate certainty |
| **P2** | PE | 2.73 | **0.0312** | 1.45 | Cat 3: Moderately useful (48.6%) | **High uncertainty (narrow spread)** |
| **P2** | BI | 3.19 | **0.1175** | 1.76 | Cat 3: Might consider buying (48.0%) | **Highest signal / Decisive** |
| **P2** | EE | 3.46 | 0.0591 | 1.96 | Cat 3: Neither easy nor difficult (31.7%) | Dispersed distribution |
| **P2** | HM | 2.80 | **0.0291** | 1.87 | Cat 3: Moderately entertaining (42.2%) | **High uncertainty (narrow spread)** |
| **P2** | SI | 3.32 | **0.0953** | 1.36 | Cat 3: Neutral social influence (67.0%) | Highly concentrated |
| **P2** | PP | 2.98 | 0.0788 | 1.97 | Cat 4: Agree privacy is protected (30.6%) | Dispersed distribution |
| **P3** | PE | 2.63 | **0.0283** | 1.82 | Cat 2: Disagree useful for tasks (38.7%) | **High uncertainty (narrow spread)** |
| **P3** | BI | 3.21 | **0.1086** | 1.73 | Cat 3: Might consider buying (50.0%) | **High signal / Decisive** |
| **P3** | EE | 3.63 | **0.0953** | 1.78 | Cat 3: Clear and understandable (43.8%) | High signal / Decisive |
| **P3** | HM | 2.74 | **0.0379** | 1.94 | Cat 4: Agree entertaining (34.2%) | **High uncertainty (narrow spread)** |
| **P3** | SI | 3.34 | 0.0824 | 1.67 | Cat 3: Neutral social influence (53.8%) | High signal / Decisive |
| **P3** | PP | 3.08 | 0.0872 | 1.93 | Cat 4: Agree privacy is protected (35.7%) | High signal / Decisive |
| **P4** | PE | 2.44 | 0.0403 | 1.74 | Cat 2: Disagree useful for tasks (44.1%) | Moderate certainty |
| **P4** | BI | 3.01 | **0.1008** | 1.48 | Cat 3: Neutral purchase intention (62.3%) | **Highest concentration** |
| **P4** | EE | 3.13 | **0.0278** | 1.92 | Cat 5: Strongly agree easy (31.4%) | **Bimodal / High sensitivity** |
| **P4** | HM | 2.72 | **0.0395** | 1.66 | Cat 3: Moderately entertaining (51.6%) | Moderate certainty |
| **P4** | SI | 3.07 | **0.0928** | 1.72 | Cat 3: Neutral social influence (50.2%) | High signal / Decisive |
| **P4** | PP | 2.80 | 0.0412 | 1.97 | Cat 2: Disagree privacy is protected (28.4%) | Dispersed distribution |

---

## 2. Temperature Sensitivity Analysis ($T \in [0.5, 1.0, 1.5, 2.0]$)

The canonical SSR formula uses $T=1.0$. We test whether temperature shifts distort persona rankings:
- **Low Temperature ($T=0.5$)**: Sharpens the distribution toward the most similar anchor.
  - P1-PE shifts from $3.14 \to 3.25$
  - P4-PE shifts from $2.44 \to 2.34$
  - **Ranking Invariance**: Under $T=0.5$, the PE gradient remains strictly **P1 (3.25) > P2 (2.60) > P3 (2.58) > P4 (2.34)**.
- **High Temperature ($T=2.0$)**: Flattens probabilities toward uniform distribution.
  - Expected ratings across all personas contract toward the neutral center ($pprox 3.0$).
- **Conclusion**: The choice of $T=1.0$ is methodologically sound; temperature scaling does not invert persona orderings.

---

## 3. Embedding Model Discussion: `nomic-embed-text` vs. `text-embedding-3-small`

1. **Dimensionality & Geometry**:
   - `nomic-embed-text` produces 768-dimensional embeddings trained on modern open benchmarks.
   - `text-embedding-3-small` produces 1536-dimensional embeddings.
   - While absolute cosine similarities between different models cannot be directly compared, the minimum-subtraction normalisation in canonical SSR subtracts the lowest similarity, making PMFs invariant to global similarity offsets.
2. **Local Reproducibility**:
   - Using local `nomic-embed-text` eliminates API downtime, rate limits, pricing changes, and privacy risks associated with external clouds.
