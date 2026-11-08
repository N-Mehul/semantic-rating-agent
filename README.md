# Semantic Rating Agent (SRA): Digital Persona + E-SSR

## Project Overview & Purpose
The **Semantic Rating Agent (SRA)** implements an empirical technology adoption simulation framework combining **Empirical Digital Personas** with **Enhanced Semantic Similarity Rating (E-SSR)**. 

Synthetic agents ("twins") conditioned on empirical survey respondents (persona profile, spending habits, current smartphone brand, price tier) answer multi-question open-ended feedback without ever outputting numeric ratings directly. Natural language answers are scored against validated psychometric anchor statements (Behavioral Intention) using embedding-based projection into continuous expected rating spaces ($\mathbb{E}[R] \in [1, 5]$).

### Core Components
- **Inference & Simulation**: Fully offline local execution via **Ollama**.
  - Text Generation: `qwen2.5:7b-instruct`
  - Embedding Projection: `nomic-embed-text` (768-dimensional embeddings)
- **Scoring Pipeline**:
  - **E-SSR**: Top-$k$ filtering ($k=3$), sharp softmax temperature scaling ($T=0.75$), projection onto 5 Likert anchor vectors.
  - **Calibrated E-SSR**: Affine variance and mean matching against held-out human response baselines ($z$-score re-scaling).
  - **Hybrid Twin Scoring**: Ensemble combining calibrated E-SSR with direct Likert elicitation baseline.

---

## Installation & Setup

1. **Clone & Set Up Virtual Environment (PowerShell / Windows)**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Ollama Setup & Models**:
   Ensure [Ollama](https://ollama.ai) is running locally (`http://localhost:11434`):
   ```powershell
   ollama pull qwen2.5:7b-instruct
   ollama pull nomic-embed-text
   ```

---

## Running the Interactive Dashboard
To launch the full 6-step interactive workflow demonstration:
```powershell
python dashboard\server.py
```
Open your browser at `http://127.0.0.1:8765` (or specify `--port 8766` if port 8765 is occupied).

The dashboard visualizes:
1. Persona selection and question prompt configuration.
2. Natural language synthetic response generation.
3. Batch sentence embedding via `nomic-embed-text`.
4. Cosine similarity calculation against psychometric anchor statements.
5. Probability mass function (PMF) computation with top-$k$ temperature scaling.
6. Continuous expected rating calculation and verification against reference baselines.

---

## Running Phase 19 Twin Validation

The Phase 19 validation pipeline generates paired digital twins for real survey respondents and evaluates group-level and individual-level psychometric alignment:

- **Pilot Validation (80 respondents, 20 per persona)**:
  ```powershell
  python evaluation\phase19_twin_validation\run_twin_validation.py --pilot 80
  ```

- **Analysis Only (re-score existing generated twins without re-running LLM)**:
  ```powershell
  python evaluation\phase19_twin_validation\run_twin_validation.py --pilot 80 --analyse-only
  ```

- **Held-Out Test Set (40 completely unseen respondents, stratified by persona)**:
  ```powershell
  python evaluation\phase19_twin_validation\run_twin_validation.py --heldout 40
  ```

*Note: Generation is fully resumable; already generated twins in JSONL are skipped automatically on re-run.*

---

## Results Summary (Placeholder)

| Experiment / Evaluation | Group Pearson ($r$) | Group Spearman ($\rho$) | Distribution $W_1$ | Individual Exact Acc | Individual Within-1 | Individual QWK |
|---|---|---|---|---|---|---|
| **Phase 19 Pilot (N=80) — E-SSR** | 0.76 (persona) / 0.93 (brand) | 0.80 (persona) / 0.50 (brand) | 0.522 | 47.5% | 88.8% | 0.363 |
| **Phase 19 Pilot (N=80) — Calibrated E-SSR** | 0.76 (persona) / 0.93 (brand) | 0.80 (persona) / 0.50 (brand) | 0.329 | 40.0% | 82.5% | 0.376 |
| **Phase 19 Pilot (N=80) — Hybrid (E-SSR + DLE)** | 0.86 (persona) / 0.99 (brand) | 1.00 (persona) / 1.00 (brand) | 0.346 | 53.8% | 88.8% | 0.461 |
| **Phase 19 Held-Out (N=40) — Confirmation** | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* | *In Progress* |

---

## Legacy notes

## Default Showcase Result: D1 Aspect-Based Random Forest

The default model to showcase in project summaries and demonstrations is the **D1 Random Forest using five aspect-rating inputs** (`battery_life_rating`, `camera_rating`, `performance_rating`, `design_rating`, `display_rating`). Its verified evaluation metrics from [data/aspect_model/metrics.json](data/aspect_model/metrics.json) are:

| Accuracy | Macro F1 | Weighted F1 | MAE | RMSE | Spearman |
|---:|---:|---:|---:|---:|---:|
| 73.34% | 73.70% | 73.32% | 0.2680 | 0.5204 | 0.9166 |

**Interpretation:** This is an aspect-based rating model, not a text-only model and not a result from the canonical five-anchor SSR pipeline. The five aspect ratings are co-assigned sub-target inputs and carry potential target leakage; the saved metrics label this model diagnostic-only. Present the result with that qualification. The existing 46.14% text-based baseline remains a separate historical comparator, not the default showcase result. Its saved baseline artifacts are unchanged.

## 1. Project Title
**Semantic Similarity Rating (SSR) for LLM-Based Synthetic Consumer Survey Responses**

---

## 2. Project Overview
This repository explores **Semantic Similarity Rating (SSR)** and related rating models. For current project-result demonstrations, the primary showcase is the D1 aspect-based Random Forest described above; the repository's canonical SSR design and the D1 model are distinct methods.

Rather than asking an LLM to directly output a numerical integer (e.g., `"Rate this product from 1 to 5"`), the SSR paradigm instructs the LLM persona to express its qualitative opinion naturally in free-form text. The generated natural-language response is embedded into a dense semantic vector space using a sentence-transformer model and compared via cosine similarity against **five standardized Likert anchor statements** (representing ratings 1 through 5). The resulting similarity scores are converted via temperature-scaled softmax into a discrete probability distribution, from which a continuous **Expected Likert Score** \(\mathbb{E}[\text{Rating}] \in [1.0, 5.0]\) and a rounded discrete Likert rating are mathematically derived.

> **Note on Project Focus:** The dataset utilized for empirical validation consists of mobile phone consumer reviews (`Mobile Reviews Sentiment.csv`). However, **the primary research objective is NOT simple mobile review sentiment analysis**. The core contribution is evaluating whether vector-space semantic anchor projection solves the structural biases (such as scale compression, central tendency bias, and extreme digit hallucination) inherent in LLM-generated survey ratings.

---

## 3. Problem Statement
In market research and social science simulations, synthetic consumer panels built using Large Language Models (LLMs) are increasingly used to pilot survey questionnaires. However, asking an LLM to output a direct numerical Likert score (e.g., `1, 2, 3, 4, 5`) leads to critical statistical distortions:
* **Digit Bias & Compression:** LLMs frequently overuse extreme values (e.g., over-indexing on `4` or `5`) or collapse toward central options (`3`), underutilizing mid-tier negative scale steps.
* **Lack of Continuous Variance:** Discrete integer outputs fail to capture nuanced consumer sentiment intensity or uncertainty.
* **Prompt Sensitivity:** Small changes in prompt phrasing skew integer output distributions unpredictably without altering underlying preference representations.

---

## 4. Proposed Solution — Semantic Similarity Rating (SSR)
The SSR framework decouples **qualitative preference reasoning** from **quantitative rating assignment**:
1. **Persona Prompting:** The LLM generates a natural-language narrative review or response matching consumer demographic/psychographic attributes.
2. **Semantic Vector Mapping:** The narrative response is projected into an embedding space.
3. **Anchor Projection:** The response embedding is compared against five fixed, mathematically grounded semantic anchor statements (representing 1 to 5 stars/points).
4. **Probabilistic Scoring:** Cosine similarities are transformed into a probability distribution and integrated into a smooth expected value \(\mathbb{E}[\text{Rating}]\).

---

## 5. Why Direct Likert Rating from LLMs Is a Problem
When an LLM directly outputs a Likert digit:
* It treats Likert choices as categorical text tokens rather than ordinal interval values.
* Token logits are strongly biased by pre-training word frequency patterns rather than fine-grained sentiment intensity.
* It prevents calculating fine-grained variance, standard error, or smooth probability distributions across response categories.

---

## 6. Synthetic Consumer / Persona
The synthetic consumer system constructs persona prompts incorporating consumer profiles, behavioral constraints, and domain metadata (e.g., price sensitivity, brand loyalty, tech savviness). In `agent.py`, the `MemorySystem` dynamically selects relevant persona exemplars to condition the LLM narrative response generation.

---

## 7. Natural-Language Response Generation
Instead of forcing a single digit, the synthetic consumer receives a prompt such as:
> *"Describe your complete user experience with this product, detailing performance, build quality, value for money, and key grievances or highlights."*

The LLM outputs a rich, unstructured qualitative response \(R\).

---

## 8. Five Likert Anchor Statements
The system maintains five standardized semantic anchor statements representing each step of a 5-point Likert scale:

| Rating Level | Anchor Statement Definition |
| :---: | :--- |
| **1 (Terrible)** | *"Extremely dissatisfied. Product failed completely, unusable quality, terrible experience."* |
| **2 (Poor)** | *"Dissatisfied. Multiple notable flaws, poor value for money, below expectations."* |
| **3 (Average)** | *"Neutral experience. Meets basic expectations, acceptable but uninspiring, average performance."* |
| **4 (Good)** | *"Satisfied. High quality overall, minor issues, solid performance, good value."* |
| **5 (Excellent)** | *"Extremely satisfied. Outstanding quality, exceeds expectations in every aspect, highly recommended."* |

---

## 9. Embeddings and Semantic Similarity
The response text \(R\) and each anchor statement \(A_k\) (\(k \in \{1, 2, 3, 4, 5\}\)) are embedded using a SentenceTransformer model (`all-MiniLM-L6-v2` by default):

\[
v_R = \text{Embed}(R), \quad v_{A_k} = \text{Embed}(A_k)
\]

Cosine similarity \(s_k\) is computed between the response vector and each anchor vector:

\[
s_k = \cos(v_R, v_{A_k}) = \frac{v_R \cdot v_{A_k}}{\|v_R\| \|v_{A_k}\|}
\]

---

## 10. Probability Distribution
Raw similarity scores \(s = [s_1, s_2, s_3, s_4, s_5]\) are normalized into a proper probability distribution \(P(k)\) using a temperature-scaled Softmax function (\(\tau > 0\)):

\[
P(\text{Likert} = k) = \frac{\exp(s_k / \tau)}{\sum_{j=1}^5 \exp(s_j / \tau)}
\]

Additionally, normalized Shannon Entropy \(H \in [0, 1]\) measures rating uncertainty:

\[
H = \frac{-\sum_{k=1}^5 P(k) \log_2 P(k)}{\log_2(5)}
\]

---

## 11. Expected Likert Score
The continuous Expected Likert Score \(\mathbb{E}[\text{Rating}]\) is mathematically calculated as:

\[
\mathbb{E}[\text{Rating}] = \sum_{k=1}^5 k \cdot P(\text{Likert} = k)
\]

The discrete predicted rating is assigned as:

\[
\hat{y} = \arg\max_{k} P(\text{Likert} = k) \quad \text{or} \quad \text{round}(\mathbb{E}[\text{Rating}])
\]

---

## 12. End-to-End Architecture

```text
Survey / Product / Consumer Context
        ↓
Synthetic Consumer Persona Prompting
        ↓
LLM Narrative Generation Engine
        ↓
Natural-Language Response (Qualitative Text)
        ↓
SentenceTransformer Embedding Model (all-MiniLM-L6-v2)
        ↓
Five Standardized Likert Anchor Vectors (1..5)
        ↓
Cosine / Semantic Similarity Computation
        ↓
Raw Vector Similarity Scores [s₁, s₂, s₃, s₄, s₅]
        ↓
Temperature-Scaled Softmax Normalization
        ↓
Probability Distribution P(Likert = k)
        ↓
Expected Likert Score E[Rating] & Entropy (Uncertainty)
        ↓
Evaluation Against Human Survey/Review Ground Truth
```

---

## 13. Agent Architecture
Implemented primarily in `agent.py`, the core agent consists of decoupled modules:
* `SentenceTransformerEmbeddings`: Handles vector encoding via PyTorch/HuggingFace models.
* `LikertAnchorBank`: Stores, embeds, and computes similarity against standard 5-point Likert scale anchors.
* `MemorySystem`: Manages JSON vector storage (`memory.json`), dynamic persona retrieval, and aspect exemplars.
* `LocalLLM`: Interfacing wrapper supporting local LLM inference engines (e.g., Ollama) or deterministic fallback reasoning.
* `SemanticRatingAgent`: Master orchestrator unifying data processing, semantic search, rating prediction, calibration, and interactive CLI sessions.

---

## 14. Dataset
Empirical evaluation is performed on mobile phone consumer reviews:
* **Primary Raw Dataset:** `data/Mobile Reviews Sentiment.csv` (~10,000 records containing review text, ground-truth integer ratings 1–5, actual sentiment, and specs).
* **Validation Datasets:** `data/actual_reviews.xlsx` and `data/reviews_only.csv`.

---

## 15. Current Implementation
The current production pipeline consists of:
* **`agent.py`**: Complete SSR agent engine.
* **`main.py`**: Interactive CLI for ad-hoc review rating and dataset querying.
* **`batch_predict.py`**: Multi-threaded/parallel batch runner generating predictions over review sets (`data/predictions.csv`).
* **`evaluate_predictions.py`**: Evaluation engine computing MAE, RMSE, Pearson/Spearman correlation, within-±1 accuracy, and confusion matrices.

---

## 16. Experiments
Standalone research experiment scripts present in the repository:
* `embedding_comparison_experiment.py`: Benchmarks `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, and `bge-base-en-v1.5`.
* `model_comparison_experiment.py`: Evaluates classical ML vs. neural classifiers on text embeddings.
* `optimized_model_comparison_experiment.py`: Evaluates tuned models (Logistic Regression, Ridge, SVM, MLP, XGBoost, LightGBM, CatBoost, Ordinal Regression).
* `safe_feature_model_comparison_experiment.py`: Strict leak-free feature evaluation ensuring text-only representation.
* `xgboost_batch_experiment.py`: Runs standalone XGBoost baseline benchmark.
* `correlation_results.py`: Computes cross-variable correlation metrics.

---

## 17. Evaluation Metrics
* **Exact Accuracy (%)**: Percentage of predicted discrete ratings matching ground truth exactly.
* **Within-±1 Accuracy (%)**: Percentage of predictions within 1 rating point of ground truth.
* **Mean Absolute Error (MAE)**: Average absolute error \(\frac{1}{N} \sum |y - \hat{y}|\).
* **Root Mean Squared Error (RMSE)**: Square root of mean squared error \(\sqrt{\frac{1}{N} \sum (y - \hat{y})^2}\).
* **Spearman Rank Correlation (\(\rho\))**: Monotonic correlation between predicted and actual ratings.
* **Pearson Correlation (\(r\))**: Linear correlation between continuous expected ratings and actual ratings.

---

## 18. Research / Investigation Work
Documented research findings in markdown reports:
* `target_leakage_audit_report.md`: Verifies strict isolation between input review text and target ratings.
* `rating_calibration_investigation.md`: Details non-linear score calibration to adjust scale boundary contraction.
* `rating_uncertainty_implementation.md`: Formulates normalized entropy calculation for rating confidence bounds.
* `sentiment_rating_correlation_analysis.md`: Analyzes statistical coupling between sentiment labels and rating scales.
* `text_to_aspect_rating_investigation.md`: Investigates multi-aspect qualitative breakdown.
* `persona_rating5_investigation.md` & `rating5_training_signal_investigation.md`: Analyzes extreme rating frequency bias.

---

## 19. Project Structure

```text
semantic_rating_agent/
├── .gitignore
├── README.md
├── requirements.txt
├── agent.py
├── main.py
├── batch_predict.py
├── evaluate_predictions.py
├── correlation_results.py
├── embedding_comparison_experiment.py
├── model_comparison_experiment.py
├── optimized_model_comparison_experiment.py
├── safe_feature_model_comparison_experiment.py
├── xgboost_batch_experiment.py
├── persona_rating5_investigation.md
├── rating5_training_signal_investigation.md
├── rating_calibration_investigation.md
├── rating_intensity_investigation.md
├── rating_uncertainty_implementation.md
├── sentiment_rating_correlation_analysis.md
├── sentiment_rating_investigation.md
├── target_leakage_audit_report.md
├── text_to_aspect_rating_investigation.md
├── memory.json
└── data/
    ├── Mobile Reviews Sentiment.csv
    ├── actual_reviews.xlsx
    ├── reviews_only.csv
    ├── correlation_metrics_summary.csv
    ├── embedding_comparison_results.csv
    ├── evaluation_results.csv
    ├── model_comparison_results.csv
    ├── optimized_model_comparison_results.csv
    ├── predictions.csv
    └── xgboost_evaluation_results.csv
```

---

## 20. Installation

```bash
# Clone repository
git clone https://github.com/your-username/semantic_rating_agent.git
cd semantic_rating_agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 21. How to Run

### Interactive CLI Session
```bash
python main.py
```

### Run Batch Prediction
```bash
python batch_predict.py --input "data/Mobile Reviews Sentiment.csv" --output "data/predictions.csv"
```

### Run Evaluation Metrics
```bash
python evaluate_predictions.py --predictions "data/predictions.csv" --actual "data/Mobile Reviews Sentiment.csv"
```

---

## 22. Example Input and Output

### Input Review Text
> *"The phone has an incredible camera and vibrant display, but the battery life barely lasts 6 hours and charging is frustratingly slow."*

### Agent SSR Output
* **Predicted Sentiment:** `Mixed / Neutral`
* **Raw Similarity Vector:** `[0.18, 0.42, 0.61, 0.58, 0.29]`
* **Likert Probability Distribution:**
  * Rating 1 (Terrible): `4.1%`
  * Rating 2 (Poor): `18.2%`
  * Rating 3 (Average): `38.5%`
  * Rating 4 (Good): `31.4%`
  * Rating 5 (Excellent): `7.8%`
* **Expected Likert Score \(\mathbb{E}[\text{Rating}]\):** `3.21 / 5.0`
* **Discrete Predicted Rating:** `3`
* **Rating Uncertainty (Entropy):** `0.74 (Moderate)`

---

## 23. Results — Verified Empirical Findings

### Core SSR Engine & Embedding Model Comparison (`data/embedding_comparison_results.csv`)

| Embedding Model | Exact Accuracy (%) | Within ±1 Acc (%) | MAE | RMSE | Spearman \(\rho\) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MiniLM (`all-MiniLM-L6-v2`)** | 36.79% | 82.82% | 0.6387 | 0.8005 | 0.6919 |
| **MPNet (`all-mpnet-base-v2`)** | 36.84% | 83.12% | 0.6393 | 0.8006 | 0.6910 |
| **BGE (`bge-base-en-v1.5`)** | 37.04% | 83.36% | 0.6396 | 0.8013 | 0.6923 |

### Linear & Machine Learning Baselines (`data/correlation_metrics_summary.csv` & `data/optimized_model_comparison_results.csv`)

| Method / Model | Pearson \(r\) | Spearman \(\rho\) | MAE | RMSE | Within ±1 Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SSR Expected Rating (MiniLM)** | **0.7817** | **0.6996** | **0.6293** | **0.7870** | 82.82% |
| **5-Fold CV Ridge Regression** | 0.7806 | 0.7056 | 0.6299 | 0.7882 | 95.31% |
| **Optimized XGBoost Baseline** | - | 0.6892 | 0.6359 | 0.7891 | 93.66% |
| **Actual Sentiment ↔ Rating Correlation** | 0.7757 | 0.7805 | - | - | - |

---

## 24. Limitations
* **Scale Edge Contraction:** Raw cosine similarities between text embeddings and anchor statements rarely yield extreme values near 0.0 or 1.0, pulling expected ratings toward central values (2.5–3.5) without temperature tuning or non-linear calibration.
* **Aspect Blur:** Multi-aspect reviews containing opposing aspect sentiments (e.g., great screen, terrible battery) collapse into an average score rather than preserving multi-dimensional ratings.

---

## 25. Future Scope
* **Aspect-Specific SSR:** Extending anchor projection to separate aspect dimensions (e.g., Performance Anchor, Battery Anchor, Price Anchor).
* **Dynamic Anchor Generation:** Automatically synthesizing context-aware Likert anchor statements tailored to specific survey domain contexts.
* **LLM Logit Integration:** Combining vector similarity probabilities with direct LLM token probability logits.

---

## 26. Technologies Used
* **Python 3.11**
* **Sentence-Transformers / HuggingFace Transformers**
* **PyTorch**
* **Scikit-Learn**
* **SciPy & NumPy**
* **Pandas**

---

## 27. Research Objective
To establish a statistically robust, continuous, and un-biased rating derivation mechanism for synthetic consumer survey simulations, bridging qualitative natural language generation and quantitative Likert survey analytics.

---

## 28. Author
**Capstone Research Project — Semantic Rating Agent Team**  
*Shri Vile Parle Kelavani Mandal (SVKM)*
