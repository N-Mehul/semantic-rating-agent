# Comprehensive Sentiment and Review Text vs. Likert Rating Correlation Analysis

## 1. Executive Summary

This study provides an exhaustive statistical analysis examining the relationship between **Sentiment**, **Review Text Embeddings**, and **Numeric Likert Ratings (1–5)** across **10,003 mobile review records** from `data/actual_reviews.xlsx` and `data/predictions.csv`.

### Key Metric Summary

| Relationship / Model | Spearman $\rho$ | $p$-value | Pearson $r$ | MAE | RMSE | $R^2$ | Relationship Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Actual Sentiment $\leftrightarrow$ Actual Rating** | **+0.7805** | $< 10^{-300}$ | +0.7757 | — | — | — | **Very Strong Positive** |
| **Predicted Sentiment $\leftrightarrow$ Actual Rating** | **+0.7774** | $< 10^{-300}$ | +0.7706 | — | — | — | **Very Strong Positive** |
| **Review Text $\leftrightarrow$ Rating (Agent Expected Rating)** | **+0.6997** | $< 10^{-300}$ | +0.7817 | 0.6293 | 0.7870 | 0.6106 | **Strong Positive (Nonlinear)** |
| **Review Text $\leftrightarrow$ Rating (5-Fold CV Ridge on Embeddings)** | **+0.7056** | $< 10^{-300}$ | +0.7806 | 0.6299 | 0.7882 | 0.6094 | **Strong Linear Generalization** |

---

## 2. Dataset & Methodology

* **Dataset Source**: `data/actual_reviews.xlsx` (Ground truth) & `data/predictions.csv` (Agent outputs)
* **Dataset Size**: **10,003 total evaluation records** ($100\%$ matched on review text alignment)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors)
* **Ordinal Sentiment Mapping** (strictly for correlation testing):
  * $\text{Negative} = -1$
  * $\text{Neutral} = 0$
  * $\text{Positive} = +1$
* **Statistical Tests**:
  * **Spearman Rank Correlation ($\rho$)**: Monotonic relationship assessment between ranked variables.
  * **Pearson Correlation ($r$)**: Linear relationship assessment.
  * **Evaluation Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$).
  * **Cross-Validation**: 5-Fold cross-validation on dense sentence embeddings to ensure strictly zero train/test data leakage.

---

## 3. Analysis A: Actual Sentiment vs. Actual Rating

### Descriptive Statistics by Actual Sentiment Class

| Sentiment Class | Code | Count ($N$) | Percentage (%) | Mean Rating | Median | Std Dev ($\sigma$) | IQR | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Negative** | -1 | 2,010 | 20.09% | **1.59** | 1.0 | 0.68 | 1.0 | 1.0 | 4.0 |
| **Neutral** | 0 | 2,516 | 25.15% | **2.47** | 2.0 | 0.83 | 1.0 | 1.0 | 5.0 |
| **Positive** | +1 | 5,477 | 54.75% | **3.97** | 4.0 | 0.80 | 2.0 | 1.0 | 5.0 |
| **Overall Dataset** | — | 10,003 | 100.00% | **3.12** | 3.0 | 1.26 | 2.0 | 1.0 | 5.0 |

### Crosstab: Actual Sentiment $\times$ Actual Rating

#### Counts:
| Actual Sentiment | Rating 1 | Rating 2 | Rating 3 | Rating 4 | Rating 5 | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Negative** | 1,047 | 757 | 196 | 10 | 0 | 2,010 |
| **Neutral** | 268 | 1,053 | 942 | 241 | 12 | 2,516 |
| **Positive** | 5 | 179 | 1,274 | 2,534 | 1,485 | 5,477 |
| **All** | **1,320** | **1,989** | **2,412** | **2,785** | **1,497** | **10,003** |

#### Row Proportions (%):
| Actual Sentiment | Rating 1 (%) | Rating 2 (%) | Rating 3 (%) | Rating 4 (%) | Rating 5 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Negative** | 52.09% | 37.66% | 9.75% | 0.50% | 0.00% |
| **Neutral** | 10.65% | 41.85% | 37.44% | 9.58% | 0.48% |
| **Positive** | 0.09% | 3.27% | 23.26% | 46.27% | 27.11% |

### Statistical Findings for Part A:
1. **Strong Monotonic Alignment ($\rho = 0.7805, p < 10^{-300}$)**:
   * Higher sentiment is strongly associated with higher numeric ratings.
2. **Distinct Class Means**:
   * Negative reviews average **$1.59 \pm 0.68$** stars (89.75% are 1–2 stars).
   * Neutral reviews average **$2.47 \pm 0.83$** stars (79.29% are 2–3 stars).
   * Positive reviews average **$3.97 \pm 0.80$** stars (73.38% are 4–5 stars).
3. **Multimodality of Neutral and Rating 3**:
   * Rating 3 is shared across Positive (1,274 reviews) and Neutral (942 reviews), explaining why 3-class sentiment alone cannot resolve finer Likert distinctions.

---

## 4. Analysis B: Predicted Sentiment vs. Actual Rating

The model predicts sentiment via cosine similarity with class centroids ($98.74\%$ accuracy, 9,877 / 10,003 correct).

### Descriptive Statistics by Predicted Sentiment Class

| Predicted Sentiment | Code | Count ($N$) | Percentage (%) | Mean Rating | Median | Std Dev ($\sigma$) | IQR | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Negative** | -1 | 2,054 | 20.53% | **1.62** | 1.0 | 0.71 | 1.0 | 1.0 | 5.0 |
| **Neutral** | 0 | 2,472 | 24.71% | **2.46** | 2.0 | 0.83 | 1.0 | 1.0 | 5.0 |
| **Positive** | +1 | 5,477 | 54.75% | **3.97** | 4.0 | 0.80 | 2.0 | 1.0 | 5.0 |

### Crosstab: Predicted Sentiment $\times$ Actual Rating

#### Counts:
| Predicted Sentiment | Rating 1 | Rating 2 | Rating 3 | Rating 4 | Rating 5 | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Negative** | 1,032 | 781 | 227 | 13 | 1 | 2,054 |
| **Neutral** | 283 | 1,029 | 911 | 238 | 11 | 2,472 |
| **Positive** | 5 | 179 | 1,274 | 2,534 | 1,485 | 5,477 |
| **All** | **1,320** | **1,989** | **2,412** | **2,785** | **1,497** | **10,003** |

### Statistical Comparison (Actual vs. Predicted):
* **Spearman $\rho$**: $0.7805$ (Actual) vs. $0.7774$ (Predicted) — **$\Delta = -0.0031$ ($99.6\%$ fidelity retained)**.
* **Pearson $r$**: $0.7757$ (Actual) vs. $0.7706$ (Predicted) — **$\Delta = -0.0051$**.
* The 126 centroid boundary errors (misclassified between Negative and Neutral) introduce virtually zero degradation in the underlying rating correlation.

---

## 5. Analysis C: Review Text vs. Actual Rating

We evaluated two distinct, leak-free methods to quantify how well the raw text semantics relate to numeric ratings:

### Approach 1: Agent Semantic Expected Rating (Cosine Retrieval over Example Bank)
* Derived continuously from softmax-weighted k-nearest neighbor retrieval on sentence embeddings.
* **Spearman Correlation ($\rho$)**: **$+0.6997$** ($p < 10^{-300}$)
* **Pearson Correlation ($r$)**: **$+0.7817$** ($p < 10^{-300}$)
* **MAE**: **$0.6293$**
* **RMSE**: **$0.7870$**
* **$R^2$**: **$0.6106$** (61.06% of rating variance explained purely by text embeddings)

### Approach 2: 5-Fold Cross-Validated Ridge Regression on Sentence Embeddings
* 384-dimensional dense sentence embeddings trained with Ridge Regularization across 5 out-of-fold validation splits.
* **Spearman Correlation ($\rho$)**: **$+0.7056$** ($p < 10^{-300}$)
* **Pearson Correlation ($r$)**: **$+0.7806$** ($p < 10^{-300}$)
* **MAE**: **$0.6299$**
* **RMSE**: **$0.7882$**
* **$R^2$**: **$0.6094$** (60.94% of rating variance explained)

---

## 6. Visualizations

### 1. Sentiment vs Average Rating
![Sentiment vs Average Rating](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/plot1_sentiment_vs_avg_rating.png)

### 2. Rating Distribution by Sentiment
![Rating Distribution by Sentiment](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/plot2_sentiment_rating_distribution.png)

### 3. Sentiment vs Rating Relationship & Quartiles
![Sentiment vs Rating Quartiles](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/plot3_sentiment_rating_relationship.png)

### 4. Review Text Semantic Score vs Actual Rating
![Semantic Score vs Actual Rating](file:///c:/Users/mehul/OneDrive%20-%20Shri%20Vile%20Parle%20Kelavani%20Mandal/Capstone/semantic_rating_agent/data/plot4_semantic_score_vs_rating.png)

---

## 7. Simple Language Explanation of Metrics

1. **Spearman Correlation ($\rho \approx 0.78$ for Sentiment, $\rho \approx 0.71$ for Text)**:
   * **What it means**: Spearman measures whether higher sentiment or more positive text reliably leads to higher star ratings. A value of $+0.78$ indicates a very strong, consistent ranking relationship: as review tone improves, star rating increases in lockstep.
2. **Pearson Correlation ($r \approx 0.78$)**:
   * **What it means**: Measures straight-line linear proportionality. A value near $+0.78$ confirms that text semantic scores scale smoothly and linearly with star ratings.
3. **$p$-value ($< 10^{-300}$)**:
   * **What it means**: The probability that this strong relationship happened by pure random chance is essentially zero. The finding is statistically irrefutable.
4. **MAE ($0.63$)**:
   * **What it means**: On average, the text-derived rating prediction misses the customer's true rating by just **$0.63$ of a star** on a 1–5 scale.
5. **RMSE ($0.79$)**:
   * **What it means**: Similar to MAE, but penalizes rare large mistakes more heavily. An RMSE of $0.79$ confirms that large prediction misses are uncommon.
6. **$R^2$ ($0.61$ / $61.0\%$)**:
   * **What it means**: $61\%$ of all variation in 1-to-5 star ratings is directly explained by the semantic content of the review text alone. The remaining $39\%$ is due to human rater subjectivity and unstated personal expectations.

---

## 8. Limitations & Final Conclusion

### Limitations:
* **Subjectivity Variance**: Multiple users writing identical reviews rate products differently (e.g. one user gives 4 stars, another gives 5 stars for the exact same praise).
* **3-Point Sentiment vs 5-Point Likert Resolution**: Discrete sentiment (-1, 0, +1) lumps Rating 4 and Rating 5 together as "Positive", making continuous text embeddings necessary for fine-grained rating estimation.

### Final Conclusion:
* **Sentiment and Rating are heavily correlated ($\rho = 0.7805$)**, but sentiment is a coarse 3-class summary.
* **Dense review embeddings capture continuous nuances ($\rho = 0.7056, R^2 = 0.6106, \text{MAE} = 0.63$)**, providing the optimal basis for calibrated rating predictions.
