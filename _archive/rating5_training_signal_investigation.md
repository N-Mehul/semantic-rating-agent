# Investigation Report: Why Does the Model Never Predict Rating 5?

## 1. Executive Summary & Core Conclusion

### **CLASSIFICATION: DO NOT CHANGE (Mathematically Correct Mode on Current Dataset)**

This read-only investigation reveals why the semantic rating model predicts zero Rating 5s in discrete point mode ($\arg\max$):

1. **Ground Truth Data Reality**:
   * The dataset contains 10,003 records spanning only **110 unique review texts**.
   * Among the **40 unique positive review texts**, **100.0% of them (40/40)** were rated across multiple contradictory star levels by human annotators (spanning 3★, 4★, and 5★).
   * In **100.0% of these 40 positive texts**, human raters chose **4★ more frequently than 5★** (typically $44\%–55\%$ chose 4★, while only $20\%–29\%$ chose 5★).
   * **There is not a single review text in the entire dataset where human raters chose 5★ more frequently than 4★.**
2. **Mathematical Mode Selection**:
   * In point prediction mode under discrete 0-1 loss, the optimal decision rule to maximize exact accuracy is the Bayes mode: $\hat{y}(x) = \arg\max_{r} P(r \mid x)$.
   * Because $P(4 \mid x) > P(5 \mid x)$ for every positive text, the mode is always **4**.
3. **Empirical Degradation of Forcing Rating 5**:
   * Offline Bayesian prior correction (Experiment B) produced 4,003 Rating 5 predictions, but achieved only **$27.25\%$ precision** (nearly 3 out of 4 were false positives). Exact accuracy **plummeted from $46.14\%$ to $38.81\%$**, and MAE degraded from $0.6055$ to $0.7858$.
   * Strict thresholding ($P_5 \ge 0.28$) produced 438 Rating 5 predictions with only **$26.48\%$ precision** and degraded accuracy.
4. **The Role of the Uncertainty Layer**:
   * The newly implemented **Uncertainty Layer** is the exact, proper solution: it does not corrupt the argmax point accuracy, but exposes $P(5)$ directly (averaging **$26.07\%$** on positive reviews, max **$29.06\%$**) and provides 80% credible prediction intervals where **$54.75\%$ of all reviews ($5,477$ records)** span $[3, 5]$ or $[4, 5]$.

---

## 2. Training Data Inspection

| Metric | Measured Value | Percentage / Notes |
| :--- | :---: | :---: |
| **Total Training Records** | 10,003 rows | `data/Mobile Reviews Sentiment.csv` |
| **Unique Review Texts** | 110 unique texts | Average 90.9 repetitions per text |
| **Rating 1 Examples** | 1,320 rows | **13.20%** |
| **Rating 2 Examples** | 1,989 rows | **19.88%** |
| **Rating 3 Examples** | 2,412 rows | **24.11%** |
| **Rating 4 Examples** | 2,785 rows | **27.84%** |
| **Rating 5 Examples** | 1,497 rows | **14.97%** |
| **Unique Texts with Rating 5** | 40 unique texts | 100% categorized under Positive sentiment |
| **Unique Texts with Rating 4** | 70 unique texts | 40 positive texts + 30 neutral/mixed texts |
| **Text Overlap (Rating 4 vs Rating 5)** | **40 / 40 texts** | **100.00% overlap** |

> **Key Finding**: Rating 5 examples exist in the dataset (1,497 occurrences across 40 unique texts), but every single text associated with Rating 5 was also assigned Rating 4 and Rating 3 by human annotators.

---

## 3. Rating Bank Structure (`example_bank` in `memory.json`)

The exemplar bank contains 110 unique text entries. For each entry, it stores the empirical rating distribution computed across all identical occurrences.

### Dominant Rating Distribution Across 110 Exemplars:
* **Dominant Rating 1**: **26 exemplars (23.64%)**
* **Dominant Rating 2**: **22 exemplars (20.00%)**
* **Dominant Rating 3**: **22 exemplars (20.00%)**
* **Dominant Rating 4**: **40 exemplars (36.36%)**
* **Dominant Rating 5**: **0 exemplars (0.00%)**

### Mean Rating Probability Inside Exemplar Bank:
* $P(\text{Rating } 1) = 19.55\%$
* $P(\text{Rating } 2) = 19.97\%$
* $P(\text{Rating } 3) = 18.06\%$
* $P(\text{Rating } 4) = 27.93\%$
* $P(\text{Rating } 5) = \mathbf{14.49\%}$ (averages $\mathbf{26.07\%}$ across positive exemplars)

> **Why are there 0 exemplars with dominant rating 5?**
> Because for every positive review text in the dataset, human raters voted for Rating 4 more frequently than Rating 5 ($44\%–55\%$ vs $20\%–29\%$). Hence, the cluster mode for all positive exemplars is 4.

---

## 4. Rating Similarity & Probability Score Analysis

| Metric | Probability for Rating 4 ($P_4$) | Probability for Rating 5 ($P_5$) | Gap ($P_4 - P_5$) |
| :--- | :---: | :---: | :---: |
| **Mean Across All Reviews** | $27.93\%$ | $14.49\%$ | $+13.44\%$ |
| **Mean on Positive Reviews** | $46.12\%$ | $26.07\%$ | $+20.05\%$ |
| **Mean on True 5★ Reviews** | $45.72\%$ | $26.74\%$ | $+18.98\%$ |
| **Min Gap on Positive Reviews** | $43.51\%$ | $29.06\%$ | $\mathbf{+14.45\%}$ |
| **Max $P_5$ on Any Review** | — | $\mathbf{29.06\%}$ | — |
| **Reviews Where $P_5 > P_4$** | — | — | **0 / 10,003 (0.00%)** |

> **Key Finding**: In every positive review, $P_4$ ranges between $44\%–47\%$, while $P_5$ ranges between $24\%–29\%$. $P_4$ strictly dominates $P_5$ for all reviews without exception.

---

## 5. Top 20 Candidate Reviews with Highest $P(\text{Rating } 5)$

Below are the 20 evaluation review texts that received the highest Rating 5 probability from the model:

| # | True Rating | Model Point Pred | $P_1$ | $P_2$ | $P_3$ | $P_4$ | $P_5$ | $P_5 / P_4$ | Review Text Snippet |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | 5★ | **4★** | 0.0% | 2.5% | 23.7% | 44.7% | **29.1%** | 0.65 | `"Battery easily lasts a day with heavy use. No regrets..."` |
| **2** | 4★ | **4★** | 0.0% | 2.5% | 24.3% | 44.4% | **28.7%** | 0.65 | `"Battery easily lasts a day with heavy use. Best purchase..."` |
| **3** | 4★ | **4★** | 0.0% | 2.5% | 24.1% | 44.9% | **28.4%** | 0.63 | `"Battery easily lasts a day with heavy use. Loving it so..."` |
| **4** | 5★ | **4★** | 0.0% | 2.7% | 24.3% | 45.1% | **27.9%** | 0.62 | `"Battery easily lasts a day with heavy use. Absolutely..."` |
| **5** | 3★ | **4★** | 0.0% | 2.7% | 24.0% | 45.4% | **27.8%** | 0.61 | `"Worth every penny. Highly recommended! Best purchase..."` |
| **6** | 5★ | **4★** | 0.0% | 2.6% | 24.4% | 45.5% | **27.5%** | 0.60 | `"Display is gorgeous, colors pop nicely. Loving it so..."` |
| **7** | 5★ | **4★** | 0.0% | 2.8% | 24.4% | 45.4% | **27.4%** | 0.60 | `"Display is gorgeous, colors pop nicely. Absolutely..."` |
| **8** | 5★ | **4★** | 0.1% | 2.8% | 24.0% | 46.0% | **27.1%** | 0.59 | `"Worth every penny. Highly recommended! Loving it so..."` |
| **9** | 5★ | **4★** | 0.4% | 2.9% | 24.9% | 44.7% | **27.0%** | 0.60 | `"Worth every penny. Highly recommended! Absolutely..."` |
| **10** | 3★ | **4★** | 0.0% | 2.6% | 24.8% | 45.5% | **27.0%** | 0.59 | `"Smooth performance even after months of use. Best..."` |
| **11** | 4★ | **4★** | 0.0% | 3.2% | 24.6% | 45.2% | **27.0%** | 0.60 | `"Face unlock is instant, super smooth. Best purchase..."` |
| **12** | 5★ | **4★** | 0.0% | 2.9% | 24.0% | 46.2% | **26.8%** | 0.58 | `"Loving the clean UI and fast updates. Loving it so..."` |
| **13** | 4★ | **4★** | 0.0% | 3.2% | 24.2% | 46.0% | **26.6%** | 0.58 | `"Loving the clean UI and fast updates. Absolutely..."` |
| **14** | 4★ | **4★** | 0.1% | 3.4% | 23.8% | 46.1% | **26.6%** | 0.58 | `"Smooth performance even after months of use. Loving..."` |
| **15** | 3★ | **4★** | 0.1% | 3.2% | 25.2% | 45.1% | **26.5%** | 0.59 | `"Worth every penny. Highly recommended! No regrets..."` |
| **16** | 5★ | **4★** | 0.1% | 3.0% | 24.9% | 45.5% | **26.5%** | 0.58 | `"Face unlock is instant, super smooth. Absolutely..."` |
| **17** | 5★ | **4★** | 1.2% | 3.9% | 24.0% | 44.5% | **26.4%** | 0.59 | `"Fast charging is a lifesaver. Loving it so far..."` |
| **18** | 4★ | **4★** | 0.0% | 3.1% | 24.6% | 45.9% | **26.3%** | 0.57 | `"Face unlock is instant, super smooth. Loving it so..."` |
| **19** | 4★ | **4★** | 0.0% | 3.2% | 24.4% | 46.1% | **26.3%** | 0.57 | `"Face unlock is instant, super smooth. No regrets..."` |
| **20** | 5★ | **4★** | 0.0% | 2.9% | 24.6% | 46.3% | **26.2%** | 0.57 | `"Smooth performance even after months of use. Absol..."` |

---

## 6. Offline Prior Correction & Rebalancing Experiments

We tested principled statistical methods to adjust class priors:

$$\text{Prior}(1)=0.132, \quad \text{Prior}(2)=0.199, \quad \text{Prior}(3)=0.241, \quad \text{Prior}(4)=0.278, \quad \text{Prior}(5)=0.150$$

| Experiment Description | Exact Accuracy | Within $\pm 1$ Acc | MAE | Spearman $\rho$ | Predicted 5 Count | Rating 5 Precision | Rating 5 Recall | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp A: Production Baseline** | **46.14%** | **93.55%** | **0.6055** | **0.7755** | **0** | **0.00%** | **0.00%** | **Optimal Baseline Mode** |
| **Exp B: Bayesian Prior Div ($P / \text{Prior}$)** | 38.81% | 84.23% | 0.7858 | 0.7378 | 4,003 | **27.25%** | 72.88% | **Severe Degradation (-7.33% Acc)** |
| **Exp C: Moderate Prior ($P / \sqrt{\text{Prior}}$)** | 46.33% | 93.57% | 0.6037 | 0.7805 | 0 | 0.00% | 0.00% | Neutral (Still mode 4 for pos) |
| **Exp D: 4-vs-5 Prior Rebalance** | 38.62% | 84.21% | 0.7876 | 0.7335 | 4,003 | **27.25%** | 72.88% | **Severe Degradation (-7.52% Acc)** |

> **Conclusion**: Dividing by the class prior artificially forces 4,003 predictions into Rating 5, but because $72.75\%$ of those reviews are actually 4★ or 3★ in ground truth, it produces massive false-positive inflation and severely degrades overall model accuracy.

---

## 7. Decision Threshold Grid Search Simulations

We tested various threshold rules to observe whether a "safe" cutoff for Rating 5 exists:

| Threshold Rule | Pred 5 Count | True 5 Matches | Rating 5 Precision | Rating 5 Recall | Exact Accuracy | Within $\pm 1$ | MAE | Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (No R5)** | **0** | **0** | **0.00%** | **0.00%** | **46.14%** | **93.55%** | **0.6055** | **0.7755** |
| **$P_5 \ge 0.28$** | 438 | 116 | **26.48%** | 7.75% | 45.24% | 92.50% | 0.6261 | 0.7591 |
| **$P_5 \ge 0.27$** | 1,409 | 370 | **26.26%** | 24.72% | 43.22% | 90.17% | 0.6724 | 0.7341 |
| **$P_5 \ge 0.26$** | 3,304 | 897 | **27.15%** | 59.92% | 39.87% | 85.85% | 0.7565 | 0.7248 |
| **$P_5 \ge 0.25$** | 3,866 | 1,051 | **27.19%** | 70.21% | 38.79% | 84.55% | 0.7819 | 0.7315 |
| **$P_5 \ge 0.24$** | 5,214 | 1,421 | **27.25%** | 94.92% | 36.30% | 81.46% | 0.8426 | 0.7655 |
| **$P_5 / P_4 \ge 0.60$** | 856 | 215 | **25.12%** | 14.36% | 44.17% | 91.51% | 0.6481 | 0.7454 |
| **$P_5 / P_4 \ge 0.55$** | 3,734 | 1,015 | **27.18%** | 67.80% | 39.18% | 84.74% | 0.7759 | 0.7283 |

> **Critical Observation**: Across all threshold rules, **Rating 5 precision never exceeds $27.25\%$**. Every rule that predicts Rating 5 incurs approximately **$73\%$ false positives**, dragging down exact accuracy and worsening MAE.

---

## 8. Dataset Limitation: Human Rater Inconsistency on Identical Texts

Why is Rating 5 precision capped at $\approx 27\%$?

In the dataset:
* There are only 40 unique positive texts.
* For each positive text, identical wording was evaluated by $\approx 140–160$ different human raters.
* Example text: `"Battery easily lasts a day with heavy use. No regrets buying this."`
  * True human ratings: **4★: 75 raters (44.4%)**, **5★: 48 raters (28.4%)**, **3★: 41 raters (24.3%)**, **2★: 5 raters (2.9%)**.
  * If the model predicts **4★**: it is correct for 75 people ($44.4\%$) and within $\pm 1$ for 164 people ($97.0\%$).
  * If the model forces **5★**: it is correct for only 48 people ($28.4\%$), incorrect for 121 people ($71.6\%$), and incurs larger errors against 3★ and 2★ raters.

Because human ratings for the exact same text have high entropy ($\sigma \approx 0.8$ stars), the single point value that minimizes expected error is the distribution mode: **Rating 4**.

---

## 9. Is Additional Training Data Necessary?

Evaluating the 4 structural possibilities:

| Possibility | Evaluation on Current Dataset | Assessment |
| :--- | :--- | :--- |
| **A. Existing data sufficient, logic suppresses 5** | FALSE. The decision logic correctly chooses the mode ($P_4 > P_5$). | Model is mathematically optimal. |
| **B. Rating 5 training examples are too few** | PARTIALLY TRUE. Rating 5 accounts for 15.0% vs Rating 4 at 27.8%. | Imbalance exists, but not the primary cause. |
| **C. Rating 5 semantically indistinguishable from 4** | **TRUE**. Identical text phrases represent both 4★ and 5★. | Pure text semantics cannot separate 4 vs 5. |
| **D. Duplicated texts create fundamental ambiguity** | **TRUE**. 110 unique texts across 10,003 rows. | Bayes ceiling limits discrete point accuracy. |

### Would More Training Data Help?
* **More duplicates of the existing 110 texts**: **NO**. Repeating the same 110 texts will not change the empirical distribution ($P(4) \approx 46\% > P(5) \approx 26\%$).
* **New, unique, highly specific 5-star texts**: **YES**. If a new dataset is curated containing distinct phrases associated exclusively with 5★ ratings (e.g., *"Absolute perfection, 10/10, flawless in every single way, zero flaws"* where $P(5) > 90\%$), the model would naturally create new exemplars whose mode is 5, enabling genuine 5-star point predictions.

---

## 10. Final Recommendation & Decision

### **FINAL DECISION: DO NOT CHANGE POINT PREDICTION ENGINE**

1. **Do Not Introduce Rating 5 Thresholds or Overrides**:
   * Forcing Rating 5 point predictions on the current dataset decreases exact accuracy by up to $7.5\%$, inflates MAE by up to $+0.18$, and yields precision $< 28\%$.
2. **Rely on the Calibrated Uncertainty Layer**:
   * The uncertainty output layer already correctly and transparently reports the true state of knowledge:
     * Point Prediction: **4★** (the Bayes-optimal mode)
     * Rating Distribution: **`{"3": 0.24, "4": 0.46, "5": 0.27}`**
     * Credible Interval: **`[3, 5]` or `[4, 5]`**
     * Uncertainty Status: **`"ambiguous"`** (correctly signaling human-rater variance)
3. **Future Path for 5-Star Predictions**:
   * If point prediction of 5-star ratings is a strict business requirement, the upstream dataset must be augmented with genuinely distinct 5-star review texts that are not reused for 3-star and 4-star ratings.
