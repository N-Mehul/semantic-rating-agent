# Target Leakage Audit Report: Rating Prediction

**Dataset Analyzed**: `data/Mobile Reviews Sentiment.csv` (50,000 source rows, 39,998 informative rows)  
**Target Variable**: `rating` (Integer Likert Scale: 1 to 5)  
**Audit Objective**: Identify features that exhibit Direct Leakage, Potential/Post-Hoc Leakage, or are Safe for genuine real-world rating prediction from review text and product metadata.

---

## 1. Executive Summary & Classification Matrix

| Feature Name | Dtype | Correlation with `rating` | Classification | Real-World Availability & Justification |
| :--- | :---: | :---: | :---: | :--- |
| **`review_text`** | string | Semantic Embeddings | 🟢 **SAFE (Primary)** | Always available. The primary input for NLP rating prediction. |
| **`review_length`** | int | +0.1825 | 🟢 **SAFE** | Derived directly from `review_text`. |
| **`word_count`** | int | +0.2360 | 🟢 **SAFE** | Derived directly from `review_text`. |
| **`language`** | string | ~0.0000 | 🟢 **SAFE** | Detectable directly from text or user platform settings. |
| **`brand`** | string | ~0.0000 | 🟢 **SAFE** | Product metadata known prior to review authoring. |
| **`model`** | string | ~0.0000 | 🟢 **SAFE** | Product metadata known prior to review authoring. |
| **`price_usd`** | float | +0.0010 | 🟢 **SAFE** | Product catalog price known at purchase/review time. |
| **`price_local`** | float | +0.0010 | 🟢 **SAFE** | Product catalog price known at purchase/review time. |
| **`currency`** | string | ~0.0000 | 🟢 **SAFE** | Storefront / regional transaction metadata. |
| **`exchange_rate_to_usd`** | float | +0.0001 | 🟢 **SAFE** | Macroeconomic rate at transaction date. |
| **`customer_name`** | string | ~0.0000 | 🟢 **SAFE / IRRELEVANT** | Known at submission (high cardinality, non-predictive). |
| **`age`** | float | +0.0080 | 🟢 **SAFE / IRRELEVANT** | Demographic metadata from user profile. |
| **`country`** | string | ~0.0000 | 🟢 **SAFE / IRRELEVANT** | User profile / storefront region. |
| **`verified_purchase`** | bool | +0.0010 | 🟢 **SAFE** | Order verification flag known at submission. |
| **`source`** | string | ~0.0000 | 🟢 **SAFE** | Review portal / platform origin (e.g. Amazon, Flipkart). |
| **`review_date`** | string | ~0.0000 | 🟢 **SAFE** | Timestamp when the review was authored. |
| **`review_id`** | int | ~0.0000 | 🟢 **SAFE / IRRELEVANT** | Arbitrary database row identifier. |
| **`helpful_votes`** | int | **+0.4600** | 🟡 **POTENTIAL / POST-HOC LEAKAGE** | **NOT AVAILABLE AT INGESTION**. Upvotes accumulate over weeks/months after the rating is already published. |
| **`sentiment` (Ground Truth)** | string | **Strong (Cramer's V ~0.68)** | 🔴 **DIRECT LEAKAGE (If raw input)** | Ground truth sentiment label is co-assigned with the rating. *(SAFE only if predicted strictly by an NLP model from `review_text`).* |
| **`camera_rating`** | float | **+0.7616** | 🔴 **DIRECT TARGET LEAKAGE** | Simultaneous sub-aspect rating provided by user. Not present in text-only inputs. |
| **`battery_life_rating`** | float | **+0.7608** | 🔴 **DIRECT TARGET LEAKAGE** | Simultaneous sub-aspect rating provided by user. Not present in text-only inputs. |
| **`display_rating`** | float | **+0.7568** | 🔴 **DIRECT TARGET LEAKAGE** | Simultaneous sub-aspect rating provided by user. Not present in text-only inputs. |
| **`performance_rating`** | float | **+0.7544** | 🔴 **DIRECT TARGET LEAKAGE** | Simultaneous sub-aspect rating provided by user. Not present in text-only inputs. |
| **`design_rating`** | float | **+0.7547** | 🔴 **DIRECT TARGET LEAKAGE** | Simultaneous sub-aspect rating provided by user. Not present in text-only inputs. |

---

## 2. In-Depth Analysis of Critical Features

### A. Sub-Aspect Ratings (`camera_rating`, `battery_life_rating`, `display_rating`, `performance_rating`, `design_rating`)
- **Leakage Classification**: 🔴 **DIRECT TARGET LEAKAGE**
- **Statistical Evidence**:
  - Each individual sub-aspect rating has a Pearson correlation of **~0.755 – 0.762** with the overall `rating`.
  - The arithmetic mean of the 5 sub-ratings has a **0.9292 correlation** with the overall `rating`.
  - Merely rounding the mean of these 5 sub-ratings achieves **55.2% exact rating accuracy** and **98.4% within ±1 accuracy** without looking at the text at all!
- **Real-World Availability**:
  - In a real-world NLP rating agent (e.g. scoring scraped reviews, social media comments, feedback emails, or `reviews_only.csv`), these sub-ratings do not exist.
  - Feeding sub-ratings into the model creates a synthetic shortcut that replaces genuine semantic understanding with simple averaging of leaked sub-targets.

### B. `helpful_votes`
- **Leakage Classification**: 🟡 **POTENTIAL / POST-HOC TEMPORAL LEAKAGE**
- **Statistical Evidence**:
  - Pearson correlation with `rating` is **+0.4600** (positive reviews systematically receive more upvotes over time in this dataset).
- **Real-World Availability**:
  - At the moment a new review is written, `helpful_votes = 0`.
  - Using historical helpful votes to predict the initial rating introduces look-ahead bias.

### C. `sentiment`
- **Leakage Classification**: 🔴 **DIRECT LEAKAGE (If using tabular column)** / 🟢 **SAFE (If model-predicted)**
- **Statistical Evidence**:
  - Negative reviews: 89.2% are Ratings 1 & 2.
  - Positive reviews: 72.4% are Ratings 4 & 5.
- **Real-World Availability**:
  - If a model takes the *ground truth* `sentiment` column as a tabular feature to predict `rating`, it is leaking the post-hoc classification.
  - If `sentiment` is **predicted upstream from `review_text`** using sentence embeddings and profile matching, it is completely safe and serves as a valid intermediate representation.

---

## 3. Recommended Feature Set for Genuine Production Rating Prediction

For true out-of-sample rating prediction on unseen reviews (such as `reviews_only.csv` or incoming live customer feedback), the only valid feature set is:

1. **Semantic Text Embeddings**: SentenceTransformer embeddings (`all-MiniLM-L6-v2`) extracted from `review_text`.
2. **Text Surface Features**: `word_count`, `review_length`, character count.
3. **Product & Catalog Metadata** *(Optional prior features)*: `brand`, `model`, `price_usd`.
4. **Intermediate Model Signals**: Model-predicted sentiment score / polarity derived strictly from the text.
