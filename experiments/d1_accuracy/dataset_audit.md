# D1 Accuracy Project — Dataset Audit

Audit scope: read-only inspection of all 50,000 rows and 25 columns of `data/Mobile Reviews Sentiment.csv`, the D1 held-out workbook `data/actual_reviews.xlsx`, D1 source/artifacts, and existing leakage/baseline reports. Original source data and saved results were not changed. Numeric summaries below use observed rows; no row or label was removed from the source.

## Dataset shape, types, nulls, uniqueness

Source CSV: **50,000 rows × 25 columns**. Exactly 10,002 rows are entirely empty; one additional row has a rating but an empty review text. There are **10,001 exact duplicate full rows**. The D1 training loader requires non-null review text, target and all five aspects, yielding **39,997 training rows** (the source has 39,998 rows with non-null review text/rating before requiring all aspects). These facts are retained as audit findings, not preprocessing decisions.

| Column | Read dtype | Missing | Unique non-null | Type / audit interpretation |
|---|---|---:|---:|---|
| `review_id` | float64 | 10,002 | 39,998 | Identifier; exclude as predictor |
| `customer_name` | object | 10,002 | 37,224 | Personal/identifier-like; exclude |
| `age` | float64 | 10,002 | 48 | Numeric respondent attribute |
| `brand` | object | 10,002 | 7 | Categorical product metadata |
| `model` | object | 10,002 | 22 | Categorical product metadata |
| `price_usd` | float64 | 10,002 | 33,337 | Numeric product metadata |
| `price_local` | object | 10,002 | 38,960 | Currency-formatted/high-cardinality price; parse and validate before modeling; raw category is unsuitable |
| `currency` | object | 10,002 | 8 | Categorical metadata |
| `exchange_rate_to_usd` | float64 | 10,002 | 8 | Numeric metadata, mostly country/currency encoding |
| `rating` | float64 | 10,002 | 5 | Target; never a feature |
| `review_text` | object | 10,003 | 110 | Free text; valid labeled data contains only 110 distinct strings |
| `sentiment` | object | 10,002 | 3 | Derived/annotated sentiment; test separately, not an independent raw predictor |
| `country` | object | 10,002 | 8 | Categorical respondent metadata |
| `language` | object | 10,002 | 4 | Categorical metadata |
| `review_date` | object | 10,002 | 1,096 | Timestamp text; observed dates span 2022-01-11 to 2025-12-10 |
| `verified_purchase` | object | 10,002 | 2 | Boolean-like categorical metadata |
| `battery_life_rating` | float64 | 10,002 | 5 | Simultaneously assigned aspect sub-rating; leakage-prone |
| `camera_rating` | float64 | 10,002 | 5 | Simultaneously assigned aspect sub-rating; leakage-prone |
| `performance_rating` | float64 | 10,002 | 5 | Simultaneously assigned aspect sub-rating; leakage-prone |
| `design_rating` | float64 | 10,002 | 5 | Simultaneously assigned aspect sub-rating; leakage-prone |
| `display_rating` | float64 | 10,002 | 5 | Simultaneously assigned aspect sub-rating; leakage-prone |
| `review_length` | float64 | 10,002 | 34 | Text-derived numeric feature; potentially available if computed from input text |
| `word_count` | float64 | 10,002 | 10 | Text-derived numeric feature; potentially available if computed from input text |
| `helpful_votes` | float64 | 10,002 | 18 | Post-publication engagement; exclude from pre-rating predictions, diagnostic only |
| `source` | object | 10,002 | 5 | Categorical platform/source metadata |

Observed low-cardinality values: brand = Realme, Google, Xiaomi, Motorola, Apple, OnePlus, Samsung; model = 22 listed phone models; sentiment = Negative, Neutral, Positive; country = India, Brazil, UAE, Australia, Germany, UK, Canada, USA; language = Hindi, Portuguese, English, German; source = Amazon, Flipkart, AliExpress, BestBuy, eBay; currency = INR, BRL, AED, AUD, EUR, GBP, CAD, USD; `verified_purchase` = True/False. Target and all five aspect columns each take values 1–5. `word_count` observed values are integers 6–15. `price_local` and `exchange_rate_to_usd` include 8 country/currency conversion levels. Other high-cardinality unique values are enumerated by counts above rather than dumping thousands of distinct values.

## Target distribution

Among all 39,998 non-null target rows: rating 1 = 5,057 (12.64%), 2 = 7,814 (19.54%), 3 = 10,037 (25.09%), 4 = 11,245 (28.11%), 5 = 5,845 (14.61%). Majority-class accuracy is therefore **28.11%** (predict class 4). The 10,002 empty rows have no rating.

## Feature-target relationships

Numeric Pearson / Spearman correlations against overall rating, computed over non-null labeled rows:

| Feature | Pearson r | Spearman rho | Interpretation |
|---|---:|---:|---|
| `battery_life_rating` | 0.76084 | 0.76767 | Co-rated aspect; leakage-prone |
| `camera_rating` | 0.76161 | 0.76867 | Co-rated aspect; leakage-prone |
| `performance_rating` | 0.75439 | 0.76115 | Co-rated aspect; leakage-prone |
| `design_rating` | 0.75466 | 0.76106 | Co-rated aspect; leakage-prone |
| `display_rating` | 0.75681 | 0.76366 | Co-rated aspect; leakage-prone |
| `helpful_votes` | 0.45997 | 0.48895 | Strong but post-publication outcome; exclude from valid prediction |
| `word_count` | 0.23604 | 0.21519 | Text-derived; available from input text |
| `review_length` | 0.18252 | 0.16917 | Text-derived; available from input text |
| `age` | 0.00798 | 0.00796 | Weak respondent attribute |
| `price_usd` | 0.00099 | 0.00071 | No observed marginal linear/rank signal |
| `exchange_rate_to_usd` | 0.00007 | 0.00243 | No observed marginal signal |

Categorical associations with target (Cramer's V; high-cardinality caution applies): brand 0.01255; model 0.02120; currency 0.01100; country 0.01100; language 0.00826; verified_purchase 0.00955; source 0.00908. `sentiment` has Cramer's V 0.61513 (mutual information 0.4204 nats), consistent with a target-correlated label derived from the review; only a separately labeled sentiment-feature experiment can determine its usefulness and whether it is legitimately available at deployment. Raw `price_local` has 38,960 categories and a misleading Cramer's V 0.98725; this is a high-cardinality statistic, not evidence of a valid target relationship. It must be parsed as numeric/currency and re-audited, never one-hot encoded as 38,960 nominal levels.

The existing `data/experiment_d_aspect_features/leakage_analysis.md` additionally reports the mean of the five aspects has Pearson r=0.9292 with target, and rounded aspect mean alone gives 55.20% accuracy. This is strong evidence these are simultaneous sub-targets, not ordinary predictors available in text-only intake.

## Duplicate and label-conflict audit

There are **110 unique non-null review texts** among 39,997 text-bearing labeled rows. All 110 exact text groups have multiple overall ratings; each text has 3–5 distinct labels (maximum 5). Thus all 39,997 source rows belong to conflicting-label text groups. Example texts recur hundreds of times with ratings spanning multiple classes. This is not safely repairable by deduplication or relabeling without external annotation provenance; the source has been preserved.

The D1 held-out workbook contains 10,003 distinct review IDs, no exact duplicate rows, and 9,893 repeated-text rows (110 unique texts). Review-ID overlap between source CSV and held-out workbook is **0**, but the workbooks share **90 of the 110** unique review texts. The held-out workbook therefore tests unseen IDs, not unseen text. A separate group split by exact `review_text` is required for unseen-text claims. Because only 110 text groups exist and their labels conflict, grouped results will have high uncertainty and cannot be directly conflated with row-level results.

## Existing D1 and baseline protocol

`aspect_rating_model.py` loads `data/Mobile Reviews Sentiment.csv`, drops rows missing text, target, or any of the five aspects, and reads the fixed 10,003-row `data/actual_reviews.xlsx` workbook as held-out evaluation. It fits `StandardScaler` on training only, then a 100-tree Random Forest (`max_depth=10`, `random_state=42`, `n_jobs=-1`) on exactly the five aspect inputs. The source code writes to `data/aspect_model/`; that script was not run in this audit because rerunning it would overwrite protected artifacts. Existing saved metrics report 73.3380% accuracy, macro F1 73.7047%, weighted F1 73.3162%, MAE 0.2680, RMSE 0.5204, Spearman 0.9166. Existing confusion matrix and per-class outputs are preserved.

The frozen text-only baseline is a separate MiniLM example-bank top-k=3 model and has existing 46.1362% held-out accuracy on the same workbook. Its saved files under `data/current_baseline_evaluation/` and `data/calibrated_discretization/` are not part of this new comparison and must remain untouched.

## Feature eligibility decided from this audit

- **Diagnostic-only aspect group:** the five aspect columns reproduce D1 but are simultaneous sub-targets; report distinctly and never call leakage-free or text-only.
- **Eligible structured group:** age, brand, model, numeric price after validation, currency, exchange rate, country, language, verified-purchase flag, source, and submission date only where available at the stated prediction time. Use training-fitted preprocessing. Brand/model/country/etc. show weak univariate association, but still test jointly.
- **Text-derived group:** review text, and length/word-count computed solely from that text. Evaluate TF-IDF in row-level and grouped-by-text settings.
- **Separate derived-sentiment group:** `sentiment`; document its provenance/availability and never silently mix it into text-only.
- **Exclude from valid predictive groups:** `rating`; `review_id`; `customer_name`; post-publication `helpful_votes`; raw high-cardinality `price_local`; any target-derived aspect or target encoding computed outside training folds.
- **Engineered features:** aspect mean/spread/differences only in diagnostic aspect experiments; any structured/text feature engineering must be fitted/computed without using held-out labels.

No model feature comparison or tuning was run before completing this audit. Next, retain the fixed D1 workbook protocol for direct comparability while also add a separate grouped-by-review-text evaluation protocol for unseen-text generalization. All new outputs belong only under `experiments/d1_accuracy/`; existing outputs remain protected.
