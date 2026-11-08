# Phase 19 digital-twin validation

Pairs analysed: 80 (model qwen2.5:7b-instruct, E-SSR k=3 T=0.75, repurchase anchors, nomic-embed-text). Gate first-pass rate 95.0%.

| Level | Groups | Twin E-SSR vs human Q21 (Pearson, 95% CI) | Spearman | MAE | Twin direct rating Pearson | MAE |
|---|---|---|---|---|---|---|
| persona | 4 | 0.69 [-1.0, 1.0] | 0.20 | 0.37 | 0.82 | 0.39 |
| brand | 3 | 0.99 [0.9904468417545752, 1.0] | 1.00 | 0.43 | 0.90 | 0.25 |
| brand_x_persona | 5 | 0.85 [0.1245537148329282, 1.0] | 0.80 | 0.38 | 0.87 | 0.49 |

Distribution vs human Q21: Wasserstein E-SSR 0.580 vs direct 0.488; KS E-SSR 0.225 vs direct 0.300.
SD: human 1.04, twin E-SSR 0.33, twin direct 0.56.

Human text vs own Q21 (Spearman): all 0.223 (Q26) / 0.209 (Q23-Q26 combined); 3+ words 0.481 / 0.435.
Notes: twins never saw the respondent's ratings or text. Groups below the minimum size are excluded. Small group counts give wide intervals.
