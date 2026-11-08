# Phase 19 digital-twin validation

Pairs analysed: 80 (model qwen2.5:7b-instruct, E-SSR k=3 T=0.75, BI anchors, nomic-embed-text). Gate first-pass rate 95.0%.

| Level | Groups | Twin E-SSR vs human Q21 (Pearson, 95% CI) | Spearman | MAE | Twin direct rating Pearson | MAE |
|---|---|---|---|---|---|---|
| persona | 4 | 0.36 [-1.0, 1.0] | 0.40 | 0.40 | 0.82 | 0.39 |
| brand | 3 | 0.88 [-1.0, 1.0] | 0.50 | 0.51 | 0.90 | 0.25 |
| brand_x_persona | 5 | 0.70 [0.18518186138362175, 1.0] | 0.90 | 0.37 | 0.87 | 0.49 |

Distribution vs human Q21: Wasserstein E-SSR 0.594 vs direct 0.488; KS E-SSR 0.212 vs direct 0.300.
SD: human 1.04, twin E-SSR 0.35, twin direct 0.56.

Human text vs own Q21 (Spearman): all 0.221 (Q26) / 0.157 (Q23-Q26 combined); 3+ words 0.423 / 0.317.
Notes: twins never saw the respondent's ratings or text. Groups below the minimum size are excluded. Small group counts give wide intervals.
