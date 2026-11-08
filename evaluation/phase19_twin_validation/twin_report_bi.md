# Phase 19 digital-twin validation

Pairs analysed: 80 (model qwen2.5:7b-instruct, E-SSR k=3 T=0.75, bi anchors, nomic-embed-text). Gate first-pass rate 97.5%.

| Level | Groups | Twin E-SSR vs human Q21 (Pearson, 95% CI) | Spearman | MAE | Twin direct rating Pearson | MAE |
|---|---|---|---|---|---|---|
| persona | 4 | 0.76 [-1.0, 1.0] | 0.80 | 0.23 | 0.89 | 0.34 |
| brand | 3 | 0.93 [-1.0, 1.0] | 0.50 | 0.45 | 0.90 | 0.24 |
| brand_x_persona | 5 | 0.66 [-0.9999999999999999, 1.0] | 0.50 | 0.30 | 0.80 | 0.43 |

Distribution vs human Q21: Wasserstein E-SSR 0.522 vs direct 0.338; KS E-SSR 0.212 vs direct 0.150.
SD: human 1.04, twin E-SSR 0.45, twin direct 0.67.

Human text vs own Q21 (Spearman): all 0.258 (Q26) / 0.259 (Q23-Q26 combined); 3+ words 0.515 / 0.482.
Notes: twins never saw the respondent's ratings or text. Groups below the minimum size are excluded. Small group counts give wide intervals.
