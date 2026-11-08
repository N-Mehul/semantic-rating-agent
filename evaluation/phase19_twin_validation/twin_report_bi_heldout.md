# Phase 19 digital-twin validation

Pairs analysed: 40 (model qwen2.5:7b-instruct, E-SSR k=3 T=0.75, bi anchors, nomic-embed-text). Gate first-pass rate 97.5%.

| Level | Groups | Twin E-SSR vs human Q21 (Pearson, 95% CI) | Spearman | MAE | Twin direct rating Pearson | MAE |
|---|---|---|---|---|---|---|
| persona | 4 | -0.03 [-1.0, 1.0] | 0.32 | 0.25 | -0.11 | 0.67 |
| brand | 1 | nan None | nan | nan | nan | nan |
| brand_x_persona | 4 | -0.03 [-1.0, 1.0] | 0.32 | 0.25 | -0.11 | 0.67 |

Distribution vs human Q21: Wasserstein E-SSR 0.809 vs direct 0.625; KS E-SSR 0.225 vs direct 0.250.
SD: human 1.30, twin E-SSR 0.36, twin direct 0.74.

Human text vs own Q21 (Spearman): all 0.435 (Q26) / 0.455 (Q23-Q26 combined); 3+ words 0.617 / 0.609.
Notes: twins never saw the respondent's ratings or text. Groups below the minimum size are excluded. Small group counts give wide intervals.
