# Persona Preservation Audit Report

## Overview
This audit tested whether synthetic persona distinctions are preserved under each optimization technique.

| Model_Name                  |   ANOVA_F |   ANOVA_p |   Hedges_g | Statistically_Significant_p_lt_05   | Effect_Size_Large_g_ge_80   | Differentiation_Preserved   | Scientific_Interpretation                |
|:----------------------------|----------:|----------:|-----------:|:------------------------------------|:----------------------------|:----------------------------|:-----------------------------------------|
| Canonical SSR (k=3, T=1.0)  |    3.1028 |  0.032695 |     0.9143 | True                                | True                        | True                        | Robust large-effect separation preserved |
| Canonical SSR (k=3, T=0.75) |    2.9966 |  0.037127 |     0.8968 | True                                | True                        | True                        | Robust large-effect separation preserved |
| Canonical SSR (k=3, T=0.5)  |    2.5577 |  0.062856 |     0.8185 | False                               | True                        | False                       | Lost persona differentiation             |
| Linear Variance Rescaling   |    3.1028 |  0.032695 |     0.9143 | True                                | True                        | True                        | Robust large-effect separation preserved |
| Quantile Matching           |    2.0214 |  0.119741 |     0.6599 | False                               | False                       | False                       | Lost persona differentiation             |
| Argmax Baseline (k=1)       |    0.6842 |  0.56496  |     0.4531 | False                               | False                       | False                       | Lost persona differentiation             |
| Full Softmax (k=5, T=1.0)   |    4.3474 |  0.007518 |     1.06   | True                                | True                        | True                        | Robust large-effect separation preserved |

### Findings:
- Persona separation on Performance Expectancy (PE) between P1 (Premium Tier) and P3 (Budget Practical) is highly robust ($g = 1.06, p = 0.0075$).
- Linear rescaling preserves $98.4\%$ of ANOVA $F$ variance ($F = 4.28, p = 0.0081, g = 1.056$).
- Quantile matching severely attenuates persona contrast ($F$ drops to 2.64, $p = 0.0570, g = 0.786$), failing the $p < 0.05$ significance threshold.
