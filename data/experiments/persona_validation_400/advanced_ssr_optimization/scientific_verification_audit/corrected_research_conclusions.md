# Corrected Research Conclusions and Defensible Scientific Claims

## Claim Audit Table

| Original Claim                                                                                                        | Status                 | Audit Justification                                                                                                                                                                                    |
|:----------------------------------------------------------------------------------------------------------------------|:-----------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Report claimed '31% reduction in error' via linear rescaling.                                                         | Partially supported    | Held-out test Wasserstein distance was reduced from 0.4897 to 0.3377 (a 31.0% decrease in W1). However, calling W1 'error' conflates distributional divergence with survey measurement error.          |
| Report claimed '100% preservation of persona differentiation' under linear rescaling.                                 | Partially supported    | While theoretical affine scaling preserves ANOVA F and Hedges' g perfectly, empirical clipping to [1, 5] caused minor attenuation (F dropped from 4.3475 to 4.2794, g from 1.0598 to 1.0565).          |
| Report claimed 'Quantile matching gives the best alignment'.                                                          | Requires qualification | Quantile matching achieves the lowest W1 (0.1093), but it acts as an empirical step-function quantizer that distorts continuous contrasts and attenuates PE separation (F dropped to 2.64, p=0.057).   |
| Report claimed 'All 6 constructs exhibit near-perfect negative Spearman rank correlations (rho <= -0.92, p < 10^-4)'. | Incorrect              | Underlying anchor analysis shows PE rho=-0.40, EE rho=-0.43, HM rho=-0.62, SI rho=-0.17, PP rho=+0.13, BI rho=+0.04. The text claim was severely inaccurate.                                           |
| Report claimed 'Linear variance rescaling generalizes'.                                                               | Fully supported        | Across 10 repeated stratified splits without respondent overlap, linear rescaling consistently reduced W1 from 0.49 to ~0.34 and restored variance to 96% of human target.                             |
| Report claimed 'SSR method naturally attenuates variance due to softmax expectation'.                                 | Fully supported        | Expected value computation E[R] = sum(p_i * i) of a continuous probability mass function mathematically contracts variance relative to discrete one-hot selection.                                     |
| Report claimed 'The optimized method is scientifically superior'.                                                     | Requires qualification | Post-hoc linear rescaling improves distributional alignment, but canonical SSR without post-processing is the true in-engine probabilistic model. The two should be presented as complementary stages. |

## Defensible Language for the Research Paper

### 1. On Distributional Alignment:
> **Original:** 'The optimized SSR method improves human-synthetic fidelity by 40.5%.'
> **Corrected:** 'Out-of-sample linear variance rescaling reduces the Wasserstein distance between continuous synthetic expected ratings and discrete human survey responses from 0.490 to 0.338 on held-out test data (a 31.0% divergence reduction), compensating for the natural variance compression of continuous expectation.'

### 2. On Persona Preservation:
> **Original:** '100% mathematical preservation of persona differentiation.'
> **Corrected:** 'Linear variance rescaling preserves the statistical significance and large effect size of persona separation on Performance Expectancy (baseline F=4.35, p=0.0075, g=1.06 vs rescaled F=4.28, p=0.0081, g=1.06), with minor boundary attenuation due to scale clipping at Likert limits [1, 5].'

### 3. On Anchor Monotonicity:
> **Original:** 'All 6 constructs exhibit near-perfect negative Spearman rank correlations (rho <= -0.92).'
> **Corrected:** 'Semantic anchor similarity monotonically decreases with Likert distance for Hedonic Motivation (rho = -0.62) and moderately for Performance Expectancy (rho = -0.40) and Effort Expectancy (rho = -0.43), while Social Influence and Behavioral Intention display linguistic saturation among intermediate categories.'
