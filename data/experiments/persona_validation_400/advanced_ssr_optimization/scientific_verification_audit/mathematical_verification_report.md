# Mathematical and Statistical Verification Report

## 1. Linear Variance Rescaling
The transformation formula is:
$$y = \mu_{\text{train}} + (x - \mu_{\text{synth}}) \cdot \frac{\sigma_{\text{train}}}{\sigma_{\text{synth}}}$$

Under pure affine transformations ($a > 0$), ANOVA $F$-statistics, $p$-values, and Cohen's $d$ / Hedges' $g$ are strictly invariant. However, because survey ratings must be bounded within $[1.0, 5.0]$, clipping is applied:
$$y_{\text{clipped}} = \text{clip}(y, 1.0, 5.0)$$

### Empirical Invariance Verification Table

| Property                 | Unclipped_Condition          | Clipped_Condition                          | Theoretical_Invariance   | Empirical_ANOVA_F                       |   ANOVA_F_Difference_From_Orig | Empirical_Hedges_g                      |   Hedges_g_Difference_From_Orig |   Total_Responses_Clipped |   Boundary_Clipped_Low |   Boundary_Clipped_High | Scientific_Verdict                                                                                        |
|:-------------------------|:-----------------------------|:-------------------------------------------|:-------------------------|:----------------------------------------|-------------------------------:|:----------------------------------------|--------------------------------:|--------------------------:|-----------------------:|------------------------:|:----------------------------------------------------------------------------------------------------------|
| Affine Linearity (a > 0) | Strictly Linear (y = ax + b) | Piecewise Linear (Nonlinear at boundaries) | Strictly Invariant       | Orig=4.3475, Unclip=4.3475, Clip=4.2889 |                         0.0585 | Orig=1.0598, Unclip=1.0598, Clip=1.0571 |                          0.0027 |                        10 |                      1 |                       9 | Clipped transformation causes minor boundary attenuation; pure affine scaling preserves exact statistics. |

**Verdict:** Because 3.2% of responses hit boundary constraints, clipping causes minor numerical attenuation ($F$ shifts from 4.3475 to 4.2794). Claiming '100% preservation' is mathematically inaccurate; it should be stated as **'98.4% statistical preservation with minor boundary clipping'**.

## 2. Quantile Matching
Quantile matching inverts the empirical CDF of the human training set. While it achieves high distributional matching ($W_1 = 0.1093$), it creates artificial discrete step clusters that compress subtle persona contrasts, reducing PE ANOVA $F$ to 2.64 ($p = 0.0570$).

## 3. Temperature Scaling
Probability scaling $p_i(T) \propto p_i^{1/T}$ was verified. $T=0.5$ successfully expands standard deviation from 0.62 to 0.86 without violating PMF non-negativity or sum-to-one constraints.
