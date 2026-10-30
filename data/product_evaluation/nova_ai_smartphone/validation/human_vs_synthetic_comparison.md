# Human Survey Evidence vs. Synthetic Product Evaluation Comparison

**Fielded Human Survey Sample:** $N=258$ eligible primary respondents  
**Simulated Evaluations:** 4 digital personas reacting to Nova AI Smartphone (₹24,999)  

---

## 1. Methodological Boundary & Comparison Integrity

> [!WARNING]
> **Essential Conceptual Distinction:**
> The fielded human survey measured respondents' evaluations of **their own existing smartphone and current brand loyalty** in 2026.
> The synthetic product evaluation measures simulated persona reactions to a **new, hypothetical third-party product concept (Nova AI Smartphone at ₹24,999)**.
> 
> Therefore, human Likert ratings cannot be treated as a direct numerical ground truth for the Nova AI Smartphone. 
> Instead, this comparison assesses **relative persona alignments, behavioral tendencies, and construct divergence**.

---

## 2. Behavioral Intention (`BI`) vs. Human Brand Repurchase (`Q21`)

In the human survey, `Q21` asked: *"How likely are you to buy the same smartphone brand again?"* (1–5 Likert).  
In the product evaluation, `BI` measures simulated intention to purchase the **Nova AI Smartphone**.

| Persona | Human Q21 Mean (Current Brand) | Human Q21 Distribution (1 to 5) | Synthetic Nova AI BI Rating ($E[R]$) | Synthetic BI PMF $[p_1, p_2, p_3, p_4, p_5]$ | Behavioral Interpretation |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **P1 (Premium)** | **4.01** | `1:5, 2:1, 3:18, 4:27, 5:36` | **3.26** | `[0.00, 0.22, 0.43, 0.22, 0.13]` | High existing loyalty to Apple/iOS explains reluctance to switch to a mid-range Android phone. |
| **P2 (Mid-Heavy)** | **3.00** | `1:8, 2:2, 3:27, 4:12, 5:3` | **3.19** | `[0.00, 0.21, 0.48, 0.23, 0.08]` | Moderate human satisfaction matches neutral/cautious synthetic consideration ($p_3 = 0.48$). |
| **P3 (Mid-Mod)** | **3.26** | `1:9, 2:5, 3:36, 4:28, 5:9` | **3.21** | `[0.00, 0.18, 0.50, 0.24, 0.08]` | Consistent neutral-to-moderate stance across both human baseline and synthetic concept. |
| **P4 (Budget)** | **3.66** | `1:2, 2:3, 3:6, 4:14, 5:7` | **3.01** | `[0.00, 0.22, 0.62, 0.08, 0.08]` | Drops from human brand satisfaction (3.66) to 3.01 due to the severe ₹24,999 price barrier. |

### Key Finding on Behavioral Intention
- In human data, P1 has the highest brand repurchase intention (**4.01**) because they own premium devices (primarily iPhones) and intend to stay with Apple.
- When presented with the Nova AI concept, P1's simulated purchase intention drops to **3.26** precisely because adopting Nova requires abandoning iOS.
- P4 drops from **3.66** to **3.01**, with **62.3% of the probability mass on Category 3 (Neutral)**, demonstrating that high price suppresses adoption regardless of feature interest.

---

## 3. Effort Expectancy (`EE`) vs. Human Tech Comfort (`Q20`)

In the human survey, `Q20` asked: *"How comfortable are you with using new technology?"* (1–5 Likert).  
In the product evaluation, `EE` evaluates the perceived clarity, ease of use, and interface learning curve.

| Persona | Human Q20 Tech Comfort Mean | Human Q20 Distribution | Synthetic Nova AI EE Rating ($E[R]$) | Synthetic EE PMF $[p_1, p_2, p_3, p_4, p_5]$ | Usability Interpretation |
| :--- | :---: | :--- | :---: | :--- | :--- |
| **P1 (Premium)** | **3.91** | `1:1, 2:5, 3:20, 4:36, 5:25` | **3.38** | `[0.13, 0.00, 0.45, 0.19, 0.23]` | Recognizes clean UI, but notes ecosystem transition effort. |
| **P2 (Mid-Heavy)** | **3.56** | `1:3, 2:4, 3:17, 4:17, 5:11` | **3.46** | `[0.17, 0.00, 0.32, 0.24, 0.28]` | Expresses concern that AI suggestions might complicate navigation. |
| **P3 (Mid-Mod)** | **3.62** | `1:2, 2:5, 3:32, 4:33, 5:15` | **3.63** | `[0.07, 0.00, 0.44, 0.22, 0.27]` | Highest EE score; views clean interface and shortcuts favorably. |
| **P4 (Budget)** | **3.69** | `1:2, 2:3, 3:5, 4:15, 5:7` | **3.13** | `[0.31, 0.00, 0.25, 0.12, 0.31]` | Shows bimodal uncertainty regarding whether AI features complicate daily use. |

---

## 4. Constructs Where Direct Human Survey Validation is Not Possible

1. **Performance Expectancy (`PE`)**: The fielded human survey lacked multi-item Likert PE scales; it captured performance qualitatively in `Q25`. Simulated PE (2.44 to 3.14) reflects qualitative themes but has no numeric survey counter-item.
2. **Hedonic Motivation (`HM`) & Social Influence (`SI`)**: Measured in human survey via discrete behavioral proxies (`Q18` screen time, `Q19` entertainment, `Q22` info sources). Direct item-by-item Likert comparison is unavailable.
3. **Price Value (`PV`)**: Unanchored for SSR; qualitative consensus confirms ₹24,999 is perceived as high by all personas except P1 (who finds it low-spec rather than expensive).

**Conclusion:** The simulated persona results are behaviorally coherent with the human survey dataset, showing expected divergence based on brand inertia and budget constraints.
