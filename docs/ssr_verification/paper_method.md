# Original SSR Method: Paper-Grounded Specification

## Source and scope

Primary source: Maier et al., *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings*, arXiv:2510.08338v3 (27 October 2025), especially Sections 3.4 and A.4.1–A.4.3 and Appendix C.1–C.2: <https://arxiv.org/html/2510.08338v3>.

The supplied local files `docs/ssr_verification/source_excerpts/maier_section_a43.txt` and `docs/ssr_verification/source_excerpts/maier_appendix_c.txt` are text extractions with missing equations/anchor wording. I checked the full v3 arXiv HTML and the paper-linked implementation at <https://github.com/pymc-labs/semantic-similarity-rating>. The paper is the authority for the study settings below; the linked package is used only to inspect available implementation details.

## Personas and free-text elicitation

The paper pairs each human consumer with a synthetic consumer: an LLM is prompted to impersonate that person using the same or a subset of their demographic attributes. Attributes named in the paper include age, gender, income, location/region, and ethnicity, with availability varying in the source surveys. The synthetic consumer sees the product concept, normally the full concept image (text description and sometimes product illustration), and is asked, “How likely are you to purchase the product?” The SSR route uses a brief, otherwise unconstrained natural-language response, not a directly requested Likert number. A text-stimulus condition used only the product description; the paper says GPT-4o transcribed that text from the concept image.

The paper reports two samples per prompt (`n = 2`). Production experiments used GPT-4o and Gemini-2.0-flash, with LLM `top_p = 0.9` and LLM generation temperature `T_LLM` in `{0.5, 1.5}`; main-text results use `T_LLM = 0.5`. This generation temperature is distinct from SSR post-mapping temperature `T`.

The article describes zero-shot/prompt-based synthetic response generation and explicitly says consumer-response training/fine-tuning was not used. It does not provide enough prompt text to reconstruct every production prompt verbatim; exact system prompt wording is therefore unresolved from the paper text.

## Reference statements and embeddings

Each anchor set contains exactly five ordered statements, one for each Likert response $r \in \{1,2,3,4,5\}$. The paper uses six similar-but-not-identical sets, manually designed to capture different ways consumers express purchase intent. Appendix C.1 describes their intended gradient: rating 1 expresses unlikely purchase; rating 3 indifference/uncertainty; rating 5 intent/possible intent to purchase; ratings 2 and 4 fall semantically between their neighbors. The statements are described as short, generic, domain-independent consumer-product purchase-intent references.

**Exact anchor text is not printed in the supplied appendix or the v3 HTML text.** The six actual five-statement sets cannot be reconstructed faithfully from the available paper text. The linked package's README demonstrates generic agreement anchors, but those examples are not evidence that they were the six experimental purchase-intent sets. Do not substitute those examples or this repository's smartphone-review anchors and call them original paper anchors.

The study embeds both response and reference text with OpenAI `text-embedding-3-small`; the paper says tests using `text-embedding-3-large` left results virtually unchanged. It specifies cosine similarity. It does not state a vector dimension, normalization option, batching size, or embedding API request settings. Those details are unresolved and should not be invented.

## SSR equations

For response embedding $\mathbf v_t$ and the embedding $\mathbf v_{r,i}$ of rating-$r$ anchor in set $i$, paper Eq. 7 defines raw cosine similarity:

$$
\gamma(\sigma_{r,i},t) = \frac{\mathbf v_{r,i} \cdot \mathbf v_t}{\|\mathbf v_{r,i}\|\,\|\mathbf v_t\|}.
$$

For each anchor set $i$, let $\ell$ denote the rating whose reference statement has the minimum similarity to this response. The unnormalized weight in paper Eq. 8 is:

$$
w_{i}(r) = \gamma(\sigma_{r,i},t) - \gamma(\sigma_{\ell,i},t) + \epsilon\,\delta_{\ell,r}.
$$

The five-point PMF is the normalized weight:

$$
p_i(r) = \frac{w_i(r)}{\sum_{j=1}^{5} w_i(j)}
= \frac{\gamma(\sigma_{r,i},t)-\gamma(\sigma_{\ell,i},t)+\epsilon\delta_{\ell,r}}
{\sum_{j=1}^{5}\gamma(\sigma_{j,i},t)-5\gamma(\sigma_{\ell,i},t)+\epsilon}.
$$

Thus epsilon is added only at the minimum-similarity rating, and also contributes once to the denominator. It is not added to every class. This minimum subtraction offsets low variation across cosine scores. At $\epsilon=0$, the least-similar rating has zero probability (unless ties create more than one minimum; the paper does not specify a tie convention). Epsilon is described as a control for that zero-mass behavior and division-by-zero edge cases.

Paper Eq. 9 optionally temperature-scales an already formed PMF:

$$
p_i(r,T) = \frac{p_i(r)^{1/T}}{\sum_{j=1}^{5}p_i(j)^{1/T}}.
$$

**Reported SSR settings are $\epsilon=0$ and $T=1$.** These are the values to reproduce; they mean no epsilon offset and no temperature reshaping. The paper discusses tuning epsilon and temperature as possible future/control levers, not as settings used for its reported study. Do not substitute the repository baseline's softmax temperature 0.1.

For the six sets, the paper averages the six single-response PMFs rating-wise:

$$
p_{\mathrm{SSR}}(r) = \frac{1}{6}\sum_{i=0}^{5}p_i(r,T).
$$

At the reported $T=1$, the optional temperature operation is the identity, so there is no ambiguity about whether it is applied before or after averaging. The paper-linked package currently scales after taking the mean when called with reference set `mean`; non-default temperature ordering is not needed to reproduce the study and should not be treated as a new paper result.

The resulting five probabilities form the response PMF. When a scalar purchase-intent summary is needed, its expected rating is:

$$
\mathbb{E}[r] = \sum_{r=1}^{5} r\,p_{\mathrm{SSR}}(r).
$$

For survey-level distributions, the paper averages individual response PMFs; for mean purchase intent it uses the PMF expectation. It does not require converting every free-text response to a single argmax class as the primary SSR output.

## Implementation settings and what is not specified

| Setting | Paper-reported value | Notes |
|---|---|---|
| Likert points | 1–5 | Five ordered responses |
| Anchor sets | 6 sets × 5 statements | Average single-response PMFs across all six |
| Embedding model | OpenAI `text-embedding-3-small` | `text-embedding-3-large` reportedly gave virtually unchanged results |
| Similarity | Raw cosine similarity | Equation 7 |
| Epsilon | 0 | Study setting; epsilon is an optional equation parameter |
| SSR temperature $T$ | 1 | Study setting; not the LLM generation temperature |
| LLMs | GPT-4o, Gemini-2.0-flash | Production runs described in Appendix A.4 |
| LLM `top_p` | 0.9 | Response-generation setting |
| LLM temperature | 0.5 and 1.5; main text reports 0.5 | Not the SSR PMF temperature |
| Samples per persona/prompt | 2 | Paper reports this was sufficient for stability |
| Embedding dimension / normalization / API batching | Not specified | Unresolved; do not infer as paper settings |
| Exact six anchor texts | Not present in available paper text | Unresolved; essential for exact experiment-level replication |
| Smartphone construct anchors | Not in the paper | Any PE, EE, HM, SI, PV, BI anchors must be labeled proposed adaptations |

The linked Python package currently defaults to `all-MiniLM-L6-v2`, demonstrating that package defaults are not interchangeable with the paper's stated experimental embedding model. Its implementation offers a useful independent check of the PMF algebra, but package defaults do not override the paper.
