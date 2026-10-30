# Canonical SSR Reproduction Readiness

**Status: IMPLEMENTATION VALIDATED; FULL REPRODUCTION PENDING**

## Verified Methodology

Maier et al., arXiv:2510.08338v3, specify six reference-statement sets of five Likert points (1–5). For each set, the response and five anchors are embedded, cosine similarities are computed, the minimum similarity is subtracted, the epsilon indicator term is applied, and the five weights are normalized into a PMF. The paper reports `epsilon = 0` and post-mapping `T = 1`; the six per-set PMFs are averaged rating-wise. The named embedding model is OpenAI `text-embedding-3-small`.

## Implementation and Tests

[`canonical_ssr.py`](../../canonical_ssr.py) preserves six sets, five ordered inputs per set, `RATINGS = (1, 2, 3, 4, 5)`, `PAPER_EPSILON = 0.0`, `PAPER_TEMPERATURE = 1.0`, arithmetic six-set PMF averaging, and the `text-embedding-3-small` adapter. It accepts anchors from the caller and does not embed any alleged original anchor strings.

[`tests/test_canonical_ssr.py`](../../tests/test_canonical_ssr.py) uses deterministic, test-only statements/vectors and a fake embedding client. It validates the equation, parameter behavior, set shape, and probability outputs; it does not verify the original anchor wording or actual OpenAI runtime behavior.

Test command: `python -m pytest --confcutdir=tests tests/test_canonical_ssr.py -q`  
Result: **42 passed in 0.54s**. Tests use a fake client; no live API call was made.

## Original Anchor Evidence

**Not verified:** the exact 30 original purchase-intent statements, their six set IDs, and the literal 1-to-5 ordering within each set. The paper's Appendix C.1 describes the six-set design and intent gradient but does not list the statements in the available text. The local paper extracts likewise omit them.

The official [arXiv v3 record](https://arxiv.org/abs/2510.08338v3) links the PDF, HTML, and TeX source, but no separate supplementary-material or anchor-data link was listed there. The arXiv HTML text does not expose the literal sets; PDF extraction failed and the TeX archive request was rate-limited during this audit. The paper-linked [PyMC Labs SSR repository](https://github.com/pymc-labs/semantic-similarity-rating) provides a generic reusable rater and README examples, not an original purchase-intent anchor configuration. Those generic agreement examples and the project's smartphone-satisfaction anchors are not substitutes.

## Remaining Configuration Evidence

The paper's embedding model name is verified. The inspected paper text does not specify all API/runtime details such as embedding dimension, normalization, batching, or request options. The local adapter selects the specified model name, but no live provider request or author-provided execution configuration was checked. These details remain unverified and must not be inferred.

## Before Claiming Faithful Reproduction

1. Obtain the six original anchor sets verbatim from an author-controlled, versioned source or supplementary dataset, including set identifiers and rating-to-statement mapping.
2. Verify and cite the provenance and confirm each set's statement order is ratings 1, 2, 3, 4, 5. Do not paraphrase or infer missing statements.
3. Confirm any additional embedding configuration from a reproducible author source. The model name alone is not proof of identical runtime configuration.
4. Only then add the supplied sets as a separate, provenance-bearing configuration and add tests that assert its exact contents, mapping, and structure.

Until those items are available and verified, the equation and deterministic implementation are validated, but the original paper's full SSR experiment is not reproduced.
