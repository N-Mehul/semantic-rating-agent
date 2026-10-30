# Canonical SSR Verification Record

## Implementation added

Added [canonical_ssr.py](../../canonical_ssr.py), an isolated implementation of the paper's response-to-PMF calculations. It does not import `agent.py`, alter `five_anchor_ssr.py`, invoke an LLM, create personas, or write experiment outputs. The existing MiniLM/top-k=3 text baseline and D1 aspect model remain separate and unchanged.

The new module accepts six ordered groups of five anchor statements and a caller-provided embedding function. `openai_text_embedding_3_small()` is an explicit adapter that requests the paper-named embedding model from an injected OpenAI client; no API key, client, or remote call is created implicitly. Deterministic tests use fixed vectors and test-only anchor strings, not the paper's undisclosed original anchor texts.

## Parameters and equation

Defaults match the reported study: `PAPER_EPSILON = 0.0`, `PAPER_TEMPERATURE = 1.0`, six anchor sets, five ratings per set, and embedding model name `text-embedding-3-small`.

For raw cosine similarities $s_{i,r}$ in anchor set $i$, let $\ell_i=\arg\min_r s_{i,r}$. The implementation uses:

$$
w_{i,r}=s_{i,r}-s_{i,\ell_i}+\epsilon\,\mathbf{1}[r=\ell_i], \qquad
p_i(r)=\frac{w_{i,r}}{\sum_{k=1}^{5}w_{i,k}}.
$$

Optional paper Eq. 9 scaling is:

$$
p_i(r,T)=\frac{p_i(r)^{1/T}}{\sum_{k=1}^{5}p_i(k)^{1/T}}.
$$

The per-set distributions are averaged:

$$
p(r)=\frac{1}{6}\sum_{i=0}^{5}p_i(r,T), \qquad
\mathbb{E}[r]=\sum_{r=1}^{5}r\,p(r).
$$

Probability validity is checked with absolute tolerance `1e-12`: every value must be finite and in [0, 1], and the five probabilities must sum to 1. Equal similarities with epsilon zero make the source equation's denominator zero; the module reports this as undefined instead of silently inventing a fallback. For tied minima and epsilon greater than zero, deterministic `argmin` selects the first (lowest-rating) index; the paper does not specify tie handling.

## Automated test results

Run command: `python -m pytest tests/test_canonical_ssr.py -v`

The test suite covers raw cosine values, paper min-subtraction and epsilon placement, PMF normalization and five-rating shape, six-set averaging, expected rating, one-call encoding of a fixed text plus 30 anchors, model selection in the OpenAI adapter, and undefined zero-denominator handling.

**Run result:** 10 passed in 0.52 seconds on Python 3.13.2 / pytest 9.1.1. The first run exposed an invalid test fixture with identical anchor vectors; after changing only that deterministic fixture to use varied embeddings, the same focused command passed all 10 tests. No live embedding API call was made.

Final focused rerun: `python -m pytest --confcutdir=tests tests/test_canonical_ssr.py -v` — **10 passed in 0.44 seconds**. The `--confcutdir=tests` option prevents unrelated scripts under `scratch/` from being collected.

Final scoped regression run: `python -m pytest --confcutdir=tests tests/test_canonical_ssr.py tests/test_analysis.py -q` — **135 passed in 5.24 seconds**.

Full-suite attempt: `python -m pytest -q` did not complete. Pytest collected `scratch/test_groq_single.py`, which raised during collection because `GROQ_API_KEY` is not set; execution then ended with `KeyboardInterrupt` during model/network activity. This is an existing credential-dependent scratch integration script, not a failure in the canonical SSR tests. The full repository suite is therefore unverified.

## Verification status and unresolved differences

- **Mathematical implementation:** implemented for caller-supplied embeddings/anchor texts, including the paper's six-set averaging and reported epsilon/temperature settings.
- **Deterministic implementation check:** passed for fixed, test-only embeddings and anchor strings; this verifies the equation and aggregation code without external services.
- **Full original-paper reproduction:** not yet verified. The supplied Appendix C.1 extraction and the full v3 HTML describe the six anchor sets but do not contain their literal statements. No substitute anchors are silently installed. Exact reproduction requires obtaining the authors' six original sets and making them explicit inputs.
- **Embedding-model operation:** the adapter selects `text-embedding-3-small`, but no OpenAI API call was made. Remote model output and exact reproduction cannot be validated without an authorized client/credentials. The paper does not specify embedding dimension, normalization, batch size, or API options.
- **Persona/generation stage:** not implemented or run, by request. The paper's human-persona conditioning, concept stimuli, prompt wording, and LLM generation temperatures are documented in [paper_method.md](paper_method.md), not recreated here.
- **Smartphone constructs:** PE, EE, HM, SI, PV, and BI are future application targets. Their six-by-five anchor sets would be proposed domain-specific adaptations and require construct-specific validation; none are presented as original paper anchors or generated in this task.
- **Reported project metrics:** 73.34% D1 accuracy uses five aspect-rating inputs; 46.14% belongs to the separate MiniLM/top-k=3 text baseline. Neither is evidence of canonical SSR reproduction.

## Next step

Obtain the authors' exact six purchase-intent anchor sets (or a versioned supplementary/source artifact that contains them), then run the isolated module with those texts and the specified embedding service. Only after that can the whole paper-specific SSR scoring path be called reproduced. Smartphone construct adaptations should follow as a separate, explicitly labeled validation study.
