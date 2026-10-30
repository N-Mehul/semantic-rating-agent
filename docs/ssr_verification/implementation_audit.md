# Existing Implementation Audit Against Maier et al. SSR

## Scope and source of truth

Compared the repository's `five_anchor_ssr.py`, `agent.py`, `batch_predict.py`, `aspect_rating_model.py`, available tests/configuration, README/report/audit documents, saved metrics, the supplied Maier text extracts, the full arXiv v3 HTML, and the paper-linked SSR package. Paper-specific facts are documented in [paper_method.md](paper_method.md).

This report was written before adding any new canonical implementation. No baseline/D1 code or output is changed by the audit.

## Paper vs repository

| Topic | Maier et al. paper | Existing repository | Finding |
|---|---|---|---|
| Synthetic respondent | LLM impersonates a real consumer using the same/subset of demographic attributes; sees product-concept image or text and answers purchase-intent question | No connected reproducible persona-to-survey production pipeline. Persona prompt probes are experiments and focus on smartphone reviews | Missing as end-to-end capability |
| Response generation | Brief free-text purchase-intent response; two samples; GPT-4o or Gemini-2.0-flash; `top_p=0.9`; main-text generation temperature 0.5 | `agent.py` contains a direct numerical rating path; persona probe is separate and not integrated | Different/incomplete workflow |
| Anchor data | Six paper-specific sets, five purchase-intent statements per set; average per-set PMFs | `five_anchor_ssr.py` contains one locally written five-statement smartphone satisfaction set plus generic fallback; no six paper sets | Not paper anchor data; only one domain-specific set |
| Embedding model | OpenAI `text-embedding-3-small` | `agent.py` loads `all-MiniLM-L6-v2`; canonical helper delegates to that agent | Wrong model for paper reproduction |
| Similarity | Raw cosine similarity, Eq. 7 | Existing helper computes cosine, compatible in isolation | Partial overlap only |
| Probability mapping | Minimum-subtract similarities; add epsilon only to minimum class; normalize, Eq. 8 | `five_anchor_ssr.py` applies ordinary softmax to raw similarities | Different equation |
| Paper SSR settings | `epsilon=0`, post-mapping `T=1` | Existing helper defaults `temperature=0.1`; no epsilon or six-set averaging | Different settings and behavior |
| Multiple anchors | Average the six five-class PMFs per response | Existing helper accepts one set of five anchors | Missing six-set aggregation |
| Output | A five-value PMF; expected rating may summarize it | Existing helper returns five probabilities and expected rating, but computed from its different softmax method | Output shape similar; semantics differ |
| Text-only baseline | Repository's held-out text baseline is 46.14%; code uses MiniLM, example-bank top-k=3, softmax `T=0.1` | It remains frozen and is not the paper's canonical SSR | Preserve as a separate baseline |
| D1 model | Not part of paper SSR | Random Forest using five aspect-rating inputs; verified accuracy 73.34%; artifact flags potential target leakage | Separate diagnostic model; not canonical SSR |

## Code paths inspected

- `five_anchor_ssr.py`: isolated one-set anchor helper, calls `SemanticRatingAgent._embed`, then `_softmax(..., temp=0.1)`. Its phone-review/satisfaction anchor text was written for this repository and does not establish the original purchase-intent anchors.
- `agent.py`: `_get_model()` selects `all-MiniLM-L6-v2`; `_embed()` uses batch size 64 and normalized embeddings. `analyze_new_text()`/`predict_text()` use the learned example bank or rating profiles; top-k neighbor weights use temperature 0.1. `predict_rating_agentic()` asks an LLM for a numeric rating, which is not SSR free-text scoring.
- `batch_predict.py`: `predict_single(..., top_k=3)` embeds with the agent, retrieves the top three example-bank entries, and applies softmax temperature 0.1. This is an exemplar-neighbor baseline.
- `aspect_rating_model.py` and `data/aspect_model/metrics.json`: D1 uses five aspect-rating columns as features and the artifact records `POTENTIAL TARGET LEAKAGE — DIAGNOSTIC ONLY`; verified accuracy is 73.3379986% (73.34%).
- `tests/`: pytest is used; current test coverage is in `tests/test_analysis.py`. No tests for `five_anchor_ssr.py` or six-set SSR were found. No root `pyproject.toml`, `pytest.ini`, `setup.cfg`, `tox.ini`, or `conftest.py` was found.

## Existing reported results are not SSR-reproduction evidence

- The text-only result of 46.14% is from the repository's MiniLM example-bank/top-k=3 path, not the original paper's text-embedding-3-small + minimum-subtracted PMF + six-anchor-set method.
- The D1 73.34% accuracy uses five aspect-rating features; it is neither text-only nor a canonical SSR result.
- Neither number demonstrates reproduction of Maier et al.'s reported panel-level distributional similarity/correlation-attainment experiments.

The paper-linked package's current `compute.cosine_similarity_matrix()` maps cosine to `(1 + cosine) / 2`, whereas paper Eq. 7 writes raw cosine. With the paper's `epsilon=0`, this positive affine transformation cancels after minimum subtraction and normalization, so the PMF is unchanged. For nonzero epsilon, adding the same numeric epsilon after that rescaling is not generally equivalent to adding it to raw cosine-derived differences; this task implements the paper's raw-cosine equation.

## Paper-specific differences and unresolved items

1. Replace the ordinary-softmax five-anchor helper with a separate implementation of the paper's Eq. 7–9 method: raw cosine, per-set min subtraction, epsilon indicator term, normalization, and six-set PMF averaging.
2. Keep `agent.py`, `batch_predict.py`, the top-k=3 baseline, its artifacts, and D1 model/output files unchanged.
3. Accept the paper's six anchor sets as explicit inputs until the exact text is recovered. The appendix describes their properties but does not print their literal strings in the supplied extraction or v3 HTML. The locally authored smartphone-review anchors and README examples are not substitutes.
4. Bind a real runner to OpenAI `text-embedding-3-small` before claiming a full operational reproduction. An injected encoder can validate the deterministic mathematics without making API calls, but it does not verify remote embeddings or reproduce the paper's experiment.
5. Exact system-prompt text, embedding dimensions/normalization/batch settings, and the tie behavior for equal minimum similarities are not fully specified in the paper text and remain unresolved.

## Smartphone adaptation boundary

The six future constructs PE, EE, HM, SI, PV, and BI are not the original paper's purchase-intent construct. The paper itself notes that new questions may require their own reference sets. Any six-by-five smartphone anchor library must be designed and validated as a **proposed domain-specific adaptation**, not attributed to Maier et al. No smartphone anchors or synthetic responses are generated in this audit.
