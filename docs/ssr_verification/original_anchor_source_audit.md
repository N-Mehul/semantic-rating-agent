# Original SSR Anchor Source Audit

**Audit date:** 2026-10-11  
**Paper:** Maier et al., *LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings*, arXiv:2510.08338v3 (27 October 2025).

## Finding

The exact six sets of five purchase-intent anchor statements could not be verified from repository files or the official paper materials inspected. No anchor strings are reproduced or inferred here. No anchor configuration or anchor-content tests were added because their contents and provenance are unavailable.

The paper does verify that each set has five reference statements corresponding to Likert ratings 1 through 5, that there are six similar but non-identical sets, and that the six per-set PMFs are averaged. Appendix C.1 describes the intent gradient (unlikely purchase at the low end, indifference/uncertainty in the middle, intent/possible intent at the high end, with intermediate statements between) but does not print the literal statements in the available paper text. Therefore the conceptual rating order is documented, but the actual order/content of statements in each original set cannot be checked.

## Sources Inspected

### Official paper record and materials

- [arXiv v3 record](https://arxiv.org/abs/2510.08338v3): confirms the version and provides official PDF, HTML, and TeX-source links.
- [arXiv v3 HTML](https://arxiv.org/html/2510.08338v3): inspected the paper sections on methods and Appendix C.1–C.2. The text describes six sets and their intended gradient, but does not provide the six literal five-statement lists.
- [arXiv v3 PDF](https://arxiv.org/pdf/2510.08338v3): the PDF is not present in the project. The webpage extraction tool could not extract the PDF. The corresponding TeX-source download attempt was rate-limited, so direct inspection of the source archive could not be completed in this session.
- No supplementary-material link or separate anchor-data download was listed on the arXiv v3 record inspected.

### Official paper-linked implementation

- [PyMC Labs SSR repository](https://github.com/pymc-labs/semantic-similarity-rating): Appendix C.2 points to this repository. Its current tree contains the reusable `ResponseRater`, computation module, tests, project files, and README, but no purchase-intent anchor dataset/configuration file.
- The package README examples use generic agreement/disagreement reference sets and explicitly show reference text as user-provided inputs. Those examples are not identified as the six purchase-intent sets used in the paper and must not be substituted for them.
- Lexical searches of the repository for the paper's described anchor wording and purchase-intent phrases returned no original anchor-set data.

### Local project

A repository-wide search found:
- `docs/ssr_verification/source_excerpts/maier_appendix_c.txt` and `docs/ssr_verification/source_excerpts/maier_section_a43.txt`: extracted paper text; C.1 contains the anchor-set description but no literal sets.
- `docs/ssr_verification/source_excerpts/match_57037.txt`: a duplicate paper-text extract with the same omission.
- `five_anchor_ssr.py`: one project-authored smartphone satisfaction anchor set, not six paper purchase-intent sets.
- `pymc_response_rater.py`: a local package-style rater with generic README-style example statements, not the paper's six sets.
- `canonical_ssr.py` and `tests/test_canonical_ssr.py`: accept caller-supplied anchor sets; current tests use deterministic test-only labels/vectors, not purported original statements.
- No local PDF, TeX source archive, supplement, or anchor-data file was found.

## Paper Settings Compared with the Isolated Implementation

| Requirement | Paper verification | `canonical_ssr.py` comparison | Status |
|---|---|---|---|
| Six ordered sets, five ratings per set | Appendix C.1 and A.4.3 specify six sets and five statements per 1–5 scale | Module requires six sets; text interface requires five statements in each set | PASS for structure |
| Exact original statement text and per-set ordering | Not listed in C.1 text; no original set artifact found | Inputs remain caller-supplied; no strings are claimed as original | NOT VERIFIED |
| Embedding model name | A.4.3 names OpenAI `text-embedding-3-small`; it reports `text-embedding-3-large` tests as virtually unchanged | `EMBEDDING_MODEL` and OpenAI adapter use `text-embedding-3-small` | PASS for model name |
| Embedding request/runtime configuration | The paper text inspected does not specify dimension, normalization, API request options, or batching; no live API call was verified | Adapter forwards text to an injected client; no runtime configuration is pinned beyond the model name | NOT VERIFIED |
| Epsilon | A.4.3 says study uses `epsilon = 0` | Default is `PAPER_EPSILON = 0.0` | PASS |
| SSR temperature | A.4.3 says study uses `T = 1` | Default is `PAPER_TEMPERATURE = 1.0` | PASS |
| PMF and six-set aggregation | Paper subtracts the per-set minimum similarity, normalizes, optionally temperature-scales, then averages the six per-set PMFs | Module implements minimum subtraction, epsilon term, power temperature scaling, and arithmetic mean across six sets | PASS for equation/path; original anchors remain unverified |
| Rating order in supplied original strings | Paper says one reference per integer 1–5, but literal lists are absent | Module preserves the caller-provided order and does not infer/correct statement ordering | NOT VERIFIED for original sets |

These findings do not establish full paper reproduction. The model name is source-verified, but actual service behavior/configuration and the original anchor strings remain open.

## Blockers and Evidence Needed

1. Obtain the exact six five-statement sets from an author-controlled supplementary artifact, a versioned source/data release, or the corresponding author. The paper's Appendix C.1 description and the package's generic examples are insufficient provenance.
2. Verify that each original set lists the five statements in rating order 1, 2, 3, 4, 5 before storing it as configuration. Keep each set verbatim with a source/version citation; do not normalize or paraphrase its language.
3. Verify the intended OpenAI embedding API configuration from the authors or a reproducible study script. The paper establishes the model name but not all API/runtime details.
4. After those materials are obtained, add a separate configuration file and provenance/shape/order tests. Until then, the current API's caller-supplied test anchors remain test fixtures only.

## Scope

No code, tests, anchor definitions, project model outputs, baseline artifacts, or D1 artifacts were changed for this source audit. No personas or synthetic responses were generated. No full-reproduction claim is made.
