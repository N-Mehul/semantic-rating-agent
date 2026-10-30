# Full Project Audit (2026-10-29)

Scope: Digital Persona-Based Synthetic Consumer Research using Semantic Similarity Rating (SSR). This audit was done in place; no project was rebuilt. Every status below was checked against code, saved data, or a command run on 2026-10-29. Things that could not be checked are marked **unverifiable**.

## 1. What the project does

Human survey (215 raw / 200 primary respondents) -> four rule-based personas (P1-P4) -> construct-specific prompt -> LLM writes free text (Gemini / OpenAI / local Ollama `qwen2.5:0.5b`) -> sentence embedding (three local encoders) -> cosine similarity with the five approved anchors of the construct -> canonical SSR probability mass function -> expected rating E[R] = sum(i * P(i)). The LLM is never asked for a number. A single-screen dashboard (`python dashboard/server.py`) walks through the six steps.

A separate legacy line of work (mobile-phone review rating with a kNN example bank, `agent.py`, `batch_predict.py`) exists. It is not part of this pipeline; see section 8.

## 2. Architecture and directory structure

| Path | Role |
|---|---|
| `canonical_ssr.py` | Canonical SSR mathematics (protected) |
| `survey_ssr_bridge.py` | Loads and validates approved anchors from `docs/survey_instrument/smartphone_construct_anchor_statements.csv` |
| `five_anchor_ssr.py` | Earlier five-anchor implementation (protected, superseded; depends on `agent.py`) |
| `scripts/` | Pipeline: `clean_human_survey`, `build_personas`, `dp_*` (persona generator, constructs, dataset, validation), `pilot_generate_{gemini,openai,api,condition}`, `ollama_utils`, `gemini_utils`, `ssr_*` (encoders, persona-matched evaluation), `run_pipeline`, `verify_pipeline_outputs` |
| `dashboard/` | HTTP server (127.0.0.1 only), workflow, product research, survey batches, static front end |
| `analysis/`, `experiments/d1_accuracy/` | Evaluation utilities and grouped-CV model comparison (legacy review-rating line) |
| `tests/` | pytest suite |
| `docs/` | Survey instrument and anchors, SSR verification, project status reports |
| `data/` | Raw and cleaned survey data, personas, prompts, saved LLM responses, synthetic dataset, validation outputs, historical experiment outputs |
| `results/`, `paper/`, `_archive/` | Saved metrics; paper; legacy and superseded files (see `_archive/README.md` and `CLEANUP_LOG.md`) |

## 3. Commands

```
python scripts/run_pipeline.py all          # personas -> prompts -> dataset -> validate -> status (never calls an LLM)
python scripts/verify_pipeline_outputs.py
python -m pytest --confcutdir=tests tests/
python dashboard/server.py [--port 8765]    # http://127.0.0.1:8765
```
LLM generation is separate and explicit: `scripts/test_gemini_api.py`, `scripts/pilot_generate_gemini.py`, `scripts/pilot_generate_openai.py --model <id>`; Ollama generation runs from the dashboard.

## 4. Five-stage audit

### Stage 1 - Human survey and preparation: **Implemented, verified**
- `scripts/clean_human_survey.py`; workbook `data/Capstone_Reviews(Responses).xlsx` (unmodified; verifier confirms hash).
- Re-counted from `data/human_survey_inclusion_exclusion_log.csv`: 215 raw; 4 no consent; 6 under 18; 1 not a smartphone owner (HS051 has two reasons); 4 unresolved age eligibility; **200 primary cohort** (204 before the unresolved-age exclusions). Matches the prior audit; nothing changed.
- 195 respondents have at least one free-text answer, 181 have all four.
- Limitation: convenience sample, mostly students; no de-identification review beyond the `HSnnn` ids was performed in this audit (**unverifiable** here).

### Stage 2 - Digital personas: **Implemented, verified**
- `scripts/build_personas.py`: rule-based segments over non-outcome variables (price paid, usage hours); `persona_assignments.csv` has 200 rows over 4 personas (P1-P4). Q21 and Q26 are excluded from profiles. Personas are descriptive segments, not statistically derived clusters (stated in the Step 4 report). P4 is small and mixes budget buyers with respondents who gave no price.
- **Leakage risk (open):** assignments and profile statistics use all 200 respondents, including the 97 respondents in the frozen test half. Outcomes were not used, but the author had seen overall Q21 results, so persona-to-outcome comparisons are not fully blind.

### Stage 3 - Persona-conditioned generation: **Implemented; partly exercised**
- Persona prompt: `data/construct_prompts/*_prompts.jsonl` (`system` + `user` containing "Respondent profile: ..."). Control: same system prompt and question with no profile.
- Live test on 2026-10-29 through `scripts/ollama_utils.generate` with `qwen2.5:0.5b`: persona prompt included the profile, control prompt did not (asserted in the test). Both returned real text.
- Error handling present: probe, timeouts (`CHAT_TIMEOUT_S`, per-prompt `timeout_s`), error classification, empty-response detection, `response_quality` (flags truncation, missing terminal punctuation, numbers/ratings in text).
- Saved responses: Gemini v1 20/40 (frozen, 19 of 20 truncated, disabled by default), Gemini v2 20/40 (P1 and P2 only, stopped on quota errors), OpenAI 0/40. Construct-specific FT_PE01 / FT_BI01 responses: none generated.
- Not verified here: live Gemini or OpenAI generation (no API keys used); live batch generation is documented in the README as tested with mocked providers only.

### Stage 4 - SSR: **Implemented, verified**
- `canonical_ssr.py`: `cosine_similarities` -> `similarities_to_pmf` (subtract the minimum similarity, divide by the sum; epsilon 0, temperature 1) -> `expected_rating` = dot((1..5), P). Probabilities are validated to sum to 1 (tolerance 1e-12). Equal similarities raise an error rather than inventing a distribution.
- Live check: persona response gave P = [0, 0.0567, 0.1298, 0.4252, 0.3883], sum 1.0, E[R] 4.1451, equal to the manual sum; control gave E[R] 4.0170. Encoder `BAAI/bge-base-en-v1.5` (the dashboard default of three local encoders), PE construct.
- Unsupported construct: `anchor_info('ZZ')` raises `KeyError` listing the available constructs; no rating is produced.
- Deviations from the paper / diagram (documented, not changed):
  1. The diagram's "softmax" is not used: the approved method is minimum-subtraction normalisation, so the least similar anchor always gets P = 0.
  2. The app uses **one** adapted anchor set per construct. `canonical_ssr.score_text_over_six_sets` (paper's six sets) is implemented and tested but is not used by the app.
  3. The anchors are labelled `PROPOSED_DOMAIN_ADAPTATION_NOT_ORIGINAL_MAIER_ANCHOR`.
  4. The paper's embedding model (`text-embedding-3-small`) is supported only by an injected-client adapter; the app uses local encoders.
  5. `survey_ssr_bridge.CONSTRUCT_CODE_MAP` used to map `FC` (Facilitating Conditions) to Perceived Protection. The alias was removed on 2026-10-29: FC now returns unsupported. Caveat: older handover documents listed FC as the fifth construct, while the codebook, instrument and anchor CSV contain PP (the instrument marks its PP definition as a working interpretation). Which name is intended is an open decision for the research team.
- `k=3` is the nearest-neighbour setting of the legacy kNN path (`agent.py`, `batch_predict.py`), not of canonical SSR. It was not modified.

### Stage 5 - Validation: **Partially implemented**
- Frozen split: `data/validation_split/frozen_split_manifest.json` (seed 20261028, 97 development / 97 test, stratified by Q21). Preserved; hash unchanged.
- Metrics implemented in `scripts/ssr_eval_lib.py` and `dp_methodology_eval.py`: Spearman with bootstrap CI, MAE against a mean-predictor baseline, exact accuracy against the majority class. Saved results (Step 11 report): Spearman with Q21 about 0.15-0.30 depending on encoder, MAE close to the baseline, accuracy not clearly above majority. Pearson, Cohen's kappa, KS, Wasserstein and Jensen-Shannon: not confirmed as saved results in this audit.
- TSTR: **not computable** (labels would be SSR-derived and the saved synthetic responses fall into two classes). No TSTR value was fabricated.
- Remaining risks: proxy target (Q21 asks about rebuying the brand, anchors describe continued use); persona definitions use the test half; encoders disagree on response ordering (verifier-confirmed in the report); only 20 Gemini v2 responses exist for two personas.

## 5. Dependencies

Entry points and the import graph are in the 2026-10-29 Phase 1 audit. No kept file imports an archived file. `tests/` imports only kept modules.

## 6. Verification on 2026-10-29

| Check | Result |
|---|---|
| `python scripts/run_pipeline.py all` | exit 0 |
| `python scripts/verify_pipeline_outputs.py` | exit 0, 39/39 checks passed |
| `python -m pytest --confcutdir=tests tests/` | exit 0, 491 passed, 2 warnings (ConstantInputWarning in `tests/test_ssr_eval.py`), 191 s |
| Dashboard on port 8791 | `/`, `/api/health`, `/api/workflow/options`, `/api/personas`, `/api/validation`, `/api/limitations`, `/api/simple/options`, `/api/product/options` returned 200; `/api/response` without an id returned 400; unknown route 404; server stopped afterwards |
| Live Ollama request (persona and control) | succeeded, then scored with canonical SSR |
| Protected-file hashes (210 files, before and after) | 209 identical. The one difference is `data/dashboard_demo_runs/runs.jsonl`, an append-only run log (38 lines now). Dashboard workflow tests and the live dashboard session appended entries stamped 2026-10-29 (e.g. 14:32, 14:36, 14:44 UTC). No research input, anchor, split or response file changed. The entries were left in place since the new lines were not separately identified; see limitations |

## 7. Cleanup

See `CLEANUP_LOG.md` at the repository root.

## 8. Limitations and recommendations

- Legacy kNN / LLM-asked-rating code (`agent.py:3312 predict_rating_agentic`, `batch_predict.py:306 run_agentic_batch`) remains and contradicts the "never ask for a number" principle if used. It is outside the pipeline; marked for rewrite by the owner.
- Duplicated cosine/softmax logic exists in `agent.py`, `batch_predict.py` and `five_anchor_ssr.py`; `canonical_ssr.py` is the only tested copy.
- `five_anchor_ssr.py` (superseded; depends on `agent.py`) and `README.md` still describe or depend on older material; update the README when the owner rewrites the legacy path.
- Test isolation: fixed on 2026-10-29 (`tests/conftest.py`, `tests/test_run_log_isolation.py`, env var `DP_DASHBOARD_RUN_LOG_DIR`). The existing 38 `runs.jsonl` entries were kept because their origin is unproven.
- Decide whether the fifth construct is Perceived Protection or Facilitating Conditions (see Stage 4 deviation 5).
- Collect more persona-conditioned responses (P3, P4, construct-specific questions) before drawing any conclusions about persona fidelity.

## 10. Hardening pass results (2026-10-29)
- Run-log pollution fixed (see `CLEANUP_LOG.md`); `runs.jsonl` hash unchanged after a full test run (494 passed, 2 warnings), pipeline `all` exit 0, verifier 39/39 exit 0, 153 protected non-code files byte-identical.
- `FC` alias removed: the instrument has six constructs (PE, EE, HM, SI, PP, BI); there are no Facilitating Conditions items or anchors, and Perceived Protection means privacy/security of personal data (PP01-PP04). FC now returns unsupported, never PP anchors.
- Legacy direct-rating code labelled, not changed: `legacy_direct_rating_paths.md`.
- Persona leakage documented with a dev-only proposal: `persona_leakage_and_dev_only_proposal.md`. Synthetic profiles are also sampled from marginals over all 200 respondents.
- New Ollama generation (model `qwen2.5:0.5b`, canonical SSR, local_bge_base): FT_PE01 persona 20/20 scored (mean E[R] 4.04), control 20/20 (3.68, wider spread); FT_BI01 persona 18/20 scored, 2 excluded by the quality check (3.54), control 20/20 (3.41). Five profiles per persona for P1-P4. This is coverage, not validation: no human reference exists for FT_PE01/FT_BI01, and a 0.5B model's text is not evidence of real consumer attitudes.
- Still missing: Gemini/OpenAI responses for P3/P4 and the new questions (no keys used), Gemini v2 beyond 20/40, OpenAI 0/40.
