# Digital Persona Pipeline: Component Checklist and Implementation Plan

Audit of the repository against the intended workflow: human survey -> data preparation -> Digital Persona Generator -> persona profiles -> persona-conditioned LLM responses -> SSR rating engine (PMF, expected rating) -> synthetic survey dataset -> Validation Agent -> reports. Statuses were checked against files and tests on disk when this document was written.

| # | Component | Status before this work | Evidence | Action |
|---|---|---|---|---|
| 1 | Human survey data preparation | **Complete** | `scripts/clean_human_survey.py`, `data/human_survey_cleaned.csv`, inclusion/exclusion log, quality report, verification report, tests | none |
| 2 | Persona assignments and aggregate profiles | **Complete (descriptive)** | `scripts/build_personas.py`, `data/personas/` | none |
| 3 | Digital Persona Generator (reproducible profiles + respondent specs) | **Partial / disconnected** | specs were produced by `pilot_make_specs.py` (runs on import, writes into `data/pilot/`); no persona cards, no provenance hashes | **new** `scripts/dp_persona_generator.py` (reproduces the 40 pilot specs exactly) |
| 4 | Synthetic Survey Agent (LLM text generation) | **Partial / fragmented** | three separate provider scripts (`pilot_generate_api.py`, `pilot_generate_gemini.py`, `pilot_generate_openai.py`); Gemini v2 has 20/40 responses (P1, P2), OpenAI 0, Gemini v1 truncated | keep scripts (shared prompts); pipeline reads saved outputs only; generation stays an explicit, separate step |
| 5 | SSR rating engine | **Complete, with documented deviations** | `canonical_ssr.py` (42 tests), single adapted Behavioral Intention anchor set, local encoders | unchanged |
| 6 | PMF, expected rating, uncertainty export | **Partial** | per-source score CSVs lacked PMF SD / entropy; scoring script was Gemini-v2 specific | **new** `scripts/dp_synthetic_dataset.py` (all encoders, SD, entropy, encoder spread) |
| 7 | Synthetic dataset with provenance | **Missing** | only per-experiment CSV fragments | **new** `data/synthetic_dataset/` (dataset, coverage, manifest) |
| 8 | Validation Agent | **Partial** | metric and persona-matched scripts existed separately; no consistency or stability checks, no unified report | **new** `scripts/dp_validation_agent.py` |
| 9 | Baselines | **Complete / frozen** | mean and majority baselines in `comparison_metrics.json`; D1 aspect model documented as leakage-flagged diagnostic on a different dataset | preserved, referenced read-only |
| 10 | End-to-end runner | **Missing** | no single entry point | **new** `scripts/run_pipeline.py` |
| 11 | README for the Digital Persona pipeline | **Missing** | README described the older review-rating SSR project (`agent.py`, softmax SSR) | README section added; legacy content untouched |
| 12 | Legacy review-rating SSR (`agent.py`, `batch_predict.py`) | **Separate, disconnected** | uses a softmax SSR variant, not `canonical_ssr.py`; modified uncommitted files in git | not part of this pipeline; not touched |

## Cannot be completed without an external dependency
- Gemini responses for the remaining 20 prompts (P3, P4 and the rest) need quota and `GEMINI_API_KEY`.
- OpenAI responses need account credit, `OPENAI_API_KEY` and a model id.
- The paper's six anchor sets and `text-embedding-3-small` results are unavailable (anchors unrecovered; OpenAI embedding access blocked by credit).
- Repeated-generation stability needs the same specs generated at least twice by the same model.

## Safeguards followed
No edits to `canonical_ssr.py`, approved anchors, the survey workbook, frozen baselines or the Gemini v1/v2 outputs; no LLM or paid API call is made by any new module (a test scans the sources); outputs from different experiments stay in separate folders and rows; output writing never overwrites differing content.
