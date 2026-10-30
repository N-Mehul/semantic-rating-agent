# Legacy direct-rating and kNN paths (not the canonical SSR pipeline)

Written 2026-10-29. Purpose: make sure nothing here is mistaken for the target pipeline, in which the LLM writes free text and the rating is derived geometrically by `canonical_ssr.py`. Code behaviour and outputs were not changed; only labels were added.

## Code that asks an LLM for a numeric rating

| Location | What it does |
|---|---|
| `agent.py` `_AGENTIC_SYSTEM_PROMPT` (about L3239) | System prompt demanding `{"rating": <integer 1-5>, "reasoning": ...}` |
| `agent.py` `_build_agentic_prompt`, `_parse_agentic_response`, `predict_rating_agentic` | Builds the prompt, parses the integer, calls Groq or Ollama |
| `batch_predict.py` `run_agentic_batch` and `--mode agentic` / `--mode all` | Runs the above over a review CSV and writes `data/agentic_predictions.csv`; the mode's old description said "SSR sampling", which was misleading |
| (removed) | Two scratch probes of the same functions (`test_agentic_unit.py`, `test_groq_single.py`) were deleted in the final cleanup; see `CLEANUP_LOG.md` |
| `_archive/evaluate_agentic.py` | Evaluator for the agentic output (archived 2026-10-29) |

Nothing in `scripts/`, `dashboard/` or `tests/` calls these functions. The pipeline prompts (`scripts/pilot_generate_*`, `dashboard/workflow.py`) tell the model to use words only, and `ssr_eval_lib.response_quality` flags responses containing ratings.

## kNN example-bank scoring (also legacy)

`agent.py` `_build_example_bank` and `predict_text` (top_k = 3), `answer_question`, and `batch_predict.py` `predict_single` / default `--mode deterministic`. These produce the review-rating baselines under `data/` (for example `data/predictions.csv`, `data/current_baseline_evaluation/`). They are kept because those baselines depend on them. `analysis/label_structure.py` imports this path. The `k = 3` setting belongs here and was not changed.

## What was changed

- LEGACY banners in the module docstrings of `agent.py` and `batch_predict.py`, and above the agentic section of `agent.py`.
- `run_agentic_batch` docstring corrected, and it prints one `[LEGACY]` line to stderr when invoked. Predictions and file outputs are unchanged.
- No function was removed, renamed or modified, so existing commands (`python batch_predict.py ...`) still work.

## Recommended next step (owner)

Rewrite or retire these paths only after deciding whether the review-rating baselines must remain reproducible. If retired, move `agent.py`, `batch_predict.py` and `analysis/label_structure.py` together, because they import each other.
