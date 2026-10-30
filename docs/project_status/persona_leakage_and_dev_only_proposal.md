# Persona leakage: what is affected, and a proposed future experiment

Written 2026-10-29. No persona, split or result was rebuilt or changed.

## Finding
- `data/personas/persona_assignments.csv` assigns all 200 primary-cohort respondents to P1-P4 with fixed rules over price paid and daily usage (`scripts/build_personas.py`, `assign_persona`).
- Persona profile statistics (`persona_profiles.json`, theme counts) and the marginal distributions used to sample synthetic respondent profiles (`scripts/dp_persona_generator.py`, `generate_specs` / `generate_extended`) are computed over all 200 respondents.
- The frozen evaluation split (`data/validation_split/frozen_split_manifest.json`: 97 development, 97 test, stratified by Q21, drawn from respondents with a usable Q26 answer) is a subset of those 200. So the 97 test respondents contributed to the persona rules' calibration, the persona statistics and the synthetic profiles.
- Mitigating facts (from the code and the Step 4 report): Q21 and Q26 text are not inputs to persona rules, profiles or specs. The rule thresholds were chosen by judgement, and the author had seen overall Q21 results, so the rules are not blind to the outcome distribution.

## Implication for held-out evaluation
Held-out comparisons of persona-conditioned synthetic responses with test-half Q21 are not fully independent: information about test respondents' non-outcome attributes reached the generator. The effect on the reported metrics is unquantified; it is likely small because the outcome was excluded, but that is an argument, not a measurement. Reported results should be called preliminary.

## Proposal (separate future experiment, do not overwrite existing results)
1. Reuse the frozen split unchanged.
2. Re-derive persona rules, thresholds, profile statistics and sampling marginals from the 97 development respondents only (plus the respondents with no usable Q26 answer, if the protocol allows).
3. Generate with a new seed, store everything in a new directory (for example `data/dev_only_personas/`), and record hashes of the inputs.
4. Score with the same canonical SSR configuration and compare against test-half Q21 with the existing metric code.
5. Report both results side by side as "all-200 personas" and "dev-only personas".
