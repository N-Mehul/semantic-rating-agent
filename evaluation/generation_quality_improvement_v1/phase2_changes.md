# Phase 2: Generation Prompt and Context Corrections

**Date:** 2026-11-04 · **Product:** Nova AI Smartphone (₹24,999, `scripts/run_product_evaluation.py`) · **Model:** Qwen 2.5 7B Instruct via local Ollama
**Basis:** `phase1_generation_audit.md`. SSR maths, anchors, encoder, k = 3 and T = 0.75 were not touched; no SSR or validation was run, and only one diagnostic generation call was made.

## 0. Design decision: a new v2 module, legacy builder untouched

The legacy builder `build_persona_multiconstruct_prompt` (`scripts/product_evaluation_prompts.py`) is part of your uncommitted work. It is used by `dashboard/evaluation_service.py`, and your own tests assert its current behaviour:
- `tests/test_persona_bi_calibration.py` requires the sentence "I am certain I will continue using this smartphone".
- `tests/test_product_evaluation_prompts.py` requires "Current smartphone:" and "pocket-friendly".

Fixing those prompts in place would overwrite your work and break those tests. All corrections are therefore in a **new module**, `scripts/product_evaluation_prompts_v2.py`, which the Phase 3 generation will use. Nothing existing was edited.

| File | Status |
|---|---|
| `scripts/product_evaluation_prompts_v2.py` | **new**: v2 prompts, context planning and diagnostics, parser |
| `tests/test_product_evaluation_prompts_v2.py` | **new**: 7 offline tests |
| `evaluation/generation_quality_improvement_v1/run_phase2_pilot.py` | **new**: one-call pilot; refuses to run twice |
| `evaluation/generation_quality_improvement_v1/phase2_diagnostic_pilot.json` | **new**: pilot record |
| `evaluation/generation_quality_improvement_v1/phase2_changes.md` | **new**: this file |
| `scripts/ollama_utils.py`, `scripts/product_evaluation_prompts.py`, `dashboard/*`, codebook, anchors, data | **unchanged**. `ollama_utils.generate` already accepts a per-call `num_ctx`, so no edit was needed. |

## 1. Task 1: Context window

| | Before (legacy, 6 constructs) | After (v2, 6 constructs) | After (v2, PP only, pilot) |
|---|---:|---:|---:|
| Prompt characters | 9,742–9,846 | 6,767–6,841 (**−31%**) | 3,696 |
| Prompt tokens | ≈ 2,080–2,100* | ≈ 1,440–1,460* (planning estimate 1,644–1,661) | **788 (measured)** |
| `num_ctx` | 2,048 (fixed in `ollama_utils.SETTINGS`) | **3,072** (per call) | 2,048 (per call) |
| Output budget (`num_predict`) | 800 | 700 (110 per construct + 40) | 150 |
| Fits? | **No**: the prompt alone exceeds the window, so Ollama silently drops the start (the persona and product blocks) | Yes: 1,411–1,428 tokens spare by the conservative estimate | Yes: 1,260 tokens spare (measured) |

\*Converted at the **4.69 characters/token** measured in the pilot (Ollama `prompt_eval_count`, including the chat template).

**How the value is chosen.** `plan_context()` estimates prompt tokens at 4.2 chars/token, which deliberately over-estimates by about 12%. It adds the output budget and a 15% margin, then picks the smallest standard size from 2048/3072/4096/6144/8192. For the six-construct prompt: 1,661 + 700 = 2,361 × 1.15 = 2,715, so **3,072**. Each call carries its own `num_ctx`, so the global default is unchanged and the dashboard is unaffected.

**Hardware implications.** For Qwen 2.5 7B (28 layers, 4 KV heads × 128 dims, fp16), the KV cache is about 57 KB per token: about 117 MB at 2,048 and about 176 MB at 3,072. That is **+59 MB** of RAM against a 4.7 GB model, which is negligible. Each step costs slightly more attention compute, but that is far outweighed by the 31% shorter prompt.

**Diagnostic.** `plan_context()` and `context_diagnostic()` record:
- `prompt_tokens_estimate` and `prompt_tokens_actual` (from Ollama)
- `num_ctx`
- `generation_token_budget` (estimated and actual)
- `max_output_tokens`
- `truncation_likely`

## 2. Task 2: Ownership contradictions

| Before | After |
|---|---|
| The system prompt says "You do NOT own the evaluated product or any competing device", while the persona block says "Current smartphone: Flagship-tier smartphone" and BI must say "continue using this smartphone as my primary phone" | One consistent frame: "*The survey asks about 'your current smartphone'. Answer as if the Nova AI Smartphone described above were your current smartphone. This is a hypothetical: judge it from its listed features and price and your own profile; do not describe events that did not happen.*" |
| The persona block was the same for every respondent of a segment | Segment line plus **one real respondent spec** of that persona, drawn with a seed from `respondent_specs_extended_v2.jsonl`: price paid, monthly spending, daily use, main uses, tech comfort, replacement frequency, info sources. It records `respondent_spec_id`. Brand is omitted so the model doesn't name brands. |

This matches the anchors' first-person "my current smartphone" framing without inventing ownership history. The rules forbid invented incidents, repairs and long usage history.

## 3. Task 3: Exact PP correction

**Removed:** the legacy rules "*In PP, evaluate physical build quality, hardware durability, and reliability. Do NOT fabricate data privacy … concerns*" and "*PP: Physical build durability, hardware reliability, and longevity over time. Discuss data security ONLY if …*". These are still present in the legacy module, which was left untouched, but are not used by v2.

**New PP block** (codebook PP01–PP04 verbatim; definition from the instrument's working interpretation):
```
[PP] Perceived Protection
Definition: How well the phone protects your personal information and accounts from unauthorized access and misuse,
and how much control its privacy and security controls give you.
Survey statements: My current smartphone protects my personal information from unauthorized access.; I feel that my
personal data are protected when I use my current smartphone.; The privacy and security controls on my current
smartphone give me control over my information.; I feel safe using my current smartphone for activities involving
personal information.
Write about: privacy and data protection: personal data sent to the cloud versus processed on the device,
location-based suggestions, account and payment safety, control over your information.
Do not write about: physical durability, build quality or hardware reliability (not part of this construct).
```
The cues map to Nova's listed features ("On-Device AI Processing … reduce the need to transmit personal data"; "suggestions based on time and location"). No new features are introduced.

## 4. Task 4: Exact HM correction

**Removed:** "*How pleasant, satisfying, and enjoyable it feels to use for daily tasks and media viewing based ONLY on listed features. (CRITICAL: Do NOT mention gaming …)*".

**New HM block** (HM01–HM04 verbatim):
```
[HM] Hedonic Motivation
Definition: How enjoyable, fun, entertaining and pleasurable using the phone is.
Survey statements: Using my current smartphone is enjoyable.; I have fun using my current smartphone.; Using my current
smartphone is entertaining.; I find using my current smartphone pleasurable.
Write about: fun and pleasure: taking and enhancing photos and short videos, casual gaming, entertainment, the smooth display.
Do not write about: productivity, task usefulness or work/study benefits (that is PE).
```
The gaming ban is lifted because Nova lists "casual gaming, and entertainment". PE now explicitly excludes enjoyment, so the two constructs are cleanly separated.

## 5. Task 5: Construct separation

There is still **one call per persona** (efficiency), but each construct is a self-contained block in the same order: `[CODE] name`, `Definition`, `Survey statements` (verbatim codebook), `Write about`, `Do not write about`. The output is a JSON object with exactly the requested keys.

**Anchors are not shown to the model, deliberately.** Showing the verbatim SSR anchors would let the generator copy their wording, which would inflate SSR agreement artificially (target leakage). The pilot also showed that showing even the **scale labels** makes Qwen print a rating (§8). The survey statements carry the construct meaning; anchor content is unchanged and used only by SSR.

## 6. Task 6: Restrictive rules

| Legacy | v2 |
|---|---|
| About 15 "NEVER" rules, duplicated in the system and user messages, plus about 40 regex rewrites | 5 rules, stated once |
| Price concerns forbidden in 5 constructs and deleted by the sanitizer | Price belongs to BI; concerns are allowed anywhere they're genuine; **no content rewriting** (the parser only reads JSON) |
| Positive adoption by default; the forced opening sentence | "*Take your own honest position. It can be positive, mixed or negative; mention a real concern when you have one.*" |
| "Pocket-friendly" for both P1 and P4 | Each persona judges ₹24,999 against its own survey price tier (`Segment`, `Price you paid…`) |
| A parse failure copies the raw text into all constructs | The construct is left **missing** and flagged |
| Kept safeguards | Only the listed product facts; no invented features or events; words only (no ratings, scale points or labels, stars, %); no other brands; JSON only. These are checked by `RATING_LEAK`. |

## 7. Task 7: Efficiency
- The prompt is 31% shorter, with no duplicated rule lists.
- `num_predict` dropped from 800 to 700.
- Still 4 calls for 24 responses. No judging or quality stage was added and nothing regenerates automatically.

## 8. Task 8: Diagnostic pilot (one call; `phase2_diagnostic_pilot.json`)

Setup: P2 mid-range heavy user (spec `P2_S08`), PP only, Nova AI Smartphone, `qwen2.5:7b-instruct`, temperature 0.7, `num_ctx` 2048.

| Check | Result |
|---|---|
| Truncation | **None**: 788 actual prompt tokens, 1,260 spare; `finish_reason` STOP |
| Elapsed | **63.3 s** (61 output tokens; includes any model warm-up) |
| Privacy / data protection | ✔ on-device processing, location-based suggestions, "who gets my data" |
| Durability content | ✔ none |
| Product context | ✔ "This phone claims to process selected AI tasks locally…"; no other device |
| Naturalness / stance | ✔ natural, 2 sentences, honest mixed view |
| Invented events | ✔ none |
| **Direct rating** | ✘ **FAIL**: the output began with "**3 Neither agree nor disagree ·**" |

Output: *"3 Neither agree nor disagree · This phone claims to process selected AI tasks locally, which sounds good for privacy, but I'm not sure if it's enough. The location-based suggestions could be a bit invasive, and I'm still concerned about who gets my data."*

**Cause:** the pilot prompt printed the answer-scale labels, and Qwen answered with one. The pilot's leak check also missed it.

**Corrections, made without re-running** (per the one-call rule):
1. Scale labels were removed from the prompt.
2. System rule 4 now forbids stating "an answer option, scale point or scale label (for example a number such as 3, or 'Neither agree nor disagree')".
3. The task line asks for "your own everyday words (not an answer option or scale label)".
4. `RATING_LEAK` now flags a leading 1–5 and verbatim scale labels; re-parsing the pilot output now flags it.

**The corrected prompt has not yet been confirmed by a generation call.** Check the first Phase 3 call; the parser flags any recurrence.

## 9. Estimated time for the final 24 responses
- The legacy runs took about 91 s per persona call (median of 39 saved runs) with a truncated prompt of about 2,050 tokens and about 400 output tokens.
- v2 has about 1,450 prompt tokens and about 350–450 output tokens. The pilot ran at about 63 s for 788 + 61 tokens, which includes warm-up.
- **Estimate: about 75–100 s per persona call, so 4 calls ≈ 5–7 minutes** sequentially. Batch embedding for scoring adds under 1 minute.

## 10. Tests
Run with `python -m pytest --confcutdir=tests <files>`: the 7 new v2 tests, plus the existing prompt, BI-calibration, canonical-SSR and improved-SSR test files. These last are unaffected because no existing code changed.

## 11. Unchanged
SSR code (`canonical_ssr.py`, `improved_ssr.py`), anchors CSV, codebook, encoder, k = 3, T = 0.75, all datasets, previous evaluation reports and results, and the legacy prompt module and dashboard.
