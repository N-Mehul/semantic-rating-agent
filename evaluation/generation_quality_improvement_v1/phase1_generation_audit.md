# Phase 1: Generation Quality Audit (read-only)

**Date:** 2026-11-04 · **Scope:** the existing mobile-phone product pipeline (Qwen 2.5 7B via Ollama). Nothing was generated, scored or modified.
**Evidence:** code reading, prompt construction for the existing product *Nova AI Smartphone* (₹24,999; `scripts/run_product_evaluation.py`) without any LLM call, statistics over the 31 saved mobile-phone runs (631 construct texts), `git diff` of the uncommitted working-tree changes, and `evaluation/quick_ssr_validation/quick_validation_report.md`.

## 1. Files inspected

| Area | File |
|---|---|
| Multi-construct prompt, persona context, category detection, sanitizer, parser | `scripts/product_evaluation_prompts.py` (uncommitted: +771 lines) |
| Generation loop, parameters, embedding, SSR call | `dashboard/evaluation_service.py` (uncommitted: +486 lines) |
| Ollama client and settings | `scripts/ollama_utils.py` (uncommitted: `num_ctx` pinned to 2048) |
| Existing mobile product and personas | `scripts/run_product_evaluation.py` (`NOVA_AI_SMARTPHONE`, `PERSONA_IDS`) |
| Persona definitions | `data/personas/persona_profiles.json`, `persona_assignments.csv`; `data/persona_generator/persona_cards*.json`, `respondent_specs*.jsonl` |
| Construct definitions | `SUPPORTED_CONSTRUCTS` in `product_evaluation_prompts.py`; `scripts/dp_constructs.py`; `survey_ssr_bridge.py` |
| Survey questions and codebook | `docs/survey_instrument/survey_codebook.csv`, `survey_instrument.md` |
| Approved anchors | `docs/survey_instrument/smartphone_construct_anchor_statements.csv` |
| Saved outputs | `data/product_evaluation/*/persona_responses.json`, `product_concept.json` |
| Previous validation | `evaluation/quick_ssr_validation/quick_validation_report.md` |

## 2. How generation currently works

- **One call per persona** produces all six constructs plus an overall summary as a single JSON object. That is 4 calls per product.
- **Settings:** `qwen2.5:7b-instruct`, temperature 0.7, `num_predict` 800, `num_ctx` **2048**, JSON mode, timeout 360 s. There is **no retry**: one failed call aborts the whole run.
- **Persona input** is a fixed segment block (title, price-tier benchmark, top 3 Q23–Q25 themes, tech comfort). The varied respondent specs in `data/persona_generator/respondent_specs*.jsonl` are **not used**.
- **Post-processing:** `parse_multiconstruct_response` parses the JSON, then `sanitize_invented_facts` rewrites phrases with about 40 regex substitutions. If parsing fails, the **whole raw text is copied into every construct**.
- **SSR input:** every construct text is embedded with `nomic-embed-text` (one HTTP call per text) and compared with anchors that are all phrased "*my current smartphone…*".

## 3. Root causes

### 3.1 Generic phone text instead of the specific product
1. **Prompt likely exceeds the context window (high-impact).** System + user prompt is 9,742–9,846 characters, about **2,400–2,700 tokens** for each persona. Ollama is called with `num_ctx=2048`, plus up to 800 output tokens. When a prompt overflows, Ollama silently drops tokens from the start. The persona block and product block are at the start of the user message. *Verification is cheap: one call's `prompt_eval_count` (Phase 2).*
2. **Anchor framing pulls the text toward "my phone".** All 30 anchors say "my current smartphone". The prompt says the product is a proposed concept the persona does not own. Two prompt lines go further:
   - BI must begin "*I am certain I will continue using this smartphone as my primary phone*" (continuation of ownership).
   - The persona block says "*Current smartphone: Flagship-tier smartphone*".

   The model resolves this contradiction by writing about a generic phone it already owns.
3. **Over-restrictive grounding rules.** About 15 "NEVER" rules forbid inventing features, needs, routines or ownership. The safest output is generic hardware talk ("processor, display, battery"). In the saved phone runs, only **7.6%** of texts name the product.
4. **Category detection uses the whole description** (a wrong-category risk, not a mobile bug). The smartphone regex is tested against the full description. Any product whose description mentions "smartphones" becomes a smartphone, which caused the glasses failure. For a real phone it is harmless.

### 3.2 PP discusses durability instead of privacy
**Direct prompt contradiction, introduced in the uncommitted edits:**
- System prompt: "*In PP, evaluate physical build quality, hardware durability, and reliability. Do NOT fabricate data privacy … concerns unless explicitly stated*".
- Construct instruction (generic and phone branch): "*PP: Physical build durability, hardware reliability, and longevity … data security ONLY if the product explicitly lists security features*".
- Meanwhile `SUPPORTED_CONSTRUCTS["PP"]`, the codebook (PP01–PP04) and the anchors all define PP as **protection of personal information**.

Older saved phone runs, generated before this edit, discuss privacy in **90%** of PP texts. The regression comes from the prompt edit, not from the model.

### 3.3 HM drifts into performance
- The HM instruction is "*how pleasant … to use for daily tasks and media viewing based ONLY on listed features*". "Daily tasks" overlaps with PE, and "only listed features" forces hardware talk.
- PE, HM and EE share the same feature list and are written in one JSON object, so wording bleeds between neighbouring keys.
- In saved runs, 37% of HM texts contain performance terms, and the quick validation found 3 of 4 HM texts closest to the PE anchors.
- The "NEVER mention gaming" rule also removes a main source of enjoyment for phones.

### 3.4 Personas produce near-identical responses
1. **Same template for every persona.** The persona difference is one "Evaluation Focus" line and one price sentence; the construct instructions are identical.
2. **Forced identical BI opening** for all four personas.
3. **Contradictory price lines.** For prices under ₹25,000, P1 *and* P4 are both told the price is "*completely affordable, pocket-friendly*". Nova costs ₹24,999, so the premium and budget personas receive the same message.
4. **No respondent-level variation.** Each persona gets one fixed profile, the seeded specs are unused, and there is one generation per persona. Quick validation measured cross-persona cosine similarity of 0.82 for the same construct.
5. **Truncation (3.1)** may remove most of the persona block, which would erase the differences that remain.
6. **Sanitizer homogenises text.** Regex replacements push all personas toward the same phrases ("for the price", "on the higher side", "practical features for daily use").

### 3.5 Limited attitude and Likert variability
- **Positivity is built in.** Every BI guide starts "*If the product delivers … express confident/decisive positive adoption*"; negative intent is only an afterthought. P1 and P2 have no negative branch at all.
- **Negativity is filtered out.** Price complaints are forbidden in five of six constructs, and the sanitizer deletes price-concern clauses ("the price is too high …").
- **Short answers.** Each value is limited to 2–3 sentences, which gives little room for mixed or critical attitudes. Only 20% of saved phone texts contain any hedge or negation.
- **Positive product input.** The concept text is marketing copy with no drawbacks, so the model has nothing to criticise.
- **Anchor/frame mismatch compresses SSR.** "Proposed product" text is compared against "my current smartphone" anchors, so similarities are close together and E[r] stays between 3 and 4.4.

### 3.6 Slow generation
- About **91 s per persona call** (median over 39 saved runs), run sequentially, so roughly 6 minutes per product before embedding.
- The prompt is about 2.5k tokens, and prompt evaluation on CPU is a large part of that time. About 40% of it is duplicated (rules appear in both system and user messages).
- `num_predict` 800 is high for roughly 350 tokens of needed output.
- Embedding is one HTTP request per text (24 texts plus 30 anchors), re-embedded on every run.
- There is no retry: a single timeout loses the whole run and forces full regeneration.

## 4. Prompt weaknesses (summary)

| # | Weakness | Location |
|---|---|---|
| W1 | PP redefined as durability (contradicts codebook and anchors) | system prompt; construct instruction 5 |
| W2 | Forced identical, positive BI opening | `pricing_guide` P1–P4; instruction 6 |
| W3 | "Pocket-friendly" for both P1 and P4 below ₹25k | `p1_price_eval`, `p4_price_eval` |
| W4 | "Current smartphone: Flagship-tier" conflicts with "you do NOT own …" | `build_persona_context` vs system prompt |
| W5 | HM is defined through "daily tasks" and "listed features"; gaming banned | instruction 3 |
| W6 | Prompt longer than `num_ctx` 2048 | prompt size vs `ollama_utils.SETTINGS` |
| W7 | About 15 negative "NEVER" rules; rules duplicated in system and user | both messages |
| W8 | Positive-only adoption guidance; negativity filtered | `pricing_guide`, sanitizer |
| W9 | All constructs in one JSON call, so wording bleeds between them | `build_persona_multiconstruct_prompt` |
| W10 | Parse failure copies raw text into all six constructs | `parse_multiconstruct_response` |
| W11 | Respondent specs unused; one sample per persona | evaluation loop |

## 5. Recommended improvements (Phase 2 candidates; SSR, k, T, encoder and anchors untouched)

1. **Restore PP to privacy and data protection**: lock screen, app permissions, data sharing, account safety. Keep "do not invent features not listed" for protection features only.
2. **Remove the forced BI sentence.** Ask for a stance *chosen by the persona* (adopt / undecided / would not), each with a concrete reason.
3. **Fix the price lines** so each persona reacts to ₹24,999 according to its own tier: P1 "good value / unremarkable", P4 "a stretch".
4. **Align the frame with the anchors**: "imagine you have been using the Nova AI Smartphone as your main phone for a month…". This keeps the anchors' first-person "my phone" framing while naming the product. Remove "Current smartphone: Flagship-tier".
5. **Redefine HM** as fun, enjoyment and entertainment (camera, media, games and social apps, if listed), and explicitly exclude speed and productivity.
6. **Shrink the prompt to under about 1,400 tokens.** State each rule once, convert "NEVER" lists into 4–5 short positive instructions, and raise `num_ctx` to 4096 for safety.
7. **Add persona-specific attitude drivers** from survey data (e.g. P4: price sensitivity and essentials; P1: brand, camera and premium finish), plus one required drawback per construct.
8. **Draw 2–3 respondent specs per persona** from the existing `respondent_specs*.jsonl` to add within-persona variation.
9. **Make the sanitizer lighter**: keep only rating-leak and brand checks, and stop deleting price-concern clauses.
10. **Validate per-construct output** with a cheap check (keyword presence for PP/HM, and no identical opening), and regenerate only the failing construct.

## 6. Risks to generation time

| Change | Effect on time |
|---|---|
| Shorter prompt (rec. 6) | **Faster**: prompt evaluation drops by roughly 40–50% |
| `num_predict` 800 → about 450 | Faster (caps runaway output) |
| `num_ctx` 2048 → 4096 | Slightly slower per token and more RAM; only needed if the prompt stays long |
| Per-construct calls instead of one multi-construct call | **6× more calls**: about 3–4× slower even with shorter prompts. Use only for failing constructs. |
| 2–3 respondent specs per persona | Linear 2–3× increase; keep it optional |
| Retry-on-failure | Saves time overall (avoids full reruns) |
| Batch embeddings (`embed_texts`) and anchor caching | Removes about 50 HTTP round-trips per run |

## 7. Improving quality without excessive regeneration

1. **Prompt-only pilot first:** 1 persona × 1 call. Check `prompt_eval_count` (confirms or rules out truncation), PP and HM content, and BI stance. Then run all 4 personas once: 4 calls in about 4–6 minutes.
2. **Regenerate only PP, HM and BI.** PE, EE and SI were acceptable in the quick validation (judge MAE 0.13–0.30). Use a short three-construct prompt.
3. **Gate before scoring:** reject outputs that fail the construct keyword checks, repeat a previous opening, or fail to parse. Retry only that persona, at most twice.
4. **Reuse cached anchor embeddings** and batch the response embeddings.
5. **Run validation on the same 24-cell design** as the quick validation for a like-for-like comparison. This time, keep the generator and judge distinct if credits allow.

---
*No existing file was modified. This report is the only file created (`evaluation/generation_quality_improvement_v1/phase1_generation_audit.md`).*
