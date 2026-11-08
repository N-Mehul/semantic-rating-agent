# Quick Validation: Synthetic Responses and Improved SSR Ratings

**Date:** 2026-11-04 · **Product:** Echo Vision Smart Glasses (AI Smart Glasses / Wearable Technology, ₹14,999)
**Responses evaluated:** 24 (4 personas × 6 constructs: PE, EE, HM, SI, PP, BI)
**Reproduce:** `python evaluation/quick_ssr_validation/run_quick_validation.py`

> ⚠ **All agreement numbers below compare Improved SSR with LLM-JUDGE REFERENCE RATINGS. These are not human ground truth.**

## 1. Setup

| Step | What was used |
|---|---|
| Generation | Existing pipeline `ProductEvaluationService.run_evaluation()` (unmodified), `qwen2.5:7b-instruct` via Ollama, multi-construct prompt, saved concept from `echo_vision_smart_glasses_20261103_154257`. New run saved as `data/product_evaluation/echo_vision_smart_glasses_20261104_194521/` (the pipeline's normal behaviour; no existing run touched). |
| SSR | Pipeline output: Improved SSR (k=3, T=0.75), `nomic-embed-text` embeddings, approved anchors. Canonical SSR also reported. |
| Judge | `qwen2.5:7b-instruct`, temperature 0, separate call. It saw the response text, construct, the five approved anchors, the persona profile and the product description, and **never** the SSR rating. |
| Judge caveat | **The judge is the same model as the generator.** OpenAI was tried first but returned HTTP 429 (no credits remaining). Same-model judging can share biases with the generator, so the agreement is likely optimistic. |

## 2. Rating agreement (Improved SSR vs LLM judge)

| Metric | Improved SSR (rounded E[r]) | Improved SSR (argmax) | Canonical SSR (rounded) | Constant "always 4" |
|---|---:|---:|---:|---:|
| Exact agreement | **75.0%** | 62.5% | 79.2% | **66.7%** |
| Within ±1 | 100% | 87.5% | 100% | 100% |
| MAE (E[r]) | 0.389 | 0.389 | 0.330 | 0.333 |
| RMSE (E[r]) | 0.571 | 0.571 | 0.456 | 0.577 |
| Spearman ρ | **0.18 (p = 0.40)** | 0.18 | 0.23 (p = 0.28) | — |
| Quadratic-weighted κ | **0.40** | 0.04 | 0.48 | 0.00 |
| Mean bias (SSR − judge) | +0.18 | | +0.04 | |

**Confusion matrix** (rows = judge, columns = Improved SSR rounded; `confusion_matrix.csv` / `.png`):

| Judge \ SSR | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | **4** | 4 | 0 |
| 4 | 0 | 0 | 2 | **14** | 0 |
| 5 | 0 | 0 | 0 | 0 | 0 |

**Reading:** the judge used only ratings 3 and 4, and SSR E[r] ranged from 2.98 to 4.37. With such a narrow range, 75% exact agreement is only **8 points above a constant "always 4" guess**. The rank correlation is weak and not significant, so SSR does not reliably order the responses the way the judge does. QW-κ = 0.40 is *fair–moderate*. Canonical SSR agrees slightly better than Improved on all metrics, but that difference is a single response.

**Per construct (MAE vs judge):** PE 0.13, EE 0.17, SI 0.30, BI 0.35, HM 0.42, **PP 0.96**. PP is the main source of error: the judge rated all four PP responses 3, while SSR gave three of them ≈ 4.1–4.4.

## 3. Response quality

| Check | Result | Method |
|---|---|---|
| **Product relevance** | **0 / 24 mention any smart-glasses feature** (voice assistant, navigation, lenses, frame, hands-free). 14 / 24 explicitly call the product a "smartphone" or "primary phone"; the other 9 are generic device text (processor, display, battery). The judge marked 58% as relevant, but manual reading shows that is too lenient. | manual + regex + judge |
| Construct alignment | Judge: **70.8%**; embedding nearest anchor set: **66.7%** | judge / nomic |
| PP construct | 3 / 4 PP responses talk about *durability / build quality*, not data protection or privacy. Only P3 mentions security. | manual |
| HM | 3 / 4 HM responses are closest to the PE anchors (they describe performance, not enjoyment) | embedding |
| Persona consistency | Judge: 100% (lenient). Manually, P1 sounds premium ("high standards"), and P2–P4 say "entry-level", "pocket-friendly", "affordable". Persona tone is present, but **P1 also says "pocket-friendly price"**. | judge + manual |
| Naturalness | Judge mean 4.4 / 5; short (28.5 words), fluent, generic | judge |
| Repetition | BI opening "I am certain I will continue using this smartphone as my primary phone" is repeated **verbatim in all 4 personas**. Same-construct, cross-persona cosine similarity is 0.82. | manual / embedding |
| Direct numerical ratings | 0 / 24 | regex |

## 4. Persona differentiation (n = 6 per persona; descriptive only)

| Persona | Improved SSR mean E[r] | Judge mean |
|---|---:|---:|
| P1 premium | 4.20 | 3.83 |
| P2 mid-range heavy | 3.52 | 3.67 |
| P3 mid-range moderate | 3.85 | 3.67 |
| P4 budget | 3.80 | 3.50 |
| Kruskal–Wallis | H = 5.79, **p = 0.12** | H = 1.44, p = 0.70 |

Both scorers put P1 highest, but neither difference is significant. The texts are highly similar across personas (cosine 0.82 for the same construct).

## 5. Verdict

- **Rating quality: MODERATE at best, and weaker than the headline suggests.** SSR stays within ±1 of the judge for every response and matches exactly in 75%. However, the judge's ratings barely vary, a constant guess gets 67%, and the ranking agreement is weak (ρ = 0.18, n.s.). PP ratings are systematically too high.
- **Response quality: POOR on the most important criterion.** The responses are fluent and rating-free, but they describe **a smartphone instead of the smart glasses**. The PP and HM responses also drift away from their constructs. SSR ratings of these texts therefore measure attitudes toward a generic phone, not toward the product.

## 6. Main errors

1. **Wrong product.** The generator ignores the glasses concept (0/24 glasses-specific content).
2. **PP mismatch.** The responses discuss durability while the anchors measure data protection, and SSR overrates them (MAE 0.96).
3. **HM drifts toward PE** (performance language instead of enjoyment).
4. **Template repetition** across personas (identical BI opening sentence).
5. **Low rating variance.** Everything falls between 3 and 4.4, so there is little for SSR to discriminate.

## 7. Limitations
n = 24 with one product and one generation per persona, so all statistics have very wide uncertainty. The judge is the same model as the generator and is not human ground truth. No bootstrap or large-scale analysis was run, as you requested.

## Files
- `validation_results.csv`: all 24 responses with SSR E[r] (improved and canonical), PMF, judge rating and quality flags, embedding construct alignment
- `validation_metrics.json`: all metrics
- `confusion_matrix.csv`, `confusion_matrix.png` (300 dpi)
- `judge_raw.jsonl`: raw judge outputs
- `generation_run.json`: pointer to the generation run
