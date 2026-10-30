"""New mobile product concept research with digital personas (the dashboard's main page).

Flow per click:  for each selected persona  ->  POST /api/product/respond  (one NEW LLM response, text only)
                 then                       ->  POST /api/product/report   (an LLM analyst report built ONLY from those stored responses)

Reused, unchanged: the four persona cards and respondent profiles (built from the human survey), the provider integration and its
quality checks, the approved anchor sets and the canonical SSR functions (via Workflow). Nothing here computes a rating itself.

SSR scoring is applied only when the research question has an approved anchor set. In this project the approved anchors cover the
smartphone survey constructs and are matched only to the project's own survey questions; a new-product question (purchase intention,
price acceptability, recommendation, ...) therefore gets "SSR rating unavailable" instead of a rating from an unrelated anchor set.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np

try:                                              # imported as dashboard.product_research (tests) or product_research (server)
    from . import workflow as _wfm
except ImportError:
    import workflow as _wfm

StepError = _wfm.StepError
RATING_REQUEST = _wfm.RATING_REQUEST
UNRATED = "Likert rating unavailable: no compatible approved anchor set."
DEFAULT_QUESTION = "Would you consider buying this product? Why or why not?"
SUGGESTED_QUESTIONS = ["Would you consider buying this smartphone?", "Does the proposed price seem reasonable?", "How useful are the AI camera features?",
                       "Would this product meet your needs?", "How likely are you to recommend this product?", "What might prevent you from purchasing this product?"]
LIMITS = {"concept_min": 20, "concept_max": 3000, "field_max": 300, "question_max": 400, "price_max": 1_000_000}
RESPONDENT_SYSTEM = ("You role-play one consumer from a market-research panel reacting to a NEW mobile product concept. Answer in your own words, "
                     "3-6 sentences, informal and honest, from the profile's point of view: mention what appeals to you, what worries you, how the price "
                     "feels for someone like you, and whether you would consider buying it. Positive, negative, mixed or neutral answers are all fine; "
                     "do not default to praise. Use words only: no numbers for scores, no ratings, no stars, no percentages. Do not mention being an AI.")
ANALYST_SYSTEM = ("You are a careful market-research analyst. You receive SYNTHETIC consumer responses written by an LLM for persona profiles that are "
                  "based on a small student-heavy survey sample. Summarise ONLY what the responses and profiles support. Never invent statistics, "
                  "percentages, market share, sales, willingness-to-pay figures, conversion rates or a launch-success score, and never give numeric ratings. "
                  "Mark anything not supported by the responses as an open question. Reply with JSON only, no prose outside the JSON.")
REPORT_KEYS = {"executive_summary": dict, "personas": list, "customer_segments": str, "price_and_features": str, "opportunities": list, "risks": list,
               "missing_information": list, "assumptions_to_test": list, "recommendations": list, "launch_readiness": dict}
REPORT_SCHEMA = """{
 "executive_summary": {"overview": str, "opportunities": [str], "concerns": [str], "caveat": str},
 "personas": [{"persona_id": str, "reaction": str, "motivations": [str], "objections": [str], "valued_features": [str],
               "price_perception": str, "would_consider_if": [str], "would_reject_because": [str]}],
 "customer_segments": str,
 "price_and_features": str,
 "opportunities": [str], "risks": [str], "missing_information": [str], "assumptions_to_test": [str],
 "recommendations": [str],
 "launch_readiness": {"positive_signals": [str], "negative_signals": [str], "unresolved_questions": [str], "required_validation": [str]}
}"""
SUMMARY_SYSTEM = ("You rewrite one consumer's answer into a short, clear summary in everyday English. Use ONLY what the answer says: do not add "
                  "preferences, features or opinions that are not in it, and do not repeat yourself. If something is not mentioned, write \"Not mentioned\". "
                  "Never give numbers, scores or ratings. Reply with JSON only.")
SUMMARY_SCHEMA = """{"overall_opinion": str (one short sentence), "likes": [str] (short items), "main_concern": str, "important_features": [str],
 "suggested_improvement": str, "purchase_intention": str (one short phrase describing how interested the consumer sounds)}"""
SUMMARY_KEYS = ("overall_opinion", "likes", "main_concern", "important_features", "suggested_improvement", "purchase_intention")
INTENTION_NOTE = "This is a synthetic response, not a confirmed purchase decision."
STOP = set("the a an and or but of to for in on with this that it its is are was be my me i you your they their very really just more most much "
           "would could should will can not no yes some any about from into than then so as at by have has had do does like".split())
SUMMARY_RATING = re.compile(r"\b[1-5]\s*/\s*5\b|\bout of (5|five)\b|\b(rating|score)\s*(of|:)?\s*\d", re.I)


def content_words(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (t or "").lower()) if len(w) > 2 and w not in STOP}


def grounded(item: str, source: str) -> bool:
    """True when the summary item shares at least one content word (or its 5-letter stem) with the raw response/concept."""
    src = content_words(source)
    stems = {w[:5] for w in src}
    return any(w in src or w[:5] in stems for w in content_words(item))


NUMBER = re.compile(r"\d[\d,.]*")


def numbers_grounded(text: str, source: str) -> bool:
    """Every number in the summary text must also appear in the raw response or the concept (e.g. no invented '1600 mAh')."""
    src = {n.replace(",", "").rstrip(".") for n in NUMBER.findall(source)}
    return all(n.replace(",", "").rstrip(".") in src for n in NUMBER.findall(text))


def clean_summary(raw: dict, source: str) -> dict:
    """Validated summary. List items that cannot be traced to the raw response or concept are moved to 'not_in_response', never shown as fact."""
    if not isinstance(raw, dict) or any(k not in raw for k in SUMMARY_KEYS):
        raise ValueError("summary is missing sections")
    if SUMMARY_RATING.search(json.dumps(raw)):                      # checked on the raw output, before any cleaning
        raise ValueError("summary contains a numeric rating")
    out, flagged = {}, []
    for k in SUMMARY_KEYS:
        v = raw[k]
        if k in ("likes", "important_features"):
            items = [str(x).strip()[:120] for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
            items = list(dict.fromkeys(items, None))                         # drop exact repeats
            items = [it for i, it in enumerate(items) if it.lower() not in {x.lower() for x in items[:i]}][:8]
            keep = []
            for it in items:
                if it.lower() != "not mentioned" and not (grounded(it, source) and numbers_grounded(it, source)):
                    flagged.append(it)
                else:
                    keep.append(it)
            out[k] = keep
        else:
            txt = (str(v).strip() if v is not None else "")[:300] or "Not mentioned"
            concrete = k in ("main_concern", "suggested_improvement")   # opinion/intention fields paraphrase, so only their numbers are checked
            if txt.lower() != "not mentioned" and not (numbers_grounded(txt, source) and (grounded(txt, source) or not concrete)):
                flagged.append(txt)
                txt = "Not clearly stated in the response."
            out[k] = txt
    if SUMMARY_RATING.search(json.dumps(out)):
        raise ValueError("summary contains a numeric rating")
    out["purchase_intention_note"] = INTENTION_NOTE
    out["not_in_response"] = flagged
    return out


def explain_ssr(sc: dict) -> str:
    """Plain-language reading of the ACTUAL similarities and probabilities (it creates no new numbers)."""
    a = sorted(sc["anchors"], key=lambda x: -x["cosine"])
    top, second = a[0], a[1]
    spread = a[0]["cosine"] - a[-1]["cosine"]
    e = sc["expected_rating"]
    side = "the agreeing (higher) end" if e > 3.5 else "the disagreeing (lower) end" if e < 2.5 else "the middle"
    txt = (f"The response is most similar to anchor {top['rating']} ({top['label']}: \"{top['statement']}\") and next to anchor {second['rating']} "
           f"({second['label']}). Its expected rating of {e:.2f} out of 5 sits toward {side} of the scale. ")
    if spread < 0.05:
        txt += f"All five anchors are almost equally similar (similarities differ by only {spread:.3f}), so this rating is uncertain. "
    txt += "In the canonical SSR method the least similar anchor always receives probability 0."
    return txt


NUMERIC_CLAIM = re.compile(r"\b\d{1,3}(\.\d+)?\s*%|\bmarket share\b|\bconversion rate\b|\bunits sold\b|\bsuccess (score|probability)\b", re.I)


def _clean_text(v, limit):
    if v is None:
        return ""
    if not isinstance(v, str):
        raise StepError("invalid_input", "Text fields must be text.")
    v = v.strip()
    if len(v) > limit:
        raise StepError("invalid_input", f"A field is too long ({len(v)} characters; limit {limit}).")
    return v


def validate_concept(body: dict) -> dict:
    concept = _clean_text(body.get("concept"), LIMITS["concept_max"])
    if len(concept) < LIMITS["concept_min"]:
        raise StepError("invalid_input", f"Describe the product concept in at least {LIMITS['concept_min']} characters.")
    fields = {k: _clean_text(body.get(k), LIMITS["field_max"]) for k in ("target_group", "features", "market")}
    price = body.get("price_inr")
    if price in (None, ""):
        price = None
    else:
        try:
            price = float(price)
        except (TypeError, ValueError):
            raise StepError("invalid_input", "The proposed price must be a number in INR.")
        if not 0 < price <= LIMITS["price_max"]:
            raise StepError("invalid_input", "The proposed price must be between 1 and 10,00,000 INR.")
    question = _clean_text(body.get("question"), LIMITS["question_max"]) or DEFAULT_QUESTION
    if RATING_REQUEST.search(question):
        raise StepError("invalid_question", "The research question must not ask for a rating, score or number: the LLM only writes text; SSR calculates ratings where approved anchors exist.")
    if _wfm.prompt_leak_findings(question) or _wfm.prompt_leak_findings(concept):
        raise StepError("invalid_input", "The text mentions evaluation labels (for example Q21 or a /5 score) and cannot be used.")
    return {"concept": concept, "price_inr": price, **fields, "question": question, "question_was_default": not _clean_text(body.get("question"), LIMITS["question_max"])}


def concept_block(c: dict) -> str:
    lines = [f"Product concept: {c['concept']}"]
    if c["price_inr"] is not None:
        lines.append(f"Proposed price: {c['price_inr']:,.0f} INR")
    for k, label in (("features", "Main features"), ("target_group", "Target customers"), ("market", "Intended market")):
        if c[k]:
            lines.append(f"{label}: {c[k]}")
    return "\n".join(lines)


def parse_json_object(text: str):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        raise ValueError("no JSON object")
    return json.loads(t[a:b + 1])


def validate_report(rep: dict, persona_ids: list[str]) -> list[str]:
    """Returns problems; an empty list means the structure is usable. Missing personas are reported, never filled in."""
    probs = [f"missing or wrong type: {k}" for k, typ in REPORT_KEYS.items() if not isinstance(rep.get(k), typ)]
    got = [p.get("persona_id") for p in rep.get("personas", []) if isinstance(p, dict)]
    probs += [f"no summary for {pid}" for pid in persona_ids if pid not in got]
    return probs


class ProductResearch:
    def __init__(self, workflow):
        self.w = workflow
        self._results: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def options(self) -> dict:
        base = self.w.simple_options()
        if not base.get("available"):
            return base
        return {"available": True, "title": "Digital Persona — Synthetic Consumer Research",
                "subtitle": "Explore potential consumer reactions and evaluate new mobile product concepts before launch.",
                "personas": base["personas"], "suggested_questions": SUGGESTED_QUESTIONS, "default_question": DEFAULT_QUESTION, "limits": LIMITS,
                "generation_available": base["generation_available"], "generation_unavailable_reason": base["generation_unavailable_reason"],
                "rating_scope": "SSR ratings are calculated only for questions with an approved anchor set (the project's own smartphone survey questions). "
                                "New-product questions are answered and analysed but not rated."}

    def _backend(self):
        be = next((b for b in _wfm.backends_status(self.w._env, self.w._find_spec, self.w._ollama_probe) if b["available"]), None)
        if be is None:
            raise StepError("no_backend", "No LLM is available, so no response can be generated. Start Ollama with the model qwen2.5:0.5b, or set GEMINI_API_KEY "
                            "(or OPENAI_API_KEY and OPENAI_GEN_MODEL) in the window that starts the dashboard, then restart it.")
        return be

    def _call(self, be, prompt):
        try:
            rec, err = self.w._generator(be["provider"], be["model"], prompt)
        except Exception as e:
            raise StepError("generation_failed", f"The LLM request failed ({type(e).__name__}). No response was generated.")
        if rec is None:
            kind = (err or {}).get("kind")
            raise StepError(kind if kind in ("invalid_api_key", "location_not_supported", "model_not_found_or_no_access", "rate_limit_or_quota") else "generation_failed",
                            _wfm.explain_provider_error(err, be["provider"]))
        return rec

    # ---------- one persona
    def respond(self, body: dict) -> dict:
        try:
            if body.get("confirm_generation") is not True:
                raise StepError("confirmation_required", "Generation must be requested with the button.")
            cards = {c["persona_id"]: c for c in self.w._cards()}
            pid = body.get("persona_id")
            if pid not in cards:
                raise StepError("invalid_persona", "Choose one of the four personas.")
            c = validate_concept(body)
            be = self._backend()
            spec = next((s for s in self.w._read_jsonl("data/persona_generator/respondent_specs_extended.jsonl") if s["persona_id"] == pid), None)
            if spec is None:
                raise StepError("missing_profiles", "No respondent profile found for this persona. Run: python scripts/run_pipeline.py all")
            persona_desc = next(p["description"] for p in self.w.simple_options()["personas"] if p["persona_id"] == pid)
            user = (f"Respondent profile: {spec['profile_text_v2']}.\nGroup this respondent belongs to (from a real survey): {persona_desc}\n\n"
                    f"{concept_block(c)}\n\nResearch question: {c['question']}")
            prompt = {"synthetic_id": spec["synthetic_id"], "persona_id": pid, "system": RESPONDENT_SYSTEM, "user": user}
            rec = self._call(be, prompt)
            text = rec["response_text"]
            q = _wfm.response_quality(text, rec.get("finish_reason"))
            task, meta = self.w._construct_for(c["question"])
            summary, summary_error = None, None
            if text.strip():
                try:
                    sp = {"synthetic_id": spec["synthetic_id"], "persona_id": pid, "system": SUMMARY_SYSTEM, "json": True, "max_output_tokens": 600,
                          "user": f"Consumer's answer:\n{text}\n\nReturn JSON with exactly this structure:\n{SUMMARY_SCHEMA}"}
                    srec = self._call(be, sp)
                    summary = clean_summary(parse_json_object(srec["response_text"]), text + "\n" + concept_block(c))
                except (StepError, ValueError, json.JSONDecodeError) as e:
                    summary_error = ("Simple summary unavailable (" + (e.message if isinstance(e, StepError) else "the LLM did not return a usable summary")
                                     + "). The original response is shown.")
            result = {"persona_id": pid, "question": c["question"], "response_text": text, "origin": "NEW LLM-generated synthetic response",
                      "generation": {"provider": be["provider"], "model": rec.get("model_returned") or be["model"], "generated_at_utc": rec.get("timestamp_utc"),
                                     "respondent_profile": spec["synthetic_id"], "finish_reason": rec.get("finish_reason")},
                      "quality": {k: bool(v) for k, v in q.items()}, "summary": summary, "summary_error": summary_error,
                      "raw_response_note": "Original LLM text, unchanged. This is the text SSR scores; the simple summary is for reading only."}
            if not q["safe_for_ssr"]:
                bad = [k for k in ("empty", "finish_reason_not_stop", "no_terminal_punctuation", "contains_rating") if q[k]]
                result["ssr"] = {"available": False, "reason": "The response failed quality checks (" + ", ".join(bad) + "), so it was not rated."}
            elif meta is None:
                result["ssr"] = {"available": False, "reason": UNRATED}
            else:
                result["ssr"] = {"available": True, **self._score(text, meta["construct"]), "question_scored": c["question"], "task_id": task}
                result["ssr"]["explanation"] = explain_ssr(result["ssr"])
            rid = uuid.uuid4().hex[:12]
            with self._lock:
                self._results[rid] = {**result, "concept": c}
                while len(self._results) > 200:
                    self._results.popitem(last=False)
            return {"ok": True, "result_id": rid, **result}
        except StepError as e:
            return {"ok": False, "persona_id": body.get("persona_id"), "error_code": e.code, "error": e.message}

    def _score(self, text: str, construct: str) -> dict:
        info = _wfm.anchor_info(construct)
        from survey_ssr_bridge import get_construct_anchors
        labels = list(get_construct_anchors(construct).labels)                 # Likert labels from the approved anchor file
        embed = self.w._embedder(_wfm.DEFAULT_ENCODER)
        A = np.asarray(embed(info["statements"]), dtype=float)
        v = np.asarray(embed([text]), dtype=float)[0]
        sims = _wfm.cosine_similarities(v, A)
        pmf = _wfm.similarities_to_pmf(sims)
        chk = _wfm.validate_probabilities(pmf)
        if not chk["valid"]:
            raise StepError("invalid_probabilities", "The SSR probabilities failed validation; no rating is shown.")
        return {"construct": info["construct_name"], "anchor_set": info["status"], "anchor_sha": info["anchor_sha"], "embedding_model": _wfm.ENCODERS[_wfm.DEFAULT_ENCODER],
                "anchors": [{"rating": i + 1, "label": labels[i], "statement": s, "cosine": float(sims[i])} for i, s in enumerate(info["statements"])],
                "probabilities": [float(p) for p in pmf], "probability_sum": chk["sum"], "expected_rating": float(_wfm.expected_rating(pmf))}

    # ---------- report from the stored responses
    def report(self, body: dict) -> dict:
        try:
            ids = body.get("result_ids")
            if not isinstance(ids, list) or not 1 <= len(ids) <= 4:
                raise StepError("invalid_input", "Provide the ids of 1 to 4 generated persona responses.")
            with self._lock:
                results = [self._results.get(i) for i in ids]
            if any(r is None for r in results):
                raise StepError("unknown_result", "Some responses are no longer available. Run the analysis again.")
            if len({json.dumps(r["concept"], sort_keys=True) for r in results}) != 1:
                raise StepError("invalid_input", "All responses in a report must come from the same product concept.")
            be = self._backend()
            descs = {p["persona_id"]: p["description"] for p in self.w.simple_options()["personas"]}
            blocks = "\n\n".join(f"[{r['persona_id']}] profile: {descs.get(r['persona_id'], '')}\nSynthetic response: {r['response_text']}" for r in results)
            user = (f"{concept_block(results[0]['concept'])}\nResearch question asked: {results[0]['question']}\n\n{blocks}\n\n"
                    f"Return JSON with exactly this structure (lists of short strings):\n{REPORT_SCHEMA}")
            prompt = {"synthetic_id": "report", "persona_id": "analyst", "system": ANALYST_SYSTEM, "user": user, "max_output_tokens": 8192, "json": True, "timeout_s": 300}
            rec = self._call(be, prompt)
            if str(rec.get("finish_reason") or "").upper() not in ("STOP", "FINISH_REASON_STOP", ""):
                raise StepError("report_incomplete", f"The analyst report was cut off (finish reason {rec.get('finish_reason')}). No report is shown; try again.")
            try:
                rep = parse_json_object(rec["response_text"])
            except (ValueError, json.JSONDecodeError):
                raise StepError("report_unparseable", "The LLM did not return a valid report structure. No report is shown; try again.")
            problems = validate_report(rep, [r["persona_id"] for r in results])
            if any(p.startswith("missing or wrong type") for p in problems):
                raise StepError("report_unparseable", "The LLM report was missing required sections (" + "; ".join(problems[:4]) + "). No report is shown.")
            flagged = sorted({m.group(0) for m in NUMERIC_CLAIM.finditer(json.dumps(rep))})
            comparison = []
            summ = {p.get("persona_id"): p for p in rep.get("personas", []) if isinstance(p, dict)}
            for rid, r in zip(ids, results):
                s = summ.get(r["persona_id"], {})
                comparison.append({"persona_id": r["persona_id"], "result_id": rid, "main_reaction": s.get("reaction"), "motivations": s.get("motivations"),
                                   "concerns": s.get("objections"), "price_perception": s.get("price_perception"),
                                   "ssr_expected_rating": r["ssr"].get("expected_rating") if r["ssr"]["available"] else None,
                                   "ssr_status": "rated (" + r["ssr"]["construct"] + ")" if r["ssr"]["available"] else r["ssr"]["reason"]})
            rated = {r["ssr"].get("construct") for r in results if r["ssr"]["available"]}
            return {"ok": True, "report": rep, "comparison": comparison, "report_warnings": problems + ([f"The report contains numeric or market claims that are NOT supported by synthetic data: {', '.join(flagged)}"] if flagged else []),
                    "ratings_comparable": len(rated) == 1 and all(r["ssr"]["available"] for r in results),
                    "labels": {"responses": "LLM-generated synthetic responses (not real customers)", "report": "LLM-generated interpretation of the synthetic responses (hypotheses, not market evidence)",
                               "ratings": "SSR-computed from the response text with approved anchors (only where available)", "personas": "Persona profiles derived from the existing human survey sample"},
                    "generation": {"provider": be["provider"], "model": rec.get("model_returned") or be["model"], "generated_at_utc": rec.get("timestamp_utc") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
        except StepError as e:
            return {"ok": False, "error_code": e.code, "error": e.message}
