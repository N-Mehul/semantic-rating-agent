"""Batches of synthetic SURVEY responses for the "Synthetic Survey Responses" section of the dashboard.

Source of truth is the project's human survey work, reused through dashboard/workflow.py:
  * the four digital personas (data/persona_generator/persona_cards.json, built from the cleaned human survey),
  * their synthetic respondent profiles (respondent_specs_extended.jsonl: attributes sampled from each persona's real answers),
  * the supported survey questions and constructs (dp_constructs: collected-survey Q26 and codebook FT_BI01 / FT_PE01, or a custom
    question scored with an approved construct's anchors),
  * the approved anchor statements and the unmodified canonical SSR functions.

Survey context given to the LLM is AGGREGATED and anonymised: the persona's most frequent themes in the experience questions
(Q23-Q25, computed in Step 4). No individual respondent's answer, identifier or Q21/Q26 value is ever placed in a prompt (Q21/Q26 are
evaluation material and stay out of generation, as in the earlier pilots).

Generation happens only after an explicit, confirmed request with a configured backend. Each response uses a different respondent
profile of the persona, so a batch is diverse by construction; the prompt also asks the model not to default to praise (prompt variant
"survey_batch_v1", recorded in provenance). The LLM writes text only; every number shown is computed by canonical SSR.
Batches are saved to data/dashboard_survey_batches/<batch_id>.json; earlier batches are never deleted or overwritten.
"""
from __future__ import annotations

import csv
import io
import json
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np

try:                                              # imported as dashboard.survey_batches (tests) or as survey_batches (server)
    from . import workflow as _wfm
except ImportError:
    import workflow as _wfm
from dp_constructs import ANCHOR_CODES, TASKS, load_codebook, task_question  # noqa: E402  (scripts/ is on sys.path via workflow)

CUSTOM_LABEL, ENCODERS, SYSTEM, USER_TEMPLATE = _wfm.CUSTOM_LABEL, _wfm.ENCODERS, _wfm.SYSTEM, _wfm.USER_TEMPLATE
StepError, Workflow, anchor_info, backends_status, clean = _wfm.StepError, _wfm.Workflow, _wfm.anchor_info, _wfm.backends_status, _wfm.clean
cosine_similarities, expected_rating, similarities_to_pmf = _wfm.cosine_similarities, _wfm.expected_rating, _wfm.similarities_to_pmf
prompt_leak_findings, response_quality, sha = _wfm.prompt_leak_findings, _wfm.response_quality, _wfm.sha
validate_probabilities, validate_question = _wfm.validate_probabilities, _wfm.validate_question

MAX_PER_BATCH = 10            # one per respondent profile of the persona; keeps provider calls bounded
PROMPT_VARIANT = "survey_batch_v1"
THEME_WORDS = {"positive_general": "general positive remarks", "negative_issue": "problems or complaints", "performance_speed": "speed and performance",
               "camera": "the camera", "battery": "battery life", "display": "the display", "design_build": "design and build",
               "value_price": "price and value", "brand_ecosystem": "the brand or its ecosystem", "storage": "storage"}
DIVERSITY_NOTE = ("Answer honestly from this respondent's own point of view. Real respondents like this give positive, negative, mixed or "
                  "neutral answers; do not default to praise, and avoid stock phrases.")
STOP_KINDS = {"invalid_api_key", "location_not_supported", "authentication_or_permission", "model_not_found_or_no_access", "bad_request", "rate_limit_or_quota", "insufficient_quota",
              "authentication", "permission_or_model_access", "model_not_found", "model_check_failed", "unknown_provider"}
EXPORT_FIELDS = ["batch_id", "response_id", "origin", "persona_id", "spec_id", "task_id", "question_id", "question", "construct", "anchor_construct",
                 "anchor_sha", "response_text", "status", "quality_warnings", "provider", "model_requested", "model_returned", "finish_reason",
                 "generated_at_utc", "prompt_variant", "prompt_sha", "encoder", "embedding_model", "s1", "s2", "s3", "s4", "s5",
                 "p1", "p2", "p3", "p4", "p5", "probability_sum", "expected_rating"]


def persona_context(card: dict) -> str:
    """Aggregated, anonymised survey evidence for the persona (no individual answers)."""
    themes = card.get("experience_themes_Q23_Q25") or []
    parts = [f"{THEME_WORDS.get(t['theme'], t['theme'].replace('_', ' '))} ({t['pct_of_usable_answers']}% of answers)" for t in themes]
    n = card.get("n_human_respondents")
    if not parts:
        return ""
    return (f"Survey context (aggregated and anonymised from {n} real respondents in this persona's group): when describing their phones "
            f"they most often mentioned {', '.join(parts)}")


def jaccard(a: str, b: str) -> float:
    A, B = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
    return len(A & B) / len(A | B) if A | B else 0.0


class BatchManager:
    def __init__(self, workflow: Workflow, store_dir: Path | None = None):
        self.w = workflow
        self.store = Path(store_dir) if store_dir else workflow.root / "data/dashboard_survey_batches"
        self.jobs: dict = {}
        self._lock = threading.Lock()

    # ---------- validation of a request (no provider call)
    def _resolve(self, p: dict) -> dict:
        cards = {c["persona_id"]: c for c in self.w._cards()}
        pid = p.get("persona_id")
        if pid not in cards:
            raise StepError("invalid_persona", "Choose one of the four personas.")
        task = p.get("task_id")
        cb = load_codebook()
        if task == CUSTOM_LABEL:
            question = validate_question(p.get("custom_question"))
            construct = p.get("custom_construct")
            if construct not in ANCHOR_CODES:
                raise StepError("invalid_construct", "Choose which approved construct's anchors score the custom question.")
            meta = {"task_id": CUSTOM_LABEL, "construct": construct, "question_id": "custom", "profile_version": "v2", "human_reference": None}
        elif task in TASKS:
            question, meta = task_question(task, cb), {"task_id": task, **TASKS[task]}
        else:
            raise StepError("invalid_task", "Choose a supported survey question/construct or a custom question.")
        try:
            n = int(p.get("count", 0))
        except (TypeError, ValueError):
            n = 0
        if not 1 <= n <= MAX_PER_BATCH:
            raise StepError("invalid_count", f"Choose between 1 and {MAX_PER_BATCH} responses per batch.")
        enc = p.get("encoder") or "local_bge_base"
        if enc not in ENCODERS:
            raise StepError("invalid_encoder", f"Unknown encoder '{enc}'.")
        if p.get("confirm_generation") is not True:
            raise StepError("confirmation_required", "Tick the confirmation box: generation calls an external LLM provider and may cost money.")
        be = {b["provider"]: b for b in backends_status(self.w._env, self.w._find_spec, self.w._ollama_probe)}.get(p.get("backend"))
        if not be:
            raise StepError("invalid_backend", "Choose a generation backend.")
        if not be["available"]:
            raise StepError("no_backend", f"{be['label']} is not available: {be['unavailable_reason']}. Previously saved responses remain available below.")
        specs = [s for s in self.w._read_jsonl("data/persona_generator/respondent_specs_extended.jsonl") if s["persona_id"] == pid]
        if len(specs) < n:
            raise StepError("not_enough_profiles", f"Only {len(specs)} respondent profiles exist for this persona.")
        return {"persona_id": pid, "card": cards[pid], "question": question, "task": meta, "construct": meta["construct"], "count": n,
                "encoder": enc, "backend": be, "model": p.get("model") or be["model"], "specs": specs[:n]}

    def build_prompt(self, r: dict, spec: dict) -> dict:
        profile = spec["profile_text_v1"] if r["task"]["profile_version"] == "v1" else spec["profile_text_v2"]
        ctx = persona_context(r["card"])
        user = USER_TEMPLATE.format(profile=profile + (". " + ctx if ctx else ""), question=r["question"]) + "\n\n" + DIVERSITY_NOTE
        p = {"synthetic_id": spec["synthetic_id"], "persona_id": r["persona_id"], "system": SYSTEM, "user": user,
             "task_id": r["task"]["task_id"], "construct": r["construct"], "question_id": r["task"]["question_id"]}
        if prompt_leak_findings(p["system"] + "\n" + p["user"]):
            raise StepError("prompt_rejected", "A prompt failed the leak check and was not sent.")
        return p

    def _score(self, text: str, construct: str, encoder: str, info: dict) -> dict:
        embed = self.w._embedder(encoder)
        A = np.asarray(embed(info["statements"]), dtype=float)
        v = np.asarray(embed([text]), dtype=float)[0]
        s = cosine_similarities(v, A)
        pmf = similarities_to_pmf(s)
        chk = validate_probabilities(pmf)
        if not chk["valid"]:
            raise StepError("invalid_probabilities", "Canonical SSR output failed probability validation.")
        er = float(expected_rating(pmf))
        return {"similarities": [float(x) for x in s], "probabilities": [float(x) for x in pmf], "probability_sum": chk["sum"], "expected_rating": er,
                "expected_rating_check": abs(er - float(np.dot(np.arange(1, 6), pmf))) < 1e-9, "embedding_dimensions": int(v.shape[0])}

    # ---------- the batch itself
    def run_batch(self, p: dict, progress=None) -> dict:
        r = self._resolve(p)
        info = anchor_info(r["construct"])
        bid = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
        items, stopped = [], None
        for i, spec in enumerate(r["specs"], 1):
            rid = f"{bid}-{i:02d}"
            base = {"response_id": rid, "origin": "NEW generation (this batch)", "persona_id": r["persona_id"], "spec_id": spec["synthetic_id"],
                    "task_id": r["task"]["task_id"], "question_id": r["task"]["question_id"], "question": r["question"], "construct": r["construct"],
                    "anchor_construct": info["construct_name"], "anchor_sha": info["anchor_sha"], "provider": r["backend"]["provider"],
                    "model_requested": r["model"], "prompt_variant": PROMPT_VARIANT, "encoder": r["encoder"], "embedding_model": ENCODERS[r["encoder"]]}
            if stopped:
                items.append({**base, "status": "not_attempted", "warnings": [f"Not attempted: the batch stopped earlier ({stopped})."]})
                continue
            prompt = self.build_prompt(r, spec)
            base["prompt_sha"] = sha(prompt["system"], prompt["user"])
            try:
                rec, err = self.w._generator(r["backend"]["provider"], r["model"], prompt)
            except Exception as e:
                rec, err = None, {"kind": f"exception:{type(e).__name__}"}
            if rec is None:
                kind = (err or {}).get("kind", "unknown")
                items.append({**base, "status": "generation_failed", "warnings": [f"Generation failed (kind: {kind}); nothing was scored."]})
                if kind in STOP_KINDS or (err or {}).get("stop_run"):
                    stopped = kind
            else:
                text = rec["response_text"]
                q = response_quality(text, rec.get("finish_reason"))
                warns = []
                if q["too_short_lt3_words"]:
                    warns.append("Very short response.")
                item = {**base, "response_text": text, "model_returned": rec.get("model_returned"), "finish_reason": rec.get("finish_reason"),
                        "generated_at_utc": rec.get("timestamp_utc"), "quality": q}
                if not q["safe_for_ssr"]:
                    bad = [k for k in ("empty", "finish_reason_not_stop", "no_terminal_punctuation", "contains_rating") if q[k]]
                    item.update(status="excluded_quality", warnings=warns + ["Failed quality checks (" + ", ".join(bad) + "); not scored."])
                else:
                    try:
                        item.update(status="scored", ssr=self._score(text, r["construct"], r["encoder"], info), warnings=warns)
                    except StepError as e:
                        item.update(status="generated_not_scored", warnings=warns + [e.message])
                items.append(item)
            if progress:
                progress(i, len(r["specs"]))
        texts = [(x["response_id"], x["response_text"]) for x in items if x.get("response_text")]
        for a in range(len(texts)):
            for b in range(a + 1, len(texts)):
                if jaccard(texts[a][1], texts[b][1]) >= 0.8:
                    for x in items:
                        if x["response_id"] == texts[b][0]:
                            x.setdefault("warnings", []).append(f"Near-duplicate of {texts[a][0]} (word overlap at least 0.8).")
        scored = [x for x in items if x["status"] == "scored"]
        es = [x["ssr"]["expected_rating"] for x in scored]
        batch = {"batch_id": bid, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "persona_id": r["persona_id"],
                 "persona_rule": r["card"].get("defining_rule"), "task": r["task"], "question": r["question"], "construct": r["construct"],
                 "anchors": {"construct_name": info["construct_name"], "statements": info["statements"], "anchor_sha": info["anchor_sha"], "status": info["status"]},
                 "provider": r["backend"]["provider"], "model_requested": r["model"], "encoder": r["encoder"], "embedding_model": ENCODERS[r["encoder"]],
                 "prompt_variant": PROMPT_VARIANT, "survey_context": persona_context(r["card"]), "requested": r["count"],
                 "counts": {k: sum(1 for x in items if x["status"] == k) for k in ("scored", "excluded_quality", "generation_failed", "generated_not_scored", "not_attempted")},
                 "stopped_early_reason": stopped,
                 "expected_rating_summary": ({"n": len(es), "min": min(es), "mean": float(np.mean(es)), "max": max(es)} if es else None),
                 "items": items}
        if scored or any(x.get("response_text") for x in items):
            self._save(batch)
        batch["saved"] = bool(scored or any(x.get("response_text") for x in items))
        return batch

    def _save(self, batch: dict):
        self.store.mkdir(parents=True, exist_ok=True)
        path = self.store / f"{batch['batch_id']}.json"
        if path.exists():                       # batch ids are unique; never overwrite an earlier batch
            raise StepError("batch_exists", "A batch with this id already exists.")
        path.write_text(json.dumps(batch, indent=2, allow_nan=False, default=clean), encoding="utf-8")

    # ---------- background jobs
    def start(self, p: dict) -> dict:
        try:
            self._resolve(p)                      # validate before starting anything
        except StepError as e:
            return {"ok": False, "error_code": e.code, "error": e.message}
        jid = uuid.uuid4().hex[:10]
        job = {"job_id": jid, "state": "running", "done": 0, "total": int(p.get("count")), "batch_id": None, "error": None}
        with self._lock:
            self.jobs[jid] = job

        def run():
            try:
                b = self.run_batch(p, progress=lambda i, n: job.update(done=i, total=n))
                job.update(state="finished", batch_id=b["batch_id"] if b.get("saved") else None, counts=b["counts"],
                           stopped_early_reason=b["stopped_early_reason"], batch=None if b.get("saved") else b)
            except StepError as e:
                job.update(state="failed", error=e.message, error_code=e.code)
            except Exception as e:
                job.update(state="failed", error=f"Unexpected error ({type(e).__name__}).", error_code="internal")
        threading.Thread(target=run, daemon=True).start()
        return {"ok": True, "job_id": jid}

    def job(self, jid: str) -> dict | None:
        j = self.jobs.get(jid)
        return None if j is None else dict(j)

    # ---------- reading results
    def list_batches(self) -> list[dict]:
        out = []
        if self.store.exists():
            for f in sorted(self.store.glob("*.json"), reverse=True):
                try:
                    b = json.loads(f.read_text(encoding="utf-8"))
                    out.append({k: b.get(k) for k in ("batch_id", "created_utc", "persona_id", "question", "construct", "provider", "model_requested",
                                                      "encoder", "requested", "counts", "stopped_early_reason")})
                except Exception:
                    out.append({"batch_id": f.stem, "error": "unreadable batch file"})
        return out

    def get_batch(self, bid: str) -> dict | None:
        if not re.fullmatch(r"[0-9A-Za-z-]{6,40}", bid or ""):
            return None
        f = self.store / f"{bid}.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def saved_pilot(self, encoder: str = "local_bge_base") -> dict:
        """Earlier saved synthetic responses (Gemini v2 pilot) with their STORED canonical-SSR scores, explicitly labelled as saved."""
        if encoder not in ENCODERS:
            encoder = "local_bge_base"
        df = self.w._dataset()
        if df is None:
            return {"available": False, "error": "No synthetic dataset found. Run: python scripts/run_pipeline.py all", "items": []}
        cb = load_codebook()
        items = []
        for r in df.itertuples():
            row = df.loc[r.Index]
            scored = row["status"] == "scored"
            t = row["task_id"]
            item = {"response_id": f"{row['source_key']}:{row['synthetic_id']}", "origin": f"SAVED earlier response ({row['source_key']}), not newly generated",
                    "persona_id": row["persona_id"], "spec_id": row["synthetic_id"], "task_id": t, "question_id": row["question_id"],
                    "question": task_question(t, cb) if t in TASKS else None, "construct": row["construct"], "anchor_construct": row["anchor_construct"],
                    "anchor_sha": row["anchor_sha"], "response_text": row["response_text"], "provider": row["provider"], "model_requested": row["model_requested"],
                    "model_returned": row["model_returned"], "finish_reason": row["finish_reason"] or None, "generated_at_utc": row["generation_timestamp_utc"] or None,
                    "prompt_sha": row["prompt_sha"] or None, "prompt_variant": "pilot (Q26 prompt, no survey-context block)", "encoder": encoder,
                    "embedding_model": ENCODERS[encoder], "status": "scored" if scored else row["status"], "warnings": [] if scored else [str(row["exclusion_reasons"])]}
            try:
                item["anchor_statements"] = anchor_info(row["construct"])["statements"]
            except Exception:
                item["anchor_statements"] = None
            if scored:
                p = [clean(row[f"{encoder}__p{k}"]) for k in range(1, 6)]
                s = [clean(row[f"{encoder}__s{k}"]) for k in range(1, 6)]
                item["ssr"] = {"similarities": s, "probabilities": p, "probability_sum": sum(p), "expected_rating": clean(row[f"{encoder}__E"]),
                               "stored_scores": True}
            items.append(item)
        bi = anchor_info("BI")
        return {"available": True, "origin": "saved", "encoder": encoder, "items": items,
                "anchors": {"construct_name": bi["construct_name"], "statements": bi["statements"], "anchor_sha": bi["anchor_sha"], "status": bi["status"]},
                "note": "These responses were generated earlier (Gemini v2 pilot, personas P1 and P2 only) and are shown with their stored scores."}


def export_rows(items: list[dict], batch_id: str) -> list[dict]:
    rows = []
    for x in items:
        s = x.get("ssr") or {}
        sims, pr = s.get("similarities") or [None] * 5, s.get("probabilities") or [None] * 5
        rows.append({"batch_id": batch_id, **{k: x.get(k) for k in EXPORT_FIELDS if k in x}, "quality_warnings": " | ".join(x.get("warnings") or []),
                     **{f"s{i + 1}": sims[i] for i in range(5)}, **{f"p{i + 1}": pr[i] for i in range(5)},
                     "probability_sum": s.get("probability_sum"), "expected_rating": s.get("expected_rating")})
    return [{k: r.get(k) for k in EXPORT_FIELDS} for r in rows]


def to_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=EXPORT_FIELDS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    return buf.getvalue()
