"""Six-step demonstration workflow behind the dashboard:

  1 persona + question -> synthetic TEXT (new LLM generation, or an existing saved response, clearly labelled)
  2 text -> sentence embedding            (existing local encoders; dimensions come from the real vector)
  3 embedding vs the five approved anchors (canonical cosine_similarities; anchors read from the stored approved CSV)
  4 similarities -> probabilities          (canonical similarities_to_pmf, unchanged configuration)
  5 probabilities -> expected rating       (canonical expected_rating)
  6 complete result with provenance and warnings

Nothing here computes a rating with an LLM: the model writes a sentence and the canonical SSR functions compute everything else.
Live generation happens ONLY when step 1 is called with mode "generate" and an explicit confirmation flag; it is never triggered by
loading a page. API keys are read from the environment by the SDKs and are never returned, logged or stored.
Completed runs are appended to data/dashboard_demo_runs/runs.jsonl (new folder; no other project file is written).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from canonical_ssr import (PAPER_EPSILON, PAPER_TEMPERATURE, PROBABILITY_TOLERANCE, cosine_similarities, expected_rating,  # noqa: E402
                           similarities_to_pmf)
from dp_constructs import ANCHOR_CODES, TASKS, anchor_info, load_codebook, task_question  # noqa: E402
from pilot_generate_api import SYSTEM, USER_TEMPLATE  # noqa: E402
from ssr_eval_lib import prompt_leak_findings, response_quality  # noqa: E402

ENCODERS = {"local_mpnet_baseline": "sentence-transformers/all-mpnet-base-v2",
            "local_minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "local_bge_base": "BAAI/bge-base-en-v1.5"}
DEFAULT_ENCODER = "local_bge_base"
RATING_REQUEST = re.compile(r"\b(rate|rating|ratings|score|scores|stars?|likert|out of \d|scale of|[1-5]\s*(to|-)\s*[1-5])\b", re.I)
MAX_QUESTION_CHARS = 400
CUSTOM_LABEL = "custom"
SSR_CONFIG = {"epsilon": PAPER_EPSILON, "temperature": PAPER_TEMPERATURE, "probability_tolerance": PROBABILITY_TOLERANCE}


class StepError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def clean(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def sha(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def validate_question(q) -> str:
    """A custom question must be non-empty text that asks for an explanation, never for a rating or number."""
    if not isinstance(q, str) or not q.strip():
        raise StepError("invalid_question", "Enter a question (it cannot be empty).")
    q = q.strip()
    if len(q) > MAX_QUESTION_CHARS:
        raise StepError("invalid_question", f"The question is too long ({len(q)} characters; limit {MAX_QUESTION_CHARS}).")
    if RATING_REQUEST.search(q):
        raise StepError("invalid_question", "The question must not ask for a rating, score, stars or a number: the LLM only writes sentences and SSR calculates the rating.")
    if prompt_leak_findings(q):
        raise StepError("invalid_question", "The question mentions evaluation labels (for example Q21) and cannot be used.")
    return q


def validate_probabilities(p) -> dict:
    p = np.asarray(p, dtype=float)
    ok_shape = p.shape == (5,)
    finite = bool(ok_shape and np.all(np.isfinite(p)))
    nonneg = bool(finite and np.all(p >= 0))
    total = float(p.sum()) if finite else None
    return {"five_values": ok_shape, "finite": finite, "non_negative": nonneg, "sum": total,
            "sum_within_tolerance": bool(finite and abs(total - 1.0) <= PROBABILITY_TOLERANCE), "tolerance": PROBABILITY_TOLERANCE,
            "valid": bool(ok_shape and finite and nonneg and abs(total - 1.0) <= PROBABILITY_TOLERANCE)}


def gemini_default_model() -> str:
    from gemini_utils import DEFAULT_MODEL          # single source of truth for the Gemini default
    return DEFAULT_MODEL


def gemini_model_from(env) -> str:
    from gemini_utils import get_model
    return get_model(env)


def backends_status(env=None, find_spec=None, ollama_probe=None) -> list[dict]:
    """Which generation backends could be used right now, in order of preference. Key values are never read into the result.

    LLM_PROVIDER=ollama|gemini|openai pins one provider (the others are reported as not selected: no silent fallback).
    Unset: a running local Ollama is preferred (no key needed), then Gemini, then OpenAI.
    The local Ollama server is probed only for the real environment (env=None) unless a probe is injected (tests)."""
    real_env = env is None
    env = os.environ if env is None else env
    import ollama_utils
    if ollama_probe is None and real_env:
        ollama_probe = ollama_utils.probe
    if ollama_probe is not None:
        pr = ollama_probe(env)
        oll = {"provider": "ollama", "label": "Ollama (local)", "sdk_installed": True, "key_configured": True, "model": ollama_utils.model(env),
               "available": bool(pr["running"] and pr["model_available"]), "unavailable_reason": pr["error"], "host": ollama_utils.host(env),
               "how_to_configure": f"Start Ollama ('ollama serve' or the Ollama app) and make sure '{ollama_utils.model(env)}' is installed; optional: set OLLAMA_MODEL / OLLAMA_HOST"}
    else:
        oll = {"provider": "ollama", "label": "Ollama (local)", "sdk_installed": True, "key_configured": True, "model": ollama_utils.model(env), "available": False,
               "unavailable_reason": "Ollama status not checked", "host": ollama_utils.host(env), "how_to_configure": "Start Ollama"}
    out = [oll] + _cloud_backends(env, find_spec)
    pinned = (env.get("LLM_PROVIDER") or "").strip().lower()
    if pinned in {b["provider"] for b in out}:
        for b in out:
            if b["provider"] != pinned:
                b["available"], b["unavailable_reason"] = False, f"not selected (LLM_PROVIDER={pinned})"
        out.sort(key=lambda b: b["provider"] != pinned)
    elif pinned:
        for b in out:
            b["available"], b["unavailable_reason"] = False, f"LLM_PROVIDER={pinned} is not a supported provider (use ollama, gemini or openai)"
    return out


def _cloud_backends(env, find_spec=None) -> list[dict]:
    if find_spec is None:
        import importlib.util
        find_spec = importlib.util.find_spec
    out = []
    for prov, label, key, mod, model_env, default in (("gemini", "Google Gemini", "GEMINI_API_KEY", "google.genai", "GEMINI_MODEL", gemini_default_model()),
                                                      ("openai", "OpenAI", "OPENAI_API_KEY", "openai", "OPENAI_GEN_MODEL", None)):
        try:
            sdk = find_spec(mod) is not None
        except (ImportError, ValueError):
            sdk = False
        has_key = bool(env.get(key))
        fmt_issue = key_format_issue(env.get(key))
        model = gemini_model_from(env) if prov == "gemini" else (env.get(model_env) or default)
        reasons = []
        if not has_key:
            reasons.append(f"{key} is not set in the server's environment")
        elif fmt_issue:
            reasons.append(f"{key} looks malformed: {fmt_issue}")
        if not sdk:
            reasons.append(f"the '{mod.split('.')[0]}' package is not installed")
        if has_key and sdk and not model:
            reasons.append(f"no model chosen (set {model_env})")
        out.append({"provider": prov, "label": label, "sdk_installed": sdk, "key_configured": has_key, "model": model,
                    "available": has_key and not fmt_issue and sdk and bool(model), "unavailable_reason": "; ".join(reasons) or None,
                    "how_to_configure": f"In the CMD window that starts the dashboard: set {key}=your-key" + (f" and set {model_env}=model-id" if default is None else "")})
    return out


def key_format_issue(value: str | None) -> str | None:
    """Detects common copy/paste mistakes WITHOUT revealing the key (Windows CMD keeps quotes typed in `set NAME="..."`)."""
    if not value:
        return None
    if value[0] in "\"'" or value[-1] in "\"'":
        return "the key value starts or ends with a quote character (in CMD write set NAME=value without quotes)"
    if value != value.strip() or any(c.isspace() for c in value):
        return "the key value contains spaces or line breaks"
    return None


def explain_provider_error(err: dict | None, provider: str) -> str:
    """Safe, useful text for the page: confirmed reason when the provider gave one, otherwise its sanitized status and message."""
    err = err or {}
    kind, det = err.get("kind", "unknown"), err.get("provider") or {}
    name = {"gemini": "Gemini", "openai": "OpenAI", "ollama": "Ollama"}.get(provider, provider)
    if provider == "ollama":
        if kind == "ollama_not_running":
            return f"Ollama is not running or not reachable: {det.get('message')} Start Ollama and try again. No response was generated."
        if kind == "model_not_found_or_no_access":
            return f"Ollama could not find the model ({det.get('message')}). Check OLLAMA_MODEL and 'ollama list'. No response was generated."
        if kind == "network_or_timeout":
            return f"{det.get('message')} No response was generated."
    var = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    if kind == "invalid_api_key" or det.get("reason") == "API_KEY_INVALID":
        return (f"{name} rejected the API key (HTTP {det.get('http_status')}, {det.get('reason') or 'invalid key'}). Re-create or re-copy the key, then in a new "
                f"Command Prompt run  set {var}=your-key  without quotes or spaces, and restart the dashboard in that window. No response was generated.")
    if kind == "location_not_supported":
        return f"{name} refused the request because the API is not available in your location ({det.get('message')}). No response was generated."
    parts = " ".join(str(x) for x in (f"HTTP {det['http_status']}" if det.get("http_status") else None, det.get("provider_status"), det.get("reason")) if x)
    msg = f": {det['message']}" if det.get("message") else ""
    if kind in ("rate_limit_or_quota", "insufficient_quota"):
        return f"{name} quota or rate limit reached ({kind}{', ' + parts if parts else ''}){msg}. No response was generated."
    if det:
        return f"{name} request failed ({kind}{', ' + parts if parts else ''}){msg}. No response was generated."
    return f"The LLM did not return a usable response ({kind}). No response was generated."


def default_generator(provider: str, model: str, prompt: dict):
    """Real provider call, reusing the project's tested generate_one functions (quality checks included). Returns (record|None, error|None)."""
    state = {"quota_hits": 0, "max_rate_limit_hits": 1}
    if provider == "ollama":
        import ollama_utils
        return ollama_utils.generate(model, prompt)
    if provider == "gemini":
        from google import genai
        from google.genai import types
        import pilot_generate_gemini as gg
        mot = int(prompt.get("max_output_tokens") or gg.SETTINGS["max_output_tokens"])     # optional per-request budget (e.g. the report)
        return gg.generate_one(genai.Client(), types, model, prompt, 2, 1, state, settings={**gg.SETTINGS, "timeout_s": 90, "max_output_tokens": mot})
    if provider == "openai":
        from openai import OpenAI
        import pilot_generate_openai as og
        client = OpenAI(max_retries=0, timeout=60)
        ok, msg = og.verify_model(client, model)
        if not ok:
            return None, {"kind": "model_check_failed", "detail": msg}
        mot = int(prompt.get("max_output_tokens") or og.SETTINGS["max_completion_tokens"])
        return og.generate_one(client, model, prompt, 2, state, settings={**og.SETTINGS, "max_completion_tokens": mot})
    return None, {"kind": "unknown_provider"}


def default_embedder_factory(cfg: str):
    from ssr_embedding_comparison import local_embedder
    return local_embedder(ENCODERS[cfg])


class Workflow:
    def __init__(self, root: Path | str = REPO_ROOT, embedder_factory=None, generator=None, env=None, log_dir=None, find_spec=None, ollama_probe=None):
        self.root = Path(root)
        self._factory = embedder_factory or default_embedder_factory
        self._generator = generator or default_generator
        self._env = env
        self._find_spec = find_spec
        self._ollama_probe = ollama_probe
        self._embedders: dict = {}
        self._sessions: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.log_dir = Path(log_dir) if log_dir else Path(os.environ.get("DP_DASHBOARD_RUN_LOG_DIR") or self.root / "data/dashboard_demo_runs")

    # ---------- data
    def _read_jsonl(self, rel):
        p = self.root / rel
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []

    def _read_json(self, rel):
        p = self.root / rel
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def _dataset(self):
        import pandas as pd
        p = self.root / "data/synthetic_dataset/synthetic_survey_dataset.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, keep_default_na=False, dtype={"response_text": str})
        for c in df.columns:
            if "__" in c or c == "encoder_spread_sd":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _cards(self):
        cards = self._read_json("data/persona_generator/persona_cards.json")
        if not isinstance(cards, list) or not cards:
            raise StepError("missing_personas", "Persona cards not found. Run: python scripts/run_pipeline.py all")
        return cards

    def _embedder(self, cfg):
        if cfg not in ENCODERS:
            raise StepError("invalid_encoder", f"Unknown encoder '{cfg}'.")
        with self._lock:
            if cfg not in self._embedders:
                try:
                    self._embedders[cfg] = self._factory(cfg)
                except Exception as e:
                    raise StepError("embedding_unavailable", f"Could not load the embedding model for {cfg} ({type(e).__name__}). "
                                    "The local sentence-transformer model must be installed/cached.")
            return self._embedders[cfg]

    # ---------- options (no generation, no model loading)
    def options(self) -> dict:
        try:
            cards = self._cards()
        except StepError as e:
            return {"available": False, "error": e.message}
        ext = self._read_jsonl("data/persona_generator/respondent_specs_extended.jsonl")
        df = self._dataset()
        cb = load_codebook()
        personas = []
        for c in cards:
            pid = c["persona_id"]
            n_saved = int(((df.persona_id == pid) & (df.status == "scored")).sum()) if df is not None else None
            personas.append({"persona_id": pid, "n_human_respondents": c.get("n_human_respondents"), "defining_rule": c.get("defining_rule"),
                             "saved_scored_responses": n_saved, "specs": [s["synthetic_id"] for s in ext if s["persona_id"] == pid]})
        tasks = [{"task_id": t, "label": f"{v['question_id']}: {task_question(t, cb)}", "construct": v["construct"], "question_text": task_question(t, cb),
                  "profile_version": v["profile_version"], "human_reference": v["human_reference"]} for t, v in TASKS.items()]
        saved = []
        if df is not None:
            for r in df[df.status == "scored"].itertuples():
                saved.append({"source_key": r.source_key, "synthetic_id": r.synthetic_id, "persona_id": r.persona_id, "task_id": r.task_id,
                              "construct": r.construct, "model_requested": r.model_requested, "preview": str(r.response_text)[:90]})
        return {"available": True, "title": "Digital Persona-Based Synthetic Survey Response Generation using SSR", "personas": personas, "tasks": tasks,
                "custom": {"id": CUSTOM_LABEL, "constructs": [{"code": k, "name": v} for k, v in ANCHOR_CODES.items()], "max_chars": MAX_QUESTION_CHARS},
                "encoders": [{"id": k, "model": v, "default": k == DEFAULT_ENCODER} for k, v in ENCODERS.items()],
                "backends": backends_status(self._env, self._find_spec, self._ollama_probe), "saved_responses": saved,
                "safety": "Nothing is generated when this page loads. New generation needs a configured backend, an explicit choice and a confirmation."}

    # ---------- sessions
    def _new_session(self, ctx: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._lock:
            self._sessions[rid] = ctx
            while len(self._sessions) > 40:
                self._sessions.popitem(last=False)
        return rid

    def _ctx(self, rid, need: str | None = None) -> dict:
        ctx = self._sessions.get(rid)
        if ctx is None:
            raise StepError("unknown_run", "This run no longer exists. Press Reset and start again.")
        if need and need not in ctx:
            raise StepError("step_order", f"Run the earlier step first (missing: {need}).")
        return ctx

    # ---------- step 1
    def step1(self, p: dict) -> dict:
        cards = {c["persona_id"]: c for c in self._cards()}
        pid = p.get("persona_id")
        if pid not in cards:
            raise StepError("invalid_persona", "Choose one of the four personas.")
        enc = p.get("encoder") or DEFAULT_ENCODER
        if enc not in ENCODERS:
            raise StepError("invalid_encoder", f"Unknown encoder '{enc}'.")
        mode = p.get("mode")
        if mode not in ("generate", "existing"):
            raise StepError("invalid_mode", "Choose 'generate a new response' or 'use an existing saved response'.")
        task = p.get("task_id")
        cb = load_codebook()
        if task == CUSTOM_LABEL:
            question = validate_question(p.get("custom_question"))
            construct = p.get("custom_construct")
            if construct not in ANCHOR_CODES:
                raise StepError("invalid_construct", "Choose which construct's approved anchors score a custom question.")
            task_meta = {"task_id": CUSTOM_LABEL, "construct": construct, "question_id": "custom", "profile_version": "v2", "human_reference": None}
        elif task in TASKS:
            question, task_meta = task_question(task, cb), {"task_id": task, **TASKS[task]}
            construct = task_meta["construct"]
        else:
            raise StepError("invalid_task", "Choose a supported question/construct or the custom option.")
        base = {"persona_id": pid, "persona_card": {k: cards[pid].get(k) for k in ("persona_id", "n_human_respondents", "pct_of_primary_cohort", "defining_rule")},
                "task": task_meta, "question": question, "construct": construct, "encoder": enc, "mode": mode,
                "compare_encoders": bool(p.get("compare_encoders"))}
        if mode == "existing":
            df = self._dataset()
            if df is None:
                raise StepError("missing_dataset", "No synthetic dataset found. Run: python scripts/run_pipeline.py all")
            m = df[(df.source_key == p.get("source_key")) & (df.synthetic_id == p.get("synthetic_id"))]
            if m.empty:
                raise StepError("no_such_response", "Select a saved response from the list.")
            r = m.iloc[0]
            if r.status != "scored":
                raise StepError("response_not_usable", f"That saved response did not pass the quality checks (status: {r.status}) and is not scored.")
            if r.persona_id != pid:
                raise StepError("persona_mismatch", "The selected saved response belongs to a different persona.")
            if task != CUSTOM_LABEL and r.task_id != task:
                raise StepError("task_mismatch", f"The selected saved response answers a different question ({r.task_id}), not {task}.")
            if task == CUSTOM_LABEL:
                raise StepError("task_mismatch", "Saved responses were not written for a custom question. Choose a supported question, or generate a new response.")
            text = str(r.response_text)
            q = response_quality(text, r.finish_reason if r.finish_reason != "" else None)
            prov = {"origin": "EXISTING saved response (not newly generated)", "source_key": r.source_key, "synthetic_id": r.synthetic_id,
                    "provider": r.provider, "model_requested": r.model_requested, "model_returned": r.model_returned, "finish_reason": r.finish_reason or None,
                    "generated_at_utc": r.generation_timestamp_utc or None, "prompt_sha": r.prompt_sha or None, "profile_version": r.profile_version}
            base["stored_expected_rating"] = clean(r.get(f"{enc}__E"))
            base["saved_ref"] = {"source_key": r.source_key, "synthetic_id": r.synthetic_id}
        else:
            if p.get("confirm_generation") is not True:
                raise StepError("confirmation_required", "Tick the confirmation box: new generation calls an external LLM provider and may cost money.")
            be = {b["provider"]: b for b in backends_status(self._env, self._find_spec, self._ollama_probe)}.get(p.get("backend"))
            if not be:
                raise StepError("invalid_backend", "Choose a generation backend.")
            if not be["available"]:
                raise StepError("no_backend", f"{be['label']} is not available: {be['unavailable_reason']}. You can still use an existing saved response.")
            ext = [s for s in self._read_jsonl("data/persona_generator/respondent_specs_extended.jsonl") if s["persona_id"] == pid]
            spec = next((s for s in ext if s["synthetic_id"] == p.get("spec_id")), None)
            if spec is None:
                raise StepError("invalid_spec", "Choose a synthetic respondent profile of this persona.")
            profile = spec["profile_text_v1"] if task_meta["profile_version"] == "v1" else spec["profile_text_v2"]
            prompt = {"synthetic_id": spec["synthetic_id"], "persona_id": pid, "system": SYSTEM, "user": USER_TEMPLATE.format(profile=profile, question=question)}
            bad = prompt_leak_findings(prompt["system"] + "\n" + prompt["user"])
            if bad:
                raise StepError("prompt_rejected", "The prompt failed the leak check and was not sent.")
            model = (p.get("model") or be["model"])
            try:
                rec, err = self._generator(be["provider"], model, prompt)
            except Exception as e:  # SDK/network errors: report the class only (never request details or keys)
                raise StepError("generation_failed", f"Generation failed ({type(e).__name__}).")
            if rec is None:
                kind = (err or {}).get("kind", "unknown")
                raise StepError("generation_failed", f"The provider did not return a usable response (kind: {kind}). Nothing was scored.")
            text = rec["response_text"]
            q = response_quality(text, rec.get("finish_reason"))
            prov = {"origin": "NEW generation (this run)", "provider": be["provider"], "model_requested": rec.get("model_requested", model),
                    "model_returned": rec.get("model_returned"), "finish_reason": rec.get("finish_reason"), "generated_at_utc": rec.get("timestamp_utc"),
                    "prompt_sha": sha(prompt["system"], prompt["user"]), "spec_id": spec["synthetic_id"], "profile_version": task_meta["profile_version"],
                    "response_id": rec.get("response_id")}
        base["text"], base["provenance"], base["quality"] = text, prov, q
        stop = None
        if not q["safe_for_ssr"]:
            reasons = [k for k in ("empty", "finish_reason_not_stop", "no_terminal_punctuation", "contains_rating") if q[k]]
            stop = "The response failed the quality checks (" + ", ".join(reasons) + ") and was not scored."
        rid = self._new_session(base)
        return {"run_id": rid, "persona": base["persona_card"], "question": question, "construct": construct, "task": task_meta, "mode": mode, "text": text,
                "provenance": prov, "quality": q, "stop": stop,
                "llm_note": "The LLM only wrote this sentence. It did not provide any rating." if mode == "generate" else "This text was written earlier; it is shown as an existing saved response."}

    # ---------- step 2
    def step2(self, rid: str) -> dict:
        ctx = self._ctx(rid, "text")
        if not ctx["quality"]["safe_for_ssr"]:
            raise StepError("not_scorable", "This response failed the quality checks and is not scored.")
        cfg = ctx["encoder"]
        embed = self._embedder(cfg)
        t0 = time.time()
        v = np.asarray(embed([ctx["text"]]), dtype=float)
        if v.ndim != 2 or v.shape[0] != 1 or not np.all(np.isfinite(v)):
            raise StepError("bad_embedding", "The embedding function returned an invalid vector.")
        vec = v[0]
        ctx["vec"], ctx["embed_model"] = vec, ENCODERS[cfg]
        return {"encoder": cfg, "model": ENCODERS[cfg], "dimensions": int(vec.shape[0]), "norm": float(np.linalg.norm(vec)),
                "first_values": [float(x) for x in vec[:8]], "seconds": round(time.time() - t0, 2),
                "note": f"The first 8 of {vec.shape[0]} actual values are shown; the full vector is used in the next step."}

    # ---------- step 3
    def step3(self, rid: str) -> dict:
        ctx = self._ctx(rid, "vec")
        info = anchor_info(ctx["construct"])
        embed = self._embedder(ctx["encoder"])
        A = np.asarray(embed(info["statements"]), dtype=float)
        sims = cosine_similarities(ctx["vec"], A)
        ctx["sims"] = sims
        ctx["anchor_info"] = {k: info[k] for k in ("construct_name", "anchor_sha", "status")}
        return {"construct": info["construct_name"], "anchor_sha": info["anchor_sha"], "anchor_status": info["status"],
                "anchors": [{"rating": i + 1, "statement": s, "cosine": float(sims[i])} for i, s in enumerate(info["statements"])],
                "note": "Anchors are read from the approved project file (one adapted anchor set for this construct; not the paper's six sets). Cosine similarity: 1 = same direction."}

    # ---------- step 4
    def step4(self, rid: str) -> dict:
        ctx = self._ctx(rid, "sims")
        pmf = similarities_to_pmf(ctx["sims"])           # canonical function, default (paper) configuration
        chk = validate_probabilities(pmf)
        if not chk["valid"]:
            raise StepError("invalid_probabilities", "The canonical SSR output failed probability validation; the run was stopped.")
        ctx["pmf"] = pmf
        return {"probabilities": [{"rating": i + 1, "probability": float(p)} for i, p in enumerate(pmf)], "validation": chk, "config": SSR_CONFIG,
                "zero_probability_ratings": [i + 1 for i, p in enumerate(pmf) if p == 0.0],
                "method": "Canonical SSR: subtract the smallest similarity from all five, then divide by their sum (epsilon 0, temperature 1). "
                          "The least similar anchor therefore always gets probability 0."}

    # ---------- step 5
    def step5(self, rid: str) -> dict:
        ctx = self._ctx(rid, "pmf")
        pmf = ctx["pmf"]
        er = expected_rating(pmf)                           # canonical function
        terms = [{"rating": i + 1, "probability": float(p), "product": float((i + 1) * p)} for i, p in enumerate(pmf)]
        total = float(sum(t["product"] for t in terms))
        sd = float(np.sqrt(np.dot(pmf, (np.arange(1, 6) - er) ** 2)))
        ent = float(-(pmf[pmf > 0] * np.log(pmf[pmf > 0])).sum())
        ctx.update(er=float(er), sd=sd, entropy=ent)
        return {"expected_rating": float(er), "terms": terms, "sum_of_terms": total, "matches_canonical_function": bool(abs(total - er) < 1e-9),
                "pmf_sd": sd, "entropy": ent, "most_probable_rating": int(np.argmax(pmf)) + 1,
                "formula": "Expected Rating = sum of (Likert rating x its probability)",
                "who_does_what": "The LLM wrote the sentence. The SSR pipeline calculated this rating."}

    # ---------- step 6
    def _score_all(self, text: str, construct: str) -> list[dict]:
        info = anchor_info(construct)
        rows = []
        for cfg in ENCODERS:
            try:
                embed = self._embedder(cfg)
                A = np.asarray(embed(info["statements"]), dtype=float)
                v = np.asarray(embed([text]), dtype=float)[0]
                s = cosine_similarities(v, A)
                pm = similarities_to_pmf(s)
                rows.append({"encoder": cfg, "model": ENCODERS[cfg], "expected_rating": float(expected_rating(pm)), "probabilities": [float(x) for x in pm],
                             "similarities": [float(x) for x in s]})
            except Exception as e:
                rows.append({"encoder": cfg, "model": ENCODERS[cfg], "error": getattr(e, "message", type(e).__name__)})
        return rows

    def step6(self, rid: str) -> dict:
        ctx = self._ctx(rid, "er")
        warnings = []
        q = ctx["quality"]
        if q["too_short_lt3_words"]:
            warnings.append({"level": "warning", "text": "The response is very short (fewer than 3 words); SSR has little text to work with."})
        if ctx["mode"] == "existing":
            warnings.append({"level": "info", "text": "This is an EXISTING saved response, not a new generation."})
            stored = ctx.get("stored_expected_rating")
            if stored is not None:
                warnings.append({"level": "info", "text": f"Recomputed expected rating {ctx['er']:.4f} vs stored dataset value {stored:.4f}: "
                                 + ("they match." if abs(ctx["er"] - stored) < 1e-6 else "they DIFFER; investigate before relying on either.")})
        else:
            warnings.append({"level": "info", "text": "This is a single new generation (one response). It is not a sample and is not saved into the research dataset."})
        warnings.append({"level": "info", "text": f"Uncertainty: PMF standard deviation {ctx['sd']:.3f}, entropy {ctx['entropy']:.3f} nats. A wide PMF means the text is similar to several anchors."})
        if ctx["task"]["human_reference"] is None:
            warnings.append({"level": "warning", "text": "No comparable human answers exist for this question in the collected survey, so this result cannot be validated against humans."})
        else:
            warnings.append({"level": "info", "text": "Human reference for this question is a proxy (Q26 text / Q21 rebuy intention), not a direct measure of this construct."})
        summ = self._read_json("data/validation/validation_summary.json") or {}
        ea = [x["spearman_E"] for x in summ.get("encoder_agreement", []) if x.get("spearman_E") is not None]
        if ea and max(ea) < 0.5:
            warnings.append({"level": "warning", "text": f"Saved validation shows the three encoders disagree on the ordering of responses (pairwise Spearman {min(ea):.2f} to {max(ea):.2f}); the expected rating of a single response depends on the encoder."})
        st = (summ.get("stability") or {}).get("status")
        if st and st != "computed":
            warnings.append({"level": "info", "text": "Repeated-generation stability is not supported by the current saved data."})
        df = self._dataset()
        if df is not None:
            n = int(((df.persona_id == ctx["persona_id"]) & (df.status == "scored")).sum())
            if n == 0:
                warnings.append({"level": "warning", "text": f"{ctx['persona_id']} has no saved generated responses in the research dataset; this demonstration cannot be compared with saved results for it."})
        warnings.append({"level": "info", "text": "Results are preliminary and do not establish robust persona replication or equivalence between synthetic and human answers."})
        comparison = self._score_all(ctx["text"], ctx["construct"]) if ctx.get("compare_encoders") else None
        spread = None
        if comparison:
            es = [r["expected_rating"] for r in comparison if "expected_rating" in r]
            spread = float(np.std(es)) if len(es) > 1 else None
            if len(es) > 1 and spread is not None and spread > 0.1:
                warnings.append({"level": "warning", "text": f"The three encoders give expected ratings spread over {max(es) - min(es):.2f} points for this response."})
        result = {"persona": ctx["persona_card"], "question": ctx["question"], "construct": ctx["construct"], "task": ctx["task"], "text": ctx["text"],
                  "embedding_model": ctx["embed_model"], "encoder": ctx["encoder"],
                  "similarities": [float(x) for x in ctx["sims"]], "probabilities": [float(x) for x in ctx["pmf"]], "expected_rating": ctx["er"],
                  "pmf_sd": ctx["sd"], "entropy": ctx["entropy"], "anchor": ctx["anchor_info"], "provenance": ctx["provenance"], "quality": q,
                  "warnings": warnings, "encoder_comparison": comparison, "encoder_spread_sd": spread, "ssr_config": SSR_CONFIG}
        self._log(rid, result)
        return result

    def _log(self, rid, result):
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_dir / "runs.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": rid, "logged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **result}, allow_nan=False, default=clean) + "\n")
        except Exception:
            pass  # logging must never break the demonstration

    # ---------- simple one-page demonstration ("Digital Persona — Synthetic Consumer")
    def _construct_for(self, question: str):
        """Approved anchors exist only for the project's survey constructs. A question gets a construct only if it IS one of the
        project's survey questions; otherwise no anchor set is assumed (no rating is invented)."""
        norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
        cb = load_codebook()
        for t, meta in TASKS.items():
            if norm(task_question(t, cb)) == norm(question):
                return t, meta
        return None, None

    def simple_options(self) -> dict:
        try:
            cards = self._cards()
        except StepError as e:
            return {"available": False, "error": e.message}
        top = lambda c, k: ", ".join(x.split(" (")[0] for x in (c.get(k) or [])[:2])
        personas = [{"persona_id": c["persona_id"], "label": c["persona_id"].split("_")[0] + " — " + " ".join(c["persona_id"].split("_")[1:]).replace("or ", "/ "),
                     "description": f"{c.get('defining_rule')}. Based on {c.get('n_human_respondents')} survey respondents; median age {c.get('median_age')}; "
                                    f"mostly {top(c, 'top_current_status').lower()}; common brands: {top(c, 'top_brands')}; daily use: {top(c, 'top_daily_usage')}."}
                    for c in cards]
        cb = load_codebook()
        be = backends_status(self._env, self._find_spec, self._ollama_probe)
        return {"available": True, "title": "Digital Persona — Synthetic Consumer", "personas": personas, "default_persona": personas[0]["persona_id"],
                "default_question": "Do you think AI virtual try-on technology will become widely used in the future?",
                "questions_with_approved_anchors": [{"question": task_question(t, cb), "construct": ANCHOR_CODES[m["construct"]]} for t, m in TASKS.items()],
                "generation_available": any(b["available"] for b in be),
                "generation_unavailable_reason": None if any(b["available"] for b in be) else "; ".join(b["unavailable_reason"] for b in be)}

    def simple_run(self, body: dict) -> dict:
        """Generate ONE new response with the configured LLM and score it with canonical SSR. Never returns a fabricated value."""
        try:
            cards = {c["persona_id"]: c for c in self._cards()}
            pid = body.get("persona_id")
            if pid not in cards:
                raise StepError("invalid_persona", "Choose one of the four personas.")
            question = validate_question(body.get("question"))
            if body.get("confirm_generation") is not True:
                raise StepError("confirmation_required", "Generation must be requested with the button.")
            be = next((b for b in backends_status(self._env, self._find_spec, self._ollama_probe) if b["available"]), None)
            if be is None:
                raise StepError("no_backend", "No LLM is available, so no response can be generated. Start Ollama with the model qwen2.5:0.5b, or set GEMINI_API_KEY "
                                "(or OPENAI_API_KEY and OPENAI_GEN_MODEL) in the window that starts the dashboard, then restart it.")
            task, meta = self._construct_for(question)
            spec = next((s for s in self._read_jsonl("data/persona_generator/respondent_specs_extended.jsonl") if s["persona_id"] == pid), None)
            if spec is None:
                raise StepError("missing_profiles", "No respondent profile found for this persona. Run: python scripts/run_pipeline.py all")
            profile = spec["profile_text_v1"] if meta and meta["profile_version"] == "v1" else spec["profile_text_v2"]
            prompt = {"synthetic_id": spec["synthetic_id"], "persona_id": pid, "system": SYSTEM, "user": USER_TEMPLATE.format(profile=profile, question=question)}
            if prompt_leak_findings(prompt["system"] + "\n" + prompt["user"]):
                raise StepError("prompt_rejected", "The question cannot be used (it mentions evaluation labels).")
            try:
                rec, err = self._generator(be["provider"], be["model"], prompt)
            except Exception as e:
                raise StepError("generation_failed", f"The LLM request failed ({type(e).__name__}). No response was generated.")
            if rec is None:
                raise StepError((err or {}).get("kind") if (err or {}).get("kind") in ("invalid_api_key", "location_not_supported") else "generation_failed",
                                explain_provider_error(err, be["provider"]))
            text = rec["response_text"]
            q = response_quality(text, rec.get("finish_reason"))
            out = {"ok": True, "persona_id": pid, "question": question, "response_text": text,
                   "generation": {"provider": be["provider"], "model": rec.get("model_returned") or be["model"], "generated_at_utc": rec.get("timestamp_utc"),
                                  "respondent_profile": spec["synthetic_id"]}}
            if not q["safe_for_ssr"]:
                bad = [k for k in ("empty", "finish_reason_not_stop", "no_terminal_punctuation", "contains_rating") if q[k]]
                return {**out, "rating_available": False, "rating_unavailable_reason": "The response failed quality checks (" + ", ".join(bad) + "), so it was not rated."}
            if meta is None:
                cb = load_codebook()
                return {**out, "rating_available": False,
                        "rating_unavailable_reason": "No approved anchor statements exist for this question's construct (the project's approved anchors cover "
                                                     "its smartphone survey constructs only), so no rating is calculated. Use one of the project's survey questions to see a rating.",
                        "questions_with_approved_anchors": [task_question(t, cb) for t in TASKS]}
            info = anchor_info(meta["construct"])
            embed = self._embedder(DEFAULT_ENCODER)
            A = np.asarray(embed(info["statements"]), dtype=float)
            v = np.asarray(embed([text]), dtype=float)[0]
            sims = cosine_similarities(v, A)
            pmf = similarities_to_pmf(sims)
            if not validate_probabilities(pmf)["valid"]:
                raise StepError("invalid_probabilities", "The SSR probabilities failed validation; no rating is shown.")
            return {**out, "rating_available": True, "construct": info["construct_name"], "embedding_model": ENCODERS[DEFAULT_ENCODER],
                    "anchors": [{"rating": i + 1, "statement": s, "cosine": float(sims[i])} for i, s in enumerate(info["statements"])],
                    "probabilities": [float(p) for p in pmf], "expected_rating": float(expected_rating(pmf))}
        except StepError as e:
            return {"ok": False, "error_code": e.code, "error": e.message}

    # ---------- dispatch used by the server
    def run_step(self, n: int, body: dict) -> dict:
        try:
            if n == 1:
                return {"ok": True, "step": 1, **self.step1(body)}
            if n in (2, 3, 4, 5, 6):
                return {"ok": True, "step": n, **getattr(self, f"step{n}")(body.get("run_id"))}
            return {"ok": False, "step": n, "error_code": "unknown_step", "error": "Unknown step."}
        except StepError as e:
            return {"ok": False, "step": n, "error_code": e.code, "error": e.message}
        except Exception as e:
            return {"ok": False, "step": n, "error_code": "internal", "error": f"Unexpected error ({type(e).__name__}) in step {n}."}
