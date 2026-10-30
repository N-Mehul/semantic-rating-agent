"""Methodology stage 3: persona-conditioned generation vs an UNCONDITIONED control group, as separate labelled conditions.

  persona  : the project's existing prompt (SYSTEM + respondent profile from data/persona_generator + survey question)
  control  : the same SYSTEM instruction and the same question, with NO persona/profile information

Each (provider, condition, task) writes to its own folder data/pilot_<provider>_<condition>_<task>/ (never mixed with other runs):
  prompts.jsonl, responses.jsonl, failures.jsonl, generation_config.json, condition_scores.csv (canonical SSR, selected encoder)
Responses are checkpointed; reruns skip finished ids. The LLM writes text only; ratings come from canonical SSR afterwards.

Run (local Ollama, no key):  python scripts/pilot_generate_condition.py --condition control --provider ollama --per-persona 1
                             python scripts/pilot_generate_condition.py --condition persona --provider ollama --per-persona 1
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts", ROOT / "dashboard"):
    sys.path.insert(0, str(p))
from dp_constructs import TASKS, anchor_info, load_codebook, task_question  # noqa: E402
from gemini_utils import append_jsonl, done_ids  # noqa: E402
from pilot_generate_api import SYSTEM, USER_TEMPLATE  # noqa: E402
from ssr_eval_lib import prompt_leak_findings, response_quality  # noqa: E402

PERSONAS = ["P1_premium_tier_owner", "P2_midrange_heavy_user", "P3_midrange_moderate_user", "P4_budget_or_undisclosed_price"]
CONTROL_TEMPLATE = "Question: {question}"


def build_prompts(condition: str, task: str, per_persona: int, specs: list[dict]) -> list[dict]:
    """Persona prompts use the existing respondent profiles; control prompts carry no persona information at all.
    The control group gets the same number of prompts as the persona condition, so the two are directly comparable."""
    meta = TASKS[task]
    question = task_question(task, load_codebook())
    chosen = [s for pid in PERSONAS for s in [x for x in specs if x["persona_id"] == pid][:per_persona]]
    out = []
    for i, s in enumerate(chosen, 1):
        if condition == "persona":
            profile = s["profile_text_v1"] if meta["profile_version"] == "v1" else s["profile_text_v2"]
            p = {"synthetic_id": s["synthetic_id"], "persona_id": s["persona_id"], "user": USER_TEMPLATE.format(profile=profile, question=question)}
        elif condition == "control":
            p = {"synthetic_id": f"CTRL_{i:02d}", "persona_id": None, "user": CONTROL_TEMPLATE.format(question=question)}
        else:
            raise ValueError("condition must be 'persona' or 'control'")
        p.update(system=SYSTEM, condition=condition, task_id=task, construct=meta["construct"], question_id=meta["question_id"])
        if prompt_leak_findings(p["system"] + "\n" + p["user"]):
            raise RuntimeError(f"prompt {p['synthetic_id']} failed the leak check")
        out.append(p)
    return out


def score(rows: list[dict], construct: str, embed) -> pd.DataFrame:
    """Canonical SSR on each quality-passing response (cosine -> similarities_to_pmf -> expected_rating)."""
    from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf
    info = anchor_info(construct)
    A = np.asarray(embed(info["statements"]), dtype=float)
    out = []
    for r in rows:
        q = response_quality(r["response_text"], r.get("finish_reason"))
        base = {"synthetic_id": r["synthetic_id"], "persona_id": r.get("persona_id"), "condition": r["condition"], "model": r.get("model_returned"),
                "response_text": r["response_text"], "anchor_construct": info["construct_name"], "anchor_sha": info["anchor_sha"]}
        if not q["safe_for_ssr"]:
            out.append({**base, "status": "excluded_quality"})
            continue
        v = np.asarray(embed([r["response_text"]]), dtype=float)[0]
        s = cosine_similarities(v, A)
        pm = similarities_to_pmf(s)
        out.append({**base, "status": "scored", **{f"s{k}": float(s[k - 1]) for k in range(1, 6)}, **{f"p{k}": float(pm[k - 1]) for k in range(1, 6)},
                    "expected_rating": float(expected_rating(pm))})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["persona", "control"])
    ap.add_argument("--provider", default="ollama", choices=["ollama", "gemini", "openai"])
    ap.add_argument("--task", default="Q26_BI_proxy", choices=list(TASKS))
    ap.add_argument("--per-persona", type=int, default=1, help="profiles per persona (control gets the same total count)")
    ap.add_argument("--encoder", default="local_bge_base")
    a = ap.parse_args(argv)
    import workflow as wf
    be = {b["provider"]: b for b in wf.backends_status()}[a.provider]
    if not be["available"]:
        print(f"{a.provider} is not available: {be['unavailable_reason']}. Nothing was generated.")
        return 2
    specs = [json.loads(l) for l in (ROOT / "data/persona_generator/respondent_specs_extended.jsonl").read_text(encoding="utf-8").splitlines()]
    prompts = build_prompts(a.condition, a.task, a.per_persona, specs)
    out = ROOT / f"data/pilot_{a.provider}_{a.condition}_{a.task}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8")
    (out / "generation_config.json").write_text(json.dumps({"provider": a.provider, "model": be["model"], "condition": a.condition, "task": a.task,
                                                            "per_persona": a.per_persona, "system_prompt": SYSTEM,
                                                            "user_template": USER_TEMPLATE if a.condition == "persona" else CONTROL_TEMPLATE,
                                                            "note": "Text only; no ratings requested. Control prompts contain no persona information."}, indent=2), encoding="utf-8")
    resp, fail = out / "responses.jsonl", out / "failures.jsonl"
    finished = done_ids(resp)
    for i, p in enumerate([p for p in prompts if p["synthetic_id"] not in finished], 1):
        rec, err = wf.default_generator(a.provider, be["model"], p)
        if rec is None:
            append_jsonl(fail, {"synthetic_id": p["synthetic_id"], "error": err, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
            print(f"[{i}] {p['synthetic_id']} FAILED {(err or {}).get('kind')}", flush=True)
            if (err or {}).get("kind") in ("ollama_not_running", "model_not_found_or_no_access", "invalid_api_key", "no_backend"):
                break
            continue
        append_jsonl(resp, {**rec, "synthetic_id": p["synthetic_id"], "persona_id": p["persona_id"], "condition": a.condition, "task_id": a.task})
        print(f"[{i}] {p['synthetic_id']} ok ({len(rec['response_text'])} chars)", flush=True)
    rows = [json.loads(l) for l in resp.read_text(encoding="utf-8").splitlines() if l.strip()] if resp.exists() else []
    if rows:
        df = score(rows, TASKS[a.task]["construct"], wf.default_embedder_factory(a.encoder))
        df.to_csv(out / "condition_scores.csv", index=False)
        print(f"scored {int((df.status == 'scored').sum())}/{len(df)} -> {(out / 'condition_scores.csv').relative_to(ROOT)}")
    return 0 if len(done_ids(resp)) == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
