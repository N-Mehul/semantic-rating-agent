"""Construct registry and construct-specific prompt builder (workflow stage: "Construct-Specific Prompt").

Everything is read from the stored project files: free-text prompts from docs/survey_instrument/survey_codebook.csv and anchor
sets from the approved docs/survey_instrument/smartphone_construct_anchor_statements.csv (via survey_ssr_bridge). Nothing is
invented: a construct without an approved free-text prompt or without an approved anchor set is reported as blocked.

A "task" = one question wording + the anchor construct used to score it + whether the collected human survey contains a
comparable human answer:
  Q26_BI_proxy  collected-survey Q26 wording (used by all existing Gemini/OpenAI runs); scored with Behavioral Intention anchors;
                human reference = Q26 text / Q21 (a proxy: Q26 mixes price-worth and rebuy, anchors describe continued use)
  FT_BI01, FT_PE01  codebook free-text questions; scored with the BI / PE anchors; the collected survey has NO human answers to
                them, so synthetic results for these tasks cannot be compared with human text.

Run:  python scripts/dp_constructs.py        (writes prompts only; no LLM call)
Outputs (non-destructive): data/construct_prompts/construct_registry.json and <task>_prompts.jsonl (40 prompts per task)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from pilot_generate_api import QUESTION as Q26_QUESTION, SYSTEM, USER_TEMPLATE  # noqa: E402
from ssr_eval_lib import prompt_leak_findings  # noqa: E402
from ssr_persona_matched_evaluation import safe_write_bytes  # noqa: E402
from survey_ssr_bridge import DEFAULT_ANCHOR_CSV_PATH, get_construct_anchors  # noqa: E402

CODEBOOK = ROOT / "docs/survey_instrument/survey_codebook.csv"
OUT = ROOT / "data/construct_prompts"
ANCHOR_CODES = {"PE": "Performance Expectancy", "EE": "Effort Expectancy", "HM": "Hedonic Motivation",
                "SI": "Social Influence", "PP": "Perceived Protection", "BI": "Behavioral Intention"}

# Tasks that can be generated and scored. profile_version says which persona wording the prompt uses.
TASKS = {
    "Q26_BI_proxy": {"construct": "BI", "question_id": "Q26", "profile_version": "v1", "human_reference": "Q26_Q21_proxy",
                     "question_source": "collected survey (Q26)"},
    "FT_BI01": {"construct": "BI", "question_id": "FT_BI01", "profile_version": "v2", "human_reference": None,
                "question_source": "survey_codebook.csv"},
    "FT_PE01": {"construct": "PE", "question_id": "FT_PE01", "profile_version": "v2", "human_reference": None,
                "question_source": "survey_codebook.csv"},
}


def sha_text(*parts: str) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def load_codebook(path: Path = CODEBOOK) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def task_question(task_id: str, codebook: pd.DataFrame) -> str:
    t = TASKS[task_id]
    if t["question_id"] == "Q26":
        return Q26_QUESTION
    row = codebook[codebook.question_id == t["question_id"]]
    if row.empty:
        raise KeyError(f"{t['question_id']} not in codebook")
    return row.iloc[0].question_text


def anchor_info(code: str) -> dict:
    a = get_construct_anchors(code)
    stm = list(a.statements)
    assert len(stm) == 5 and list(a.ratings) == [1, 2, 3, 4, 5], "anchors must be five statements in rating order"
    return {"construct_code": code, "construct_name": a.construct_name, "statements": stm, "anchor_sha": sha_text(*stm),
            "status": "PROPOSED_DOMAIN_ADAPTATION_NOT_ORIGINAL_MAIER_ANCHOR"}


def construct_registry(codebook: pd.DataFrame | None = None) -> list[dict]:
    """One entry per instrument construct, with an explicit readiness status. Computed from the stored files."""
    cb = codebook if codebook is not None else load_codebook()
    rows = []
    names = {**ANCHOR_CODES, "PV": "Price Value (open-ended only)"}
    for code, name in names.items():
        items = cb[cb.construct == name]
        ft = items[items.question_type.str.startswith("Free text")]
        lik = items[~items.question_type.str.startswith("Free text")]
        has_anchors = code in ANCHOR_CODES
        info = anchor_info(code) if has_anchors else None
        has_prompt = len(ft) > 0
        tasks = [t for t, v in TASKS.items() if v["construct"] == code]
        if has_prompt and has_anchors:
            status = "ready_for_generation_and_scoring"
        elif has_anchors:
            status = "blocked_needs_approved_free_text_prompt"
        else:
            status = "blocked_needs_approved_anchor_set"
        human = "none" if code != "BI" else "Q26/Q21 proxy only (collected survey has no FT_BI01 answers)"
        rows.append({"construct_code": code, "construct": name, "free_text_prompt_ids": list(ft.question_id),
                     "free_text_prompts": list(ft.question_text), "likert_item_ids": list(lik.question_id),
                     "approved_anchor_set": has_anchors, "anchor_sha": info["anchor_sha"] if info else None,
                     "anchor_status_label": info["status"] if info else None, "generation_tasks": tasks,
                     "status": status, "human_text_reference_in_collected_survey": human})
    return rows


def build_task_prompts(task_id: str, extended_specs: list[dict], v1_specs: list[dict] | None = None, codebook=None) -> list[dict]:
    """Prompts (text only; no rating requested) for every spec. v1 tasks use the v1 profile text exactly as in earlier runs."""
    cb = codebook if codebook is not None else load_codebook()
    t = TASKS[task_id]
    question = task_question(task_id, cb)
    out = []
    for s in extended_specs:
        profile = s["profile_text_v1"] if t["profile_version"] == "v1" else s["profile_text_v2"]
        p = {"synthetic_id": s["synthetic_id"], "persona_id": s["persona_id"], "task_id": task_id, "construct": t["construct"],
             "question_id": t["question_id"], "profile_version": t["profile_version"], "system": SYSTEM,
             "user": USER_TEMPLATE.format(profile=profile, question=question)}
        bad = prompt_leak_findings(p["system"] + "\n" + p["user"])
        if bad:
            raise RuntimeError(f"prompt for {s['synthetic_id']} ({task_id}) failed leak check: {bad}")
        out.append(p)
    return out


def load_prompts_file(path: Path, limit: int | None = None) -> list[dict]:
    """Validate a prompts JSONL for the generators: required keys, unique ids, leak-free, text only."""
    rows = [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]
    need = {"synthetic_id", "persona_id", "system", "user"}
    ids = [r.get("synthetic_id") for r in rows]
    if not rows or any(need - set(r) for r in rows):
        raise ValueError(f"{path}: every prompt needs {sorted(need)}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate synthetic_id values")
    for r in rows:
        bad = prompt_leak_findings(r["system"] + "\n" + r["user"])
        if bad:
            raise ValueError(f"{path}: prompt {r['synthetic_id']} failed leak check: {bad}")
    return rows[:limit] if limit else rows


def main() -> int:
    ext_path = ROOT / "data/persona_generator/respondent_specs_extended.jsonl"
    if not ext_path.exists():
        print("Run  python scripts/dp_persona_generator.py  first (extended specs missing).")
        return 2
    ext = [json.loads(l) for l in ext_path.open(encoding="utf-8")]
    OUT.mkdir(parents=True, exist_ok=True)
    reg = construct_registry()
    written = []
    summary = {}
    for task in TASKS:
        prompts = build_task_prompts(task, ext)
        written.append(safe_write_bytes("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts).encode("utf-8"),
                                        OUT / f"{task}_prompts.jsonl").name)
        summary[task] = {"n_prompts": len(prompts), **{k: v for k, v in TASKS[task].items()},
                         "anchor_sha": anchor_info(TASKS[task]["construct"])["anchor_sha"],
                         "prompts_file": f"data/construct_prompts/{task}_prompts.jsonl"}
    doc = {"constructs": reg, "tasks": summary, "anchor_file_sha256": hashlib.sha256(Path(DEFAULT_ANCHOR_CSV_PATH).read_bytes()).hexdigest(),
           "note": "Prompts only; no LLM was called. EE/HM/SI/PP have approved anchors but no approved free-text prompt; PV has a prompt but no approved anchors."}
    written.append(safe_write_bytes(json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8"), OUT / "construct_registry.json").name)
    for r in reg:
        print(f"{r['construct_code']:3s} {r['status']}")
    print("written", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
