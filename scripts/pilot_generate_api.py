"""Step 5 (API version): generate synthetic Q26-style answers with an OpenAI chat model.

STATUS: written but NOT yet executed (OPENAI_API_KEY was absent when this was authored).

Setup (PowerShell):  pip install openai ; $env:OPENAI_API_KEY="sk-..."   (never write the key to a file)
Run from repo root:  python scripts/pilot_generate_api.py [--model gpt-4o-mini] [--limit 40]

Properties
  * inputs: data/pilot/synthetic_respondent_specs.jsonl (non-outcome attributes only; no Q21/Q26/labels)
  * every prompt is leak-checked before any call; the model is told not to give numbers or ratings
  * checkpointing: each finished response is appended immediately; reruns skip finished IDs
  * retries with exponential backoff for rate limits / transient errors; failures logged, not hidden
  * outputs (separate from the manual pilot): data/pilot_api/api_generated_responses.jsonl, failures.jsonl,
    generation_config.json, prompts.jsonl. The API key is never read into any saved artifact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ssr_eval_lib import prompt_leak_findings, response_contains_rating  # noqa: E402

SPECS = ROOT / "data/pilot/synthetic_respondent_specs.jsonl"
OUT = ROOT / "data/pilot_api"
QUESTION = ("Do you think your smartphone is worth the price you paid, and would you choose the same brand again? "
            "Why or why not?")
SYSTEM = ("You role-play one survey respondent answering an open-ended question in your own words, 1-4 sentences, "
          "informal student register. Give reasons that fit the profile. Use words only: no numbers, no scores, "
          "no ratings, no stars. Do not mention being an AI.")
USER_TEMPLATE = "Respondent profile: {profile}.\n\nQuestion: {question}"
SETTINGS = {"temperature": 1.0, "max_tokens": 150, "top_p": 1.0}


def profile_text(s: dict) -> str:
    return (f"{s['gender']}, student in a {s['area_type'].lower()}; phone brand {s['brand']} ({s['operating_system']}), "
            f"paid {s['price_paid']}; has used it for {s['usage_duration']}; replaces phones {s['replacement_frequency'].lower()}; "
            f"upgraded because: {s['upgrade_reason'].lower().rstrip('.')}; uses it {s['daily_usage']} per day; "
            f"comfort with new technology {s['tech_comfort_1to5']} on a 1-5 scale")


def load_specs(limit):
    specs = [json.loads(l) for l in open(SPECS, encoding="utf-8")]
    return specs[:limit] if limit else specs


def build_prompts(specs):
    out = []
    for s in specs:
        user = USER_TEMPLATE.format(profile=profile_text(s), question=QUESTION)
        bad = prompt_leak_findings(SYSTEM + "\n" + user)
        if bad:
            raise RuntimeError(f"prompt for {s['synthetic_id']} failed leak check: {bad}")
        out.append({"synthetic_id": s["synthetic_id"], "persona_id": s["persona_id"], "system": SYSTEM, "user": user})
    return out


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["synthetic_id"] for l in path.open(encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_GEN_MODEL", "gpt-4o-mini"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true", help="build and leak-check prompts only; no API call")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(load_specs(a.limit))
    with open(OUT / "prompts.jsonl", "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if a.dry_run:
        print(f"dry run ok: {len(prompts)} prompts passed leak check")
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Set it in your shell (PowerShell: $env:OPENAI_API_KEY='sk-...') and rerun.")
        return 2
    try:
        from openai import APIError, OpenAI, RateLimitError
    except ImportError:
        print("The openai package is missing: pip install openai")
        return 2

    client = OpenAI()
    resp_path, fail_path = OUT / "api_generated_responses.jsonl", OUT / "failures.jsonl"
    finished = done_ids(resp_path)
    (OUT / "generation_config.json").write_text(json.dumps({
        "model_requested": a.model, "settings": SETTINGS, "system_prompt": SYSTEM, "user_template": USER_TEMPLATE,
        "question": QUESTION, "specs_file": str(SPECS.relative_to(ROOT)), "label": "api_generated",
        "note": "No Q21, Q26 or evaluation labels are included in any prompt (leak-checked)."}, indent=2), encoding="utf-8")

    for p in prompts:
        if p["synthetic_id"] in finished:
            continue
        for attempt in range(a.max_retries):
            try:
                r = client.chat.completions.create(
                    model=a.model, messages=[{"role": "system", "content": p["system"]},
                                             {"role": "user", "content": p["user"]}], **SETTINGS)
                text = (r.choices[0].message.content or "").strip()
                rec = {"synthetic_id": p["synthetic_id"], "persona_id": p["persona_id"], "response_text": text,
                       "source": "api_generated", "model_returned": r.model, "response_id": r.id,
                       "finish_reason": r.choices[0].finish_reason,
                       "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                       "flag_contains_rating": response_contains_rating(text)}
                with open(resp_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                break
            except (RateLimitError, APIError) as exc:
                wait = min(60, 2 ** attempt)
                if attempt == a.max_retries - 1:
                    with open(fail_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"synthetic_id": p["synthetic_id"], "error": type(exc).__name__,
                                            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}) + "\n")
                else:
                    time.sleep(wait)
    print(f"done: {len(done_ids(resp_path))}/{len(prompts)} responses in {resp_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
