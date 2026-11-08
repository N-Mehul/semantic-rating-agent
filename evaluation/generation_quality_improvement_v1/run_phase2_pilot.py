"""Phase 2 one-call diagnostic pilot: 1 persona x PP x Nova AI Smartphone. Not part of the 24-response experiment. No SSR."""
from __future__ import annotations
import json, re, sys, time, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent / "phase2_diagnostic_pilot.json"
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts.run_product_evaluation import NOVA_AI_SMARTPHONE as PRODUCT  # noqa
from scripts import ollama_utils  # noqa
import product_evaluation_prompts_v2 as v2  # noqa

if OUT.exists():
    sys.exit(f"{OUT} exists; refusing to overwrite (one-call pilot already run)")
PERSONA, CONSTRUCTS, MODEL, TEMPERATURE = "P2_midrange_heavy_user", ["PP"], "qwen2.5:7b-instruct", 0.7
prompt = v2.build_prompt_v2(PRODUCT, PERSONA, CONSTRUCTS)
plan = v2.plan_context(prompt)
ollama_utils.SETTINGS["temperature"] = TEMPERATURE
t0 = time.perf_counter()
rec, err = ollama_utils.generate(MODEL, v2.ollama_request(prompt, plan), timeout=300.0)
elapsed = round(time.perf_counter() - t0, 1)
result = {"label": "PHASE 2 DIAGNOSTIC PILOT - not part of the 24-response experiment; no SSR computed",
          "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "model": MODEL, "temperature": TEMPERATURE,
          "prompt_version": prompt["prompt_version"], "persona_id": PERSONA, "respondent_spec_id": prompt["respondent_spec_id"],
          "constructs": CONSTRUCTS, "product": PRODUCT.name, "elapsed_seconds": elapsed, "error": err,
          "prompt": {"system": prompt["system"], "user": prompt["user"]}}
if rec:
    usage = rec.get("usage") or {}
    result["context"] = v2.context_diagnostic(plan, usage.get("prompt_tokens"), usage.get("output_tokens"))
    result["finish_reason"] = rec.get("finish_reason")
    result["raw_output"] = rec["response_text"]
    parsed = v2.parse_response_v2(rec["response_text"], CONSTRUCTS); result["parsed"] = parsed
    t = parsed["responses"].get("PP") or ""
    result["checks"] = {
        "parse_ok": parsed["parse_ok"], "pp_present": bool(t),
        "privacy_terms": sorted(set(m.lower() for m in re.findall(r"privacy|private|personal (?:data|information)|data|secur\w*|cloud|on-device|locally|location|account\w*|payment\w*|permission\w*|misuse|unauthori[sz]ed|control", t, re.I))),
        "durability_terms": sorted(set(m.lower() for m in re.findall(r"durab\w*|build quality|sturdy|robust|wear and tear|last\w* long|hardware reliab\w*", t, re.I))),
        "refers_to_product": bool(re.search(r"nova|this phone", t, re.I)),
        "mentions_generic_other_device": bool(re.search(r"\b(laptop|glasses|earbuds|tablet)\b", t, re.I)),
        "rating_leak": parsed["rating_leak"],
        "sentences": len([s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]), "words": len(t.split()),
        "invented_events": bool(re.search(r"\b(last (week|month|year)|was hacked|got hacked|repair\w*|for (two|three|six) months)\b", t, re.I)),
    }
OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: result.get(k) for k in ["elapsed_seconds", "error", "context", "finish_reason", "checks"]}, indent=1, ensure_ascii=False))
print(result.get("parsed", {}).get("responses"))
