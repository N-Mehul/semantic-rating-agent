"""Score the quality-passing responses of data/pilot_gemini_v2/ with the existing SSR pipeline (local encoders).

Reads only; calls no API. Uses the approved Behavioral Intention anchors and the canonical SSR functions through
ssr_embedding_comparison.score_texts. Responses failing ssr_eval_lib.response_quality (empty, non-STOP finish reason,
no terminal punctuation, numeric rating in text) are listed in the manifest and not scored.

Outputs (never overwritten; differing content gets _v2.. siblings), in data/pilot_embedding_comparison/:
  synthetic_scores_<config>_gemini_v2.csv   one per local encoder
  gemini_v2_scoring_manifest.json           counts, exclusions, config
Run from repo root:  set HF_HUB_OFFLINE=1  &&  python scripts/ssr_score_gemini_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ssr_embedding_comparison import LOCAL, local_embedder, safe_write_csv, score_texts, synthetic_score_frame  # noqa: E402
from ssr_eval_lib import split_by_quality  # noqa: E402
from survey_ssr_bridge import get_construct_anchors  # noqa: E402

V2 = ROOT / "data/pilot_gemini_v2"
CMP = ROOT / "data/pilot_embedding_comparison"


def load_v2(directory: Path = V2):
    rows = [json.loads(l) for l in (directory / "gemini_generated_responses.jsonl").open(encoding="utf-8") if l.strip()]
    fails = ([json.loads(l) for l in (directory / "failures.jsonl").open(encoding="utf-8") if l.strip()]
             if (directory / "failures.jsonl").exists() else [])
    prompt_ids = [json.loads(l)["synthetic_id"] for l in (directory / "prompts.jsonl").open(encoding="utf-8") if l.strip()]
    return rows, fails, prompt_ids


def main():
    rows, fails, prompt_ids = load_v2()
    passed, excluded = split_by_quality(rows)
    anchors = get_construct_anchors("BI").to_anchor_list()
    written = {}
    for cfg, model in LOCAL.items():
        embed = local_embedder(model)
        P, er = score_texts([r["response_text"] for r in passed], embed, anchors)
        df = synthetic_score_frame(passed, P, er, cfg, "gemini_v2")
        written[cfg] = safe_write_csv(df, CMP / f"synthetic_scores_{cfg}_gemini_v2.csv").name
    got = {r["synthetic_id"] for r in rows}
    manifest = {
        "source_dir": "data/pilot_gemini_v2", "prompts": len(prompt_ids), "responses_on_disk": len(rows),
        "unique_ids": len(got), "passed_quality_and_scored": len(passed),
        "excluded_by_quality": [{"synthetic_id": e["synthetic_id"], "reasons": e["exclusion_reasons"]} for e in excluded],
        "prompt_ids_without_response": sorted(set(prompt_ids) - got),
        "failure_records": len(fails), "failure_kinds": sorted({f.get("kind") for f in fails}),
        "scored_by_persona": {p: sum(1 for r in passed if r["persona_id"].startswith(p)) for p in ("P1", "P2", "P3", "P4")},
        "finish_reasons": sorted({r.get("finish_reason") or "unknown" for r in rows}),
        "anchor_set": "Behavioral Intention (single approved set)", "configs": list(LOCAL), "score_files": written}
    from ssr_persona_matched_evaluation import safe_write_bytes
    p = safe_write_bytes(json.dumps(manifest, indent=2).encode("utf-8"), CMP / "gemini_v2_scoring_manifest.json")
    print(json.dumps({k: manifest[k] for k in ("responses_on_disk", "passed_quality_and_scored", "scored_by_persona",
                                                "score_files")}, indent=1), p.name)


if __name__ == "__main__":
    main()
