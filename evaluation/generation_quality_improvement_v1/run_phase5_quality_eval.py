"""Phase 5: generation-quality evaluation (no SSR, no Likert validation).
Deterministic metrics on BOTH baseline (legacy pipeline) and improved (v3) sets; Qwen 2.5 7B judge on the 24 improved only.
Writes: phase5_quality_evaluation.json (+ judge cache phase5_judge_raw.jsonl). Responses are read, never modified.
"""
from __future__ import annotations
import itertools, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts.run_product_evaluation import NOVA_AI_SMARTPHONE as P, PERSONA_IDS  # noqa: E402
from scripts import ollama_utils  # noqa: E402
import product_evaluation_prompts_v2 as v2  # noqa: E402
from survey_ssr_bridge import get_construct_anchors  # noqa: E402

IMP = ROOT / "data/product_evaluation/nova_ai_smartphone_v3_improved_20261104_211250"
BASE = ROOT / "data/product_evaluation/nova_ai_smartphone_20261103_132323"
C = list(v2.CONSTRUCT_ORDER)
JUDGE = "qwen2.5:7b-instruct"


def load(d):
    r = json.loads((d / "persona_responses.json").read_text(encoding="utf-8"))
    by = {x["persona_id"]: x for x in r}
    return {(p, c): (by[p].get("construct_responses") or {}).get(c) or "" for p in PERSONA_IDS for c in C}


sets = {"baseline": load(BASE), "improved": load(IMP)}
specs = {p: v2.load_spec(p) for p in PERSONA_IDS}
from sentence_transformers import SentenceTransformer  # noqa: E402
enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
emb = lambda xs: (lambda E: E / np.linalg.norm(E, axis=1, keepdims=True))(np.asarray(enc.encode(list(xs), show_progress_bar=False)))
ANCH = {c: emb(get_construct_anchors(c).to_anchor_list()) for c in C}

HEDGE = re.compile(r"\b(but|though|although|however|not sure|unsure|worr\w*|concern\w*|a bit|might|maybe|could be better|pricey|expensive|hesita\w*|doubt\w*|not really|wish)\b", re.I)
NEG = re.compile(r"\b(not|no|never|don't|doesn't|won't|wouldn't|isn't|bad|poor|disappoint\w*|annoy\w*|problem\w*|issue\w*|lack\w*)\b", re.I)
POS = re.compile(r"\b(good|great|love|enjoy\w*|fun|easy|smooth|helpful|handy|useful|safe\w*|happy|nice|cool|impressive|reliable)\b", re.I)


def det(texts, use_spec):
    keys = list(texts); T = [texts[k] for k in keys]; E = emb(T)
    out = {"n": len(T), "empty": sum(not t for t in T), "mean_words": float(np.mean([len(t.split()) for t in T]))}
    openings3 = [v2.opening(t, 3) for t in T]
    out["starts_this_phone_or_smartphone"] = sum(bool(re.match(r"\s*(this|the) (phone|smartphone)\b", t, re.I)) for t in T)
    out["distinct_3word_openings"] = len(set(openings3))
    w = [x for t in T for x in re.findall(r"[a-z']+", t.lower())]; bg = list(zip(w, w[1:]))
    out["distinct_2"] = len(set(bg)) / max(1, len(bg))
    cross = []; shared = []
    for c in C:
        idx = [i for i, k in enumerate(keys) if k[1] == c]
        for i, j in itertools.combinations(idx, 2):
            cross.append(float(E[i] @ E[j]))
            shared.append(len(v2._ngrams(T[i]) & v2._ngrams(T[j])))
    out["cross_persona_same_construct_cosine_mean"] = float(np.mean(cross))
    out["persona_differentiation_score_1_minus_cos"] = float(1 - np.mean(cross))
    out["shared_5grams_per_pair_mean"] = float(np.mean(shared)); out["pairs_with_shared_5gram"] = int(sum(s > 0 for s in shared)); out["pairs"] = len(shared)
    hm = [T[i] for i, k in enumerate(keys) if k[1] == "HM"]
    out["HM_same_4word_opening_max_count"] = max(openings3.count(o) for o in openings3)
    out["HM_openings"] = [v2.opening(t, 6) for t in hm]
    # construct relevance: nearest construct anchor set (mpnet) + lexicon alignment
    near = [max(C, key=lambda c: float((ANCH[c] @ E[i]).mean())) for i in range(len(T))]
    out["nearest_anchor_set_matches_construct"] = sum(n == k[1] for n, k in zip(near, keys))
    per = {}
    flags = []
    for i, k in enumerate(keys):
        others = [texts[(p, k[1])] for p in PERSONA_IDS if p != k[0]]
        q = v2.quality_flags(T[i], k[1], P, specs[k[0]] if use_spec else None, other_texts=others); flags.append(q)
        per.setdefault(k[1], []).append({"nearest_anchor_set": near[i], "aligned_lexicon": q["construct_aligned"], "off_construct": q["off_construct_content"],
                                         "contamination": q["dominant_other_construct"]})
    out["lexicon_construct_aligned"] = sum(q["construct_aligned"] for q in flags)
    out["off_construct_content"] = sum(q["off_construct_content"] for q in flags)
    out["possible_contamination"] = sum(bool(q["dominant_other_construct"]) for q in flags)
    out["rating_leak"] = sum(bool(q["rating_leak"]) for q in flags)
    out["refers_to_product_or_feature"] = sum(q["refers_to_product"] or q["mentions_listed_feature"] for q in flags)
    out["names_product"] = sum(bool(re.search(r"\bnova\b", t, re.I)) for t in T)
    out["unsupported_claims"] = sum(bool(q["unsupported_spec_claims"] or q["unsupported_price"]) for q in flags)
    out["other_brand_or_device"] = sum(q["other_brand_mentioned"] or q["other_device_mentioned"] for q in flags)
    out["invented_history_or_contradiction"] = sum(bool(q["persona_or_history_contradiction"]) for q in flags)
    out["responses_with_hedge_or_mixed_view"] = sum(bool(HEDGE.search(t)) for t in T)
    pol = [len(POS.findall(t)) - len(NEG.findall(t)) for t in T]
    out["lexical_polarity_sd"] = float(np.std(pol)); out["share_purely_positive"] = float(np.mean([bool(POS.search(t)) and not HEDGE.search(t) and not NEG.search(t) for t in T]))
    out["responses_with_any_flag"] = sum(bool(q["issues"]) for q in flags)
    out["issue_counts"] = {}
    for q in flags:
        for i_ in q["issues"]:
            out["issue_counts"][i_] = out["issue_counts"].get(i_, 0) + 1
    out["per_construct"] = {c: {"nearest_anchor_matches": sum(x["nearest_anchor_set"] == c for x in v),
                                "lexicon_aligned": sum(x["aligned_lexicon"] for x in v), "off_construct": sum(x["off_construct"] for x in v),
                                "contamination": [x["contamination"] for x in v if x["contamination"]]} for c, v in per.items()}
    return out


results = {"baseline_run": str(BASE.relative_to(ROOT)), "improved_run": str(IMP.relative_to(ROOT)),
           "deterministic": {k: det(v, use_spec=(k == "improved")) for k, v in sets.items()}}

# ---------------- LLM judge: improved 24 only (same Qwen model as the generator -> bias disclosed)
SYS = ("You are a strict survey-methodology reviewer. You assess ONE synthetic survey answer. Do not rate the product. "
       "Be critical: generic, templated or overly positive text should get low scores. Return JSON only.")
DIMS = ["construct_relevance", "construct_isolation", "persona_consistency", "naturalness", "product_relevance", "nuance"]
cache_f = HERE / "phase5_judge_raw.jsonl"; cache = {}
if cache_f.exists():
    for line in cache_f.read_text(encoding="utf-8").splitlines():
        j = json.loads(line); cache[(j["persona_id"], j["construct"])] = j
ollama_utils.SETTINGS["temperature"] = 0.0
for p in PERSONA_IDS:
    for c in C:
        if (p, c) in cache:
            continue
        cd = v2.CONSTRUCTS[c]
        user = (f"{v2.persona_block(p, specs[p])}\n\n{v2.persona_drivers(p, specs[p], P)}\n\n{v2.product_block(P)}\n\n"
                f"CONSTRUCT: {c} {cd['name']} - {cd['definition']}\nSurvey statements: {'; '.join(v2.survey_items(c))}\n\n"
                f"ANSWER TO REVIEW:\n\"\"\"{sets['improved'][(p, c)]}\"\"\"\n\n"
                "Score each 1-5 (5 = best): construct_relevance (addresses this construct), construct_isolation (does not drift into other "
                "constructs), persona_consistency (fits this respondent's profile, needs and price position), naturalness (sounds like a real "
                "person, not a template), product_relevance (about this specific phone and its listed features), nuance (honest, specific, "
                "includes reservations where reasonable rather than generic praise). Also: unsupported_claims (true/false: claims facts not in "
                "the product description or profile), invented_details (list), issues (short list).\n"
                'Return JSON: {"construct_relevance":int,"construct_isolation":int,"persona_consistency":int,"naturalness":int,'
                '"product_relevance":int,"nuance":int,"unsupported_claims":bool,"invented_details":[str],"issues":[str]}')
        rec, err = ollama_utils.generate(JUDGE, {"system": SYS, "user": user, "json": True, "timeout_s": 300.0, "max_output_tokens": 250, "num_ctx": 2048}, timeout=300.0)
        if err:
            raise RuntimeError(f"judge failed {p} {c}: {err}")
        j = json.loads(re.search(r"\{.*\}", rec["response_text"], re.S).group(0))
        j.update({"persona_id": p, "construct": c, "judge_model": JUDGE, "prompt_tokens": (rec.get("usage") or {}).get("prompt_tokens")})
        cache[(p, c)] = j
        with cache_f.open("a", encoding="utf-8") as f:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
J = [cache[(p, c)] for p in PERSONA_IDS for c in C]
score = lambda j, d: float(j.get(d)) if isinstance(j.get(d), (int, float)) else np.nan
dim = {d: float(np.nanmean([score(j, d) for j in J])) for d in DIMS}
results["llm_judge_improved_only"] = {
    "judge": JUDGE + " (temperature 0) - SAME model family as the generator; LLM-judge bias; NOT human ground truth",
    "n": len(J), "dimension_means_1to5": dim, "overall_mean_1to5": float(np.mean(list(dim.values()))),
    "by_construct": {c: {d: float(np.nanmean([score(cache[(p, c)], d) for p in PERSONA_IDS])) for d in DIMS} for c in C},
    "by_persona": {p: float(np.nanmean([score(cache[(p, c)], d) for c in C for d in DIMS])) for p in PERSONA_IDS},
    "unsupported_claims_true": sum(bool(j.get("unsupported_claims")) for j in J),
    "responses_with_judge_issues": sum(bool(j.get("issues")) for j in J),
    "low_scores_le2": [{"persona": j["persona_id"], "construct": j["construct"], "dims": [d for d in DIMS if score(j, d) <= 2]} for j in J if any(score(j, d) <= 2 for d in DIMS)],
    "issues_examples": [{"persona": j["persona_id"], "construct": j["construct"], "issues": j.get("issues"), "invented": j.get("invented_details")} for j in J if j.get("issues") or j.get("invented_details")][:12],
    "score_variance_note": f"distinct overall scores: {sorted({round(np.mean([score(j, d) for d in DIMS]), 2) for j in J})}",
}
(HERE / "phase5_quality_evaluation.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps(results, indent=1, ensure_ascii=False, default=float)[:9000])
