"""FAST validation: 1 product x 4 personas x 6 constructs = 24 responses.
Generation + Improved SSR via the existing pipeline (ProductEvaluationService.run_evaluation, unmodified).
Independent LLM-judge reference ratings (judge never sees SSR). Outputs -> evaluation/quick_ssr_validation/.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "dashboard"))
from dashboard.evaluation_service import ProductEvaluationService  # noqa
from scripts import ollama_utils  # noqa
from survey_ssr_bridge import get_construct_anchors  # noqa
from scripts.product_evaluation_prompts import load_persona_data  # noqa

CONS = ["PE", "EE", "HM", "SI", "PP", "BI"]; L = [1, 2, 3, 4, 5]
SRC = ROOT / "data/product_evaluation/echo_vision_smart_glasses_20261103_154257/product_concept.json"

# ---------------- 1. generation + SSR through the existing pipeline (or reuse a previous quick run)
state = OUT / "generation_run.json"
if state.exists():
    eval_id = json.loads(state.read_text())["eval_id"]
else:
    c = json.loads(SRC.read_text(encoding="utf-8"))
    body = {k: c[k] for k in ["name", "category", "description", "price_inr", "target_group", "target_age_group", "market", "brand_or_ecosystem", "features"]}
    body.update({"constructs": CONS, "ssr_method": "compare", "target_persona": "all", "target_construct": "all"})
    res = ProductEvaluationService(ROOT).run_evaluation(body)
    eval_id = res.get("eval_id") or res.get("id") or sorted((ROOT / "data/product_evaluation").glob("echo_vision_smart_glasses_*"))[-1].name
    state.write_text(json.dumps({"eval_id": eval_id, "source_concept": str(SRC.relative_to(ROOT))}, indent=2))
run = ROOT / "data/product_evaluation" / eval_id
concept = json.loads((run / "product_concept.json").read_text(encoding="utf-8"))
resps = json.loads((run / "persona_responses.json").read_text(encoding="utf-8"))
ssr = pd.read_csv(run / "ssr_results/ssr_results.csv")
sj = json.loads((run / "ssr_results/ssr_results.json").read_text(encoding="utf-8"))

# improved PMF per (persona, construct) from the JSON detail
imp_pmf = {}
def walk(o, pid=None):
    if isinstance(o, dict):
        pid = o.get("persona_id", pid)
        code = o.get("construct_code") or o.get("anchor_set_code") or o.get("code")
        if "improved" in o and isinstance(o["improved"], dict) and "pmf_probabilities" in o["improved"] and code:
            imp_pmf[(pid, code)] = o["improved"]["pmf_probabilities"]
        for v in o.values(): walk(v, pid)
    elif isinstance(o, list):
        for v in o: walk(v, pid)
walk(sj)

rows = []
for r in resps:
    for code in CONS:
        s = ssr[(ssr.persona_id == r["persona_id"]) & (ssr.construct_code == code)].iloc[0]
        rows.append({"persona_id": r["persona_id"], "construct": code, "response_text": (r.get("construct_responses") or {}).get(code) or r["response_text"],
                     "used_overall_fallback": not bool((r.get("construct_responses") or {}).get(code)),
                     "improved_E": float(s.improved_expected_rating), "canonical_E": float(s.canonical_expected_rating),
                     "improved_pmf": imp_pmf.get((r["persona_id"], code))})
D = pd.DataFrame(rows)
D["improved_rounded"] = np.clip(np.round(D.improved_E), 1, 5).astype(int)
D["improved_argmax"] = [int(np.argmax(p)) + 1 if p else np.nan for p in D.improved_pmf]

# ---------------- 2. independent LLM judge (no SSR shown)
JUDGE_MODEL = "qwen2.5:7b-instruct"
SYS = ("You are a strict, independent survey-methodology rater. You rate ONE consumer response on ONE construct. "
       "Use only the anchors given. Return JSON only.")
def persona_desc(pid):
    try:
        p = load_persona_data(pid, ROOT); keys = ["persona_id", "defining_rule", "median_age", "current_status", "operating_system", "brand"]
        return json.dumps({k: p.get(k) for k in keys if k in p}, ensure_ascii=False)[:900]
    except Exception:
        return pid
cache_f = OUT / "judge_raw.jsonl"; cache = {}
if cache_f.exists():
    for l in cache_f.read_text(encoding="utf-8").splitlines():
        j = json.loads(l); cache[(j["persona_id"], j["construct"])] = j
for i, r in D.iterrows():
    key = (r.persona_id, r.construct)
    if key in cache: continue
    an = get_construct_anchors(r.construct)
    anchors = "\n".join(f"{k}: {t}" for k, t in zip(an.ratings, an.statements))
    user = f"""PRODUCT: {concept['name']} ({concept.get('category','')}) - {concept['description'][:700]}
PERSONA PROFILE: {persona_desc(r.persona_id)}
CONSTRUCT: {r.construct} ({an.construct_name})
APPROVED LIKERT ANCHORS (1-5):
{anchors}

RESPONSE:
\"\"\"{r.response_text}\"\"\"

Tasks:
1. reference_rating: integer 1-5 = which anchor best matches the attitude expressed in the response for this construct.
2. construct_aligned: true if the response actually addresses this construct ({an.construct_name}).
3. product_relevant: true if it is clearly about this product ({concept.get('category','')}), not a different product type.
4. persona_consistent: true if it fits the persona profile.
5. naturalness: integer 1-5 (5 = natural, realistic consumer voice).
6. issues: short list of problems (empty if none).
Return JSON: {{"reference_rating":int,"construct_aligned":bool,"product_relevant":bool,"persona_consistent":bool,"naturalness":int,"issues":[str]}}"""
    ollama_utils.SETTINGS["temperature"] = 0.0
    rec, err = ollama_utils.generate(JUDGE_MODEL, {"system": SYS, "user": user, "json": True, "timeout_s": 300.0, "max_output_tokens": 300}, timeout=300.0)
    if err: raise RuntimeError(f"judge failed {key}: {err}")
    txt = rec["response_text"]; m = re.search(r"\{.*\}", txt, re.S); j = json.loads(m.group(0))
    j.update({"persona_id": r.persona_id, "construct": r.construct, "judge_model": JUDGE_MODEL, "raw": txt})
    cache[key] = j
    with cache_f.open("a", encoding="utf-8") as f: f.write(json.dumps(j, ensure_ascii=False) + "\n")
for col in ["reference_rating", "construct_aligned", "product_relevant", "persona_consistent", "naturalness", "issues"]:
    D["judge_" + col] = [cache[(p, c)].get(col) for p, c in zip(D.persona_id, D.construct)]
D["judge_reference_rating"] = D.judge_reference_rating.astype(int)

# embedding-based construct alignment (nearest anchor set, same nomic encoder as the pipeline)
from scripts.ollama_embedding_adapter import get_ollama_embedding_adapter  # noqa
ad = get_ollama_embedding_adapter("nomic-embed-text")
AE = {c: np.asarray(ad.embed_texts(list(get_construct_anchors(c).statements))) for c in CONS}
def cos(a, B): return B @ a / (np.linalg.norm(B, axis=1) * np.linalg.norm(a))
D["nearest_anchor_set"] = [max(CONS, key=lambda c: cos(np.asarray(ad.embed_text(t)), AE[c]).mean()) for t in D.response_text]
D["embedding_construct_aligned"] = D.nearest_anchor_set == D.construct
D["mentions_other_product_type"] = D.response_text.str.contains(r"\b(smartphone|laptop|earbuds|tablet|watch)\b", case=False)
D["contains_numeric_rating"] = D.response_text.str.contains(r"\b[1-5]\s*(/|out of)\s*5\b|\bstars?\b|\brating\b", case=False)

# ---------------- 3. metrics (LLM-judge reference, NOT human ground truth)
y = D.judge_reference_rating.values; e = D.improved_E.values; pr = D.improved_rounded.values
def mets(pred_int, pred_cont):
    return {"exact_agreement_pct": float((pred_int == y).mean() * 100), "within_1_pct": float((np.abs(pred_int - y) <= 1).mean() * 100),
            "MAE": float(np.abs(pred_cont - y).mean()), "RMSE": float(np.sqrt(((pred_cont - y) ** 2).mean())),
            "spearman_rho": float(stats.spearmanr(pred_cont, y)[0]), "spearman_p": float(stats.spearmanr(pred_cont, y)[1]),
            "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred_int, weights="quadratic", labels=L)),
            "linear_weighted_kappa": float(cohen_kappa_score(y, pred_int, weights="linear", labels=L)),
            "mean_signed_error_pred_minus_judge": float((pred_cont - y).mean())}
M = {"label": "Agreement with LLM-JUDGE REFERENCE ratings (NOT human ground truth)", "n_responses": int(len(D)), "product": concept["name"],
     "generation_run": f"data/product_evaluation/{eval_id}", "generator": "qwen2.5:7b-instruct (existing pipeline)", "judge": JUDGE_MODEL + " temperature 0, blind to SSR",
     "judge_independence_caveat": "judge is the same model family as the generator (OpenAI unavailable: no credits)",
     "improved_ssr_rounded_E": mets(pr, e), "improved_ssr_argmax": mets(D.improved_argmax.values.astype(int), e) if D.improved_argmax.notna().all() else None,
     "canonical_ssr_rounded_E": mets(np.clip(np.round(D.canonical_E), 1, 5).astype(int), D.canonical_E.values),
     "constant_baseline_judge_mode": {"exact_agreement_pct": float((y == stats.mode(y, keepdims=False).mode).mean() * 100)},
     "distribution": {"judge": {int(k): int((y == k).sum()) for k in L}, "improved_rounded": {int(k): int((pr == k).sum()) for k in L}},
     "spread": {"judge_sd": float(y.std(ddof=1)), "improved_E_sd": float(e.std(ddof=1)), "improved_E_range": [float(e.min()), float(e.max())]}}
cm = confusion_matrix(y, pr, labels=L); M["confusion_matrix_rows_judge_cols_improved_rounded"] = cm.tolist()
M["per_construct"] = {c: {"judge_mean": float(g.judge_reference_rating.mean()), "ssr_mean": float(g.improved_E.mean()), "MAE": float(np.abs(g.improved_E - g.judge_reference_rating).mean()),
                          "exact_pct": float((g.improved_rounded == g.judge_reference_rating).mean() * 100)} for c, g in D.groupby("construct")}
pd_ = {}
for nm, col in [("improved_E", "improved_E"), ("judge_rating", "judge_reference_rating")]:
    g = [D[D.persona_id == p][col].values for p in sorted(D.persona_id.unique())]
    H, p = stats.kruskal(*g) if np.ptp(D[col]) > 0 else (np.nan, np.nan)
    pd_[nm] = {"persona_means": {pp: float(D[D.persona_id == pp][col].mean()) for pp in sorted(D.persona_id.unique())}, "kruskal_H": float(H), "kruskal_p": float(p),
               "range_of_persona_means": float(D.groupby("persona_id")[col].mean().ptp() if hasattr(pd.Series, 'ptp') else np.ptp(D.groupby("persona_id")[col].mean().values))}
# text-level persona differentiation: mean cosine between same-construct responses of different personas
txt_emb = np.array([ad.embed_text(t) for t in D.response_text]); nrm = txt_emb / np.linalg.norm(txt_emb, axis=1, keepdims=True)
sims = [float(nrm[i] @ nrm[j]) for i in range(len(D)) for j in range(i + 1, len(D)) if D.construct[i] == D.construct[j]]
pd_["text_same_construct_cross_persona_mean_cosine"] = float(np.mean(sims)); pd_["note"] = "n=6 per persona: very low power; descriptive only"
M["persona_differentiation"] = pd_
M["quality"] = {"construct_alignment_judge_pct": float(D.judge_construct_aligned.astype(bool).mean() * 100), "construct_alignment_embedding_pct": float(D.embedding_construct_aligned.mean() * 100),
                "product_relevance_judge_pct": float(D.judge_product_relevant.astype(bool).mean() * 100), "persona_consistency_judge_pct": float(D.judge_persona_consistent.astype(bool).mean() * 100),
                "naturalness_judge_mean_1to5": float(pd.to_numeric(D.judge_naturalness).mean()), "mentions_other_product_type_n": int(D.mentions_other_product_type.sum()),
                "contains_numeric_rating_n": int(D.contains_numeric_rating.sum()), "overall_fallback_used_n": int(D.used_overall_fallback.sum()),
                "unique_texts": int(D.response_text.nunique()), "mean_words": float(D.response_text.str.split().str.len().mean())}
D["abs_error"] = (D.improved_E - D.judge_reference_rating).abs()
M["largest_disagreements"] = D.sort_values("abs_error", ascending=False).head(5)[["persona_id", "construct", "judge_reference_rating", "improved_E", "response_text"]].to_dict("records")

D.drop(columns=["improved_pmf"]).assign(improved_pmf=[json.dumps(p) for p in D.improved_pmf]).to_csv(OUT / "validation_results.csv", index=False)
(OUT / "validation_metrics.json").write_text(json.dumps(M, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
pd.DataFrame(cm, index=[f"judge_{k}" for k in L], columns=[f"ssr_{k}" for k in L]).to_csv(OUT / "confusion_matrix.csv")
fig, ax = plt.subplots(figsize=(5, 4.3)); ax.imshow(cm, cmap="Blues")
for i in range(5):
    for j in range(5): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks(range(5), L); ax.set_yticks(range(5), L); ax.set_xlabel("Improved SSR (rounded E[r])"); ax.set_ylabel("LLM-judge reference rating")
ax.set_title(f"Improved SSR vs LLM judge (n={len(D)})\nNOT human ground truth", fontsize=10); fig.savefig(OUT / "confusion_matrix.png", dpi=300, bbox_inches="tight")
print(json.dumps({k: M[k] for k in ["n_responses", "improved_ssr_rounded_E", "distribution", "spread", "quality"]}, indent=1, default=str))
