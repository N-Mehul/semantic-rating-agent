"""Phase 6: SSR scoring of the 24 improved responses + same evaluation on the 24 baseline responses.
SSR maths imported unmodified (canonical_ssr / improved_ssr); k=3, T=0.75; nomic-embed-text; six approved anchor sets.
Baseline: stored canonical SSR is re-used (cosines read from its intermediate CSV, never re-embedded or overwritten).
Reference ratings: blind LLM-judge protocol of evaluation/quick_ssr_validation (identical prompt), Qwen 2.5 7B.
All outputs -> evaluation/generation_quality_improvement_v1/phase6_ssr/.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent / "phase6_ssr"; OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from canonical_ssr import similarities_to_pmf, cosine_similarities, expected_rating, PAPER_EPSILON, PAPER_TEMPERATURE  # noqa: E402
from improved_ssr import improved_similarities_to_pmf, DEFAULT_IMPROVED_K, DEFAULT_IMPROVED_TEMPERATURE  # noqa: E402
from survey_ssr_bridge import get_construct_anchors  # noqa: E402
from scripts.ollama_embedding_adapter import get_ollama_embedding_adapter  # noqa: E402
from scripts.product_evaluation_prompts import load_persona_data  # noqa: E402
from scripts.run_product_evaluation import PERSONA_IDS  # noqa: E402
from scripts import ollama_utils  # noqa: E402

assert (DEFAULT_IMPROVED_K, DEFAULT_IMPROVED_TEMPERATURE) == (3, 0.75)
IMP = ROOT / "data/product_evaluation/nova_ai_smartphone_v3_improved_20261104_211250"
BASE = ROOT / "data/product_evaluation/nova_ai_smartphone_20261103_132323"
C = ["PE", "EE", "HM", "SI", "PP", "BI"]; L = [1, 2, 3, 4, 5]; R = np.arange(1, 6)
EMB = "nomic-embed-text"; JUDGE = "qwen2.5:7b-instruct"


def texts(d):
    by = {x["persona_id"]: x for x in json.loads((d / "persona_responses.json").read_text(encoding="utf-8"))}
    return {(p, c): (by[p].get("construct_responses") or {}).get(c) or "" for p in PERSONA_IDS for c in C}


def score_rows(sims_by_key, set_name, T):
    rows = []
    for (p, c), s in sims_by_key.items():
        s = np.asarray(s, float)
        pc = similarities_to_pmf(s, epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE)
        pi = improved_similarities_to_pmf(s, k=DEFAULT_IMPROVED_K, temperature=DEFAULT_IMPROVED_TEMPERATURE)
        ei, ec = expected_rating(pi), expected_rating(pc)
        ent = float(-(pi[pi > 0] * np.log2(pi[pi > 0])).sum())
        rows.append({"set": set_name, "persona_id": p, "construct": c, "response_text": T[(p, c)],
                     **{f"cos_{k}": float(s[k - 1]) for k in L},
                     **{f"canon_p{k}": float(pc[k - 1]) for k in L}, "canon_E": float(ec),
                     **{f"imp_p{k}": float(pi[k - 1]) for k in L}, "imp_E": float(ei),
                     "imp_rounded": int(np.clip(np.round(ei), 1, 5)), "imp_argmax": int(np.argmax(pi)) + 1,
                     "imp_pmf_sd": float(np.sqrt((pi * (R - ei) ** 2).sum())), "imp_entropy_bits": ent, "imp_max_prob": float(pi.max()),
                     "cos_spread": float(s.max() - s.min())})
    return rows


# ---------- improved: embed with nomic-embed-text (embedding only, no generation)
Ti, Tb = texts(IMP), texts(BASE)
ad = get_ollama_embedding_adapter(EMB)
A = {c: np.asarray(ad.embed_texts(list(get_construct_anchors(c).statements))) for c in C}
keys = [(p, c) for p in PERSONA_IDS for c in C]
E = np.asarray(ad.embed_texts([Ti[k] for k in keys]))
sims_i = {k: cosine_similarities(E[i], A[k[1]]) for i, k in enumerate(keys)}
# ---------- baseline: stored cosines (canonical results already exist; reproduced, not overwritten)
inter = pd.read_csv(BASE / "ssr_results/ssr_intermediate_calculations.csv")
sims_b = {(r.persona_id, r.anchor_set_code): [r[f"cos_sim_{k}"] for k in L] for _, r in inter.iterrows() if r.anchor_set_code in C}
assert len(sims_b) == 24 and len(sims_i) == 24
S = pd.DataFrame(score_rows(sims_i, "improved", Ti) + score_rows(sims_b, "baseline", Tb))
stored = pd.read_csv(BASE / "ssr_results/ssr_results.csv").set_index(["persona_id", "construct_code"]).expected_rating
repro = S[S.set == "baseline"].apply(lambda r: abs(r.canon_E - stored[(r.persona_id, r.construct)]), axis=1).max()

# ---------- reference ratings: identical blind judge prompt to quick_ssr_validation (judge never sees SSR)
SYS = ("You are a strict, independent survey-methodology rater. You rate ONE consumer response on ONE construct. "
       "Use only the anchors given. Return JSON only.")


def persona_desc(pid):
    p = load_persona_data(pid, ROOT); ks = ["persona_id", "defining_rule", "median_age", "current_status", "operating_system", "brand"]
    return json.dumps({k: p.get(k) for k in ks if k in p}, ensure_ascii=False)[:900]


cache_f = OUT / "reference_ratings_llm_judge.jsonl"; cache = {}
if cache_f.exists():
    for line in cache_f.read_text(encoding="utf-8").splitlines():
        j = json.loads(line); cache[(j["set"], j["persona_id"], j["construct"])] = j
ollama_utils.SETTINGS["temperature"] = 0.0
for set_name, d, T in [("improved", IMP, Ti), ("baseline", BASE, Tb)]:
    concept = json.loads((d / "product_concept.json").read_text(encoding="utf-8"))
    for p, c in keys:
        if (set_name, p, c) in cache:
            continue
        an = get_construct_anchors(c); anchors = "\n".join(f"{k}: {t}" for k, t in zip(an.ratings, an.statements))
        user = f"""PRODUCT: {concept['name']} ({concept.get('category','')}) - {concept['description'][:700]}
PERSONA PROFILE: {persona_desc(p)}
CONSTRUCT: {c} ({an.construct_name})
APPROVED LIKERT ANCHORS (1-5):
{anchors}

RESPONSE:
\"\"\"{T[(p, c)]}\"\"\"

Tasks:
1. reference_rating: integer 1-5 = which anchor best matches the attitude expressed in the response for this construct.
2. construct_aligned: true if the response actually addresses this construct ({an.construct_name}).
3. product_relevant: true if it is clearly about this product ({concept.get('category','')}), not a different product type.
4. persona_consistent: true if it fits the persona profile.
5. naturalness: integer 1-5 (5 = natural, realistic consumer voice).
6. issues: short list of problems (empty if none).
Return JSON: {{"reference_rating":int,"construct_aligned":bool,"product_relevant":bool,"persona_consistent":bool,"naturalness":int,"issues":[str]}}"""
        rec, err = ollama_utils.generate(JUDGE, {"system": SYS, "user": user, "json": True, "timeout_s": 300.0, "max_output_tokens": 300}, timeout=300.0)
        if err:
            raise RuntimeError(f"judge failed {set_name} {p} {c}: {err}")
        j = json.loads(re.search(r"\{.*\}", rec["response_text"], re.S).group(0))
        j.update({"set": set_name, "persona_id": p, "construct": c, "judge_model": JUDGE, "raw": rec["response_text"]})
        cache[(set_name, p, c)] = j
        with cache_f.open("a", encoding="utf-8") as f:
            f.write(json.dumps(j, ensure_ascii=False) + "\n")
S["ref"] = [int(cache[(s, p, c)]["reference_rating"]) for s, p, c in zip(S.set, S.persona_id, S.construct)]
S.to_csv(OUT / "ssr_scores_with_reference.csv", index=False)


# ---------- metrics
def metrics(df, pred_col="imp_rounded", cont_col="imp_E"):
    y, yp, e = df.ref.values, df[pred_col].values, df[cont_col].values
    m = {"n": int(len(df)), "exact_acc": float((yp == y).mean()), "within_1_acc": float((abs(yp - y) <= 1).mean()),
         "MAE_continuous": float(abs(e - y).mean()), "RMSE_continuous": float(np.sqrt(((e - y) ** 2).mean())),
         "MAE_discrete": float(abs(yp - y).mean()), "RMSE_discrete": float(np.sqrt(((yp - y) ** 2).mean())),
         "bias_E_minus_ref": float((e - y).mean())}
    for nm, fn in [("spearman", stats.spearmanr), ("pearson", stats.pearsonr)]:
        if np.ptp(y) > 0 and np.ptp(e) > 0:
            r = fn(e, y); m[nm] = {"r": float(r[0]), "p": float(r[1])}
        else:
            m[nm] = {"r": None, "p": None, "note": "undefined: constant ratings"}
    labs = [k for k in L if k in set(y) | set(yp)]
    m["qw_kappa"] = float(cohen_kappa_score(y, yp, weights="quadratic", labels=L)) if len(set(y) | set(yp)) > 1 else None
    m["lw_kappa"] = float(cohen_kappa_score(y, yp, weights="linear", labels=L)) if len(set(y) | set(yp)) > 1 else None
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=L, zero_division=0)
    m["per_rating"] = {k: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support_ref": int(sup[i]), "n_pred": int((yp == k).sum())} for i, k in enumerate(L)}
    for avg in ["macro", "weighted"]:
        a = precision_recall_fscore_support(y, yp, labels=labs, average=avg, zero_division=0)
        m[avg] = {"precision": float(a[0]), "recall": float(a[1]), "f1": float(a[2]), "labels_used": labs}
    m["confusion_rows_ref_cols_pred"] = confusion_matrix(y, yp, labels=L).tolist()
    m["dist_ref"] = {k: int((y == k).sum()) for k in L}; m["dist_pred"] = {k: int((yp == k).sum()) for k in L}
    m["E_mean"], m["E_sd"], m["ref_mean"], m["ref_sd"] = float(e.mean()), float(e.std(ddof=1)), float(y.mean()), float(y.std(ddof=1))
    m["majority_baseline_exact"] = float((y == stats.mode(y, keepdims=False).mode).mean())
    return m


res = {"label": "Phase 6 SSR vs LLM-JUDGE REFERENCE ratings (NOT human ground truth). n = 24 per set.",
       "reference_source": f"{JUDGE}, temperature 0, blind to SSR, prompt identical to evaluation/quick_ssr_validation; same model family as the generator",
       "ssr_config": {"primary": "Improved SSR k=3 T=0.75 (rounded E[r]); argmax and canonical also reported", "embedding_model": EMB,
                      "anchors": "docs/survey_instrument/smartphone_construct_anchor_statements.csv (unchanged)"},
       "baseline_canonical_reproduction_max_abs_diff": float(repro), "sets": {}}
for s in ["baseline", "improved"]:
    d = S[S.set == s]
    res["sets"][s] = {"improved_ssr_rounded": metrics(d), "improved_ssr_argmax": metrics(d, "imp_argmax"),
                      "canonical_ssr_rounded": metrics(d.assign(cr=np.clip(np.round(d.canon_E), 1, 5).astype(int)), "cr", "canon_E"),
                      "per_construct": {c: {"ref_mean": float(g.ref.mean()), "E_mean": float(g.imp_E.mean()), "MAE": float(abs(g.imp_E - g.ref).mean()),
                                            "exact": float((g.imp_rounded == g.ref).mean()), "refs": g.ref.tolist(), "preds": g.imp_rounded.tolist()} for c, g in d.groupby("construct")},
                      "per_persona": {p: {"ref_mean": float(g.ref.mean()), "E_mean": float(g.imp_E.mean()), "MAE": float(abs(g.imp_E - g.ref).mean()),
                                          "exact": float((g.imp_rounded == g.ref).mean())} for p, g in d.groupby("persona_id")},
                      "uncertainty": {"mean_pmf_sd": float(d.imp_pmf_sd.mean()), "mean_entropy_bits": float(d.imp_entropy_bits.mean()),
                                      "mean_max_prob": float(d.imp_max_prob.mean()), "mean_cos_spread": float(d.cos_spread.mean()),
                                      "zero_prob_cells_share": float((d[[f"imp_p{k}" for k in L]].values == 0).mean())},
                      "persona_E_kruskal_p": float(stats.kruskal(*[g.imp_E.values for _, g in d.groupby("persona_id")])[1])}
# paired comparison (same persona x construct cells; different texts and different reference ratings per set)
b, i = S[S.set == "baseline"].set_index(["persona_id", "construct"]), S[S.set == "improved"].set_index(["persona_id", "construct"])
ab = (b.imp_E - b.ref).abs()
ai = (i.imp_E - i.ref).abs().reindex(ab.index)
w = stats.wilcoxon(ai, ab) if (ai - ab).abs().sum() > 0 else None
res["paired_baseline_vs_improved"] = {"mean_abs_err_baseline": float(ab.mean()), "mean_abs_err_improved": float(ai.mean()),
                                      "wilcoxon_p": float(w.pvalue) if w else None,
                                      "mcnemar_exact_counts": {"both_correct": int(((b.imp_rounded == b.ref) & (i.imp_rounded == i.ref).reindex(b.index)).sum()),
                                                               "only_baseline": int(((b.imp_rounded == b.ref) & ~(i.imp_rounded == i.ref).reindex(b.index)).sum()),
                                                               "only_improved": int((~(b.imp_rounded == b.ref) & (i.imp_rounded == i.ref).reindex(b.index)).sum())},
                                      "note": "references differ per set (different texts were judged); n=24; low power"}
(OUT / "phase6_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
for s in ["baseline", "improved"]:
    pd.DataFrame(res["sets"][s]["improved_ssr_rounded"]["confusion_rows_ref_cols_pred"], index=[f"ref_{k}" for k in L], columns=[f"ssr_{k}" for k in L]).to_csv(OUT / f"confusion_{s}_improved_ssr.csv")
print("repro", repro); print(json.dumps({s: {k: v for k, v in res["sets"][s]["improved_ssr_rounded"].items() if k not in ("per_rating",)} for s in res["sets"]}, indent=1, default=float))
print(json.dumps(res["paired_baseline_vs_improved"], indent=1))
