"""Phase 15: synthetic BI ratings vs human Q21. No generation, no LLM/embedding calls, nothing refit.
Synthetic E[r]: stored Phase 6 Improved SSR (k=3, T=0.75, BI anchors, nomic) for the Phase 4 v3 responses.
Calibrated (EXPERIMENTAL): Phase 14 Method C thresholds = mean of the 20 train-split fits, loaded from phase14_metrics.json.
"""
from __future__ import annotations
import json, re, sys, itertools, collections
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ssr_experimental_decisions as X  # noqa: E402

L = np.arange(1, 6); RNG = np.random.default_rng(20261105)
P14 = json.loads((ROOT / "evaluation/phase14_human_calibration/phase14_metrics.json").read_text(encoding="utf-8"))
THR = tuple(P14["thresholds_mean_sd"]["C_quantile_matched_thresholds (fit on train)"][0])
S6 = pd.read_csv(ROOT / "evaluation/generation_quality_improvement_v1/phase6_ssr/ssr_scores_with_reference.csv")
syn = S6[S6.set == "improved"].reset_index(drop=True)
gen_dir = ROOT / "data/product_evaluation/nova_ai_smartphone_v3_improved_20261104_211250"
gen = {(r["persona_id"], c): t for r in json.loads((gen_dir / "persona_responses.json").read_text(encoding="utf-8")) for c, t in r["construct_responses"].items()}
assert all(gen[(p, c)] == t for p, c, t in zip(syn.persona_id, syn.construct, syn.response_text)), "Phase 6 texts must equal Phase 4 output"
syn["canonical"] = np.clip(np.floor(syn.imp_E + 0.5), 1, 5).astype(int)
syn["calibrated"] = X.OrdinalCutpoints(THR).predict(syn.imp_E.values)
bi = syn[syn.construct == "BI"].reset_index(drop=True)

# human reference: all eligible respondents (Q21), personas by the same rules as Phases 12-13
d = pd.read_excel(ROOT / "data/Capstone_Reviews(Responses).xlsx"); d.columns = [f"c{i:02d}" for i in range(d.shape[1])]
age = pd.to_numeric(d.c03, errors="coerce")
Hh = d[(d.c01.str.startswith("Yes")) & (d.c02 == "Yes") & (d.c04 == "Yes") & (age >= 18)].copy()
Hh["q21"] = Hh.c20.str[0].astype(int)
tier = {"More than ₹60,000": "P1", "₹35,001–₹60,000": "P1", "₹20,001–₹35,000": "mid", "₹10,000–₹20,000": "mid"}
Hh["persona"] = Hh.c12.map(tier).fillna("P4")
Hh.loc[Hh.persona == "mid", "persona"] = np.where(Hh.loc[Hh.persona == "mid", "c17"].isin(["6–8 hours", "More than 8 hours"]), "P2", "P3")
hum = Hh.q21.values


def dist(v):
    v = np.asarray(v, int); return np.bincount(v, minlength=6)[1:] / max(1, len(v))


def compare(v, ref=hum, n_null=5000):
    v = np.asarray(v, int); p, q = dist(v), dist(ref)
    w1 = float(stats.wasserstein_distance(v, ref)); js = float(jensenshannon(p, q, base=2) ** 2)
    # reference: how close are random samples of the SAME size drawn from the human ratings themselves?
    null_w = np.array([stats.wasserstein_distance(RNG.choice(ref, len(v)), ref) for _ in range(n_null)])
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else None,
            "proportions_1to5": p.round(4).tolist(), "neg_1_2": float(p[:2].sum()), "neu_3": float(p[2]), "pos_4_5": float(p[3:].sum()),
            "wasserstein_W1_vs_human": w1, "JSD_vs_human_base2": js,
            "share_of_random_human_samples_with_W1_at_least_this": float((null_w >= w1 - 1e-12).mean()),
            "null_W1_median_same_n": float(np.median(null_w))}


res = {"label": "Phase 15: synthetic BI vs human Q21. Canonical = original Improved SSR rounded E (k=3,T=0.75). "
                "Calibrated = EXPERIMENTAL Phase 14 Method C thresholds (not refit).",
       "calibration_thresholds_loaded": THR, "human_n": int(len(hum)),
       "human": {"n": int(len(hum)), "mean": float(hum.mean()), "sd": float(hum.std(ddof=1)), "proportions_1to5": dist(hum).round(4).tolist(),
                 "neg_1_2": float(dist(hum)[:2].sum()), "neu_3": float(dist(hum)[2]), "pos_4_5": float(dist(hum)[3:].sum())},
       "synthetic_BI_canonical": compare(bi.canonical), "synthetic_BI_calibrated": compare(bi.calibrated),
       "synthetic_BI_E": {"values": bi.imp_E.round(3).tolist(), "mean": float(bi.imp_E.mean()), "sd": float(bi.imp_E.std(ddof=1))},
       "SECONDARY_all24_not_comparable_to_Q21": {"canonical": compare(syn.canonical), "calibrated_BI_thresholds_misapplied": compare(syn.calibrated),
                                                 "note": "non-BI constructs use other anchor sets; Q21 is not their human counterpart"}}
# persona level
per = []
for p in ["P1", "P2", "P3", "P4"]:
    hv = Hh[Hh.persona == p].q21.values; s = bi[bi.persona_id.str.startswith(p)]
    per.append({"persona": p, "human_n": int(len(hv)), "human_mean": float(hv.mean()), "human_sd": float(hv.std(ddof=1)), "human_props": dist(hv).round(3).tolist(),
                "syn_BI_text": s.response_text.iloc[0], "syn_E": float(s.imp_E.iloc[0]), "syn_canonical": int(s.canonical.iloc[0]),
                "syn_calibrated": int(s.calibrated.iloc[0]), "judge_ref_phase6": int(s.ref.iloc[0]),
                "human_pct_at_or_below_syn_calibrated": float((hv <= s.calibrated.iloc[0]).mean()),
                "syn_all6_E_mean": float(syn[syn.persona_id.str.startswith(p)].imp_E.mean())})
res["persona"] = per
hm = [x["human_mean"] for x in per]; se = [x["syn_E"] for x in per]
res["persona_rank_agreement"] = {"human_mean_order": [x["persona"] for x in sorted(per, key=lambda r: -r["human_mean"])],
                                 "synthetic_E_order": [x["persona"] for x in sorted(per, key=lambda r: -r["syn_E"])],
                                 "spearman_n4": float(stats.spearmanr(hm, se)[0]), "human_kruskal_p": float(stats.kruskal(*[Hh[Hh.persona == p].q21 for p in ["P1", "P2", "P3", "P4"]])[1]),
                                 "human_P1_minus_P2": hm[0] - hm[1], "synthetic_P1_minus_P2_E": se[0] - se[1]}
# text quality / diversity (existing texts only)
W = lambda t: re.findall(r"[a-z']+", t.lower())
def txt(df):
    w = [W(t) for t in df.response_text]; g5 = collections.Counter(tuple(x[i:i + 5]) for x in w for i in range(len(x) - 4))
    return {"n": len(w), "words_mean": float(np.mean([len(x) for x in w])), "words_sd": float(np.std([len(x) for x in w], ddof=1)),
            "starts_this_phone": int(sum(" ".join(x[:2]) == "this phone" for x in w)), "distinct_3w_openings": len({" ".join(x[:3]) for x in w}),
            "repeated_5grams": int(sum(c > 1 for c in g5.values())), "openings": [" ".join(x[:6]) for x in w]}
human_q26 = Hh.c25.fillna("").astype(str); hw = human_q26.map(lambda t: len(W(t)))
res["text"] = {"synthetic_BI": txt(bi), "synthetic_all24": txt(syn),
               "human_Q26_words_mean_median": [float(hw[hw > 0].mean()), float(hw[hw > 0].median())],
               "human_Q26_starts_yes_no": [int(human_q26.str.lower().str.match(r"^(yes|yeah|ya)").sum()), int(human_q26.str.lower().str.match(r"^(no|nope|not)").sum())],
               "phase10_reference": "persona differentiation 0.157 (cross-persona same-construct mpnet cosine 0.843); see Phase 10"}
syn[["persona_id", "construct", "response_text", "imp_E", "canonical", "calibrated", "ref"]].to_csv(OUT / "phase15_synthetic_ratings.csv", index=False)
(OUT / "phase15_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "persona"}, indent=1, ensure_ascii=False, default=float))
for x in per: print({k: v for k, v in x.items() if k != "syn_BI_text"}, "|", x["syn_BI_text"][:110])
