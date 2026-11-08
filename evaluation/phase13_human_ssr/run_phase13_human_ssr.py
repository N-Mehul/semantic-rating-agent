"""Phase 13: SSR vs real human Q21 (rebuy intention) using each respondent's own Q26 text.
SSR code imported unchanged (canonical_ssr + improved_ssr k=3 T=0.75), BI anchors, nomic-embed-text (local embedding only).
Reads data/Capstone_Reviews(Responses).xlsx read-only. Writes only into evaluation/phase13_human_ssr/.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from canonical_ssr import similarities_to_pmf, cosine_similarities, expected_rating, PAPER_EPSILON, PAPER_TEMPERATURE  # noqa: E402
from improved_ssr import improved_similarities_to_pmf, DEFAULT_IMPROVED_K, DEFAULT_IMPROVED_TEMPERATURE  # noqa: E402
from survey_ssr_bridge import get_construct_anchors  # noqa: E402
from scripts.ollama_embedding_adapter import get_ollama_embedding_adapter  # noqa: E402
import ssr_hybrid_experiments as H  # metrics helper (read-only reuse)  # noqa: E402

assert (DEFAULT_IMPROVED_K, DEFAULT_IMPROVED_TEMPERATURE, PAPER_EPSILON, PAPER_TEMPERATURE) == (3, 0.75, 0.0, 1.0)
p = improved_similarities_to_pmf(np.array([.1, .2, .3, .4, .5]), k=3, temperature=.75); w = np.array([.2, .3, .4]) ** (1 / .75)
assert np.allclose(p, [0, 0, *(w / w.sum())]) and np.allclose(similarities_to_pmf(np.array([.1, .2, .3, .4, .5])), [0, .1, .2, .3, .4])

d = pd.read_excel(ROOT / "data/Capstone_Reviews(Responses).xlsx"); d.columns = [f"c{i:02d}" for i in range(d.shape[1])]
age = pd.to_numeric(d.c03, errors="coerce")
E = d[(d.c01.str.startswith("Yes")) & (d.c02 == "Yes") & (d.c04 == "Yes") & (age >= 18)].copy().reset_index(names="row")
E["q21"] = E.c20.str[0].astype(int)
E["text"] = E.c25.fillna("").astype(str).str.strip()


def category(t):
    w = re.findall(r"[A-Za-z]+", t)
    if not w or (len(w) == 1 and len(w[0]) >= 5 and not re.search(r"[aeiou]", w[0], re.I)) or re.search(r"(h[ds]){2,}|(dh){2,}", t.lower()):
        return "missing_or_unusable"
    return "one_word" if len(w) == 1 else "short_2" if len(w) == 2 else "meaningful_3plus"


E["category"] = E.text.map(category)
price = {"More than ₹60,000": "P1 premium", "₹35,001–₹60,000": "P1 premium", "₹20,001–₹35,000": "mid", "₹10,000–₹20,000": "mid"}
E["persona"] = E.c12.map(price).fillna("P4 entry/undisclosed")
E.loc[E.persona == "mid", "persona"] = np.where(E.loc[E.persona == "mid", "c17"].isin(["6–8 hours", "More than 8 hours"]), "P2 mid heavy", "P3 mid moderate")
E["os"] = np.where(E.c13 == "Android", "Android", "iOS")

S = E[E.category != "missing_or_unusable"].reset_index(drop=True)
ad = get_ollama_embedding_adapter("nomic-embed-text")
A = np.asarray(ad.embed_texts(list(get_construct_anchors("BI").statements)))
R = np.asarray(ad.embed_texts(list(S.text)))
cos = np.array([cosine_similarities(r, A) for r in R])
Pi = np.array([improved_similarities_to_pmf(c, k=3, temperature=0.75) for c in cos])
Pc = np.array([similarities_to_pmf(c, epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE) for c in cos])
S["imp_E"] = [expected_rating(x) for x in Pi]; S["imp_pred"] = H.round_half_up(S.imp_E)
S["can_E"] = [expected_rating(x) for x in Pc]; S["can_pred"] = H.round_half_up(S.can_E)
for k in range(5):
    S[f"cos_{k+1}"] = cos[:, k]; S[f"imp_p{k+1}"] = Pi[:, k]


def M(df, pred="imp_pred", cont="imp_E"):
    m = H.metrics(df.q21.values, df[pred].values, df[cont].values)
    m["n"] = int(len(df)); m["mean_signed_err_E"] = float((df[cont] - df.q21).mean())
    m["under"] = float((df[pred] < df.q21).mean()); m["over"] = float((df[pred] > df.q21).mean())
    m["majority_exact"] = float((df.q21 == df.q21.mode()[0]).mean())
    return m


res = {"label": "Phase 13: SSR vs HUMAN Q21 (own Q26 text). BI anchors (proxy: Q21 = rebuy SAME BRAND; anchors = continue using phone).",
       "n_eligible": int(len(E)), "category_counts": E.category.value_counts().to_dict(),
       "human_q21_dist_all_eligible": {int(k): int(v) for k, v in E.q21.value_counts().sort_index().items()},
       "overall_improved": M(S), "overall_canonical": M(S, "can_pred", "can_E"),
       "by_category": {c: M(g) for c, g in S.groupby("category")},
       "by_category_canonical": {c: M(g, "can_pred", "can_E") for c, g in S.groupby("category")},
       "by_persona": {p: M(g) for p, g in S.groupby("persona")}, "by_os": {o: M(g) for o, g in S.groupby("os")},
       "pred_dist_improved": {int(k): int((S.imp_pred == k).sum()) for k in range(1, 6)},
       "E_mean_sd": [float(S.imp_E.mean()), float(S.imp_E.std())], "q21_mean_sd_scored": [float(S.q21.mean()), float(S.q21.std())],
       "persona_q21_vs_E_means": S.groupby("persona")[["q21", "imp_E"]].mean().round(3).to_dict("index"),
       "os_q21_vs_E_means": S.groupby("os")[["q21", "imp_E"]].mean().round(3).to_dict("index")}
for nm, g in [("persona", "persona"), ("os", "os")]:
    res[f"kruskal_E_by_{nm}"] = float(stats.kruskal(*[x.imp_E.values for _, x in S.groupby(g)])[1])
    res[f"kruskal_q21_by_{nm}"] = float(stats.kruskal(*[x.q21.values for _, x in S.groupby(g)])[1])
S.drop(columns=[c for c in S.columns if re.fullmatch(r"c\d\d", c) and c not in ("c12", "c13", "c17")]).to_csv(OUT / "phase13_scored_human_q26.csv", index=False)
(OUT / "phase13_metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if not k.startswith("by_")}, indent=1, default=float)[:6000])
