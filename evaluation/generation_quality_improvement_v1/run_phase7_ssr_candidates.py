"""Phase 7: experimental SSR decision rules / calibrators. No LLM calls, no regeneration, canonical SSR untouched.
TEST (held out, never used for fitting): 24 improved Nova responses (phase6_ssr/ssr_scores_with_reference.csv, set=improved).
CALIBRATION (disjoint texts, same SSR config + same blind-judge protocol): 24 baseline Nova responses (phase6) +
24 quick-validation responses (evaluation/quick_ssr_validation/validation_results.csv). Candidates are fixed a priori.
Output: phase7_ssr_candidates/ (metrics JSON + comparison CSV).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]; HERE = Path(__file__).resolve().parent
OUT = HERE / "phase7_ssr_candidates"; OUT.mkdir(exist_ok=True)
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from improved_ssr import improved_similarities_to_pmf  # noqa: E402
import ssr_experimental_decisions as X  # noqa: E402

SEED, NBOOT, L = 20261104, 2000, [1, 2, 3, 4, 5]
S = pd.read_csv(HERE / "phase6_ssr/ssr_scores_with_reference.csv")
test = S[S.set == "improved"].reset_index(drop=True)
cal_b = S[S.set == "baseline"][["construct", "imp_E", "ref"]].assign(source="nova_baseline")
q = pd.read_csv(ROOT / "evaluation/quick_ssr_validation/validation_results.csv")
cal_q = pd.DataFrame({"construct": q.construct, "imp_E": q.improved_E, "ref": q.judge_reference_rating, "source": "quick_validation_glasses"})
cal = pd.concat([cal_b, cal_q], ignore_index=True)
assert not set(test.response_text) & set(S[S.set == "baseline"].response_text)   # disjoint texts

y = test.ref.values.astype(int)
P_imp = test[[f"imp_p{k}" for k in L]].values; E_imp = test.imp_E.values
cos = test[[f"cos_{k}" for k in L]].values
P_k5 = np.array([improved_similarities_to_pmf(c, k=5, temperature=0.75) for c in cos])   # top-k ablation (a priori)
fits = {"bias": X.BiasShift.fit(cal.imp_E, cal.ref), "construct": X.ConstructShift.fit(cal.imp_E, cal.ref, cal.construct),
        "varmatch": X.VarianceMatch.fit(cal.imp_E, cal.ref), "cutpoints": X.OrdinalCutpoints.fit(cal.imp_E, cal.ref)}
maj = int(stats.mode(cal.ref.values, keepdims=False).mode)
C = {  # name: (discrete prediction, continuous score used for correlation / E-MAE)
    f"Majority baseline (cal mode = {maj})": (np.full(len(y), maj), np.full(len(y), float(maj))),
    "Canonical SSR, rounded E": (X.round_expected(test.canon_E), test.canon_E.values),
    "Improved SSR k3 T.75, rounded E (current)": (X.round_expected(E_imp), E_imp),
    "Improved SSR, PMF argmax": (X.pmf_argmax(P_imp), E_imp),
    "Improved SSR, PMF median": (X.pmf_median(P_imp), E_imp),
    "Nearest anchor (k=1)": (X.nearest_anchor(cos), X.nearest_anchor(cos).astype(float)),
    "Improved SSR k=5 (no truncation) T.75, rounded E": (X.round_expected(X.expected(P_k5)), X.expected(P_k5)),
    "Cal: global bias shift": (X.round_expected(fits["bias"].transform(E_imp)), fits["bias"].transform(E_imp)),
    "Cal: construct shift (shrunk)": (X.round_expected(fits["construct"].transform(E_imp, test.construct)), fits["construct"].transform(E_imp, test.construct)),
    "Cal: variance matching": (X.round_expected(fits["varmatch"].transform(E_imp)), fits["varmatch"].transform(E_imp)),
    "Cal: ordinal cutpoints (QWK on cal)": (fits["cutpoints"].predict(E_imp), E_imp),
}
CURRENT = "Improved SSR k3 T.75, rounded E (current)"


def full(yp, e, yy=y, df=test):
    yp = np.asarray(yp, int); e = np.asarray(e, float)
    m = {"exact": float((yp == yy).mean()), "within1": float((abs(yp - yy) <= 1).mean()),
         "MAE": float(abs(yp - yy).mean()), "RMSE": float(np.sqrt(((yp - yy) ** 2).mean())),
         "E_MAE": float(abs(e - yy).mean()), "E_RMSE": float(np.sqrt(((e - yy) ** 2).mean()))}
    lab = sorted(set(yy) | set(yp))
    m["qwk"] = float(cohen_kappa_score(yy, yp, weights="quadratic", labels=L)) if len(lab) > 1 else 0.0
    m["lwk"] = float(cohen_kappa_score(yy, yp, weights="linear", labels=L)) if len(lab) > 1 else 0.0
    for nm, f in [("spearman", stats.spearmanr), ("pearson", stats.pearsonr)]:
        r = f(e, yy) if np.ptp(e) > 0 else (np.nan, np.nan); m[nm], m[nm + "_p"] = float(r[0]), float(r[1])
    pr, rc, f1, sup = precision_recall_fscore_support(yy, yp, labels=L, zero_division=0)
    m["per_class"] = {k: {"P": float(pr[i]), "R": float(rc[i]), "F1": float(f1[i]), "support": int(sup[i]), "n_pred": int((yp == k).sum())} for i, k in enumerate(L)}
    present = [k for k in L if (yy == k).any()]   # classes with reference support (3,4,5)
    for avg in ("macro", "weighted"):
        a = precision_recall_fscore_support(yy, yp, labels=present, average=avg, zero_division=0)
        m[f"{avg}_P"], m[f"{avg}_R"], m[f"{avg}_F1"] = map(float, a[:3])
    m["confusion"] = confusion_matrix(yy, yp, labels=L).tolist()
    m["dist_pred"] = {k: int((yp == k).sum()) for k in L}
    m["five_star_FP"] = int(((yp == 5) & (yy != 5)).sum()); m["five_star_FN"] = int(((yp != 5) & (yy == 5)).sum())
    m["per_construct_exact"] = {c: float((yp[df.construct == c] == yy[df.construct == c]).mean()) for c in sorted(df.construct.unique())}
    m["per_construct_MAE"] = {c: float(abs(yp[df.construct == c] - yy[df.construct == c]).mean()) for c in sorted(df.construct.unique())}
    m["per_persona_exact"] = {p[:2]: float((yp[df.persona_id == p] == yy[df.persona_id == p]).mean()) for p in sorted(df.persona_id.unique())}
    return m


res = {cand: full(*v) for cand, v in C.items()}
# reliability vs current: paired bootstrap (resample the 24 cases) + exact McNemar on correctness
rng = np.random.default_rng(SEED); idx = rng.integers(0, len(y), (NBOOT, len(y)))
cur_p, cur_e = C[CURRENT]


def stat(yp, e, ii):
    yy = y[ii]; yp = np.asarray(yp)[ii]
    lab = set(yy) | set(yp)
    present = [k for k in L if (yy == k).any()]
    return ((yp == yy).mean(), cohen_kappa_score(yy, yp, weights="quadratic", labels=L) if len(lab) > 1 else 0.0,
            precision_recall_fscore_support(yy, yp, labels=present, average="macro", zero_division=0)[2],
            abs(yp - yy).mean())


base_bs = np.array([stat(cur_p, cur_e, ii) for ii in idx])
for cand, (yp, e) in C.items():
    if cand == CURRENT:
        continue
    d = np.array([stat(yp, e, ii) for ii in idx]) - base_bs
    a, b = int(((np.asarray(yp) == y) & (cur_p != y)).sum()), int(((np.asarray(yp) != y) & (cur_p == y)).sum())
    res[cand]["vs_current"] = {"exact_diff_ci95": np.percentile(d[:, 0], [2.5, 97.5]).tolist(), "qwk_diff_ci95": np.percentile(d[:, 1], [2.5, 97.5]).tolist(),
                               "macroF1_diff_ci95": np.percentile(d[:, 2], [2.5, 97.5]).tolist(), "MAE_diff_ci95": np.percentile(d[:, 3], [2.5, 97.5]).tolist(),
                               "mcnemar_only_candidate_right": a, "only_current_right": b,
                               "mcnemar_exact_p": float(stats.binomtest(a, a + b).pvalue) if a + b else 1.0}
# calibration-set sanity: what the fitted calibrators do on the calibration data itself
cal_eval = {"n_cal": len(cal), "cal_sources": cal.source.value_counts().to_dict(), "cal_ref_dist": cal.ref.value_counts().sort_index().to_dict(),
            "cal_spearman_E_vs_ref": float(stats.spearmanr(cal.imp_E, cal.ref)[0]),
            "fitted": {"bias_shift": fits["bias"].shift, "construct_shifts": dict(fits["construct"].shifts), "construct_prior_m": fits["construct"].m,
                       "variance_match": vars(fits["varmatch"]), "cutpoints": fits["cutpoints"].thresholds}}
out = {"label": "Phase 7 EXPLORATORY candidates; test n=24 (LLM-judge reference, not human ground truth); calibration n=48 disjoint",
       "seed": SEED, "n_boot": NBOOT, "test_ref_dist": {k: int((y == k).sum()) for k in L}, "calibration": cal_eval, "candidates": res}
(OUT / "phase7_metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
cols = ["exact", "within1", "macro_P", "macro_R", "macro_F1", "weighted_P", "weighted_R", "weighted_F1", "spearman", "spearman_p", "pearson", "qwk", "lwk", "MAE", "RMSE", "E_MAE", "E_RMSE", "five_star_FP", "five_star_FN"]
tab = pd.DataFrame([{"method": k, **{c: v[c] for c in cols}, "R5": v["per_class"][5]["R"], "F1_5": v["per_class"][5]["F1"], "R3": v["per_class"][3]["R"],
                     "PE_exact": v["per_construct_exact"]["PE"], "pred_dist": v["dist_pred"]} for k, v in res.items()])
tab.to_csv(OUT / "phase7_comparison.csv", index=False)
pd.set_option("display.width", 250); print(tab.drop(columns=["pred_dist"]).round(3).to_string()); print(json.dumps(cal_eval, indent=1, default=float))
for k, v in res.items():
    if "vs_current" in v: print(k, {a: (np.round(b, 3).tolist() if isinstance(b, list) else b) for a, b in v["vs_current"].items()})
