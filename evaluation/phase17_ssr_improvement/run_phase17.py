"""Phase 17: experimental E[r]/PMF -> Likert conversion methods. Frozen SSR outputs are the ONLY inputs
(Phase 13 human: Improved SSR k=3 T=0.75, BI anchors, nomic; Phase 16 synthetic: same). No LLM / embedding calls.

Protocol (declared before running):
  TRAIN/VALIDATION: human respondents only, repeated stratified 5-fold CV (4 repeats x 5 folds = 20 held-out folds),
                    respondent-level, seeds 20261106+r. Methods have NO tuned hyperparameters (all fixed a priori).
  SELECTION RULE:   highest mean held-out human QWK among methods whose mean 1-star AND 5-star recall > 0 and whose
                    QWK beats canonical in >= 15/20 folds; tie-break lower MAE. If none qualifies -> keep canonical.
  FINAL TEST:       methods refit on ALL human data, applied ONCE to the 200 Phase 16 synthetic responses;
                    intended stances used ONLY as evaluation labels (never for fitting).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats, optimize
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ssr_hybrid_experiments as Hx   # metrics helper (read-only)
L = np.arange(1, 6); SEED = 20261106

Hd = pd.read_csv(ROOT / "evaluation/phase13_human_ssr/phase13_scored_human_q26.csv")
Sd = pd.read_csv(ROOT / "evaluation/phase16_bi_only_generation/phase16_scored_responses.csv")
feat = lambda d, pre: (d["imp_E" if "imp_E" in d else "E"].values, d[[f"{pre}{k}" for k in L]].values, d[[f"cos_{k}" for k in L]].values)
EH, PH, CH = feat(Hd, "imp_p"); yH = Hd.q21.values.astype(int)
ES, PS, CS = feat(Sd, "p"); yS = Sd.intended_stance.values.astype(int)
assert np.allclose(PH @ L, EH, atol=1e-6) and np.allclose(PS @ L, ES, atol=1e-6)
rnd = lambda x: np.clip(np.floor(np.asarray(x, float) + 0.5), 1, 5).astype(int)
cut = lambda E, t: 1 + (np.asarray(E)[:, None] >= np.asarray(t)[None, :]).sum(1)


# ------------------------------------------------------------------ methods: fit(E,P,C,y) -> predict(E,P,C) -> ratings
def m_canonical(E, P, C, y):
    return lambda E2, P2, C2: rnd(E2)


def m_quantile_thresholds(E, P, C, y):          # Phase 14 Method C on all lengths
    cum = np.cumsum(np.bincount(y, minlength=6)[1:]) / len(y); t = np.quantile(E, cum[:4])
    return lambda E2, P2, C2: cut(E2, t)


def m_variance_match(E, P, C, y):               # affine moment matching (same maths as improved_ssr.LinearVarianceRescaler)
    a, b, mu, sd = E.mean(), E.std(ddof=1), y.mean(), y.std(ddof=1)
    return lambda E2, P2, C2: rnd(mu + (E2 - a) * sd / b)


def m_isotonic(E, P, C, y):                     # order-preserving regression of y on E, then round
    iso = IsotonicRegression(y_min=1, y_max=5, out_of_bounds="clip").fit(E, y)
    return lambda E2, P2, C2: rnd(iso.predict(E2))


def _ordlogit_fit(X, y, l2=1.0):
    """Proportional-odds (cumulative logit) model, L2 on slopes; returns predict_proba."""
    X = (X - (mu := X.mean(0))) / (sd := X.std(0) + 1e-9); k = X.shape[1]
    def unpack(th): return th[:k], np.cumsum(np.concatenate([[th[k]], np.exp(th[k + 1:])]))
    def nll(th):
        b, c = unpack(th); eta = X @ b
        F = 1 / (1 + np.exp(-(c[None, :] - eta[:, None]))); F = np.hstack([np.zeros((len(X), 1)), F, np.ones((len(X), 1))])
        p = np.clip(F[np.arange(len(X)), y] - F[np.arange(len(X)), y - 1], 1e-12, 1)
        return -np.log(p).sum() + l2 * (b ** 2).sum()
    th0 = np.concatenate([np.zeros(k), [-2.0], np.log(np.ones(3))])
    th = optimize.minimize(nll, th0, method="L-BFGS-B").x; b, c = unpack(th)
    def proba(X2):
        X2 = (X2 - mu) / sd; eta = X2 @ b
        F = 1 / (1 + np.exp(-(c[None, :] - eta[:, None]))); F = np.hstack([np.zeros((len(X2), 1)), F, np.ones((len(X2), 1))])
        return np.diff(F, axis=1)
    return proba


def m_ordlogit_E(E, P, C, y):                   # ordinal regression on E only; predict = PMF median of fitted probs
    pr = _ordlogit_fit(E[:, None], y)
    return lambda E2, P2, C2: (np.cumsum(pr(E2[:, None]), 1) >= 0.5).argmax(1) + 1


def m_ordlogit_feats(E, P, C, y):               # ordinal regression on E + centred anchor cosines (PMF/similarity features)
    f = lambda E_, C_: np.column_stack([E_, C_ - C_.mean(1, keepdims=True)])
    pr = _ordlogit_fit(f(E, C), y)
    return lambda E2, P2, C2: (np.cumsum(pr(f(E2, C2)), 1) >= 0.5).argmax(1) + 1


def m_prior_corrected_pmf(E, P, C, y):          # probability-based: reweight SSR PMF by class weights w so that the mean
    target = np.bincount(y, minlength=6)[1:] / len(y)   # reweighted train PMF matches the train label prior (Saerens-style)
    w = np.ones(5)
    for _ in range(200):
        Q = P * w; Q /= Q.sum(1, keepdims=True); w *= target / np.clip(Q.mean(0), 1e-9, None)
    def pred(E2, P2, C2):
        Q = P2 * w; Q /= Q.sum(1, keepdims=True); return rnd(Q @ L)
    return pred


def m_rank_quantile_transductive(E, P, C, y):   # label-free on the TARGET set: map target E PERCENTILES to train label cum. props
    cum = np.cumsum(np.bincount(y, minlength=6)[1:]) / len(y)
    return lambda E2, P2, C2: cut(E2, np.quantile(E2, cum[:4]))   # uses target E only (no target labels); distribution ~ human by design


METHODS = {"M0 canonical rounded E (ORIGINAL)": m_canonical, "M1 quantile thresholds on E": m_quantile_thresholds,
           "M2 variance-matched E": m_variance_match, "M3 isotonic E->rating": m_isotonic,
           "M4 ordinal logit on E": m_ordlogit_E, "M5 ordinal logit on E+cosines": m_ordlogit_feats,
           "M6 prior-corrected PMF": m_prior_corrected_pmf, "M7 rank-quantile (transductive, label-free)": m_rank_quantile_transductive}


def evaluate(y, p, ref_dist=None):
    m = Hx.metrics(y, p, p)
    out = {k: m[k] for k in ["exact", "within1", "MAE", "RMSE", "spearman", "qwk", "lwk", "macro_F1", "macro_R"]}
    out.update({f"R{c}": m["per_class"][c]["R"] for c in L}); out["pred_dist"] = np.bincount(p, minlength=6)[1:].tolist()
    ref = y if ref_dist is None else ref_dist
    out["W1_vs_ref"] = float(stats.wasserstein_distance(p, ref))
    out["JSD_vs_ref"] = float(jensenshannon(np.bincount(p, minlength=6)[1:] / len(p), np.bincount(ref, minlength=6)[1:] / len(ref), base=2) ** 2)
    out["confusion"] = m["confusion"]
    return out


# ------------------------------------------------------------------ human repeated CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED)
rows = []
for f, (tr, te) in enumerate(cv.split(EH, yH)):
    assert not set(tr) & set(te)
    for nm, fn in METHODS.items():
        pred = fn(EH[tr], PH[tr], CH[tr], yH[tr])(EH[te], PH[te], CH[te])
        r = evaluate(yH[te], pred); r.update({"fold": f, "method": nm}); rows.append(r)
CVd = pd.DataFrame(rows)
num = ["exact", "within1", "MAE", "RMSE", "spearman", "qwk", "lwk", "macro_F1", "macro_R", "R1", "R2", "R3", "R4", "R5", "W1_vs_ref", "JSD_vs_ref"]
summ = CVd.groupby("method")[num].agg(["mean", "std"])
base = CVd[CVd.method.str.startswith("M0")].set_index("fold")
stab = {}
for nm in METHODS:
    c = CVd[CVd.method == nm].set_index("fold")
    stab[nm] = {"qwk_better_folds": int((c.qwk > base.qwk + 1e-12).sum()), "exact_better_folds": int((c.exact > base.exact + 1e-12).sum()),
                "MAE_better_folds": int((c.MAE < base.MAE - 1e-12).sum()), "qwk_diff_mean": float((c.qwk - base.qwk).mean()),
                "qwk_diff_ci95_over_folds": np.percentile(c.qwk - base.qwk, [2.5, 97.5]).tolist(),
                "exact_diff_mean": float((c.exact - base.exact).mean()), "MAE_diff_mean": float((c.MAE - base.MAE).mean())}
# selection (pre-declared rule)
elig = [nm for nm in METHODS if not nm.startswith("M0") and summ.loc[nm, ("R1", "mean")] > 0 and summ.loc[nm, ("R5", "mean")] > 0
        and stab[nm]["qwk_better_folds"] >= 15]
selected = min(elig, key=lambda nm: (-summ.loc[nm, ("qwk", "mean")], summ.loc[nm, ("MAE", "mean")])) if elig else "M0 canonical rounded E (ORIGINAL)"

# ------------------------------------------------------------------ final: fit on ALL human, apply ONCE to synthetic
syn = {}
rng = np.random.default_rng(SEED)
preds_syn = {}
for nm, fn in METHODS.items():
    p = fn(EH, PH, CH, yH)(ES, PS, CS); preds_syn[nm] = p
    r = evaluate(yS, p); r_h = evaluate(yS, p, ref_dist=yH)
    r["W1_vs_human_Q21"], r["JSD_vs_human_Q21"] = r_h["W1_vs_ref"], r_h["JSD_vs_ref"]
    r["persona_means"] = {q: float(p[Sd.persona == q].mean()) for q in ["P1", "P2", "P3", "P4"]}
    syn[nm] = r
cur = preds_syn["M0 canonical rounded E (ORIGINAL)"]
for nm, p in preds_syn.items():
    if nm.startswith("M0"):
        continue
    idx = rng.integers(0, len(yS), (2000, len(yS)))
    q = lambda a, ii: Hx.metrics(yS[ii], a[ii], a[ii])["qwk"]
    dq = np.array([q(p, ii) - q(cur, ii) for ii in idx[:500]])     # 500 resamples (speed)
    de = np.array([(p[ii] == yS[ii]).mean() - (cur[ii] == yS[ii]).mean() for ii in idx])
    a, b = int(((p == yS) & (cur != yS)).sum()), int(((p != yS) & (cur == yS)).sum())
    syn[nm]["vs_canonical"] = {"qwk_diff_ci95": np.percentile(dq, [2.5, 97.5]).tolist(), "exact_diff_ci95": np.percentile(de, [2.5, 97.5]).tolist(),
                               "mcnemar_p": float(stats.binomtest(a, a + b).pvalue) if a + b else 1.0, "only_method_right": a, "only_canonical_right": b}
res = {"label": "Phase 17 EXPERIMENTAL rating-conversion methods on frozen SSR outputs",
       "data": {"human_n": int(len(yH)), "human_dist": np.bincount(yH, minlength=6)[1:].tolist(), "synthetic_n": int(len(yS)),
                "synthetic_stance_dist": np.bincount(yS, minlength=6)[1:].tolist(),
                "E_mean_human": float(EH.mean()), "E_mean_synthetic": float(ES.mean()),
                "E_by_label_human": {int(k): float(EH[yH == k].mean()) for k in L}, "E_by_label_synthetic": {int(k): float(ES[yS == k].mean()) for k in L}},
       "protocol": {"cv": "RepeatedStratifiedKFold 5x4 on human respondents", "seed": SEED, "selection_rule": __doc__.split("SELECTION RULE:")[1].split("FINAL TEST:")[0].strip()},
       "human_cv_summary": {nm: {k: [float(summ.loc[nm, (k, "mean")]), float(summ.loc[nm, (k, "std")])] for k in num} for nm in METHODS},
       "human_cv_stability_vs_canonical": stab, "eligible_under_rule": elig, "selected": selected, "synthetic_final_test": syn}
CVd.to_csv(OUT / "phase17_human_cv_per_fold.csv", index=False)
pd.DataFrame({"id": Sd.id, "persona": Sd.persona, "intended_stance": yS, "E": ES, **{nm: p for nm, p in preds_syn.items()}}).to_csv(OUT / "phase17_synthetic_predictions.csv", index=False)
(OUT / "phase17_metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
pd.set_option("display.width", 260)
print(json.dumps(res["data"], indent=1))
print("\nHUMAN CV (mean over 20 held-out folds)"); print(summ.xs("mean", axis=1, level=1).round(3).to_string())
print("\nSD"); print(summ.xs("std", axis=1, level=1)[["exact", "qwk", "macro_F1", "R1", "R5", "MAE"]].round(3).to_string())
print("\nSTABILITY", json.dumps(stab, indent=0)); print("ELIGIBLE", elig, "SELECTED", selected)
print("\nSYNTHETIC FINAL TEST (labels = intended stances)")
print(pd.DataFrame({nm: {k: v for k, v in r.items() if k not in ("confusion", "vs_canonical", "persona_means")} for nm, r in syn.items()}).T.to_string())
for nm, r in syn.items(): print(nm, "| persona means", {k: round(v, 2) for k, v in r["persona_means"].items()}, "| vs canonical", r.get("vs_canonical"))
