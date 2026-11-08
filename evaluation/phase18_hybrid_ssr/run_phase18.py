"""Phase 18: hybrid E[r]/PMF -> Likert conversions on FROZEN SSR outputs (no LLM, no embedding calls).
Inputs: Phase 13 human scores (251 respondents, Q21) and Phase 16 synthetic scores (200, intended stances = eval labels only).
Same CV as Phase 17: RepeatedStratifiedKFold(5 x 4, seed 20261106). Everything is fit on human TRAIN folds only;
transductive methods additionally use the UNLABELLED target E[r] distribution (never target labels).
Selection (declared a priori, human-only): highest mean human QWK among methods with QWK > M0 in >= 15/20 folds,
mean exact >= M0 - 0.02, and mean 1-star and 5-star recall > 0; tie -> lower MAE. Synthetic labels are used ONLY
afterwards as a held-out PASS/FAIL check (QWK CI lower bound vs M0 > -0.05); they never choose among methods.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ssr_hybrid_experiments as Hx  # read-only metric helper
L = np.arange(1, 6); SEED = 20261106
Hd = pd.read_csv(ROOT / "evaluation/phase13_human_ssr/phase13_scored_human_q26.csv")
Sd = pd.read_csv(ROOT / "evaluation/phase16_bi_only_generation/phase16_scored_responses.csv")
EH, PH, yH = Hd.imp_E.values, Hd[[f"imp_p{k}" for k in L]].values, Hd.q21.values.astype(int)
ES, PS, yS = Sd.E.values, Sd[[f"p{k}" for k in L]].values, Sd.intended_stance.values.astype(int)
assert np.allclose(PH @ L, EH, atol=1e-6) and np.allclose(PS @ L, ES, atol=1e-6)
rnd = lambda x: np.clip(np.floor(np.asarray(x, float) + 0.5), 1, 5).astype(int)
cut = lambda x, t: 1 + (np.asarray(x)[:, None] >= np.asarray(t)[None, :]).sum(1)
cumprops = lambda y: (np.cumsum(np.bincount(y, minlength=6)[1:]) / len(y))[:4]


# ---------- components (identical maths to Phase 17 where reused)
def reweight_fit(P, y, iters=200):
    target = np.bincount(y, minlength=6)[1:] / len(y); w = np.ones(5)
    for _ in range(iters):
        Q = P * w; Q /= Q.sum(1, keepdims=True); w *= target / np.clip(Q.mean(0), 1e-9, None)
    return w
def reweight_E(P, w):
    Q = P * w; Q /= Q.sum(1, keepdims=True); return Q @ L
def affine_fit(x, y): return (x.mean(), x.std(ddof=1), y.mean(), y.std(ddof=1))
def affine(x, a): return a[2] + (x - a[0]) * a[3] / a[1]


# ---------- methods: fit(E,P,y) -> f(E2,P2) returning (discrete, continuous score or None)
def M0(E, P, y): return lambda E2, P2: (rnd(E2), E2)
def M2(E, P, y):
    a = affine_fit(E, y); return lambda E2, P2: (rnd(affine(E2, a)), affine(E2, a))
def M6(E, P, y):
    w = reweight_fit(P, y); return lambda E2, P2: (rnd(reweight_E(P2, w)), reweight_E(P2, w))
def M7(E, P, y):
    c = cumprops(y); return lambda E2, P2: (cut(E2, np.quantile(E2, c)), None)          # transductive
def M8(E, P, y):
    c = cumprops(y); z = (E - E.mean()) / E.std(ddof=1); t = np.quantile(z, c)          # thresholds in TRAIN z-space
    return lambda E2, P2: (cut((E2 - E2.mean()) / E2.std(ddof=1), t), (E2 - E2.mean()) / E2.std(ddof=1))  # target self-standardised
def M9(E, P, y):
    w = reweight_fit(P, y); a = affine_fit(reweight_E(P, w), y)
    return lambda E2, P2: (rnd(affine(reweight_E(P2, w), a)), affine(reweight_E(P2, w), a))
def M10(E, P, y):                                                                      # = M7 (affine is rank-preserving)
    a = affine_fit(E, y); c = cumprops(y)
    return lambda E2, P2: (cut(affine(E2, a), np.quantile(affine(E2, a), c)), None)
def M11(E, P, y):
    w = reweight_fit(P, y); c = cumprops(y)
    return lambda E2, P2: (cut(reweight_E(P2, w), np.quantile(reweight_E(P2, w), c)), None)


MEMBERS = {"M2": M2, "M6": M6, "M7": M7, "M8": M8}       # ensemble members fixed a priori (Phase 17 QWK > M0 + new M8)
def _member_preds(E, P, y, E2, P2): return np.column_stack([MEMBERS[k](E, P, y)(E2, P2)[0] for k in MEMBERS])
def M12_avg(E, P, y): return lambda E2, P2: (rnd(_member_preds(E, P, y, E2, P2).mean(1)), _member_preds(E, P, y, E2, P2).mean(1))
def M12_vote(E, P, y):
    def f(E2, P2):
        X = _member_preds(E, P, y, E2, P2)
        out = [int(np.median(r)) if np.bincount(r).max() == 1 or (np.bincount(r) == np.bincount(r).max()).sum() > 1 else int(np.bincount(r).argmax()) for r in X]
        return np.array(out), None                                                      # majority; ties -> lower median
    return f
WGRID = [w for w in itertools.product([0, .25, .5, .75, 1], repeat=4) if abs(sum(w) - 1) < 1e-9]
def M12_weighted(E, P, y):
    """weights chosen by inner 3-fold CV on the TRAIN fold only (max QWK, tie: closer to uniform)."""
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED); score = np.zeros(len(WGRID))
    for a, b in inner.split(E, y):
        X = _member_preds(E[a], P[a], y[a], E[b], P[b])
        for i, w in enumerate(WGRID):
            score[i] += Hx.metrics(y[b], rnd(X @ np.array(w)), rnd(X @ np.array(w)))["qwk"]
    best = max(range(len(WGRID)), key=lambda i: (round(score[i], 10), -np.abs(np.array(WGRID[i]) - .25).sum()))
    w = np.array(WGRID[best])
    f = lambda E2, P2: (rnd(_member_preds(E, P, y, E2, P2) @ w), _member_preds(E, P, y, E2, P2) @ w)
    f.weights = w.tolist(); return f


METHODS = {"M0 canonical (ORIGINAL)": M0, "M2 mean/SD alignment": M2, "M6 probability reweighting": M6,
           "M7 rank-quantile (transductive)": M7, "M8 source-wise z-score (transductive)": M8, "M9 hybrid M6->M2": M9,
           "M10 hybrid M2->M7 (redundant check)": M10, "M11 hybrid M6->M7 (transductive)": M11,
           "M12a ensemble average": M12_avg, "M12b ensemble majority vote": M12_vote, "M12c ensemble weighted (inner-CV)": M12_weighted}
TRANSDUCTIVE = {"M7 rank-quantile (transductive)", "M8 source-wise z-score (transductive)", "M10 hybrid M2->M7 (redundant check)",
                "M11 hybrid M6->M7 (transductive)", "M12a ensemble average", "M12b ensemble majority vote", "M12c ensemble weighted (inner-CV)"}


def ev(y, p, ref):
    m = Hx.metrics(y, p, p)
    o = {k: m[k] for k in ["exact", "within1", "MAE", "RMSE", "spearman", "qwk", "lwk", "macro_P", "macro_R", "macro_F1"]}
    o.update({f"R{c}": m["per_class"][c]["R"] for c in L}); o["pred_mean"], o["pred_sd"] = float(p.mean()), float(p.std(ddof=1))
    o["pred_dist"] = np.bincount(p, minlength=6)[1:].tolist()
    q = lambda v: np.bincount(v, minlength=6)[1:] / len(v)
    o["W1_vs_human"] = float(stats.wasserstein_distance(p, ref)); o["JSD_vs_human"] = float(jensenshannon(q(p), q(ref), base=2) ** 2)
    o["confusion"] = m["confusion"]; return o


# ---------- human CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED)
rows, oof, wlog = [], {nm: np.zeros((4, len(yH)), int) for nm in METHODS}, []
for f, (tr, te) in enumerate(cv.split(EH, yH)):
    for nm, fn in METHODS.items():
        g = fn(EH[tr], PH[tr], yH[tr]); p = g(EH[te], PH[te])[0]
        if nm.startswith("M12c"): wlog.append(g.weights)
        r = ev(yH[te], p, yH[tr]); r.update({"fold": f, "method": nm}); rows.append(r); oof[nm][f // 5, te] = p
CV = pd.DataFrame(rows)
num = ["exact", "within1", "MAE", "RMSE", "spearman", "qwk", "lwk", "macro_P", "macro_R", "macro_F1", "R1", "R2", "R3", "R4", "R5", "pred_mean", "pred_sd", "W1_vs_human", "JSD_vs_human"]
summ = CV.groupby("method", sort=False)[num].agg(["mean", "std"])
b0 = CV[CV.method.str.startswith("M0")].set_index("fold")
stab = {}
for nm in METHODS:
    c = CV[CV.method == nm].set_index("fold")
    stab[nm] = {"qwk_better_folds": int((c.qwk > b0.qwk + 1e-12).sum()), "exact_better_folds": int((c.exact > b0.exact + 1e-12).sum()),
                "exact_worse_folds": int((c.exact < b0.exact - 1e-12).sum()), "MAE_better_folds": int((c.MAE < b0.MAE - 1e-12).sum()),
                **{f"{k}_diff_mean": float((c[k] - b0[k]).mean()) for k in ["qwk", "exact", "MAE", "macro_F1", "spearman"]}}
redundant_M10_equals_M7 = bool((oof["M10 hybrid M2->M7 (redundant check)"] == oof["M7 rank-quantile (transductive)"]).all())

# paired bootstrap on repeat-0 out-of-fold predictions (n=251, each respondent predicted once)
rng = np.random.default_rng(SEED); IDX = rng.integers(0, len(yH), (1000, len(yH)))
def boot(y, a, b, idx):
    d = {k: [] for k in ["exact", "qwk", "MAE", "macro_F1"]}
    for ii in idx:
        ma, mb = Hx.metrics(y[ii], a[ii], a[ii]), Hx.metrics(y[ii], b[ii], b[ii])
        for k in d: d[k].append(ma[k] - mb[k])
    ci = {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in d.items()}
    A, B = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    ci["mcnemar_p"] = float(stats.binomtest(A, A + B).pvalue) if A + B else 1.0; return ci
hboot = {nm: boot(yH, oof[nm][0], oof["M0 canonical (ORIGINAL)"][0], IDX[:400]) for nm in METHODS if not nm.startswith("M0")}
h_persona = {nm: {p: float((oof[nm][0][Hd.persona.values == p] == yH[Hd.persona.values == p]).mean()) for p in sorted(Hd.persona.unique())} for nm in METHODS}

# ---------- selection (human only)
elig = [nm for nm in METHODS if not nm.startswith("M0") and not nm.startswith("M10")
        and stab[nm]["qwk_better_folds"] >= 15 and summ.loc[nm, ("exact", "mean")] >= summ.loc["M0 canonical (ORIGINAL)", ("exact", "mean")] - 0.02
        and summ.loc[nm, ("R1", "mean")] > 0 and summ.loc[nm, ("R5", "mean")] > 0]
best_human = min(elig, key=lambda nm: (-summ.loc[nm, ("qwk", "mean")], summ.loc[nm, ("MAE", "mean")])) if elig else "M0 canonical (ORIGINAL)"

# ---------- synthetic: refit on ALL human, apply once
syn, sp = {}, {}
for nm, fn in METHODS.items():
    g = fn(EH, PH, yH); p = g(ES, PS)[0]; sp[nm] = p
    r = ev(yS, p, yH); r["persona_means"] = {q: float(p[Sd.persona.values == q].mean()) for q in ["P1", "P2", "P3", "P4"]}
    r["persona_exact"] = {q: float((p[Sd.persona.values == q] == yS[Sd.persona.values == q]).mean()) for q in ["P1", "P2", "P3", "P4"]}
    if nm.startswith("M12c"): r["weights_full_human"] = g.weights
    syn[nm] = r
SIDX = rng.integers(0, len(yS), (1000, len(yS)))
sboot = {nm: boot(yS, sp[nm], sp["M0 canonical (ORIGINAL)"], SIDX[:500]) for nm in METHODS if not nm.startswith("M0")}
syn_check = {nm: bool(sboot[nm]["qwk"][0] > -0.05) for nm in sboot}
balanced = [nm for nm in elig if syn_check.get(nm)]
best_balanced = min(balanced, key=lambda nm: (-summ.loc[nm, ("qwk", "mean")], summ.loc[nm, ("MAE", "mean")])) if balanced else "M0 canonical (ORIGINAL)"
best_synthetic_descriptive = max(syn, key=lambda nm: syn[nm]["qwk"])

res = {"label": "Phase 18 EXPERIMENTAL hybrid conversions on frozen SSR; human CV 5x4 (seed 20261106); synthetic labels = evaluation only",
       "human_n": int(len(yH)), "synthetic_n": int(len(yS)), "human_q21_dist": np.bincount(yH, minlength=6)[1:].tolist(),
       "synthetic_stance_dist": np.bincount(yS, minlength=6)[1:].tolist(), "transductive_methods": sorted(TRANSDUCTIVE),
       "M10_identical_to_M7_on_all_human_oof": redundant_M10_equals_M7,
       "M10_identical_to_M7_on_synthetic": bool((sp["M10 hybrid M2->M7 (redundant check)"] == sp["M7 rank-quantile (transductive)"]).all()),
       "human_cv": {nm: {k: [float(summ.loc[nm, (k, "mean")]), float(summ.loc[nm, (k, "std")])] for k in num} for nm in METHODS},
       "human_stability_vs_M0": stab, "human_paired_bootstrap_vs_M0_repeat0_oof": hboot, "human_persona_exact_oof": h_persona,
       "M12c_weights_per_fold": wlog, "eligible_human_rule": elig, "best_human": best_human,
       "synthetic": syn, "synthetic_paired_bootstrap_vs_M0": sboot, "synthetic_nondeterioration_check": syn_check,
       "best_balanced": best_balanced, "best_synthetic_descriptive_only": best_synthetic_descriptive,
       "human_persona_q21_means": Hd.groupby("persona").q21.mean().round(3).to_dict(),
       "synthetic_persona_stance_means": Sd.groupby("persona").intended_stance.mean().round(3).to_dict()}
(OUT / "phase18_metrics.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
CV.drop(columns=["confusion"]).to_csv(OUT / "phase18_human_cv_per_fold.csv", index=False)
pd.DataFrame({"id": Sd.id, "persona": Sd.persona, "intended_stance": yS, "E": ES, **sp}).to_csv(OUT / "phase18_synthetic_predictions.csv", index=False)
pd.DataFrame({"respondent_row": Hd.row, "persona": Hd.persona, "q21": yH, "E": EH, **{nm: oof[nm][0] for nm in METHODS}}).to_csv(OUT / "phase18_human_oof_predictions_repeat0.csv", index=False)
comp = pd.DataFrame([{"method": nm, **{f"human_{k}": summ.loc[nm, (k, "mean")] for k in ["exact", "within1", "MAE", "RMSE", "spearman", "qwk", "lwk", "macro_F1", "macro_R", "R1", "R5", "W1_vs_human"]},
                      "human_qwk_better_folds": stab[nm]["qwk_better_folds"],
                      **{f"syn_{k}": syn[nm][k] for k in ["exact", "within1", "MAE", "RMSE", "spearman", "qwk", "lwk", "macro_F1", "macro_R", "R1", "R5", "W1_vs_human", "JSD_vs_human"]},
                      "syn_qwk_diff_ci_lo": sboot.get(nm, {}).get("qwk", [np.nan])[0], "syn_qwk_diff_ci_hi": sboot.get(nm, {}).get("qwk", [np.nan, np.nan])[1]} for nm in METHODS])
comp.to_csv(OUT / "phase18_comparison.csv", index=False)

# ---------- plots
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
show = ["M0 canonical (ORIGINAL)", "M2 mean/SD alignment", "M6 probability reweighting", "M7 rank-quantile (transductive)", "M8 source-wise z-score (transductive)", best_human]
show = list(dict.fromkeys(show))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for ax, title, ref, preds in [(axs[0], "Synthetic predicted distribution (n=200)", yS, sp), (axs[1], "Human OOF predicted distribution (n=251, repeat 0)", yH, {k: oof[k][0] for k in oof})]:
    wdt = 0.8 / (len(show) + 1)
    ax.bar(L - 0.4 + wdt / 2, np.bincount(ref, minlength=6)[1:] / len(ref), wdt, label="reference labels", color="#444")
    for i, nm in enumerate(show):
        ax.bar(L - 0.4 + wdt * (i + 1.5), np.bincount(preds[nm], minlength=6)[1:] / len(ref), wdt, label=nm.split(" (")[0])
    ax.set_xticks(L); ax.set_title(title, fontsize=10); ax.set_xlabel("rating"); ax.set_ylabel("share")
axs[1].legend(fontsize=7, frameon=False); fig.tight_layout(); fig.savefig(OUT / "phase18_distributions.png", dpi=200); plt.close(fig)
fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, (nm, src) in zip(axs.ravel(), [(m, "human") for m in show[:3]] + [(m, "synthetic") for m in show[:3]]):
    cm = np.array(syn[nm]["confusion"]) if src == "synthetic" else np.array(Hx.metrics(yH, oof[nm][0], oof[nm][0])["confusion"])
    ax.imshow(cm, cmap="Blues")
    for i, j in itertools.product(range(5), range(5)): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xticks(range(5), L); ax.set_yticks(range(5), L); ax.set_title(f"{src}: {nm.split(' (')[0]}", fontsize=9); ax.set_xlabel("predicted"); ax.set_ylabel("label")
fig.tight_layout(); fig.savefig(OUT / "phase18_confusion_matrices.png", dpi=200); plt.close(fig)

pd.set_option("display.width", 300)
print("HUMAN CV mean"); print(summ.xs("mean", axis=1, level=1).round(3).to_string())
print("\nHUMAN CV sd"); print(summ.xs("std", axis=1, level=1)[["exact", "qwk", "MAE", "macro_F1", "R1", "R5"]].round(3).to_string())
print("\nSTABILITY"); print(pd.DataFrame(stab).T.round(3).to_string())
print("\nHUMAN BOOT"); print(pd.DataFrame({k: {a: (np.round(b, 3).tolist() if isinstance(b, list) else round(b, 4)) for a, b in v.items()} for k, v in hboot.items()}).T.to_string())
print("\nhuman persona exact"); print(pd.DataFrame(h_persona).T.round(3).to_string())
print("\nSYNTHETIC"); print(pd.DataFrame({nm: {k: v for k, v in r.items() if k not in ("confusion", "persona_means", "persona_exact", "weights_full_human")} for nm, r in syn.items()}).T.to_string())
print("\nSYN BOOT"); print(pd.DataFrame({k: {a: (np.round(b, 3).tolist() if isinstance(b, list) else round(b, 4)) for a, b in v.items()} for k, v in sboot.items()}).T.to_string())
print("\nsyn persona means"); print(pd.DataFrame({nm: r["persona_means"] for nm, r in syn.items()}).T.round(2).to_string())
print(json.dumps({k: res[k] for k in ["M10_identical_to_M7_on_all_human_oof", "M10_identical_to_M7_on_synthetic", "eligible_human_rule", "best_human", "synthetic_nondeterioration_check", "best_balanced", "best_synthetic_descriptive_only", "human_persona_q21_means", "synthetic_persona_stance_means"]}, indent=1))
print("M12c weights per fold", wlog, "| full-human", syn["M12c ensemble weighted (inner-CV)"].get("weights_full_human"))
