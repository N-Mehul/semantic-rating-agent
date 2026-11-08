"""Read-only model performance audit. Writes ONLY into evaluation/model_performance_audit/.
Imports canonical_ssr / improved_ssr unmodified. Run: python evaluation/model_performance_audit/run_audit.py
"""
from __future__ import annotations
import json, re, sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support, cohen_kappa_score,
                             r2_score, accuracy_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
(OUT / "metrics").mkdir(exist_ok=True); (OUT / "figures").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from canonical_ssr import similarities_to_pmf, cosine_similarities, expected_rating  # noqa
from improved_ssr import improved_similarities_to_pmf  # noqa
from survey_ssr_bridge import get_construct_anchors  # noqa

SEED = 20261028
RNG = np.random.default_rng(SEED)
L = [1, 2, 3, 4, 5]
RESULTS: dict = {}
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})
C = {"ssr": "#2a6fbb", "base": "#999999", "alt": "#d9822b", "good": "#2e8b57", "bad": "#c0392b"}


def save(name, obj):
    if isinstance(obj, pd.DataFrame):
        obj.to_csv(OUT / "metrics" / f"{name}.csv", index=False)
    else:
        (OUT / "metrics" / f"{name}.json").write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def fig_save(fig, name):
    fig.savefig(OUT / "figures" / f"{name}.png"); fig.savefig(OUT / "figures" / f"{name}.pdf"); plt.close(fig)


# ---------------------------------------------------------------- metrics
def reg_metrics(y, pred_cont, pred_int):
    y = np.asarray(y, float); pc = np.asarray(pred_cont, float); pi = np.asarray(pred_int, float)
    out = {"n": int(len(y)), "exact_acc": float((pi == y).mean()), "pm1_acc": float((np.abs(pi - y) <= 1).mean()),
           "mae_int": float(np.abs(pi - y).mean()), "mse_int": float(((pi - y) ** 2).mean()),
           "mae": float(np.abs(pc - y).mean()), "mse": float(((pc - y) ** 2).mean())}
    out["rmse"] = float(np.sqrt(out["mse"])); out["rmse_int"] = float(np.sqrt(out["mse_int"]))
    out["bias_mean_error"] = float((pc - y).mean())
    if np.std(pc) > 0 and np.std(y) > 0:
        out["pearson"] = float(stats.pearsonr(pc, y)[0]); out["spearman"] = float(stats.spearmanr(pc, y)[0])
    else:
        out["pearson"] = out["spearman"] = float("nan")
    out["r2"] = float(r2_score(y, pc))
    out["qw_kappa"] = float(cohen_kappa_score(y.astype(int), pi.astype(int), weights="quadratic"))
    out["macro_f1"] = float(precision_recall_fscore_support(y.astype(int), pi.astype(int), labels=L, average="macro", zero_division=0)[2])
    out["pred_sd"] = float(pc.std(ddof=1)); out["true_sd"] = float(y.std(ddof=1))
    return out


def boot(y, pc, pi, groups=None, n=1000, seed=0):
    """Percentile bootstrap CIs; cluster bootstrap over `groups` when given."""
    rng = np.random.default_rng(seed); y, pc, pi = map(np.asarray, (y, pc, pi))
    if groups is not None:
        g = pd.factorize(groups)[0]; idx_by = [np.where(g == k)[0] for k in range(g.max() + 1)]
    keys = ["exact_acc", "pm1_acc", "mae", "rmse", "pearson", "spearman", "r2"]
    acc = {k: [] for k in keys}
    for _ in range(n):
        if groups is None:
            ix = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(idx_by), len(idx_by)); ix = np.concatenate([idx_by[p] for p in pick])
        m = reg_metrics(y[ix], pc[ix], pi[ix])
        for k in keys:
            if np.isfinite(m[k]): acc[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [np.nan, np.nan] for k, v in acc.items()}


def per_rating(y, pi):
    p, r, f, s = precision_recall_fscore_support(y.astype(int), pi.astype(int), labels=L, zero_division=0)
    return pd.DataFrame({"rating": L, "support": s, "precision": p, "recall": r, "f1": f})


def grade(name, v):
    t = {"exact_acc": (.60, .40), "pm1_acc": (.90, .75), "pearson": (.70, .40), "spearman": (.70, .40), "r2": (.50, .20)}
    if name in t:
        return "good" if v >= t[name][0] else "moderate" if v >= t[name][1] else "poor"
    if name == "mae": return "good" if v <= .5 else "moderate" if v <= .9 else "poor"
    if name == "rmse": return "good" if v <= .7 else "moderate" if v <= 1.1 else "poor"
    return ""


def cm_heatmap(cm, title, name, normalize=True):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    m = cm / cm.sum(1, keepdims=True).clip(1) if normalize else cm
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    for i, j in itertools.product(range(5), range(5)):
        ax.text(j, i, f"{cm[i, j]}\n({m[i, j]*100:.0f}%)" if normalize else cm[i, j], ha="center", va="center",
                fontsize=8, color="white" if m[i, j] > .5 else "black")
    ax.set_xticks(range(5), L); ax.set_yticks(range(5), L); ax.set_xlabel("Predicted rating"); ax.set_ylabel("Actual rating")
    ax.set_title(title, fontsize=10); fig.colorbar(im, label="row-normalised share" if normalize else "count"); fig_save(fig, name)


# ================================================================ PART A: review-rating SSR (human ground truth, n=10,003)
print("== A: review-rating SSR")
src = pd.read_csv(ROOT / "data/Mobile Reviews Sentiment.csv", low_memory=False)
h = pd.read_csv(ROOT / "data/calibrated_discretization/heldout_predictions.csv")
wb = pd.read_excel(ROOT / "data/actual_reviews.xlsx", header=None); wb.columns = src.columns  # held-out workbook has no header row
wb["review_id"] = pd.to_numeric(wb.review_id, errors="coerce")
h = h.merge(wb[["review_id", "brand", "model", "sentiment", "battery_life_rating", "camera_rating", "performance_rating",
                 "design_rating", "display_rating"]], on="review_id", how="left")
train = src[~src.review_id.isin(h.review_id) & src.rating.notna() & src.review_text.notna()].copy()
y = h.actual_rating.values.astype(int)
A = {"data": {"n_eval": len(h), "unique_texts_eval": int(h.review_text.nunique()), "n_train_source": len(train),
              "unique_texts_train": int(train.review_text.nunique()),
              "eval_rows_with_text_seen_in_train": float(h.review_text.isin(train.review_text).mean()),
              "eval_unique_texts_seen_in_train": float(pd.Series(h.review_text.unique()).isin(train.review_text).mean()),
              "rating_dist_eval": {int(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()}}}
# per-text label noise ceiling: best any text-only predictor could do on eval (oracle per-text mode)
mode = h.groupby("review_text").actual_rating.agg(lambda s: s.mode().iloc[0])
A["data"]["text_only_oracle_ceiling_exact_acc"] = float((h.review_text.map(mode) == h.actual_rating).mean())
A["data"]["text_only_oracle_ceiling_pm1"] = float((np.abs(h.review_text.map(mode) - h.actual_rating) <= 1).mean())
A["data"]["texts_with_multiple_distinct_ratings_eval"] = float((h.groupby("review_text").actual_rating.nunique() > 1).mean())

models = {"SSR argmax (PMF mode)": (h.expected_rating.values, h.argmax_rating.values),
          "SSR rounded E[r]": (h.expected_rating.values, np.clip(np.round(h.expected_rating.values), 1, 5)),
          "SSR calibrated cutpoints": (h.expected_rating.values, h.calibrated_rating.values)}
tr_mode = int(train.rating.mode().iloc[0]); tr_prior = train.rating.value_counts(normalize=True).sort_index().reindex(L, fill_value=0).values
rs = np.random.default_rng(SEED)
rnd = np.mean([rs.choice(L, len(y), p=tr_prior) for _ in range(1)], 0)  # one seeded draw; seed-sweep below
models["Random (train prior)"] = (rnd, rnd)
models["Majority class (train mode=%d)" % tr_mode] = (np.full(len(y), float(tr_mode)), np.full(len(y), tr_mode))
models["Train-mean constant"] = (np.full(len(y), train.rating.mean()), np.full(len(y), round(train.rating.mean())))
# sentiment-label baseline: dataset sentiment label -> mean/mode rating learned from TRAIN (label is derived from the rating -> optimistic)
sm = train.groupby("sentiment").rating.agg(["mean", lambda s: s.mode().iloc[0]]); sm.columns = ["mean", "mode"]
A["data"]["eval_rows_missing_sentiment_after_merge"] = int(h.sentiment.isna().sum())
models["Sentiment-label lookup (leaky label)"] = (h.sentiment.map(sm["mean"]).fillna(train.rating.mean()).values, h.sentiment.map(sm["mode"]).fillna(tr_mode).values.astype(int))
# text-only supervised baseline trained on train split
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1); Xtr = tf.fit_transform(train.review_text); Xte = tf.transform(h.review_text)
lr = LogisticRegression(max_iter=2000).fit(Xtr, train.rating.astype(int)); pp = lr.predict_proba(Xte)
models["TF-IDF+LogReg text-only (train->eval)"] = (pp @ lr.classes_, lr.classes_[pp.argmax(1)])
# aspect RF baseline (existing saved predictions) - leakage-flagged
asp = pd.read_csv(ROOT / "data/aspect_model/predictions.csv")
models["Aspect RF D1 (existing; TARGET LEAKAGE)"] = (asp.predicted_rating.values.astype(float), asp.predicted_rating.values.astype(int))
yasp = asp.actual_rating.values.astype(int)
rows, cis = [], {}
for nme, (pc, pi) in models.items():
    yy = yasp if "Aspect RF" in nme else y
    gg = None if "Aspect RF" in nme else h.review_text.values
    m = reg_metrics(yy, pc, pi); m["model"] = nme; rows.append(m)
    cis[nme] = boot(yy, pc, pi, groups=gg, n=500, seed=1)
cmp = pd.DataFrame(rows).set_index("model"); save("A_review_model_comparison", cmp.reset_index())
save("A_review_bootstrap_ci_cluster_by_text", cis)
# random-seed sensitivity
seeds = []
for s in range(30):
    r = np.random.default_rng(s).choice(L, len(y), p=tr_prior); seeds.append(reg_metrics(y, r, r))
A["random_baseline_30_seeds"] = {k: [float(np.mean([m[k] for m in seeds])), float(np.std([m[k] for m in seeds]))] for k in ["exact_acc", "mae", "rmse", "pearson"]}
# main model detail
pc, pi = models["SSR argmax (PMF mode)"]; cm = confusion_matrix(y, pi, labels=L)
pr = per_rating(y, pi); save("A_per_rating_argmax", pr); save("A_confusion_argmax", pd.DataFrame(cm, index=L, columns=L).reset_index())
pr2 = per_rating(y, models["SSR calibrated cutpoints"][1]); save("A_per_rating_calibrated", pr2)
dist = pd.DataFrame({"rating": L, "actual": [int((y == k).sum()) for k in L], "argmax_pred": [int((pi == k).sum()) for k in L],
                     "calibrated_pred": [int((models["SSR calibrated cutpoints"][1] == k).sum()) for k in L]}); save("A_rating_distribution", dist)
err = pc - y; bias_by = pd.DataFrame({"actual": L, "mean_signed_error": [float(err[y == k].mean()) for k in L], "mean_abs_error": [float(np.abs(err[y == k]).mean()) for k in L]}); save("A_bias_by_actual_rating", bias_by)
A["error_summary"] = {"mean_signed_error": float(err.mean()), "sd_error": float(err.std()), "skew": float(stats.skew(err))}
# per-brand generalisation
bm = []
for b, g in h.groupby("brand"):
    m = reg_metrics(g.actual_rating.values, g.expected_rating.values, g.argmax_rating.values); m["brand"] = b; bm.append(m)
save("A_per_brand", pd.DataFrame(bm)[["brand", "n", "exact_acc", "pm1_acc", "mae", "rmse", "spearman"]])
# paired tests: SSR vs baselines (cluster bootstrap on |err| difference), McNemar-style via text-cluster bootstrap
def paired_boot(ya, pa, pb, groups, n=1000, seed=3):
    rng = np.random.default_rng(seed); g = pd.factorize(groups)[0]; idx = [np.where(g == k)[0] for k in range(g.max() + 1)]
    d_abs = np.abs(pa - ya) - np.abs(pb - ya); d_acc = (np.round(pa) == ya).astype(float) - (np.round(pb) == ya).astype(float)
    r = []
    for _ in range(n):
        ix = np.concatenate([idx[p] for p in rng.integers(0, len(idx), len(idx))]); r.append((d_abs[ix].mean(), d_acc[ix].mean()))
    r = np.array(r); return {"mae_diff": float(d_abs.mean()), "mae_diff_ci": list(np.percentile(r[:, 0], [2.5, 97.5])),
                             "acc_diff": float(d_acc.mean()), "acc_diff_ci": list(np.percentile(r[:, 1], [2.5, 97.5]))}
A["paired_vs_ssr_argmax"] = {k: paired_boot(y, pc, v[0] if k != "Random (train prior)" else v[0], h.review_text.values)
                             for k, v in models.items() if "SSR" not in k and "Aspect" not in k}
# leave-one-brand-out for supervised text baseline
lobo = []
for b in sorted(src.brand.dropna().unique()):
    tr_ = train[train.brand != b]; te_ = h[h.brand == b]
    if len(te_) == 0: continue
    t2 = TfidfVectorizer(ngram_range=(1, 2)); l2 = LogisticRegression(max_iter=1000).fit(t2.fit_transform(tr_.review_text), tr_.rating.astype(int))
    p2 = l2.predict(t2.transform(te_.review_text)); lobo.append({"held_out_brand": b, "n": len(te_), "tfidf_lr_exact": float((p2 == te_.actual_rating).mean()),
                                                                  "tfidf_lr_mae": float(np.abs(p2 - te_.actual_rating).mean()),
                                                                  "ssr_argmax_exact": float((te_.argmax_rating == te_.actual_rating).mean())})
save("A_leave_one_brand_out", pd.DataFrame(lobo))
RESULTS["A_review"] = A
# aspect leakage: how well do aspect ratings alone track rating
A["aspect_leakage"] = {"spearman_mean_aspect_vs_rating": float(stats.spearmanr(src[["battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating"]].mean(1), src.rating, nan_policy="omit")[0]),
                       "sentiment_label_vs_rating_cramers_note": "sentiment label is a near-deterministic function of rating (see crosstab in report)"}
# figures A
cm_heatmap(cm, "Review SSR (argmax) - confusion matrix, n=10,003", "fig_A_confusion_matrix_review_ssr")
fig, ax = plt.subplots(figsize=(5, 4.4)); jit = np.random.default_rng(0).uniform(-.18, .18, len(y))
ax.scatter(y + jit, pc, s=4, alpha=.15, color=C["ssr"]); ax.plot([1, 5], [1, 5], "k--", lw=1)
ax.plot(L, [pc[y == k].mean() for k in L], "o-", color=C["alt"], label="mean E[r] per actual rating")
ax.set_xlabel("Actual human rating"); ax.set_ylabel("SSR expected rating E[r]"); ax.legend(frameon=False); ax.set_title("Actual vs predicted (review SSR)")
fig_save(fig, "fig_A_actual_vs_predicted_scatter")
fig, ax = plt.subplots(figsize=(6.2, 3.8)); w = .27
for i, (col, lab, c) in enumerate([("actual", "Actual", C["base"]), ("argmax_pred", "SSR argmax", C["ssr"]), ("calibrated_pred", "SSR calibrated", C["alt"])]):
    ax.bar(np.array(L) + (i - 1) * w, dist[col] / dist[col].sum() * 100, w, label=lab, color=c)
ax.set_xlabel("Rating"); ax.set_ylabel("% of rows"); ax.legend(frameon=False); ax.set_title("Rating distribution comparison"); fig_save(fig, "fig_A_rating_distribution")
fig, ax = plt.subplots(figsize=(6, 3.8)); ax.hist(err, bins=40, color=C["ssr"], alpha=.8); ax.axvline(0, color="k", lw=1); ax.axvline(err.mean(), color=C["bad"], ls="--", label=f"mean={err.mean():.2f}")
ax.set_xlabel("E[r] - actual rating"); ax.set_ylabel("count"); ax.legend(frameon=False); ax.set_title("Error distribution (review SSR)"); fig_save(fig, "fig_A_error_distribution")
sel = ["SSR argmax (PMF mode)", "SSR calibrated cutpoints", "Random (train prior)", [k for k in models if k.startswith("Majority")][0], "Sentiment-label lookup (leaky label)",
       "TF-IDF+LogReg text-only (train->eval)", "Aspect RF D1 (existing; TARGET LEAKAGE)"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, key in zip(axs, ["mae", "rmse"]):
    vals = [cmp.loc[k, key] for k in sel]; lo = [cmp.loc[k, key] - cis[k][key][0] for k in sel]; hi = [cis[k][key][1] - cmp.loc[k, key] for k in sel]
    ax.barh(range(len(sel)), vals, xerr=[np.clip(lo, 0, None), np.clip(hi, 0, None)], color=[C["ssr"], C["ssr"], C["base"], C["base"], C["alt"], C["alt"], C["bad"]])
    ax.set_yticks(range(len(sel)), [s.replace(" (", "\n(") for s in sel], fontsize=7); ax.set_xlabel(key.upper()); ax.invert_yaxis()
fig.suptitle("MAE / RMSE across models (human ratings, 95% cluster-bootstrap CI)"); fig_save(fig, "fig_A_mae_rmse_model_comparison")

# ================================================================ PART B: Digital-Persona SSR vs human Q21 (n=194)
print("== B: DP SSR vs human Q21")
from sentence_transformers import SentenceTransformer
import os; os.environ.setdefault("HF_HUB_OFFLINE", "1")
enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2"); emb = lambda xs: np.asarray(enc.encode(list(xs), show_progress_bar=False), float)
c = pd.read_csv(ROOT / "data/human_survey_cleaned.csv", keep_default_na=False, dtype=str)
hh = c[(c.in_primary_cohort == "True") & (c.text_value_rebuy_usable == "True")].reset_index(drop=True)
htexts = hh.text_value_rebuy.str.strip().tolist(); q21 = hh.rebuy_brand_1to5.astype(float).astype(int).values
BI = get_construct_anchors("BI").to_anchor_list(); AE = emb(BI); HE = emb(htexts)
HS = np.array([cosine_similarities(e, AE) for e in HE])
split = json.loads((ROOT / "data/validation_split/frozen_split_manifest.json").read_text())
ids = hh.respondent_id.tolist(); idx_of = {r: i for i, r in enumerate(ids)}
dev_i = np.array([idx_of[r] for r in split["dev_ids"] if r in idx_of]); test_i = np.array([idx_of[r] for r in split["test_ids"] if r in idx_of])


def ssr_pred(S, k=None, T=None):
    P = np.array([similarities_to_pmf(s) if k is None else improved_similarities_to_pmf(s, k=k, temperature=T) for s in S])
    return P, P @ np.arange(1, 6), P.argmax(1) + 1


PC, ec, ac = ssr_pred(HS); PI, ei, ai = ssr_pred(HS, 3, 0.75)
# ties in argmax resolve to first max (canonical convention np.argmax) - note in report
B = {"n": len(q21), "n_dev": len(dev_i), "n_test": len(test_i), "q21_dist": {int(k): int((q21 == k).sum()) for k in L},
     "ground_truth_note": "Q21 = self-reported likelihood to rebuy same brand; SSR scores the free-text Q26 answer against ADAPTED BI anchors (proxy construct)"}
def eval_set(idx, name):
    yy = q21[idx]; res = {}
    sets = {"Canonical SSR (argmax)": (ec[idx], ac[idx]), "Canonical SSR (rounded E[r])": (ec[idx], np.clip(np.round(ec[idx]), 1, 5).astype(int)),
            "Improved SSR k=3 T=0.75 (argmax)": (ei[idx], ai[idx]), "Improved SSR k=3 T=0.75 (rounded E[r])": (ei[idx], np.clip(np.round(ei[idx]), 1, 5).astype(int))}
    return yy, sets
rowsB = []
for nm, idx in [("full", np.arange(len(q21))), ("frozen_test", test_i), ("frozen_dev", dev_i)]:
    yy, sets = eval_set(idx, nm)
    maj = int(pd.Series(q21[dev_i] if nm != "full" else q21).mode().iloc[0])
    sets["Majority class"] = (np.full(len(yy), float(maj)), np.full(len(yy), maj))
    sets["Mean constant (fit on same eval y - optimistic)"] = (np.full(len(yy), yy.mean()), np.full(len(yy), int(round(yy.mean()))))
    for k, (pc_, pi_) in sets.items():
        m = reg_metrics(yy, pc_, pi_); m["model"] = k; m["split"] = nm; m["ci"] = boot(yy, pc_, pi_, n=1000, seed=2) if k != "Majority class" and "constant" not in k else None; rowsB.append(m)
B_df = pd.DataFrame(rowsB); save("B_human_q21_metrics", B_df.drop(columns=["ci"])); save("B_human_q21_bootstrap_ci", {f"{r['split']}|{r['model']}": r["ci"] for r in rowsB if r["ci"]})
# random baseline seeds (full)
prior = np.bincount(q21, minlength=6)[1:] / len(q21)
rr = [reg_metrics(q21, *(2 * [np.random.default_rng(s).choice(L, len(q21), p=prior)])) for s in range(100)]
B["random_prior_baseline_100_seeds_mean_sd"] = {k: [float(np.mean([m[k] for m in rr])), float(np.std([m[k] for m in rr]))] for k in ["exact_acc", "mae", "pearson"]}
# supervised text-only baselines via repeated stratified CV (class 1/2 tiny -> merge for stratification via 3-bin key)
strat = np.where(q21 <= 2, 0, q21 - 2)
def cv_text(model_fn, reps=10, folds=2):
    res = []
    for r in range(reps):
        skf = StratifiedKFold(folds, shuffle=True, random_state=r); P = np.zeros(len(q21), int); PCt = np.zeros(len(q21))
        for a, b in skf.split(htexts, strat):
            t = TfidfVectorizer(ngram_range=(1, 2)); m = model_fn().fit(t.fit_transform([htexts[i] for i in a]), q21[a]); pr_ = m.predict_proba(t.transform([htexts[i] for i in b]))
            P[b] = m.classes_[pr_.argmax(1)]; PCt[b] = pr_ @ m.classes_
        res.append(reg_metrics(q21, PCt, P))
    return res
tf_cv = cv_text(lambda: LogisticRegression(max_iter=2000, C=1.0))
B["tfidf_logreg_repeated_2fold_x10"] = {k: [float(np.mean([m[k] for m in tf_cv])), float(np.std([m[k] for m in tf_cv]))] for k in ["exact_acc", "pm1_acc", "mae", "rmse", "spearman"]}
# frozen split: train on dev, test on test
t = TfidfVectorizer(ngram_range=(1, 2)); lm = LogisticRegression(max_iter=2000).fit(t.fit_transform([htexts[i] for i in dev_i]), q21[dev_i]); pr_ = lm.predict_proba(t.transform([htexts[i] for i in test_i]))
B["tfidf_logreg_dev_to_test"] = reg_metrics(q21[test_i], pr_ @ lm.classes_, lm.classes_[pr_.argmax(1)])
# sentiment lexicon baseline (a-priori word lists, no fitting)
POS = set("good great love loved best excellent amazing happy satisfied smooth fast reliable nice perfect worth recommend premium better awesome fine comfortable".split())
NEG = set("bad poor worst hate slow lag laggy issue issues problem problems expensive disappointed disappointing heating hang hangs battery-drain overpriced switch change never".split())
def lex(txt):
    w = re.findall(r"[a-z\-]+", txt.lower()); p = sum(x in POS for x in w); n = sum(x in NEG for x in w); return 3 + np.clip(p - n, -2, 2) * 0.75
lx = np.array([lex(x) for x in htexts]); lxi = np.clip(np.round(lx), 1, 5).astype(int)
B["lexicon_sentiment_baseline_full"] = reg_metrics(q21, lx, lxi)
B["lexicon_sentiment_baseline_frozen_test"] = reg_metrics(q21[test_i], lx[test_i], lxi[test_i])
# paired bootstrap SSR vs mean baseline & vs lexicon (full)
def pb(pa, pbb, n=2000, seed=5):
    rng = np.random.default_rng(seed); d = np.abs(pa - q21) - np.abs(pbb - q21); r = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"mean_abs_err_diff(A-B)": float(d.mean()), "ci": list(np.percentile(r, [2.5, 97.5])), "p_two_sided": float(2 * min((np.array(r) >= 0).mean(), (np.array(r) <= 0).mean()))}
B["paired_abs_error_improved_vs_majority_full"] = pb(ei, np.full(len(q21), float(pd.Series(q21).mode().iloc[0])))
B["paired_abs_error_improved_vs_lexicon_full"] = pb(ei, lx)
B["paired_abs_error_improved_vs_canonical_full"] = pb(ei, ec)
# seed sensitivity of split-level metrics (SSR is deterministic; variance = split/sample)
sp = []
for s in range(50):
    from sklearn.model_selection import train_test_split as tts
    _, te = tts(np.arange(len(q21)), test_size=.5, random_state=s, stratify=strat); m = reg_metrics(q21[te], ei[te], ai[te]); sp.append(m)
B["improved_ssr_50_random_half_splits"] = {k: [float(np.mean([m[k] for m in sp])), float(np.std([m[k] for m in sp]))] for k in ["exact_acc", "mae", "spearman", "pearson"]}
# k / T sensitivity on human GT (exploratory, no model change)
grid = []
for k in [1, 2, 3, 4, 5]:
    for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        _, e_, a_ = ssr_pred(HS, k, T); m = reg_metrics(q21, e_, a_); grid.append({"k": k, "T": T, "exact_acc": m["exact_acc"], "mae": m["mae"], "rmse": m["rmse"], "spearman": m["spearman"], "pred_sd": m["pred_sd"], "r2": m["r2"]})
G = pd.DataFrame(grid); save("B_k_T_sensitivity_human_q21", G)
# calibration / uncertainty on human GT using PMF
def calib(P, yy, name):
    conf = P.max(1); pred = P.argmax(1) + 1; correct = (pred == yy).astype(float)
    bins = np.linspace(conf.min() - 1e-9, conf.max() + 1e-9, 6); ece = 0; rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mk = (conf >= lo) & (conf < hi)
        if mk.sum(): ece += mk.mean() * abs(conf[mk].mean() - correct[mk].mean()); rel.append((conf[mk].mean(), correct[mk].mean(), int(mk.sum())))
    onehot = np.eye(5)[yy - 1]; brier = float(((P - onehot) ** 2).sum(1).mean()); eps = 1e-3
    nll = float(-np.log(np.clip(P[np.arange(len(yy)), yy - 1], eps, 1)).mean())
    cov = {}
    for lvl in [.5, .7, .8, .9]:
        c_ = []
        for p_, y_ in zip(P, yy):
            o = np.argsort(-p_); cs = np.cumsum(p_[o]); n_ = int(np.searchsorted(cs, lvl) + 1); c_.append(y_ - 1 in o[:n_])
        cov[str(lvl)] = float(np.mean(c_))
    ent = float((-(np.where(P > 0, P * np.log2(np.where(P > 0, P, 1)), 0)).sum(1)).mean())
    return {"model": name, "ECE_top1": float(ece), "brier": brier, "nll_eps1e-3": nll, "set_coverage_at_nominal": cov, "mean_entropy_bits": ent,
            "uniform_nll_ref": float(np.log(5)), "pmf_zero_mass_share": float((P == 0).mean()), "reliability_bins": rel,
            "conf_vs_correct_spearman": float(stats.spearmanr(conf, correct)[0]), "conf_vs_abs_err_spearman": float(stats.spearmanr(conf, np.abs(P @ np.arange(1, 6) - yy))[0])}
B["calibration_canonical"] = calib(PC, q21, "canonical"); B["calibration_improved"] = calib(PI, q21, "improved k3 T.75")
# prior-only (train prior) calibration reference
Pprior = np.tile(prior, (len(q21), 1)); B["calibration_prior_reference"] = calib(Pprior, q21, "constant prior PMF")
RESULTS["B_human_q21"] = B
# confusion + per-rating on full for improved argmax
cmB = confusion_matrix(q21, ai, labels=L); save("B_confusion_improved_argmax_full", pd.DataFrame(cmB, index=L, columns=L).reset_index()); save("B_per_rating_improved_argmax_full", per_rating(q21, ai))
save("B_per_rating_canonical_argmax_full", per_rating(q21, ac))
cm_heatmap(cmB, "Improved SSR (k=3, T=0.75) vs human Q21, n=194", "fig_B_confusion_matrix_improved_vs_human")
cm_heatmap(confusion_matrix(q21, ac, labels=L), "Canonical SSR vs human Q21, n=194", "fig_B_confusion_matrix_canonical_vs_human")
fig, axs = plt.subplots(1, 2, figsize=(9, 4)); jit = np.random.default_rng(1).uniform(-.15, .15, len(q21))
for ax, e_, t_ in [(axs[0], ec, "Canonical"), (axs[1], ei, "Improved k=3,T=0.75")]:
    ax.scatter(q21 + jit, e_, s=14, alpha=.5, color=C["ssr"]); ax.plot([1, 5], [1, 5], "k--", lw=1); ax.plot(L, [e_[q21 == k].mean() for k in L], "o-", color=C["alt"])
    r = stats.spearmanr(e_, q21)[0]; ax.set_title(f"{t_} (Spearman {r:.2f})"); ax.set_xlabel("Human Q21 rating"); ax.set_ylabel("SSR E[r]")
fig_save(fig, "fig_B_actual_vs_predicted_human_q21")
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for i, (lab, vals, col) in enumerate([("Human Q21", np.bincount(q21, minlength=6)[1:] / len(q21), C["base"]), ("Canonical argmax", np.bincount(ac, minlength=6)[1:] / len(q21), C["alt"]),
                                      ("Improved argmax", np.bincount(ai, minlength=6)[1:] / len(q21), C["ssr"])]):
    ax.bar(np.array(L) + (i - 1) * .27, vals * 100, .27, label=lab, color=col)
ax.set_xlabel("Rating"); ax.set_ylabel("% of respondents"); ax.legend(frameon=False); ax.set_title("Rating distribution: human vs SSR (n=194)"); fig_save(fig, "fig_B_rating_distribution_human_vs_ssr")
fig, ax = plt.subplots(figsize=(6.2, 3.8)); ax.hist(ei - q21, bins=30, color=C["ssr"], alpha=.8, label="Improved"); ax.hist(ec - q21, bins=30, color=C["alt"], alpha=.5, label="Canonical")
ax.axvline(0, color="k", lw=1); ax.set_xlabel("E[r] - Q21"); ax.legend(frameon=False); ax.set_title("Error distribution vs human Q21"); fig_save(fig, "fig_B_error_distribution")
piv = G.pivot(index="k", columns="T", values="mae")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, key, cm_ in [(axs[0], "mae", "viridis_r"), (axs[1], "spearman", "viridis")]:
    pv = G.pivot(index="k", columns="T", values=key); im = ax.imshow(pv.values, cmap=cm_, aspect="auto")
    for i_, j_ in itertools.product(range(pv.shape[0]), range(pv.shape[1])): ax.text(j_, i_, f"{pv.values[i_, j_]:.2f}", ha="center", va="center", fontsize=7, color="w")
    ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index); ax.set_xlabel("T"); ax.set_ylabel("k"); ax.set_title(f"{key.upper()} vs human Q21 (n=194)"); fig.colorbar(im, ax=ax)
fig_save(fig, "fig_B_k_T_sensitivity_heatmaps")
# PMF visualisation: mean PMF per actual Q21 rating (improved) + confidence reliability
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8)); mp = np.array([PI[q21 == k].mean(0) for k in L]); im = axs[0].imshow(mp, cmap="Blues", vmin=0, vmax=.6)
for i_, j_ in itertools.product(range(5), range(5)): axs[0].text(j_, i_, f"{mp[i_, j_]:.2f}", ha="center", va="center", fontsize=8)
axs[0].set_xticks(range(5), L); axs[0].set_yticks(range(5), L); axs[0].set_xlabel("PMF category"); axs[0].set_ylabel("Actual Q21"); axs[0].set_title("Mean Improved-SSR PMF by actual rating")
cal = B["calibration_improved"]["reliability_bins"]; axs[1].plot([0, 1], [0, 1], "k--"); axs[1].plot([r[0] for r in cal], [r[1] for r in cal], "o-", color=C["ssr"])
axs[1].set_xlabel("PMF top-1 confidence"); axs[1].set_ylabel("observed accuracy"); axs[1].set_title("Reliability (improved, n=194)"); fig_save(fig, "fig_B_pmf_and_reliability")

# ================================================================ PART C: synthetic persona responses (NO ground truth)
print("== C: synthetic persona responses")
S = pd.read_csv(ROOT / "data/experiments/persona_validation_400/canonical_vs_improved_comparison/canonical_vs_improved_responses.csv")
sim = S[[f"Sim_Anchor_{i}" for i in L]].values
Pc = np.array([similarities_to_pmf(s) for s in sim]); Pi = np.array([improved_similarities_to_pmf(s, 3, 0.75) for s in sim])
Cx = {"n": len(S), "per_persona_n": S.Persona_ID.value_counts().sort_index().to_dict(), "per_construct_n": S.Construct.value_counts().to_dict()}
Cx["reproduction_check"] = {"canonical_E_max_abs_diff_vs_saved": float(np.abs(Pc @ np.arange(1, 6) - S.Canonical_Expected_Rating).max()),
                            "improved_E_max_abs_diff_vs_saved": float(np.abs(Pi @ np.arange(1, 6) - S.Improved_Expected_Rating).max())}
def pmf_diag(P, name):
    E = P @ np.arange(1, 6); sd = np.sqrt((P * (np.arange(1, 6) - E[:, None]) ** 2).sum(1)); ent = -(np.where(P > 0, P * np.log2(np.where(P > 0, P, 1)), 0)).sum(1)
    return {"model": name, "pmf_valid": bool(np.allclose(P.sum(1), 1) and (P >= 0).all()), "E_mean": float(E.mean()), "E_sd": float(E.std(ddof=1)), "E_min": float(E.min()), "E_max": float(E.max()),
            "share_E_ge_4.5": float((E >= 4.5).mean()), "share_E_le_1.5": float((E <= 1.5).mean()), "mean_within_pmf_sd": float(sd.mean()), "mean_entropy_bits": float(ent.mean()),
            "share_zero_prob_cells": float((P == 0).mean()), "argmax_dist": {int(k): float((P.argmax(1) + 1 == k).mean()) for k in L},
            "share_argmax_endpoints": float(np.isin(P.argmax(1) + 1, [1, 5]).mean()), "pmf_mean": P.mean(0).tolist()}
Cx["pmf_canonical"] = pmf_diag(Pc, "canonical"); Cx["pmf_improved"] = pmf_diag(Pi, "improved k3 T.75")
hum_all = np.concatenate([pd.to_numeric(pd.read_csv(ROOT / "data/clean/human_survey_clean.csv")[col], errors="coerce").dropna().values for col in ["rebuy_brand_1to5", "tech_comfort_1to5"]])
Cx["human_pooled_q20_q21"] = {"n": len(hum_all), "mean": float(hum_all.mean()), "sd": float(hum_all.std(ddof=1)), "share_endpoints": float(np.isin(hum_all, [1, 5]).mean())}
Cx["variance_ratio_synthetic_to_human"] = {"canonical": float(Cx["pmf_canonical"]["E_sd"] / hum_all.std(ddof=1)), "improved": float(Cx["pmf_improved"]["E_sd"] / hum_all.std(ddof=1))}
# distribution vs human (BI<->Q21, EE<->Q20): DISTRIBUTIONAL only
Hc = pd.read_csv(ROOT / "data/clean/human_survey_clean.csv"); dist_rows = []
for cons, col in [("BI", "rebuy_brand_1to5"), ("EE", "tech_comfort_1to5")]:
    hv = pd.to_numeric(Hc[Hc.in_primary_cohort == True][col], errors="coerce").dropna().values
    for nm, P in [("canonical", Pc), ("improved", Pi)]:
        sv = (P @ np.arange(1, 6))[S.Construct.values == cons]
        dist_rows.append({"construct": cons, "ssr": nm, "n_syn": len(sv), "n_hum": len(hv), "syn_mean": sv.mean(), "hum_mean": hv.mean(), "mean_diff": sv.mean() - hv.mean(),
                          "cohens_d": (sv.mean() - hv.mean()) / np.sqrt(((len(sv) - 1) * sv.var(ddof=1) + (len(hv) - 1) * hv.var(ddof=1)) / (len(sv) + len(hv) - 2)),
                          "wasserstein_W1": stats.wasserstein_distance(sv, hv), "ks_p": stats.ks_2samp(sv, hv)[1], "sd_ratio_syn_over_hum": sv.std(ddof=1) / hv.std(ddof=1)})
save("C_synthetic_vs_human_distribution_NOT_ACCURACY", pd.DataFrame(dist_rows))
# persona differentiation
S["E_can"] = Pc @ np.arange(1, 6); S["E_imp"] = Pi @ np.arange(1, 6)
def hedges(a, b):
    na, nb = len(a), len(b); sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)); d = (a.mean() - b.mean()) / sp
    d *= 1 - 3 / (4 * (na + nb) - 9); se = np.sqrt((na + nb) / (na * nb) + d ** 2 / (2 * (na + nb))); return d, d - 1.96 * se, d + 1.96 * se
def eta_ci(v, g, n=1000, seed=0):
    rng = np.random.default_rng(seed); gs = np.asarray(g)
    def eta(vv, gg):
        grand = vv.mean(); ssb = sum(((vv[gg == k].mean() - grand) ** 2) * (gg == k).sum() for k in np.unique(gg)); return ssb / ((vv - grand) ** 2).sum()
    pt = eta(v, gs); bs = [eta(v[ix], gs[ix]) for ix in (rng.integers(0, len(v), len(v)) for _ in range(n))]; return pt, np.percentile(bs, [2.5, 97.5])
prow, pair_rows = [], []
for nm, col in [("canonical", "E_can"), ("improved", "E_imp")]:
    for cons in ["ALL"] + sorted(S.Construct.unique()):
        d = S if cons == "ALL" else S[S.Construct == cons]
        groups = [d[d.Persona_ID == p][col].values for p in sorted(d.Persona_ID.unique())]
        F, pF = stats.f_oneway(*groups); H, pH = stats.kruskal(*groups); et, eci = eta_ci(d[col].values, d.Persona_ID.values)
        prow.append({"ssr": nm, "construct": cons, "n": len(d), "ANOVA_F": F, "ANOVA_p": pF, "Kruskal_H": H, "Kruskal_p": pH, "eta_sq": et, "eta_ci_lo": eci[0], "eta_ci_hi": eci[1], "levene_p": stats.levene(*groups)[1]})
        ps = []
        for a, b in itertools.combinations(sorted(d.Persona_ID.unique()), 2):
            x, yv = d[d.Persona_ID == a][col].values, d[d.Persona_ID == b][col].values; t_, p_ = stats.ttest_ind(x, yv, equal_var=False); g_, lo, hi = hedges(x, yv)
            ps.append({"ssr": nm, "construct": cons, "pair": f"{a}-{b}", "n_a": len(x), "n_b": len(yv), "mean_diff": x.mean() - yv.mean(), "welch_p": p_, "mannwhitney_p": stats.mannwhitneyu(x, yv)[1], "hedges_g": g_, "g_lo": lo, "g_hi": hi})
        order = np.argsort([q["welch_p"] for q in ps]); m = len(ps)
        for rank, i in enumerate(order): ps[i]["welch_p_holm"] = min(1, max(ps[j]["welch_p"] * (m - r_) for r_, j in enumerate(order[:rank + 1])))
        pair_rows += ps
save("C_persona_omnibus_tests", pd.DataFrame(prow)); save("C_persona_pairwise_tests", pd.DataFrame(pair_rows))
# persona text-embedding classifiability (is persona identifiable from text?)
SE = emb(S.Response_Text.tolist()); np.save(OUT / "metrics" / "synthetic_response_embeddings_mpnet.npy", SE)
yp = S.Persona_ID.values; accs = []
for r in range(5):
    skf = StratifiedKFold(5, shuffle=True, random_state=r); pr_ = np.empty(len(S), object)
    for a, b in skf.split(SE, yp): pr_[b] = LogisticRegression(max_iter=2000).fit(SE[a], yp[a]).predict(SE[b])
    accs.append(accuracy_score(yp, pr_))
perm = []
for r in range(100):
    yperm = np.random.default_rng(r).permutation(yp); skf = StratifiedKFold(5, shuffle=True, random_state=0); pr_ = np.empty(len(S), object)
    for a, b in skf.split(SE, yperm): pr_[b] = LogisticRegression(max_iter=2000).fit(SE[a], yperm[a]).predict(SE[b])
    perm.append(accuracy_score(yperm, pr_))
Cx["persona_identifiability_from_text"] = {"cv_acc_mean": float(np.mean(accs)), "cv_acc_sd": float(np.std(accs)), "chance": .25, "perm_null_mean": float(np.mean(perm)), "perm_p": float((np.sum(np.array(perm) >= np.mean(accs)) + 1) / 101)}
# confound check: construct balance across personas
ct = pd.crosstab(S.Persona_ID, S.Construct); Cx["persona_x_construct_counts"] = ct.to_dict(); Cx["construct_persona_chi2_p"] = float(stats.chi2_contingency(ct)[1])
# construct-level 'performance' of SSR: correct-construct anchor alignment. Mean cosine of response to each construct's anchor set
cons_anch = {}
for cc in ["PE", "EE", "HM", "SI", "PP", "BI"]:
    try: cons_anch[cc] = emb(get_construct_anchors(cc).to_anchor_list())
    except Exception as e: Cx.setdefault("anchor_load_errors", {})[cc] = str(e)
keys = list(cons_anch); M = np.array([[np.mean(cosine_similarities(e, cons_anch[k])) for k in keys] for e in SE])
S["nearest_construct_anchor_set"] = [keys[i] for i in M.argmax(1)]
Cx["construct_alignment_nearest_anchor_set_acc"] = float((S.nearest_construct_anchor_set == S.Construct).mean()); Cx["construct_alignment_chance"] = 1 / len(keys)
Cx["construct_alignment_by_construct"] = S.groupby("Construct").apply(lambda d: (d.nearest_construct_anchor_set == d.Construct).mean()).to_dict()
# diversity and quality
def distinct(texts, n):
    g = [tuple(w[i:i + n]) for w in (t.lower().split() for t in texts) for i in range(len(w) - n + 1)]; return len(set(g)) / max(1, len(g))
cs = SE @ SE.T / np.outer(np.linalg.norm(SE, axis=1), np.linalg.norm(SE, axis=1)); iu = np.triu_indices(len(S), 1)
same_p = (yp[:, None] == yp[None, :])[iu]; same_c = (S.Construct.values[:, None] == S.Construct.values[None, :])[iu]
first3 = S.Response_Text.str.lower().str.split().str[:3].str.join(" ").value_counts()
sent_split = S.Response_Text.str.split(r"(?<=[.!?])\s+")
Cx["diversity"] = {"unique_texts": int(S.Response_Text.nunique()), "distinct_1": distinct(S.Response_Text, 1), "distinct_2": distinct(S.Response_Text, 2), "distinct_3": distinct(S.Response_Text, 3),
                   "mean_pairwise_cosine": float(cs[iu].mean()), "pairs_cos_gt_0.90": int((cs[iu] > .9).sum()), "pairs_cos_gt_0.95": int((cs[iu] > .95).sum()),
                   "mean_cos_same_persona": float(cs[iu][same_p].mean()), "mean_cos_diff_persona": float(cs[iu][~same_p].mean()), "mean_cos_same_construct": float(cs[iu][same_c].mean()), "mean_cos_diff_construct": float(cs[iu][~same_c].mean()),
                   "top_repeated_openings_3words": first3.head(8).to_dict(), "mean_words": float(S.Response_Text.str.split().str.len().mean()), "mean_sentences": float(sent_split.str.len().mean()),
                   "sentence_dup_rate": float(1 - pd.Series([s for l_ in sent_split for s in l_]).nunique() / sum(len(l_) for l_ in sent_split))}
dup_idx = np.argwhere(np.triu(cs > .93, 1))
Cx["near_duplicate_examples"] = [{"a": S.Sample_ID[i], "b": S.Sample_ID[j], "cos": float(cs[i, j]), "text_a": S.Response_Text[i], "text_b": S.Response_Text[j]} for i, j in dup_idx[:5]]
RATING_RE = re.compile(r"(\b\d+(\.\d+)?\s*/\s*\d+\b)|(\b\d+(\.\d+)?\s*(out of|stars?)\b)|(\b(rate|rating|score)\w*\s*[:=]?\s*\d)|(\b[1-5]\s*(/|out of)\s*5\b)", re.I)
Cx["direct_numeric_rating_in_text"] = {"regex_hits": int(S.Response_Text.str.contains(RATING_RE).sum()), "any_digit_responses": int(S.Response_Text.str.contains(r"\d").sum()),
                                       "star_or_rating_words": int(S.Response_Text.str.contains(r"\b(stars?|rating|rated|score|out of)\b", case=False).sum()),
                                       "digit_examples": S[S.Response_Text.str.contains(r"\d")].Response_Text.head(5).tolist()}
POSW = set("great excellent love amazing fantastic perfect smooth fast reliable impressive solid good happy worth".split()); NEGW = set("poor bad terrible slow lag laggy disappointing disappointed overpriced expensive issues problems issue throttling stutter stuttering heating overheating drain".split())
def mixed(t):
    w = set(re.findall(r"[a-z]+", t.lower())); return bool(w & POSW) and bool(w & NEGW)
S["mixed_polarity_flag"] = S.Response_Text.map(mixed); Cx["contradiction_screen"] = {"mixed_polarity_share": float(S.mixed_polarity_flag.mean()), "note": "lexical screen only; mixed polarity is often legitimate nuance, not contradiction"}
Cx["mixed_polarity_examples"] = S[S.mixed_polarity_flag].Response_Text.head(4).tolist()
# persona consistency: persona-stereotype keyword lexicons (premium / budget-student)
PREM = re.compile(r"\b(flagship|premium|high-end|long-term|reliab\w+|build quality)\b", re.I); BUD = re.compile(r"\b(afford\w*|budget|price|cheap\w*|student\w*|value|cost|pocket|expens\w+)\b", re.I)
S["prem_kw"] = S.Response_Text.str.count(PREM); S["bud_kw"] = S.Response_Text.str.count(BUD)
Cx["persona_keyword_rates_per_response"] = S.groupby("Persona_ID")[["prem_kw", "bud_kw"]].mean().to_dict()
kw_p = stats.kruskal(*[S[S.Persona_ID == p].bud_kw.values for p in sorted(S.Persona_ID.unique())])[1]; Cx["budget_keyword_persona_kruskal_p"] = float(kw_p)
# SSR-vs-text consistency within synthetic: E[r] vs lexicon polarity (convergent validity, not accuracy)
S["lex"] = [lex(t) for t in S.Response_Text]
Cx["convergent_validity_E_vs_lexicon_spearman"] = {"canonical": float(stats.spearmanr(S.E_can, S.lex)[0]), "improved": float(stats.spearmanr(S.E_imp, S.lex)[0]), "note": "agreement between two automatic scorers; NOT accuracy against humans"}
# k/T sensitivity on synthetic (no GT): variance, endpoints, persona effect, rank agreement with default
sens = []
base_E = S.E_imp.values
for k in [1, 2, 3, 4, 5]:
    for T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        P = np.array([improved_similarities_to_pmf(s, k, T) for s in sim]); E = P @ np.arange(1, 6)
        et, _ = eta_ci(E, S.Persona_ID.values, n=1); pe = S.Construct == "PE"
        sens.append({"k": k, "T": T, "E_mean": E.mean(), "E_sd": E.std(ddof=1), "share_E_ge_4.5_or_le_1.5": float(((E >= 4.5) | (E <= 1.5)).mean()), "mean_entropy_bits": float((-(np.where(P > 0, P * np.log2(np.where(P > 0, P, 1)), 0)).sum(1)).mean()),
                     "persona_eta_sq_all": et, "spearman_vs_default": float(stats.spearmanr(E, base_E)[0]) if E.std() > 0 else np.nan})
SG = pd.DataFrame(sens); save("C_k_T_sensitivity_synthetic", SG)
# extreme compression table by rating bins
Cx["rating_bin_share"] = {nm: {f"[{a},{b})": float(((E >= a) & (E < b)).mean()) for a, b in [(1, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5), (4.5, 5.01)]} for nm, E in [("canonical", S.E_can.values), ("improved", S.E_imp.values)]}
# persona x construct heatmap
for nm, col in [("improved", "E_imp"), ("canonical", "E_can")]:
    pv = S.pivot_table(index="Persona_ID", columns="Construct", values=col, aggfunc="mean")[["PE", "EE", "HM", "SI", "PP", "BI"]]; pv.reset_index().to_csv(OUT / "metrics" / f"C_persona_x_construct_mean_E_{nm}.csv", index=False)
pv = S.pivot_table(index="Persona_ID", columns="Construct", values="E_imp", aggfunc="mean")[["PE", "EE", "HM", "SI", "PP", "BI"]]
cn = S.pivot_table(index="Persona_ID", columns="Construct", values="E_imp", aggfunc="count")[["PE", "EE", "HM", "SI", "PP", "BI"]]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6)); im = axs[0].imshow(pv.values, cmap="RdYlBu", vmin=2.5, vmax=4.5)
for i_, j_ in itertools.product(range(4), range(6)): axs[0].text(j_, i_, f"{pv.values[i_, j_]:.2f}\n(n={cn.values[i_, j_]})", ha="center", va="center", fontsize=7)
axs[0].set_xticks(range(6), pv.columns); axs[0].set_yticks(range(4), pv.index); axs[0].set_title("Mean Improved-SSR E[r] (synthetic; no ground truth)"); fig.colorbar(im, ax=axs[0])
om = pd.DataFrame(prow); om = om[(om.ssr == "improved") & (om.construct != "ALL")]; axs[1].bar(om.construct, om.eta_sq, yerr=[(om.eta_sq - om.eta_ci_lo).clip(lower=0), (om.eta_ci_hi - om.eta_sq).clip(lower=0)], color=C["ssr"], capsize=3)
axs[1].axhline(.01, color=C["base"], ls=":"); axs[1].axhline(.06, color=C["base"], ls="--"); axs[1].set_ylabel("eta-squared (persona effect), 95% CI"); axs[1].set_title("Persona differentiation by construct")
for i_, (cn_, p_) in enumerate(zip(om.construct, om.ANOVA_p)): axs[1].text(i_, 0.005, f"p={p_:.3f}", ha="center", fontsize=7, color="w")
fig_save(fig, "fig_C_persona_x_construct_heatmap_and_effect_sizes")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8)); ex = S.sample(6, random_state=3).index
for ax, P, t_ in [(axs[0], Pc, "Canonical"), (axs[1], Pi, "Improved k=3,T=0.75")]:
    for i_ in ex: ax.plot(L, P[i_], "o-", alpha=.8, label=S.Sample_ID[i_][4:])
    ax.set_xlabel("Rating"); ax.set_ylabel("P(rating)"); ax.set_title(f"{t_}: example PMFs"); ax.set_xticks(L)
axs[1].legend(frameon=False, fontsize=6, ncol=2); fig_save(fig, "fig_C_example_pmfs")
fig, ax = plt.subplots(figsize=(6.2, 3.8)); bins = np.linspace(1, 5, 33)
ax.hist(hum_all, bins=np.arange(.5, 6), density=True, alpha=.35, color=C["base"], label="Human Q20/Q21 (pooled)"); ax.hist(S.E_can, bins=bins, density=True, histtype="step", lw=2, color=C["alt"], label="Canonical E[r]"); ax.hist(S.E_imp, bins=bins, density=True, histtype="step", lw=2, color=C["ssr"], label="Improved E[r]")
ax.set_xlabel("rating / E[r]"); ax.legend(frameon=False, fontsize=8); ax.set_title("Variance compression (synthetic vs human; distributional)"); fig_save(fig, "fig_C_variance_compression")
RESULTS["C_synthetic"] = Cx
S.drop(columns=["lex"]).to_csv(OUT / "metrics" / "C_synthetic_scored_with_flags.csv", index=False)

# ================================================================ PART D: product-evaluation outputs (relevance to concept)
print("== D: product evaluation relevance")
Dv = {"runs": 0}
rows_d = []
for run in sorted((ROOT / "data/product_evaluation").glob("*/")):
    pj, rj = run / "product_concept.json", run / "persona_responses.json"
    if not (pj.exists() and rj.exists()): continue
    try: pc_ = json.loads(pj.read_text(encoding="utf-8")); rs_ = json.loads(rj.read_text(encoding="utf-8"))
    except Exception: continue
    name = pc_.get("name", ""); desc = pc_.get("description", "") + " " + " ".join(map(str, pc_.get("features", []))) if isinstance(pc_.get("features", []), list) else pc_.get("description", "")
    for r in rs_:
        txts = {"overall": r.get("response_text", "")}; txts.update({f"c_{k}": v for k, v in (r.get("construct_responses") or {}).items()})
        for k, t_ in txts.items(): rows_d.append({"run": run.name, "product": name, "category": pc_.get("category", ""), "persona": r.get("persona_id"), "slot": k, "text": t_, "concept": desc})
D = pd.DataFrame(rows_d)
if len(D):
    D = D[D.text.astype(str).str.len() > 0].reset_index(drop=True)
    DE = emb(D.text.tolist()); CE = emb(D.concept.tolist()); D["cos_to_own_concept"] = (DE * CE).sum(1) / (np.linalg.norm(DE, axis=1) * np.linalg.norm(CE, axis=1))
    shuf = np.random.default_rng(0).permutation(len(D)); D["cos_to_random_concept"] = (DE * CE[shuf]).sum(1) / (np.linalg.norm(DE, axis=1) * np.linalg.norm(CE[shuf], axis=1))
    D["product_name_mentioned"] = [bool(n and n.lower().split()[0] in t.lower()) for n, t in zip(D["product"], D.text)]
    D["says_smartphone_for_non_phone"] = D.text.str.contains(r"\b(smartphone|phone)\b", case=False) & ~D.category.str.contains("phone", case=False, na=False) & ~D["product"].str.contains("phone|5G|Pro Max|Ultra|Flagship", case=False, na=False)
    Dv = {"runs": int(D.run.nunique()), "responses": len(D), "categories": D.category.value_counts().head(8).to_dict(), "mean_cos_own": float(D.cos_to_own_concept.mean()), "mean_cos_random": float(D.cos_to_random_concept.mean()),
          "wilcoxon_own_vs_random_p": float(stats.wilcoxon(D.cos_to_own_concept, D.cos_to_random_concept)[1]), "product_name_mention_rate": float(D.product_name_mentioned.mean()),
          "smartphone_word_in_non_phone_product_rate": float(D.says_smartphone_for_non_phone.sum() / max(1, (~D.category.str.contains("phone", case=False, na=False)).sum())),
          "examples_smartphone_text_for_non_phone": D[D.says_smartphone_for_non_phone][["product", "text"]].head(4).to_dict("records"), "unique_text_share": float(D.text.nunique() / len(D))}
    D.drop(columns=["concept"]).to_csv(OUT / "metrics" / "D_product_evaluation_relevance.csv", index=False)
RESULTS["D_product_eval"] = Dv

# ---------------------------------------------------------------- write everything
save("ALL_RESULTS", RESULTS)
cmp_out = cmp.reset_index(); cmp_out.insert(1, "mae_grade", cmp_out.mae.map(lambda v: grade("mae", v)))
print("DONE")
print(cmp[["exact_acc", "pm1_acc", "mae", "rmse", "pearson", "spearman", "r2"]].round(3).to_string())
