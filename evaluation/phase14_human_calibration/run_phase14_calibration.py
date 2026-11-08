"""Phase 14: human-fitted E[r] -> Likert thresholds (EXPERIMENTAL calibrated SSR). Canonical/Improved SSR untouched;
input = stored Phase 13 E[r] (Improved SSR k=3 T=0.75, BI anchors, nomic). No LLM / embedding calls.
Thresholds are fit ONLY on each training split and evaluated on the held-out split. 20 stratified 50/50 splits.
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import ssr_hybrid_experiments as H          # metrics (read-only reuse)
import ssr_experimental_decisions as X      # Phase 7 OrdinalCutpoints.predict (threshold rule)

BASE_SEED, N_SPLITS, L = 20261105, 20, np.arange(1, 6)
D = pd.read_csv(ROOT / "evaluation/phase13_human_ssr/phase13_scored_human_q26.csv")
D = D[D.category == "meaningful_3plus"].reset_index(drop=True)
E_all, y_all = D.imp_E.values, D.q21.values.astype(int)


def qwk_fast(y, p):
    O = np.zeros((5, 5)); np.add.at(O, (y - 1, p - 1), 1)
    W = (np.subtract.outer(L, L) ** 2) / 16.0
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return 1 - (W * O).sum() / (W * Ex).sum() if (W * Ex).sum() > 0 else 0.0


def predict(E, t):
    return 1 + (np.asarray(E)[:, None] >= np.asarray(t)[None, :]).sum(1)


def fit_qwk_cutpoints(E, y, n_grid=14):
    """Phase 7 objective (max QWK, tie: lower MAE, then closest to rounding cutpoints) over all 4 thresholds,
    searched on a quantile grid of TRAIN E[r] (all 5 classes present)."""
    grid = np.unique(np.quantile(E, np.linspace(0, 1, n_grid)))
    grid = np.concatenate([[E.min() - 1e-6], (grid[:-1] + grid[1:]) / 2, [E.max() + 1e-6]])
    default = np.array([1.5, 2.5, 3.5, 4.5]); best, key = default, None
    for t in itertools.combinations_with_replacement(grid, 4):
        p = predict(E, t); k = (-qwk_fast(y, p), float(np.abs(p - y).mean()), float(np.abs(np.array(t) - default).sum()))
        if key is None or k < key:
            best, key = np.array(t), k
    return tuple(float(x) for x in best)


def fit_quantile_match(E, y):
    """Thresholds at TRAIN E[r] quantiles equal to the TRAIN cumulative Q21 proportions (distribution matching)."""
    cum = np.cumsum(np.bincount(y, minlength=6)[1:]) / len(y)
    return tuple(float(np.quantile(E, c)) for c in cum[:4])


METHODS = {"A_baseline_rounded_E (original SSR)": lambda Etr, ytr: (1.5, 2.5, 3.5, 4.5),
           "B_phase7_QWK_cutpoints (fit on train)": fit_qwk_cutpoints,
           "C_quantile_matched_thresholds (fit on train)": fit_quantile_match}
KEYS = ["exact", "within1", "MAE", "RMSE", "spearman", "qwk", "lwk", "macro_P", "macro_R", "macro_F1", "weighted_P", "weighted_R", "weighted_F1"]
rows, conf, thr, preds_dist, bias = [], {m: np.zeros((5, 5), int) for m in METHODS}, {m: [] for m in METHODS}, {m: np.zeros(5, int) for m in METHODS}, {m: [] for m in METHODS}
for s in range(N_SPLITS):
    tr, te = train_test_split(np.arange(len(D)), test_size=0.5, random_state=BASE_SEED + s, stratify=y_all)
    assert not set(tr) & set(te)
    for m, fit in METHODS.items():
        t = fit(E_all[tr], y_all[tr]); thr[m].append(t)
        p = X.OrdinalCutpoints(t).predict(E_all[te]); y = y_all[te]
        mt = H.metrics(y, p, p)
        rows.append({"split": s, "method": m, **{k: mt[k] for k in KEYS}, **{f"R{c}": mt["per_class"][c]["R"] for c in L},
                     **{f"n_pred_{c}": mt["per_class"][c]["n_pred"] for c in L}, "mean_bias_pred_minus_true": float((p - y).mean())})
        conf[m] += np.array(mt["confusion"]); preds_dist[m] += np.bincount(p, minlength=6)[1:]
R = pd.DataFrame(rows); R.to_csv(OUT / "phase14_per_split_metrics.csv", index=False)
cols = KEYS + [f"R{c}" for c in L] + ["mean_bias_pred_minus_true"]
summary = R.groupby("method")[cols].agg(["mean", "std"]).round(4)
# paired differences vs baseline across splits
base = R[R.method.str.startswith("A_")].set_index("split")
paired = {}
for m in METHODS:
    if m.startswith("A_"):
        continue
    cur = R[R.method == m].set_index("split")
    paired[m] = {k: {"mean_diff": float((cur[k] - base[k]).mean()), "sd": float((cur[k] - base[k]).std()),
                     "splits_better": int(((cur[k] - base[k]) > 1e-12).sum()) if k not in ("MAE", "RMSE") else int(((cur[k] - base[k]) < -1e-12).sum()),
                     "splits_worse": int(((cur[k] - base[k]) < -1e-12).sum()) if k not in ("MAE", "RMSE") else int(((cur[k] - base[k]) > 1e-12).sum())}
                 for k in ["exact", "qwk", "macro_F1", "macro_R", "MAE", "R1", "R5", "spearman"]}
out = {"label": "Phase 14 EXPERIMENTAL calibrated SSR vs original SSR on HUMAN Q21; 142 meaningful Q26 responses; 20 stratified 50/50 splits",
       "n": int(len(D)), "class_counts": {int(k): int(v) for k, v in pd.Series(y_all).value_counts().sort_index().items()},
       "test_set_size": int(len(D) - len(D) // 2), "seeds": [BASE_SEED + s for s in range(N_SPLITS)],
       "summary_mean_sd": {m: {c: [float(summary.loc[m, (c, "mean")]), float(summary.loc[m, (c, "std")])] for c in cols} for m in METHODS},
       "paired_vs_baseline": paired,
       "pooled_test_confusion_rows_true_cols_pred": {m: conf[m].tolist() for m in METHODS},
       "pooled_test_pred_distribution": {m: preds_dist[m].tolist() for m in METHODS},
       "true_distribution_pooled_test": np.bincount(np.concatenate([y_all[train_test_split(np.arange(len(D)), test_size=.5, random_state=BASE_SEED + s, stratify=y_all)[1]] for s in range(N_SPLITS)]), minlength=6)[1:].tolist(),
       "thresholds_mean_sd": {m: [np.mean(thr[m], 0).round(3).tolist(), np.std(thr[m], 0).round(3).tolist()] for m in METHODS},
       "spearman_note": "computed on discrete predictions (calibration is monotone, so continuous E[r] Spearman is unchanged by design)"}
(OUT / "phase14_metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
pd.set_option("display.width", 250)
print(summary.to_string()); print(json.dumps({k: out[k] for k in ["class_counts", "paired_vs_baseline", "pooled_test_confusion_rows_true_cols_pred", "pooled_test_pred_distribution", "true_distribution_pooled_test", "thresholds_mean_sd"]}, indent=1, default=float))
