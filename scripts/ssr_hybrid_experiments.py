"""Phase 8: hybrid SSR decision rules (experimental). Inputs are existing Improved-SSR PMFs (k=3, T=0.75, nomic-embed-text);
canonical_ssr.py / improved_ssr.py are not modified or re-run. No LLM calls.

Leakage protocol
  TEST  = 24 improved Nova responses (Phase 6). Never used for any choice.
  CAL   = 48 disjoint responses with the same SSR config and the same blind-judge protocol
          (24 baseline Nova + 24 quick-validation). ALL selections (per-construct rule, gate/threshold values,
          calibration sufficiency) are made on CAL only. Pre-specified variants without free choices are all reported.
  "test-oracle" rows pick the best option ON TEST; they are leaky diagnostics that quantify optimism, never candidates.

Run: python scripts/ssr_hybrid_experiments.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RATINGS = np.arange(1, 6)
LOG2_5 = float(np.log2(5))


# ------------------------------------------------------------------ primitives (deterministic)
def round_half_up(x) -> np.ndarray:
    return np.clip(np.floor(np.asarray(x, float) + 0.5), 1, 5).astype(int)


def expected(P) -> np.ndarray:
    return np.asarray(P, float) @ RATINGS


def pmf_median(P) -> np.ndarray:
    cdf = np.cumsum(np.asarray(P, float), axis=1)
    return (cdf >= 0.5 - 1e-12).argmax(axis=1) + 1


def confidence_entropy(P) -> np.ndarray:
    """1 - H(p)/log2(5) in [0, 1]; 1 = all mass on one rating. Uses only the SSR PMF."""
    P = np.asarray(P, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.where(P > 0, P * np.log2(P), 0.0).sum(axis=1)
    return 1.0 - H / LOG2_5


def confidence_maxprob(P) -> np.ndarray:
    """(max p - 1/5) / (1 - 1/5) in [0, 1]. Uses only the SSR PMF."""
    return (np.asarray(P, float).max(axis=1) - 0.2) / 0.8


# ------------------------------------------------------------------ hybrid rules: f(P) -> integer ratings
def rule_rounded_E(P) -> np.ndarray:
    return round_half_up(expected(P))


def rule_median(P) -> np.ndarray:
    return pmf_median(P)


def rule_weighted(P, w_E: float) -> np.ndarray:
    """A: round_half_up(w_E * E[r] + (1 - w_E) * median)."""
    return round_half_up(w_E * expected(P) + (1 - w_E) * pmf_median(P))


def rule_integer_average(P) -> np.ndarray:
    """A: average of round(E) and median; a .5 tie goes to the integer closer to E[r] (exact tie -> round(E))."""
    E, r, m = expected(P), rule_rounded_E(P), pmf_median(P)
    avg = (r + m) / 2.0
    lo, hi = np.floor(avg).astype(int), np.ceil(avg).astype(int)
    pick = np.where(np.abs(hi - E) < np.abs(lo - E), hi, lo)
    return np.where(avg == np.round(avg), avg.astype(int), np.where(np.abs(hi - E) == np.abs(lo - E), r, pick))


def rule_upper_gate(P, tau: float) -> np.ndarray:
    """A: rounded E by default; PMF median when P(5) >= tau (enough upper-rating mass)."""
    P = np.asarray(P, float)
    return np.where(P[:, 4] >= tau, pmf_median(P), rule_rounded_E(P))


def rule_confidence_weighted(P, conf_fn: Callable = confidence_entropy) -> np.ndarray:
    """C: c*median + (1-c)*E[r], c from the PMF only (peaked PMF -> trust its median)."""
    c = conf_fn(P)
    return round_half_up(c * pmf_median(P) + (1 - c) * expected(P))


def rule_conf_threshold(P, theta_lo: float, theta_hi: float, conf_fn: Callable = confidence_entropy) -> np.ndarray:
    """D: c < theta_lo -> rounded E; c >= theta_hi -> median; in between -> integer average of both."""
    c = conf_fn(P)
    return np.where(c >= theta_hi, pmf_median(P), np.where(c < theta_lo, rule_rounded_E(P), rule_integer_average(P)))


def rule_by_construct(P, constructs: Sequence[str], choice: Mapping[str, str]) -> np.ndarray:
    """B: per-construct choice between 'E' (rounded E) and 'median'; unseen construct -> 'E' (default primary)."""
    r, m = rule_rounded_E(P), pmf_median(P)
    return np.array([m[i] if choice.get(str(c), "E") == "median" else r[i] for i, c in enumerate(constructs)])


# ------------------------------------------------------------------ metrics
def metrics(y, yp, cont=None, constructs=None, personas=None) -> dict:
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
    y, yp = np.asarray(y, int), np.asarray(yp, int)
    cont = np.asarray(yp if cont is None else cont, float)
    L = list(RATINGS); present = [k for k in L if (y == k).any()]
    m = {"exact": float((yp == y).mean()), "within1": float((abs(yp - y) <= 1).mean()),
         "MAE": float(abs(yp - y).mean()), "RMSE": float(np.sqrt(((yp - y) ** 2).mean())),
         "E_MAE": float(abs(cont - y).mean()), "E_RMSE": float(np.sqrt(((cont - y) ** 2).mean()))}
    multi = len(set(y) | set(yp)) > 1
    m["lwk"] = float(cohen_kappa_score(y, yp, weights="linear", labels=L)) if multi else 0.0
    m["qwk"] = float(cohen_kappa_score(y, yp, weights="quadratic", labels=L)) if multi else 0.0
    # *_cont = on the continuous score (comparable with Phases 6-7); plain = on the discrete prediction
    for nm, f, x in [("spearman", stats.spearmanr, yp), ("pearson", stats.pearsonr, yp),
                     ("spearman_cont", stats.spearmanr, cont), ("pearson_cont", stats.pearsonr, cont)]:
        r = f(x, y) if np.ptp(x) > 0 else (np.nan, np.nan)
        m[nm], m[nm + "_p"] = float(r[0]), float(r[1])
    for avg in ("macro", "weighted"):
        a = precision_recall_fscore_support(y, yp, labels=present, average=avg, zero_division=0)
        m[f"{avg}_P"], m[f"{avg}_R"], m[f"{avg}_F1"] = map(float, a[:3])
    pr, rc, f1, sup = precision_recall_fscore_support(y, yp, labels=L, zero_division=0)
    m["per_class"] = {int(k): {"P": float(pr[i]), "R": float(rc[i]), "F1": float(f1[i]), "support": int(sup[i]), "n_pred": int((yp == k).sum())} for i, k in enumerate(L)}
    m["confusion"] = confusion_matrix(y, yp, labels=L).tolist()
    m["five_star_FP"], m["five_star_FN"] = int(((yp == 5) & (y != 5)).sum()), int(((yp != 5) & (y == 5)).sum())
    if constructs is not None:
        c = np.asarray(constructs)
        m["per_construct_exact"] = {k: float((yp[c == k] == y[c == k]).mean()) for k in sorted(set(c))}
    if personas is not None:
        p = np.asarray(personas)
        m["per_persona_exact"] = {k[:2]: float((yp[p == k] == y[p == k]).mean()) for k in sorted(set(p))}
    return m


def selection_key(y, yp) -> tuple:
    """Pre-declared selection objective (used on CAL only): maximise QWK, then exact accuracy, then lower MAE."""
    from sklearn.metrics import cohen_kappa_score
    y, yp = np.asarray(y, int), np.asarray(yp, int)
    q = cohen_kappa_score(y, yp, weights="quadratic", labels=list(RATINGS)) if len(set(y) | set(yp)) > 1 else 0.0
    return (q, float((yp == y).mean()), -float(abs(yp - y).mean()))


# ------------------------------------------------------------------ selection on CAL only
TAU_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
THETA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01)


def select_tau(P_cal, y_cal, grid=TAU_GRID) -> float:
    # tie -> larger tau (closer to the current primary rule = fewer median overrides)
    return max(grid, key=lambda t: (selection_key(y_cal, rule_upper_gate(P_cal, t)), t))


def select_thresholds(P_cal, y_cal, grid=THETA_GRID, conf_fn=confidence_entropy) -> tuple[float, float]:
    pairs = [(a, b) for a, b in itertools.product(grid, grid) if a <= b]
    # tie -> highest thresholds (closest to the current primary rule)
    return max(pairs, key=lambda ab: (selection_key(y_cal, rule_conf_threshold(P_cal, ab[0], ab[1], conf_fn)), ab[0], ab[1]))


def select_construct_rules(P_cal, y_cal, constructs_cal) -> dict[str, str]:
    """Per construct, 'median' only if it beats rounded E on CAL by the selection key; ties keep 'E' (the primary)."""
    c = np.asarray(constructs_cal); y = np.asarray(y_cal, int); out = {}
    for k in sorted(set(c)):
        mk = c == k
        e_key, m_key = selection_key(y[mk], rule_rounded_E(P_cal[mk])), selection_key(y[mk], rule_median(P_cal[mk]))
        out[str(k)] = "median" if m_key > e_key else "E"
    return out


CAL_SUFFICIENCY = {"min_n": 100, "min_per_present_class": 10, "min_spearman_E_vs_ref": 0.30}   # declared a priori


def calibration_sufficient(E_cal, y_cal) -> dict:
    from scipy import stats
    y = np.asarray(y_cal, int); counts = {int(k): int((y == k).sum()) for k in np.unique(y)}
    rho = float(stats.spearmanr(E_cal, y)[0]) if np.ptp(E_cal) > 0 and np.ptp(y) > 0 else 0.0
    checks = {"n": (len(y), len(y) >= CAL_SUFFICIENCY["min_n"]),
              "min_class_count": (min(counts.values()), min(counts.values()) >= CAL_SUFFICIENCY["min_per_present_class"]),
              "spearman_E_vs_ref": (rho, rho >= CAL_SUFFICIENCY["min_spearman_E_vs_ref"]),
              "classes_present": (sorted(counts), len(counts) >= 3)}
    return {"criteria": CAL_SUFFICIENCY, "checks": {k: {"value": v[0], "pass": bool(v[1])} for k, v in checks.items()},
            "sufficient": all(v[1] for v in checks.values()), "class_counts": counts}


def assert_disjoint(test_texts: Sequence[str], cal_texts: Sequence[str]) -> None:
    overlap = set(map(str, test_texts)) & set(map(str, cal_texts))
    if overlap:
        raise ValueError(f"calibration/test overlap: {len(overlap)} texts")


# ------------------------------------------------------------------ experiment
def load_data():
    import pandas as pd
    S = pd.read_csv(ROOT / "evaluation/generation_quality_improvement_v1/phase6_ssr/ssr_scores_with_reference.csv")
    L = [f"imp_p{k}" for k in RATINGS]
    test = S[S.set == "improved"].reset_index(drop=True)
    base = S[S.set == "baseline"].reset_index(drop=True)
    q = pd.read_csv(ROOT / "evaluation/quick_ssr_validation/validation_results.csv")
    Pq = np.array([json.loads(p) for p in q.improved_pmf], float)
    cal = {"P": np.vstack([base[L].values, Pq]), "y": np.concatenate([base.ref.values, q.judge_reference_rating.values]).astype(int),
           "construct": np.concatenate([base.construct.values, q.construct.values]),
           "text": list(base.response_text) + list(q.response_text),
           "E_stored": np.concatenate([base.imp_E.values, q.improved_E.values])}
    tst = {"P": test[L].values, "y": test.ref.values.astype(int), "construct": test.construct.values, "persona": test.persona_id.values,
           "text": list(test.response_text), "E_stored": test.imp_E.values, "canon_E": test.canon_E.values}
    return tst, cal


def paired_vs(y, yp_cand, yp_ref, n_boot=2000, seed=20261104) -> dict:
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support
    y, a, b = map(lambda v: np.asarray(v, int), (y, yp_cand, yp_ref))
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(y), (n_boot, len(y)))

    def st(yp, ii):
        yy, pp = y[ii], yp[ii]; present = [k for k in RATINGS if (yy == k).any()]
        return ((pp == yy).mean(), cohen_kappa_score(yy, pp, weights="quadratic", labels=list(RATINGS)) if len(set(yy) | set(pp)) > 1 else 0.0,
                precision_recall_fscore_support(yy, pp, labels=present, average="macro", zero_division=0)[2], abs(pp - yy).mean())
    d = np.array([np.subtract(st(a, ii), st(b, ii)) for ii in idx])
    only_c, only_r = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    ci = lambda j: [float(x) for x in np.percentile(d[:, j], [2.5, 97.5])]
    out = {"exact_diff_ci95": ci(0), "qwk_diff_ci95": ci(1), "macroF1_diff_ci95": ci(2), "MAE_diff_ci95": ci(3),
           "only_candidate_right": only_c, "only_current_right": only_r,
           "mcnemar_exact_p": float(stats.binomtest(only_c, only_c + only_r).pvalue) if only_c + only_r else 1.0,
           "n_changed_predictions": int((a != b).sum())}
    out["supported"] = bool(out["mcnemar_exact_p"] < 0.05 or out["qwk_diff_ci95"][0] > 0 or out["macroF1_diff_ci95"][0] > 0)
    return out


def run(out_dir: Path | None = None) -> dict:
    import pandas as pd
    out_dir = out_dir or ROOT / "evaluation/generation_quality_improvement_v1/phase8_hybrid"
    out_dir.mkdir(parents=True, exist_ok=True)
    T, C = load_data()
    assert_disjoint(T["text"], C["text"])
    assert np.allclose(expected(T["P"]), T["E_stored"], atol=1e-3), "PMFs must reproduce stored Improved-SSR E[r]"
    P, y = T["P"], T["y"]
    sel = {"tau_upper_gate": select_tau(C["P"], C["y"]),
           "thresholds_entropy": select_thresholds(C["P"], C["y"], conf_fn=confidence_entropy),
           "thresholds_maxprob": select_thresholds(C["P"], C["y"], conf_fn=confidence_maxprob),
           "construct_rules": select_construct_rules(C["P"], C["y"], C["construct"])}
    suff = calibration_sufficient(expected(C["P"]), C["y"])
    cands = {
        "Canonical SSR, rounded E": (round_half_up(T["canon_E"]), T["canon_E"]),
        "CURRENT: Improved SSR, rounded E": (rule_rounded_E(P), expected(P)),
        "Improved SSR, PMF median": (rule_median(P), expected(P)),
        "A1 integer average (round E, median)": (rule_integer_average(P), expected(P)),
        "A2 weighted 0.25*E + 0.75*median": (rule_weighted(P, 0.25), 0.25 * expected(P) + 0.75 * pmf_median(P)),
        "A3 weighted 0.50*E + 0.50*median": (rule_weighted(P, 0.50), 0.50 * expected(P) + 0.50 * pmf_median(P)),
        "A4 weighted 0.75*E + 0.25*median": (rule_weighted(P, 0.75), 0.75 * expected(P) + 0.25 * pmf_median(P)),
        f"A5 upper gate: median if P5>=tau (tau={sel['tau_upper_gate']} from CAL)": (rule_upper_gate(P, sel["tau_upper_gate"]), expected(P)),
        f"B construct-specific (rules from CAL: {sel['construct_rules']})": (rule_by_construct(P, T["construct"], sel["construct_rules"]), expected(P)),
        "C1 confidence-weighted (entropy)": (rule_confidence_weighted(P, confidence_entropy), None),
        "C2 confidence-weighted (max prob)": (rule_confidence_weighted(P, confidence_maxprob), None),
        f"D1 threshold hybrid, entropy conf, thetas={sel['thresholds_entropy']} from CAL": (rule_conf_threshold(P, *sel["thresholds_entropy"], confidence_entropy), expected(P)),
        f"D2 threshold hybrid, max-prob conf, thetas={sel['thresholds_maxprob']} from CAL": (rule_conf_threshold(P, *sel["thresholds_maxprob"], confidence_maxprob), expected(P)),
        "Majority baseline (CAL mode)": (np.full(len(y), int(np.bincount(C["y"]).argmax())), None),
    }
    for nm, fn in [("C1 confidence-weighted (entropy)", confidence_entropy), ("C2 confidence-weighted (max prob)", confidence_maxprob)]:
        c = fn(P); cands[nm] = (cands[nm][0], c * pmf_median(P) + (1 - c) * expected(P))
    cur = cands["CURRENT: Improved SSR, rounded E"][0]
    res = {}
    for nm, (yp, cont) in cands.items():
        res[nm] = metrics(y, yp, cont, T["construct"], T["persona"])
        if not nm.startswith("CURRENT"):
            res[nm]["vs_current"] = paired_vs(y, yp, cur)
    # Holm correction over the McNemar p-values of the hybrid candidates
    hyb = [k for k in res if k[:2] in ("A1", "A2", "A3", "A4", "A5", "B ", "C1", "C2", "D1", "D2")]
    order = sorted(hyb, key=lambda k: res[k]["vs_current"]["mcnemar_exact_p"])
    for r_, k in enumerate(order):
        res[k]["vs_current"]["mcnemar_holm_p"] = float(min(1.0, max((len(order) - j) * res[order[j]]["vs_current"]["mcnemar_exact_p"] for j in range(r_ + 1))))
    # E: calibration only if CAL is sufficient (declared criteria); otherwise skipped and reported
    calib = {"sufficiency": suff}
    if suff["sufficient"]:
        sys.path.insert(0, str(ROOT / "scripts"))
        import ssr_experimental_decisions as X
        cp = X.OrdinalCutpoints.fit(0.5 * expected(C["P"]) + 0.5 * pmf_median(C["P"]), C["y"])
        yp = cp.predict(0.5 * expected(P) + 0.5 * pmf_median(P))
        res["E hybrid A3 + ordinal cutpoints (CAL)"] = metrics(y, yp, None, T["construct"], T["persona"]) | {"vs_current": paired_vs(y, yp, cur)}
        calib["status"] = "run"
    else:
        calib["status"] = "SKIPPED: calibration partition fails the pre-declared sufficiency criteria; forcing it would be unreliable"
    # leaky diagnostics: options picked ON TEST (never candidates) -> optimism of selection
    diag = {}
    best_tau = max(TAU_GRID, key=lambda t: selection_key(y, rule_upper_gate(P, t)))
    best_th = max([(a, b) for a, b in itertools.product(THETA_GRID, THETA_GRID) if a <= b], key=lambda ab: selection_key(y, rule_conf_threshold(P, *ab)))
    best_cr = select_construct_rules(P, y, T["construct"])
    for nm, yp, ch in [("tau", rule_upper_gate(P, best_tau), best_tau), ("thresholds_entropy", rule_conf_threshold(P, *best_th), best_th),
                       ("construct_rules", rule_by_construct(P, T["construct"], best_cr), best_cr)]:
        m = metrics(y, yp)
        diag[nm] = {"choice_on_test": ch, "exact": m["exact"], "qwk": m["qwk"], "macro_F1": m["macro_F1"], "R5": m["per_class"][5]["R"]}
    out = {"label": "Phase 8 EXPLORATORY hybrid SSR. Test n=24 (references: Qwen 2.5 7B blind judge, NOT human; no 1-2 star labels). "
                    "All selections on a disjoint calibration set (n=48).",
           "selection_objective": "max QWK, then exact, then lower MAE; ties -> option closest to current rule",
           "test_ref_dist": {int(k): int((y == k).sum()) for k in RATINGS}, "cal_ref_dist": {int(k): int((C["y"] == k).sum()) for k in RATINGS},
           "selections_from_cal": sel, "calibration_candidate_E": calib, "candidates": res,
           "LEAKY_test_oracle_diagnostics_not_candidates": diag}
    (out_dir / "phase8_metrics.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    cols = ["exact", "macro_P", "macro_R", "macro_F1", "weighted_P", "weighted_R", "weighted_F1", "spearman", "spearman_p", "pearson",
            "spearman_cont", "pearson_cont", "lwk", "qwk", "MAE", "RMSE", "E_MAE"]
    rows = []
    for k, v in res.items():
        r = {"method": k, **{c: v[c] for c in cols}, **{f"R{s}": v["per_class"][s]["R"] for s in (3, 4, 5)},
             "five_star_FP": v["five_star_FP"], **{f"exact_{c}": a for c, a in v["per_construct_exact"].items()},
             **{f"exact_{p}": a for p, a in v["per_persona_exact"].items()}}
        if "vs_current" in v:
            r.update({"mcnemar_p": v["vs_current"]["mcnemar_exact_p"], "holm_p": v["vs_current"].get("mcnemar_holm_p"),
                      "changed_preds": v["vs_current"]["n_changed_predictions"], "supported": v["vs_current"]["supported"]})
        rows.append(r)
    pd.DataFrame(rows).to_csv(out_dir / "phase8_comparison.csv", index=False)
    return out


if __name__ == "__main__":
    import pandas as pd
    o = run()
    pd.set_option("display.width", 260)
    df = pd.read_csv(ROOT / "evaluation/generation_quality_improvement_v1/phase8_hybrid/phase8_comparison.csv")
    print(df[["method", "exact", "macro_F1", "macro_R", "weighted_F1", "spearman", "qwk", "lwk", "MAE", "RMSE", "E_MAE", "R3", "R4", "R5", "five_star_FP", "exact_PE", "mcnemar_p", "holm_p", "changed_preds"]].round(3).to_string())
    print(json.dumps({k: o[k] for k in ["selections_from_cal", "calibration_candidate_E", "LEAKY_test_oracle_diagnostics_not_candidates"]}, indent=1, default=float))
