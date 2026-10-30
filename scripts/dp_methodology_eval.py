"""Methodology stage 5 (validation against held-out human data): frozen dev/test split, metrics, TSTR status and leakage audit.

Reads existing outputs only (human SSR scores from ssr_embedding_comparison.py, the cleaned survey, persona assignments, optional
condition runs). Never regenerates or redefines the test set:
  * The split is the one already used by ssr_embedding_comparison.py (ssr_eval_lib.dev_test_split, 50/50 stratified by Q21,
    seed 20261028). It is written once to data/validation_split/frozen_split_manifest.json; later runs must reproduce it exactly,
    otherwise the script stops instead of silently using a different split.
  * The encoder is the one selected on the DEV half (comparison_metrics.json "selected_on_dev"); metrics are reported on the TEST half.
Metrics that the data cannot support are reported as unavailable with the reason, never as 0.

Run: python scripts/dp_methodology_eval.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ssr_eval_lib import boot_ci, dev_test_split  # noqa: E402

SEED = 20261028
CMP = ROOT / "data/pilot_embedding_comparison"
SPLIT = ROOT / "data/validation_split/frozen_split_manifest.json"
OUT = ROOT / "data/validation/methodology_metrics.json"
ENCODERS = ["local_mpnet_baseline", "local_minilm", "local_bge_base"]
LEVELS = [1, 2, 3, 4, 5]


def unavailable(reason: str) -> dict:
    return {"status": "unavailable", "reason": reason}


# ---------- frozen split
def current_split(root: Path = ROOT) -> dict:
    h = pd.read_csv(root / "data/pilot_embedding_comparison/human_scores_local_bge_base.csv")
    q21 = h.q21.astype(int).values
    dev, test = dev_test_split(q21, 0.5, SEED)
    ids = h.respondent_id.tolist()
    d, t = sorted(ids[i] for i in dev), sorted(ids[i] for i in test)
    digest = hashlib.sha256(json.dumps({"dev": d, "test": t}).encode()).hexdigest()
    return {"seed": SEED, "method": "ssr_eval_lib.dev_test_split(q21, test_size=0.5, seed=20261028), stratified by Q21",
            "source": "usable Q26 answers of the 200-person primary cohort (row order of human_scores_*.csv)",
            "n_dev": len(d), "n_test": len(t), "dev_ids": d, "test_ids": t, "sha256": digest}


def freeze_split(root: Path = ROOT, path: Path = SPLIT) -> dict:
    """Write the manifest once; afterwards only verify. Raises if the recomputed split differs from the frozen one."""
    cur = current_split(root)
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen["sha256"] != cur["sha256"]:
            raise RuntimeError("The recomputed dev/test split differs from the frozen manifest; refusing to continue with a different test set.")
        return {**frozen, "status": "verified_against_frozen_manifest"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
    return {**cur, "status": "frozen_now (manifest written; identical to the split used by earlier runs)"}


# ---------- metrics
def kappa(a, b, weights):
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(set(a)) < 2 or len(set(b)) < 2:
        return unavailable("needs at least two observed categories in both ratings")
    return {"status": "computed", "value": float(cohen_kappa_score(a, b, labels=LEVELS, weights=weights))}


def jsd(p, q):
    """Jensen-Shannon DIVERGENCE with log base 2 (range 0-1) = square of scipy's Jensen-Shannon distance."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() <= 0 or q.sum() <= 0:
        return unavailable("empty distribution")
    return {"status": "computed", "value": float(jensenshannon(p / p.sum(), q / q.sum(), base=2) ** 2)}


def paired_metrics(e, argmax, q21, n_boot=2000) -> dict:
    """Individual-level agreement between SSR on a respondent's own Q26 text and the same respondent's Q21 (proxy)."""
    e, argmax, q21 = np.asarray(e, float), np.asarray(argmax, int), np.asarray(q21, int)
    n = len(q21)
    out = {"n": n}
    if n < 10:
        return {**out, "status": "unavailable", "reason": f"only {n} pairs"}
    const = np.ptp(e) == 0
    out["exact_accuracy_argmax"] = float((argmax == q21).mean())
    out["mae_expected_rating"] = float(np.abs(e - q21).mean())
    out["baseline_majority_accuracy"] = float((q21 == np.bincount(q21).argmax()).mean())
    out["baseline_mean_mae"] = float(np.abs(q21.mean() - q21).mean())
    out["spearman_rho"] = unavailable("constant predictions") if const else {
        "status": "computed", "value": float(spearmanr(e, q21).statistic), "ci95": boot_ci(lambda a, b: spearmanr(a, b).statistic, [e, q21], n_boot, SEED)}
    out["pearson_r"] = unavailable("constant predictions") if const else {"status": "computed", "value": float(pearsonr(e, q21).statistic)}
    out["cohen_kappa_linear"] = kappa(argmax, q21, "linear")
    out["cohen_kappa_quadratic"] = kappa(argmax, q21, "quadratic")
    out["cohen_kappa_unweighted"] = kappa(argmax, q21, None)
    return out


def distribution_metrics(pmfs: np.ndarray, q21: np.ndarray) -> dict:
    """Distribution-level comparison: mean SSR PMF vs the empirical Q21 distribution of the same respondents."""
    emp = np.array([(q21 == k).mean() for k in LEVELS])
    mean_pmf = pmfs.mean(0)
    return {"human_q21_distribution": emp.round(4).tolist(), "mean_ssr_pmf": mean_pmf.round(4).tolist(), "jsd_base2": jsd(mean_pmf, emp)}


def tstr(train_X, train_y, test_X, test_y, label_source: str, min_per_class: int = 5) -> dict:
    """Train-on-Synthetic, Test-on-Real with prerequisite checks. Returns 'unavailable' (with reasons) unless the design is valid."""
    reasons = []
    if label_source != "independent":
        reasons.append("synthetic labels are not independent of the scoring method (e.g. SSR-derived), so TSTR would be circular")
    train_y = np.asarray(train_y)
    counts = {int(k): int(v) for k, v in zip(*np.unique(train_y, return_counts=True))} if len(train_y) else {}
    if len(counts) < 2:
        reasons.append("synthetic training labels contain fewer than two classes")
    if counts and min(counts.values()) < min_per_class:
        reasons.append(f"fewer than {min_per_class} synthetic examples in some class ({counts})")
    if len(test_y) < 20:
        reasons.append("fewer than 20 real test examples")
    if reasons:
        return {**unavailable("; ".join(reasons)), "n_train": int(len(train_y)), "n_test": int(len(test_y)), "train_class_counts": counts}
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000).fit(train_X, train_y)
    pred = clf.predict(test_X)
    return {"status": "computed", "n_train": int(len(train_y)), "n_test": int(len(test_y)), "accuracy": float((pred == np.asarray(test_y)).mean()),
            "cohen_kappa_quadratic": kappa(pred, test_y, "quadratic")}


# ---------- leakage audit
def leakage_audit(split: dict, root: Path = ROOT) -> dict:
    dev, test = set(split["dev_ids"]), set(split["test_ids"])
    c = pd.read_csv(root / "data/human_survey_cleaned.csv", keep_default_na=False, dtype=str)
    prim = c[c.in_primary_cohort == "True"]
    txt = prim.set_index("respondent_id").text_value_rebuy.str.strip().str.lower()
    shared = sorted(set(txt[list(dev)]) & set(txt[list(test)]))
    cm = json.loads((root / "data/pilot_embedding_comparison/comparison_metrics.json").read_text(encoding="utf-8"))
    cfg = json.loads((root / "data/pilot/ssr_config.json").read_text(encoding="utf-8"))["sha256"]
    sha = lambda p: hashlib.sha256((root / p).read_bytes()).hexdigest()
    return {
        "respondent_overlap_dev_test": len(dev & test),
        "duplicate_respondent_ids_primary": int(prim.respondent_id.duplicated().sum()),
        "identical_q26_texts_in_both_halves": {"count": len(shared), "examples": shared[:8],
                                               "note": "short answers such as 'yes' occur in both halves; this is identical input text, not a shared respondent"},
        "encoder_selected_on": "dev half only" if cm.get("selected_on_dev") else "unknown",
        "anchors_unchanged_since_step6": sha("docs/survey_instrument/smartphone_construct_anchor_statements.csv") == cfg["anchors_csv"],
        "canonical_ssr_unchanged_since_step6": sha("canonical_ssr.py") == cfg["canonical_ssr.py"],
        "anchors_tuned_on_data": "no evidence: the anchor file is a pre-approved artefact whose hash is unchanged",
        "persona_construction_used_test_respondents": True,
        "persona_construction_note": ("Personas and their Q23-Q25 theme summaries were built from the whole 200-person primary cohort, which "
                                      "includes test-half respondents. Q21 and Q26 (the evaluation pair) were excluded from persona profiles and "
                                      "prompts, but this is a residual leakage risk for persona-level comparisons."),
        "prompt_development_on_test": "unable to verify from the repository history; no prompt was tuned against test metrics in the recorded steps",
        "test_split_persisted_before_this_run": "no (the split was recomputed deterministically from a fixed seed; now frozen in a manifest)",
    }


# ---------- conditions (persona-conditioned vs unconditioned control)
def condition_runs(root: Path = ROOT) -> list[dict]:
    rows = []
    for d in sorted((root / "data").glob("pilot_*_persona_*")) + sorted((root / "data").glob("pilot_*_control_*")):
        f = d / "condition_scores.csv"
        if f.exists():
            df = pd.read_csv(f)
            ok = df[df.status == "scored"]
            rows.append({"folder": d.name, "condition": df.condition.iloc[0] if len(df) else None, "n_saved": len(df), "n_scored": len(ok),
                         "mean_pmf": ok[[f"p{k}" for k in LEVELS]].mean().round(4).tolist() if len(ok) else None,
                         "mean_expected_rating": float(ok.expected_rating.mean()) if len(ok) else None,
                         "expected_ratings": ok.expected_rating.round(4).tolist()})
    return rows


def run(root: Path = ROOT) -> dict:
    split = freeze_split(root)
    test_ids, dev_ids = set(split["test_ids"]), set(split["dev_ids"])
    cm = json.loads((root / "data/pilot_embedding_comparison/comparison_metrics.json").read_text(encoding="utf-8"))
    selected = cm["selected_on_dev"]
    per_encoder = {}
    for enc in ENCODERS:
        h = pd.read_csv(root / f"data/pilot_embedding_comparison/human_scores_{enc}.csv")
        t = h[h.respondent_id.isin(test_ids)]
        pm = t[[f"p{k}" for k in LEVELS]].values
        per_encoder[enc] = {"test": paired_metrics(t.expected_rating, t.argmax_rating, t.q21.astype(int)),
                            "test_distribution": distribution_metrics(pm, t.q21.astype(int).values)}
    hs = pd.read_csv(root / f"data/pilot_embedding_comparison/human_scores_{selected}.csv").set_index("respondent_id")
    ds = pd.read_csv(root / "data/synthetic_dataset/synthetic_survey_dataset.csv", keep_default_na=False)
    a = pd.read_csv(root / "data/personas/persona_assignments.csv").set_index("respondent_id").persona_id
    syn = []
    for pid, g in ds[ds.status == "scored"].groupby("persona_id"):
        hum = hs[hs.index.isin(test_ids) & hs.index.isin(a[a == pid].index)]
        sp = g[[f"{selected}__p{k}" for k in LEVELS]].astype(float).values
        se = g[f"{selected}__E"].astype(float).values
        syn.append({"persona_id": pid, "n_synthetic": len(g), "n_human_test": len(hum),
                    "ks_expected_rating": ({"status": "computed", "statistic": float(ks_2samp(se, hum.expected_rating).statistic),
                                            "note": "descriptive; p-value not reported (n_synthetic small)"} if len(hum) >= 5 and len(se) >= 5 else unavailable("too few responses")),
                    "jsd_mean_pmf_vs_human_text_pmf": jsd(sp.mean(0), hum[[f"p{k}" for k in LEVELS]].values.mean(0)) if len(hum) else unavailable("no human test responses"),
                    "jsd_mean_pmf_vs_human_q21": jsd(sp.mean(0), [(hum.q21 == k).mean() for k in LEVELS]) if len(hum) else unavailable("no human test responses")})
    labels = ds[ds.status == "scored"][f"{selected}__argmax"].astype(int).values
    tstr_res = tstr(np.zeros((len(labels), 1)), labels, np.zeros((len(test_ids), 1)), np.zeros(len(test_ids)), label_source="ssr_derived")
    res = {"split": {k: split[k] for k in ("status", "method", "source", "n_dev", "n_test", "sha256", "seed")},
           "selected_encoder_on_dev": selected, "selected_encoder_test": per_encoder[selected], "all_encoders_test_for_information": per_encoder,
           "synthetic_vs_human_test_by_persona": syn, "tstr": tstr_res, "conditions": condition_runs(root), "leakage_audit": leakage_audit(split, root),
           "metric_definitions": {"exact_accuracy": "argmax of the SSR PMF equals Q21", "mae": "|E[r] - Q21|", "spearman/pearson": "E[r] vs Q21",
                                  "cohen_kappa": "argmax vs Q21 on labels 1-5; linear and quadratic weights (ordinal) and unweighted",
                                  "jsd": "Jensen-Shannon divergence, log base 2 (0 = identical, 1 = disjoint)", "ks": "two-sample KS statistic on E[r]",
                                  "proxy": "Q21 (rebuy-brand likelihood) is a proxy for the Behavioral Intention anchors"}}
    return res


def main() -> int:
    try:
        res = run()
    except RuntimeError as e:
        print("STOPPED:", e)
        return 2
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    t = res["selected_encoder_test"]["test"]
    print("split:", res["split"]["status"], res["split"]["n_dev"], "/", res["split"]["n_test"], "| encoder (chosen on dev):", res["selected_encoder_on_dev"])
    print("test: acc", round(t["exact_accuracy_argmax"], 3), "| MAE", round(t["mae_expected_rating"], 3), "| rho", round(t["spearman_rho"]["value"], 3),
          "| r", round(t["pearson_r"]["value"], 3), "| kappa(q)", t["cohen_kappa_quadratic"], "| JSD", res["selected_encoder_test"]["test_distribution"]["jsd_base2"])
    print("TSTR:", res["tstr"]["status"], "-", res["tstr"].get("reason", ""))
    print("written:", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
