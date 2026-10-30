"""Compare embedding configurations for SSR on the SAME 194 human Q26/Q21 observations.

Canonical SSR functions are used unmodified with the single approved Behavioral Intention anchor set
(domain adaptation, NOT the paper's six sets, which are unverified and are not fabricated here).

Design (to keep selection and reporting separate):
  * full sample: descriptive only, every configuration, no selection.
  * dev/test split (respondent-level, stratified by Q21, seed fixed): the configuration is chosen by Spearman
    on the DEV half ONLY; the TEST half is then reported for the chosen configuration (final held-out result).
    Test results of non-selected configurations are shown for information, not used to choose.
  * The original local mpnet baseline is re-computed and checked against data/pilot/evaluation_metrics.json.
  * openai text-embedding-3-small runs only if OPENAI_API_KEY and the openai package exist; otherwise it is
    recorded as skipped (no numbers invented).
Run from repo root: python scripts/ssr_embedding_comparison.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf  # noqa: E402
from ssr_eval_lib import check_pmf, dev_test_split, metrics_vs_q21  # noqa: E402
from survey_ssr_bridge import get_construct_anchors  # noqa: E402

OUT = ROOT / "data/pilot_embedding_comparison"
REPORT = ROOT / "docs/project_status/step8_embedding_comparison_report.md"
SEED = 20261028
LOCAL = {"local_mpnet_baseline": "sentence-transformers/all-mpnet-base-v2",
         "local_minilm": "sentence-transformers/all-MiniLM-L6-v2",
         "local_bge_base": "BAAI/bge-base-en-v1.5"}


def local_embedder(name):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name)
    return lambda xs: np.asarray(m.encode(list(xs), show_progress_bar=False))


def openai_embedder():
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "OPENAI_API_KEY not set"
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed"
    from canonical_ssr import openai_text_embedding_3_small
    client = OpenAI()
    return (lambda xs: openai_text_embedding_3_small(list(xs), client)), ""


def score_texts(texts, embed, anchors):
    A = np.asarray(embed(anchors), dtype=float)
    E = np.asarray(embed(texts), dtype=float)
    rows = []
    for e in E:
        pmf = similarities_to_pmf(cosine_similarities(e, A))
        rows.append((pmf, expected_rating(pmf)))
    P = np.stack([r[0] for r in rows])
    assert check_pmf(P), "PMF normalisation failed"
    er = np.array([r[1] for r in rows])
    assert np.allclose(er, P @ np.arange(1, 6)), "expected rating mismatch"
    return P, er


def synthetic_score_frame(rows, P, er, config, source):
    """Per-response SSR scores for generated texts (rows are the saved JSONL records)."""
    assert len(rows) == len(P) == len(er) and check_pmf(P)
    return pd.DataFrame({
        "synthetic_id": [r["synthetic_id"] for r in rows], "persona_id": [r["persona_id"] for r in rows],
        "response_text": [r["response_text"] for r in rows], "embedding_config": config, "source": source,
        "expected_rating": er, **{f"p{k}": P[:, k - 1] for k in range(1, 6)},
        "argmax_rating": P.argmax(1) + 1})


def safe_write_csv(df, path):
    """Never overwrite: identical existing file is kept; a differing one gets a _v2, _v3... sibling."""
    path = Path(path)
    new = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    cand, i = path, 1
    while cand.exists():
        if cand.read_bytes().replace(b"\r\n", b"\n") == new:  # ignore Windows line endings
            return cand
        i += 1
        cand = path.with_name(f"{path.stem}_v{i}{path.suffix}")
    cand.write_bytes(new)
    return cand


def describe_api_failure(exc) -> str:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return f"API unavailable ({type(exc).__name__}, status {code}); OpenAI comparison skipped, no numbers invented"


def run_config_safely(name, evaluate, skipped):
    """Run one configuration. The optional OpenAI config is skipped (and recorded) on any failure such as a
    429 insufficient-credits error; local configurations still raise so real bugs are not hidden."""
    try:
        return evaluate()
    except Exception as exc:
        if not name.startswith("openai"):
            raise
        skipped[name] = describe_api_failure(exc)
        return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(ROOT / "data/human_survey_cleaned.csv", keep_default_na=False, dtype=str)
    h = c[(c.in_primary_cohort == "True") & (c.text_value_rebuy_usable == "True")].reset_index(drop=True)
    texts = h.text_value_rebuy.str.strip().tolist()
    q21 = h.rebuy_brand_1to5.astype(float).astype(int).values
    anchors = get_construct_anchors("BI").to_anchor_list()
    manual = [json.loads(l) for l in open(ROOT / "data/pilot/synthetic_responses.jsonl", encoding="utf-8")]
    def load(rel):
        path = ROOT / rel
        return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    api = load("data/pilot_api/api_generated_responses.jsonl")  # OpenAI pilot (rate-limited: none produced so far)
    gem = load("data/pilot_gemini/gemini_generated_responses.jsonl")
    dev, test = dev_test_split(q21, 0.5, SEED)
    assert not set(dev) & set(test) and len(dev) + len(test) == len(q21)

    embedders = {k: (lambda n=v: local_embedder(n)) for k, v in LOCAL.items()}
    skipped = {}
    oe, why = openai_embedder()
    if oe:
        embedders["openai_text_embedding_3_small"] = lambda: oe
    else:
        skipped["openai_text_embedding_3_small"] = why

    results, syn_rows, per_cfg = {}, [], {}

    def evaluate(name, make):
        """Score one configuration fully; nothing is committed until it completes without error."""
        embed = make()
        P, er = score_texts(texts, embed, anchors)
        am = P.argmax(1) + 1
        scores = pd.DataFrame({"respondent_id": h.respondent_id, "q21": q21, "expected_rating": er, "argmax_rating": am,
                               **{f"p{k}": P[:, k - 1] for k in range(1, 6)}})
        res = {"full": metrics_vs_q21(er, am, q21, seed=SEED),
               "dev": metrics_vs_q21(er[dev], am[dev], q21[dev], seed=SEED),
               "test": metrics_vs_q21(er[test], am[test], q21[test], seed=SEED)}
        rows_out, gem_scores = [], None
        for label, rows in (("manual_pilot", manual), ("api_generated", api), ("gemini_generated", gem)):
            if rows:
                Ps, ers = score_texts([r["response_text"] for r in rows], embed, anchors)
                if label == "gemini_generated":
                    gem_scores = synthetic_score_frame(rows, Ps, ers, name, "gemini")
                rows_out.append({"config": name, "source": label, "n": len(rows),
                                 "mean_E": float(ers.mean()), "sd_E": float(ers.std(ddof=1)),
                                 "human_text_mean_E": float(er.mean()), "human_text_sd_E": float(er.std(ddof=1)),
                                 "mean_pmf": Ps.mean(0).round(4).tolist(),
                                 "human_text_mean_pmf": P.mean(0).round(4).tolist(),
                                 "w1_vs_human_text": float(np.abs(np.cumsum(Ps.mean(0)) - np.cumsum(P.mean(0))).sum())})
        return res, er, scores, rows_out, gem_scores

    for name, make in embedders.items():
        out_ = run_config_safely(name, lambda n=name, m=make: evaluate(n, m), skipped)
        if out_ is None:
            continue
        res, er, scores, rows_out, gem_scores = out_
        scores.to_csv(OUT / f"human_scores_{name}.csv", index=False)
        if gem_scores is not None:
            print("saved", safe_write_csv(gem_scores, OUT / f"synthetic_scores_{name}_gemini.csv").name)
        results[name], per_cfg[name] = res, er
        syn_rows += rows_out

    # baseline reproduction check
    base = json.load(open(ROOT / "data/pilot/evaluation_metrics.json"))
    r0 = results["local_mpnet_baseline"]["full"]
    repro = {"spearman_stored": base["spearman"]["rho"], "spearman_now": round(r0["spearman"], 3),
             "mae_stored": base["mae_expected_vs_q21"], "mae_now": round(r0["mae"], 3),
             "accuracy_stored": base["argmax_accuracy"], "accuracy_now": round(r0["accuracy_exact"], 3)}
    repro["match"] = (repro["spearman_stored"] == repro["spearman_now"] and repro["mae_stored"] == repro["mae_now"]
                      and repro["accuracy_stored"] == repro["accuracy_now"])

    # selection on DEV only
    selected = max(results, key=lambda k: results[k]["dev"]["spearman"])
    out = {"seed": SEED, "n_dev": len(dev), "n_test": len(test), "skipped": skipped, "baseline_reproduction": repro,
           "selected_on_dev": selected, "results": results, "synthetic_vs_human_text": syn_rows,
           "anchor_set": "Behavioral Intention (single, PROPOSED_DOMAIN_ADAPTATION_NOT_ORIGINAL_MAIER_ANCHOR)",
           "anchors": anchors}
    (OUT / "comparison_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(results)
    xs = np.arange(len(names))
    for i, split in enumerate(["full", "dev", "test"]):
        v = [results[n][split]["spearman"] for n in names]
        lo = [results[n][split]["spearman"] - results[n][split]["spearman_ci95"][0] for n in names]
        hi = [results[n][split]["spearman_ci95"][1] - results[n][split]["spearman"] for n in names]
        ax.errorbar(xs + (i - 1) * 0.15, v, yerr=[lo, hi], fmt="o", label=split, capsize=3)
    ax.axhline(0, color="gray", lw=.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("local_", "") for n in names], rotation=15)
    ax.set_ylabel("Spearman rho vs Q21 (95% bootstrap CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "spearman_by_embedding.png", dpi=150)
    plt.close(fig)
    write_report(out)
    print(json.dumps({k: out[k] for k in ["baseline_reproduction", "selected_on_dev", "skipped"]}, indent=1))
    for n in names:
        print(n, {s: (round(results[n][s]["spearman"], 3), round(results[n][s]["mae"], 3), round(results[n][s]["accuracy_exact"], 3)) for s in ["full", "dev", "test"]})


def write_report(o):
    def row(n, s):
        r = o["results"][n][s]
        return f"| {n} | {r['n']} | {r['spearman']:.3f} [{r['spearman_ci95'][0]:.3f}, {r['spearman_ci95'][1]:.3f}] | {r['mae']:.3f} ({r['mae_baseline_mean']:.3f}) | {r['accuracy_exact']:.3f} ({r['accuracy_baseline_majority']:.3f}) |"
    hdr = "| config | n | Spearman [95% CI] | MAE (mean-baseline) | exact acc (majority baseline) |\n|---|---|---|---|---|\n"
    tabs = {s: hdr + "\n".join(row(n, s) for n in o["results"]) for s in ["full", "dev", "test"]}
    syn = "None generated yet." if not o["synthetic_vs_human_text"] else (
        "| config | source | n | mean E[r] synthetic | mean E[r] human text | W1 between mean PMFs |\n|---|---|---|---|---|---|\n" + "\n".join(
            f"| {r['config']} | {r['source']} | {r['n']} | {r['mean_E']:.3f} | {r['human_text_mean_E']:.3f} | {r['w1_vs_human_text']:.3f} |"
            for r in o["synthetic_vs_human_text"]))
    sk = "; ".join(f"{k}: {v}" for k, v in o["skipped"].items()) or "none"
    srcs = {r["source"] for r in o["synthetic_vs_human_text"]}
    api_note = ("; ".join([
        "OpenAI-generated: " + ("scored" if "api_generated" in srcs else "none exist (the OpenAI run hit RateLimitError for every request)"),
        "Gemini-generated: " + ("scored (n=%d of 40 planned; the run stopped on quota errors)" % next(r["n"] for r in o["synthetic_vs_human_text"] if r["source"] == "gemini_generated") if "gemini_generated" in srcs else "none exist (GEMINI_API_KEY not available when the report was generated; run scripts/pilot_generate_gemini.py)")]))
    REPORT.write_text(f"""# Step 8: Embedding Configuration Comparison (same human observations)

Generated by `scripts/ssr_embedding_comparison.py`. Canonical SSR code unmodified; single approved Behavioral Intention anchor set; the paper's six anchor sets are not available and were not invented.

## Status
- OpenAI embedding (`text-embedding-3-small`, the paper's documented model): **skipped** ({sk}). No OpenAI numbers exist.
- {api_note}
- Baseline reproduction (local mpnet, full sample): {json.dumps(o['baseline_reproduction'])}.

## Design
Respondent-level split of the {o['n_dev'] + o['n_test']} paired observations into dev ({o['n_dev']}) and test ({o['n_test']}), stratified by Q21, seed {o['seed']}. The configuration is selected on **dev** Spearman only: **{o['selected_on_dev']}**. The test half is the held-out result for that choice. Nothing is fitted to the test half. Full-sample rows are descriptive only. Note SSR has no trained parameters here, so dev/test mainly guards against selection bias when picking among encoders.

## Full sample (descriptive, no selection)
{tabs['full']}

## Dev half (used for selection)
{tabs['dev']}

## Test half (held-out; selected config = {o['selected_on_dev']})
{tabs['test']}

## Synthetic vs human-text score distributions (unpaired; not individual predictions)
{syn}

## Interpretation
Compare Spearman CIs, not point estimates: with n≈97 per half the intervals are wide, so differences between encoders are mostly within noise. MAE and exact accuracy are judged against the mean and majority baselines shown in parentheses; an SSR configuration that does not beat them has no demonstrated absolute accuracy. Q21 asks about rebuying the brand; the anchors are about continuing to use the phone (proxy).

Caveat: the three local encoders were chosen as candidates in advance, but the full-sample table was computed in the same run and seen by the author, so the dev/test separation is procedural rather than blind. The test half is only 97 observations and the test CIs of the three encoders overlap heavily; do not read the ranking as established. Observed gains over the mpnet baseline are modest, and no configuration clearly beats the mean/majority baselines on MAE and exact accuracy.

Plot: `data/pilot_embedding_comparison/spearman_by_embedding.png`. Scores: `data/pilot_embedding_comparison/human_scores_*.csv`; metrics: `comparison_metrics.json`.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
