"""Phase 19: Digital-twin validation of the persona + E-SSR pipeline against the real survey.

For every eligible survey respondent we build a synthetic "twin" that shares the respondent's
persona, profile and product (their own phone: brand + price band), but never sees the
respondent's ratings or text. The twin answers the same open questions (Q23-Q26) in free text.
Human and twin answers are scored with the project's E-SSR (k=3, T=0.75, nomic-embed-text,
approved BI anchors) and compared at brand, persona, brand x persona and distribution level.
A direct 1-5 question to the twin gives the direct-elicitation (DLE) baseline.

Place this file in:   evaluation/phase19_twin_validation/run_twin_validation.py
Run from the repo root (Ollama running, models pulled):
    python evaluation/phase19_twin_validation/run_twin_validation.py --pilot 80
    python evaluation/phase19_twin_validation/run_twin_validation.py            (all respondents)
    python evaluation/phase19_twin_validation/run_twin_validation.py --analyse-only
Generation is resumable: finished twins are kept in twins.jsonl and skipped on restart.
Reads the survey workbook read-only; writes only into this folder.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from improved_ssr import improved_similarities_to_pmf  # noqa: E402
from canonical_ssr import cosine_similarities, similarities_to_pmf, expected_rating  # noqa: E402
from survey_ssr_bridge import get_construct_anchors  # noqa: E402

OLLAMA = "http://localhost:11434"
GEN_MODEL = "qwen2.5:7b-instruct"
EMB_MODEL = "nomic-embed-text"
SEED = 20261108

Q = {
    "Q23": "How would you describe your overall experience with your current smartphone?",
    "Q24": "How would you describe your smartphone's battery life and camera quality?",
    "Q25": "How would you describe your smartphone's performance, speed, and display quality?",
    "Q26": "Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?",
}
PERSONA_TEXT = {
    "P1": "a premium-tier phone owner who values build quality, polish and long-term reliability",
    "P2": "a mid-range phone owner and very heavy daily user; under heavy gaming and multitasking you notice heating, lag and battery drain, and you often look at other brands",
    "P3": "a mid-range phone owner with moderate daily use who values battery life, a simple interface and dependability",
    "P4": "a budget phone owner who paid little and feels the phone gives fair value for its price; you accept its limits because it covers your daily needs",
}
REPURCHASE_ANCHORS = [
    "I would definitely not buy a smartphone from this brand again.",
    "I am unlikely to buy a smartphone from this brand again.",
    "I am not sure whether I would buy a smartphone from this brand again.",
    "I would probably buy a smartphone from this brand again.",
    "I would definitely buy a smartphone from this brand again.",
]
ANCHOR_SET = "bi"
SUFFIX = ""
NUM_RE = re.compile(r"\d|\bout of\b|\bstars?\b|/\s*\d|percent", re.I)


# ----------------------------------------------------------------------------- survey
def col(df: pd.DataFrame, prefix: str) -> str:
    return next(c for c in df.columns if str(c).startswith(prefix))


def load_survey(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path)
    g = lambda p: raw[col(raw, p)]
    age = pd.to_numeric(g("Q3.").astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df = pd.DataFrame({
        "rid": [f"HS{i+1:03d}" for i in range(len(raw))],
        "consent": g("Q1.").astype(str).str.startswith("Yes"),
        "adult": g("Q2.").astype(str).eq("Yes") & (age >= 18),
        "owns": g("Q4.").astype(str).eq("Yes"),
        "age": age, "gender": g("Q6."), "education": g("Q7."), "status": g("Q8."),
        "spending": g("Q10."), "area": g("Q11."), "brand": g("Q12."), "price": g("Q13."),
        "os": g("Q14."), "replace": g("Q16."), "upgrade_reason": g("Q17."), "usage": g("Q18."),
        "main_use": g("Q19."), "comfort": g("Q20."),
        "q21": pd.to_numeric(g("Q21.").astype(str).str[0], errors="coerce"),
        "sources": g("Q22."),
    })
    for k in Q:
        df[f"h_{k}"] = g(k + ".").fillna("").astype(str).str.strip()
    df = df[df.consent & df.adult & df.owns & df.q21.notna()].copy()
    df["persona"] = df.apply(persona_of, axis=1)
    return df.reset_index(drop=True)


def persona_of(r) -> str:
    if r.price in ("More than ₹60,000", "₹35,001–₹60,000"):
        return "P1"
    if r.price in ("₹10,000–₹20,000", "₹20,001–₹35,000"):
        return "P2" if r.usage in ("6–8 hours", "More than 8 hours") else "P3"
    return "P4"


# ----------------------------------------------------------------------------- ollama
def post(path: str, payload: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(system: str, user: str, seed: int, json_mode: bool, num_predict: int) -> str:
    p = {"model": GEN_MODEL, "stream": False, "messages": [{"role": "system", "content": system},
                                                         {"role": "user", "content": user}],
         "options": {"temperature": 0.7, "seed": seed, "num_predict": num_predict, "num_ctx": 2048}}
    if json_mode:
        p["format"] = "json"
    return post("/api/chat", p)["message"]["content"]


def embed(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 32):
        out.extend(post("/api/embed", {"model": EMB_MODEL, "input": texts[i:i + 32]})["embeddings"])
    return np.asarray(out, dtype=float)


# ----------------------------------------------------------------------------- twin prompts
def system_prompt(r) -> str:
    brand = "a phone of a less common brand" if r.brand == "Other" else f"a {r.brand} smartphone"
    price = "an undisclosed price" if r.price == "Prefer not to say" else r.price.replace("–", " to ")
    return (
        f"You are answering a university survey about your own smartphone. You are {PERSONA_TEXT[r.persona]}. "
        f"Profile: {int(r.age)}-year-old {str(r.gender).lower()}, {str(r.status).lower()}, {str(r.education).lower()} education, "
        f"lives in a {str(r.area).lower()}, monthly personal spending: {r.spending}. "
        f"You currently own {brand} ({r.os}) bought for {price}; you have used it for a while. "
        f"Main uses: {r.main_use}. Comfort with new technology: {r.comfort}. "
        f"Upgrade habit: {r.replace}; reason for buying this phone: {r.upgrade_reason}.\n"
        "Answer like a real Indian student filling a form: honest, specific, plain English, mixed feelings are fine. "
        "Do not invent exact specifications. Never use digits, ratings, stars, fractions or percentages."
    )


def twin_user_prompt() -> str:
    qs = "\n".join(f'"{k}": {v}' for k, v in Q.items())
    return ("Answer each question in 1-3 sentences (20-60 words). Return JSON with keys Q23, Q24, Q25, Q26.\n" + qs)


DLE_USER = ("Question: How likely are you to buy the same smartphone brand again? "
            "Reply with a single digit only: 1 = very unlikely, 2 = unlikely, 3 = not sure, 4 = likely, 5 = very likely.")


def gate(ans: dict) -> list[str]:
    issues = []
    for k in Q:
        t = str(ans.get(k, "")).strip()
        if len(t.split()) < 8:
            issues.append(f"{k}: too short")
        if NUM_RE.search(t):
            issues.append(f"{k}: contains a number or rating")
    return issues


def generate(df: pd.DataFrame, out: Path) -> None:
    done = set()
    if out.exists():
        done = {json.loads(l)["rid"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = df[~df.rid.isin(done)]
    print(f"twins done: {len(done)}, to generate: {len(todo)}")
    for n, (_, r) in enumerate(todo.iterrows(), 1):
        t0, seed = time.time(), SEED + int(r.rid[2:])
        sysm, ans, issues, tries = system_prompt(r), {}, ["not run"], 0
        while issues and tries < 3:
            tries += 1
            try:
                ans = json.loads(chat(sysm, twin_user_prompt() + ("" if tries == 1 else
                                 "\nRewrite: no digits or ratings, 20-60 words each."), seed + tries, True, 700))
            except Exception as e:  # noqa: BLE001
                ans = {"error": str(e)}
            issues = gate(ans)
        dle_raw = chat(sysm, DLE_USER, seed, False, 5)
        m = re.search(r"[1-5]", dle_raw)
        rec = {"rid": r.rid, "persona": r.persona, "brand": r.brand, "tries": tries, "issues": issues,
               "answers": {k: ans.get(k, "") for k in Q}, "dle_raw": dle_raw, "dle": int(m.group()) if m else None,
               "seconds": round(time.time() - t0, 1), "model": GEN_MODEL}
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{n}/{len(todo)}] {r.rid} {r.persona} {r.brand}: tries={tries} issues={len(issues)} dle={rec['dle']} {rec['seconds']}s")


# ----------------------------------------------------------------------------- scoring
def score(texts: list[str], A: np.ndarray) -> pd.DataFrame:
    rows, E = [], embed([t if t.strip() else "." for t in texts])
    for t, v in zip(texts, E):
        if len(t.split()) < 1:
            rows.append({"E": np.nan, "E_can": np.nan})
            continue
        s = cosine_similarities(v, A)
        p = improved_similarities_to_pmf(s)
        try:
            ec = expected_rating(similarities_to_pmf(s))
        except Exception:  # noqa: BLE001
            ec = np.nan
        rows.append({"E": expected_rating(p), "E_can": ec, **{f"p{i+1}": p[i] for i in range(5)}})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- analysis
def w1(a, b) -> float:
    return float(stats.wasserstein_distance(a, b))


def boot_ci(x, y, f, n=2000, seed=SEED):
    rng, x, y, vals = np.random.default_rng(seed), np.asarray(x), np.asarray(y), []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            vals.append(f(x[i], y[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [np.nan, np.nan]


def group_compare(d: pd.DataFrame, key, min_n: int) -> tuple[pd.DataFrame, dict]:
    g = d.groupby(key).agg(n=("rid", "size"), human_q21=("q21", "mean"), human_text_E=("h_E", "mean"),
                           twin_E=("t_E", "mean"), twin_cal=("t_cal", "mean"), twin_hyb=("t_hyb", "mean"),
                           twin_DLE=("dle", "mean")).reset_index()
    g = g[g.n >= min_n]
    res = {"groups": int(len(g))}
    for c in ("twin_E", "twin_cal", "twin_hyb", "twin_DLE", "human_text_E"):
        if len(g) >= 3:
            rho = stats.spearmanr(g[c], g.human_q21).statistic
            r = stats.pearsonr(g[c], g.human_q21).statistic
            res[c] = {"spearman": float(rho), "pearson": float(r),
                      "pearson_ci95": boot_ci(g[c].values, g.human_q21.values, lambda a, b: stats.pearsonr(a, b).statistic),
                      "MAE_vs_human_mean": float((g[c] - g.human_q21).abs().mean())}
    return g, res


def analyse(df: pd.DataFrame, twins_path: Path, full: pd.DataFrame | None = None) -> None:
    tw = pd.DataFrame([json.loads(l) for l in twins_path.read_text(encoding="utf-8").splitlines() if l.strip()])
    tw = tw[tw.issues.apply(len) == 0]
    d = df.merge(tw[["rid", "answers", "dle", "tries"]], on="rid", how="inner")
    print(f"analysing {len(d)} respondent/twin pairs")
    A = embed(REPURCHASE_ANCHORS if ANCHOR_SET == "repurchase" else list(get_construct_anchors("BI").statements))
    d["h_text"] = d.h_Q26
    d["t_text"] = d.answers.apply(lambda a: a["Q26"])
    d["h_all"] = d[[f"h_{k}" for k in Q]].agg(" ".join, axis=1)
    d["t_all"] = d.answers.apply(lambda a: " ".join(a[k] for k in Q))
    hs, ts = score(d.h_text.tolist(), A), score(d.t_text.tolist(), A)
    d["h_E"], d["t_E"], d["t_E_can"] = hs.E.values, ts.E.values, ts.E_can.values
    d["h_E_all"], d["t_E_all"] = score(d.h_all.tolist(), A).E.values, score(d.t_all.tolist(), A).E.values
    d["h_words"] = d.h_text.str.split().str.len()
    d["brand_g"] = d.brand.where(d.brand.map(d.brand.value_counts()) >= 10, "Other")
    d["t_round"] = d.t_E.round().clip(1, 5)
    # Calibration: map twin E-SSR onto the human rating scale using the mean/SD of Q21 from
    # respondents NOT in the evaluated sample (held-out) and the twins' own unlabeled mean/SD.
    held = full[~full.rid.isin(d.rid)] if full is not None else d
    ref = held if len(held) >= 30 else d
    mu_h, sd_h = float(ref.q21.mean()), float(ref.q21.std())
    d["t_cal"] = ((d.t_E - d.t_E.mean()) / d.t_E.std() * sd_h + mu_h).clip(1, 5)
    d["t_hyb"] = ((d.t_cal + d.dle) / 2).clip(1, 5)

    M = {"n_pairs": int(len(d)), "model": GEN_MODEL, "embedding": EMB_MODEL, "ssr": f"E-SSR k=3 T=0.75, {ANCHOR_SET} anchors"}
    # distribution
    hq = d.q21.values
    M["distribution"] = {
        "human_q21_share": d.q21.value_counts(normalize=True).sort_index().round(3).to_dict(),
        "twin_essr_share": d.t_round.value_counts(normalize=True).sort_index().round(3).to_dict(),
        "twin_dle_share": d.dle.value_counts(normalize=True).sort_index().round(3).to_dict(),
        "W1_twin_essr_vs_human": w1(d.t_E, hq), "W1_twin_dle_vs_human": w1(d.dle.dropna(), hq),
        "KS_twin_essr_vs_human": float(stats.ks_2samp(d.t_round, hq).statistic),
        "KS_twin_dle_vs_human": float(stats.ks_2samp(d.dle.dropna(), hq).statistic),
        "SD": {"human_q21": float(np.std(hq)), "twin_E": float(d.t_E.std()), "twin_DLE": float(d.dle.std())},
    }
    # levels
    tables = {}
    for name, key, mn in [("persona", "persona", 5), ("brand", "brand_g", 10), ("brand_x_persona", ["brand_g", "persona"], 5)]:
        tables[name], M[name] = group_compare(d, key, mn)
        tables[name].to_csv(HERE / f"twin_{name}_table_{ANCHOR_SET}{SUFFIX}.csv", index=False)
    # human text vs own Q21 (replicates Phase 13 with combined text)
    for nm, sub in [("all", d), ("q26_3plus_words", d[d.h_words >= 3])]:
        M[f"human_text_vs_own_q21_{nm}"] = {
            "n": int(len(sub)),
            "spearman_Q26": float(stats.spearmanr(sub.h_E, sub.q21, nan_policy="omit").statistic),
            "spearman_all4": float(stats.spearmanr(sub.h_E_all, sub.q21, nan_policy="omit").statistic)}
    M["gate_first_pass_rate"] = float((d.tries == 1).mean())
    def indiv(pred, true):
        pred = pd.Series(pred).reset_index(drop=True).round().clip(1, 5)
        true = pd.Series(true).reset_index(drop=True)
        ok = pred.notna() & true.notna()
        p, t = pred[ok].astype(int).values, true[ok].astype(int).values
        from sklearn.metrics import cohen_kappa_score
        return {"n": int(len(p)), "exact": float((p == t).mean()), "within1": float((abs(p - t) <= 1).mean()),
                "MAE": float(abs(p - t).mean()), "QWK": float(cohen_kappa_score(t, p, weights="quadratic")),
                "spearman": float(stats.spearmanr(p, t).statistic)}
    rng = np.random.default_rng(SEED)
    shares = ref.q21.value_counts(normalize=True).sort_index()
    chance = rng.choice(shares.index.values, size=len(d), p=shares.values)
    M["calibration_reference"] = {"n_heldout": int(len(ref)), "mean": mu_h, "sd": sd_h}
    M["individual"] = {"twin_essr": indiv(d.t_E, d.q21), "twin_essr_calibrated": indiv(d.t_cal, d.q21),
                       "twin_hybrid": indiv(d.t_hyb, d.q21), "twin_direct": indiv(d.dle, d.q21),
                       "chance_baseline": indiv(chance, d.q21)}
    for nm, cc in [("cal", "t_cal"), ("hyb", "t_hyb")]:
        M["distribution"][f"W1_twin_{nm}_vs_human"] = w1(d[cc], hq)
        M["distribution"][f"KS_twin_{nm}_vs_human"] = float(stats.ks_2samp(d[cc].round(), hq).statistic)
        M["distribution"]["SD"][f"twin_{nm}"] = float(d[cc].std())
    print("CALIBRATION REFERENCE:", M["calibration_reference"])
    print("INDIVIDUAL LEVEL:", json.dumps(M["individual"], indent=2))
    for k in ("persona", "brand", "brand_x_persona"):
        print("GROUP", k, {c: {m: round(v, 3) for m, v in M[k][c].items() if m != "pearson_ci95"}
                           for c in ("twin_E", "twin_cal", "twin_hyb", "twin_DLE") if c in M[k]})
    print("DISTRIBUTION:", json.dumps(M["distribution"], indent=2, default=float))
    (HERE / f"twin_metrics_{ANCHOR_SET}{SUFFIX}.json").write_text(json.dumps(M, indent=2, default=float), encoding="utf-8")
    d.drop(columns=["answers"]).to_csv(HERE / f"twin_pairs_scored_{ANCHOR_SET}{SUFFIX}.csv", index=False)
    figures(d, tables)
    report(M)


def figures(d, T) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    x = np.arange(1, 6)
    for i, (lab, s) in enumerate([("Human Q21", d.q21), ("Twin E-SSR", d.t_round), ("Twin direct rating", d.dle.dropna())]):
        sh = s.value_counts(normalize=True).reindex(x, fill_value=0)
        ax[0].bar(x + (i - 1) * 0.27, sh.values, 0.27, label=lab)
    ax[0].set(title="Rating distribution", xlabel="Rating", ylabel="Share"); ax[0].legend(fontsize=7)
    for a, (nm, t) in zip(ax[1:], [("Persona", T["persona"]), ("Brand", T["brand"])]):
        a.scatter(t.human_q21, t.twin_E, label="Twin E-SSR")
        a.scatter(t.human_q21, t.twin_DLE, marker="x", label="Twin direct")
        lab = t.iloc[:, 0].astype(str)
        for xx, yy, l in zip(t.human_q21, t.twin_E, lab):
            a.annotate(l, (xx, yy), fontsize=7)
        lo, hi = 2.5, 4.6
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set(title=f"{nm} means: twin vs human", xlabel="Human mean Q21", ylabel="Twin mean rating"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / f"twin_validation_{ANCHOR_SET}{SUFFIX}.png", dpi=200)


def report(M) -> None:
    f = lambda k, c, s: M[k].get(c, {}).get(s, float("nan"))
    txt = f"""# Phase 19 digital-twin validation

Pairs analysed: {M['n_pairs']} (model {M['model']}, {M['ssr']}, {M['embedding']}). Gate first-pass rate {M['gate_first_pass_rate']:.1%}.

| Level | Groups | Twin E-SSR vs human Q21 (Pearson, 95% CI) | Spearman | MAE | Twin direct rating Pearson | MAE |
|---|---|---|---|---|---|---|
""" + "\n".join(
        f"| {k} | {M[k]['groups']} | {f(k,'twin_E','pearson'):.2f} {M[k].get('twin_E',{}).get('pearson_ci95')} | {f(k,'twin_E','spearman'):.2f} | {f(k,'twin_E','MAE_vs_human_mean'):.2f} | {f(k,'twin_DLE','pearson'):.2f} | {f(k,'twin_DLE','MAE_vs_human_mean'):.2f} |"
        for k in ("persona", "brand", "brand_x_persona")) + f"""

Distribution vs human Q21: Wasserstein E-SSR {M['distribution']['W1_twin_essr_vs_human']:.3f} vs direct {M['distribution']['W1_twin_dle_vs_human']:.3f}; KS E-SSR {M['distribution']['KS_twin_essr_vs_human']:.3f} vs direct {M['distribution']['KS_twin_dle_vs_human']:.3f}.
SD: human {M['distribution']['SD']['human_q21']:.2f}, twin E-SSR {M['distribution']['SD']['twin_E']:.2f}, twin direct {M['distribution']['SD']['twin_DLE']:.2f}.

Human text vs own Q21 (Spearman): all {M['human_text_vs_own_q21_all']['spearman_Q26']:.3f} (Q26) / {M['human_text_vs_own_q21_all']['spearman_all4']:.3f} (Q23-Q26 combined); 3+ words {M['human_text_vs_own_q21_q26_3plus_words']['spearman_Q26']:.3f} / {M['human_text_vs_own_q21_q26_3plus_words']['spearman_all4']:.3f}.
Notes: twins never saw the respondent's ratings or text. Groups below the minimum size are excluded. Small group counts give wide intervals.
"""
    (HERE / f"twin_report_{ANCHOR_SET}{SUFFIX}.md").write_text(txt, encoding="utf-8")
    print(txt)


# ----------------------------------------------------------------------------- main
def main() -> None:
    global GEN_MODEL, OLLAMA, ANCHOR_SET, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", default=str(ROOT / "data" / "Capstone_Reviews(Responses).xlsx"))
    ap.add_argument("--pilot", type=int, default=0, help="stratified sample size (e.g. 80); 0 = everyone")
    ap.add_argument("--heldout", type=int, default=0, help="held-out sample size of unseen respondents (e.g. 40)")
    ap.add_argument("--model", default=GEN_MODEL)
    ap.add_argument("--host", default=OLLAMA)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--redo-personas", default="", help="e.g. P2,P4: back up twins.jsonl and regenerate only these personas")
    ap.add_argument("--anchors", choices=["bi", "repurchase"], default="bi")
    a = ap.parse_args()
    GEN_MODEL, OLLAMA, ANCHOR_SET = a.model, a.host, a.anchors
    df = load_survey(Path(a.survey))
    print(f"eligible respondents: {len(df)}; personas: {df.persona.value_counts().sort_index().to_dict()}")
    full_df = df.copy()
    if a.heldout > 0 and not a.pilot:
        SUFFIX = "_heldout"
        used = set()
        for p in [HERE / "twins.jsonl", *HERE.glob("twins_backup_*.jsonl")]:
            if p.exists():
                for l in p.read_text(encoding="utf-8").splitlines():
                    if l.strip():
                        try:
                            used.add(json.loads(l)["rid"])
                        except Exception:
                            pass
        pool = full_df[~full_df.rid.isin(used)]
        per = max(1, a.heldout // 4)
        sample = pd.concat([g.sample(min(len(g), per), random_state=SEED + 1)
                            for _, g in pool.groupby("persona")]).reset_index(drop=True)
        print(f"held-out pool: {len(pool)}; sample: {len(sample)}; personas: {sample.persona.value_counts().sort_index().to_dict()}")
        out = HERE / "twins_heldout.jsonl"
        if not a.analyse_only:
            generate(sample, out)
        analyse(sample, out, full_df)
        return

    if a.pilot:
        per = max(1, a.pilot // 4)
        df = pd.concat([g.sample(min(len(g), per), random_state=SEED)
                        for _, g in df.groupby("persona")]).reset_index(drop=True)
        print(f"pilot sample: {len(df)}")
    out = HERE / "twins.jsonl"
    redo = [x.strip() for x in a.redo_personas.split(",") if x.strip()]
    if redo and out.exists():
        lines = out.read_text(encoding="utf-8").splitlines()
        backup = HERE / f"twins_backup_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        backup.write_text("\n".join(lines) + "\n", encoding="utf-8")
        keep = [l for l in lines if l.strip() and json.loads(l)["persona"] not in redo]
        out.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
        print(f"backup saved: {backup.name}; removed {len(lines) - len(keep)} twins of {redo} for regeneration")
    if not a.analyse_only:
        generate(df, out)
    analyse(df, out, full_df)


if __name__ == "__main__":
    main()
