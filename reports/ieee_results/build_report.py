"""Build the IEEE results package: results_summary.json, figures/F*.png, DigitalPersona_IEEE_Results_Report.docx.
Read-only on the repo; writes only under reports/ieee_results/. Every number is read from a saved output file."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "ieee_results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

S = {}          # key -> {"value": ..., "source": ...}


def put(key, value, source):
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    S[key] = {"value": value, "source": source}
    return value


def v(key):
    return S[key]["value"]


def jload(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- STEP 1: collect
P19 = "evaluation/phase19_twin_validation/"
SRC_DEV, SRC_HO, SRC_V1 = P19 + "twin_metrics_bi.json", P19 + "twin_metrics_bi_heldout.json", P19 + "twin_metrics.json"
dev, ho, v1 = jload(SRC_DEV), jload(SRC_HO), jload(SRC_V1)
METHODS = [("twin_essr", "E-SSR"), ("twin_hybrid", "Hybrid"), ("twin_essr_calibrated", "Calibrated E-SSR"),
           ("twin_direct", "Direct LLM rating"), ("chance_baseline", "Chance")]
for tag, M, src in (("dev", dev, SRC_DEV), ("ho", ho, SRC_HO)):
    put(f"{tag}.n_pairs", M["n_pairs"], src)
    put(f"{tag}.gate_first_pass", M["gate_first_pass_rate"], src)
    put(f"{tag}.calib_ref.n", M["calibration_reference"]["n_heldout"], src)
    put(f"{tag}.calib_ref.mean", M["calibration_reference"]["mean"], src)
    put(f"{tag}.calib_ref.sd", M["calibration_reference"]["sd"], src)
    for m, _ in METHODS:
        for k in ("exact", "within1", "MAE", "QWK", "spearman"):
            put(f"{tag}.ind.{m}.{k}", M["individual"][m][k], src)
    for m in ("twin_E", "twin_cal", "twin_hyb", "twin_DLE", "human_text_E"):
        if m in M["persona"]:
            put(f"{tag}.persona.{m}.spearman", M["persona"][m]["spearman"], src)
            put(f"{tag}.persona.{m}.pearson", M["persona"][m]["pearson"], src)
            put(f"{tag}.persona.{m}.MAE", M["persona"][m]["MAE_vs_human_mean"], src)
    D = M["distribution"]
    for k in ("W1_twin_essr_vs_human", "W1_twin_dle_vs_human", "W1_twin_cal_vs_human", "W1_twin_hyb_vs_human",
              "KS_twin_essr_vs_human", "KS_twin_dle_vs_human", "KS_twin_cal_vs_human", "KS_twin_hyb_vs_human"):
        put(f"{tag}.dist.{k}", D[k], src)
    for k, val in D["SD"].items():
        put(f"{tag}.dist.SD.{k}", val, src)
    for share in ("human_q21_share", "twin_essr_share", "twin_dle_share"):
        put(f"{tag}.dist.{share}", {str(int(float(a))): b for a, b in D[share].items()}, src)
    put(f"{tag}.human_text_q21.all.n", M["human_text_vs_own_q21_all"]["n"], src)
    put(f"{tag}.human_text_q21.all.spearman_Q26", M["human_text_vs_own_q21_all"]["spearman_Q26"], src)
    put(f"{tag}.human_text_q21.3plus.n", M["human_text_vs_own_q21_q26_3plus_words"]["n"], src)
    put(f"{tag}.human_text_q21.3plus.spearman_Q26", M["human_text_vs_own_q21_q26_3plus_words"]["spearman_Q26"], src)
for k in ("brand", "brand_x_persona"):
    for m in ("twin_E", "twin_hyb", "twin_DLE"):
        put(f"dev.{k}.{m}.spearman", dev[k][m]["spearman"], SRC_DEV)
        put(f"dev.{k}.{m}.MAE", dev[k][m]["MAE_vs_human_mean"], SRC_DEV)
put("dev.brand.groups", dev["brand"]["groups"], SRC_DEV)
put("ho.brand.groups", ho["brand"]["groups"], SRC_HO)

# v1 (original anchors, before v2 refinement)
put("v1.n_pairs", v1["n_pairs"], SRC_V1)
put("v1.gate_first_pass", v1["gate_first_pass_rate"], SRC_V1)
for m in ("twin_E", "twin_DLE"):
    put(f"v1.persona.{m}.spearman", v1["persona"][m]["spearman"], SRC_V1)
    put(f"v1.persona.{m}.MAE", v1["persona"][m]["MAE_vs_human_mean"], SRC_V1)

# persona tables
TAB = {}
for tag, f in (("dev", P19 + "twin_persona_table_bi.csv"), ("ho", P19 + "twin_persona_table_bi_heldout.csv"),
               ("v1", P19 + "twin_persona_table.csv")):
    t = pd.read_csv(ROOT / f)
    TAB[tag] = t
    put(f"{tag}.persona_table", t.to_dict(orient="records"), f)

# per-respondent pairs (for bootstrap CIs and calibrated share)
pairs_dev = pd.read_csv(ROOT / (P19 + "twin_pairs_scored_bi.csv"))
pairs_ho = pd.read_csv(ROOT / (P19 + "twin_pairs_scored_bi_heldout.csv"))
SRC_PH = P19 + "twin_pairs_scored_bi_heldout.csv"
cal_round = pairs_ho.t_cal.round().clip(1, 5).astype(int)
put("ho.dist.twin_cal_share (derived: round(t_cal))",
    {str(k): float(x) for k, x in cal_round.value_counts(normalize=True).sort_index().items()}, SRC_PH)
put("ho.check.cal_exact_recomputed", float((cal_round == pairs_ho.q21).mean()), SRC_PH)
assert abs(v("ho.check.cal_exact_recomputed") - v("ho.ind.twin_essr_calibrated.exact")) < 1e-9

# twin runtime
secs = []
for f in ("twins.jsonl", "twins_heldout.jsonl"):
    secs += [json.loads(l).get("seconds") for l in (ROOT / P19 / f).read_text(encoding="utf-8").splitlines() if l.strip()]
secs = [s for s in secs if s is not None]
put("runtime.twin_seconds_median", float(np.median(secs)), P19 + "twins.jsonl + twins_heldout.jsonl ('seconds' field)")
put("runtime.twin_count", len(secs), P19 + "twins.jsonl + twins_heldout.jsonl")
put("model.generator", ho["model"], SRC_HO)
put("model.embedding", ho["embedding"], SRC_HO)
put("model.ssr", ho["ssr"], SRC_HO)

# Phase 7: judge benchmark ablation
SRC7 = "evaluation/generation_quality_improvement_v1/phase7_ssr_candidates/phase7_metrics.json"
p7 = jload(SRC7)
P7 = [("Majority baseline (cal mode = 4)", "Majority baseline"), ("Nearest anchor (k=1)", "Nearest anchor (k=1)"),
      ("Canonical SSR, rounded E", "Canonical SSR (k=5, T=1)"),
      ("Improved SSR k=5 (no truncation) T.75, rounded E", "E-SSR k=5, T=0.75"),
      ("Improved SSR k3 T.75, rounded E (current)", "E-SSR k=3, T=0.75")]
put("p7.label", p7["label"], SRC7)
put("p7.n_test", sum(p7["test_ref_dist"].values()), SRC7)
for key, lab in P7:
    for k in ("exact", "within1", "MAE", "qwk", "spearman"):
        val = p7["candidates"][key][k]
        put(f"p7.{lab}.{k}", None if (isinstance(val, float) and np.isnan(val)) else val, SRC7)

# Phase 13: E-SSR on real human text
SRC13 = "evaluation/phase13_human_ssr/phase13_metrics.json"
p13 = jload(SRC13)
put("p13.label", p13["label"], SRC13)
put("p13.n_eligible", p13["n_eligible"], SRC13)
for k in ("n", "exact", "within1", "qwk", "spearman", "spearman_p"):
    put(f"p13.all.{k}", p13["overall_improved"][k], SRC13)
for c in ("meaningful_3plus", "one_word"):
    for k in ("n", "exact", "within1", "qwk", "spearman"):
        put(f"p13.{c}.{k}", p13["by_category"][c][k], SRC13)
put("p13.kruskal_q21_by_persona_p", p13["kruskal_q21_by_persona"], SRC13)

# Phase 18: calibration study on human text (5x4 repeated CV)
SRC18 = "evaluation/phase18_hybrid_ssr/phase18_metrics.json"
p18 = jload(SRC18)
put("p18.best_human", p18["best_human"], SRC18)
for key, lab in (("M0 canonical (ORIGINAL)", "M0 uncalibrated"), ("M2 mean/SD alignment", "M2 mean/SD alignment"),
                 ("M9 hybrid M6->M2", "M9 hybrid")):
    for k in ("exact", "within1", "MAE", "qwk", "spearman", "pred_sd", "W1_vs_human"):
        put(f"p18.{lab}.{k}", p18["human_cv"][key][k][0], SRC18)

# Cohort
SRCC, SRCP = "data/clean/human_survey_clean.csv", "data/personas/persona_assignments.csv"
hs = pd.read_csv(ROOT / SRCC)
coh = hs[hs.in_primary_cohort == True]
put("cohort.n_raw", len(hs), SRCC)
put("cohort.n", len(coh), SRCC)
put("cohort.consent_true_share", float((coh.consent == True).mean()), SRCC)
put("cohort.gender_share", {k: float(x) for k, x in coh.gender.value_counts(normalize=True).items()}, SRCC)
age = pd.to_numeric(coh.age.astype(str).str.extract(r"(\d+)")[0])
put("cohort.age_median", float(age.median()), SRCC)
put("cohort.age_min", float(age.min()), SRCC)
put("cohort.age_max", float(age.max()), SRCC)
put("cohort.os_share", {k: float(x) for k, x in coh.operating_system.value_counts(normalize=True).items()}, SRCC)
pa = pd.read_csv(ROOT / SRCP)
pa = pa[pa.respondent_id.isin(coh.respondent_id)]
put("cohort.persona_counts", pa.persona_id.value_counts().sort_index().to_dict(), SRCP)
put("cohort.persona_share", {k: float(x) for k, x in pa.persona_id.value_counts(normalize=True).sort_index().items()}, SRCP)

# Case studies (latest run per phone)
CASES = [("Google Pixel 8a", "google_pixel_8a_20261106_125851"), ("OnePlus Nord CE4 5G", "oneplus_nord_ce4_5g_20261106_123600"),
         ("Samsung Galaxy M35 5G", "samsung_galaxy_m35_5g_20261106_110236")]
case = {}
for name, d in CASES:
    src = f"data/product_evaluation/{d}/ssr_results/ssr_results.json"
    r = jload(src)
    case[name] = src
    put(f"case.{name}.construct_averages", r["construct_averages"], src)
    put(f"case.{name}.primary_adoption_average", r["primary_adoption_average"], src)
    put(f"case.{name}.completeness", {k: r.get("completeness", {}).get(k, "not recorded") for k in
        ("expected_responses", "scored_responses", "passed_first_attempt", "corrected", "complete")}, src)
    gs = r.get("generation_stats") or {}
    put(f"case.{name}.total_seconds", gs.get("total_seconds"), src + " → generation_stats")
    put(f"case.{name}.qwen_calls", gs.get("qwen_calls"), src + " → generation_stats")
    put(f"case.{name}.canonical_BI", r["comparison_summary"]["canonical"]["primary_adoption_average"], src)
    put(f"case.{name}.persona_BI", {p["persona_id"]: p["improved_adoption_rating"]
                                     for p in r["comparison_summary"]["persona_comparison"]}, src)
runs = [p for p in (ROOT / "data/product_evaluation").iterdir() if (p / "ssr_results/ssr_results.json").exists()]
put("runs.saved_product_evaluations", len(runs), "data/product_evaluation/*/ssr_results/ssr_results.json (count)")

# tests
SRCT = "reports/ieee_results/pytest_summary.txt"
put("tests.summary", (ROOT / SRCT).read_text(encoding="utf-8").strip().splitlines()[-1], SRCT)

# anchors
SRCA = "docs/survey_instrument/smartphone_construct_anchor_statements.csv"
anc = pd.read_csv(ROOT / SRCA)
bi = anc[anc.construct == "Behavioral Intention"].sort_values("anchor_rating")
put("anchors.BI_v2", list(bi.anchor_statement), SRCA)

# ---------------------------------------------------------------- STEP 2: figures
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
C = {"E-SSR": "#2563EB", "Hybrid": "#0F766E", "Calibrated E-SSR": "#7C3AED", "Direct LLM rating": "#D97706",
     "Chance": "#9CA3AF", "Human": "#111827"}
made, skipped = [], []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300)
    plt.close(fig)
    made.append(name)


def bars_individual(tag, name, title):
    labs = [l for _, l in METHODS]
    w1 = [v(f"{tag}.ind.{m}.within1") * 100 for m, _ in METHODS]
    ex = [v(f"{tag}.ind.{m}.exact") * 100 for m, _ in METHODS]
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    b1 = ax.bar(x - 0.2, w1, 0.4, color=[C[l] for l in labs], label="Within one point")
    b2 = ax.bar(x + 0.2, ex, 0.4, color=[C[l] for l in labs], alpha=0.45, hatch="//", edgecolor="white", label="Exact match")
    for b, val in list(zip(b1, w1)) + list(zip(b2, ex)):
        ax.text(b.get_x() + b.get_width() / 2, val + 1, f"{val:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x, labs, fontsize=10)
    ax.set_ylabel("% of respondents")
    ax.set_ylim(0, 100)
    ax.set_title(title)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#555"), Patch(facecolor="#555", alpha=0.45, hatch="//", edgecolor="white")],
              labels=["Within one point", "Exact match"], loc="upper right", frameon=False)
    save(fig, name)


bars_individual("ho", "F1_heldout_individual_accuracy.png",
                f"Held-out twins (n={v('ho.n_pairs')}): rating agreement with each person's own Q21")
bars_individual("dev", "F2_dev_individual_accuracy.png",
                f"Development twins (n={v('dev.n_pairs')}): rating agreement with each person's own Q21")

# F3 persona means with bootstrap CI
rng = np.random.default_rng(20261108)
PERS = ["P1", "P2", "P3", "P4"]
ci = {}
for p in PERS:
    g = pairs_dev[pairs_dev.persona == p]
    for col in ("q21", "t_E"):
        a = g[col].values
        bs = [rng.choice(a, len(a)).mean() for _ in range(2000)]
        ci[(p, col)] = (a.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
put("dev.F3_bootstrap_ci (derived, 2000 resamples, seed 20261108)",
    {f"{p}.{c}": [round(x, 4) for x in val] for (p, c), val in ci.items()}, P19 + "twin_pairs_scored_bi.csv")
fig, ax = plt.subplots(figsize=(7.5, 4.4))
x = np.arange(4)
for off, col, lab, colr in ((-0.18, "q21", "Human Q21 (actual)", C["Human"]), (0.18, "t_E", "Digital twin E-SSR", C["E-SSR"])):
    m = [ci[(p, col)][0] for p in PERS]
    lo = [ci[(p, col)][0] - ci[(p, col)][1] for p in PERS]
    hi = [ci[(p, col)][2] - ci[(p, col)][0] for p in PERS]
    ax.bar(x + off, m, 0.36, color=colr, alpha=0.85 if col == "t_E" else 0.75, label=lab)
    ax.errorbar(x + off, m, yerr=[lo, hi], fmt="none", ecolor="black", capsize=4, lw=1)
    for xi, mi in zip(x + off, m):
        ax.text(xi, 1.08, f"{mi:.2f}", ha="center", color="white", fontsize=9, fontweight="bold")
ax.set_xticks(x, ["P1 premium", "P2 mid-range\nheavy", "P3 mid-range\nmoderate", "P4 budget /\nundisclosed"])
ax.set_ylim(1, 5)
ax.set_ylabel("Mean rating (1-5)")
ax.set_title(f"Persona means, development (n={v('dev.n_pairs')}); bars = 95% bootstrap CI")
ax.legend(frameon=False, loc="upper right")
save(fig, "F3_persona_means_dev.png")

# F4 persona MAE
fig, ax = plt.subplots(figsize=(7, 4.2))
groups = [("Development (n=80)", "dev"), ("Held-out (n=40)", "ho")]
x = np.arange(2)
for off, m, lab in ((-0.2, "twin_E", "E-SSR"), (0.2, "twin_DLE", "Direct LLM rating")):
    vals = [v(f"{t}.persona.{m}.MAE") for _, t in groups]
    bb = ax.bar(x + off, vals, 0.4, color=C[lab], label=lab)
    for b, val in zip(bb, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=10)
ax.set_xticks(x, [g for g, _ in groups])
ax.set_ylabel("MAE vs human persona means (lower is better)")
ax.set_title("Persona-average error: E-SSR vs direct LLM rating")
ax.legend(frameon=False)
save(fig, "F4_persona_mae.png")

# F5 distribution (held-out)
lv = [1, 2, 3, 4, 5]
series = [("Human Q21", v("ho.dist.human_q21_share"), C["Human"]),
          ("Calibrated E-SSR", v("ho.dist.twin_cal_share (derived: round(t_cal))"), C["Calibrated E-SSR"]),
          ("Uncalibrated E-SSR", v("ho.dist.twin_essr_share"), C["E-SSR"]),
          ("Direct LLM rating", v("ho.dist.twin_dle_share"), C["Direct LLM rating"])]
fig, ax = plt.subplots(figsize=(8, 4.4))
w = 0.2
for i, (lab, sh, col) in enumerate(series):
    vals = [sh.get(str(l), 0) * 100 for l in lv]
    ax.bar(np.array(lv) + (i - 1.5) * w, vals, w, color=col, label=lab)
ax.set_xticks(lv)
ax.set_xlabel("Rating")
ax.set_ylabel("% of respondents")
ax.set_title(f"Held-out rating distribution (n={v('ho.n_pairs')}); KS: calibrated {v('ho.dist.KS_twin_cal_vs_human'):.3f}, "
             f"uncal. {v('ho.dist.KS_twin_essr_vs_human'):.3f}, direct {v('ho.dist.KS_twin_dle_vs_human'):.3f}", fontsize=11)
ax.legend(frameon=False, fontsize=9.5)
save(fig, "F5_heldout_distribution.png")

# F6 ablation
labs = [l for _, l in P7]
ex = [v(f"p7.{l}.exact") * 100 for l in labs]
qw = [v(f"p7.{l}.qwk") for l in labs]
cols = ["#9CA3AF", "#93C5FD", "#60A5FA", "#3B82F6", "#1D4ED8"]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.3))
for ax, vals, yl, fmt in ((axs[0], ex, "Exact agreement (%)", "{:.1f}"), (axs[1], qw, "Quadratic weighted kappa", "{:.3f}")):
    bb = ax.barh(labs, vals, color=cols)
    for b, val in zip(bb, vals):
        ax.text(val + (1 if yl.startswith("Exact") else 0.01), b.get_y() + b.get_height() / 2, fmt.format(val), va="center", fontsize=9.5)
    ax.set_xlabel(yl)
    ax.set_xlim(0, 100 if yl.startswith("Exact") else 0.85)
axs[1].set_yticklabels([])
fig.suptitle(f"Ablation on the {v('p7.n_test')}-response LLM-judge benchmark (reference = LLM judge, not humans)", fontsize=12)
save(fig, "F6_ablation_judge_benchmark.png")

# F7 human text
fig, ax = plt.subplots(figsize=(7, 4.2))
labs = [f"One-word answers\n(n={v('p13.one_word.n')})", f"All answers\n(n={v('p13.all.n')})",
        f"3+ word answers\n(n={v('p13.meaningful_3plus.n')})"]
vals = [v("p13.one_word.spearman"), v("p13.all.spearman"), v("p13.meaningful_3plus.spearman")]
bb = ax.bar(labs, vals, color=["#BFDBFE", "#60A5FA", C["E-SSR"]])
for b, val in zip(bb, vals):
    ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Spearman correlation with own Q21")
ax.set_title("E-SSR on real human Q26 text vs the same person's rating")
save(fig, "F7_human_text_spearman.png")

# F8 v1 -> v2
fig, axs = plt.subplots(1, 2, figsize=(9, 4.1))
for ax, k, yl in ((axs[0], "spearman", "Persona Spearman (higher is better)"), (axs[1], "MAE", "Persona MAE (lower is better)")):
    vals = [v(f"v1.persona.twin_E.{k}"), v(f"dev.persona.twin_E.{k}")]
    bb = ax.bar(["v1 (original anchors)", "v2 (purchase-intention\nanchors + refined personas)"], vals, color=["#9CA3AF", C["E-SSR"]])
    for b, val in zip(bb, vals):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center")
    ax.set_ylabel(yl)
    ax.tick_params(axis="x", labelsize=9)
fig.suptitle(f"Twin E-SSR at persona level, development n={v('dev.n_pairs')}: v1 -> v2", fontsize=12)
save(fig, "F8_v1_to_v2.png")

# F9 pipeline
fig, ax = plt.subplots(figsize=(11, 2.8))
ax.axis("off")
steps = [("Human survey", f"n={v('cohort.n')} cohort\nprofile + Q21-Q26"),
         ("Personas", "4 data-driven\nsegments"),
         ("Zero-numeric\ngeneration", f"{v('model.generator')}\nno digits/ratings"),
         ("2-gate\nvalidation", "length >= 8 words\nno numbers/ratings"),
         ("E-SSR", f"{v('model.embedding')}\ntop-3, T=0.75"),
         ("Rating", "E[r] in [1,5]\n(+ calibration)")]
for i, (h, sub) in enumerate(steps):
    x0 = i * 1.85
    ax.add_patch(FancyBboxPatch((x0, 0.2), 1.5, 1.4, boxstyle="round,pad=0.05", fc="#EFF6FF", ec="#2563EB", lw=1.5))
    ax.text(x0 + 0.75, 1.15, h, ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(x0 + 0.75, 0.55, sub, ha="center", va="center", fontsize=8.5, color="#374151")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x0 + 1.82, 0.9), xytext=(x0 + 1.58, 0.9), arrowprops=dict(arrowstyle="->", lw=1.6))
ax.set_xlim(-0.1, len(steps) * 1.85)
ax.set_ylim(0, 1.8)
save(fig, "F9_pipeline.png")

# F10 case studies
cons = ["PE", "EE", "HM", "SI", "PP", "BI"]
fig, ax = plt.subplots(figsize=(8.5, 4.3))
for i, (name, _) in enumerate(CASES):
    vals = [v(f"case.{name}.construct_averages")[c] for c in cons]
    ax.bar(np.arange(6) + (i - 1) * 0.27, vals, 0.27, label=name, color=["#2563EB", "#0F766E", "#D97706"][i])
ax.set_xticks(range(6), ["Performance\nexpectancy", "Effort\nexpectancy", "Hedonic\nmotivation", "Social\ninfluence",
                         "Perceived\nprotection", "Behavioral\nintention"], fontsize=9)
ax.set_ylim(1, 5)
ax.set_ylabel("Mean E-SSR rating (1-5)")
ax.set_title("Construct profiles of three real phones (4 personas, E-SSR)")
ax.legend(frameon=False, fontsize=9)
save(fig, "F10_case_study_constructs.png")

(OUT / "results_summary.json").write_text(json.dumps(S, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------------------------------------------------------- STEP 3: Word report
pct = lambda x: f"{x * 100:.1f}%"
f3 = lambda x: f"{x:.3f}"
doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
CAP = {}
fig_no = {}


def H(text, lvl=1):
    doc.add_heading(text, lvl)


def P(text, bold_prefix=None, italic=False):
    p = doc.add_paragraph()
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    r = p.add_run(text)
    r.italic = italic
    return p


def B(text):
    doc.add_paragraph(text, style="List Bullet")


def T(header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for c, h in zip(t.rows[0].cells, header):
        c.text = ""
        c.paragraphs[0].add_run(str(h)).bold = True
    for r in rows:
        cells = t.add_row().cells
        for c, val in zip(cells, r):
            c.text = str(val)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9)
    doc.add_paragraph()


def SRC(text):
    p = doc.add_paragraph()
    r = p.add_run("Source: " + text)
    r.italic, r.font.size, r.font.color.rgb = True, Pt(8.5), RGBColor(0x55, 0x55, 0x55)


def FIGURE(fname, caption):
    n = len(fig_no) + 1
    fig_no[fname] = n
    CAP[n] = caption
    doc.add_picture(str(FIG / fname), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(f"Fig. {n}. ").bold = True
    p.add_run(caption).italic = True


# convenience values
hoE, hoD, hoC = (lambda k: v(f"ho.ind.twin_essr.{k}")), (lambda k: v(f"ho.ind.twin_direct.{k}")), (lambda k: v(f"ho.ind.chance_baseline.{k}"))
dvE, dvH, dvD, dvC = [(lambda m: (lambda k: v(f"dev.ind.{m}.{k}")))(m) for m in ("twin_essr", "twin_hybrid", "twin_direct", "chance_baseline")]
pers = v("cohort.persona_counts")
pshare = v("cohort.persona_share")
g = v("cohort.gender_share")

# 1 title
t = doc.add_heading("Digital Persona: Persona-Grounded Synthetic Consumers Scored with Enhanced Semantic Similarity Rating (E-SSR) "
                    "— Validation Against Real Survey Respondents", 0)
P("Results package for IEEE paper drafting. All numbers are read from saved repository outputs; the Appendix lists the source "
  "file of every number. ★ = compulsory section, ★★ = most important for acceptance.", italic=True)
H("1. ★ Summary", 1)
P(f"We built digital twins of real smartphone owners from a {v('cohort.n')}-person Indian student survey and asked a local LLM "
  f"({v('model.generator')}) to answer the survey's open questions in its own words, without ever seeing the person's ratings or text "
  f"and without being allowed to write any number. Each answer was converted to a 1–5 purchase-intention rating with E-SSR "
  f"(embedding similarity to five anchor statements, top-3 truncation, T=0.75). On a held-out sample of {v('ho.n_pairs')} twins, "
  f"E-SSR placed {pct(hoE('within1'))} of individuals within one point of their own rating, versus {pct(hoD('within1'))} for "
  f"directly asking the LLM for a number and {pct(hoC('within1'))} for chance; its MAE was {f3(hoE('MAE'))} versus "
  f"{f3(hoD('MAE'))} (direct) and {f3(hoC('MAE'))} (chance). At persona level, E-SSR means were within {f3(v('ho.persona.twin_E.MAE'))} "
  f"points of the human means on held-out data, compared with {f3(v('ho.persona.twin_DLE.MAE'))} for direct rating. On the "
  f"{v('dev.n_pairs')}-person development set, a hybrid of calibrated E-SSR and direct rating reached a QWK of "
  f"{f3(dvH('QWK'))} (direct {f3(dvD('QWK'))}, chance {f3(dvC('QWK'))}), and moving from v1 to v2 anchors and personas raised "
  f"persona-level Spearman from {f3(v('v1.persona.twin_E.spearman'))} to {f3(v('dev.persona.twin_E.spearman'))}.")

# 2 key results
H("2. ★★ Key Results at a Glance", 1)
KEY = [
    ("Held-out individual accuracy (within one point)", f"E-SSR {pct(hoE('within1'))}", f"Direct {pct(hoD('within1'))}; chance {pct(hoC('within1'))}", SRC_HO),
    ("Held-out individual MAE (lower better)", f"E-SSR {f3(hoE('MAE'))}", f"Direct {f3(hoD('MAE'))}; chance {f3(hoC('MAE'))}", SRC_HO),
    ("Held-out persona-mean error (MAE)", f"E-SSR {f3(v('ho.persona.twin_E.MAE'))}", f"Direct {f3(v('ho.persona.twin_DLE.MAE'))}", SRC_HO),
    ("Held-out distribution match (KS, lower better)", f"Calibrated E-SSR {f3(v('ho.dist.KS_twin_cal_vs_human'))}",
     f"Uncalibrated {f3(v('ho.dist.KS_twin_essr_vs_human'))}; direct {f3(v('ho.dist.KS_twin_dle_vs_human'))}", SRC_HO),
    ("Held-out rating spread (SD; human = " + f"{v('ho.dist.SD.human_q21'):.3f})", f"Calibrated E-SSR {v('ho.dist.SD.twin_cal'):.3f}",
     f"Uncalibrated {v('ho.dist.SD.twin_E'):.3f}; direct {v('ho.dist.SD.twin_DLE'):.3f}", SRC_HO),
    ("Development individual QWK", f"Hybrid {f3(dvH('QWK'))}", f"Direct {f3(dvD('QWK'))}; chance {f3(dvC('QWK'))}", SRC_DEV),
    ("Development individual within one point", f"Hybrid / E-SSR {pct(dvH('within1'))}", f"Direct {pct(dvD('within1'))}; chance {pct(dvC('within1'))}", SRC_DEV),
    ("Development persona-mean error (MAE)", f"E-SSR {f3(v('dev.persona.twin_E.MAE'))}", f"Direct {f3(v('dev.persona.twin_DLE.MAE'))}", SRC_DEV),
    ("v1 → v2 persona Spearman (development, E-SSR)", f"v2 {f3(v('dev.persona.twin_E.spearman'))}", f"v1 {f3(v('v1.persona.twin_E.spearman'))}", f"{SRC_V1}; {SRC_DEV}"),
    ("LLM-judge benchmark exact agreement (n=24)", f"E-SSR k=3 {pct(v('p7.E-SSR k=3, T=0.75.exact'))} (QWK {f3(v('p7.E-SSR k=3, T=0.75.qwk'))})",
     f"Canonical SSR {pct(v('p7.Canonical SSR (k=5, T=1).exact'))} (QWK {f3(v('p7.Canonical SSR (k=5, T=1).qwk'))}); majority {pct(v('p7.Majority baseline.exact'))} (QWK {f3(v('p7.Majority baseline.qwk'))})", SRC7),
]
T(["Result", "Our method", "Baseline(s)", "Source"], KEY)
put("key_results_table", [dict(zip(["result", "ours", "baseline", "source"], r)) for r in KEY], "derived from the sources listed per row")

# 3 data
H("3. ★ Data and Personas", 1)
P(f"The survey file contains {v('cohort.n_raw')} responses; {v('cohort.n')} form the primary cohort used throughout "
  f"(in_primary_cohort = True). Gender: female {pct(g.get('Female', 0))}, male {pct(g.get('Male', 0))}. Median age "
  f"{v('cohort.age_median'):.0f} (range {v('cohort.age_min'):.0f}–{v('cohort.age_max'):.0f}). Respondents rated whether they would choose "
  f"the same brand again (Q21, 1–5) and answered four open questions (Q23–Q26) about experience, battery/camera, "
  f"performance/display and value/rebuy.")
T(["Persona", "Description", "n", "Share"],
  [(k.split("_")[0], " ".join(k.split("_")[1:]), pers[k], pct(pshare[k])) for k in sorted(pers)])
SRC(f"{SRCC} (in_primary_cohort == True); {SRCP}")
P(f"Consent and ethics: every primary-cohort respondent has consent = True ({pct(v('cohort.consent_true_share'))}) and self-reported "
  f"being 18 or older. Respondents are referenced only by pseudonymous IDs (e.g. HS001); twins receive demographic and usage profile "
  f"fields only. [Add institutional ethics approval details here.]")

# 4 method
H("4. ★ Method", 1)
FIGURE("F9_pipeline.png", "End-to-end Digital Persona pipeline: survey → personas → zero-numeric generation → two-gate validation → E-SSR → rating.")
B("Persona grounding: each twin is built from one real respondent's profile (age, gender, education, status, spending, area, brand, "
  "price band, OS, replacement habits, usage, main uses, tech comfort, information sources) and their assigned persona description. "
  "The twin never sees the person's Q21 rating or any of their Q23–Q26 text.")
B("Zero-numeric generation: the twin answers Q23–Q26 in natural language; the prompt forbids digits, ratings, stars, fractions and percentages, "
  "so the rating must come from meaning, not from a stated number.")
B(f"Two-gate validation: gate 1 rejects answers shorter than eight words; gate 2 rejects any answer containing a number or rating. Failed "
  f"answers are regenerated with a rewrite instruction. First-pass rate: {pct(v('ho.gate_first_pass'))} (held-out), "
  f"{pct(v('dev.gate_first_pass'))} (development).")
B("E-SSR: the Q26 answer and five anchor statements are embedded (nomic-embed-text, 768 dimensions). With cosine similarities γ(r), "
  "w(r) = γ(r) − min_j γ(j); only the top k = 3 weights are kept; p_raw(r) = w(r)/Σ w; p(r) = p_raw(r)^(1/T) / Σ_j p_raw(j)^(1/T) with "
  "T = 0.75; rating E[r] = Σ r·p(r). Canonical SSR is the special case k = 5, T = 1.")
B("v2 purchase-intention anchors (1→5): " + " | ".join(f"“{a}”" for a in v("anchors.BI_v2")))
B(f"Calibration: twin E-SSR scores are z-scored and mapped to the human Q21 mean and SD of respondents not in the evaluated sample "
  f"(held-out run: n={v('ho.calib_ref.n')}, mean {f3(v('ho.calib_ref.mean'))}, SD {f3(v('ho.calib_ref.sd'))}; development run: "
  f"n={v('dev.calib_ref.n')}, mean {f3(v('dev.calib_ref.mean'))}, SD {f3(v('dev.calib_ref.sd'))}). No evaluated person's rating is used.")
B("Hybrid: the average of the calibrated E-SSR score and the twin's direct 1–5 answer, clipped to [1, 5]. Individual metrics round "
  "every continuous score to the nearest integer.")
B("Chance baseline: one random draw per person from the calibration-reference Q21 distribution (a single fixed-seed sample).")
SRC("evaluation/phase19_twin_validation/run_twin_validation.py (gate, calibration, hybrid definitions); improved_ssr.py (E-SSR formula); " + SRCA)

# 5 validation
H("5. ★★ Validation Against Real Humans (Digital Twins)", 1)
P("Design: for each sampled respondent a twin is generated from their profile alone, answers the open questions, and is scored. "
  "The twin's rating is then compared with that person's real Q21 answer. The development sample (80, stratified 20 per persona) "
  "was used while refining anchors and personas; the held-out sample (40, 10 per persona) was drawn from respondents not used in "
  "development and was generated once with the frozen v2 pipeline.")
H("5.1 Individual level — held-out (n=40)", 2)
rows = lambda tag: [(l, pct(v(f"{tag}.ind.{m}.within1")), pct(v(f"{tag}.ind.{m}.exact")), f3(v(f"{tag}.ind.{m}.MAE")),
                     f3(v(f"{tag}.ind.{m}.QWK")), f3(v(f"{tag}.ind.{m}.spearman"))) for m, l in METHODS]
T(["Method", "Within one point", "Exact", "MAE", "QWK", "Spearman"], rows("ho"))
SRC(SRC_HO + " → individual")
FIGURE("F1_heldout_individual_accuracy.png",
       f"Held-out twins (n={v('ho.n_pairs')}): share of people whose twin rating is within one point of / exactly equal to their own Q21. "
       "Direct LLM rating and chance shown as baselines.")
P(f"On unseen respondents E-SSR gives the best within-one-point accuracy ({pct(hoE('within1'))}) and lowest MAE ({f3(hoE('MAE'))}) of all "
  f"methods, ahead of direct LLM rating ({pct(hoD('within1'))}, {f3(hoD('MAE'))}). Rank-based agreement is weaker on this small sample: "
  f"E-SSR QWK {f3(hoE('QWK'))} and Spearman {f3(hoE('spearman'))} versus direct {f3(hoD('QWK'))} and {f3(hoD('spearman'))}; the single "
  f"chance draw happened to reach QWK {f3(hoC('QWK'))} and Spearman {f3(hoC('spearman'))}, so individual rank agreement on n=40 is not "
  "claimed as a strength.")
H("5.2 Individual level — development (n=80)", 2)
T(["Method", "Within one point", "Exact", "MAE", "QWK", "Spearman"], rows("dev"))
SRC(SRC_DEV + " → individual")
FIGURE("F2_dev_individual_accuracy.png",
       f"Development twins (n={v('dev.n_pairs')}): within-one-point and exact agreement with each person's own Q21, with direct and chance baselines.")
P(f"On the development set every model-based method beats chance by a wide margin (within one point {pct(dvE('within1'))} for E-SSR "
  f"vs {pct(dvC('within1'))} for chance). The hybrid has the highest QWK ({f3(dvH('QWK'))}), Spearman ({f3(dvH('spearman'))}) and lowest "
  f"MAE ({f3(dvH('MAE'))}); direct rating has the highest exact agreement ({pct(dvD('exact'))}).")
H("5.3 Persona level", 2)
pt = lambda tag: [(r["persona"], r["n"], f"{r['human_q21']:.2f}", f"{r['twin_E']:.3f}", f"{r['twin_cal']:.3f}", f"{r['twin_hyb']:.3f}",
                   f"{r['twin_DLE']:.2f}") for r in v(f"{tag}.persona_table")]
P("Development (n=80):", bold_prefix=None)
T(["Persona", "n", "Human Q21", "Twin E-SSR", "Calibrated", "Hybrid", "Direct"], pt("dev"))
SRC(P19 + "twin_persona_table_bi.csv")
P("Held-out (n=40):")
T(["Persona", "n", "Human Q21", "Twin E-SSR", "Calibrated", "Hybrid", "Direct"], pt("ho"))
SRC(P19 + "twin_persona_table_bi_heldout.csv")
T(["Persona-level metric", "E-SSR", "Calibrated", "Hybrid", "Direct"],
  [(f"{name} {k}", *[f3(v(f"{tag}.persona.{m}.{k}")) for m in ("twin_E", "twin_cal", "twin_hyb", "twin_DLE")])
   for tag, name in (("dev", "Development"), ("ho", "Held-out")) for k in ("spearman", "MAE")])
SRC(f"{SRC_DEV}; {SRC_HO} → persona")
FIGURE("F3_persona_means_dev.png", "Development persona means of human Q21 and twin E-SSR with 95% bootstrap confidence intervals "
       "(2000 resamples of each persona's 20 respondents).")
FIGURE("F4_persona_mae.png", "Mean absolute error between twin persona means and human persona means: E-SSR vs direct LLM rating, "
       "development and held-out.")
P(f"E-SSR reproduces persona-level averages more closely than direct rating on both samples (MAE {f3(v('dev.persona.twin_E.MAE'))} vs "
  f"{f3(v('dev.persona.twin_DLE.MAE'))} in development; {f3(v('ho.persona.twin_E.MAE'))} vs {f3(v('ho.persona.twin_DLE.MAE'))} held-out). "
  f"In development, E-SSR ranks the four personas with Spearman {f3(v('dev.persona.twin_E.spearman'))} and the hybrid with "
  f"{f3(v('dev.persona.twin_hyb.spearman'))}. Brand-level hybrid Spearman in development is {f3(v('dev.brand.twin_hyb.spearman'))} "
  f"over {v('dev.brand.groups')} brand groups.")
H("5.4 Distribution", 2)
FIGURE("F5_heldout_distribution.png", "Held-out rating distributions: human Q21, calibrated E-SSR, uncalibrated E-SSR and direct LLM rating. "
       "Calibrated shares are derived by rounding the saved calibrated scores (t_cal).")
P(f"Uncalibrated E-SSR compresses ratings toward 3–4 (SD {v('ho.dist.SD.twin_E'):.3f} vs human {v('ho.dist.SD.human_q21'):.3f}). "
  f"Calibration restores the spread (SD {v('ho.dist.SD.twin_cal'):.3f}) and gives the closest distribution to humans: KS "
  f"{f3(v('ho.dist.KS_twin_cal_vs_human'))} and Wasserstein {f3(v('ho.dist.W1_twin_cal_vs_human'))}, versus "
  f"{f3(v('ho.dist.KS_twin_essr_vs_human'))}/{f3(v('ho.dist.W1_twin_essr_vs_human'))} uncalibrated and "
  f"{f3(v('ho.dist.KS_twin_dle_vs_human'))}/{f3(v('ho.dist.W1_twin_dle_vs_human'))} for direct rating.")

# 6 baselines
H("6. ★★ Comparison with Baselines and Ablation", 1)
P(f"Across both twin samples E-SSR is compared with (i) directly asking the same LLM for a 1–5 rating and (ii) chance. On held-out "
  f"respondents E-SSR leads on within-one-point accuracy, MAE, persona-mean error and (after calibration) distribution match "
  f"(Sections 5.1–5.4). To isolate each component, the 24-response benchmark below scores the same generated texts with "
  f"different conversion rules against an LLM-judge reference (not human ground truth).")
T(["Conversion rule", "Exact", "Within one point", "MAE", "QWK", "Spearman"],
  [(l, pct(v(f"p7.{l}.exact")), pct(v(f"p7.{l}.within1")), f3(v(f"p7.{l}.MAE")), f3(v(f"p7.{l}.qwk")),
    "n/a" if v(f"p7.{l}.spearman") is None else f3(v(f"p7.{l}.spearman"))) for _, l in P7])
SRC(SRC7)
FIGURE("F6_ablation_judge_benchmark.png", "Ablation on the 24-response LLM-judge benchmark: exact agreement and QWK for the majority "
       "baseline, nearest anchor, canonical SSR, E-SSR without truncation (k=5) and E-SSR (k=3).")
B(f"Soft scoring vs nearest anchor: using the similarity distribution instead of the single closest anchor raises exact agreement "
  f"from {pct(v('p7.Nearest anchor (k=1).exact'))} to {pct(v('p7.Canonical SSR (k=5, T=1).exact'))}.")
B(f"Temperature sharpening (T=0.75): E-SSR k=5 reaches {pct(v('p7.E-SSR k=5, T=0.75.exact'))} exact, QWK {f3(v('p7.E-SSR k=5, T=0.75.qwk'))}.")
B(f"Top-3 truncation removes the two least similar anchors: {pct(v('p7.E-SSR k=3, T=0.75.exact'))} exact, QWK "
  f"{f3(v('p7.E-SSR k=3, T=0.75.qwk'))}, MAE {f3(v('p7.E-SSR k=3, T=0.75.MAE'))} — the best of all rules.")
B(f"The majority baseline gets {pct(v('p7.Majority baseline.exact'))} exact but QWK {f3(v('p7.Majority baseline.qwk'))}, showing that exact agreement alone rewards always predicting 4.")
P(f"Calibration was studied separately on real human text with 5×4 repeated cross-validation (Phase 18, n={p18['human_n']}). The "
  f"pre-registered selection rule chose {v('p18.best_human')}: QWK {f3(v('p18.M9 hybrid.qwk'))} vs {f3(v('p18.M0 uncalibrated.qwk'))} "
  f"uncalibrated, Spearman {f3(v('p18.M9 hybrid.spearman'))} vs {f3(v('p18.M0 uncalibrated.spearman'))}, prediction SD "
  f"{f3(v('p18.M9 hybrid.pred_sd'))} vs {f3(v('p18.M0 uncalibrated.pred_sd'))} (mean over folds).")
SRC(SRC18 + " → human_cv, best_human")

# 7 human text
H("7. ★ E-SSR on Real Human Text", 1)
P(f"To check that E-SSR reads real consumer language, it was applied to each respondent's own Q26 answer and compared with the same "
  f"person's Q21 (n={v('p13.all.n')}). Spearman is {f3(v('p13.all.spearman'))} overall (p = {v('p13.all.spearman_p'):.2e}), "
  f"{f3(v('p13.meaningful_3plus.spearman'))} for answers of three or more words (n={v('p13.meaningful_3plus.n')}) and "
  f"{f3(v('p13.one_word.spearman'))} for one-word answers (n={v('p13.one_word.n')}). Within one point: {pct(v('p13.all.within1'))} overall. "
  f"The held-out twin sample shows the same pattern for its 40 humans: {f3(v('ho.human_text_q21.all.spearman_Q26'))} overall and "
  f"{f3(v('ho.human_text_q21.3plus.spearman_Q26'))} for 3+ word answers (n={v('ho.human_text_q21.3plus.n')}).")
SRC(f"{SRC13}; {SRC_HO}")
FIGURE("F7_human_text_spearman.png", "E-SSR applied to real human Q26 answers: Spearman correlation with the same person's Q21, by answer length.")

# 8 v1 -> v2
H("8. ★ Development Process: v1 → v2", 1)
P("v1 used the original behavioral-intention anchors (continued use). v2 replaced them with purchase-intention anchors that match the "
  "rebuy question and refined the P2 and P4 persona descriptions. Both were evaluated on the same 80 development respondents.")
T(["Metric (development, n=80)", "v1", "v2"],
  [("Persona Spearman, twin E-SSR", f3(v("v1.persona.twin_E.spearman")), f3(v("dev.persona.twin_E.spearman"))),
   ("Persona MAE, twin E-SSR", f3(v("v1.persona.twin_E.MAE")), f3(v("dev.persona.twin_E.MAE"))),
   ("Persona Spearman, direct rating", f3(v("v1.persona.twin_DLE.spearman")), f3(v("dev.persona.twin_DLE.spearman"))),
   ("Persona MAE, direct rating", f3(v("v1.persona.twin_DLE.MAE")), f3(v("dev.persona.twin_DLE.MAE"))),
   ("Gate first-pass rate", pct(v("v1.gate_first_pass")), pct(v("dev.gate_first_pass")))])
SRC(f"{SRC_V1}; {SRC_DEV}")
FIGURE("F8_v1_to_v2.png", "Persona-level Spearman and MAE of twin E-SSR before (v1) and after (v2) anchor and persona refinement, development set.")
P(f"Persona ranking doubled from {f3(v('v1.persona.twin_E.spearman'))} to {f3(v('dev.persona.twin_E.spearman'))} and persona-mean error "
  f"fell from {f3(v('v1.persona.twin_E.MAE'))} to {f3(v('dev.persona.twin_E.MAE'))}. These gains were measured on the development set; "
  "the held-out run (Section 5) is the unbiased confirmation.")

# 9 case studies
H("9. Case Studies: Three Real Phones", 1)
cons_names = dict(zip(cons, ["Performance expectancy", "Effort expectancy", "Hedonic motivation", "Social influence",
                             "Perceived protection", "Behavioral intention"]))
T(["Construct"] + [n for n, _ in CASES],
  [(cons_names[c], *[f"{v(f'case.{n}.construct_averages')[c]:.3f}" for n, _ in CASES]) for c in cons] +
  [("Responses scored / expected", *[f"{v(f'case.{n}.completeness')['scored_responses']}/{v(f'case.{n}.completeness')['expected_responses']}" for n, _ in CASES]),
   ("Passed gate on first attempt", *[str(v(f'case.{n}.completeness')['passed_first_attempt']) for n, _ in CASES]),
   ("Total runtime (s)", *[str(v(f'case.{n}.total_seconds')) for n, _ in CASES]),
   ("LLM calls (generation + corrections)", *[str(v(f'case.{n}.qwen_calls')) for n, _ in CASES])])
P("Saved runs used: " + "; ".join(f"{n}: {d}" for n, d in CASES) + ". Each evaluates four personas with the offline stack.")
SRC("; ".join(case.values()))
FIGURE("F10_case_study_constructs.png", "E-SSR construct profiles for Google Pixel 8a, OnePlus Nord CE4 5G and Samsung Galaxy M35 5G (four personas each).")

# 10 implementation
H("10. ★ Implementation and Reproducibility", 1)
B(f"Fully offline stack: generation {v('model.generator')} and embeddings {v('model.embedding')} (768-d), both served locally by Ollama; no cloud API.")
B(f"Twin generation runtime: median {v('runtime.twin_seconds_median'):.1f} s per twin (four answers + direct rating) over {v('runtime.twin_count')} saved twins.")
B(f"Gate first-pass rate {pct(v('ho.gate_first_pass'))} (held-out) and {pct(v('dev.gate_first_pass'))} (development).")
B(f"{v('runs.saved_product_evaluations')} saved product-evaluation runs with complete SSR outputs in data/product_evaluation/.")
B(f"Automated test suite: {v('tests.summary')}.")
B("Fixed seeds; calibration uses only respondents outside the evaluated sample; held-out twins were generated once with the frozen v2 pipeline.")
SRC("; ".join([SRC_HO, SRC_DEV, P19 + "twins*.jsonl", SRCT]))

# 11 limitations
H("11. ★ Limitations and Future Work", 1)
B(f"Pilot sample sizes: {v('dev.n_pairs')} development and {v('ho.n_pairs')} held-out twins; persona-level statistics rest on four groups, so confidence intervals are wide.")
B("The method is strongest at group level (persona means, distributions); individual-level rank agreement is modest (held-out QWK "
  f"{f3(hoE('QWK'))}).")
B(f"A single generator model ({v('model.generator')}) and a single embedding model were evaluated.")
B(f"Uncalibrated E-SSR compresses the rating spread (held-out SD {v('ho.dist.SD.twin_E'):.3f} vs human {v('ho.dist.SD.human_q21'):.3f}); calibration is needed for distribution-level use.")
held = v("ho.persona_table")
hq = [r["human_q21"] for r in held]
B(f"Persona ranking cannot be tested on held-out data: human persona means lie within {max(hq) - min(hq):.1f} points of each other ({min(hq)}–{max(hq)}).")
B(f"Planned: a full-cohort run over all {v('cohort.n')} respondents to tighten persona-level estimates.")

# 12 paper text
H("12. ★★ Ready-to-Paste Paper Text", 1)
H("Abstract", 2)
abstract = (f"Can a language model stand in for real consumers when estimating purchase intention? We present Digital Persona, an offline "
            f"pipeline that builds digital twins of real survey respondents, lets each twin describe its phone in free text with numbers "
            f"forbidden, and converts the text to a 1–5 rating with Enhanced Semantic Similarity Rating (E-SSR): embedding similarity to five "
            f"purchase-intention anchor statements with top-3 truncation and temperature sharpening. Twins were built from a {v('cohort.n')}-person "
            f"smartphone survey and never saw the person's ratings or answers. On {v('ho.n_pairs')} held-out respondents, E-SSR placed "
            f"{pct(hoE('within1'))} of individuals within one point of their own rating, compared with {pct(hoD('within1'))} for directly "
            f"asking the model for a number and {pct(hoC('within1'))} for chance, with a mean absolute error of {f3(hoE('MAE'))} versus "
            f"{f3(hoD('MAE'))} and {f3(hoC('MAE'))}. Persona-level means were within {f3(v('ho.persona.twin_E.MAE'))} points of human means "
            f"(direct rating: {f3(v('ho.persona.twin_DLE.MAE'))}), and mean/SD calibration matched the human rating distribution best "
            f"(Kolmogorov–Smirnov {f3(v('ho.dist.KS_twin_cal_vs_human'))} vs {f3(v('ho.dist.KS_twin_dle_vs_human'))}). On an {v('dev.n_pairs')}-person "
            f"development set a hybrid score reached a quadratic weighted kappa of {f3(dvH('QWK'))}. Results support semantic rating as a "
            f"more faithful route than direct numeric elicitation for group-level synthetic consumer research.")
P(abstract)
put("abstract.word_count", len(abstract.split()), "computed from report text")
P(f"[{len(abstract.split())} words]", italic=True)
H("Contributions", 2)
for c in [
    "An offline, persona-grounded digital-twin pipeline that elicits consumer opinions in free text with a zero-numeric constraint enforced by a two-gate validator.",
    "E-SSR, an extension of semantic similarity rating with top-k truncation and temperature sharpening, which outperforms nearest-anchor and canonical SSR in an ablation.",
    f"A twin-level validation against real respondents' own ratings, including a held-out confirmation, where E-SSR beats direct LLM rating on within-one-point accuracy, MAE and persona-mean error.",
    "A leakage-free calibration that uses only non-evaluated respondents' rating mean and SD and restores the human rating spread.",
    "Evidence that domain-matched purchase-intention anchors and refined personas double persona-level rank agreement (v1 → v2).",
]:
    B(c)
H("Conclusion", 2)
P(f"Digital Persona shows that grounding an LLM in real respondent profiles and reading its free-text answers semantically yields "
  f"ratings that track real people better than asking the model for a number. On unseen respondents, E-SSR reached {pct(hoE('within1'))} "
  f"within-one-point accuracy and persona-mean error of {f3(v('ho.persona.twin_E.MAE'))}, outperforming direct rating on both, and calibration "
  f"brought the rating distribution into line with the human one. The approach is most reliable for group-level questions such as persona "
  f"averages and rating distributions; a full-cohort run and additional generator models are the next steps.")
H("Figure captions", 2)
for n in sorted(CAP):
    P(CAP[n], bold_prefix=f"Fig. {n}. ")

# 13 appendix
H("13. Appendix: Every Number and Its Source", 1)
T(["Key", "Value", "Source file"],
  [(k, json.dumps(d["value"], ensure_ascii=False) if not isinstance(d["value"], str) else d["value"], d["source"])
   for k, d in S.items() if k not in ("key_results_table",)])

(OUT / "results_summary.json").write_text(json.dumps(S, indent=2, ensure_ascii=False), encoding="utf-8")
doc.save(OUT / "DigitalPersona_IEEE_Results_Report.docx")
print("figures:", made)
print("skipped:", skipped)
print("abstract words:", v("abstract.word_count"))
for r in KEY:
    print(" | ".join(r))
