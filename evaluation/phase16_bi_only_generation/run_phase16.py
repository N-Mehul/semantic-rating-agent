"""Phase 16: 200 BI-only Q26-style synthetic answers (50 per persona) + canonical SSR + comparison with human Q21.
Stages:  python run_phase16.py generate   (resumable; local Ollama qwen2.5:7b-instruct; text only)
         python run_phase16.py evaluate   (nomic-embed-text + frozen SSR k=3 T=0.75 BI anchors; no LLM)
Intended stances are sampled from the persona's HUMAN Q21 distribution and passed as WORDS (never numbers).
Outputs only in this folder.
"""
from __future__ import annotations
import collections, datetime as dt, json, re, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from scripts.run_product_evaluation import NOVA_AI_SMARTPHONE as PRODUCT  # noqa: E402
import product_evaluation_prompts_v2 as v2  # noqa: E402

SEED, N_PER, MODEL, TEMP = 20261105, 50, "qwen2.5:7b-instruct", 0.9
PIDS = {"P1": "P1_premium_tier_owner", "P2": "P2_midrange_heavy_user", "P3": "P3_midrange_moderate_user", "P4": "P4_budget_or_undisclosed_price"}
REC = OUT / "generation_records.jsonl"
Q26 = "Do you think your smartphone is worth the price you paid, and would you choose the same brand again? Why or why not?"
STANCE_WORDS = {
    1: "You are clearly unhappy with it: you feel it is not worth the money and you would definitely not choose this brand again.",
    2: "You lean negative: you doubt it is worth the money and you probably would not choose this brand again.",
    3: "You are genuinely unsure or mixed: some things are fine, some are not, and you have not decided about choosing this brand again.",
    4: "You lean positive: it seems mostly worth the money and you would probably choose this brand again, with a small reservation.",
    5: "You are very happy with it: it feels fully worth the money and you would definitely choose this brand again.",
}
LEN_BUCKETS = [(1, 2, "one or two words"), (3, 5, "a few words (about three to five)"), (6, 12, "one short sentence"), (13, 25, "one or two sentences"), (26, 60, "two or three sentences")]


def human():
    d = pd.read_excel(ROOT / "data/Capstone_Reviews(Responses).xlsx"); d.columns = [f"c{i:02d}" for i in range(d.shape[1])]
    age = pd.to_numeric(d.c03, errors="coerce")
    H = d[(d.c01.str.startswith("Yes")) & (d.c02 == "Yes") & (d.c04 == "Yes") & (age >= 18)].copy()
    H["q21"] = H.c20.str[0].astype(int)
    tier = {"More than ₹60,000": "P1", "₹35,001–₹60,000": "P1", "₹20,001–₹35,000": "mid", "₹10,000–₹20,000": "mid"}
    H["persona"] = H.c12.map(tier).fillna("P4")
    H.loc[H.persona == "mid", "persona"] = np.where(H.loc[H.persona == "mid", "c17"].isin(["6–8 hours", "More than 8 hours"]), "P2", "P3")
    H["q26"] = H.c25.fillna("").astype(str).str.strip()
    H["q26_words"] = H.q26.map(lambda t: len(re.findall(r"[A-Za-z']+", t)))
    return H


def plan(H):
    """Deterministic plan: stance ~ persona's human Q21 distribution; length ~ human Q26 word counts (non-empty); spec cycles."""
    rng = np.random.default_rng(SEED)
    lens = H.q26_words[H.q26_words > 0].values
    specs = [json.loads(l) for l in (ROOT / "data/persona_generator/respondent_specs_extended_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for p, pid in PIDS.items():
        probs = np.bincount(H[H.persona == p].q21, minlength=6)[1:] / (H.persona == p).sum()
        st = rng.choice(np.arange(1, 6), N_PER, p=probs); ln = rng.choice(lens, N_PER)
        mine = [s for s in specs if s["persona_id"] == pid]
        for i in range(N_PER):
            b = next(x for x in LEN_BUCKETS if x[0] <= ln[i] <= x[1]) if ln[i] <= 60 else LEN_BUCKETS[-1]
            rows.append({"id": f"{p}_{i+1:03d}", "persona": p, "persona_id": pid, "intended_stance": int(st[i]),
                         "target_len_words_sampled": int(ln[i]), "length_instruction": b[2], "spec": mine[i % len(mine)]})
    return rows


SYS = ("You are filling in a Google Form survey about your smartphone. Write exactly like a real Indian college student types "
       "into a form: casual, short, sometimes no capital letters or full stop, simple words, your own honest opinion. "
       "Words only: never give a number, score, rating, stars or percentage. Do not name other brands. "
       'Return JSON only: {"response": "..."}')


def prompt(r):
    s = r["spec"]
    prof = (f"About you: {s.get('current_status','')}, {s.get('area_type','')}; main phone uses: {s.get('main_uses','')}; "
            f"last phone change: {s.get('upgrade_reason','')}; you replace phones {str(s.get('replacement_frequency','')).lower()}; "
            f"price you paid for your last phone: {s.get('price_paid','')}.")
    feats = "; ".join(f"{f.name} ({f.explanation.rstrip('.')})" for f in PRODUCT.features)
    user = (f"{prof}\nImagine your current smartphone is the {PRODUCT.name} (₹{PRODUCT.price_inr:,.0f}): {feats}. "
            f"{v2.price_position(float(PRODUCT.price_inr), s.get('price_paid'))}\n"
            f"Your honest feeling: {STANCE_WORDS[r['intended_stance']]}\n\n"
            f"Survey question: \"{Q26}\"\n"
            f"Answer in {r['length_instruction']}. Give your own reason if you write more than a few words. "
            "Do not start with 'This phone' or 'The phone'.")
    return {"system": SYS, "user": user}


def generate():
    from scripts import ollama_utils
    H = human(); rows = plan(H)
    done = {}
    if REC.exists():
        for l in REC.read_text(encoding="utf-8").splitlines():
            j = json.loads(l); done[j["id"]] = j
    probe = ollama_utils.probe(env={"OLLAMA_MODEL": MODEL})
    if not probe["running"] or not probe["model_available"]:
        sys.exit(f"Ollama not ready: {probe['error']}")
    ollama_utils.SETTINGS["temperature"] = TEMP
    t_all = time.perf_counter()
    for k, r in enumerate(rows):
        if r["id"] in done:
            continue
        pr = prompt(r); attempts = []; text = None; user = pr["user"]
        for a in (1, 2):
            t0 = time.perf_counter()
            rec, err = ollama_utils.generate(MODEL, {"system": pr["system"], "user": user, "json": True, "timeout_s": 180.0,
                                                     "max_output_tokens": 120, "num_ctx": 1024}, timeout=180.0)
            parsed = v2.parse_single_v3(rec["response_text"]) if rec else {"text": None, "parse_ok": False}
            leak = v2.detect_rating_leak(parsed["text"])
            attempts.append({"attempt": a, "elapsed_s": round(time.perf_counter() - t0, 1), "error": err, "raw": (rec or {}).get("response_text"),
                             "parse_ok": parsed["parse_ok"], "rating_leak": leak, "usage": (rec or {}).get("usage")})
            text = parsed["text"]
            if err or (text and not leak):
                break
            user = pr["user"] + "\n\n" + v2.CORRECTIVE_NOTE
        status = ("generation_error" if attempts[-1]["error"] else "parse_failed" if not text
                  else "rating_leak_unresolved" if v2.detect_rating_leak(text) else "ok_after_correction" if len(attempts) == 2 else "ok")
        out = {k2: v for k2, v in r.items() if k2 != "spec"} | {"respondent_spec_id": r["spec"]["synthetic_id"], "text": text, "status": status,
               "n_calls": len(attempts), "attempts": attempts, "prompt": pr, "model": MODEL, "temperature": TEMP, "seed": SEED,
               "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        with REC.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"[{k+1}/200] {r['id']} stance={r['intended_stance']} {status} {attempts[-1]['elapsed_s']}s | {text}", flush=True)
    print("generation wall time s", round(time.perf_counter() - t_all, 1))


def evaluate():
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
    from canonical_ssr import cosine_similarities, expected_rating
    from improved_ssr import improved_similarities_to_pmf
    from survey_ssr_bridge import get_construct_anchors
    from scripts.ollama_embedding_adapter import get_ollama_embedding_adapter
    import ssr_experimental_decisions as X
    H = human(); hum = H.q21.values
    G = pd.DataFrame([json.loads(l) for l in REC.read_text(encoding="utf-8").splitlines()])
    res = {"generation": {"planned": 200, "records": int(len(G)), "status_counts": G.status.value_counts().to_dict(),
                          "corrective_retries": int((G.n_calls == 2).sum()), "llm_calls": int(G.n_calls.sum()),
                          "total_gen_seconds": float(sum(a["elapsed_s"] for at in G.attempts for a in at))}}
    S = G[G.text.notna()].reset_index(drop=True)
    ad = get_ollama_embedding_adapter("nomic-embed-text")
    A = np.asarray(ad.embed_texts(list(get_construct_anchors("BI").statements))); Em = np.asarray(ad.embed_texts(list(S.text)))
    cos = np.array([cosine_similarities(e, A) for e in Em])
    P = np.array([improved_similarities_to_pmf(c, k=3, temperature=0.75) for c in cos])
    S["E"] = [expected_rating(x) for x in P]; S["ssr"] = np.clip(np.floor(S.E + 0.5), 1, 5).astype(int)
    thr = tuple(json.loads((ROOT / "evaluation/phase14_human_calibration/phase14_metrics.json").read_text(encoding="utf-8"))["thresholds_mean_sd"]["C_quantile_matched_thresholds (fit on train)"][0])
    S["calibrated_EXPERIMENTAL"] = X.OrdinalCutpoints(thr).predict(S.E.values)
    for k in range(5):
        S[f"cos_{k+1}"] = cos[:, k]; S[f"p{k+1}"] = P[:, k]
    rng = np.random.default_rng(SEED)

    def desc(v):
        v = np.asarray(v, int); p = np.bincount(v, minlength=6)[1:] / len(v)
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        return {"n": int(len(v)), "counts": np.bincount(v, minlength=6)[1:].tolist(), "props": p.round(4).tolist(), "mean": float(v.mean()),
                "mean_ci95": np.percentile(bs, [2.5, 97.5]).round(3).tolist(), "sd": float(v.std(ddof=1)),
                "neg_1_2": float(p[:2].sum()), "neu_3": float(p[2]), "pos_4_5": float(p[3:].sum())}

    def dist_vs(v, ref):
        v, ref = np.asarray(v, int), np.asarray(ref, int)
        p, q = np.bincount(v, minlength=6)[1:] / len(v), np.bincount(ref, minlength=6)[1:] / len(ref)
        w = float(stats.wasserstein_distance(v, ref)); js = float(jensenshannon(p, q, base=2) ** 2)
        bw = [stats.wasserstein_distance(rng.choice(v, len(v)), rng.choice(ref, len(ref))) for _ in range(1000)]
        null = [stats.wasserstein_distance(rng.choice(ref, len(v)), ref) for _ in range(1000)]
        return {"W1": w, "W1_boot_ci95": np.percentile(bw, [2.5, 97.5]).round(3).tolist(), "JSD_base2": js,
                "W1_null_same_n_human_resamples_p95": float(np.percentile(null, 95)), "W1_beyond_sampling_noise": bool(w > np.percentile(null, 95))}

    res["overall"] = {"human_Q21": desc(hum), "intended_stance": desc(S.intended_stance), "ssr_canonical": desc(S.ssr),
                      "calibrated_EXPERIMENTAL": desc(S.calibrated_EXPERIMENTAL),
                      "stance_vs_human": dist_vs(S.intended_stance, hum), "ssr_vs_human": dist_vs(S.ssr, hum),
                      "calibrated_vs_human_EXPERIMENTAL": dist_vs(S.calibrated_EXPERIMENTAL, hum),
                      "E_mean_sd": [float(S.E.mean()), float(S.E.std())]}
    import ssr_hybrid_experiments as Hx
    m = Hx.metrics(S.intended_stance.values, S.ssr.values, S.E.values)
    res["ssr_vs_intended_stance"] = {k: m[k] for k in ["exact", "within1", "MAE", "RMSE", "spearman_cont", "spearman_cont_p", "qwk", "lwk", "macro_F1"]} | {
        "per_class_recall": {c: m["per_class"][c]["R"] for c in range(1, 6)}, "confusion_rows_stance_cols_ssr": m["confusion"],
        "E_mean_by_stance": S.groupby("intended_stance").E.mean().round(3).to_dict()}
    per = {}
    for p in PIDS:
        s = S[S.persona == p]; h = H[H.persona == p].q21.values
        per[p] = {"human": desc(h), "intended": desc(s.intended_stance), "ssr": desc(s.ssr), "E_mean": float(s.E.mean()),
                  "ssr_vs_human_W1": float(stats.wasserstein_distance(s.ssr, h)), "stance_vs_human_W1": float(stats.wasserstein_distance(s.intended_stance, h))}
    res["persona"] = per
    order = lambda key: sorted(PIDS, key=lambda p: -key(p))
    res["persona_order"] = {"human": order(lambda p: per[p]["human"]["mean"]), "intended": order(lambda p: per[p]["intended"]["mean"]),
                            "ssr_E": order(lambda p: per[p]["E_mean"]),
                            "kruskal_p_human": float(stats.kruskal(*[H[H.persona == p].q21 for p in PIDS])[1]),
                            "kruskal_p_ssr_E": float(stats.kruskal(*[S[S.persona == p].E for p in PIDS])[1]),
                            "kruskal_p_ssr_rating": float(stats.kruskal(*[S[S.persona == p].ssr for p in PIDS])[1]),
                            "eta2_human": None, "P1_minus_P2_human": per["P1"]["human"]["mean"] - per["P2"]["human"]["mean"],
                            "P1_minus_P2_ssr_E": per["P1"]["E_mean"] - per["P2"]["E_mean"]}
    # ---- text quality
    W = lambda t: re.findall(r"[a-z']+", str(t).lower())
    def text_stats(tx):
        w = [W(t) for t in tx]; n = np.array([len(x) for x in w]); allw = [y for x in w for y in x]
        op1 = [x[0] if x else "" for x in w]; op3 = [" ".join(x[:3]) for x in w]
        g3 = collections.Counter(tuple(x[i:i + 3]) for x in w for i in range(len(x) - 2)); g5 = collections.Counter(tuple(x[i:i + 5]) for x in w for i in range(len(x) - 4))
        low = [t for t in tx]
        return {"n": len(w), "words_mean": float(n.mean()), "words_median": float(np.median(n)), "words_sd": float(n.std()),
                "len_buckets_pct": {f"{a}-{b}": float(((n >= a) & (n <= b)).mean()) for a, b, _ in LEN_BUCKETS} | {">60": float((n > 60).mean())},
                "starts_yes": float(np.mean([o in ("yes", "yeah", "ya", "yep", "haan", "yess") for o in op1])),
                "starts_no": float(np.mean([o in ("no", "nope", "nah", "not") for o in op1])),
                "starts_maybe_notsure": float(np.mean([o3.startswith(("maybe", "not sure", "i'm not sure", "im not sure", "idk", "i dont know", "i don't know", "depends")) for o3 in op3])),
                "starts_this_or_the_phone": float(np.mean([" ".join(x[:2]) in ("this phone", "the phone") for x in w])),
                "distinct_first_word": len(set(op1)), "distinct_3w_openings": len(set(op3)), "top_3w_openings": collections.Counter(op3).most_common(6),
                "type_token_ratio": len(set(allw)) / max(1, len(allw)), "distinct_2": len(set(zip(allw, allw[1:]))) / max(1, len(allw) - 1),
                "repeated_3grams": int(sum(c > 1 for c in g3.values())), "repeated_5grams": int(sum(c > 1 for c in g5.values())),
                "top_3grams": [(" ".join(g), c) for g, c in g3.most_common(6)], "exact_duplicate_texts": int(len(low) - len(set(s.lower().strip() for s in low))),
                "no_final_punct": float(np.mean([not re.search(r"[.!?]$", str(t).strip()) for t in tx])), "lowercase_start": float(np.mean([bool(re.match(r"^[a-z]", str(t).strip())) for t in tx]))}
    hq = H.q26[H.q26_words > 0].tolist()
    res["text"] = {"synthetic": text_stats(S.text.tolist()), "human_Q26_nonempty": text_stats(hq),
                   "synthetic_by_persona_words_mean": S.groupby("persona").text.apply(lambda t: float(np.mean([len(W(x)) for x in t]))).to_dict()}
    # construct relevance / leakage / other brands / copying
    BI = re.compile(r"\b(buy|again|same brand|choose|worth|money|price|keep|switch|stick|recommend|upgrade|next|value|regret|loyal|brand)\b", re.I)
    OB = re.compile(r"\b(apple|iphone|samsung|xiaomi|redmi|oneplus|vivo|oppo|realme|google|pixel|motorola|nothing phone|poco)\b", re.I)
    hn = {re.sub(r"\W+", " ", h.lower()).strip() for h in hq}
    h5 = collections.Counter(tuple(x[i:i + 5]) for x in (W(t) for t in hq) for i in range(len(x) - 4))
    res["integrity"] = {"bi_cue_present_share": float(S.text.str.contains(BI).mean()), "rating_leak_final": int(S.text.map(lambda t: bool(v2.detect_rating_leak(t))).sum()),
                        "other_brand_mentions": int(S.text.str.contains(OB).sum()),
                        "exact_matches_with_human_Q26": int(sum(re.sub(r"\W+", " ", t.lower()).strip() in hn for t in S.text)),
                        "exact_matches_excluding_1_2_word": int(sum(re.sub(r"\W+", " ", t.lower()).strip() in hn and len(W(t)) > 2 for t in S.text)),
                        "synthetic_with_5gram_shared_with_human": int(sum(any(tuple(W(t)[i:i + 5]) in h5 for i in range(len(W(t)) - 4)) for t in S.text)),
                        "mentions_nova": int(S.text.str.contains("nova", case=False).sum())}
    # sentiment variation (lexical) by intended stance
    POS = re.compile(r"\b(yes|good|great|worth|love|happy|best|satisfied|definitely|nice|awesome|value)\b", re.I)
    NEG = re.compile(r"\b(no|not|never|bad|poor|waste|regret|disappoint\w*|expensive|overpriced|pricey|won't|wouldn't|don't)\b", re.I)
    S["lex_pol"] = S.text.map(lambda t: len(POS.findall(t)) - len(NEG.findall(t)))
    res["sentiment_lexical_by_stance"] = S.groupby("intended_stance").lex_pol.mean().round(2).to_dict()
    res["sentiment_lexical_spearman_with_stance"] = float(stats.spearmanr(S.lex_pol, S.intended_stance)[0])
    # persona differentiation in text (mpnet, local)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    M = enc.encode(S.text.tolist(), show_progress_bar=False); M = M / np.linalg.norm(M, axis=1, keepdims=True)
    lab = S.persona.values; Cm = M @ M.T; iu = np.triu_indices(len(S), 1); same = (lab[:, None] == lab[None, :])[iu]
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    acc = cross_val_score(LogisticRegression(max_iter=2000), M, lab, cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).mean()
    res["persona_text_differentiation"] = {"mean_cos_within_persona": float(Cm[iu][same].mean()), "mean_cos_between_persona": float(Cm[iu][~same].mean()),
                                           "persona_identifiable_from_text_cv_acc": float(acc), "chance": 0.25,
                                           "mean_pairwise_cos_all": float(Cm[iu].mean())}
    S.drop(columns=["attempts", "prompt"]).to_csv(OUT / "phase16_scored_responses.csv", index=False)
    (OUT / "phase16_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    {"generate": generate, "evaluate": evaluate}[sys.argv[1]]()
