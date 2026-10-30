import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import dp_constructs as dc
from scripts import dp_persona_generator as pg
from scripts import dp_synthetic_dataset as sd
from scripts import dp_validation_agent as va
from scripts import pilot_generate_gemini as gg
from scripts import pilot_generate_openai as og
from scripts import run_pipeline as rp
from scripts import ssr_embedding_comparison as cmp
from scripts import ssr_eval_lib as lib

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_CSV = ROOT / "docs/survey_instrument/smartphone_construct_anchor_statements.csv"


def fake_embed(texts):
    return np.array([np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(size=16) for t in texts])


def ext_specs():
    return [json.loads(l) for l in open(ROOT / "data/persona_generator/respondent_specs_extended.jsonl", encoding="utf-8")]


# ---- construct registry
def test_registry_statuses_come_from_stored_files():
    reg = {r["construct_code"]: r for r in dc.construct_registry()}
    assert reg["PE"]["status"] == reg["BI"]["status"] == "ready_for_generation_and_scoring"
    assert all(reg[c]["status"] == "blocked_needs_approved_free_text_prompt" for c in ("EE", "HM", "SI", "PP"))
    assert reg["PV"]["status"] == "blocked_needs_approved_anchor_set" and reg["PV"]["free_text_prompt_ids"] == ["FT_PV01"]
    assert reg["PE"]["free_text_prompt_ids"] == ["FT_PE01"] and reg["BI"]["free_text_prompt_ids"] == ["FT_BI01"]


def test_anchors_are_exactly_the_approved_stored_statements():
    csv = pd.read_csv(ANCHOR_CSV)
    for code, name in dc.ANCHOR_CODES.items():
        info = dc.anchor_info(code)
        stored = csv[csv.construct == name].sort_values("anchor_rating")
        assert info["statements"] == list(stored.anchor_statement) and list(stored.anchor_rating) == [1, 2, 3, 4, 5]
        assert info["status"] == "PROPOSED_DOMAIN_ADAPTATION_NOT_ORIGINAL_MAIER_ANCHOR"


# ---- construct-specific prompts
def test_q26_task_reproduces_prompts_used_by_existing_runs_exactly():
    new = dc.build_task_prompts("Q26_BI_proxy", ext_specs())
    old = [json.loads(l) for l in open(ROOT / "data/pilot_gemini_v2/prompts.jsonl", encoding="utf-8")]
    assert [(p["synthetic_id"], p["system"], p["user"]) for p in new] == [(p["synthetic_id"], p["system"], p["user"]) for p in old]


@pytest.mark.parametrize("task", ["FT_PE01", "FT_BI01"])
def test_codebook_task_prompts_use_codebook_wording_and_full_persona(task):
    cb = dc.load_codebook()
    q = cb[cb.question_id == task].iloc[0].question_text
    ps = dc.build_task_prompts(task, ext_specs(), codebook=cb)
    assert len(ps) == 40 and {p["persona_id"][:2] for p in ps} == {"P1", "P2", "P3", "P4"}
    assert all(q in p["user"] and p["task_id"] == task and p["profile_version"] == "v2" for p in ps)
    assert all(not lib.prompt_leak_findings(p["system"] + p["user"]) for p in ps)
    assert "no ratings" in ps[0]["system"] and "Q21" not in ps[0]["user"]
    assert "year-old" in ps[0]["user"] and "education:" in ps[0]["user"] and "mainly for" in ps[0]["user"]


def test_load_prompts_file_validation(tmp_path):
    good = {"synthetic_id": "A", "persona_id": "P1", "system": "words only", "user": "Profile: x. Question: y?"}
    f = tmp_path / "p.jsonl"
    f.write_text(json.dumps(good) + "\n", encoding="utf-8")
    assert len(dc.load_prompts_file(f)) == 1
    f.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        dc.load_prompts_file(f)
    f.write_text(json.dumps({"synthetic_id": "A"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs"):
        dc.load_prompts_file(f)
    f.write_text(json.dumps({**good, "user": "They answered Q21 as 4"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="leak"):
        dc.load_prompts_file(f)


# ---- persona attributes
def test_extended_specs_keep_v1_values_and_add_demographic_behavioural_attributes():
    r = pg.build(write=False)
    v1 = r["specs"]
    for a, b in zip(v1, r["extended_specs"]):
        assert all(b[k] == a[k] for k in pg.SPEC_VARS + ["synthetic_id", "persona_id", "operating_system"])
        assert all(k in b for k in pg.EXT_VARS + ["main_uses", "profile_text_v2"]) and isinstance(b["age"], int) and b["age"] >= 18
    assert r["extended_specs"] == pg.build(write=False)["extended_specs"]               # deterministic
    cov = r["manifest"]["attribute_coverage"]
    assert {"gender", "age", "education", "current_status", "area_type", "monthly_spending"} <= set(cov["demographic"])
    assert {"brand", "price_paid", "daily_usage", "main_uses", "info_sources", "tech_comfort_1to5"} <= set(cov["behavioural"])
    card = r["cards"][0]
    assert {"top_gender", "top_current_status", "top_area_type", "top_monthly_spending", "top_info_sources", "top_main_uses"} <= set(card)


def test_profile_text_v2_has_no_outcome_or_rating_words():
    for s in pg.build(write=False)["extended_specs"]:
        t = s["profile_text_v2"]
        assert not lib.prompt_leak_findings(t) and "rating" not in t.lower() and "q21" not in t.lower()


# ---- generators accept construct prompts only with their own folder
@pytest.mark.parametrize("script", ["pilot_generate_gemini.py", "pilot_generate_openai.py"])
def test_generators_need_explicit_out_dir_for_construct_prompts_and_dry_run_is_isolated(script, tmp_path):
    pf = str(ROOT / "data/construct_prompts/FT_PE01_prompts.jsonl")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--prompts-file", pf, "--dry-run"], capture_output=True, text=True)
    assert r.returncode == 2 and "--out-dir" in r.stdout
    out = tmp_path / "run"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--prompts-file", pf, "--out-dir", str(out), "--dry-run"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "dry run ok: 40 prompts" in r.stdout
    saved = [json.loads(l) for l in (out / "prompts.jsonl").open(encoding="utf-8")]
    assert len(saved) == 40 and saved[0]["task_id"] == "FT_PE01"


def test_generated_records_carry_task_metadata():
    class C:
        models = SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text="It helps me finish tasks quickly.", model_version="m",
                                                                              candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
                                                                              usage_metadata=None))
    p = {"synthetic_id": "P1_S01", "persona_id": "P1", "system": "s", "user": "u", "task_id": "FT_PE01", "construct": "PE", "question_id": "FT_PE01"}
    types = SimpleNamespace(GenerateContentConfig=lambda **k: k, HttpOptions=lambda **k: k)
    rec, err = gg.generate_one(C(), types, "m", p, 2, 5, {"quota_hits": 0})
    assert err is None and (rec["task_id"], rec["construct"], rec["question_id"]) == ("FT_PE01", "PE", "FT_PE01")
    oc = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        id="i", model="m", system_fingerprint=None, usage=None,
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="It helps me a lot.", refusal=None))]))))
    rec, err = og.generate_one(oc, "m", p, 2, {"max_rate_limit_hits": 5}, sleep=lambda s: None)
    assert err is None and rec["task_id"] == "FT_PE01" and rec["construct"] == "PE"


# ---- SSR: similarities -> PMF -> E[r], uncertainty, provenance
def test_score_detailed_matches_existing_scoring_and_exposes_similarities():
    texts = ["good phone", "not worth it", "average"]
    anchors = ["a1", "a2", "a3", "a4", "a5"]
    S, P, E = sd.score_detailed(texts, fake_embed, anchors)
    P2, E2 = cmp.score_texts(texts, fake_embed, anchors)
    assert S.shape == (3, 5) and np.allclose(P, P2) and np.allclose(E, E2) and np.allclose(P.sum(1), 1.0)
    from canonical_ssr import similarities_to_pmf
    assert np.allclose(similarities_to_pmf(S[0]), P[0])                                # PMF derived from the exposed similarities


def make_tmp_source(tmp_path, name, rows, task="Q26_BI_proxy"):
    d = tmp_path / name
    d.mkdir()
    (d / "r.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "prompts.jsonl").write_text("".join(json.dumps({"synthetic_id": r["synthetic_id"], "persona_id": r["persona_id"], "system": "s", "user": "u"}) + "\n" for r in rows), encoding="utf-8")
    t = dc.TASKS[task]
    return {"label": name, "provider": "fake", "dir": str(d), "responses": "r.jsonl", "enabled": True, "note": "t", "task_id": task,
            "construct": t["construct"], "question_id": t["question_id"], "profile_version": t["profile_version"], "human_reference": t["human_reference"]}


def test_dataset_has_similarities_provenance_and_per_construct_anchors(tmp_path):
    rec = {"synthetic_id": "P1_S01", "persona_id": "P1_premium_tier_owner", "response_text": "It helps me finish tasks.", "finish_reason": "STOP",
           "model_requested": "m1", "model_returned": "m1-x"}
    reg = {"pe": make_tmp_source(tmp_path, "pe", [rec], "FT_PE01"), "bi": make_tmp_source(tmp_path, "bi", [rec], "Q26_BI_proxy")}
    spec = {"synthetic_id": "P1_S01", "persona_id": "P1_premium_tier_owner", "brand": "Samsung", "spec_sha": "x", "age": 20}
    fac = {c: (lambda: fake_embed) for c in sd.ENCODERS}
    df, cov, man = sd.build_dataset(["pe", "bi"], {"v1": [spec], "v2": [spec]}, fac, registry=reg)
    pe, bi = df.set_index("source_key").loc["pe"], df.set_index("source_key").loc["bi"]
    assert pe.construct == "PE" and bi.construct == "BI" and pe.anchor_sha != bi.anchor_sha                 # anchors follow the construct
    assert pe.anchor_sha == dc.anchor_info("PE")["anchor_sha"] and pe.human_reference == "none" and bi.human_reference == "Q26_Q21_proxy"
    for cfg in sd.ENCODERS:
        s = [pe[f"{cfg}__s{k}"] for k in range(1, 6)]
        assert all(-1 <= v <= 1 for v in s)
        from canonical_ssr import similarities_to_pmf
        assert np.allclose(similarities_to_pmf(np.array(s)), [pe[f"{cfg}__p{k}"] for k in range(1, 6)])
    for c in ("model_requested", "prompt_sha", "task_id", "question_id", "persona_id", "status", "finish_reason", "spec_sha", "anchor_sha"):
        assert c in df.columns and df[c].notna().all()
    assert man["sources"]["pe"]["anchors"] == dc.anchor_info("PE")["statements"]


# ---- validation only compares sources that have a human reference
def test_validation_skips_sources_without_comparable_human_answers():
    rng = np.random.default_rng(0)
    rows = []
    for key, ref in (("a", "Q26_Q21_proxy"), ("b", "none")):
        d = pd.DataFrame({"source_key": key, "task_id": "t", "human_reference": ref, "synthetic_id": [f"P1_S{i}" for i in range(10)],
                          "persona_id": "P1_premium_tier_owner", "status": "scored"})
        for cfg in va.ENCODERS:
            P = rng.dirichlet(np.ones(5), 10)
            for k in range(5):
                d[f"{cfg}__p{k + 1}"] = P[:, k]
            d[f"{cfg}__E"], d[f"{cfg}__sd"], d[f"{cfg}__entropy"] = P @ np.arange(1, 6), 1.0, 1.0
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    P = rng.dirichlet(np.ones(5), 30)
    h = {cfg: pd.DataFrame({"persona_id": "P1_premium_tier_owner", "expected_rating": P @ np.arange(1, 6), "q21": 3,
                            **{f"p{k + 1}": P[:, k] for k in range(5)}}) for cfg in va.ENCODERS}
    dist = va.distribution_comparison(df, h, n_boot=50)
    assert set(dist.source_key) == {"a"}
    nc = va.not_comparable_sources(df)
    assert [x["source_key"] for x in nc] == ["b"] and "no human answers" in nc[0]["reason"]


def test_persona_ordering_direction_logic():
    df = pd.DataFrame({"source_key": "a", "human_reference": "Q26_Q21_proxy", "status": "scored",
                       "persona_id": ["P1_x"] * 3 + ["P2_y"] * 3, **{f"{c}__E": [4.0] * 3 + [3.0] * 3 for c in va.ENCODERS}})
    hum = {c: pd.DataFrame({"persona_id": ["P1_x", "P2_y"], "expected_rating": [3.5, 3.2], "q21": [4.0, 3.0]}) for c in va.ENCODERS}
    o = va.persona_ordering(df, hum)
    assert len(o) == 3 and o.same_direction_as_human_text.all() and o.same_direction_as_human_q21.all()
    hum2 = {c: pd.DataFrame({"persona_id": ["P1_x", "P2_y"], "expected_rating": [3.0, 3.5], "q21": [3.0, 4.0]}) for c in va.ENCODERS}
    assert not va.persona_ordering(df, hum2).same_direction_as_human_text.any()
    assert va.persona_ordering(df[df.persona_id == "P1_x"], hum).empty


# ---- runner
def test_runner_has_prompts_stage_and_lists_construct_sources():
    out = rp.status(print_fn=lambda *a: None)
    assert out["stages"]["construct prompts"] is True
    assert {"gemini_FT_PE01", "openai_FT_BI01"} <= set(out["sources"]) and out["sources"]["gemini_FT_PE01"]["responses"] == 0
    assert rp.main(["prompts"]) == 0
