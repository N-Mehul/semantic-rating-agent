import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import dp_persona_generator as pg
from scripts import dp_synthetic_dataset as sd
from scripts import dp_validation_agent as va
from scripts import run_pipeline as rp

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ["a1 anchor", "a2 anchor", "a3 anchor", "a4 anchor", "a5 anchor"]


def fake_embed(texts):
    """Deterministic pseudo-embeddings; no model is loaded."""
    out = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        out.append(np.random.default_rng(seed).normal(size=16))
    return np.array(out)


FACTORIES = {"local_mpnet_baseline": lambda: fake_embed, "local_minilm": lambda: fake_embed, "local_bge_base": lambda: fake_embed}


def spec(sid, persona, brand="Samsung", os_="Android"):
    return {"synthetic_id": sid, "persona_id": persona, "brand": brand, "operating_system": os_, "price_paid": "x", "usage_duration": "x",
            "replacement_frequency": "x", "upgrade_reason": "x", "daily_usage": "x", "area_type": "x", "gender": "x",
            "tech_comfort_1to5": 3, "spec_sha": "abc"}


def make_source(tmp_path, name, rows):
    d = tmp_path / name
    d.mkdir()
    (d / "gemini_generated_responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    ids = sorted({r["synthetic_id"] for r in rows} | {"P3_S01"})
    (d / "prompts.jsonl").write_text("".join(json.dumps({"synthetic_id": i, "persona_id": "P3_midrange_moderate_user" if i.startswith("P3") else "P1_premium_tier_owner",
                                                         "system": "s", "user": f"u {i}"}) + "\n" for i in ids), encoding="utf-8")
    return {"label": name, "provider": "fake", "dir": str(d), "responses": "gemini_generated_responses.jsonl", "enabled": True, "note": "t"}


def rec(sid, text, persona="P1_premium_tier_owner", fr="STOP"):
    return {"synthetic_id": sid, "persona_id": persona, "response_text": text, "finish_reason": fr, "model_requested": "m1"}


# ---- persona generator
def test_persona_generator_reproduces_existing_pilot_specs_exactly():
    r = pg.build(write=False)
    stripped = [{k: v for k, v in s.items() if k not in ("profile_text", "spec_sha")} for s in r["specs"]]
    existing = [json.loads(l) for l in open(ROOT / "data/pilot/synthetic_respondent_specs.jsonl", encoding="utf-8")]
    assert stripped == existing and len(r["specs"]) == 40
    assert r["specs"] == pg.build(write=False)["specs"]                          # deterministic


def test_persona_cards_cover_four_personas_and_exclude_outcomes():
    r = pg.build(write=False)
    ids = {c["persona_id"] for c in r["cards"]}
    assert len(ids) == 4 and all(len(c["synthetic_respondent_ids"]) == 10 for c in r["cards"])
    blob = json.dumps(r["cards"]) + json.dumps(r["specs"])
    assert "rebuy_brand" not in blob and "text_value_rebuy" not in blob and "q21" not in blob.lower().replace("q21 rebuy intention", "")
    assert all("Likert" not in s["profile_text"] and "rating" not in s["profile_text"].lower() for s in r["specs"])


# ---- dataset builder
def test_build_dataset_scores_flags_and_never_imputes(tmp_path):
    reg = {"srcA": make_source(tmp_path, "A", [rec("P1_S01", "Worth it, good camera."), rec("P1_S02", "It is worth it and the", fr="MAX_TOKENS"),
                                               rec("P1_S03", "Pretty good value.")])}
    specs = [spec("P1_S01", "P1_premium_tier_owner"), spec("P1_S02", "P1_premium_tier_owner"), spec("P1_S03", "P1_premium_tier_owner"),
             spec("P3_S01", "P3_midrange_moderate_user")]
    df, cov, man = sd.build_dataset(["srcA"], specs, FACTORIES, registry=reg, anchors=ANCHORS)
    assert len(df) == 3                                                           # no row for the never-generated P3_S01
    assert df.set_index("synthetic_id").status.to_dict() == {"P1_S01": "scored", "P1_S02": "excluded_quality", "P1_S03": "scored"}
    bad = df[df.status == "excluded_quality"].iloc[0]
    assert "finish_reason_not_stop" in bad.exclusion_reasons and np.isnan(bad["local_bge_base__E"])
    s = df[df.status == "scored"]
    for cfg in FACTORIES:
        P = s[[f"{cfg}__p{k}" for k in range(1, 6)]].values
        assert np.allclose(P.sum(1), 1.0, atol=1e-12) and np.allclose(s[f"{cfg}__E"], P @ np.arange(1, 6))
        assert (s[f"{cfg}__sd"] >= 0).all() and (s[f"{cfg}__entropy"] >= 0).all()
    assert (s.encoder_spread_sd >= 0).all() and s.prompt_sha.notna().all() and (s.spec_found).all()
    p3 = cov[(cov.persona_id == "P3_midrange_moderate_user")].iloc[0]
    assert (p3.responses_on_disk, p3.not_generated, p3.coverage_status) == (0, 1, "none")
    assert man["n_scored"] == 2 and "canonical_ssr.py" in man["sha256"]


def test_sources_stay_separate(tmp_path):
    reg = {"A": make_source(tmp_path, "A", [rec("P1_S01", "Good phone, worth it.")]),
           "B": make_source(tmp_path, "B", [rec("P1_S01", "Not great for the price.")])}
    df, _, _ = sd.build_dataset(["A", "B"], [spec("P1_S01", "P1_premium_tier_owner")], FACTORIES, registry=reg, anchors=ANCHORS)
    assert sorted(df.source_key) == ["A", "B"] and df.response_text.nunique() == 2


def test_real_dataset_matches_step8_score_exports_and_has_only_p1_p2():
    df = pd.read_csv(ROOT / "data/synthetic_dataset/synthetic_survey_dataset.csv", keep_default_na=False)
    g = df[df.source_key == "gemini_v2"]
    assert len(g) == 20 and set(g.persona_id.str[:2]) == {"P1", "P2"} and (g.status == "scored").all()
    for cfg in sd.ENCODERS:
        ref = pd.read_csv(ROOT / f"data/pilot_embedding_comparison/synthetic_scores_{cfg}_gemini_v2.csv").set_index("synthetic_id")
        assert np.allclose(g.set_index("synthetic_id")[f"{cfg}__E"].astype(float), ref.loc[g.synthetic_id, "expected_rating"], atol=1e-9)
    cov = pd.read_csv(ROOT / "data/synthetic_dataset/coverage_by_persona_source.csv")
    c34 = cov[(cov.source_key == "gemini_v2") & (cov.persona_id.str[:2].isin(["P3", "P4"]))]
    assert (c34.passed_quality_and_scored == 0).all() and (c34.not_generated == 10).all()
    assert not (df.source_key == "openai_v1").any()


# ---- validation agent
def human_frame(n=30, mean=3.5, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(5), n)
    return pd.DataFrame({"persona_id": "P1_premium_tier_owner", "expected_rating": P @ np.arange(1, 6), "q21": rng.integers(1, 6, n),
                         **{f"p{k}": P[:, k - 1] for k in range(1, 6)}})


def synth_df(n=10, source="s1", model="m1", seed=1):
    rng = np.random.default_rng(seed)
    rows = {"source_key": source, "model_requested": model, "synthetic_id": [f"P1_S{i:02d}" for i in range(n)],
            "persona_id": "P1_premium_tier_owner", "status": "scored", "brand": "Samsung", "operating_system": "Android",
            "response_text": "Worth it", "spec_spread": 0, "spec_found": True}
    df = pd.DataFrame(rows)
    for cfg in va.ENCODERS:
        P = rng.dirichlet(np.ones(5), n)
        for k in range(5):
            df[f"{cfg}__p{k + 1}"] = P[:, k]
        df[f"{cfg}__E"] = P @ np.arange(1, 6)
        df[f"{cfg}__sd"] = 1.0
        df[f"{cfg}__entropy"] = 1.0
    return df


def test_distribution_comparison_reports_ci_and_never_claims_equivalence():
    h = {cfg: human_frame() for cfg in va.ENCODERS}
    d = va.distribution_comparison(synth_df(10), h, n_boot=200)
    assert len(d) == 3 and (d.n_synthetic == 10).all() and (d.n_human == 30).all()
    assert d.diff_ci95_lo.notna().all() and (d.equivalence_claimed == False).all()   # noqa: E712
    assert np.allclose(d.diff_synth_minus_human, d.synthetic_mean_E - d.human_mean_E)


def test_no_interval_for_tiny_samples():
    h = {cfg: human_frame() for cfg in va.ENCODERS}
    d = va.distribution_comparison(synth_df(4), h, n_boot=100)
    assert d.diff_ci95_lo.isna().all() and not d.interval_includes_zero.any()


def test_persona_consistency_flags_os_contradiction():
    df = synth_df(3)
    df["response_text"] = ["My Samsung is great.", "I love my iPhone, switched from Samsung.", "Android is fine, Samsung is solid."]
    c = va.persona_consistency(df)
    assert c.os_contradiction.tolist() == [False, True, False]
    assert c.mentions_spec_brand.tolist() == [True, True, True] and c.mentions_other_brand.tolist() == [False, True, False]


def test_generation_stability_unsupported_then_computed():
    single = synth_df(5)
    assert va.generation_stability(single)["status"] == "not_supported_by_data"
    rerun = synth_df(5, source="s2", seed=2)                                       # same ids, same model, different source/run
    both = pd.concat([single, rerun], ignore_index=True)
    st = va.generation_stability(both)
    assert st["status"] == "computed" and st["n_repeated_specs"] == 5 and st["mean_within_spec_sd_E"] >= 0
    other_model = pd.concat([single, synth_df(5, source="s3", model="m2", seed=3)], ignore_index=True)
    assert va.generation_stability(other_model)["status"] == "not_supported_by_data"   # different model is not a repeat


def test_encoder_agreement_needs_minimum_n():
    assert va.encoder_agreement(synth_df(4)) == []
    assert len(va.encoder_agreement(synth_df(12))) == 3


def test_validation_run_on_real_outputs_writes_nothing_when_disabled():
    res = va.run(write=False)
    assert res["integrity"][0]["pmf_valid_and_E_consistent"] and res["stability"]["status"] == "not_supported_by_data"
    assert set(res["distribution"].persona_id.str[:2]) == {"P1", "P2"} and len(res["distribution"]) == 6
    assert (res["consistency"].os_contradiction == False).all()                      # noqa: E712


# ---- runner and safety
def test_runner_status_unknown_command_and_missing_prereq(monkeypatch, capsys):
    out = rp.status(print_fn=lambda *a: None)
    assert out["sources"]["gemini_v2"]["responses"] == 20 and out["sources"]["openai_v1"]["responses"] == 0
    assert rp.main(["bogus"]) == 2
    monkeypatch.setitem(rp.PREREQ, "persona assignments", ("data/does_not_exist.csv", "python scripts/build_personas.py"))
    assert rp.run_stage("dataset") == 2
    assert "missing persona assignments" in capsys.readouterr().out


def test_pipeline_modules_contain_no_network_or_llm_client_code():
    for name in ("dp_persona_generator", "dp_synthetic_dataset", "dp_validation_agent", "run_pipeline"):
        src = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
        for bad in ("import openai", "from openai", "google.genai", "from google import", "import requests", "urllib.request", ".chat.completions"):
            assert bad not in src, f"{name} must not call external APIs ({bad})"
