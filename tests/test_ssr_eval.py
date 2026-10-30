import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from canonical_ssr import expected_rating, similarities_to_pmf
from scripts import pilot_generate_api as gen
from scripts import ssr_eval_lib as lib

ROOT = Path(__file__).resolve().parents[1]


def test_pmf_and_expected_rating_consistent():
    pmf = similarities_to_pmf(np.array([0.1, 0.2, 0.4, 0.3, 0.25]))
    assert lib.check_pmf(pmf)
    assert expected_rating(pmf) == pytest.approx(float(pmf @ np.arange(1, 6)))
    assert not lib.check_pmf(np.array([0.5, 0.5, 0.2, 0, 0]))


def test_metrics_perfect_and_baselines():
    q = np.array([1, 2, 3, 4, 5] * 6)
    m = lib.metrics_vs_q21(q.astype(float), q, q, n_boot=50)
    assert m["spearman"] == pytest.approx(1.0) and m["mae"] == 0 and m["accuracy_exact"] == 1
    assert m["accuracy_baseline_majority"] == pytest.approx(0.2)
    assert m["q21_distribution"] == {1: 6, 2: 6, 3: 6, 4: 6, 5: 6}


def test_constant_predictor_matches_baseline_mae():
    q = np.array([1, 3, 3, 4, 5, 3, 2, 3, 4, 3] * 3)
    m = lib.metrics_vs_q21(np.full(len(q), q.mean()), np.full(len(q), 3), q, n_boot=20)
    assert m["mae"] == pytest.approx(m["mae_baseline_mean"])
    assert m["accuracy_exact"] == pytest.approx(m["accuracy_baseline_majority"])


def test_dev_test_split_disjoint_and_stratified():
    q = np.array([1] * 10 + [2] * 10 + [3] * 40 + [4] * 30 + [5] * 20)
    dev, test = lib.dev_test_split(q, 0.5, seed=1)
    assert not set(dev) & set(test) and len(dev) + len(test) == len(q)
    for k in range(1, 6):
        assert abs((q[dev] == k).sum() - (q[test] == k).sum()) <= 1
    assert (lib.dev_test_split(q, 0.5, seed=1)[0] == dev).all()


def test_prompt_leak_detection():
    assert lib.prompt_leak_findings("They answered Q21 as 4")
    assert lib.prompt_leak_findings("how likely to buy the same brand")
    assert not lib.prompt_leak_findings("Female student, brand Vivo, paid ₹10,000–₹20,000")


def test_generation_prompts_are_leak_free_and_have_no_outcome_fields():
    prompts = gen.build_prompts(gen.load_specs(40))
    assert len(prompts) == 40
    for p in prompts:
        assert not lib.prompt_leak_findings(p["system"] + p["user"])
    specs = gen.load_specs(40)
    assert not any(k in s for s in specs for k in ("q21", "rebuy_brand_1to5", "text_value_rebuy"))


def test_response_validation():
    ok = [{"synthetic_id": "a", "response_text": "Worth it, battery is good."}]
    assert lib.validate_responses(ok) == []
    bad = [{"synthetic_id": "a", "response_text": "I give it 4/5"}, {"synthetic_id": "a", "response_text": ""}]
    assert len(lib.validate_responses(bad)) >= 2


def test_manual_pilot_responses_valid_and_labelled_separately():
    rows = [json.loads(l) for l in open(ROOT / "data/pilot/synthetic_responses.jsonl", encoding="utf-8")]
    assert lib.validate_responses(rows) == []
    assert all("manual" in r["generator"] for r in rows)


def test_api_script_stops_without_key(monkeypatch):
    env = {k: v for k, v in __import__("os").environ.items() if k != "OPENAI_API_KEY"}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/pilot_generate_api.py")], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "OPENAI_API_KEY" in r.stdout


class _Err(Exception):
    def __init__(self, code):
        self.code = code


def test_openai_failure_is_skipped_and_recorded():
    from scripts import ssr_embedding_comparison as cmp
    skipped = {}

    def boom():
        raise _Err(429)
    assert cmp.run_config_safely("openai_text_embedding_3_small", boom, skipped) is None
    assert "429" in skipped["openai_text_embedding_3_small"] and "skipped" in skipped["openai_text_embedding_3_small"]


def test_local_failure_still_raises_and_success_passes_through():
    from scripts import ssr_embedding_comparison as cmp
    skipped = {}
    with pytest.raises(_Err):
        cmp.run_config_safely("local_minilm", lambda: (_ for _ in ()).throw(_Err(500)), skipped)
    assert skipped == {}
    assert cmp.run_config_safely("local_bge_base", lambda: "ok", skipped) == "ok"


def test_gemini_records_load_and_export_scores(tmp_path):
    import pandas as pd
    from scripts import ssr_embedding_comparison as cmp
    from canonical_ssr import cosine_similarities
    rows = [json.loads(l) for l in open(ROOT / "data/pilot_gemini/gemini_generated_responses.jsonl", encoding="utf-8")]
    assert len(rows) == len({r["synthetic_id"] for r in rows}) and all(r["source"] == "gemini_generated" for r in rows)
    rng = np.random.default_rng(0)
    anchors = rng.normal(size=(5, 8))
    E = rng.normal(size=(len(rows), 8))
    embed = lambda xs: anchors if len(xs) == 5 else E
    P, er = cmp.score_texts([r["response_text"] for r in rows], embed, ["a"] * 5)
    df = cmp.synthetic_score_frame(rows, P, er, "local_test", "gemini")
    assert list(df.synthetic_id) == [r["synthetic_id"] for r in rows]
    assert list(df.response_text) == [r["response_text"] for r in rows]
    assert {"persona_id", "expected_rating", "embedding_config", "source", "p1", "p2", "p3", "p4", "p5"} <= set(df.columns)
    assert np.allclose(df[[f"p{k}" for k in range(1, 6)]].sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(df.expected_rating, df[[f"p{k}" for k in range(1, 6)]].values @ np.arange(1, 6))
    sims = cosine_similarities(E[0], anchors)  # canonical path reproduces first row's argmin-zero PMF
    assert df.loc[0, "p1":"p5"].min() == pytest.approx(0.0) and np.argmin(sims) == int(np.argmin(df.loc[0, "p1":"p5"].values))


def test_safe_write_csv_never_overwrites(tmp_path):
    import pandas as pd
    from scripts import ssr_embedding_comparison as cmp
    a, b = pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})
    f = tmp_path / "s.csv"
    assert cmp.safe_write_csv(a, f) == f
    assert cmp.safe_write_csv(a, f) == f                      # identical: kept
    g = cmp.safe_write_csv(b, f)                              # differs: new sibling
    assert g.name == "s_v2.csv" and f.read_text().strip().endswith("1") and g.read_text().strip().endswith("2")
