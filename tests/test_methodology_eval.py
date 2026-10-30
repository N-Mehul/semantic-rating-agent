import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from canonical_ssr import cosine_similarities, expected_rating, similarities_to_pmf
from scripts import dp_methodology_eval as me
from scripts import pilot_generate_condition as gc

ROOT = Path(__file__).resolve().parents[1]


# ---------- frozen split
def test_split_is_written_once_then_only_verified(tmp_path):
    p = tmp_path / "split.json"
    first = me.freeze_split(ROOT, p)
    assert first["status"].startswith("frozen_now") and p.exists()
    again = me.freeze_split(ROOT, p)
    assert again["status"] == "verified_against_frozen_manifest" and again["sha256"] == first["sha256"]
    assert not set(first["dev_ids"]) & set(first["test_ids"]) and first["n_dev"] + first["n_test"] == 194


def test_a_changed_split_is_refused_not_silently_used(tmp_path):
    p = tmp_path / "split.json"
    me.freeze_split(ROOT, p)
    d = json.loads(p.read_text())
    d["sha256"] = "0" * 64
    p.write_text(json.dumps(d))
    with pytest.raises(RuntimeError, match="refusing"):
        me.freeze_split(ROOT, p)


def test_repository_manifest_matches_the_split_used_by_earlier_results():
    cur = me.current_split(ROOT)
    frozen = json.loads((ROOT / "data/validation_split/frozen_split_manifest.json").read_text(encoding="utf-8"))
    cm = json.loads((ROOT / "data/pilot_embedding_comparison/comparison_metrics.json").read_text(encoding="utf-8"))
    assert frozen["sha256"] == cur["sha256"] and (frozen["n_dev"], frozen["n_test"]) == (cm["n_dev"], cm["n_test"])


# ---------- metrics
def test_paired_metrics_on_perfect_agreement():
    q = np.array([1, 2, 3, 4, 5] * 4)
    m = me.paired_metrics(q.astype(float), q, q, n_boot=50)
    assert m["exact_accuracy_argmax"] == 1 and m["mae_expected_rating"] == 0
    for k in ("spearman_rho", "pearson_r", "cohen_kappa_linear", "cohen_kappa_quadratic", "cohen_kappa_unweighted"):
        assert m[k]["status"] == "computed" and m[k]["value"] == pytest.approx(1.0)


def test_undefined_metrics_are_unavailable_not_zero():
    q = np.array([1, 3, 3, 4, 5, 2, 3, 4, 3, 5, 2, 1])
    m = me.paired_metrics(np.full(len(q), 3.0), np.full(len(q), 3), q, n_boot=20)
    assert m["spearman_rho"]["status"] == "unavailable" and m["pearson_r"]["status"] == "unavailable"
    assert m["cohen_kappa_quadratic"]["status"] == "unavailable" and "two observed categories" in m["cohen_kappa_quadratic"]["reason"]
    assert me.paired_metrics([3.0] * 5, [3] * 5, [3] * 5)["status"] == "unavailable"


def test_kappa_variants_match_sklearn_definitions():
    from sklearn.metrics import cohen_kappa_score
    a = np.array([1, 2, 3, 3, 4, 5, 2, 4, 3, 5])
    b = np.array([1, 3, 3, 2, 4, 4, 2, 5, 3, 5])
    assert me.kappa(a, b, "quadratic")["value"] == pytest.approx(cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5], weights="quadratic"))
    assert me.kappa(a, b, None)["value"] == pytest.approx(cohen_kappa_score(a, b, labels=[1, 2, 3, 4, 5]))


def test_jsd_bounds_and_empty_input():
    assert me.jsd([0.2, 0.2, 0.2, 0.2, 0.2], [1, 1, 1, 1, 1])["value"] == pytest.approx(0.0, abs=1e-12)
    assert me.jsd([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])["value"] == pytest.approx(1.0)
    assert me.jsd([0, 0, 0, 0, 0], [1, 0, 0, 0, 0])["status"] == "unavailable"


def test_tstr_requires_independent_labels_and_enough_classes():
    X = np.random.default_rng(0).normal(size=(40, 3))
    y = np.array([1, 2] * 20)
    assert me.tstr(X, y, X, y, "ssr_derived")["status"] == "unavailable"
    assert "fewer than two classes" in me.tstr(X, np.ones(40), X, y, "independent")["reason"]
    ok = me.tstr(X, y, X, y, "independent")
    assert ok["status"] == "computed" and 0 <= ok["accuracy"] <= 1


# ---------- real outputs
def test_methodology_metrics_reproduce_the_earlier_held_out_metrics(tmp_path):
    split = me.freeze_split(ROOT, tmp_path / "s.json")
    cm = json.loads((ROOT / "data/pilot_embedding_comparison/comparison_metrics.json").read_text(encoding="utf-8"))["results"]["local_bge_base"]["test"]
    import pandas as pd
    h = pd.read_csv(ROOT / "data/pilot_embedding_comparison/human_scores_local_bge_base.csv")
    t = h[h.respondent_id.isin(split["test_ids"])]
    m = me.paired_metrics(t.expected_rating, t.argmax_rating, t.q21.astype(int), n_boot=50)
    assert m["n"] == cm["n"] and m["spearman_rho"]["value"] == pytest.approx(cm["spearman"]) and m["mae_expected_rating"] == pytest.approx(cm["mae"])
    assert m["exact_accuracy_argmax"] == pytest.approx(cm["accuracy_exact"])


def test_leakage_audit_on_real_data():
    a = me.leakage_audit(me.current_split(ROOT), ROOT)
    assert a["respondent_overlap_dev_test"] == 0 and a["duplicate_respondent_ids_primary"] == 0
    assert a["anchors_unchanged_since_step6"] and a["canonical_ssr_unchanged_since_step6"] and a["encoder_selected_on"] == "dev half only"
    assert a["persona_construction_used_test_respondents"] is True                     # documented, not hidden


# ---------- persona-conditioned vs control prompts
SPECS = [json.loads(l) for l in (ROOT / "data/persona_generator/respondent_specs_extended.jsonl").read_text(encoding="utf-8").splitlines()]


def test_control_prompts_contain_no_persona_information_and_match_persona_counts():
    per = gc.build_prompts("persona", "Q26_BI_proxy", 2, SPECS)
    ctl = gc.build_prompts("control", "Q26_BI_proxy", 2, SPECS)
    assert len(per) == len(ctl) == 8 and {p["condition"] for p in per} == {"persona"} and {p["condition"] for p in ctl} == {"control"}
    assert all(p["system"] == c["system"] for p, c in zip(per, ctl))                  # same instruction
    q = gc.task_question("Q26_BI_proxy", gc.load_codebook())
    for c in ctl:
        assert c["user"] == f"Question: {q}" and c["persona_id"] is None
        assert not any(s["profile_text_v1"][:40] in c["user"] or s["profile_text_v2"][:40] in c["user"] for s in SPECS)
    for p in per:
        spec = next(s for s in SPECS if s["synthetic_id"] == p["synthetic_id"])
        assert spec["profile_text_v1"] in p["user"] and q in p["user"]
    assert {p["persona_id"] for p in per} == set(gc.PERSONAS)


def test_condition_scoring_uses_canonical_ssr():
    def embed(xs):
        return np.array([np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)).normal(size=16) for t in xs])
    rows = [{"synthetic_id": "CTRL_01", "persona_id": None, "condition": "control", "response_text": "Yes, it is worth it and I would stay with the brand.",
             "finish_reason": "STOP", "model_returned": "m"},
            {"synthetic_id": "CTRL_02", "persona_id": None, "condition": "control", "response_text": "It was okay but the", "finish_reason": "MAX_TOKENS"}]
    df = gc.score(rows, "BI", embed)
    ok = df[df.status == "scored"].iloc[0]
    stmts = gc.anchor_info("BI")["statements"]
    pmf = similarities_to_pmf(cosine_similarities(embed([rows[0]["response_text"]])[0], embed(stmts)))
    assert [ok[f"p{k}"] for k in range(1, 6)] == pytest.approx(list(pmf)) and ok.expected_rating == pytest.approx(expected_rating(pmf))
    assert df.set_index("synthetic_id").status["CTRL_02"] == "excluded_quality"
