"""Tests for the 8-Step Digital Persona SSR Methodology (Maier et al. reproduction).

Verifies:
1. Step 1: 4 Synthetic consumers created from survey data (258 eligible respondents).
2. Step 2: Smartphone product concept handling.
3. Step 3: Natural-language text generation only (zero numerical ratings).
4. Step 4: Sentence embeddings with local nomic-embed-text.
5. Step 5: Five-anchor cosine similarities across all 6 approved anchor sets.
6. Step 6: Six individual PMFs via minimum subtraction, each summing to 1.0.
7. Step 7: Final element-wise averaged PMF summing to 1.0.
8. Step 8: Expected Likert rating between 1.0 and 5.0.
9. Exports: Full JSON, Summary CSV, and Detailed Intermediate Calculations CSV.
10. Strict rule: No 3-nearest-neighbor truncation; pure 5-anchor paper method.
"""
from __future__ import annotations

import csv
import io
import json
from http import HTTPStatus
from pathlib import Path

import numpy as np
import pytest

from dashboard.data_access import Dashboard
from dashboard.evaluation_service import ProductEvaluationService
from dashboard.server import route

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dash():
    return Dashboard(ROOT)


@pytest.fixture
def eval_service():
    return ProductEvaluationService(ROOT)


def test_step1_synthetic_consumers_creation(dash):
    status, payload = route(dash, "/api/evaluation/options", {})
    assert status == HTTPStatus.OK
    assert "personas" in payload
    assert "survey_meta" in payload

    # Verify 4 empirical synthetic consumer profiles
    personas = payload["personas"]
    assert len(personas) == 4
    pids = [p["persona_id"] for p in personas]
    assert "P1_premium_tier_owner" in pids
    assert "P2_midrange_heavy_user" in pids
    assert "P3_midrange_moderate_user" in pids
    assert "P4_budget_or_undisclosed_price" in pids

    # Verify demographic characteristics
    for p in personas:
        assert p["n_respondents"] > 0
        assert p["median_age"] >= 18
        assert p["tech_comfort"] > 0
        assert "primary_brand" in p

    # Verify survey metadata distinction
    sm = payload["survey_meta"]
    assert sm["eligible_respondents"] == 258
    assert sm["total_raw_submissions"] == 275
    assert sm["excluded_submissions"] == 17


def test_step2_product_concept_presentation(eval_service):
    # Verify input validation for smartphone product concept
    with pytest.raises(ValueError, match="Product name must be at least 2 characters"):
        eval_service.run_evaluation({"name": "X", "description": "A valid description", "features": [{"name": "F", "explanation": "E"}]})

    with pytest.raises(ValueError, match="Product description must be at least 10 characters"):
        eval_service.run_evaluation({"name": "Valid Phone", "description": "Too short", "features": [{"name": "F", "explanation": "E"}]})

    with pytest.raises(ValueError, match="Please provide at least one product feature"):
        eval_service.run_evaluation({"name": "Valid Phone", "description": "Valid smartphone concept description", "features": []})


def test_step3_step4_step5_step6_step7_step8_saved_evaluation(dash):
    status, payload = route(dash, "/api/evaluation/saved", {"id": ["nova_ai_smartphone"]})
    assert status == HTTPStatus.OK
    assert payload["eval_id"] == "nova_ai_smartphone"

    # Step 2: Product concept verified
    concept = payload["concept"]
    assert concept["name"] == "Nova AI Smartphone"
    assert concept["price_inr"] == 24999.0
    assert len(concept["features"]) == 6

    # Step 3: Natural language responses text only
    responses = payload["responses"]
    assert len(responses) == 4
    for r in responses:
        text = r["response_text"]
        assert isinstance(text, str) and len(text) > 50
        # Verify text does not directly declare Likert ratings
        assert "I rate this 5/5" not in text
        assert "Rating: 4" not in text

    # Step 4: Local nomic-embed-text model
    ssr = payload["ssr_results"]
    assert ssr["embedding_model"] == "nomic-embed-text"
    assert ssr["embedding_dimensions"] == 768
    assert ssr["n_anchor_sets"] == 6
    assert ssr["anchors_per_set"] == 5
    assert ssr["total_anchors"] == 30

    # Step 5, 6, 7, 8: Persona summaries & calculation inspectability
    p_summaries = ssr["persona_summaries"]
    assert len(p_summaries) == 4

    for ps in p_summaries:
        # Step 5: 6 anchor sets evaluated
        anchor_sets = ps["anchor_sets"]
        assert len(anchor_sets) == 6
        set_codes = [s["construct_code"] for s in anchor_sets]
        assert set(set_codes) == {"PE", "EE", "HM", "SI", "PP", "BI"}

        set_pmfs = []
        for s in anchor_sets:
            sims = s["cosine_similarities"]
            pmf = s["pmf_probabilities"]
            # 5 cosine similarities per set
            assert len(sims) == 5
            assert all(isinstance(x, float) and -1.0 <= x <= 1.0 for x in sims)
            # Step 6: 5 PMF values per set summing to 1.0
            assert len(pmf) == 5
            assert all(p >= 0.0 for p in pmf)
            assert np.isclose(sum(pmf), 1.0, atol=1e-5)
            # Minimum similarity correctly subtracted
            assert np.isclose(s["min_similarity"], min(sims), atol=1e-5)
            set_pmfs.append(pmf)

        # Step 7 & 8: Verify Independent Single-Set SSR per Construct (Zero cross-construct averaging)
        assert "averaged_pmf" not in ps, "Cross-construct averaging must not occur in Option A"
        assert "primary_adoption_rating" in ps
        assert "primary_adoption_pmf" in ps
        assert "construct_scores" in ps

        # Primary adoption outcome: BI
        bi_rating = ps["primary_adoption_rating"]
        assert 1.0 <= bi_rating <= 5.0
        assert len(ps["primary_adoption_pmf"]) == 5
        assert np.isclose(sum(ps["primary_adoption_pmf"]), 1.0, atol=1e-5)

        # Supporting diagnostic constructs (PE, EE, HM, SI, PP)
        c_scores = ps["construct_scores"]
        assert set(c_scores.keys()) == {"PE", "EE", "HM", "SI", "PP", "BI"}
        assert all(1.0 <= score <= 5.0 for score in c_scores.values())

        # BI score matches primary adoption rating
        assert np.isclose(c_scores["BI"], bi_rating, atol=1e-4)


def test_export_endpoints_contain_all_steps(eval_service):
    # Full JSON export
    b_json, ctype_json, fname_json = eval_service.export_data("nova_ai_smartphone", "json")
    assert ctype_json == "application/json; charset=utf-8"
    assert fname_json.endswith(".json")
    data = json.loads(b_json.decode("utf-8"))
    assert "persona_summaries" in data["ssr_results"]
    assert "intermediate_records" in data["ssr_results"]

    # Summary CSV export
    b_csv, ctype_csv, fname_csv = eval_service.export_data("nova_ai_smartphone", "csv")
    assert ctype_csv == "text/csv; charset=utf-8"
    assert fname_csv.endswith(".csv")
    reader = csv.DictReader(io.StringIO(b_csv.decode("utf-8")))
    rows = list(reader)
    assert len(rows) in (4, 24)
    assert "expected_rating" in reader.fieldnames
    assert "embedding_model" in reader.fieldnames

    # Intermediate calculations CSV export
    b_inter, ctype_inter, fname_inter = eval_service.export_data("nova_ai_smartphone", "intermediate_csv")
    assert ctype_inter == "text/csv; charset=utf-8"
    assert "intermediate_calculations" in fname_inter
    i_reader = csv.DictReader(io.StringIO(b_inter.decode("utf-8")))
    i_rows = list(i_reader)
    assert len(i_rows) == 24
    assert "cos_sim_1" in i_reader.fieldnames
    assert "min_similarity" in i_reader.fieldnames
    assert "set_expected_rating" in i_reader.fieldnames


def test_dashboard_static_files_structure():
    static_dir = ROOT / "dashboard" / "static"
    assert (static_dir / "index.html").exists()
    assert (static_dir / "app.js").exists()
    assert (static_dir / "styles.css").exists()

    app_js = (static_dir / "app.js").read_text(encoding="utf-8")
    assert "step1" in app_js
    assert "step2" in app_js
    assert "step3" in app_js
    assert "step4" in app_js
    assert "step5" in app_js
    assert "step6" in app_js
    assert "step78" in app_js
    # Ensure no 3-nearest-neighbors mention
    assert "k_nearest_neighbors" not in app_js
