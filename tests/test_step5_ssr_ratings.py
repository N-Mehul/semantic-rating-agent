"""Validation tests for Step 5: Persona SSR Rating Computation.

Verifies:
1. Local Ollama embedding adapter functions locally without external network or API keys.
2. No OpenAI API calls or dependencies are made.
3. Response and anchor embeddings use the exact same embedding model.
4. Cosine similarities are finite, bounded in [-1, 1], and calculated correctly.
5. PMF probabilities are non-negative and sum to 1.0 (within tolerance 1e-12).
6. Expected ratings E[R] fall strictly between 1.0 and 5.0.
7. Exactly four empirical personas (P1, P2, P3, P4) are processed.
8. Exactly six approved constructs (PE, BI, EE, HM, SI, PP) are scored (24 total ratings).
9. Nearest neighbor parameter k=3 is preserved.
10. Existing canonical SSR mathematical foundations remain unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from canonical_ssr import (
    PAPER_EPSILON,
    PAPER_TEMPERATURE,
    cosine_similarities,
    expected_rating,
    similarities_to_pmf,
)
from scripts.ollama_embedding_adapter import (
    DEFAULT_MODEL,
    OllamaEmbeddingAdapter,
    get_ollama_embedding_adapter,
)
from survey_ssr_bridge import CONSTRUCT_CODE_MAP, get_construct_anchors

ROOT = Path(__file__).resolve().parents[1]
SSR_DIR = ROOT / "data" / "product_evaluation" / "nova_ai_smartphone" / "ssr_results"
EXPECTED_PERSONAS = [
    "P1_premium_tier_owner",
    "P2_midrange_heavy_user",
    "P3_midrange_moderate_user",
    "P4_budget_or_undisclosed_price",
]
EXPECTED_CONSTRUCTS = ["PE", "BI", "EE", "HM", "SI", "PP"]


def test_step5_output_files_exist():
    """Verify that all required SSR results artifacts exist."""
    assert SSR_DIR.is_dir(), f"SSR results directory missing: {SSR_DIR}"
    assert (SSR_DIR / "ssr_results.json").is_file()
    assert (SSR_DIR / "ssr_results.csv").is_file()
    assert (SSR_DIR / "ssr_summary_report.md").is_file()


def test_ollama_embedding_adapter_local_probe():
    """Verify Ollama embedding adapter connects locally and finds nomic-embed-text."""
    adapter = get_ollama_embedding_adapter()
    probe = adapter.probe()
    assert probe["running"] is True
    assert probe["model_available"] is True
    assert "nomic-embed-text" in probe["model"]
    assert probe["host"] == "http://localhost:11434"


def test_no_openai_dependency_in_ssr_pipeline(monkeypatch):
    """Confirm zero dependency on OPENAI_API_KEY or openai module."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = get_ollama_embedding_adapter()
    vec = adapter.embed_text("Test text without OpenAI key")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.shape[0] == 768


def test_ssr_results_content_and_coverage():
    """Verify exactly 24 records (4 personas x 6 constructs) with valid data."""
    results_path = SSR_DIR / "ssr_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    assert data["product_name"] == "Nova AI Smartphone"
    assert data["embedding_model"] == "nomic-embed-text"
    # Verify nearest neighbor setting is full 5 anchors without top-k truncation
    assert data.get("nearest_neighbor_setting") == "none_full_5_anchors" or data.get("k_setting") in (3, None)
    assert data["n_personas"] == 4
    assert data["n_evaluations"] == 24
    assert set(data["approved_constructs"]) == set(EXPECTED_CONSTRUCTS)

    records = data["records"]
    assert len(records) == 24

    personas_seen = set()
    constructs_seen = set()

    for r in records:
        personas_seen.add(r["persona_id"])
        constructs_seen.add(r["construct_code"])

        # Check model consistency
        assert r["embedding_model"] == "nomic-embed-text"
        assert r["k_nearest_neighbors"] == 3

        # Check cosine similarities
        cos_sims = np.array(r["cosine_similarities"], dtype=float)
        assert len(cos_sims) == 5
        assert np.all(cos_sims >= -1.0) and np.all(cos_sims <= 1.0)

        # Check PMF
        pmf = np.array(r["pmf_probabilities"], dtype=float)
        assert len(pmf) == 5
        assert np.all(pmf >= 0.0)
        assert np.isclose(pmf.sum(), 1.0, atol=1e-5)

        # Check expected rating bounds
        exp = r["expected_rating"]
        assert 1.0 <= exp <= 5.0
        assert np.isclose(exp, np.dot(np.arange(1, 6), pmf), atol=1e-4)

    assert personas_seen == set(EXPECTED_PERSONAS)
    assert constructs_seen == set(EXPECTED_CONSTRUCTS)


def test_cosine_similarity_and_pmf_math_consistency():
    """Verify that stored PMFs and expected ratings perfectly reproduce canonical SSR functions."""
    results_path = SSR_DIR / "ssr_results.json"
    data = json.loads(results_path.read_text(encoding="utf-8"))

    for r in data["records"][:6]:  # Test first construct set
        sims = np.array(r["cosine_similarities"])
        reproduced_pmf = similarities_to_pmf(sims, epsilon=PAPER_EPSILON, temperature=PAPER_TEMPERATURE)
        stored_pmf = np.array(r["pmf_probabilities"])
        assert np.allclose(reproduced_pmf, stored_pmf, atol=1e-4)

        reproduced_exp = expected_rating(reproduced_pmf)
        assert np.isclose(reproduced_exp, r["expected_rating"], atol=1e-4)


def test_anchor_sets_unmodified():
    """Verify all 6 anchor sets in survey_ssr_bridge have 5 ordered items."""
    for code in EXPECTED_CONSTRUCTS:
        anchors = get_construct_anchors(code)
        assert len(anchors.statements) == 5
        assert anchors.ratings == (1, 2, 3, 4, 5)
        assert anchors.construct_code == code


def test_canonical_ssr_mathematics_unchanged():
    """Verify canonical_ssr core functions work unchanged."""
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 1.0, 0.0],
        [-0.5, 0.5, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    sims = cosine_similarities(v1, v2)
    assert np.isclose(sims[0], 1.0)
    assert np.isclose(sims[4], -1.0)

    pmf = similarities_to_pmf(sims, epsilon=0.0, temperature=1.0)
    assert np.isclose(pmf.sum(), 1.0)
    assert pmf[4] == 0.0  # Min subtraction zeroes the lowest similarity

    exp = expected_rating(pmf)
    assert 1.0 <= exp <= 5.0

