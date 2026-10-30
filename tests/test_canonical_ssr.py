"""Deterministic tests for the isolated paper-method SSR implementation."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from canonical_ssr import (
    ANCHOR_SET_COUNT,
    EMBEDDING_MODEL,
    cosine_similarities,
    expected_rating,
    openai_text_embedding_3_small,
    score_embedding_over_six_sets,
    score_text_over_six_sets,
    similarities_to_pmf,
)


TOLERANCE = 1e-12


def _forward_anchor_set() -> np.ndarray:
    return np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(0.75)],
        [0.0, 1.0],
        [-0.5, np.sqrt(0.75)],
        [-1.0, 0.0],
    ])


def _six_embedding_sets() -> list[np.ndarray]:
    forward = _forward_anchor_set()
    reverse = forward[::-1].copy()
    return [forward.copy() if index < 3 else reverse.copy()
            for index in range(ANCHOR_SET_COUNT)]


def _six_text_sets() -> list[list[str]]:
    return [[f"set-{set_index}-rating-{rating}" for rating in range(1, 6)]
            for set_index in range(ANCHOR_SET_COUNT)]


def _embedding_item(index, embedding):
    return SimpleNamespace(index=index, embedding=embedding)


def _fake_openai_client(response, expected_inputs=None):
    class Embeddings:
        @staticmethod
        def create(model, input):
            assert model == EMBEDDING_MODEL
            if expected_inputs is not None:
                assert input == expected_inputs
            return response

    return SimpleNamespace(embeddings=Embeddings())


def test_cosine_similarities_match_ordered_reference_vectors():
    similarities = cosine_similarities(np.array([1.0, 0.0]), _forward_anchor_set())
    assert np.allclose(similarities, [1.0, 0.5, 0.0, -0.5, -1.0], atol=TOLERANCE)


def test_paper_minimum_subtraction_produces_valid_five_rating_pmf():
    probabilities = similarities_to_pmf(np.array([1.0, 0.5, 0.0, -0.5, -1.0]))
    assert probabilities.shape == (5,)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities, [0.4, 0.3, 0.2, 0.1, 0.0], atol=TOLERANCE)
    assert np.isclose(probabilities.sum(), 1.0, atol=TOLERANCE, rtol=0.0)


def test_epsilon_is_added_only_to_first_minimum_rating():
    probabilities = similarities_to_pmf(
        np.array([1.0, 0.5, 0.0, -0.5, -1.0]), epsilon=0.5
    )
    expected = np.array([2.0, 1.5, 1.0, 0.5, 0.5]) / 5.5
    assert np.allclose(probabilities, expected, atol=TOLERANCE)
    assert np.isclose(probabilities.sum(), 1.0, atol=TOLERANCE, rtol=0.0)


def test_temperature_one_preserves_paper_study_pmf():
    probabilities = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    assert np.array_equal(similarities_to_pmf(
        np.array([1.0, 0.5, 0.0, -0.5, -1.0]), temperature=1.0
    ), probabilities)


def test_non_unit_temperature_applies_inverse_temperature_power_and_normalizes():
    probabilities = similarities_to_pmf(
        np.array([1.0, 0.5, 0.0, -0.5, -1.0]), temperature=0.5
    )
    base_pmf = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
    expected = base_pmf ** (1.0 / 0.5)
    expected /= expected.sum()
    assert np.allclose(probabilities, expected, atol=TOLERANCE)
    assert np.isclose(probabilities.sum(), 1.0, atol=TOLERANCE, rtol=0.0)


def test_six_anchor_sets_are_scored_then_averaged():
    result = score_embedding_over_six_sets(
        np.array([1.0, 0.0]), _six_embedding_sets()
    )
    assert result.similarities.shape == (6, 5)
    assert result.per_set_probabilities.shape == (6, 5)
    assert result.probabilities.shape == (5,)
    assert np.all(result.per_set_probabilities >= 0.0)
    assert np.all(result.per_set_probabilities <= 1.0)
    assert np.allclose(
        result.per_set_probabilities.sum(axis=1), 1.0, atol=TOLERANCE, rtol=0.0
    )
    assert np.all(result.probabilities >= 0.0)
    assert np.all(result.probabilities <= 1.0)
    assert np.isclose(result.probabilities.sum(), 1.0, atol=TOLERANCE, rtol=0.0)
    assert np.allclose(result.probabilities, np.full(5, 0.2), atol=TOLERANCE)


def test_expected_rating_uses_rating_values_one_through_five():
    assert expected_rating(np.array([0.4, 0.3, 0.2, 0.1, 0.0])) == pytest.approx(2.0)


def test_text_scoring_encodes_response_and_all_thirty_anchor_statements_once():
    anchor_sets = [[f"set-{set_index}-rating-{rating}" for rating in range(1, 6)]
                   for set_index in range(6)]
    encoded_inputs: list[str] = []

    def deterministic_encoder(texts):
        encoded_inputs.extend(texts)
        vectors = []
        for text in texts:
            if text == "fixed test response":
                vectors.append([1.0, 0.0])
                continue
            _, set_index, _, rating = text.split("-")
            rating_index = int(rating) - 1
            if int(set_index) >= 3:
                rating_index = 4 - rating_index
            vectors.append(_forward_anchor_set()[rating_index])
        return np.asarray(vectors)

    result = score_text_over_six_sets(
        "fixed test response", anchor_sets, deterministic_encoder
    )

    assert len(encoded_inputs) == 31
    assert encoded_inputs[0] == "fixed test response"
    assert result.probabilities.shape == (5,)
    assert np.allclose(result.probabilities.sum(), 1.0, atol=TOLERANCE)
    assert result.expected_rating == pytest.approx(3.0)


def test_zero_denominator_is_reported_instead_of_inventing_a_pmf():
    with pytest.raises(ValueError, match="equal similarities"):
        similarities_to_pmf(np.full(5, 0.25))


def test_six_anchor_count_is_required():
    with pytest.raises(ValueError, match="exactly six"):
        score_embedding_over_six_sets(np.array([1.0, 0.0]), _six_embedding_sets()[:5])


@pytest.mark.parametrize("response, anchors, message", [
    (np.array([[1.0, 0.0]]), _forward_anchor_set(), "one-dimensional"),
    (np.array([1.0, 0.0, 0.0]), _forward_anchor_set(), "dimensions must match"),
    (np.array([np.nan, 0.0]), _forward_anchor_set(), "finite values"),
    (np.array([np.inf, 0.0]), _forward_anchor_set(), "finite values"),
    (np.zeros(2), _forward_anchor_set(), "zero-norm"),
    (np.array([1.0, 0.0]), np.zeros((5, 2)), "zero-norm"),
    (np.array([1.0, 0.0]), np.zeros((4, 2)), "shape"),
])
def test_cosine_similarities_reject_invalid_embeddings(response, anchors, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarities(response, anchors)


@pytest.mark.parametrize("anchor_sets", [
    None,
    "abcdef",
    {"set1": ["a", "b", "c", "d", "e"]},
    _six_text_sets()[:5],
    ["abcde"] + _six_text_sets()[1:],
    [None] + _six_text_sets()[1:],
    [{"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}] + _six_text_sets()[1:],
    [["a", "b", "c", "d"]] + _six_text_sets()[1:],
    [["a", "b", "", "d", "e"]] + _six_text_sets()[1:],
    [["a", "b", 3, "d", "e"]] + _six_text_sets()[1:],
])
def test_text_scoring_rejects_malformed_anchor_sets(anchor_sets):
    with pytest.raises(ValueError):
        score_text_over_six_sets("fixed response", anchor_sets, lambda texts: None)


def test_openai_adapter_requests_paper_embedding_model_and_orders_results():
    response = SimpleNamespace(data=[
        _embedding_item(1, [0.0, 1.0]),
        _embedding_item(0, [1.0, 0.0]),
    ])
    vectors = openai_text_embedding_3_small(
        ["first", "second"],
        _fake_openai_client(response, expected_inputs=["first", "second"]),
    )
    assert np.array_equal(vectors, [[1.0, 0.0], [0.0, 1.0]])


@pytest.mark.parametrize("items, message", [
    ([_embedding_item(0, [1.0, 0.0]), _embedding_item(0, [0.0, 1.0])], "indices"),
    ([_embedding_item(-1, [1.0, 0.0]), _embedding_item(1, [0.0, 1.0])], "indices"),
    ([_embedding_item(0, [1.0, 0.0]), _embedding_item(2, [0.0, 1.0])], "indices"),
])
def test_openai_adapter_rejects_duplicate_negative_and_out_of_range_indices(items, message):
    response = SimpleNamespace(data=items)
    with pytest.raises(ValueError, match=message):
        openai_text_embedding_3_small(
            ["first", "second"], _fake_openai_client(response)
        )


def test_openai_adapter_rejects_missing_index():
    response = SimpleNamespace(data=[
        SimpleNamespace(embedding=[1.0, 0.0]),
        _embedding_item(1, [0.0, 1.0]),
    ])
    with pytest.raises(ValueError, match="contain an index"):
        openai_text_embedding_3_small(
            ["first", "second"], _fake_openai_client(response)
        )


@pytest.mark.parametrize("response, message", [
    (SimpleNamespace(), "contain a data sequence"),
    (SimpleNamespace(data=None), "contain a data sequence"),
    (SimpleNamespace(data=[_embedding_item(0, [1.0, 0.0])]), "different number"),
])
def test_openai_adapter_rejects_missing_or_wrong_count_data(response, message):
    with pytest.raises(ValueError, match=message):
        openai_text_embedding_3_small(
            ["first", "second"], _fake_openai_client(response)
        )


@pytest.mark.parametrize("embeddings, message", [
    ([[1.0, 0.0], [0.5]], "consistent dimensions"),
    ([[[1.0, 0.0]], [0.0, 1.0]], "one-dimensional"),
    ([[np.nan, 0.0], [0.0, 1.0]], "finite values"),
    ([[np.inf, 0.0], [0.0, 1.0]], "finite values"),
    ([[0.0, 0.0], [0.0, 1.0]], "non-zero norm"),
    (["not-a-vector", [0.0, 1.0]], "numeric vector"),
])
def test_openai_adapter_rejects_malformed_vectors(embeddings, message):
    items = [_embedding_item(index, vector)
             for index, vector in enumerate(embeddings)]
    response = SimpleNamespace(data=items)
    with pytest.raises(ValueError, match=message):
        openai_text_embedding_3_small(
            ["first", "second"], _fake_openai_client(response)
        )


def test_openai_adapter_rejects_item_without_embedding():
    items = [
        SimpleNamespace(index=0),
        _embedding_item(1, [0.0, 1.0]),
    ]
    response = SimpleNamespace(data=items)
    with pytest.raises(ValueError, match="numeric vector"):
        openai_text_embedding_3_small(
            ["first", "second"], _fake_openai_client(response)
        )
