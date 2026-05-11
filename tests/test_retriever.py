"""
Unit tests for src/retriever.py
Tests all four retrieval strategies return correct structure.
"""

import pytest
from src.retriever import (
    dense_retrieve,
    bm25_retrieve,
    hybrid_retrieve,
    retrieve,
    load_resources
)
from src.config import TOP_K


# shared fixture -- load resources once for all tests
@pytest.fixture(scope="module")
def resources():
    embedding_model, collection, bm25, abstracts = load_resources()
    return embedding_model, collection, bm25, abstracts


TEST_CLAIM = "Vitamin D supplementation improves depression symptoms"


# --- result structure checks ---

def test_dense_returns_list(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = dense_retrieve(TEST_CLAIM, embedding_model, collection, top_k=5)
    assert isinstance(results, list)


def test_dense_returns_correct_count(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = dense_retrieve(TEST_CLAIM, embedding_model, collection, top_k=5)
    assert len(results) == 5


def test_dense_result_has_required_fields(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = dense_retrieve(TEST_CLAIM, embedding_model, collection, top_k=3)
    for r in results:
        assert "id" in r
        assert "title" in r
        assert "text" in r
        assert "score" in r
        assert "method" in r
        assert r["method"] == "dense"


def test_bm25_returns_list(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = bm25_retrieve(TEST_CLAIM, bm25, abstracts, top_k=5)
    assert isinstance(results, list)


def test_bm25_returns_correct_count(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = bm25_retrieve(TEST_CLAIM, bm25, abstracts, top_k=5)
    assert len(results) == 5


def test_bm25_result_has_required_fields(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = bm25_retrieve(TEST_CLAIM, bm25, abstracts, top_k=3)
    for r in results:
        assert "id" in r
        assert "title" in r
        assert "text" in r
        assert "score" in r
        assert "method" in r
        assert r["method"] == "bm25"


def test_hybrid_returns_list(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = hybrid_retrieve(TEST_CLAIM, embedding_model, collection, bm25, abstracts, top_k=5)
    assert isinstance(results, list)


def test_hybrid_returns_correct_count(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = hybrid_retrieve(TEST_CLAIM, embedding_model, collection, bm25, abstracts, top_k=5)
    assert len(results) == 5


def test_hybrid_result_method_label(resources):
    embedding_model, collection, bm25, abstracts = resources
    results = hybrid_retrieve(TEST_CLAIM, embedding_model, collection, bm25, abstracts, top_k=3)
    for r in results:
        assert r["method"] == "hybrid"


# --- retrieve() main function ---

def test_retrieve_dense():
    results = retrieve(TEST_CLAIM, method="dense")
    assert len(results) == TOP_K
    assert all(r["method"] == "dense" for r in results)


def test_retrieve_bm25():
    results = retrieve(TEST_CLAIM, method="bm25")
    assert len(results) == TOP_K
    assert all(r["method"] == "bm25" for r in results)


def test_retrieve_hybrid():
    results = retrieve(TEST_CLAIM, method="hybrid")
    assert len(results) == TOP_K
    assert all(r["method"] == "hybrid" for r in results)


def test_retrieve_invalid_method():
    with pytest.raises(ValueError):
        retrieve(TEST_CLAIM, method="invalid_method")


def test_retrieve_scores_are_floats():
    results = retrieve(TEST_CLAIM, method="dense")
    for r in results:
        assert isinstance(r["score"], float)


def test_retrieve_ids_are_strings():
    results = retrieve(TEST_CLAIM, method="bm25")
    for r in results:
        assert isinstance(r["id"], str)


def test_retrieve_top_k_override():
    results = retrieve(TEST_CLAIM, method="bm25", top_k=7)
    assert len(results) == 7
