"""
Unit tests for src/pipeline.py
Tests end-to-end pipeline output structure.
"""

import pytest
from src.pipeline import run_pipeline


TEST_CLAIM = "Omega-3 fatty acids reduce inflammation"


# --- output structure ---

def test_pipeline_returns_dict():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert isinstance(result, dict)


def test_pipeline_has_required_keys():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert "claim" in result
    assert "method" in result
    assert "retrieved" in result
    assert "verdict" in result


def test_pipeline_claim_matches_input():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert result["claim"] == TEST_CLAIM


def test_pipeline_method_matches_input():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert result["method"] == "bm25"


def test_pipeline_retrieved_is_list():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert isinstance(result["retrieved"], list)


def test_pipeline_retrieved_not_empty():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert len(result["retrieved"]) > 0


def test_pipeline_verdict_is_string():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert isinstance(result["verdict"], str)


def test_pipeline_verdict_not_empty():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert len(result["verdict"]) > 50


def test_pipeline_verdict_contains_verdict_label():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    assert "VERDICT" in result["verdict"]


def test_pipeline_verdict_contains_evidence_sections():
    result = run_pipeline(TEST_CLAIM, method="bm25")
    verdict = result["verdict"]
    assert "SUPPORTING EVIDENCE" in verdict or "CONTRADICTING EVIDENCE" in verdict


def test_pipeline_dense_method():
    result = run_pipeline(TEST_CLAIM, method="dense")
    assert result["method"] == "dense"
    assert isinstance(result["verdict"], str)


def test_pipeline_queryreform_has_reformulated_query():
    result = run_pipeline(TEST_CLAIM, method="queryreform")
    assert result["reformulated_query"] is not None
    assert isinstance(result["reformulated_query"], str)
    assert len(result["reformulated_query"]) > 10
