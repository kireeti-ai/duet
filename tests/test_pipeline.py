"""
Unit tests for src/pipeline.py
Tests end-to-end pipeline output structure.
"""

import pytest
import src.pipeline as pipeline


TEST_CLAIM = "Omega-3 fatty acids reduce inflammation"
FAKE_VERDICT = """VERDICT: Contested

SUPPORTING EVIDENCE:
- Omega-3 studies report anti-inflammatory trends in selected cohorts.

CONTRADICTING EVIDENCE:
- Other trials report no statistically significant improvement.

MODERATING FACTORS:
- Dose, population, and baseline severity vary across studies.

CONFIDENCE: Moderate
"""


@pytest.fixture(autouse=True)
def mock_pipeline_dependencies(monkeypatch):
    def fake_retrieve(claim, method="dense", reformulated_query=None, top_k=30):
        return [
            {
                "id": "1001",
                "title": "Omega-3 anti-inflammatory effects",
                "text": "Some cohorts show reduced inflammatory markers.",
                "score": 0.95,
                "method": method,
            },
            {
                "id": "1002",
                "title": "No significant omega-3 benefit",
                "text": "Several randomized trials found no significant effect.",
                "score": 0.90,
                "method": method,
            },
        ][:top_k]

    def fake_rerank(query, retrieved, top_k=5):
        ranked = []
        for i, item in enumerate(retrieved[:top_k]):
            row = dict(item)
            row["rerank_score"] = float(1.0 - (i * 0.1))
            ranked.append(row)
        return ranked

    def fake_reformulate_query(claim):
        return f"{claim} including null-effect and contradictory evidence"

    def fake_prompt(claim, retrieved, variant):
        return f"Prompt for {claim} ({variant})"

    class _FakeCompletions:
        @staticmethod
        def create(model, messages, temperature, max_tokens):
            class _Message:
                content = FAKE_VERDICT

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _FakeChat:
        completions = _FakeCompletions()

    class FakeGroq:
        def __init__(self, api_key=None):
            self.chat = _FakeChat()

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "rerank", fake_rerank)
    monkeypatch.setattr(pipeline, "reformulate_query", fake_reformulate_query)
    monkeypatch.setattr(pipeline, "build_verdict_prompt", fake_prompt)
    monkeypatch.setattr(pipeline, "Groq", FakeGroq)


@pytest.fixture
def bm25_result():
    return pipeline.run_pipeline(TEST_CLAIM, method="bm25")


@pytest.fixture
def dense_result():
    return pipeline.run_pipeline(TEST_CLAIM, method="dense")


# --- output structure ---

def test_pipeline_returns_dict(bm25_result):
    assert isinstance(bm25_result, dict)


def test_pipeline_has_required_keys(bm25_result):
    assert "claim" in bm25_result
    assert "method" in bm25_result
    assert "retrieved" in bm25_result
    assert "verdict" in bm25_result


def test_pipeline_has_metadata_keys(bm25_result):
    assert "prompt_variant" in bm25_result
    assert "retrieval_candidate_k" in bm25_result
    assert "final_top_k" in bm25_result


def test_pipeline_claim_matches_input(bm25_result):
    assert bm25_result["claim"] == TEST_CLAIM


def test_pipeline_method_matches_input(bm25_result):
    assert bm25_result["method"] == "bm25"


def test_pipeline_retrieved_is_list(bm25_result):
    assert isinstance(bm25_result["retrieved"], list)


def test_pipeline_retrieved_not_empty(bm25_result):
    assert len(bm25_result["retrieved"]) > 0


def test_pipeline_verdict_is_string(bm25_result):
    assert isinstance(bm25_result["verdict"], str)


def test_pipeline_verdict_not_empty(bm25_result):
    assert len(bm25_result["verdict"]) > 50


def test_pipeline_verdict_contains_verdict_label(bm25_result):
    assert "VERDICT" in bm25_result["verdict"]


def test_pipeline_verdict_contains_evidence_sections(bm25_result):
    verdict = bm25_result["verdict"]
    assert "SUPPORTING EVIDENCE" in verdict or "CONTRADICTING EVIDENCE" in verdict


def test_pipeline_dense_method(dense_result):
    assert dense_result["method"] == "dense"
    assert isinstance(dense_result["verdict"], str)


def test_pipeline_queryreform_has_reformulated_query():
    result = pipeline.run_pipeline(TEST_CLAIM, method="queryreform")
    assert result["reformulated_query"] is not None
    assert isinstance(result["reformulated_query"], str)
    assert len(result["reformulated_query"]) > 10
