"""
Unit tests for src/reranker.py
"""

from src import reranker


def test_reranker_initializes_cross_encoder_once(monkeypatch):
    class DummyCrossEncoder:
        init_count = 0

        def __init__(self, model_name):
            DummyCrossEncoder.init_count += 1
            self.model_name = model_name

        def predict(self, pairs):
            return [0.3] * len(pairs)

    monkeypatch.setattr(reranker, "CrossEncoder", DummyCrossEncoder)
    reranker._cross_encoder_model = None

    retrieved = [{"id": "1", "title": "T1", "text": "A", "score": 0.1, "method": "dense"}]

    reranker.rerank("claim", list(retrieved), top_k=1)
    reranker.rerank("claim", list(retrieved), top_k=1)

    assert DummyCrossEncoder.init_count == 1


def test_rerank_preserves_scoring_shape_and_order(monkeypatch):
    class DummyCrossEncoder:
        def __init__(self, model_name):
            self.model_name = model_name

        def predict(self, pairs):
            return [0.1, 0.9, 0.5]

    monkeypatch.setattr(reranker, "CrossEncoder", DummyCrossEncoder)
    reranker._cross_encoder_model = None

    retrieved = [
        {"id": "1", "title": "T1", "text": "A", "score": 0.1, "method": "dense"},
        {"id": "2", "title": "T2", "text": "B", "score": 0.2, "method": "dense"},
        {"id": "3", "title": "T3", "text": "C", "score": 0.3, "method": "dense"},
    ]

    reranked = reranker.rerank("claim", retrieved, top_k=2)

    assert len(reranked) == 2
    assert [r["id"] for r in reranked] == ["2", "3"]
    assert all(isinstance(r["rerank_score"], float) for r in reranked)

