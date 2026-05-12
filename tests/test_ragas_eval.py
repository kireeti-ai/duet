import src.ragas_eval as ragas_eval


def test_build_ragas_embeddings_uses_current_constructor(monkeypatch):
    captured = {}

    class DummyEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ragas_eval, "RagasHFEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(ragas_eval, "_detect_device", lambda: "cpu")

    ragas_eval._build_ragas_embeddings()

    assert captured["model"] == ragas_eval.RAGAS_EMBED_MODEL
    assert captured["device"] == "cpu"
    assert captured["normalize_embeddings"] is True


def test_parse_ragas_result_with_ground_truth():
    result = {
        "faithfulness": [0.8],
        "answer_relevancy": [0.9],
        "context_precision": [0.7],
        "context_recall": [0.75],
    }

    parsed = ragas_eval._parse_ragas_result(result, ground_truth_provided=True)

    assert parsed["faithfulness"] == 0.8
    assert parsed["answer_relevancy"] == 0.9
    assert parsed["context_precision"] == 0.7
    assert parsed["context_recall"] == 0.75
    assert parsed["pass_thresholds"] is True
    assert parsed["metric_count"] == 4
