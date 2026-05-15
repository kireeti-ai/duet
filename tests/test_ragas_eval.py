import src.ragas_eval as ragas_eval


def test_build_ragas_embeddings_uses_current_constructor(monkeypatch):
    captured = {}

    class DummyEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ragas_eval, "USE_LEGACY_EVAL_METRICS", False)
    monkeypatch.setattr(ragas_eval, "ModernRagasHFEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(ragas_eval, "_detect_device", lambda: "cpu")

    ragas_eval._build_ragas_embeddings()

    assert captured["model"] == ragas_eval.RAGAS_EMBED_MODEL
    assert captured["device"] == "cpu"
    assert captured["normalize_embeddings"] is True


def test_build_ragas_embeddings_uses_legacy_constructor(monkeypatch):
    captured = {}

    class DummyEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ragas_eval, "USE_LEGACY_EVAL_METRICS", True)
    monkeypatch.setattr(ragas_eval, "LangchainHFEmbeddings", DummyEmbeddings)
    monkeypatch.setattr(ragas_eval, "_detect_device", lambda: "cpu")

    ragas_eval._build_ragas_embeddings()

    assert captured["model_name"] == ragas_eval.RAGAS_EMBED_MODEL
    assert captured["model_kwargs"] == {"device": "cpu"}
    assert captured["encode_kwargs"] == {"normalize_embeddings": True}


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


def test_parse_ragas_result_with_evaluation_result_shape():
    class DummyEvaluationResult:
        def __getitem__(self, key):
            values = {
                "faithfulness": [0.82],
                "answer_relevancy": [float("nan")],
            }
            return values[key]

    parsed = ragas_eval._parse_ragas_result(
        DummyEvaluationResult(),
        ground_truth_provided=False,
    )

    assert parsed["faithfulness"] == 0.82
    assert parsed["answer_relevancy"] is None
    assert parsed["pass_thresholds"] is True
    assert parsed["metric_count"] == 1
