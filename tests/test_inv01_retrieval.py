import investigations.inv01_retrieval as inv01
import src.config as cfg


def test_config_hash_changes_with_retrieval_candidate_k(monkeypatch):
    monkeypatch.setattr(cfg, "RETRIEVAL_CANDIDATE_K", 30)
    hash_a = inv01._config_hash(inv01._build_run_config("dense"))

    monkeypatch.setattr(cfg, "RETRIEVAL_CANDIDATE_K", 40)
    hash_b = inv01._config_hash(inv01._build_run_config("dense"))

    assert hash_a != hash_b


def test_compute_claim_scores_keeps_zero_balance_defined():
    retrieved = [{"id": "doc_contra"}]
    scores = inv01._compute_claim_scores(
        retrieved=retrieved,
        supporting_ids={"doc_support"},
        contradicting_ids={"doc_contra"},
    )

    assert scores["support_recall"] == 0.0
    assert scores["contradiction_recall"] == 1.0
    assert scores["balance_score"] == 0.0


def test_run_strategy_uses_run_config_hash_for_resume(monkeypatch):
    sample = [
        {
            "id": "1",
            "claim": "Aspirin reduces fever",
            "evidence": {"123": [{"label": "SUPPORT", "sentences": [0]}]},
        }
    ]
    captured = {}

    def fake_get_runs(*, investigation=None, run_config_hash=None, limit=1000, **kwargs):
        captured["investigation"] = investigation
        captured["run_config_hash"] = run_config_hash
        captured["limit"] = limit
        return []

    def fake_run_inv01_retrieval_only(claim_text, method):
        return {
            "claim": claim_text,
            "method": method,
            "reformulated_query": None,
            "retrieved": [{"id": "123", "title": "t", "text": "x", "score": 0.9, "method": method}],
            "verdict": "ok",
        }

    monkeypatch.setattr(inv01, "get_runs", fake_get_runs)
    monkeypatch.setattr(inv01, "_run_inv01_retrieval_only", fake_run_inv01_retrieval_only)
    monkeypatch.setattr(inv01, "log_run", lambda **kwargs: 1)

    _, run_hash = inv01.run_strategy("dense", sample)

    assert captured["investigation"] == "inv01_dense"
    assert captured["run_config_hash"] == run_hash
    assert len(run_hash) == 16
