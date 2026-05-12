import json

import investigations.inv02_topk as inv02


def test_pick_best_method_from_inv01_uses_highest_balance(tmp_path):
    summary_path = tmp_path / "inv01_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "dense": {"mean_balance_score": 1.04},
                "bm25": {"mean_balance_score": 1.01},
                "hybrid": {"mean_balance_score": 1.03},
            }
        )
    )

    method = inv02._pick_best_method_from_inv01(
        summary_path=str(summary_path),
        fallback_method="bm25",
    )

    assert method == "dense"


def test_pick_best_method_from_inv01_fallback_when_missing(tmp_path):
    missing_path = tmp_path / "missing_summary.json"
    method = inv02._pick_best_method_from_inv01(
        summary_path=str(missing_path),
        fallback_method="hybrid",
    )
    assert method == "hybrid"


def test_config_hash_changes_with_top_k():
    cfg_a = inv02._build_run_config(method="dense", top_k=3, sample_size=300)
    cfg_b = inv02._build_run_config(method="dense", top_k=10, sample_size=300)
    hash_a = inv02._config_hash(cfg_a)
    hash_b = inv02._config_hash(cfg_b)
    assert hash_a != hash_b


def test_build_ground_truth_text_contains_support_and_contradict_sections():
    claim = {
        "evidence": {
            "1": [{"label": "SUPPORT", "sentences": [0]}],
            "2": [{"label": "CONTRADICT", "sentences": [1]}],
        }
    }
    corpus_map = {
        "1": {"title": "Support title", "text": "Support abstract"},
        "2": {"title": "Contradict title", "text": "Contradict abstract"},
    }

    text = inv02._build_ground_truth_text(claim, corpus_map)

    assert text is not None
    assert "SUPPORTING EVIDENCE" in text
    assert "CONTRADICTING EVIDENCE" in text
    assert "PMID: 1" in text
    assert "PMID: 2" in text


def test_is_rate_limit_error_detects_429_message():
    err = Exception("Error code: 429 - {'error': {'code': 'rate_limit_exceeded'}}")
    assert inv02._is_rate_limit_error(err) is True


def test_extract_retry_hint_parses_wait_duration():
    err = Exception("Rate limit reached. Please try again in 21m42.912s. Need more tokens?")
    hint = inv02._extract_retry_hint(err)
    assert hint == "21m42"


def test_run_k_experiment_stops_on_rate_limit(monkeypatch):
    sample = [
        {"id": "1", "claim": "c1", "evidence": {}},
        {"id": "2", "claim": "c2", "evidence": {}},
    ]
    corpus_map = {}
    calls = {"pipeline": 0}

    def fake_get_runs(**kwargs):
        return []

    def fake_run_inv02_pipeline(**kwargs):
        calls["pipeline"] += 1
        raise Exception(
            "Error code: 429 - {'error': {'message': 'Rate limit reached. Please try again in 10m0s.'}}"
        )

    monkeypatch.setattr(inv02, "get_runs", fake_get_runs)
    monkeypatch.setattr(inv02, "_run_inv02_pipeline", fake_run_inv02_pipeline)

    results, run_hash = inv02.run_k_experiment(
        top_k=3,
        method="dense",
        sample=sample,
        corpus_map=corpus_map,
    )

    assert calls["pipeline"] == 1
    assert results == []
    assert len(run_hash) == 16
