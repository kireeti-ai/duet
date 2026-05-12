import json

from src.logger import export_to_json, get_runs, init_db, log_run


def _sample_pipeline_output(run_config_hash: str) -> dict:
    return {
        "claim": "Vitamin D improves depression",
        "method": "dense",
        "prompt_variant": "structured",
        "retrieval_candidate_k": 30,
        "final_top_k": 5,
        "run_config_hash": run_config_hash,
        "reformulated_query": None,
        "verdict": "INV-01 retrieval-only run (verdict generation skipped).",
        "retrieved": [
            {"id": "1", "title": "Doc 1", "text": "Text 1", "score": 0.9, "method": "dense", "rerank_score": 0.8}
        ],
    }


def test_log_run_persists_new_metadata_fields(tmp_path):
    db_path = str(tmp_path / "equipoise.db")
    init_db(db_path=db_path)

    log_run(
        pipeline_output=_sample_pipeline_output("cfg_hash_a"),
        eval_scores={"support_recall": 1.0, "contradiction_recall": None, "balance_score": None},
        ragas_scores=None,
        investigation="inv01_dense",
        db_path=db_path,
    )

    rows = get_runs(
        investigation="inv01_dense",
        run_config_hash="cfg_hash_a",
        db_path=db_path,
        limit=10,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["retrieval_candidate_k"] == 30
    assert row["final_top_k"] == 5
    assert row["top_k"] == 5
    assert row["run_config_hash"] == "cfg_hash_a"


def test_get_runs_filters_by_run_config_hash(tmp_path):
    db_path = str(tmp_path / "equipoise.db")
    init_db(db_path=db_path)

    log_run(
        pipeline_output=_sample_pipeline_output("cfg_hash_a"),
        eval_scores=None,
        ragas_scores=None,
        investigation="inv01_dense",
        db_path=db_path,
    )
    log_run(
        pipeline_output=_sample_pipeline_output("cfg_hash_b"),
        eval_scores=None,
        ragas_scores=None,
        investigation="inv01_dense",
        db_path=db_path,
    )

    rows_a = get_runs(
        investigation="inv01_dense",
        run_config_hash="cfg_hash_a",
        db_path=db_path,
        limit=10,
    )
    rows_b = get_runs(
        investigation="inv01_dense",
        run_config_hash="cfg_hash_b",
        db_path=db_path,
        limit=10,
    )

    assert len(rows_a) == 1
    assert len(rows_b) == 1
    assert rows_a[0]["run_config_hash"] == "cfg_hash_a"
    assert rows_b[0]["run_config_hash"] == "cfg_hash_b"


def test_export_to_json_filters_by_run_config_hash(tmp_path):
    db_path = str(tmp_path / "equipoise.db")
    out_path = str(tmp_path / "inv01_dense.json")
    init_db(db_path=db_path)

    log_run(
        pipeline_output=_sample_pipeline_output("cfg_hash_a"),
        eval_scores={"support_recall": 1.0, "contradiction_recall": None, "balance_score": None},
        ragas_scores=None,
        investigation="inv01_dense",
        db_path=db_path,
    )
    log_run(
        pipeline_output=_sample_pipeline_output("cfg_hash_b"),
        eval_scores={"support_recall": None, "contradiction_recall": 1.0, "balance_score": None},
        ragas_scores=None,
        investigation="inv01_dense",
        db_path=db_path,
    )

    export_to_json(
        out_path=out_path,
        investigation="inv01_dense",
        run_config_hash="cfg_hash_a",
        db_path=db_path,
    )

    rows = json.loads((tmp_path / "inv01_dense.json").read_text())
    assert len(rows) == 1
    assert rows[0]["run_config_hash"] == "cfg_hash_a"
