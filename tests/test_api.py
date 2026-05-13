"""
Unit tests for api.py
"""

import json

import pytest

import api


def test_pick_best_method_uses_highest_balance(tmp_path):
    summary = {
        "dense": {"mean_balance_score": 1.01},
        "bm25": {"mean_balance_score": 1.03},
        "hybrid": {"mean_balance_score": 0.98},
    }
    summary_path = tmp_path / "inv01_summary.json"
    summary_path.write_text(json.dumps(summary))

    method = api.pick_best_method(summary_path=summary_path, fallback_method="dense")
    assert method == "bm25"


def test_pick_best_method_falls_back_when_missing(tmp_path):
    missing_path = tmp_path / "missing.json"
    method = api.pick_best_method(summary_path=missing_path, fallback_method="dense")
    assert method == "dense"


def test_pick_best_method_rejects_invalid_fallback(tmp_path):
    with pytest.raises(ValueError):
        api.pick_best_method(
            summary_path=tmp_path / "inv01_summary.json",
            fallback_method="invalid",
        )


def test_run_claim_rejects_blank():
    with pytest.raises(ValueError):
        api.run_claim("   ")
