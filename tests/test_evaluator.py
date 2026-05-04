"""
Unit tests for src/evaluator.py
"""

import pytest
from src.evaluator import (
    compute_recall,
    evaluate_single,
    load_scifact_labels
)


# --- compute_recall ---

def test_recall_perfect():
    retrieved = ["123", "456", "789"]
    gold = {"123", "456", "789"}
    assert compute_recall(retrieved, gold) == 1.0


def test_recall_partial():
    retrieved = ["123", "456"]
    gold = {"123", "456", "789"}
    assert round(compute_recall(retrieved, gold), 4) == round(2/3, 4)


def test_recall_zero():
    retrieved = ["999", "888"]
    gold = {"123", "456"}
    assert compute_recall(retrieved, gold) == 0.0


def test_recall_empty_gold():
    retrieved = ["123"]
    gold = set()
    assert compute_recall(retrieved, gold) == 0.0


def test_recall_empty_retrieved():
    retrieved = []
    gold = {"123", "456"}
    assert compute_recall(retrieved, gold) == 0.0


# --- evaluate_single ---

def test_evaluate_single_support_claim_hit():
    labels = {
        "1": {
            "claim": "Aspirin reduces fever",
            "supporting_ids": {"pmid_123"},
            "contradicting_ids": set(),
            "claim_type": "SUPPORT"
        }
    }
    retrieved = [{"id": "pmid_123"}, {"id": "pmid_999"}]
    result = evaluate_single("1", retrieved, labels)
    assert result["recall"] == 1.0
    assert result["claim_type"] == "SUPPORT"
    assert result["claim_id"] == "1"


def test_evaluate_single_support_claim_miss():
    labels = {
        "1": {
            "claim": "Aspirin reduces fever",
            "supporting_ids": {"pmid_123"},
            "contradicting_ids": set(),
            "claim_type": "SUPPORT"
        }
    }
    retrieved = [{"id": "pmid_999"}]
    result = evaluate_single("1", retrieved, labels)
    assert result["recall"] == 0.0


def test_evaluate_single_contradict_claim_hit():
    labels = {
        "2": {
            "claim": "Aspirin has no effect",
            "supporting_ids": set(),
            "contradicting_ids": {"pmid_456"},
            "claim_type": "CONTRADICT"
        }
    }
    retrieved = [{"id": "pmid_456"}]
    result = evaluate_single("2", retrieved, labels)
    assert result["recall"] == 1.0
    assert result["claim_type"] == "CONTRADICT"


def test_evaluate_single_contradict_claim_miss():
    labels = {
        "2": {
            "claim": "Aspirin has no effect",
            "supporting_ids": set(),
            "contradicting_ids": {"pmid_456"},
            "claim_type": "CONTRADICT"
        }
    }
    retrieved = [{"id": "pmid_999"}]
    result = evaluate_single("2", retrieved, labels)
    assert result["recall"] == 0.0


def test_evaluate_single_missing_claim():
    labels = {}
    result = evaluate_single("nonexistent", [{"id": "123"}], labels)
    assert "error" in result


def test_evaluate_single_none_type():
    labels = {
        "3": {
            "claim": "Some claim",
            "supporting_ids": set(),
            "contradicting_ids": set(),
            "claim_type": "NONE"
        }
    }
    retrieved = [{"id": "pmid_123"}]
    result = evaluate_single("3", retrieved, labels)
    assert result["recall"] == 0.0


# --- load_scifact_labels ---

def test_load_returns_dict():
    labels = load_scifact_labels()
    assert isinstance(labels, dict)
    assert len(labels) > 0


def test_load_structure():
    labels = load_scifact_labels()
    entry = labels[next(iter(labels))]
    assert "claim" in entry
    assert "supporting_ids" in entry
    assert "contradicting_ids" in entry
    assert "claim_type" in entry
    assert isinstance(entry["supporting_ids"], set)
    assert isinstance(entry["contradicting_ids"], set)
    assert entry["claim_type"] in ["SUPPORT", "CONTRADICT", "NONE"]


def test_load_has_both_types():
    labels = load_scifact_labels()
    types = [v["claim_type"] for v in labels.values()]
    assert "SUPPORT" in types
    assert "CONTRADICT" in types


def test_load_claim_is_string():
    labels = load_scifact_labels()
    for claim_id, data in list(labels.items())[:10]:
        assert isinstance(data["claim"], str)
        assert len(data["claim"]) > 0
