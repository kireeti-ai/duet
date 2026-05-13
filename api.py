"""
api.py
Application-facing helpers for running Equipoise with best-known defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import src.config as cfg
from src.pipeline import run_pipeline


INV01_SUMMARY_PATH = Path(cfg.RESULTS_DIR) / "inv01_summary.json"
VALID_METHODS = {"dense", "bm25", "hybrid", "queryreform"}


def pick_best_method(
    summary_path: Path = INV01_SUMMARY_PATH,
    fallback_method: str = cfg.RETRIEVAL_METHOD,
) -> str:
    """
    Pick the highest Balance Score method from INV-01 summary.
    Falls back to config method when summary is missing/invalid.
    """
    if fallback_method not in VALID_METHODS:
        raise ValueError(
            "fallback_method must be one of: dense | bm25 | hybrid | queryreform"
        )

    if not summary_path.exists():
        return fallback_method

    try:
        payload = json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return fallback_method

    if not isinstance(payload, dict):
        return fallback_method

    best_method = None
    best_balance = None
    for method, stats in payload.items():
        if method not in VALID_METHODS or not isinstance(stats, dict):
            continue
        balance = stats.get("mean_balance_score")
        if not isinstance(balance, (int, float)):
            continue
        if best_balance is None or balance > best_balance:
            best_method = method
            best_balance = float(balance)

    return best_method or fallback_method


def build_runtime_config() -> dict[str, Any]:
    """
    Build fixed runtime settings used by the app.
    """
    method = pick_best_method()
    return {
        "method": method,
        "top_k": cfg.TOP_K,
        "candidate_k": cfg.RETRIEVAL_CANDIDATE_K,
        "prompt_variant": cfg.PROMPT_VARIANT,
    }


def run_claim(claim: str) -> dict[str, Any]:
    """
    Run the pipeline for a single claim using best-known defaults.
    """
    cleaned = claim.strip()
    if not cleaned:
        raise ValueError("Claim cannot be empty.")

    runtime = build_runtime_config()
    output = run_pipeline(cleaned, method=runtime["method"])
    return {
        "runtime": runtime,
        "output": output,
    }
