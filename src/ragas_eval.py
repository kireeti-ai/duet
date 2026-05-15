"""
src/ragas_eval.py

RAGAS evaluation for Equipoise RAG pipeline.
Computes: faithfulness, answer_relevancy, context_precision, context_recall.

Evaluator LLM: Groq llama-3.3-70b-versatile via OpenAI-compatible client.
Supports RAGAS evaluate() across 0.2.x-0.4.x by selecting metric objects
compatible with the installed API.

Usage as module:
    from src.ragas_eval import evaluate_ragas
    scores = evaluate_ragas(claim, verdict, contexts, ground_truth=None)

Usage as CLI:
    PYTHONPATH=. python src/ragas_eval.py
"""

import os
import sys
import json
import time
import math
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAGAS imports — fail fast with a clear message if not installed
# ---------------------------------------------------------------------------

try:
    from datasets import Dataset
    # pyrefly: ignore [missing-import]
    from ragas import evaluate
    from ragas.llms import llm_factory, LangchainLLMWrapper
    from ragas.embeddings import HuggingFaceEmbeddings as ModernRagasHFEmbeddings
    from langchain_community.embeddings import (
        HuggingFaceEmbeddings as LangchainHFEmbeddings,
    )
    from openai import OpenAI

    # RAGAS 0.4.x evaluate() validates legacy Metric objects.
    # Prefer these when present; fallback to collections constructors.
    try:
        from ragas.metrics._faithfulness import faithfulness as legacy_faithfulness
        from ragas.metrics._answer_relevance import (
            answer_relevancy as legacy_answer_relevancy,
        )
        from ragas.metrics._context_precision import (
            context_precision as legacy_context_precision,
        )
        from ragas.metrics._context_recall import context_recall as legacy_context_recall

        USE_LEGACY_EVAL_METRICS = True
    except Exception:
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )

        USE_LEGACY_EVAL_METRICS = False
except ImportError as e:
    print(
        f"Missing dependency: {e}\n"
        "Install with: pip install ragas datasets openai"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAGAS_LLM_MODEL   = "gemini-2.0-flash"
RAGAS_EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RAGAS_TIMEOUT     = 180
RAGAS_MAX_RETRIES = 2

THRESHOLDS = {
    "faithfulness":      0.70,
    "answer_relevancy":  0.70,
    "context_precision": 0.65,
    "context_recall":    0.65,
}


# ---------------------------------------------------------------------------
# LLM and embeddings setup
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    """
    Build a RAGAS-compatible LLM.
    Uses ChatGoogleGenerativeAI if GOOGLE_API_KEY is available,
    else falls back to Groq's OpenAI-compatible endpoint.
    """
    # google_api_key = os.getenv("GOOGLE_API_KEY")
    # if google_api_key:
    #     from langchain_google_genai import ChatGoogleGenerativeAI
    #     gemini = ChatGoogleGenerativeAI(
    #         model=RAGAS_LLM_MODEL, 
    #         google_api_key=google_api_key,
    #         max_retries=3
    #     )
    #     return LangchainLLMWrapper(gemini)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY not found in .env")
    
    from langchain_openai import ChatOpenAI
    
    # OpenRouter supports 'n > 1' and has much higher limits for Llama 3.3 70B
    openrouter_llm = ChatOpenAI(
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        model_name="meta-llama/llama-3.3-70b-instruct",
        max_tokens=8192,
        temperature=0,
    )
    return LangchainLLMWrapper(openrouter_llm)


def _build_ragas_embeddings():
    """
    Reuse BAAI/bge-base-en-v1.5 for answer_relevancy embedding scoring.
    Same model as the retriever — already cached on disk, no second download.
    """
    device = _detect_device()

    if USE_LEGACY_EVAL_METRICS:
        # Legacy answer_relevancy expects embed_query/embed_documents methods.
        return LangchainHFEmbeddings(
            model_name=RAGAS_EMBED_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    return ModernRagasHFEmbeddings(
        model=RAGAS_EMBED_MODEL,
        device=device,
        normalize_embeddings=True,
    )


def _detect_device() -> str:
    """Return mps on Apple Silicon, else cpu."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_ragas(
    claim: str,
    verdict: str,
    contexts: list[str],
    ground_truth: Optional[str] = None,
) -> dict:
    """
    Run RAGAS evaluation on a single claim-verdict-contexts triple.

    Args:
        claim:        The user claim (RAGAS 'question').
        verdict:      The LLM-generated verdict (RAGAS 'answer').
        contexts:     List of retrieved abstract strings (RAGAS 'contexts').
        ground_truth: Optional reference answer string.
                      When provided: context_precision and context_recall are computed.
                      When None:     only faithfulness and answer_relevancy are computed.

    Returns:
        dict with keys: faithfulness, answer_relevancy,
                        context_precision (if ground_truth provided),
                        context_recall    (if ground_truth provided),
                        pass_thresholds   (bool),
                        metric_count      (int)
    """

    if not claim or not claim.strip():
        raise ValueError("claim cannot be empty")
    if not verdict or not verdict.strip():
        raise ValueError("verdict cannot be empty")
    if not contexts:
        raise ValueError("contexts list cannot be empty")
    if not isinstance(contexts, list) or not all(isinstance(c, str) for c in contexts):
        raise TypeError("contexts must be a list of strings")

    ragas_llm        = _build_ragas_llm()
    ragas_embeddings = _build_ragas_embeddings()

    data = {
        "question": [claim],
        "answer":   [verdict],
        "contexts": [contexts],
    }

    metrics, eval_kwargs = _build_metrics_for_evaluate(
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
        with_ground_truth=(ground_truth is not None),
    )
    if ground_truth is not None:
        data["ground_truth"] = [ground_truth]

    dataset = Dataset.from_dict(data)

    try:
        result = evaluate(
            dataset,
            metrics=metrics,
            raise_exceptions=False,
            **eval_kwargs
        )
    except Exception as e:
        raise RuntimeError(f"RAGAS evaluation failed: {e}") from e

    scores = _parse_ragas_result(result, ground_truth_provided=(ground_truth is not None))
    return scores


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _build_metrics_for_evaluate(ragas_llm, ragas_embeddings, with_ground_truth: bool):
    """
    Build metric objects compatible with ragas.evaluate() for the installed version.

    Returns:
        tuple[list, dict]: (metrics, evaluate_kwargs)
    """
    if USE_LEGACY_EVAL_METRICS:
        metrics = [legacy_faithfulness, legacy_answer_relevancy]
        if with_ground_truth:
            metrics += [legacy_context_precision, legacy_context_recall]
        return metrics, {"llm": ragas_llm, "embeddings": ragas_embeddings}

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]
    if with_ground_truth:
        metrics += [ContextPrecision(llm=ragas_llm), ContextRecall(llm=ragas_llm)]
    return metrics, {}


def _parse_ragas_result(result, ground_truth_provided: bool) -> dict:
    """
    Extract scalar scores from a RAGAS EvaluationResult.
    Supports both dict-like legacy results and EvaluationResult objects.
    """
    scores = {}

    metric_keys = ["faithfulness", "answer_relevancy"]
    if ground_truth_provided:
        metric_keys += ["context_precision", "context_recall"]

    for key in metric_keys:
        raw = _get_metric_raw_value(result, key)
        value = _to_scalar_score(raw)
        scores[key] = value

    passes = [
        value >= THRESHOLDS[key]
        for key, value in scores.items()
        if value is not None and key in THRESHOLDS
    ]

    scores["pass_thresholds"] = all(passes) if passes else False
    scores["metric_count"]    = len([v for v in scores.values() if isinstance(v, float)])

    return scores


def _get_metric_raw_value(result, key: str):
    """
    Fetch a metric value from both legacy dict-like and newer EvaluationResult shapes.
    """
    if isinstance(result, dict):
        return result.get(key, None)

    if hasattr(result, "get"):
        try:
            return result.get(key, None)
        except Exception:
            pass

    # EvaluationResult implements __getitem__(key) -> list[score]
    try:
        return result[key]
    except Exception:
        pass

    # Fallback to list-of-dicts payload if present.
    rows = getattr(result, "scores", None)
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0].get(key, None)

    return None


def _to_scalar_score(raw):
    """
    Normalize raw RAGAS metric outputs into float or None.
    """
    if raw is None:
        return None

    if isinstance(raw, (list, tuple)):
        if not raw:
            return None
        raw = raw[0]

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None

    if math.isnan(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Batch evaluation (called by investigations/)
# ---------------------------------------------------------------------------

def evaluate_ragas_batch(
    records: list[dict],
    ground_truth_key: Optional[str] = None,
) -> list[dict]:
    """
    Evaluate a list of pipeline records with RAGAS.

    Each record must have: claim, verdict, contexts (list of strings).
    Optionally: a ground_truth field (key name passed via ground_truth_key).

    Returns the same list of records, each extended with a 'ragas_scores' key.
    Rate-limited to 1 request/second to avoid Groq 429s on the free tier.
    """

    if not records:
        raise ValueError("records list is empty")

    results = []
    total   = len(records)

    for i, record in enumerate(records):
        claim    = record.get("claim", "")
        verdict  = record.get("verdict", "")
        contexts = record.get("contexts", [])
        gt       = record.get(ground_truth_key) if ground_truth_key else None

        print(f"RAGAS eval {i + 1}/{total}: {claim[:60]}")

        try:
            scores = evaluate_ragas(claim, verdict, contexts, ground_truth=gt)
            record["ragas_scores"] = scores
        except Exception as e:
            print(f"  RAGAS failed for record {i + 1}: {e}")
            record["ragas_scores"] = {
                "faithfulness":     None,
                "answer_relevancy": None,
                "pass_thresholds":  False,
                "metric_count":     0,
                "error":            str(e),
            }

        results.append(record)

        if i < total - 1:
            time.sleep(1.0)

    return results


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def print_ragas_report(scores: dict, claim: str = "") -> None:
    """Print a formatted RAGAS score report to stdout."""

    separator = "-" * 52
    print(separator)
    if claim:
        print(f"Claim: {claim[:80]}")
        print(separator)

    label_map = {
        "faithfulness":      "Faithfulness       (>0.70)",
        "answer_relevancy":  "Answer Relevancy   (>0.70)",
        "context_precision": "Context Precision  (>0.65)",
        "context_recall":    "Context Recall     (>0.65)",
    }

    for key, label in label_map.items():
        if key not in scores:
            continue
        value = scores[key]
        if value is None:
            print(f"  {label}: N/A  (no ground_truth provided)")
        else:
            threshold = THRESHOLDS.get(key, 0)
            flag = "PASS" if value >= threshold else "FAIL"
            print(f"  {label}: {value:.4f}  [{flag}]")

    print(separator)
    overall = "PASS" if scores.get("pass_thresholds") else "FAIL"
    print(f"  Overall:  {overall}  ({scores.get('metric_count', 0)} metrics scored)")
    print(separator)


# ---------------------------------------------------------------------------
# CLI runner
# ---------------------------------------------------------------------------

def _run_cli() -> None:
    """
    Interactive CLI for spot-checking a single claim through RAGAS.
    Pulls a live verdict from the pipeline, then evaluates it.
    """

    print("Equipoise RAGAS Evaluator — single claim mode")
    print("Press Ctrl+C to exit")
    print()

    try:
        from src.pipeline import run_pipeline
        from src.config import RETRIEVAL_METHOD, TOP_K, PROMPT_VARIANT
    except ImportError as e:
        print(f"Could not import pipeline: {e}")
        print("Run with: PYTHONPATH=. python src/ragas_eval.py")
        sys.exit(1)

    claim = input("Enter a biomedical claim: ").strip()
    if not claim:
        print("No claim entered. Exiting.")
        sys.exit(0)

    print()
    print(f"Running pipeline  method={RETRIEVAL_METHOD}  k={TOP_K}  prompt={PROMPT_VARIANT}")
    print()

    pipeline_output = run_pipeline(claim)
    verdict   = pipeline_output.get("verdict", "")
    retrieved = pipeline_output.get("retrieved", [])
    contexts  = [r["text"] for r in retrieved if r.get("text")]

    if not verdict:
        print("Pipeline returned an empty verdict. Cannot evaluate.")
        sys.exit(1)

    if not contexts:
        print("Pipeline returned no retrieved abstracts. Cannot evaluate.")
        sys.exit(1)

    print("Verdict:")
    print(verdict)
    print()

    gt_input = input(
        "Optional — paste a reference answer for context_precision/recall scoring\n"
        "(press Enter to skip): "
    ).strip()
    ground_truth = gt_input if gt_input else None

    print()
    print("Running RAGAS evaluation...")
    scores = evaluate_ragas(claim, verdict, contexts, ground_truth=ground_truth)

    print()
    print_ragas_report(scores, claim=claim)

    out_path = "results/ragas_cli_last.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {"claim": claim, "verdict": verdict, "ragas_scores": scores},
            f, indent=2,
        )
    print(f"Scores saved to {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _run_cli()
