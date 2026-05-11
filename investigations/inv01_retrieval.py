"""
investigations/inv01_retrieval.py

INV-01: Retrieval Strategy Comparison

Runs all 4 retrieval strategies (dense, bm25, hybrid, queryreform) on 300
SciFact claims and measures Support Recall, Contradiction Recall, Balance Score.

Claim sample: 150 SUPPORT + 150 CONTRADICT from claims_train.jsonl.
NONE claims excluded — they carry no ground-truth labels for recall computation.

Results saved to:
    results/inv01_dense.json
    results/inv01_bm25.json
    results/inv01_hybrid.json
    results/inv01_queryreform.json

Resume-safe: already-logged claim+method combos are skipped on re-run.

Usage:
    PYTHONPATH=. python investigations/inv01_retrieval.py
    PYTHONPATH=. python investigations/inv01_retrieval.py --method bm25
"""

import os
import json
import random
import argparse
import time

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import src.config as cfg
from src.retriever import retrieve
from src.reranker import rerank
from src.reformulator import reformulate_query
from src.logger import init_db, log_run, get_runs, export_to_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGIES      = ["dense", "bm25", "hybrid", "queryreform"]
N_SUPPORT       = 150
N_CONTRADICT    = 150
RANDOM_SEED     = 42
INTER_CALL_SLEEP = 0.2   # seconds between claims

RESULTS_DIR     = "results"
INV_LABEL       = "inv01"

# ---------------------------------------------------------------------------
# Claim loading and stratified sampling
# ---------------------------------------------------------------------------

def _load_claims(path: str) -> list[dict]:
    claims = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    return claims


def _extract_ground_truth(claim: dict) -> tuple[set[str], set[str]]:
    """
    Parse the evidence field and return (supporting_ids, contradicting_ids).
    SciFact evidence format:
        { "doc_id": [ {"sentences": [...], "label": "SUPPORT|CONTRADICT"} ] }
    """
    supporting    = set()
    contradicting = set()
    evidence = claim.get("evidence", {})

    for doc_id, annotations in evidence.items():
        for ann in annotations:
            label = ann.get("label", "")
            if label == "SUPPORT":
                supporting.add(str(doc_id))
            elif label == "CONTRADICT":
                contradicting.add(str(doc_id))

    return supporting, contradicting


def _claim_label(claim: dict) -> str:
    """
    Derive a top-level label from the evidence field.
    A claim is SUPPORT if any evidence is SUPPORT, CONTRADICT if any is CONTRADICT, else NONE.
    """
    supporting, contradicting = _extract_ground_truth(claim)
    if contradicting:
        return "CONTRADICT"
    if supporting:
        return "SUPPORT"
    return "NONE"


def build_sample(claims_path: str) -> list[dict]:
    """
    Return a reproducible stratified sample: N_SUPPORT SUPPORT + N_CONTRADICT CONTRADICT.
    """
    all_claims = _load_claims(claims_path)

    support_pool    = [c for c in all_claims if _claim_label(c) == "SUPPORT"]
    contradict_pool = [c for c in all_claims if _claim_label(c) == "CONTRADICT"]

    rng = random.Random(RANDOM_SEED)
    selected_support    = rng.sample(support_pool,    min(N_SUPPORT, len(support_pool)))
    selected_contradict = rng.sample(contradict_pool, min(N_CONTRADICT, len(contradict_pool)))

    sample = selected_support + selected_contradict
    rng.shuffle(sample)

    print(
        f"Claim sample built: {len(selected_support)} SUPPORT + "
        f"{len(selected_contradict)} CONTRADICT = {len(sample)} total"
    )
    return sample


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_claim_scores(
    retrieved: list[dict],
    supporting_ids: set[str],
    contradicting_ids: set[str],
) -> dict:
    """
    Compute per-claim recall scores directly.
    Mirrors the definitions in evaluator.py:
        Support Recall       = |retrieved ∩ supporting|    / |supporting|
        Contradiction Recall = |retrieved ∩ contradicting| / |contradicting|
        Balance Score        = Contradiction Recall / Support Recall
    """
    retrieved_ids = {str(r.get("id", "")) for r in retrieved}

    if supporting_ids:
        support_recall = len(retrieved_ids & supporting_ids) / len(supporting_ids)
    else:
        support_recall = None   # claim has no supporting abstracts (pure CONTRADICT claim)

    if contradicting_ids:
        contradiction_recall = len(retrieved_ids & contradicting_ids) / len(contradicting_ids)
    else:
        contradiction_recall = None   # claim has no contradicting abstracts (pure SUPPORT claim)

    if support_recall and contradiction_recall:
        balance_score = contradiction_recall / support_recall if support_recall > 0 else 0.0
    else:
        balance_score = None

    return {
        "support_recall":       support_recall,
        "contradiction_recall": contradiction_recall,
        "balance_score":        balance_score,
    }


# ---------------------------------------------------------------------------
# Already-done check (resume support)
# ---------------------------------------------------------------------------

def _already_logged(claim_text: str, method: str) -> bool:
    """Return True if this claim+method combo exists in the DB."""
    runs = get_runs(investigation=f"{INV_LABEL}_{method}")
    done = {r["claim"] for r in runs}
    return claim_text in done


# ---------------------------------------------------------------------------
# Retrieval-only pipeline for INV-01
# ---------------------------------------------------------------------------

def _run_inv01_retrieval_only(claim_text: str, method: str) -> dict:
    """
    INV-01 evaluates retrieval behavior only.
    Skip verdict generation to avoid unnecessary latency and API bottlenecks.
    """
    reformulated = None
    if method == "queryreform":
        reformulated = reformulate_query(claim_text)

    retrieved = retrieve(
        claim_text,
        method=method,
        reformulated_query=reformulated,
        top_k=cfg.RETRIEVAL_CANDIDATE_K
    )
    reranked = rerank(claim_text, retrieved, top_k=cfg.TOP_K)

    return {
        "claim": claim_text,
        "method": method,
        "reformulated_query": reformulated,
        "retrieved": reranked,
        "verdict": "INV-01 retrieval-only run (verdict generation skipped).",
    }


# ---------------------------------------------------------------------------
# Per-strategy runner
# ---------------------------------------------------------------------------

def run_strategy(method: str, sample: list[dict]) -> list[dict]:
    """
    Run one retrieval strategy across all claims in the sample.
    Returns a list of result dicts (one per claim).
    """
    inv_label = f"{INV_LABEL}_{method}"
    results   = []
    total     = len(sample)

    # Override config — single-threaded, safe
    cfg.RETRIEVAL_METHOD = method

    # Count already done
    done_claims = {r["claim"] for r in get_runs(investigation=inv_label)}
    remaining   = [c for c in sample if c.get("claim", "") not in done_claims]

    if done_claims:
        print(f"  Resuming: {len(done_claims)} already done, {len(remaining)} remaining")

    for i, claim_rec in enumerate(remaining):
        claim_text = claim_rec.get("claim", "")
        global_idx = total - len(remaining) + i + 1

        print(f"  [{method}] {global_idx}/{total}: {claim_text[:70]}")

        supporting_ids, contradicting_ids = _extract_ground_truth(claim_rec)

        try:
            pipeline_output = _run_inv01_retrieval_only(claim_text, method)
            retrieved       = pipeline_output.get("retrieved", [])
            verdict         = pipeline_output.get("verdict", "")

            eval_scores = _compute_claim_scores(retrieved, supporting_ids, contradicting_ids)

            # Attach prompt_variant for logger (config value)
            pipeline_output["prompt_variant"] = cfg.PROMPT_VARIANT

            row_id = log_run(
                pipeline_output=pipeline_output,
                eval_scores=eval_scores,
                ragas_scores=None,
                investigation=inv_label,
            )

            result = {
                "claim":               claim_text,
                "claim_id":            claim_rec.get("id"),
                "method":              method,
                "supporting_ids":      list(supporting_ids),
                "contradicting_ids":   list(contradicting_ids),
                "retrieved_ids":       [str(r.get("id")) for r in retrieved],
                "verdict":             verdict,
                "eval_scores":         eval_scores,
                "db_row_id":           row_id,
            }
            results.append(result)

        except Exception as e:
            print(f"    ERROR on claim {global_idx}: {e}")
            results.append({
                "claim":     claim_text,
                "claim_id":  claim_rec.get("id"),
                "method":    method,
                "error":     str(e),
                "eval_scores": {
                    "support_recall":       None,
                    "contradiction_recall": None,
                    "balance_score":        None,
                },
            })

        if method == "queryreform" and i < len(remaining) - 1:
            time.sleep(INTER_CALL_SLEEP)

    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate(method: str) -> dict:
    """
    Compute mean Support Recall, Contradiction Recall, Balance Score
    from all rows currently logged for a method (resume-safe).
    """
    rows = get_runs(investigation=f"{INV_LABEL}_{method}", limit=5000)

    sr_vals = [row["support_recall"] for row in rows if row.get("support_recall") is not None]
    cr_vals = [row["contradiction_recall"] for row in rows if row.get("contradiction_recall") is not None]
    def mean(vals):
        return sum(vals) / len(vals) if vals else None

    mean_support = mean(sr_vals)
    mean_contradiction = mean(cr_vals)
    mean_balance = (
        mean_contradiction / mean_support
        if mean_support is not None and mean_support > 0
        else None
    )

    return {
        "n_claims":                   len(rows),
        "n_errors":                   (N_SUPPORT + N_CONTRADICT) - len(rows),
        "mean_support_recall":        mean_support,
        "mean_contradiction_recall":  mean_contradiction,
        "mean_balance_score":         mean_balance,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(strategy_summaries: dict[str, dict]) -> None:
    header = f"{'Method':<14}  {'Supp Recall':>11}  {'Contra Recall':>13}  {'Balance Score':>13}  {'Claims':>6}  {'Errors':>6}"
    print()
    print("=" * len(header))
    print("INV-01 RESULTS — Retrieval Strategy Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def fmt(v):
        return f"{v:.4f}" if v is not None else "     N/A"

    for method, agg in strategy_summaries.items():
        print(
            f"{method:<14}  "
            f"{fmt(agg['mean_support_recall']):>11}  "
            f"{fmt(agg['mean_contradiction_recall']):>13}  "
            f"{fmt(agg['mean_balance_score']):>13}  "
            f"{agg['n_claims']:>6}  "
            f"{agg['n_errors']:>6}"
        )

    print("=" * len(header))

    # Highlight winner
    valid = {m: s for m, s in strategy_summaries.items() if s["mean_balance_score"] is not None}
    if valid:
        best = max(valid, key=lambda m: valid[m]["mean_balance_score"])
        print(f"\nBest Balance Score: {best}  ({valid[best]['mean_balance_score']:.4f})")
        print(f"Recommended for INV-02: set RETRIEVAL_METHOD = \"{best}\" in src/config.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(strategies_to_run: list[str]) -> None:
    print("INV-01: Retrieval Strategy Comparison")
    print(f"Strategies : {strategies_to_run}")
    print(f"Claims     : {N_SUPPORT} SUPPORT + {N_CONTRADICT} CONTRADICT = {N_SUPPORT + N_CONTRADICT}")
    print(f"Seed       : {RANDOM_SEED}")
    print()

    init_db()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sample = build_sample(cfg.CLAIMS_TRAIN_PATH)
    print()

    strategy_summaries = {}

    for method in strategies_to_run:
        print(f"Running strategy: {method}")
        print("-" * 52)

        results = run_strategy(method, sample)

        # Export per-strategy JSON
        out_path = os.path.join(RESULTS_DIR, f"inv01_{method}.json")
        export_to_json(out_path=out_path, investigation=f"{INV_LABEL}_{method}")
        print(f"  Saved {out_path}")

        agg = _aggregate(method)
        strategy_summaries[method] = agg

        print(
            f"  Done: support_recall={fmt_maybe(agg['mean_support_recall'])}  "
            f"contradiction_recall={fmt_maybe(agg['mean_contradiction_recall'])}  "
            f"balance={fmt_maybe(agg['mean_balance_score'])}"
        )
        print()

    _print_summary(strategy_summaries)

    # Save combined summary
    summary_path = os.path.join(RESULTS_DIR, "inv01_summary.json")
    with open(summary_path, "w") as f:
        json.dump(strategy_summaries, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def fmt_maybe(v):
    return f"{v:.4f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INV-01: Retrieval Strategy Comparison")
    parser.add_argument(
        "--method",
        choices=STRATEGIES,
        default=None,
        help="Run a single strategy instead of all four (useful for resuming).",
    )
    args = parser.parse_args()

    strategies = [args.method] if args.method else STRATEGIES
    main(strategies)
