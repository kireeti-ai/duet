"""
investigations/inv02_topk.py

INV-02: Top-K Sensitivity

Tests how the number of retrieved documents (K=3, 5, 10) affects retrieval
quality, using the best strategy from INV-01 (hybrid, balance=1.0053).

Metrics (retrieval-only, zero LLM tokens):
    Support Recall       = |retrieved & supporting| / |supporting|
    Contradiction Recall = |retrieved & contradicting| / |contradicting|
    Balance Score        = Contradiction Recall / Support Recall

Claim sample: same 150 SUPPORT + 150 CONTRADICT from claims_train.jsonl (seed=42).

Results saved to:
    results/inv02_k3.json
    results/inv02_k5.json
    results/inv02_k10.json
    results/inv02_summary.json

Resume-safe: already-completed (claim, k) pairs are skipped on re-run.

Usage:
    PYTHONPATH=. python investigations/inv02_topk.py
    PYTHONPATH=. python investigations/inv02_topk.py --k 5
"""

import os
import json
import random
import argparse
import time
import hashlib

import src.config as cfg
from src.retriever import retrieve
from src.reranker import rerank
from src.logger import init_db, log_run, get_runs, export_to_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INV_LABEL        = "inv02"
METHOD           = "hybrid"          # Best balance from INV-01
K_VALUES         = [3, 5, 10]
N_SUPPORT        = 150
N_CONTRADICT     = 150
RANDOM_SEED      = 42
CANDIDATE_K      = 30                # Candidates fetched before reranking
RESULTS_DIR      = "results"

# ---------------------------------------------------------------------------
# Claim loading (same logic as INV-01 for reproducibility)
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
    supporting    = set()
    contradicting = set()
    for doc_id, annotations in claim.get("evidence", {}).items():
        for ann in annotations:
            label = ann.get("label", "")
            if label == "SUPPORT":
                supporting.add(str(doc_id))
            elif label == "CONTRADICT":
                contradicting.add(str(doc_id))
    return supporting, contradicting


def _claim_label(claim: dict) -> str:
    supporting, contradicting = _extract_ground_truth(claim)
    if contradicting:
        return "CONTRADICT"
    if supporting:
        return "SUPPORT"
    return "NONE"


def build_sample(claims_path: str) -> list[dict]:
    """Same stratified sample as INV-01 (seed=42) for comparability."""
    all_claims      = _load_claims(claims_path)
    support_pool    = [c for c in all_claims if _claim_label(c) == "SUPPORT"]
    contradict_pool = [c for c in all_claims if _claim_label(c) == "CONTRADICT"]

    rng = random.Random(RANDOM_SEED)
    selected_support    = rng.sample(support_pool,    min(N_SUPPORT,    len(support_pool)))
    selected_contradict = rng.sample(contradict_pool, min(N_CONTRADICT, len(contradict_pool)))

    sample = selected_support + selected_contradict
    rng.shuffle(sample)

    print(
        f"Claim sample: {len(selected_support)} SUPPORT + "
        f"{len(selected_contradict)} CONTRADICT = {len(sample)} total"
    )
    return sample


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _compute_scores(retrieved: list[dict], supporting_ids: set, contradicting_ids: set) -> dict:
    retrieved_ids = {str(r.get("id", "")) for r in retrieved}

    support_recall = (
        len(retrieved_ids & supporting_ids) / len(supporting_ids)
        if supporting_ids else None
    )
    contradiction_recall = (
        len(retrieved_ids & contradicting_ids) / len(contradicting_ids)
        if contradicting_ids else None
    )
    balance_score = (
        contradiction_recall / support_recall
        if (support_recall is not None and contradiction_recall is not None and support_recall > 0)
        else None
    )

    return {
        "support_recall":       support_recall,
        "contradiction_recall": contradiction_recall,
        "balance_score":        balance_score,
    }


def _config_hash(k: int) -> str:
    payload = json.dumps({
        "investigation": INV_LABEL,
        "method":        METHOD,
        "top_k":         k,
        "candidate_k":   CANDIDATE_K,
        "n_support":     N_SUPPORT,
        "n_contradict":  N_CONTRADICT,
        "random_seed":   RANDOM_SEED,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-K runner
# ---------------------------------------------------------------------------

def run_k(k: int, sample: list[dict]) -> list[dict]:
    inv_label  = f"{INV_LABEL}_k{k}"
    cfg_hash   = _config_hash(k)
    results    = []
    total      = len(sample)

    # Resume: find already-done claims
    done_claims = {
        r["claim"] for r in get_runs(
            investigation=inv_label,
            run_config_hash=cfg_hash,
            limit=5000
        )
    }
    remaining = [c for c in sample if c.get("claim", "") not in done_claims]

    if done_claims:
        print(f"  Resuming: {len(done_claims)} done, {len(remaining)} remaining")
    else:
        print(f"  Config hash: {cfg_hash}")

    for i, claim_rec in enumerate(remaining):
        claim_text = claim_rec.get("claim", "")
        global_idx = total - len(remaining) + i + 1

        print(f"  [K={k}] {global_idx}/{total}: {claim_text[:70]}")

        supporting_ids, contradicting_ids = _extract_ground_truth(claim_rec)

        try:
            t0 = time.time()

            # Retrieval + reranking only (zero LLM tokens)
            retrieved_candidates = retrieve(
                claim_text,
                method=METHOD,
                top_k=CANDIDATE_K
            )
            reranked = rerank(claim_text, retrieved_candidates, top_k=k)

            latency = time.time() - t0

            eval_scores = _compute_scores(reranked, supporting_ids, contradicting_ids)

            pipeline_output = {
                "claim":                  claim_text,
                "method":                 METHOD,
                "prompt_variant":         cfg.PROMPT_VARIANT,
                "retrieval_candidate_k":  CANDIDATE_K,
                "final_top_k":            k,
                "reformulated_query":     None,
                "retrieved":              reranked,
                "verdict":                f"INV-02 retrieval-only (K={k}).",
                "run_config_hash":        cfg_hash,
                "latency_seconds":        latency,
            }

            row_id = log_run(
                pipeline_output=pipeline_output,
                eval_scores=eval_scores,
                ragas_scores=None,
                investigation=inv_label,
            )

            results.append({
                "claim":               claim_text,
                "claim_id":            claim_rec.get("id"),
                "k":                   k,
                "method":              METHOD,
                "supporting_ids":      list(supporting_ids),
                "contradicting_ids":   list(contradicting_ids),
                "retrieved_ids":       [str(r.get("id")) for r in reranked],
                "eval_scores":         eval_scores,
                "latency_seconds":     round(latency, 4),
                "db_row_id":           row_id,
                "run_config_hash":     cfg_hash,
            })

        except Exception as e:
            print(f"    ERROR on claim {global_idx}: {e}")
            results.append({
                "claim":      claim_text,
                "claim_id":   claim_rec.get("id"),
                "k":          k,
                "method":     METHOD,
                "error":      str(e),
                "eval_scores": {
                    "support_recall":       None,
                    "contradiction_recall": None,
                    "balance_score":        None,
                },
            })

    return results, cfg_hash


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _aggregate(k: int, cfg_hash: str) -> dict:
    rows = get_runs(
        investigation=f"{INV_LABEL}_k{k}",
        run_config_hash=cfg_hash,
        limit=5000
    )

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else None

    sr_vals  = [r["support_recall"]       for r in rows if r.get("support_recall")       is not None]
    cr_vals  = [r["contradiction_recall"] for r in rows if r.get("contradiction_recall") is not None]
    lat_vals = [r.get("latency_seconds", 0) for r in rows]

    mean_sr  = mean(sr_vals)
    mean_cr  = mean(cr_vals)
    mean_bal = round(mean_cr / mean_sr, 4) if (mean_sr and mean_cr and mean_sr > 0) else None

    return {
        "k":                          k,
        "method":                     METHOD,
        "n_claims":                   len(rows),
        "mean_support_recall":        mean_sr,
        "mean_contradiction_recall":  mean_cr,
        "mean_balance_score":         mean_bal,
        "mean_latency_seconds":       mean(lat_vals),
        "run_config_hash":            cfg_hash,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(summaries: dict) -> None:
    header = (
        f"  {'K':<6}  {'Supp Recall':>11}  {'Contra Recall':>13}"
        f"  {'Balance':>9}  {'Latency(s)':>10}  {'Claims':>6}"
    )
    sep = "=" * len(header)

    print()
    print(sep)
    print("  INV-02 RESULTS — Top-K Sensitivity  (method: hybrid)")
    print(sep)
    print(header)
    print("-" * len(header))

    def fmt(v):
        return f"{v:.4f}" if v is not None else "    N/A"

    for k, agg in sorted(summaries.items(), key=lambda x: int(x[0])):
        print(
            f"  K={k:<4}  "
            f"{fmt(agg['mean_support_recall']):>11}  "
            f"{fmt(agg['mean_contradiction_recall']):>13}  "
            f"{fmt(agg['mean_balance_score']):>9}  "
            f"{fmt(agg['mean_latency_seconds']):>10}  "
            f"{agg['n_claims']:>6}"
        )

    print(sep)

    valid = {k: s for k, s in summaries.items() if s["mean_balance_score"] is not None}
    if valid:
        best_k = max(valid, key=lambda k: valid[k]["mean_balance_score"])
        print(f"\n  Best Balance Score: K={best_k}  ({valid[best_k]['mean_balance_score']:.4f})")
        print(f"  Recommended for INV-03: set TOP_K = {best_k} in src/config.py")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(k_values: list[int]) -> None:
    print("INV-02: Top-K Sensitivity")
    print(f"Method     : {METHOD}  (best balance from INV-01)")
    print(f"K values   : {k_values}")
    print(f"Claims     : {N_SUPPORT} SUPPORT + {N_CONTRADICT} CONTRADICT = {N_SUPPORT + N_CONTRADICT}")
    print(f"LLM calls  : 0  (retrieval-only, no verdict, no RAGAS)")
    print()

    init_db()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sample = build_sample(cfg.CLAIMS_TRAIN_PATH)
    print()

    summaries = {}

    for k in k_values:
        print(f"Running K={k}")
        print("-" * 52)

        results, cfg_hash = run_k(k, sample)

        out_path = os.path.join(RESULTS_DIR, f"inv02_k{k}.json")
        export_to_json(
            out_path=out_path,
            investigation=f"{INV_LABEL}_k{k}",
            run_config_hash=cfg_hash
        )
        print(f"  Saved {out_path}")

        agg = _aggregate(k, cfg_hash)
        summaries[str(k)] = agg

        print(
            f"  Done: support_recall={_fmt(agg['mean_support_recall'])}  "
            f"contradiction_recall={_fmt(agg['mean_contradiction_recall'])}  "
            f"balance={_fmt(agg['mean_balance_score'])}"
        )
        print()

    _print_summary(summaries)

    summary_path = os.path.join(RESULTS_DIR, "inv02_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved to {summary_path}")


def _fmt(v):
    return f"{v:.4f}" if v is not None else "N/A"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INV-02: Top-K Sensitivity")
    parser.add_argument(
        "--k",
        type=int,
        choices=K_VALUES,
        default=None,
        help="Run a single K value instead of all three.",
    )
    args = parser.parse_args()

    k_values = [args.k] if args.k else K_VALUES
    main(k_values)
