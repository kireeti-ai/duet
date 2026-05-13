"""
investigations/inv02_topk.py

INV-02: Top-K Sensitivity

Runs the best retrieval method from INV-01 across:
    K=3, K=5, K=10

For each K, logs:
    - Support Recall
    - Contradiction Recall
    - Balance Score
    - RAGAS scores (faithfulness, answer_relevancy, context_precision, context_recall)
    - Latency (seconds)

Outputs:
    results/inv02_k3.json
    results/inv02_k5.json
    results/inv02_k10.json
    results/inv02_summary.json

Resume-safe: run_config_hash prevents mixing runs across changed settings.

Usage:
    PYTHONPATH=. python investigations/inv02_topk.py
    PYTHONPATH=. python investigations/inv02_topk.py --k 5
    PYTHONPATH=. python investigations/inv02_topk.py --method dense --max-claims 50
"""

import os
import json
import time
import random
import argparse
import hashlib
import re

from groq import Groq

import src.config as cfg
from src.retriever import retrieve
from src.reranker import rerank
from src.reformulator import reformulate_query
from src.verdict_prompt import build_verdict_prompt
from src.ragas_eval import evaluate_ragas
from src.logger import init_db, log_run, get_runs, export_to_json


K_VALUES = [3, 5, 10]
STRATEGIES = ["dense", "bm25", "hybrid", "queryreform"]

N_SUPPORT = 150
N_CONTRADICT = 150
RANDOM_SEED = 42

RESULTS_DIR = "results"
INV_LABEL = "inv02"
INV01_SUMMARY_PATH = os.path.join(RESULTS_DIR, "inv01_summary.json")

_groq_client = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        if not cfg.GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY not found in environment/.env")
        _groq_client = Groq(api_key=cfg.GROQ_API_KEY)
    return _groq_client


def _load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_ground_truth_ids(claim: dict) -> tuple[set[str], set[str]]:
    supporting = set()
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
    supporting, contradicting = _extract_ground_truth_ids(claim)
    if contradicting:
        return "CONTRADICT"
    if supporting:
        return "SUPPORT"
    return "NONE"


def build_sample(claims_path: str, max_claims: int | None = None) -> list[dict]:
    """
    Reuse INV-01 sampling policy for comparability:
    150 SUPPORT + 150 CONTRADICT from claims_train.
    """
    all_claims = _load_jsonl(claims_path)
    support_pool = [c for c in all_claims if _claim_label(c) == "SUPPORT"]
    contradict_pool = [c for c in all_claims if _claim_label(c) == "CONTRADICT"]

    rng = random.Random(RANDOM_SEED)
    selected_support = rng.sample(support_pool, min(N_SUPPORT, len(support_pool)))
    selected_contradict = rng.sample(contradict_pool, min(N_CONTRADICT, len(contradict_pool)))

    sample = selected_support + selected_contradict
    rng.shuffle(sample)

    if max_claims is not None:
        sample = sample[:max_claims]

    print(
        f"Claim sample built: {len(sample)} total "
        f"(max_claims={max_claims if max_claims is not None else 'all'})"
    )
    return sample


def _load_corpus_map(corpus_path: str) -> dict[str, dict]:
    """
    Returns:
        {doc_id: {"title": str, "text": str}}
    """
    corpus = {}
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row.get("doc_id"))
            abstract = row.get("abstract", "")
            if isinstance(abstract, list):
                abstract_text = " ".join(str(s) for s in abstract)
            else:
                abstract_text = str(abstract)
            corpus[doc_id] = {
                "title": row.get("title", ""),
                "text": abstract_text,
            }
    return corpus


def _build_ground_truth_text(claim: dict, corpus_map: dict[str, dict]) -> str | None:
    supporting_ids, contradicting_ids = _extract_ground_truth_ids(claim)

    def gather(doc_ids: set[str]) -> list[str]:
        snippets = []
        for doc_id in sorted(doc_ids):
            doc = corpus_map.get(doc_id)
            if not doc:
                continue
            snippets.append(
                f"PMID: {doc_id}\n"
                f"Title: {doc.get('title', '')}\n"
                f"Abstract: {doc.get('text', '')}"
            )
        return snippets

    support_snippets = gather(supporting_ids)
    contradict_snippets = gather(contradicting_ids)

    sections = []
    if support_snippets:
        sections.append("SUPPORTING EVIDENCE\n" + "\n\n".join(support_snippets))
    if contradict_snippets:
        sections.append("CONTRADICTING EVIDENCE\n" + "\n\n".join(contradict_snippets))

    if not sections:
        return None
    return "\n\n".join(sections)


def _compute_claim_scores(
    retrieved: list[dict],
    supporting_ids: set[str],
    contradicting_ids: set[str],
) -> dict:
    retrieved_ids = {str(r.get("id", "")) for r in retrieved}

    if supporting_ids:
        support_recall = len(retrieved_ids & supporting_ids) / len(supporting_ids)
    else:
        support_recall = None

    if contradicting_ids:
        contradiction_recall = len(retrieved_ids & contradicting_ids) / len(contradicting_ids)
    else:
        contradiction_recall = None

    if support_recall is not None and contradiction_recall is not None:
        balance_score = contradiction_recall / support_recall if support_recall > 0 else 0.0
    else:
        balance_score = None

    return {
        "support_recall": support_recall,
        "contradiction_recall": contradiction_recall,
        "balance_score": balance_score,
    }


def _is_rate_limit_error(error: Exception) -> bool:
    """
    Detect provider rate-limit responses in a library-agnostic way.
    """
    status_code = getattr(error, "status_code", None)
    if status_code == 429:
        return True
    text = str(error).lower()
    return "429" in text or "rate limit" in text or "rate_limit_exceeded" in text


def _extract_retry_hint(error: Exception) -> str | None:
    text = str(error)
    match = re.search(r"try again in ([^\\.]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _pick_best_method_from_inv01(
    summary_path: str = INV01_SUMMARY_PATH,
    fallback_method: str = "dense",
) -> str:
    if not os.path.exists(summary_path):
        return fallback_method

    with open(summary_path) as f:
        summary = json.load(f)

    valid = {
        method: vals
        for method, vals in summary.items()
        if vals.get("mean_balance_score") is not None
    }
    if not valid:
        return fallback_method

    return max(valid, key=lambda m: valid[m]["mean_balance_score"])


def _build_run_config(method: str, top_k: int, sample_size: int) -> dict:
    candidate_k = max(cfg.RETRIEVAL_CANDIDATE_K, top_k)
    return {
        "investigation": INV_LABEL,
        "method": method,
        "top_k": top_k,
        "retrieval_candidate_k": candidate_k,
        "prompt_variant": cfg.PROMPT_VARIANT,
        "claims_path": cfg.CLAIMS_TRAIN_PATH,
        "corpus_path": cfg.CORPUS_PATH,
        "n_support": N_SUPPORT,
        "n_contradict": N_CONTRADICT,
        "sample_size": sample_size,
        "random_seed": RANDOM_SEED,
        "hybrid_dense_weight": cfg.HYBRID_DENSE_WEIGHT,
        "hybrid_bm25_weight": cfg.HYBRID_BM25_WEIGHT,
        "verdict_model": cfg.GROQ_VERDICT_MODEL,
    }


def _config_hash(run_config: dict) -> str:
    payload = json.dumps(run_config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _run_inv02_pipeline(
    claim_text: str,
    method: str,
    top_k: int,
    candidate_k: int,
) -> dict:
    reformulated = None
    if method == "queryreform":
        reformulated = reformulate_query(claim_text)

    retrieved = retrieve(
        claim_text,
        method=method,
        reformulated_query=reformulated,
        top_k=candidate_k,
    )
    reranked = rerank(claim_text, retrieved, top_k=top_k)

    prompt = build_verdict_prompt(
        claim_text,
        reranked,
        variant=cfg.PROMPT_VARIANT,
    )

    client = _get_groq_client()
    response = client.chat.completions.create(
        model=cfg.GROQ_VERDICT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    verdict_text = response.choices[0].message.content.strip()

    return {
        "claim": claim_text,
        "method": method,
        "prompt_variant": cfg.PROMPT_VARIANT,
        "reformulated_query": reformulated,
        "retrieved": reranked,
        "verdict": verdict_text,
    }


def run_k_experiment(
    top_k: int,
    method: str,
    sample: list[dict],
    corpus_map: dict[str, dict],
) -> tuple[list[dict], str]:
    inv_label = f"{INV_LABEL}_k{top_k}"
    results = []
    total = len(sample)

    run_config = _build_run_config(method, top_k=top_k, sample_size=total)
    run_config_hash = _config_hash(run_config)
    candidate_k = run_config["retrieval_candidate_k"]

    done_claims = {
        r["claim"] for r in get_runs(
            investigation=inv_label,
            run_config_hash=run_config_hash,
            limit=10000,
        )
    }
    remaining = [c for c in sample if c.get("claim", "") not in done_claims]

    if done_claims:
        print(
            f"  Resuming ({run_config_hash}): "
            f"{len(done_claims)} already done, {len(remaining)} remaining"
        )
    else:
        print(f"  Run config hash: {run_config_hash}")

    for i, claim_rec in enumerate(remaining):
        claim_text = claim_rec.get("claim", "")
        global_idx = total - len(remaining) + i + 1
        print(f"  [k={top_k}] {global_idx}/{total}: {claim_text[:70]}")

        supporting_ids, contradicting_ids = _extract_ground_truth_ids(claim_rec)
        started = time.perf_counter()

        try:
            pipeline_output = _run_inv02_pipeline(
                claim_text=claim_text,
                method=method,
                top_k=top_k,
                candidate_k=candidate_k,
            )
            retrieved = pipeline_output.get("retrieved", [])
            verdict = pipeline_output.get("verdict", "")
            latency_seconds = time.perf_counter() - started

            eval_scores = _compute_claim_scores(
                retrieved=retrieved,
                supporting_ids=supporting_ids,
                contradicting_ids=contradicting_ids,
            )

            contexts = [r.get("text", "") for r in retrieved if r.get("text")]
            ground_truth_text = _build_ground_truth_text(claim_rec, corpus_map=corpus_map)

            try:
                ragas_scores = evaluate_ragas(
                    claim=claim_text,
                    verdict=verdict,
                    contexts=contexts,
                    ground_truth=ground_truth_text,
                )
            except Exception as ragas_error:
                ragas_scores = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "context_precision": None,
                    "context_recall": None,
                    "pass_thresholds": False,
                    "metric_count": 0,
                    "error": str(ragas_error),
                }

            ragas_scores["latency_seconds"] = latency_seconds
            ragas_scores["ground_truth_used"] = bool(ground_truth_text)

            pipeline_output["retrieval_candidate_k"] = candidate_k
            pipeline_output["final_top_k"] = top_k
            pipeline_output["run_config_hash"] = run_config_hash

            row_id = log_run(
                pipeline_output=pipeline_output,
                eval_scores=eval_scores,
                ragas_scores=ragas_scores,
                investigation=inv_label,
            )

            results.append(
                {
                    "claim": claim_text,
                    "claim_id": claim_rec.get("id"),
                    "method": method,
                    "top_k": top_k,
                    "run_config_hash": run_config_hash,
                    "supporting_ids": list(supporting_ids),
                    "contradicting_ids": list(contradicting_ids),
                    "retrieved_ids": [str(r.get("id")) for r in retrieved],
                    "eval_scores": eval_scores,
                    "ragas_scores": ragas_scores,
                    "db_row_id": row_id,
                }
            )

        except Exception as e:
            latency_seconds = time.perf_counter() - started
            if _is_rate_limit_error(e):
                retry_hint = _extract_retry_hint(e)
                hint = f" (retry after {retry_hint})" if retry_hint else ""
                print(
                    f"    RATE LIMIT on claim {global_idx}{hint}. "
                    "Stopping this K run early; rerun later to resume."
                )
                break
            print(f"    ERROR on claim {global_idx}: {e}")
            results.append(
                {
                    "claim": claim_text,
                    "claim_id": claim_rec.get("id"),
                    "method": method,
                    "top_k": top_k,
                    "run_config_hash": run_config_hash,
                    "error": str(e),
                    "latency_seconds": latency_seconds,
                    "eval_scores": {
                        "support_recall": None,
                        "contradiction_recall": None,
                        "balance_score": None,
                    },
                }
            )

    return results, run_config_hash


def _aggregate(top_k: int, run_config_hash: str) -> dict:
    inv_label = f"{INV_LABEL}_k{top_k}"
    rows = get_runs(
        investigation=inv_label,
        run_config_hash=run_config_hash,
        limit=10000,
    )

    def mean(vals):
        return (sum(vals) / len(vals)) if vals else None

    sr_vals = [row["support_recall"] for row in rows if row.get("support_recall") is not None]
    cr_vals = [row["contradiction_recall"] for row in rows if row.get("contradiction_recall") is not None]
    faith_vals = [row["faithfulness"] for row in rows if row.get("faithfulness") is not None]
    cp_vals = [row["context_precision"] for row in rows if row.get("context_precision") is not None]

    latency_vals = []
    ragas_error_count = 0
    for row in rows:
        ragas_json = row.get("ragas_json")
        if not ragas_json:
            continue
        try:
            payload = json.loads(ragas_json)
        except json.JSONDecodeError:
            ragas_error_count += 1
            continue
        if payload.get("error"):
            ragas_error_count += 1
        latency = payload.get("latency_seconds")
        if isinstance(latency, (int, float)):
            latency_vals.append(float(latency))

    mean_support = mean(sr_vals)
    mean_contradiction = mean(cr_vals)
    mean_balance = (
        mean_contradiction / mean_support
        if mean_support is not None and mean_support > 0
        else None
    )

    return {
        "n_claims": len(rows),
        "mean_support_recall": mean_support,
        "mean_contradiction_recall": mean_contradiction,
        "mean_balance_score": mean_balance,
        "mean_faithfulness": mean(faith_vals),
        "mean_context_precision": mean(cp_vals),
        "mean_latency_seconds": mean(latency_vals),
        "ragas_error_count": ragas_error_count,
        "run_config_hash": run_config_hash,
    }


def _print_summary(k_summaries: dict[int, dict]) -> None:
    header = (
        f"{'K':<4}  {'Supp Recall':>11}  {'Contra Recall':>13}  {'Balance':>9}  "
        f"{'Faithful':>9}  {'CtxPrec':>8}  {'Latency(s)':>10}  {'Claims':>6}  {'RAGAS_ERR':>9}"
    )
    print()
    print("=" * len(header))
    print("INV-02 RESULTS — Top-K Sensitivity")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def fmt(v, width=8):
        if v is None:
            return "N/A".rjust(width)
        return f"{v:.4f}".rjust(width)

    for k in sorted(k_summaries):
        s = k_summaries[k]
        print(
            f"{k:<4}  {fmt(s['mean_support_recall'], 11)}  {fmt(s['mean_contradiction_recall'], 13)}  "
            f"{fmt(s['mean_balance_score'], 9)}  {fmt(s['mean_faithfulness'], 9)}  "
            f"{fmt(s['mean_context_precision'], 8)}  {fmt(s['mean_latency_seconds'], 10)}  "
            f"{s['n_claims']:>6}  {s['ragas_error_count']:>9}"
        )
    print("=" * len(header))


def fmt_maybe(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "N/A"


def main(method: str, ks: list[int], max_claims: int | None) -> None:
    print("INV-02: Top-K Sensitivity")
    print(f"Method     : {method}")
    print(f"K values   : {ks}")
    print(f"Seed       : {RANDOM_SEED}")
    print(f"Prompt     : {cfg.PROMPT_VARIANT}")
    print()

    init_db()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    sample = build_sample(cfg.CLAIMS_TRAIN_PATH, max_claims=max_claims)
    corpus_map = _load_corpus_map(cfg.CORPUS_PATH)
    print()

    summaries = {}
    for top_k in ks:
        print(f"Running K={top_k}")
        print("-" * 36)
        _, run_config_hash = run_k_experiment(
            top_k=top_k,
            method=method,
            sample=sample,
            corpus_map=corpus_map,
        )

        out_path = os.path.join(RESULTS_DIR, f"inv02_k{top_k}.json")
        export_to_json(
            out_path=out_path,
            investigation=f"{INV_LABEL}_k{top_k}",
            run_config_hash=run_config_hash,
        )
        print(f"  Saved {out_path}")

        agg = _aggregate(top_k=top_k, run_config_hash=run_config_hash)
        summaries[top_k] = agg
        print(
            f"  Done: balance={fmt_maybe(agg['mean_balance_score'])}  "
            f"faithfulness={fmt_maybe(agg['mean_faithfulness'])}  "
            f"context_precision={fmt_maybe(agg['mean_context_precision'])}  "
            f"latency={fmt_maybe(agg['mean_latency_seconds'])}"
        )
        print()

    _print_summary(summaries)

    summary_path = os.path.join(RESULTS_DIR, "inv02_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "method": method,
                "k_values": ks,
                "max_claims": max_claims,
                "summaries": summaries,
            },
            f,
            indent=2,
        )
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INV-02: Top-K Sensitivity")
    parser.add_argument(
        "--method",
        choices=STRATEGIES,
        default=None,
        help="Retrieval method to use. Default picks best Balance Score from results/inv01_summary.json.",
    )
    parser.add_argument(
        "--k",
        type=int,
        choices=K_VALUES,
        default=None,
        help="Run a single K value instead of all K=3,5,10.",
    )
    parser.add_argument(
        "--max-claims",
        type=int,
        default=None,
        help="Optional cap for faster smoke runs (applied after deterministic sampling).",
    )
    args = parser.parse_args()

    selected_method = args.method or _pick_best_method_from_inv01(
        summary_path=INV01_SUMMARY_PATH,
        fallback_method=cfg.RETRIEVAL_METHOD,
    )
    selected_ks = [args.k] if args.k else K_VALUES
    main(method=selected_method, ks=selected_ks, max_claims=args.max_claims)

