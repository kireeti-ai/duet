"""
investigations/inv03_prompts.py

INV-03: Prompt Sensitivity

Tests how different prompt instructions (neutral, biased, structured) affect
the LLM's faithfulness and answer relevancy.

Uses best settings from previous investigations:
    - Method: hybrid (INV-01)
    - Top-K:  5      (INV-02)

To avoid Groq rate limits, we run this on a small sample of 10 claims
(5 SUPPORT, 5 CONTRADICT).

Metrics:
    - RAGAS Faithfulness
    - RAGAS Answer Relevancy
    - RAGAS Context Precision
    - RAGAS Context Recall

Results saved to:
    results/inv03_neutral.json
    results/inv03_biased.json
    results/inv03_structured.json
    results/inv03_summary.json

Usage:
    PYTHONPATH=. python investigations/inv03_prompts.py
"""

import os
import json
import random
import time
import hashlib

import src.config as cfg
from src.pipeline import run_pipeline
from src.ragas_eval import evaluate_ragas
from src.logger import init_db, log_run, export_to_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INV_LABEL        = "inv03"
PROMPTS          = ["neutral", "biased", "structured"]
N_CLAIMS_PER_LBL = 2      # 2 SUPPORT, 2 CONTRADICT = 4 total
RANDOM_SEED      = 42
RESULTS_DIR      = "results"

# Override config with best values
cfg.RETRIEVAL_METHOD = "hybrid"
cfg.TOP_K = 5

# ---------------------------------------------------------------------------
# Claim loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def _load_corpus(path: str) -> dict[str, dict]:
    corpus_list = _load_jsonl(path)
    return {str(doc["doc_id"]): doc for doc in corpus_list}

def _claim_label(claim: dict) -> str:
    supporting, contradicting = False, False
    for annotations in claim.get("evidence", {}).values():
        for ann in annotations:
            label = ann.get("label", "")
            if label == "SUPPORT":
                supporting = True
            elif label == "CONTRADICT":
                contradicting = True
    if contradicting: return "CONTRADICT"
    if supporting:    return "SUPPORT"
    return "NONE"

def build_sample(claims_path: str) -> list[dict]:
    all_claims = _load_jsonl(claims_path)
    support_pool = [c for c in all_claims if _claim_label(c) == "SUPPORT"]
    contradict_pool = [c for c in all_claims if _claim_label(c) == "CONTRADICT"]

    rng = random.Random(RANDOM_SEED)
    sample = (
        rng.sample(support_pool, N_CLAIMS_PER_LBL) +
        rng.sample(contradict_pool, N_CLAIMS_PER_LBL)
    )
    rng.shuffle(sample)
    print(f"Sample: {len(sample)} claims (5 SUPPORT, 5 CONTRADICT)")
    return sample

def get_ground_truth(claim_rec: dict, corpus: dict) -> str:
    """Construct a ground truth string from the abstracts annotated as evidence."""
    evidence_ids = claim_rec.get("evidence", {}).keys()
    gt_texts = []
    for doc_id in evidence_ids:
        doc = corpus.get(str(doc_id))
        if doc:
            gt_texts.append(f"PMID {doc_id}: {doc.get('abstract', [''])[0]}") # Just use first sentence or full abstract?
            # Actually, SciFact corpus has 'abstract' as a list of strings (sentences).
            gt_texts.append(" ".join(doc.get("abstract", [])))
    return "\n\n".join(gt_texts) if gt_texts else "No ground truth available."

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_prompt(prompt_variant: str, sample: list[dict], corpus: dict):
    print(f"\n{'='*50}\nRunning Prompt: {prompt_variant.upper()}\n{'='*50}")
    
    cfg.PROMPT_VARIANT = prompt_variant
    inv_label = f"{INV_LABEL}_{prompt_variant}"
    
    # Hash config
    payload = json.dumps({"inv": INV_LABEL, "prompt": prompt_variant, "k": cfg.TOP_K, "method": cfg.RETRIEVAL_METHOD}, sort_keys=True)
    cfg_hash = hashlib.sha256(payload.encode()).hexdigest()[:16]

    results = []
    for i, claim_rec in enumerate(sample):
        claim_text = claim_rec["claim"]
        print(f"\n  [{prompt_variant}] {i+1}/{len(sample)}: {claim_text[:60]}...")
        
        try:
            # 1. Pipeline (Retrieval + Verdict)
            pipeline_output = run_pipeline(claim_text, method=cfg.RETRIEVAL_METHOD)
            verdict = pipeline_output["verdict"]
            contexts = [r["text"] for r in pipeline_output["retrieved"] if r.get("text")]
            
            # 2. RAGAS Evaluation
            print("    Running RAGAS...")
            gt_text = get_ground_truth(claim_rec, corpus)
            ragas_scores = evaluate_ragas(claim_text, verdict, contexts, ground_truth=gt_text)
            
            pipeline_output["run_config_hash"] = cfg_hash
            log_run(pipeline_output, None, ragas_scores, inv_label)
            
            results.append({
                "claim": claim_text,
                "ragas_scores": ragas_scores
            })
            
            f  = ragas_scores.get("faithfulness")
            ar = ragas_scores.get("answer_relevancy")
            cp = ragas_scores.get("context_precision")
            cr = ragas_scores.get("context_recall")
            
            f_str  = f"{f:.4f}"  if f is not None else "N/A"
            ar_str = f"{ar:.4f}" if ar is not None else "N/A"
            cp_str = f"{cp:.4f}" if cp is not None else "N/A"
            cr_str = f"{cr:.4f}" if cr is not None else "N/A"
            
            print(f"    F: {f_str} | AR: {ar_str} | CP: {cp_str} | CR: {cr_str}")
            
        except Exception as e:
            print(f"    ERROR: {e}")
            
        time.sleep(2) # Avoid rate limits
        
    # Aggregate
    f_vals  = [r["ragas_scores"]["faithfulness"]      for r in results if r["ragas_scores"].get("faithfulness") is not None]
    ar_vals = [r["ragas_scores"]["answer_relevancy"]  for r in results if r["ragas_scores"].get("answer_relevancy") is not None]
    cp_vals = [r["ragas_scores"]["context_precision"] for r in results if r["ragas_scores"].get("context_precision") is not None]
    cr_vals = [r["ragas_scores"]["context_recall"]    for r in results if r["ragas_scores"].get("context_recall") is not None]
    
    return {
        "prompt": prompt_variant,
        "mean_faithfulness":      sum(f_vals)/len(f_vals)  if f_vals else None,
        "mean_answer_relevancy":  sum(ar_vals)/len(ar_vals) if ar_vals else None,
        "mean_context_precision": sum(cp_vals)/len(cp_vals) if cp_vals else None,
        "mean_context_recall":    sum(cr_vals)/len(cr_vals) if cr_vals else None,
        "valid_claims": len(results),
        "run_config_hash": cfg_hash
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample = build_sample(cfg.CLAIMS_TRAIN_PATH)
    corpus = _load_corpus("data/scifact/data/corpus.jsonl")
    
    summaries = {}
    for p in PROMPTS:
        summary = run_prompt(p, sample, corpus)
        summaries[p] = summary
        export_to_json(f"{RESULTS_DIR}/inv03_{p}.json", f"{INV_LABEL}_{p}", summary["run_config_hash"])
        
    print("\n" + "="*60)
    print("  INV-03 RESULTS — Prompt Sensitivity (10 Claims)")
    print("="*60)
    print(f"  {'Prompt':<12} {'Faithful':>10} {'AnsRel':>10} {'CtxPrec':>10} {'CtxRec':>10}")
    print("-" * 60)
    
    for p in PROMPTS:
        s = summaries[p]
        f  = f"{s['mean_faithfulness']:.3f}"      if s['mean_faithfulness']       else "N/A"
        ar = f"{s['mean_answer_relevancy']:.3f}"  if s['mean_answer_relevancy']   else "N/A"
        cp = f"{s['mean_context_precision']:.3f}" if s['mean_context_precision'] else "N/A"
        cr = f"{s['mean_context_recall']:.3f}"    if s['mean_context_recall']    else "N/A"
        print(f"  {p:<12} {f:>10} {ar:>10} {cp:>10} {cr:>10}")
    print("="*60)
    
    with open(f"{RESULTS_DIR}/inv03_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("Saved to results/inv03_summary.json")
