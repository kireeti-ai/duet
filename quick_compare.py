"""
quick_compare.py
Runs dense / bm25 / hybrid retrieval on one claim and compares RAGAS scores.
Usage: PYTHONPATH=. python quick_compare.py
"""

import time
from src.pipeline import run_pipeline
from src.ragas_eval import evaluate_ragas, THRESHOLDS

CLAIM = "Omega-3 supplementation reduces symptoms of depression"
METHODS = ["dense", "bm25", "hybrid"]

results = {}

for method in METHODS:
    print(f"\n{'='*55}")
    print(f"  METHOD: {method.upper()}")
    print(f"{'='*55}")
    try:
        out = run_pipeline(CLAIM, method=method)
        verdict = out["verdict"]
        contexts = [r["text"] for r in out["retrieved"] if r.get("text")]

        print(f"  Retrieved {len(out['retrieved'])} docs")
        print(f"  Running RAGAS (faithfulness + answer_relevancy)...")

        scores = evaluate_ragas(CLAIM, verdict, contexts, ground_truth=None)
        results[method] = scores
        f  = scores.get("faithfulness")
        ar = scores.get("answer_relevancy")
        print(f"  Faithfulness:     {f:.4f}" if f is not None else "  Faithfulness:     N/A")
        print(f"  Answer Relevancy: {ar:.4f}" if ar is not None else "  Answer Relevancy: N/A")

    except Exception as e:
        print(f"  ERROR: {e}")
        results[method] = {"faithfulness": None, "answer_relevancy": None}

    if method != METHODS[-1]:
        print("  Waiting 3s before next method...")
        time.sleep(3)

# ── Final comparison table ────────────────────────────────────────
print(f"\n\n{'='*55}")
print(f"  RETRIEVAL METHOD COMPARISON  (claim: {CLAIM[:40]}...)")
print(f"{'='*55}")
print(f"  {'Method':<10} {'Faithfulness':>14} {'Ans.Relevancy':>14}  {'Winner'}")
print(f"  {'-'*50}")

best = {}
for metric in ["faithfulness", "answer_relevancy"]:
    vals = {m: results[m].get(metric) for m in METHODS if results[m].get(metric) is not None}
    best[metric] = max(vals, key=vals.get) if vals else None

for method in METHODS:
    s = results[method]
    f  = f"{s['faithfulness']:.4f}"  if s.get("faithfulness")  is not None else "N/A"
    ar = f"{s['answer_relevancy']:.4f}" if s.get("answer_relevancy") is not None else "N/A"
    tag = []
    if best.get("faithfulness") == method:    tag.append("F")
    if best.get("answer_relevancy") == method: tag.append("AR")
    winner = "★ " + "+".join(tag) if tag else ""
    print(f"  {method:<10} {f:>14} {ar:>14}  {winner}")

print(f"{'='*55}")
print(f"  Thresholds: Faithfulness>{THRESHOLDS['faithfulness']}  AnswerRelevancy>{THRESHOLDS['answer_relevancy']}")
print(f"{'='*55}\n")
