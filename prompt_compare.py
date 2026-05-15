"""
prompt_compare.py
Compares NEUTRAL vs STRUCTURED prompts using RAGAS scores.
"""

import time
import os
from src import config
from src.pipeline import run_pipeline
from src.ragas_eval import evaluate_ragas

CLAIM = "Omega-3 supplementation reduces symptoms of depression"
VARIANTS = ["neutral", "structured"]

results = {}

for variant in VARIANTS:
    print(f"\n{'='*55}")
    print(f"  PROMPT VARIANT: {variant.upper()}")
    print(f"{'='*55}")
    
    # Temporarily override config
    config.PROMPT_VARIANT = variant
    
    try:
        out = run_pipeline(CLAIM, method="hybrid")
        verdict = out["verdict"]
        contexts = [r["text"] for r in out["retrieved"] if r.get("text")]

        print(f"  Generated verdict using {variant} prompt.")
        print(f"  Running RAGAS (faithfulness + answer_relevancy)...")

        scores = evaluate_ragas(CLAIM, verdict, contexts, ground_truth=None)
        results[variant] = scores
        f  = scores.get("faithfulness")
        ar = scores.get("answer_relevancy")
        print(f"  Faithfulness:     {f:.4f}" if f is not None else "  Faithfulness:     N/A")
        print(f"  Answer Relevancy: {ar:.4f}" if ar is not None else "  Answer Relevancy: N/A")

    except Exception as e:
        print(f"  ERROR: {e}")
        results[variant] = {"faithfulness": None, "answer_relevancy": None}

    if variant != VARIANTS[-1]:
        print("  Waiting 5s to avoid rate limits...")
        time.sleep(5)

# ── Final comparison table ────────────────────────────────────────
print(f"\n\n{'='*55}")
print(f"  PROMPT VARIANT COMPARISON")
print(f"{'='*55}")
print(f"  {'Variant':<12} {'Faithfulness':>14} {'Ans.Relevancy':>14}")
print(f"  {'-'*50}")

for variant in VARIANTS:
    s = results[variant]
    f  = f"{s['faithfulness']:.4f}"  if s.get("faithfulness")  is not None else "N/A"
    ar = f"{s['answer_relevancy']:.4f}" if s.get("answer_relevancy") is not None else "N/A"
    print(f"  {variant:<12} {f:>14} {ar:>14}")

print(f"{'='*55}\n")
