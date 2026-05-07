# CLAUDE.md — Equipoise Project Context
> Paste this file at the start of any new chat to continue exactly where we left off.

---

## What This Project Is

**Equipoise** is a RAG (Retrieval-Augmented Generation) system for biomedical scientific claim verification.

- User types a claim: "Does Vitamin D improve depression?"
- System retrieves relevant abstracts from SciFact corpus (5183 abstracts)
- LLM reads the abstracts and writes a balanced verdict showing BOTH supporting and contradicting evidence with citations
- Goal is NOT novelty — goal is deep understanding of how RAG works

**Project document version:** V2.0 (May 2026)
**GitHub repo:** equipoise-rag
**Local path:** /Users/kireeti/Desktop/Projects/equipoise-rag

---

## The Core Problem Being Studied

Dense retrieval is structurally blind to contradiction. When you search "Vitamin D improves depression", the embedding vector retrieves papers that agree. Papers saying "no effect found" use different language (negations, null-result framing) and score low in similarity — they get missed.

This is called **Semantic Collapse** (TREC BioGen 2025):
- Dense retrieval contradiction recall: 0.023
- BM25 contradiction recall: 0.750
- BM25 outperforms dense by factor of 32 for finding contradicting evidence

---

## Three Investigations

**INV-01 — Retrieval Strategy Comparison**
Compare dense vs BM25 vs hybrid vs query-reformulation on 300 SciFact claims.
Metrics: Support Recall, Contradiction Recall, Balance Score.
You learn: how retrieval bias works in a real pipeline.

**INV-02 — Top-K Sensitivity**
Best strategy from INV-01. Test K=3 vs K=5 vs K=10.
Metrics: Balance Score, Faithfulness, Context Precision, Latency.
You learn: more context is not always better.

**INV-03 — Prompt Sensitivity**
Same retrieval, three prompt variants (neutral / biased / structured) on 50 claims.
Metrics: RAGAS scores + 20 manual human judgments.
You learn: even if retrieval finds contradicting papers, a bad prompt makes the LLM ignore them.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Language |
| ChromaDB | Vector database for dense retrieval |
| rank-bm25 | BM25 sparse retrieval |
| BAAI/bge-base-en-v1.5 | Embedding model (109M params) |
| cross-encoder/ms-marco-MiniLM-L-6-v2 | Reranker (22M params) |
| Groq llama-3.3-70b-versatile | Verdict generation |
| Groq llama-3.1-8b-instant | Query reformulation |
| LangChain | Pipeline orchestration |
| RAGAS | RAG evaluation metrics |
| LangSmith | LLM monitoring and tracing |
| Weights and Biases | Experiment tracking |
| Streamlit | Frontend UI |

---

## Dataset

**AllenAI SciFact** — downloaded directly from GitHub release (NOT HuggingFace, has loading script issues)

Local files:
- `data/scifact/data/corpus.jsonl` — 5183 abstracts
- `data/scifact/data/claims_train.jsonl` — training claims with labels
- `data/scifact/data/claims_dev.jsonl` — dev claims with labels

Label types: SUPPORT (456 claims), CONTRADICT (237 claims), NONE (416 claims)

Key detail: abstract field in corpus.jsonl is a LIST of sentences — must be joined into one string before indexing.

---

## Environment Setup

```bash
cd /Users/kireeti/Desktop/Projects/equipoise-rag
source venv/bin/activate
```

Python version: 3.11.15
Run all scripts with: `PYTHONPATH=. python src/filename.py`

---

## Files Written and Status

### src/ — ALL COMPLETE AND TESTED

| File | Status | Notes |
|---|---|---|
| src/config.py | DONE | All settings. RETRIEVAL_METHOD, TOP_K, PROMPT_VARIANT |
| src/indexer.py | DONE | Loads from local corpus.jsonl. Uses MPS on Apple Silicon |
| src/retriever.py | DONE | dense, bm25, hybrid, queryreform all working. `load_resources()` now uses module-level cache and loads once per process. |
| src/reformulator.py | DONE | Groq llama-3.1-8b rewrites claim with negation language |
| src/reranker.py | DONE | cross-encoder/ms-marco-MiniLM-L-6-v2. **KNOWN ISSUE: `CrossEncoder` is re-initialized on every `rerank()` call (performance overhead).** |
| src/verdict_prompt.py | DONE | 3 variants: neutral, biased, structured |
| src/pipeline.py | DONE | Full end-to-end pipeline working |
| src/evaluator.py | DONE | Support Recall, Contradiction Recall, Balance Score |
| src/ragas_eval.py | DONE | RAGAS faithfulness, answer_relevancy, context_precision, context_recall. Uses Groq llama-3.3-70b via OpenAI-compatible client. RAGAS 0.2.x API (instantiated metric objects, llm_factory). |
| src/logger.py | DONE | LangSmith tracing + SQLite storage at results/equipoise.db |
| src/utils.py | EMPTY | Shared utilities — low priority |

### tests/ — ALL 42 PASSING

```
42 passed in 317.43s
test_evaluator.py  — 15 tests PASSED
test_pipeline.py   — 12 tests PASSED
test_retriever.py  — 15 tests PASSED
```

Run with: `PYTHONPATH=. pytest tests/ -v`

### investigations/

| File | Status | Notes |
|---|---|---|
| investigations/inv01_retrieval.py | WRITTEN, NOT COMPLETE | Script works, ran 8 dense claims successfully before Ctrl+C. Retriever cache fix is done; ready to resume from claim 9. |
| investigations/inv02_topk.py | NOT STARTED | |
| investigations/inv03_prompts.py | NOT STARTED | |

### app.py — NOT STARTED

Streamlit UI.

---

## Indexes Already Built

ChromaDB and BM25 indexes are built and saved to disk at `chroma_db/`.
- ChromaDB: 5183 abstracts indexed with cosine similarity
- BM25: rank-bm25 index saved as `chroma_db/bm25_index.pkl`
- Abstracts list saved as `chroma_db/abstracts.pkl`
- Device used: MPS (Apple Silicon M-series Mac)
- DO NOT re-run indexer unless you want to rebuild from scratch

---

## Important Technical Decisions Made

1. **No chunking** — PubMed abstracts are 250 words max. One abstract = one retrieval unit.
2. **No live PubMed API** — SciFact corpus only. Labels needed for evaluation.
3. **BeIR/scifact removed** — Use local AllenAI files. BeIR has no CONTRADICT labels (only score=1).
4. **MPS device** — Apple Silicon detected and used automatically in indexer and retriever.
5. **experiments/ renamed to investigations/** — matches V2 document.
6. **PYTHONPATH=.** — always needed when running src/ files directly.
7. **RAGAS uses Groq via OpenAI-compatible client** — `llm_factory(model, client=OpenAI(base_url="https://api.groq.com/openai/v1"))`. No OpenAI account needed.
8. **RAGAS import path** — use `from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall` (not `ragas.metrics` — deprecated in 0.2.x).
9. **INV-01 claim sample** — 150 SUPPORT + 150 CONTRADICT from claims_train.jsonl, seed=42. NONE claims excluded (no ground-truth abstract IDs for recall computation).

---

## Current Performance Issue (Before Long INV-01 Runs)

**Problem: `src/reranker.py` re-initializes `CrossEncoder` inside `rerank()` on every call.**

This repeats model loading for every claim and adds avoidable latency in investigation loops.

**Current state:** `src/retriever.py` caching issue is already fixed (`load_resources()` caches embedding model + Chroma collection + BM25 + abstracts once per process).

**Recommended next fix:** move reranker model load to module-level cache in `src/reranker.py`, then reuse it across calls.

---

## Key Config Settings (src/config.py)

```python
RETRIEVAL_METHOD = "dense"      # dense | bm25 | hybrid | queryreform
TOP_K = 5                       # INV-02: change to 3, 5, 10
PROMPT_VARIANT = "structured"   # INV-03: neutral | biased | structured
CORPUS_PATH = "data/scifact/data/corpus.jsonl"
CLAIMS_TRAIN_PATH = "data/scifact/data/claims_train.jsonl"
CLAIMS_DEV_PATH = "data/scifact/data/claims_dev.jsonl"
```

---

## Metrics Defined

**Original metrics (custom):**
- Support Recall = supporting abstracts retrieved / total supporting abstracts
- Contradiction Recall = contradicting abstracts retrieved / total contradicting abstracts
- Balance Score = Contradiction Recall / Support Recall (1.0 = perfect, below 0.5 = biased)

**RAGAS metrics (standard):**
- Faithfulness > 0.70
- Answer Relevance > 0.70
- Context Precision > 0.65
- Context Recall > 0.65

---

## RAGAS Notes (learned during implementation)

- RAGAS 0.2.x completely changed API from 0.1.x
- Metrics must be **instantiated objects**: `Faithfulness(llm=ragas_llm)` not the old singleton `faithfulness`
- LLM wrapper: `llm_factory(model, client=OpenAI(...))` — not `LangchainLLMWrapper`
- Embeddings: `from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings`
- Import path: `from ragas.metrics.collections import ...` (not `ragas.metrics`)
- pipeline.py returns key `retrieved` (list of dicts), not `contexts` — extract text with `[r["text"] for r in retrieved]`
- Each retrieved dict has keys: `id, title, text, score, method, rerank_score`

---

## logger.py Notes

- `init_db()` — call once at investigation startup. Creates `results/equipoise.db`.
- `log_run(pipeline_output, eval_scores, ragas_scores, investigation)` — call after every pipeline run.
- `get_runs(investigation=...)` — returns list of dicts, used for resume-check.
- `export_to_json(out_path, investigation=...)` — writes results/*.json files.
- `get_summary()` — aggregate counts/averages per investigation+method, used by CLI.
- LangSmith failure is caught as a warning — never blocks a run.
- SQLite stores: claim, method, k, prompt, verdict, support/contradiction/balance recall, all RAGAS scores, full retrieved list with abstract IDs + rerank scores as JSON, investigation label.

---

## What Is Left — In Order

### Step 0 — Cache reranker model (IMMEDIATE)
Add module-level caching in `src/reranker.py` so `CrossEncoder` is loaded once per process.

### Step 1 — Re-run investigations/inv01_retrieval.py
8 dense claims already in DB — will be skipped. Resume from claim 9.
Expected runtime after retriever + reranker caching: ~25-40 min for all 4 strategies.

### Step 2 — investigations/inv02_topk.py
Run K=3, K=5, K=10 on best strategy from INV-01.
Saves to results/inv02_k3.json, inv02_k5.json, inv02_k10.json.

### Step 3 — investigations/inv03_prompts.py
Run 3 prompt variants on 50 claims.
Saves to results/inv03_neutral.json, inv03_biased.json, inv03_structured.json.
This is where RAGAS runs (50 claims × 3 prompts = 150 RAGAS evaluations).

### Step 4 — app.py
Streamlit UI. Claim input, verdict display, retrieval method selector, top-K selector, prompt variant selector, monitoring sidebar.

### Step 5 — README.md
Results table, setup instructions, architecture description.

---

## Pipeline Flow (for reference)

```
User claim
    |
    v
reformulator.py (if queryreform mode)
rewrites claim with negation language
    |
    v
retriever.py
finds top-K abstracts using selected strategy
    |
    v
reranker.py
cross-encoder reorders by true relevance
    |
    v
verdict_prompt.py
builds structured prompt (neutral/biased/structured)
    |
    v
Groq llama-3.3-70b
generates verdict
    |
    v
evaluator.py
measures Support Recall, Contradiction Recall, Balance Score
    |
    v
ragas_eval.py
measures faithfulness, answer relevance, context precision
```

---

## Rules Established in This Project

- No emojis in code
- No bullet points in print statements — use plain text
- Always run with PYTHONPATH=.
- Never hardcode paths — use config.py constants
- All API keys in .env — never committed to GitHub
- venv/ chroma_db/ data/ __pycache__ .env all in .gitignore

---

## Next Immediate Action

1. Open `src/reranker.py` and add module-level cache for `CrossEncoder`.
2. Run `PYTHONPATH=. pytest tests/ -v` to confirm nothing broke.
3. Run `PYTHONPATH=. python investigations/inv01_retrieval.py` — should resume from claim 9 of dense, then run bm25/hybrid/queryreform.
