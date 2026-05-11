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

### src/ — COMPLETE (core pipeline)

| File | Status | Notes |
|---|---|---|
| src/config.py | DONE | All settings. RETRIEVAL_METHOD, TOP_K, PROMPT_VARIANT |
| src/indexer.py | DONE | Loads from local corpus.jsonl. Uses MPS on Apple Silicon |
| src/retriever.py | DONE | dense, bm25, hybrid, queryreform all working. `load_resources()` now uses module-level cache and loads once per process. |
| src/reformulator.py | DONE | Groq llama-3.1-8b rewrites claim with negation language |
| src/reranker.py | DONE | cross-encoder/ms-marco-MiniLM-L-6-v2. Module-level cache implemented via `_get_cross_encoder()` (loaded once per process). |
| src/verdict_prompt.py | DONE | 3 variants: neutral, biased, structured |
| src/pipeline.py | DONE | Full end-to-end pipeline working |
| src/evaluator.py | DONE | Support Recall, Contradiction Recall, Balance Score |
| src/ragas_eval.py | DONE | RAGAS faithfulness, answer_relevancy, context_precision, context_recall. Uses Groq llama-3.3-70b via OpenAI-compatible client. RAGAS 0.2.x API (instantiated metric objects, llm_factory). |
| src/logger.py | DONE | LangSmith tracing + SQLite storage at results/equipoise.db |
| src/utils.py | EMPTY | Shared utilities — low priority |

### tests/ — 44 TESTS TOTAL

```
44 tests collected
test_evaluator.py  — 15 tests
test_pipeline.py   — 12 tests
test_retriever.py  — 15 tests
test_reranker.py   — 2 tests
```

Run with: `PYTHONPATH=. pytest tests/ -v`

Current runtime note: `test_pipeline.py` uses live Groq calls and can fail with 429 rate-limit errors when API quota is exhausted. In the latest local run: 32 passed, 12 failed (all in `test_pipeline.py`) due to Groq 429.

### investigations/

| File | Status | Notes |
|---|---|---|
| investigations/inv01_retrieval.py | COMPLETE | Resume-safe script. `results/inv01_*.json` and `results/inv01_summary.json` exist with 300 claims per strategy (1200 runs logged). |
| investigations/inv02_topk.py | NOT CREATED | File does not exist yet. |
| investigations/inv03_prompts.py | NOT CREATED | File does not exist yet. |

INV-01 summary snapshot (`results/inv01_summary.json`):
- dense: support 0.9328, contradiction 0.9794, balance 1.0500
- bm25: support 0.8344, contradiction 0.8650, balance 1.0366
- hybrid: support 0.9522, contradiction 0.9572, balance 1.0053
- queryreform: support 0.8567, contradiction 0.9267, balance 1.0817 (highest balance)

### app.py / api.py — NOT STARTED

Both files currently exist but are empty.

### requirements.txt — MISSING CONTENT

`requirements.txt` exists but is empty (0 lines), so dependency installation is not reproducible from repo state alone.

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

1. **No chunking** — PubMed abstracts are 250 words maxqw. One abstract = one retrieval unit.
2. **No live PubMed API** — SciFact corpus only. Labels needed for evaluation.
3. **BeIR/scifact removed** — Use local AllenAI files. BeIR has no CONTRADICT labels (only score=1).
4. **MPS device** — Apple Silicon detected and used automatically in indexer and retriever.
5. **experiments/ renamed to investigations/** — matches V2 document.
6. **PYTHONPATH=.** — always needed when running src/ files directly.
7. **RAGAS uses Groq via OpenAI-compatible client** — `llm_factory(model, client=OpenAI(base_url="https://api.groq.com/openai/v1"))`. No OpenAI account needed.
8. **RAGAS import path** — use `from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall` (not `ragas.metrics` — deprecated in 0.2.x).
9. **INV-01 claim sample** — 150 SUPPORT + 150 CONTRADICT from claims_train.jsonl, seed=42. NONE claims excluded (no ground-truth abstract IDs for recall computation).

---

## Current Reliability Issue

**Problem: pipeline tests and live runs can fail on Groq API quota/rate limits (HTTP 429).**

`tests/test_pipeline.py` invokes real Groq completions for both verdict generation and query reformulation.
When quota is exhausted, retrieval and reranking still work, but verdict-generation test assertions fail upstream.

**Current state:** retriever and reranker caching are already in place (`src/retriever.py` + `src/reranker.py`).

**Recommended next fix:** make pipeline tests deterministic by mocking Groq client calls (or gate integration tests separately from unit tests).

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

### Step 1 — Build investigations/inv02_topk.py
Implement K-sensitivity runner (K=3,5,10) on best INV-01 method and export `results/inv02_k*.json`.

### Step 2 — Build investigations/inv03_prompts.py
Implement prompt-sensitivity runner (neutral/biased/structured) over 50 claims with RAGAS scoring and JSON exports.

### Step 3 — Implement app.py (and optional API wrapper in api.py)
Create Streamlit UI for claim entry, method selector, K selector, prompt variant selector, and verdict/evidence display.

### Step 4 — Stabilize tests and packaging
Add deterministic pipeline test strategy (mock/stub Groq) and populate `requirements.txt`.

### Step 5 — Refresh README.md
Publish measured results (INV-01 complete; INV-02/03 pending), reproducible setup, and architecture updates.

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

1. Use `results/inv01_summary.json` to choose the strategy for INV-02 (current highest Balance Score: queryreform).
2. Implement `investigations/inv02_topk.py`.
3. Make `test_pipeline.py` resilient to Groq quota limits by mocking live API calls.
