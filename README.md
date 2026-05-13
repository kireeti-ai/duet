# Equipoise

**Biomedical Scientific Claim Verification using Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is Equipoise?

Equipoise is a Retrieval-Augmented Generation system that verifies biomedical scientific claims by retrieving and synthesising evidence from the SciFact corpus. Unlike standard RAG systems that retrieve only documents similar to a query — inherently biasing toward confirming evidence — Equipoise systematically studies how different retrieval strategies affect the completeness of both supporting and contradicting evidence surfaced for a given claim.

The name comes from **clinical equipoise** — the genuine uncertainty between competing treatments when evidence is balanced. That is exactly what this system measures and surfaces.

---

## The Research Problem

Standard dense retrieval has a hidden structural bias. When you embed the query *"Does Vitamin D improve depression symptoms?"* and search by vector similarity, papers that find *no effect* score lower than papers that find *positive effects* — because they use different language. This is called **Semantic Collapse** (TREC BioGen 2025).

Equipoise studies four retrieval strategies to measure how severely each is biased — using three original evaluation metrics: **Support Recall**, **Contradiction Recall**, and **Balance Score**.

---

## Research Question

> *How do retrieval strategies — dense, BM25, hybrid, and query-reformulation — affect Support Recall, Contradiction Recall, and Balance Score in biomedical claim verification, and what does this reveal about the structural bias of each retrieval method?*

---

## Key Results

| Retrieval Strategy | Support Recall | Contradiction Recall | Balance Score | Faithfulness | Latency (s) |
|---|---|---|---|---|---|
| Dense (BGE-base) | TBD | TBD | TBD | TBD | TBD |
| BM25 | TBD | TBD | TBD | TBD | TBD |
| Hybrid (dense + BM25) | TBD | TBD | TBD | TBD | TBD |
| Query-Reformulation | TBD | TBD | TBD | TBD | TBD |

*Results populated after running investigations. See `results/` for full JSON outputs.*

---

## Architecture

```
User inputs biomedical claim
          |
    Query Reformulator            (Groq llama-3.1-8b — queryreform mode only)
          |
    Retriever
    (dense | bm25 | hybrid | queryreform)
          |
    Cross-Encoder Re-ranker       (ms-marco-MiniLM-L-6-v2)
          |
    Verdict Prompt                (neutral | biased | structured)
          |
    LLM Generator                 (Groq llama-3.3-70b-versatile)
          |
    Structured Verdict + Citations
          |
    Evaluation Pipeline           (Custom metrics + RAGAS)
          |
    LangSmith Tracing + SQLite Storage
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/equipoise-rag.git
cd equipoise-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (see Environment Setup below)

# 5. Download SciFact dataset
curl -L "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz" \
     -o data/scifact/data.tar.gz
tar -xzf data/scifact/data.tar.gz -C data/scifact/

# 6. Build the index (run once — saved to chroma_db/)
PYTHONPATH=. python src/indexer.py

# 7. Run the app
streamlit run app.py --server.fileWatcherType none
```

The Streamlit app runs with fixed best-known defaults:
- Retrieval method auto-selected from `results/inv01_summary.json` by highest Balance Score
- `RETRIEVAL_CANDIDATE_K`, `TOP_K`, and `PROMPT_VARIANT` taken from `src/config.py`
- If Streamlit shows watcher-related import noise, keep `--server.fileWatcherType none`

---

## Environment Setup

Copy `.env.example` to `.env` and fill in the keys:

```bash
# Groq API — console.groq.com (free, no credit card)
GROQ_API_KEY=your_groq_api_key_here

# LangSmith — smith.langchain.com (free tier)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=equipoise-rag
```

---

## Running Investigations

```bash
# Run all 4 retrieval strategies on 300 SciFact claims (INV-01)
PYTHONPATH=. python investigations/inv01_retrieval.py

# Resume a specific strategy after interruption
PYTHONPATH=. python investigations/inv01_retrieval.py --method bm25

# Run top-K sensitivity on best strategy from INV-01 (INV-02)
PYTHONPATH=. python investigations/inv02_topk.py

# Run prompt variant comparison with RAGAS scoring (INV-03)
PYTHONPATH=. python investigations/inv03_prompts.py

# Inspect logged results
PYTHONPATH=. python src/logger.py

# Spot-check RAGAS scores on a single claim
PYTHONPATH=. python src/ragas_eval.py
```

All results are saved to `results/` as JSON and logged to SQLite at `results/equipoise.db`.

---

## Deploying to Hugging Face Spaces

Use a Streamlit Space and point it at `app.py`.

### Files needed
- `app.py`
- `api.py`
- `src/`
- `.streamlit/config.toml`
- `runtime.txt`
- `requirements.txt`
- `data/` and `chroma_db/` if you want the built app to run immediately

### Space settings
- **SDK:** Streamlit
- **Main file:** `app.py`
- **Python version:** 3.11

### Secrets
Set these in Space secrets:
- `GROQ_API_KEY`
- `LANGCHAIN_API_KEY` if tracing is enabled

### Notes
- The app uses the fixed best method from `results/inv01_summary.json`.
- The SciFact corpus is a static AllenAI snapshot, not a live data source.
- If you want the Space to build from scratch, include the dataset and existing indexes or add a separate indexing build step.
- The included Streamlit config keeps the UI dark and disables file-watcher noise.

---

## Project Structure

```
equipoise-rag/
|
+-- data/scifact/data/           SciFact corpus and claims
|   +-- corpus.jsonl             5183 abstracts
|   +-- claims_train.jsonl       Training claims with SUPPORT/CONTRADICT/NONE labels
|   +-- claims_dev.jsonl         Dev claims
|
+-- chroma_db/                   ChromaDB vector store + BM25 index (auto-generated)
|
+-- src/
|   +-- config.py                Central config (RETRIEVAL_METHOD, TOP_K, PROMPT_VARIANT)
|   +-- indexer.py               Abstract loading, embedding, ChromaDB + BM25 indexing
|   +-- retriever.py             Dense, BM25, hybrid, query-reform retrieval methods
|   +-- reformulator.py          Query reformulation using Groq llama-3.1-8b
|   +-- reranker.py              Cross-encoder re-ranking
|   +-- pipeline.py              Full end-to-end RAG pipeline
|   +-- verdict_prompt.py        Verdict prompt templates (neutral / biased / structured)
|   +-- evaluator.py             Support Recall, Contradiction Recall, Balance Score
|   +-- ragas_eval.py            RAGAS metric computation (faithfulness, answer relevancy)
|   +-- logger.py                LangSmith tracing + SQLite result storage
|   +-- utils.py                 Shared utilities
|
+-- investigations/
|   +-- inv01_retrieval.py       Dense vs BM25 vs Hybrid vs Query-reformulation (300 claims)
|   +-- inv02_topk.py            Top-K sensitivity: K=3 vs K=5 vs K=10
|   +-- inv03_prompts.py         Prompt variant comparison with RAGAS scoring (50 claims)
|
+-- results/                     JSON outputs + SQLite database
+-- tests/                       68 unit tests (all passing)
+-- app.py                       Streamlit interface
+-- requirements.txt             All dependencies
+-- .env.example                 Environment variable template
+-- CLAUDE.md                    Project context for AI-assisted development
+-- README.md                    This file
```

---

## Dataset

**AllenAI SciFact** — downloaded directly from S3 release (not HuggingFace, which has loading script issues).

| Split | Claims | Labels |
|---|---|---|
| claims_train.jsonl | 809 | SUPPORT: 456, CONTRADICT: 237, NONE: 116 |
| claims_dev.jsonl | 300 | held-out evaluation |
| corpus.jsonl | 5183 abstracts | expert-annotated |

INV-01 uses a stratified sample of 150 SUPPORT + 150 CONTRADICT claims from `claims_train.jsonl` (seed=42). NONE claims are excluded — they carry no ground-truth abstract IDs, making recall undefined.

---

## Evaluation Metrics

**Original metrics:**
- **Support Recall** — fraction of known supporting abstracts retrieved
- **Contradiction Recall** — fraction of known contradicting abstracts retrieved
- **Balance Score** — Contradiction Recall / Support Recall (1.0 = perfect, < 0.5 = biased toward support)

**RAGAS metrics (INV-03):**
- Faithfulness (> 0.70)
- Answer Relevancy (> 0.70)
- Context Precision (> 0.65)
- Context Recall (> 0.65)

---

## Tech Stack

| Component | Tool |
|---|---|
| Vector Database | ChromaDB |
| Sparse Retrieval | rank-bm25 |
| Embedding Model | BAAI/bge-base-en-v1.5 (109M params) |
| Primary LLM | Groq llama-3.3-70b-versatile |
| Reformulation LLM | Groq llama-3.1-8b-instant |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RAG Evaluation | RAGAS 0.2.x |
| LLM Monitoring | LangSmith |
| Result Storage | SQLite |
| Frontend | Streamlit |
| Hardware | Apple Silicon MPS (M-series Mac) |

---

## Running Tests

```bash
PYTHONPATH=. pytest tests/ -v
# 42 passed
```

---

## Key References

- Wadden et al. (2020). Fact or Fiction: Verifying Scientific Claims. EMNLP 2020.
- Sahoo et al. (2025). Negation is Not Semantic: Diagnosing Dense Retrieval Failure Modes. TREC BioGen 2025.
- Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv:2309.15217.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

*Equipoise — Because honest evidence shows both sides.*
