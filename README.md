# Equipoise

**Biomedical Scientific Claim Verification using Retrieval-Augmented Generation**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/yourusername/equipoise-rag)
[![W&B](https://img.shields.io/badge/Experiments-Weights%20%26%20Biases-orange)](https://wandb.ai)

---

## What is Equipoise?

Equipoise is a Retrieval-Augmented Generation system that verifies biomedical scientific claims by retrieving and synthesising evidence from PubMed literature. Unlike standard RAG systems that retrieve only documents similar to a query — inherently biasing toward confirming evidence — Equipoise systematically studies how different retrieval strategies affect the completeness of both supporting and contradicting evidence surfaced for a given claim.

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

*Results populated after running experiments. See `results/` for full JSON outputs.*

---

## Architecture

```
User inputs biomedical claim
          |
    Query Reformulator (Groq LLM)
          |
  --------+--------
  |               |
Support       Contradiction
  Query           Query
  |               |
Retrieval     Retrieval
(configurable: Dense / BM25 / Hybrid / Query-Reform)
  |               |
  +-------+-------+
          |
      Re-ranker
          |
    Verdict Prompt
          |
    LLM Generator (Groq llama-3.3-70b)
          |
  Structured Verdict + Citations
          |
  Evaluation Pipeline (RAGAS + Custom Metrics)
          |
  LangSmith Monitoring + W&B Tracking
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/equipoise-rag.git
cd equipoise-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys (see Environment Setup below)

# 5. Download SciFact dataset and build index
python src/indexer.py

# 6. Run the app
streamlit run app.py
```

---

## Environment Setup

Copy `.env.example` to `.env` and fill in all four keys:

```bash
# Groq API — console.groq.com (free, no credit card)
GROQ_API_KEY=your_groq_api_key_here

# LangSmith — smith.langchain.com (free tier)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=equipoise-rag

# Weights and Biases — wandb.ai (free tier)
WANDB_API_KEY=your_wandb_api_key_here

# HuggingFace — huggingface.co (free)
HUGGINGFACE_TOKEN=your_hf_token_here
```

---

## Running Experiments

```bash
# Build the abstract index (run once)
python src/indexer.py

# Run the main retrieval experiment (all 4 strategies)
python experiments/exp01_retrieval.py

# View results
cat results/exp01_dense.json
cat results/exp01_bm25.json
cat results/exp01_hybrid.json
cat results/exp01_queryreform.json

# Run embedding model comparison
python experiments/exp02_embedding.py
```

All experiment results are automatically logged to W&B and LangSmith.

---

## Running with Docker

```bash
# Build container
docker build -t equipoise-rag .

# Run
docker run -p 8501:8501 --env-file .env equipoise-rag

# Visit http://localhost:8501
```

---

## Project Structure

```
equipoise-rag/
|
+-- data/                    SciFact and PubMedQA dataset files
+-- chroma_db/               ChromaDB vector store (auto-generated)
|
+-- src/
|   +-- config.py            Central config for all parameters
|   +-- indexer.py           Abstract loading, embedding, ChromaDB + BM25 indexing
|   +-- reformulator.py      Query reformulation using Groq LLM
|   +-- retriever.py         Dense, BM25, hybrid, query-reform retrieval methods
|   +-- reranker.py          Cross-encoder re-ranking
|   +-- pipeline.py          Full RAG pipeline
|   +-- verdict_prompt.py    Structured verdict prompt templates
|   +-- evaluator.py         Support Recall, Contradiction Recall, Balance Score
|   +-- ragas_eval.py        RAGAS metric computation
|   +-- logger.py            LangSmith + SQLite logging
|   +-- utils.py             Shared utilities
|
+-- notebooks/
|   +-- 01_data_exploration.ipynb
|   +-- 02_retrieval_tests.ipynb
|   +-- 03_results_analysis.ipynb
|   +-- 04_visualisations.ipynb
|
+-- experiments/
|   +-- exp01_retrieval.py   Dense vs BM25 vs Hybrid vs Query-reformulation
|   +-- exp02_embedding.py   BGE-base vs PubMedBERT
|
+-- results/                 JSON files with all evaluation scores
+-- tests/                   Unit tests
+-- .github/workflows/       GitHub Actions CI
+-- app.py                   Streamlit interface
+-- api.py                   FastAPI backend
+-- Dockerfile               Container definition
+-- requirements.txt         All dependencies
+-- .env.example             Environment variable template
+-- README.md                This file
```

---

## Datasets

| Dataset | Size | Use | Access |
|---|---|---|---|
| SciFact | 1,409 claims, 5,183 abstracts | Primary evaluation — expert SUPPORT/CONTRADICT labels | `load_dataset('allenai/scifact')` |
| PubMedQA | 1K expert + 211K generated | Extended evaluation | `load_dataset('qiaojin/PubMedQA', 'pqa_labeled')` |
| PubMed Full Corpus | 35M abstracts | Extended knowledge base | NCBI Entrez API (free) |

---

## Evaluation Metrics

**Original metrics (novel contribution):**
- **Support Recall** — fraction of known supporting abstracts retrieved
- **Contradiction Recall** — fraction of known contradicting abstracts retrieved  
- **Balance Score** — ratio of contradiction to support recall (1.0 = perfect balance)

**Standard RAGAS metrics:**
- Faithfulness, Answer Relevance, Context Precision, Context Recall

---

## Tech Stack

| Component | Tool |
|---|---|
| Vector Database | ChromaDB |
| Sparse Retrieval | rank-bm25 |
| Primary LLM | llama-3.3-70b via Groq API |
| Reformulation LLM | llama-3.1-8b via Groq API |
| Local Fallback | Ollama (llama3.2:3b) |
| Embedding (baseline) | BAAI/bge-base-en-v1.5 |
| Embedding (domain) | microsoft/BiomedNLP-PubMedBERT-base |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Evaluation | RAGAS |
| Monitoring | LangSmith |
| Experiment Tracking | Weights and Biases |
| Frontend | Streamlit |
| Backend | FastAPI |
| Deployment | Hugging Face Spaces |
| Containers | Docker |

---

## Citation

If you use Equipoise or its evaluation toolkit in your research, please cite:

```bibtex
@software{equipoise2026,
  author    = {Kireeti , Gowtham},
  title     = {Equipoise: Evaluating Retrieval Strategy Bias in Biomedical Scientific Claim Verification},
  year      = {2026},
  url       = {https://github.com/kireeti-ai/equipoise-rag},
  note      = {GitHub repository}
}
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
