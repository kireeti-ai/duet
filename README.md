# DUET

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.11+-blue" alt="Python">
  <img src="https://img.shields.io/badge/frontend-Streamlit-red" alt="Frontend">
  <img src="https://img.shields.io/badge/backend-FastAPI-green" alt="Backend">
  <img src="https://img.shields.io/badge/llm-Groq%20%7C%20Ollama-purple" alt="LLM">
  <img src="https://img.shields.io/badge/vectorDB-ChromaDB-orange" alt="VectorDB">
  <img src="https://img.shields.io/badge/retrieval-BM25%20%7C%20Hybrid-yellow" alt="Retrieval">
  <img src="https://img.shields.io/badge/deployment-HuggingFace%20Spaces-teal" alt="Deployment">
</p>

---

**Dual-path Evidence Tracer**

Contradiction-aware scientific claim verification using dual-path Retrieval-Augmented Generation (RAG).

---

## Overview

DUET is a biomedical NLP research system designed to verify scientific claims by deliberately retrieving **both supporting and contradicting evidence** from biomedical literature.

Traditional RAG pipelines are structurally biased toward confirmation because semantic similarity retrieval favors documents that linguistically resemble the query, often suppressing null findings, contradictory trials, or opposing conclusions.

DUET addresses this by introducing a **dual-path retrieval architecture**:

* **Support Path:** Retrieves evidence supporting the claim
* **Contradiction Path:** Retrieves evidence opposing or nullifying the claim

These two evidence streams are independently retrieved, re-ranked, deduplicated, and synthesized into a balanced scientific verdict with source citations.

---

## Core Problem

Standard retrieval systems fail scientific balance because:

* Dense embeddings collapse negation signals
* Contradictory studies use different language patterns
* Similarity search over-rewards positive phrasing
* One-sided retrieval produces misleading conclusions

### Example

Claim:

> Vitamin D supplementation improves depression symptoms

Standard RAG may retrieve:

* Positive RCTs
* Supporting meta-analyses

But miss:

* Null-effect trials
* Contradictory systematic reviews
* Population-dependent failures

DUET explicitly searches for both sides.

---

## Key Innovation

### Dual-path Query Reformulation

Input claim is transformed into:

**Support Query:**

```text
Vitamin D supplementation positive effect depression improvement clinical trial evidence
```

**Contradiction Query:**

```text
Vitamin D supplementation no effect null result depression clinical trial contradicts
```

This architecture directly targets contradiction retrieval rather than assuming relevance equals truth.

---

## Features

### Retrieval

* Dense retrieval (ChromaDB + embeddings)
* Sparse retrieval (BM25)
* Hybrid retrieval
* Query reformulation retrieval
* Cross-encoder re-ranking
* Deduplication across support/contradiction paths

### Chunking Strategies

* Fixed-256
* Fixed-512
* Sliding Window
* Sentence-level
* Semantic chunking

### Embedding Models

* BAAI/bge-base-en-v1.5
* PubMedBERT
* S-PubMedBERT

### Generation

* Groq API (Llama models)
* Ollama local fallback
* Structured verdict generation
* Citation-grounded synthesis

### Evaluation

* RAGAS metrics
* Support Recall
* Contradiction Recall
* Balance Score
* Latency tracking
* Cost analysis

---

## Original Research Contributions

### Novel Architecture

* First contradiction-aware dual-path biomedical RAG system

### Novel Metrics

* **Support Recall**
* **Contradiction Recall**
* **Balance Score**

### Novel Evaluation Focus

* Chunking strategy effects on contradiction coverage
* Retrieval method effects on contradiction coverage
* Domain embedding effects on contradiction retrieval

---

## System Architecture

```text
User Claim
   |
   v
Query Reformulator
   |-------------------------|
   v                         v
Support Query          Contradiction Query
   |                         |
   v                         v
Retriever 1             Retriever 2
(Dense / BM25 / Hybrid) (Dense / BM25 / Hybrid)
   |                         |
   v                         v
Re-ranker                Re-ranker
   |                         |
   -----------Merge + Deduplicate-----------
                         |
                         v
                  Verdict Prompt Builder
                         |
                         v
                    LLM Generator
                         |
                         v
                Balanced Scientific Verdict
```

---

## Tech Stack

### Core Stack

* Python 3.11
* LangChain
* ChromaDB
* rank-bm25
* sentence-transformers
* transformers
* RAGAS
* FastAPI
* Streamlit
* Docker

### Inference

* Groq API (primary)
* Ollama (fallback)

### Monitoring

* LangSmith
* Weights & Biases

### Deployment

* Hugging Face Spaces
* GitHub Actions

---

## Dataset Sources

### Primary Dataset

**SciFact**

* 1,409 biomedical claims
* SUPPORT / CONTRADICT / NOINFO labels
* Expert-annotated
* Essential for contradiction-aware evaluation

### Secondary Datasets

* PubMedQA
* NLI4CT
* BEIR SciFact
* PubMed full corpus via Entrez API

---

## Project Structure

```text
duet/
│
├── data/
├── processed/
├── chroma_db/
├── src/
│   ├── chunking.py
│   ├── retriever.py
│   ├── pipeline.py
│   ├── evaluator.py
│   ├── ragas_eval.py
│   └── verdict_prompt.py
│
├── experiments/
├── notebooks/
├── results/
├── app.py
├── api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Evaluation Metrics

| Metric               | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| Support Recall       | Measures supporting evidence retrieval completeness    |
| Contradiction Recall | Measures contradictory evidence retrieval completeness |
| Balance Score        | Measures retrieval balance between both sides          |
| Faithfulness         | Checks hallucination minimization                      |
| Answer Relevance     | Measures claim alignment                               |
| Context Precision    | Measures retrieval usefulness                          |
| Context Recall       | Measures retrieval completeness                        |

---

| Metric               | Purpose                                                |
| -------------------- | ------------------------------------------------------ |
| Support Recall       | Measures supporting evidence retrieval completeness    |
| Contradiction Recall | Measures contradictory evidence retrieval completeness |
| Balance Score        | Measures retrieval balance between both sides          |
| Faithfulness         | Checks hallucination minimization                      |
| Answer Relevance     | Measures claim alignment                               |
| Context Precision    | Measures retrieval usefulness                          |
| Context Recall       | Measures retrieval completeness                        |

---

## Installation

```bash
git clone https://github.com/yourusername/duet.git
cd duet
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Environment Variables

Create `.env`:

```env
GROQ_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
WANDB_API_KEY=your_key
HUGGINGFACEHUB_API_TOKEN=your_key
```

---

## Running the Project

### Streamlit App

```bash
streamlit run app.py
```

### FastAPI Backend

```bash
uvicorn api:app --reload
```

### Docker

```bash
docker build -t duet .
docker run -p 8501:8501 duet
```

---

## Example Use Case

### Input

> Does omega-3 supplementation reduce depression symptoms?

### Output

```text
VERDICT: Contested
Supporting Evidence: Moderate evidence in clinically depressed populations
Contradicting Evidence: Large RCTs show limited or no general population benefit
Confidence: Low-to-moderate
```

---

## Publication Potential

DUET is structured for:

* EMNLP
* ACL Findings
* BioNLP Workshop
* TREC BioGen
* Information Retrieval journals

---

## Future Extensions

* Full PubMed live indexing
* Multi-hop contradiction retrieval
* Clinical trial-specific evidence weighting
* Fine-tuned contradiction retrievers
* Cross-domain scientific verification

---

## Guiding Principle

**DUET — Honest Evidence. Both Sides. Always.**
