"""
retriever.py
Implements all four retrieval strategies:
- dense: ChromaDB vector similarity search
- bm25: keyword frequency search via rank-bm25
- hybrid: reciprocal rank fusion of dense and BM25
- queryreform: LLM rewrites claim, then dense search
"""

import pickle
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import (
    EMBEDDING_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION_NAME,
    TOP_K, HYBRID_DENSE_WEIGHT, HYBRID_BM25_WEIGHT, RETRIEVAL_METHOD
)

_cache = {}


def load_resources():
    """Load embedding model, ChromaDB collection, BM25 index, and abstracts."""
    if _cache:
        return (
            _cache["embedding_model"],
            _cache["collection"],
            _cache["bm25"],
            _cache["abstracts"]
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    embedding_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)

    with open(f"{CHROMA_DB_PATH}/bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open(f"{CHROMA_DB_PATH}/abstracts.pkl", "rb") as f:
        abstracts = pickle.load(f)

    _cache["embedding_model"] = embedding_model
    _cache["collection"] = collection
    _cache["bm25"] = bm25
    _cache["abstracts"] = abstracts

    return embedding_model, collection, bm25, abstracts


def dense_retrieve(query, embedding_model, collection, top_k=TOP_K):
    """
    Dense retrieval: embed query and find most similar abstracts in ChromaDB.
    Problem: collapses negation signals -- contradicting papers score low.
    """
    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "distances"]
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "text": results["metadatas"][0][i]["text"],
            "score": 1 - results["distances"][0][i],
            "method": "dense"
        })

    return retrieved


def bm25_retrieve(query, bm25, abstracts, top_k=TOP_K):
    """
    BM25 sparse retrieval: keyword frequency matching.
    Preserves negation language -- contradicting papers score correctly.
    TREC BioGen 2025: BM25 achieved contradiction recall of 0.750.
    """
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    retrieved = []
    for idx in top_indices:
        retrieved.append({
            "id": abstracts[idx]["id"],
            "title": abstracts[idx]["title"],
            "text": abstracts[idx]["text"],
            "score": float(scores[idx]),
            "method": "bm25"
        })

    return retrieved


def hybrid_retrieve(query, embedding_model, collection, bm25, abstracts, top_k=TOP_K):
    """
    Hybrid retrieval: reciprocal rank fusion of dense and BM25 results.
    Combines semantic understanding with keyword precision.
    Dense weight: 0.6, BM25 weight: 0.4 (configurable in config.py).
    """
    dense_results = dense_retrieve(query, embedding_model, collection, top_k=top_k * 2)
    bm25_results = bm25_retrieve(query, bm25, abstracts, top_k=top_k * 2)

    # Reciprocal rank fusion
    rrf_scores = {}

    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, {"score": 0, "data": result})
        rrf_scores[doc_id]["score"] += HYBRID_DENSE_WEIGHT * (1 / (rank + 1))

    for rank, result in enumerate(bm25_results):
        doc_id = result["id"]
        if doc_id not in rrf_scores:
            rrf_scores[doc_id] = {"score": 0, "data": result}
        rrf_scores[doc_id]["score"] += HYBRID_BM25_WEIGHT * (1 / (rank + 1))

    sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    retrieved = []
    for item in sorted_docs:
        result = item["data"]
        result["score"] = item["score"]
        result["method"] = "hybrid"
        retrieved.append(result)

    return retrieved


def retrieve(query, method=RETRIEVAL_METHOD, reformulated_query=None):
    """
    Main retrieval function. Calls the correct strategy based on config.
    method options: dense | bm25 | hybrid | queryreform
    """
    embedding_model, collection, bm25, abstracts = load_resources()

    if method == "dense":
        return dense_retrieve(query, embedding_model, collection)

    elif method == "bm25":
        return bm25_retrieve(query, bm25, abstracts)

    elif method == "hybrid":
        return hybrid_retrieve(query, embedding_model, collection, bm25, abstracts)

    elif method == "queryreform":
        search_query = reformulated_query if reformulated_query else query
        return dense_retrieve(search_query, embedding_model, collection)

    else:
        raise ValueError(f"Unknown retrieval method: {method}. Choose: dense | bm25 | hybrid | queryreform")


if __name__ == "__main__":
    test_claim = "Vitamin D supplementation improves depression symptoms"
    print(f"Test claim: {test_claim}")
    print(f"Retrieval method: {RETRIEVAL_METHOD}\n")

    results = retrieve(test_claim)

    for i, r in enumerate(results):
        print(f"[{i+1}] {r['title']}")
        print(f"     Score: {r['score']:.4f} | Method: {r['method']}")
        print(f"     {r['text'][:150]}...")
        print()
