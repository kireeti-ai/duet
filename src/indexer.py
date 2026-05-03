"""
indexer.py
Loads SciFact abstracts, generates embeddings, builds ChromaDB + BM25 index.
Run once — persists to disk.
"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import torch
import os
import chromadb
from rank_bm25 import BM25Okapi
import pickle
from tqdm import tqdm
from src.config import (
    EMBEDDING_MODEL, CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME, SCIFACT_DATASET
)


def load_scifact_abstracts():
    """Load all abstracts from SciFact corpus via BEIR format."""
    print(" Loading SciFact dataset (BEIR format)...")
    dataset = load_dataset(SCIFACT_DATASET, "corpus")
    corpus = dataset['corpus']

    abstracts = []
    for item in corpus:
        abstracts.append({
            "id": str(item['_id']),
            "title": item['title'],
            "text": item['text'],
            "full_text": item['title'] + " " + item['text']
        })

    print(f" Loaded {len(abstracts)} abstracts")
    return abstracts


def build_chroma_index(abstracts):
    """Build ChromaDB vector index from abstracts."""
    print(f"🔧 Building ChromaDB index using {EMBEDDING_MODEL}...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f" Using device: {device}")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        device=device
    )

    batch_size = 32
    print(f" Batch size: {batch_size}")

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for i in tqdm(range(0, len(abstracts), batch_size), desc="Indexing"):
        batch = abstracts[i:i + batch_size]
        texts = [a['full_text'] for a in batch]
        ids = [a['id'] for a in batch]
        metadatas = [{"title": a['title'], "text": a['text']} for a in batch]

        embeddings = embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f" ChromaDB index built — {collection.count()} abstracts indexed")
    return collection


def build_bm25_index(abstracts):
    """Build BM25 keyword index from abstracts."""
    print("🔧 Building BM25 index...")

    tokenized_corpus = [a['full_text'].lower().split() for a in abstracts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(f"{CHROMA_DB_PATH}/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
    with open(f"{CHROMA_DB_PATH}/abstracts.pkl", "wb") as f:
        pickle.dump(abstracts, f)

    print(f" BM25 index built — {len(abstracts)} abstracts indexed")
    return bm25


if __name__ == "__main__":
    abstracts = load_scifact_abstracts()
    build_chroma_index(abstracts)
    build_bm25_index(abstracts)
    print("\n All indexes built and saved to disk.")
