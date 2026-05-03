import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# --- Models ---
GROQ_VERDICT_MODEL = "llama-3.3-70b-versatile"
GROQ_REFORMULATOR_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Retrieval Settings ---
RETRIEVAL_METHOD = "dense"   # options: dense | bm25 | hybrid | queryreform
TOP_K = 5
HYBRID_DENSE_WEIGHT = 0.6
HYBRID_BM25_WEIGHT = 0.4

# --- ChromaDB ---
CHROMA_DB_PATH = "chroma_db"
CHROMA_COLLECTION_NAME = "scifact_abstracts"

# --- Dataset ---
SCIFACT_DATASET = "BeIR/scifact"
VALIDATION_SPLIT_SIZE = 300

# --- Paths ---
RESULTS_DIR = "results"
