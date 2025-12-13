"""
Vector DB ingestion pipeline using ChromaDB (new PersistentClient API).

Features:
- Saves processed chunks to JSONL
- Computes embeddings (SentenceTransformers - local, free)
- Stores vectors + metadata into a persistent ChromaDB collection
- Supports overwrite / re-indexing cleanly
"""

import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# PATH HELPERS
# ------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]  # Adjust depth if needed


def ensure_processed_dir() -> Path:
    """Ensure backend/data/processed exists."""
    path = BASE_DIR / "backend" / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_chroma_dir() -> Path:
    """Ensure backend/data/chroma exists."""
    path = BASE_DIR / "backend" / "data" / "chroma"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------------------
# JSONL SAVE
# ------------------------------------------------------------------------------

def save_chunks_to_jsonl(
    chunks: List[Dict[str, Any]],
    filename: str = "knowledge_chunks.jsonl",
    overwrite: bool = True,
) -> str:
    processed_dir = ensure_processed_dir()
    path = processed_dir / filename
    mode = "w" if overwrite else "a"

    with open(path, mode, encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(chunks)} chunks → {path}")
    return str(path)


# ------------------------------------------------------------------------------
# EMBEDDING MODEL INITIALIZATION
# ------------------------------------------------------------------------------

_EMBED_MODEL = None


def get_embedding_model():
    """Lazy-load the sentence transformer model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        logger.info(
            "Loading embedding model: sentence-transformers/all-MiniLM-L6-v2"
        )
        _EMBED_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBED_MODEL


# ------------------------------------------------------------------------------
# CHROMA CLIENT (new PersistentClient API)
# ------------------------------------------------------------------------------

def get_chroma_client():
    """
    Create a persistent Chroma client using the new API.

    This replaces the deprecated `Client(Settings(...))` pattern.
    """
    chroma_dir = ensure_chroma_dir()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client


# ------------------------------------------------------------------------------
# INDEX DOCUMENTS INTO CHROMA
# ------------------------------------------------------------------------------

def index_documents(
    chunks: List[Dict[str, Any]],
    collection_name: str = "kp_knowledge_base",
    overwrite: bool = True,
    save_jsonl: bool = True,
) -> Dict[str, Any]:
    """
    Stores chunks into ChromaDB with embeddings + metadata.

    Steps:
    1. Optionally save chunks to JSONL
    2. Clean and recreate the Chroma collection if overwrite=True
    3. Compute embeddings using SentenceTransformers
    4. Insert into ChromaDB
    """

    # 1) Save JSONL snapshot
    if save_jsonl:
        save_chunks_to_jsonl(chunks, overwrite=True)

    # 2) Initialize Chroma client
    client = get_chroma_client()

    # Overwrite collection if requested
    existing = {c.name for c in client.list_collections()}
    if overwrite and collection_name in existing:
        client.delete_collection(name=collection_name)
        logger.info(f"Overwriting existing Chroma collection: {collection_name}")

    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"Using Chroma collection: {collection_name}")

    # 3) Prepare data for embedding
    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [c["id"] for c in chunks]

    logger.info(f"Computing embeddings for {len(texts)} chunks...")
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    # 4) Insert into Chroma
    logger.info("Adding embeddings to ChromaDB...")
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings,
    )

    logger.info(
        f"Indexed {len(chunks)} chunks into ChromaDB collection '{collection_name}'"
    )

    # PersistentClient persists automatically via DuckDB/parquet in path

    return {
        "collection": collection_name,
        "count": len(chunks),
        "status": "success",
    }


# ------------------------------------------------------------------------------
# QUERY FUNCTION (for testing)
# ------------------------------------------------------------------------------

def query_vector_db(
    query_text: str,
    n_results: int = 5,
    collection_name: str = "kp_knowledge_base",
):
    """Simple helper to test retrieval from Chroma."""
    client = get_chroma_client()
    collection = client.get_collection(collection_name)

    model = get_embedding_model()
    q_emb = model.encode([query_text], convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=q_emb,
        n_results=n_results,
    )

    return results
