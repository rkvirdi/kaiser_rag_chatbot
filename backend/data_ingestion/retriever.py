# backend/data_ingestion/retriever.py

"""
Retriever for Kaiser RAG practice.

- Connects to Chroma persistent DB
- Uses MiniLM embeddings
- Returns top-k chunks (documents + metadata) for a query
"""

from pathlib import Path
from typing import Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer


# ------------------------------ CONFIG ---------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "backend" / "data" / "chroma"
COLLECTION_NAME = "kp_knowledge_base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------- RETRIEVAL LOGIC ---------------------------------


def retrieve_context(
    query: str,
    top_k: int = 5,
    where: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Retrieve top_k chunks from Chroma for a given query.

    Args:
        query: user question as text
        top_k: number of results to return
        where: optional metadata filter, e.g. {"doc_type": "benefit_policy"}

    Returns:
        Chroma query result dict with keys: documents, metadatas, distances, ids
    """
    # 1) Connect to Chroma
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    # 2) Load embedding model
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 3) Embed query
    q_emb = embed_model.encode([query], convert_to_numpy=True).tolist()

    # 4) Query vector DB
    kwargs: Dict[str, Any] = {
        "query_embeddings": q_emb,
        "n_results": top_k,
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)
    return results


# ------------------------------ CLI TEST -------------------------------------


# def _demo():
#     """Small demo to test retrieval alone."""
#     query = "What is the emergency room copay for this plan?"
#     res = retrieve_context(query, top_k=3, where=None)

#     docs = res.get("documents", [[]])[0]
#     metas = res.get("metadatas", [[]])[0]
#     dists = res.get("distances", [[]])[0]

#     print(f"\nQuery: {query}\n")
#     for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
#         print(f"--- Result {i} ---")
#         print("Distance:", dist)
#         print("Source:", meta.get("source"), "| Page:", meta.get("page"))
#         print("Content:\n", doc[:500], "...\n")


#if __name__ == "__main__":
    #_demo()
