# backend/data_ingestion/ingest.py

"""
End-to-end ingestion script for Kaiser RAG:

1. Load raw PDFs from backend/data/raw
2. Chunk them by doc type (compliance vs benefit_policy)
3. Index chunks into ChromaDB + save JSONL snapshot
"""

import logging
from pathlib import Path

from .data_loading import load_raw_documents
from .chunking import create_chunks_for_documents
from .vector_db import index_documents


def main(
    raw_dir: str | Path = "backend/data/raw",
    chunk_size: int = 500,
    overlap: int = 100,
    collection_name: str = "kp_knowledge_base",
) -> None:
    logging.info("Starting ingestion pipeline")
    logging.info("Raw dir: %s | chunk_size=%d | overlap=%d | collection=%s",
                 raw_dir, chunk_size, overlap, collection_name)

    # 1) Load pages from PDFs
    docs = load_raw_documents(raw_dir)
    logging.info("Loaded %d raw pages", len(docs))

    if not docs:
        logging.warning("No documents found. Exiting.")
        return

    # 2) Chunk documents
    chunks = create_chunks_for_documents(
        docs,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    logging.info("Created %d chunks", len(chunks))

    if not chunks:
        logging.warning("No chunks created. Exiting.")
        return

    # 3) Index into Chroma (and save JSONL snapshot)
    result = index_documents(
        chunks,
        collection_name=collection_name,
        overwrite=True,
        save_jsonl=True,
    )

    logging.info("Ingestion complete: %s", result)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    main()
