# backend/tests/test_vector_db.py
import os
import sys

# Add project root (kaiser_rag) to sys.path
TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_ingestion.vector_db import (
    save_chunks_to_jsonl,
    index_documents,
    query_vector_db,
)
from pathlib import Path
import backend.data_ingestion.vector_db as vector_db
from backend.data_ingestion.vector_db import BASE_DIR as REAL_BASE_DIR

def test_save_chunks_to_jsonl(tmp_path, monkeypatch):
    # Arrange: fake BASE_DIR so it writes into tmp_path/backend/data/processed
   
    monkeypatch.setattr(vector_db, "BASE_DIR", tmp_path)

    chunks = [
        {
            "id": "c1",
            "content": "Hello world",
            "metadata": {"source": "test.pdf", "page": 0},
        },
        {
            "id": "c2",
            "content": "Second chunk",
            "metadata": {"source": "test.pdf", "page": 0},
        },
    ]

    # Act
    path_str = save_chunks_to_jsonl(chunks, filename="test_chunks.jsonl", overwrite=True)
    path = Path(path_str)

    # Assert
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2



def test_index_and_query(tmp_path, monkeypatch):
    # Make Chroma + processed dirs live under tmp_path
    monkeypatch.setattr(vector_db, "BASE_DIR", tmp_path)

    # Tiny fake corpus
    chunks = [
        {
            "id": "1",
            "content": "Emergency department visit copay is $200 per visit.",
            "metadata": {"source": "plan1.pdf", "page": 0},
        },
        {
            "id": "2",
            "content": "Primary care office visit copay is $10 per visit.",
            "metadata": {"source": "plan1.pdf", "page": 0},
        },
    ]

    # Index
    result = index_documents(chunks, collection_name="test_collection", overwrite=True)
    assert result["status"] == "success"
    assert result["count"] == 2

    # Query
    res = query_vector_db(
        "How much is the ER copay?",
        n_results=1,
        collection_name="test_collection",
    )

    # Simple sanity checks
    assert "documents" in res
    assert len(res["documents"][0]) >= 1
    top_doc = res["documents"][0][0]
    assert "Emergency department" in top_doc or "copay" in top_doc
