# backend/tests/test_chunking.py
import os
import sys

# Add project root (kaiser_rag) to sys.path
TESTS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.data_ingestion.chunking import (
    chunk_text_recursive,
    chunk_benefit_policy_page,
    create_chunks_for_documents,
)


def test_chunk_text_recursive_basic():
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = chunk_text_recursive(text, chunk_size=10, overlap=2, units="words")
    assert len(chunks) >= 1
    # Check that chunks are non-empty strings
    assert all(isinstance(c, str) and c.strip() for c in chunks)


def test_chunk_benefit_policy_page():
    # Simulate the kind of text you showed in the policy PDF
    text = """
Platinum 90 HMO 0/10 PCP + Child Dental ALT
For effective dates January 1 - December 1, 2026

Plan Provider Office Visits You Pay
Most Primary Care Visits and most Non-Physician Specialist Visits...... $10 per visit
Most Physician Specialist Visits....................................... $20 per visit

Emergency Services You Pay
Emergency department visits............................................ $200 per visit
    """

    metadata = {
        "source": "backend/data/raw/policy/plan1.pdf",
        "page": 0,
        "doc_type": "benefit_policy",
    }

    chunks = chunk_benefit_policy_page(text, metadata)

    # We expect one chunk per benefit line (3 in this example)
    assert len(chunks) >= 2  # at least two benefit lines

    c0 = chunks[0]
    assert "content" in c0 and "metadata" in c0
    assert "Platinum 90 HMO" in c0["content"]          # plan name context
    assert "You Pay" in c0["content"]                  # section context
    assert "Primary Care" in c0["content"] or "Physician Specialist" in c0["content"]

    m0 = c0["metadata"]
    assert m0["plan_name"].startswith("Platinum 90")
    assert "section" in m0
    assert "raw_line" in m0


def test_create_chunks_for_documents_dispatch():
    # One benefit_policy doc, one compliance doc
    documents = [
        {
            "text": """
Platinum 90 HMO 0/10 PCP + Child Dental ALT
Plan Provider Office Visits You Pay
Most Primary Care Visits...... $10 per visit
            """,
            "metadata": {
                "source": "backend/data/raw/policy/plan1.pdf",
                "page": 0,
                "doc_type": "benefit_policy",
            },
        },
        {
            "text": "This is a compliance document. It has longer narrative text.",
            "metadata": {
                "source": "backend/data/raw/compliance/doc1.pdf",
                "page": 0,
                "doc_type": "compliance",
            },
        },
    ]

    chunks = create_chunks_for_documents(documents, chunk_size=10, overlap=2)

    # We should have at least one chunk from each doc
    assert len(chunks) >= 2

    benefit_chunks = [c for c in chunks if "plan1" in c["metadata"].get("source", "")]
    compliance_chunks = [c for c in chunks if "doc1" in c["metadata"].get("source", "")]

    assert len(benefit_chunks) >= 1
    assert len(compliance_chunks) >= 1

    # Check that dispatcher preserved doc_type and chunk_index
    for c in chunks:
        meta = c["metadata"]
        assert "doc_type" in meta
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "created_at" in meta
