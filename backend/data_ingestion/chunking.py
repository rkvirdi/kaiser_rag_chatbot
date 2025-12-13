# chunking.py

"""
Chunking utilities for Kaiser RAG ingestion.

- Generic narrative docs (compliance, generic) use a RecursiveCharacterTextSplitter.
- Benefit / policy PDFs use a simple line-based chunker so each benefit row is a chunk.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


# ---------- Generic recursive chunking ----------

def chunk_text_recursive(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    units: str = "words",  # "words" or "chars"
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Recursive chunking with LangChain's RecursiveCharacterTextSplitter.

    - units="words": interpret chunk_size/overlap as approx word counts
    - units="chars": treat them as exact character counts
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []

    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    if units == "words":
        avg_word_chars = 6  # crude heuristic
        chunk_size_chars = int(chunk_size * avg_word_chars)
        overlap_chars = int(overlap * avg_word_chars)
    else:
        chunk_size_chars = int(chunk_size)
        overlap_chars = int(overlap)

    splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
    )
    return splitter.split_text(text)


# ---------- Benefit / policy PDFs: line-based chunking ----------

def chunk_benefit_policy_page(text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Chunk a 1–2 page benefit / policy PDF into one chunk per benefit row.

    Heuristic based on the sample you shared:
    - First non-empty line = plan_name
    - Lines that contain 'You Pay' (case-insensitive) are section headers
    - Lines under a section become individual chunks with context:
        "{plan_name} | {section} | {line}"
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return []

    plan_name = lines[0]  # simple heuristic; adjust later if needed
    chunks: List[Dict[str, Any]] = []

    current_section: Optional[str] = None

    for line in lines[1:]:
        lower = line.lower()
        # Detect section headers like "Emergency Services You Pay"
        if "you pay" in lower:
            current_section = line
            continue

        # Skip lines before first section header
        if current_section is None:
            continue

        # Treat each benefit line as a row
        content = f"{plan_name} | {current_section} | {line}"
        chunk_meta = {
            **metadata,
            "plan_name": plan_name,
            "section": current_section,
            "raw_line": line,
            "doc_type": metadata.get("doc_type", "benefit_policy"),
        }
        chunks.append({"content": content, "metadata": chunk_meta})

    logger.info(
        "Created %d benefit-policy chunks for %s (page %s)",
        len(chunks),
        metadata.get("source"),
        metadata.get("page"),
    )
    return chunks


# ---------- Top-level dispatcher ----------

def detect_doc_type(meta: Dict[str, Any]) -> str:
    return meta.get("doc_type", "generic")


def safe_id_component(value: str) -> str:
    return (
        value.replace("\\", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def create_chunks_for_documents(
    documents: List[Dict[str, Any]],
    chunk_size: int = 500,
    overlap: int = 100,
) -> List[Dict[str, Any]]:
    """
    Main entrypoint:

    Input: list of {"text": str, "metadata": dict} from data_loader.load_raw_documents
    Output: list of chunk dicts ready for JSONL / vector DB:
        {
            "id": str,
            "content": str,
            "metadata": {...}
        }
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc_idx, doc in enumerate(documents):
        text = doc.get("text", "")
        meta = dict(doc.get("metadata", {}))
        if not text.strip():
            continue

        doc_type = detect_doc_type(meta)
        source = meta.get("source", "unknown")
        page = meta.get("page", 0)
        base_id = f"{safe_id_component(Path(source).stem)}_p{page}"

        # Choose strategy based on doc_type
        if doc_type == "benefit_policy":
            page_chunks = chunk_benefit_policy_page(text, meta)
            # They already include content & metadata
            final_chunks = page_chunks
        else:
            # compliance or generic narrative: recursive splitter
            raw_chunks = chunk_text_recursive(
                text,
                chunk_size=chunk_size,
                overlap=overlap,
                units="words",
            )
            final_chunks = [
                {
                    "content": c,
                    "metadata": {
                        **meta,
                        "doc_type": doc_type,
                    },
                }
                for c in raw_chunks
            ]

        total = len(final_chunks)
        created_at = datetime.utcnow().isoformat()

        for idx, ch in enumerate(final_chunks):
            all_chunks.append(
                {
                    "id": f"{base_id}_c{idx}",
                    "content": ch["content"],
                    "metadata": {
                        **ch["metadata"],
                        "chunk_index": idx,
                        "total_chunks": total,
                        "created_at": created_at,
                    },
                }
            )

    logger.info("Created %d chunks from %d documents", len(all_chunks), len(documents))
    return all_chunks
