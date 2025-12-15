# data_loading.py

"""
Simple PDF data loader for Kaiser RAG project.

- Recursively walks data/raw/
- Loads PDFs with PyPDFLoader (one Document per page)
- Normalizes into {"text": str, "metadata": dict} records
- Infers a simple doc_type based on folder name: compliance / benefit_policy / generic
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

try:
    from langchain_community.document_loaders import PyPDFLoader
    _HAS_LANGCHAIN = True
except Exception:  # pragma: no cover
    PyPDFLoader = None
    _HAS_LANGCHAIN = False


def list_pdf_files(raw_dir: str | Path) -> List[Path]:
    """Return all PDF files under raw_dir (recursive)."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning("Raw data directory not found: %s", raw_path)
        return []
    return sorted(raw_path.rglob("*.pdf"))


def infer_doc_type(path: Path) -> str:
    """
    Infer a high-level document type from the path.

    Assumes structure like:
        data/raw/compliance/...
        data/raw/policy/...
    """
    parts = [part.lower() for part in path.parts]
    if "compliance" in parts:
        return "compliance"
    if "policies" in parts:
        return "benefit_policy"
    return "generic"


def load_raw_documents(raw_dir: str | Path = "data/raw") -> List[Dict[str, Any]]:
    """
    Load all PDFs in data/raw into a simple list of dicts:
        {"text": page_content, "metadata": {...}}

    This is the input for the chunking step.
    """
    if not _HAS_LANGCHAIN:
        raise RuntimeError(
            "langchain_community is required. Install with "
            "`pip install langchain-community`."
        )

    raw_dir = Path(raw_dir)
    pdf_files = list_pdf_files(raw_dir)
    if not pdf_files:
        logger.warning("No PDF files found under %s", raw_dir)

    docs: List[Dict[str, Any]] = []

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        doc_type = infer_doc_type(pdf_path)

        for page_idx, page in enumerate(pages):
            meta = dict(page.metadata or {})
            # Normalize metadata keys we care about
            meta["source"] = str(pdf_path)
            meta.setdefault("page", page_idx)  # fallback if not provided
            meta["doc_type"] = doc_type

            docs.append(
                {
                    "text": page.page_content,
                    "metadata": meta,
                }
            )

    logger.info("Loaded %d pages from %s", len(docs), raw_dir)
    return docs
