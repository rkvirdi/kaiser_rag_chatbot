from pathlib import Path
from typing import List, Dict, Any
import logging
import re

import pdfplumber  # required

logger = logging.getLogger(__name__)


def list_pdf_files(directory: Path | str, recursive: bool = True) -> List[Path]:
    """
    Return a list of PDF file Paths under `directory`. By default searches recursively.
    Matches .pdf and .PDF (case-insensitive).
    """
    directory = Path(directory)

    if not directory.exists():
        logger.warning("PDF directory does not exist: %s", directory)
        return []

    if directory.is_file():
        return [directory] if directory.suffix.lower() == ".pdf" else []

    pattern = "**/*" if recursive else "*"
    return sorted(p for p in directory.glob(pattern) if p.is_file() and p.suffix.lower() == ".pdf")


def _normalize_text(text: str) -> str:
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _escape_md(text: str) -> str:
    # Keep it simple—escape pipes so Markdown tables don't break
    return (text or "").replace("|", r"\|").strip()


def table_to_markdown(table: List[List[Any]]) -> str:
    """
    Convert a pdfplumber table (list of rows) into a Markdown table string.
    Uses first row as header when it looks like a header; otherwise creates generic headers.
    """
    if not table:
        return ""

    # normalize rows -> list[str]
    rows: List[List[str]] = []
    max_cols = 0
    for row in table:
        row = row or []
        row_str = [_escape_md("" if c is None else str(c)) for c in row]
        rows.append(row_str)
        max_cols = max(max_cols, len(row_str))

    if max_cols == 0:
        return ""

    # pad rows to same width
    for r in rows:
        if len(r) < max_cols:
            r.extend([""] * (max_cols - len(r)))

    # Decide header
    first = rows[0]
    has_header = any(first) and all(isinstance(x, str) for x in first)

    if has_header:
        header = first
        body = rows[1:] if len(rows) > 1 else []
    else:
        header = [f"col_{i+1}" for i in range(max_cols)]
        body = rows

    def md_row(r: List[str]) -> str:
        return "| " + " | ".join(r) + " |"

    sep = "| " + " | ".join(["---"] * max_cols) + " |"

    out = [md_row(header), sep]
    out.extend(md_row(r) for r in body)
    return "\n".join(out).strip()


def tables_to_markdown_blocks(tables: List[List[List[Any]]], page_num: int) -> str:
    """
    Convert list of tables on a page into Markdown blocks, labeled per table.
    """
    blocks: List[str] = []
    for t_idx, table in enumerate(tables or [], start=1):
        md = table_to_markdown(table)
        if md:
            blocks.append(f"### Table {t_idx} (page {page_num})\n{md}")
    return "\n\n".join(blocks).strip()


def extract_text_from_pdf(
    path: Path | str,
    include_tables_as_text: bool = True,
) -> Dict[str, Any]:
    """
    Extract PDF content using pdfplumber only.

    Returns:
      {
        "text": "<full joined text (optionally includes tables as markdown)>",
        "pages": [
          {
            "page": 1,
            "text": "...",                 # cleaned page narrative text
            "tables": [...],               # raw tables (list of rows)
            "tables_md": "..."             # markdown (if include_tables_as_text)
          }, ...
        ],
        "metadata": { ... }
      }

    Best-effort: returns empty fields on failure (does not raise).
    """
    path = Path(path)

    if not path.exists():
        logger.warning("PDF not found: %s", path)
        return {"text": "", "pages": [], "metadata": {}}

    pages: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}

    try:
        with pdfplumber.open(str(path)) as pdf:
            try:
                metadata = dict(pdf.metadata or {})
            except Exception:
                metadata = {}

            for i, page in enumerate(pdf.pages, start=1):
                page_text = ""
                tables: List[Any] = []
                tables_md = ""

                try:
                    page_text = _normalize_text(page.extract_text() or "")
                except Exception:
                    logger.debug("Failed extracting text on page %s: %s", i, path, exc_info=True)

                try:
                    tables = page.extract_tables() or []
                except Exception:
                    logger.debug("Failed extracting tables on page %s: %s", i, path, exc_info=True)

                if include_tables_as_text and tables:
                    tables_md = tables_to_markdown_blocks(tables, page_num=i)

                pages.append(
                    {
                        "page": i,
                        "text": page_text,
                        "tables": tables,
                        "tables_md": tables_md,
                    }
                )

        # Build document-level text for RAG
        chunks: List[str] = []
        for p in pages:
            if p["text"]:
                chunks.append(p["text"])
            if include_tables_as_text and p.get("tables_md"):
                chunks.append(p["tables_md"])

        full_text = _normalize_text("\n\n".join(chunks))
        return {"text": full_text, "pages": pages, "metadata": metadata}

    except Exception:
        logger.exception("pdfplumber failed to open/read %s", path)
        return {"text": "", "pages": [], "metadata": {}}
