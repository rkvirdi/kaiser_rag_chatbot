"""
PDF processing utilities for Kaiser RAG system.
Handles text extraction and document file listing.
"""

import logging
from pathlib import Path
from typing import List

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file.
    
    Returns:
        Extracted text content from the PDF.
    
    Raises:
        ImportError: If PyPDF2 is not installed.
    """
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")
    
    try:
        text = ""
        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        logger.info(f"Extracted text from PDF: {pdf_path.name}")
        return text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def list_pdf_files(directory: Path) -> List[Path]:
    """
    List all PDF files in a directory.
    
    Args:
        directory: Path to the directory to search.
    
    Returns:
        List of Path objects for PDF files found.
    """
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []
    
    pdf_files = list(directory.glob('*.pdf'))
    logger.info(f"Found {len(pdf_files)} PDF files in {directory.name}")
    return pdf_files
