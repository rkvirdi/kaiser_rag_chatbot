"""
Utility module initialization.
Exports common utilities for use throughout the system.
"""

from .pdf_processor import extract_text_from_pdf, list_pdf_files
from .data_utils import (
    load_csv_file,
    load_json_file,
    find_record_by_field,
    filter_records_by_field,
    get_unique_values,
    merge_data_sources
)

__all__ = [
    # PDF processing
    "extract_text_from_pdf",
    "list_pdf_files",
    
    # Data utilities
    "load_csv_file",
    "load_json_file",
    "find_record_by_field",
    "filter_records_by_field",
    "get_unique_values",
    "merge_data_sources",
]
