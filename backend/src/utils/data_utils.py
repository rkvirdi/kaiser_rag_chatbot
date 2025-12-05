"""
CSV and JSON data utilities for Kaiser RAG system.
Handles loading and parsing structured data files.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def load_csv_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load data from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
    
    Returns:
        List of dictionaries with CSV data.
    """
    if not file_path.exists():
        logger.warning(f"CSV file not found: {file_path}")
        return []
    
    try:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        
        logger.info(f"Loaded {len(data)} records from {file_path.name}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading CSV file {file_path}: {e}")
        return []


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: Path to the JSON file.
    
    Returns:
        Dictionary with JSON data or empty dict if error.
    """
    if not file_path.exists():
        logger.warning(f"JSON file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded JSON data from {file_path.name}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}


def find_record_by_field(data: List[Dict[str, Any]], field: str, value: str) -> Optional[Dict[str, Any]]:
    """
    Find a record in a list by matching a specific field value.
    
    Args:
        data: List of dictionaries to search.
        field: Field name to match against.
        value: Value to search for.
    
    Returns:
        First matching record or None if not found.
    
    Example:
        >>> members = [{'id': '1', 'name': 'John'}, {'id': '2', 'name': 'Jane'}]
        >>> find_record_by_field(members, 'id', '1')
        {'id': '1', 'name': 'John'}
    """
    for record in data:
        if record.get(field) == value:
            return record
    
    logger.debug(f"Record not found: {field}={value}")
    return None


def filter_records_by_field(data: List[Dict[str, Any]], field: str, value: str) -> List[Dict[str, Any]]:
    """
    Filter records by matching a specific field value.
    
    Args:
        data: List of dictionaries to filter.
        field: Field name to match against.
        value: Value to search for.
    
    Returns:
        List of matching records.
    
    Example:
        >>> visits = [{'member_id': '1', 'date': '2025-01-01'}, {'member_id': '1', 'date': '2025-01-02'}]
        >>> filter_records_by_field(visits, 'member_id', '1')
        [{'member_id': '1', 'date': '2025-01-01'}, {'member_id': '1', 'date': '2025-01-02'}]
    """
    filtered = [record for record in data if record.get(field) == value]
    
    logger.info(f"Filtered {len(filtered)} records where {field}={value}")
    return filtered


def get_unique_values(data: List[Dict[str, Any]], field: str) -> List[str]:
    """
    Get unique values for a field across all records.
    
    Args:
        data: List of dictionaries.
        field: Field name to extract unique values from.
    
    Returns:
        List of unique values.
    
    Example:
        >>> members = [{'plan': 'PPO'}, {'plan': 'HMO'}, {'plan': 'PPO'}]
        >>> get_unique_values(members, 'plan')
        ['PPO', 'HMO']
    """
    unique_values = list(set(record.get(field) for record in data if field in record))
    logger.info(f"Found {len(unique_values)} unique values for field: {field}")
    return unique_values


def merge_data_sources(csv_data: List[Dict[str, Any]], 
                       json_data: Dict[str, Any], 
                       join_key: str = None) -> Dict[str, Any]:
    """
    Merge CSV and JSON data sources.
    
    Args:
        csv_data: Data loaded from CSV.
        json_data: Data loaded from JSON.
        join_key: Optional key to join records on.
    
    Returns:
        Merged data dictionary.
    
    Example:
        >>> csv = [{'id': '1', 'name': 'John'}]
        >>> json_data = {'plans': [{'plan_id': 'PPO', 'name': 'PPO Basic'}]}
        >>> merged = merge_data_sources(csv, json_data)
        >>> 'plans' in merged and 'records' in merged
        True
    """
    merged = {
        "records": csv_data,
        "metadata": json_data
    }
    logger.info(f"Merged {len(csv_data)} CSV records with JSON metadata")
    return merged
