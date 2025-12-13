from pathlib import Path
from typing import List, Any, Dict
import csv
import json
import logging

logger = logging.getLogger(__name__)


def load_csv_file(path: Path | str, encoding: str = "utf-8") -> List[Dict[str, Any]]:
    """
    Load a CSV file and return a list of dict rows (uses header row as keys).
    """
    path = Path(path)
    if not path.exists():
        logger.warning("CSV file not found: %s", path)
        return []

    try:
        with path.open("r", encoding=encoding, newline="") as fh:
            reader = csv.DictReader(fh)
            return [dict(row) for row in reader]
    except Exception as e:
        logger.exception("Failed to read CSV %s: %s", path, e)
        return []


def load_json_file(path: Path | str, encoding: str = "utf-8") -> Any:
    """
    Load and parse a JSON file. Returns parsed object or None on error.
    """
    path = Path(path)
    if not path.exists():
        logger.warning("JSON file not found: %s", path)
        return None

    try:
        with path.open("r", encoding=encoding) as fh:
            return json.load(fh)
    except json.JSONDecodeError as jde:
        logger.exception("JSON decode error for %s: %s", path, jde)
        return None
    except Exception as e:
        logger.exception("Failed to read JSON %s: %s", path, e)
        return None