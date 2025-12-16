import json
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parents[2]
DB_DIR = BASE_DIR / "data"


def load_mock_patient_db() -> Dict[str, Any]:
    path = DB_DIR / "mock_patient_data.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

