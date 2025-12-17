from __future__ import annotations

from typing import Dict, Any


SENSITIVE_KEYWORDS = [
    "ssn",
    "social security",
    "full medical record",
    "diagnosis code",
    "mental health notes",
]


def detect_off_topic(user_query: str) -> bool:
    """
    Extremely naive off-topic detector.
    """
    keywords = ["kaiser", "kp", "co-pay", "copay", "coverage", "appointment", "doctor", "physician"]
    # If no health/insurance-ish word appears, treat as off-topic
    return not any(k in user_query.lower() for k in keywords)


def detect_potential_phi_leak(text: str) -> bool:
    """
    Naive 'did we just leak explicit PHI' detector.
    """
    lower = text.lower()
    return any(k in lower for k in SENSITIVE_KEYWORDS)


def should_handoff(state: Dict[str, Any]) -> bool:
    """
    Combined logic:
    - off topic OR
    - potential PHI leak in tool output.
    """
    user_query = state.get("user_query", "") or ""
    tool_output = state.get("tool_output", "") or ""

    if detect_off_topic(user_query):
        return True

    if isinstance(tool_output, str) and detect_potential_phi_leak(tool_output):
        return True

    if isinstance(tool_output, dict):
        joined = " ".join(str(v) for v in tool_output.values())
        if detect_potential_phi_leak(joined):
            return True

    return False
