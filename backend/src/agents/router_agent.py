from __future__ import annotations

from typing import Dict, Any

from guardrails.hipaa_safety_checks import detect_off_topic
from core.state import ACAState


def router_node(state: ACAState) -> Dict[str, Any]:
    """
    Very simple heuristic router.

    Output values (router_decision):
      - "RAG"            -> Retrieve Agent
      - "Transactional"  -> Transactional Agent
      - "Conversational" -> Conversational Agent
      - "Human"          -> Human handoff / guardrail trigger
    """
    user_query = state.get("user_query", "") or ""
    q = user_query.lower()

    if detect_off_topic(user_query):
        decision = "Human"
    elif any(word in q for word in ["copay", "co-pay", "bill", "billing", "deductible"]):
        decision = "Transactional"
    elif any(word in q for word in ["appointment", "schedule", "reschedule", "follow-up", "follow up"]):
        decision = "Transactional"
    elif any(word in q for word in ["coverage", "cover", "benefit", "physical therapy", "pt"]):
        # coverage is technically transactional (plan APIs) but we treat as transactional here
        decision = "Transactional"
    elif any(word in q for word in ["policy", "faq", "how does my plan work", "what is"]):
        decision = "RAG"
    else:
        decision = "Conversational"

    return {"router_decision": decision}
