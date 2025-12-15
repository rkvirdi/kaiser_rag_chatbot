from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ACAState(TypedDict, total=False):
    """
    Global graph state for the Kaiser Permanente ACA.

    Only keys listed in the assignment are "required", others are optional helpers
    for transactional behavior.
    """

    # Required by assignment
    user_query: str
    chat_history: List[str]
    router_decision: str  # e.g. "RAG", "Transactional", "Conversational", "Human"
    tool_output: Any  # could be dict or str
    final_response: str
    requires_human_handoff: bool

    # Helpful extras for transactional logic (optional)
    member_id: Optional[str]
    plan_id: Optional[str]
    visit_date: Optional[str]
    procedure_code: Optional[str]
    structured_request: Optional[Dict[str, Any]]
