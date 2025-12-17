from __future__ import annotations

from typing import Dict, Any

from core.state import ACAState


def conversational_node(state: ACAState) -> Dict[str, Any]:
    """
    Conversational Agent for simple Q&A / chit-chat.

    This is a simple template-based response for now.
    Later you can plug in an LLM here.
    """
    user_query = state.get("user_query", "")

    response = (
        "Thanks for reaching out to Kaiser Permanente. "
        "Right now I don't see a need to access your records. "
        "Here is a general answer based on what you asked:\n\n"
        f"- You said: \"{user_query}\"\n"
        "- At this tier, I can help with basic questions about plans, co-pays, and appointments.\n"
        "If you need specific details about your account, I can transfer you to a human representative."
    )

    return {"tool_output": {"source": "Conversational", "message": response}}
