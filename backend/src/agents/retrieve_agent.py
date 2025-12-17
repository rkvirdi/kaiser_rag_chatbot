from __future__ import annotations

from typing import Dict, Any

from core.state import ACAState
from tools.apis import RAG_Search_Tool


def retrieve_node(state: ACAState) -> Dict[str, Any]:
    """
    Retrieve Agent (RAG specialist) — runs the RAG tool and returns structured output.

    Calls `RAG_Search_Tool(query, top_k=..., include_context=True)` when available,
    falling back to simpler signatures if necessary.
    """
    user_query = (state.get("user_query") or "").strip()
    if not user_query:
        return {
            "tool_output": {
                "source": "RAG",
                "raw_results": {"error": "empty_query"},
                "answer": None,
            }
        }

    # Try the enhanced signature first; fall back if not supported
    try:
        raw = RAG_Search_Tool(user_query, top_k=3, include_context=True)
    except TypeError:
        raw = RAG_Search_Tool(user_query, top_k=3)

    # Normalize answer extraction for convenience
    answer = None
    if isinstance(raw, dict):
        answer = raw.get("answer") or raw.get("answer_text") or raw.get("result") or None
    else:
        answer = str(raw)

    return {
        "tool_output": {
            "source": "RAG",
            "raw_results": raw,
            "answer": answer,
        }
    }