from __future__ import annotations

from typing import Dict, Any

from core.state import ACAState
from guardrails.hipaa_safety_checks import should_handoff


def orchestrator_node(state: ACAState) -> Dict[str, Any]:
    """
    Orchestrator Agent (Manager).

    Responsibilities:
      - maintain chat_history
      - inspect tool_output
      - synthesize final_response
      - decide if requires_human_handoff flag should be set
    """
    user_query = state.get("user_query", "")
    chat_history = list(state.get("chat_history", []))

    if user_query:
        chat_history.append(f"User: {user_query}")

    tool_output = state.get("tool_output", {})

    # Build a natural language answer from tool_output
    parts = []

    if isinstance(tool_output, dict):
        source = tool_output.get("source")

        if source == "RAG":
            rag = tool_output.get("raw_results", {})
            results = rag.get("results", [])
            if results:
                parts.append("Here’s what I found in our general policy and FAQ documents:")
                for idx, chunk in enumerate(results, start=1):
                    parts.append(f"{idx}. {chunk}")
            else:
                parts.append("I couldn't find any matching policy information.")
        elif source == "Transactional":
            billing = tool_output.get("billing_info")
            coverage = tool_output.get("coverage_info")
            appointment = tool_output.get("appointment_confirmation")

            if billing:
                parts.append(
                    f"For your visit on {billing['visit_date']} with {billing['doctor']}, "
                    f"your co-pay was ${billing['copay']:.2f}. "
                    f"Your deductible status is '{billing['deductible_status']}' "
                    f"and your outstanding balance is ${billing['outstanding_balance']:.2f}."
                )

            if coverage:
                status = "is covered" if coverage["covered"] else "is not covered"
                parts.append(
                    f"Regarding physical therapy, your plan ({coverage['plan_id']}) "
                    f"{status}. Details: {coverage['details']}."
                )

            if appointment:
                parts.append(
                    f"I’ve scheduled a follow-up with {appointment['doctor']} on "
                    f"{appointment['scheduled_at']} for the reason: {appointment['reason']}."
                )

            if not (billing or coverage or appointment):
                parts.append("I tried to look up your account, but couldn't find matching data.")
        elif source == "Conversational":
            parts.append(tool_output.get("message", ""))

    if not parts:
        parts.append(
            "I’m sorry, I wasn’t able to confidently answer this based on the tools I have. "
            "A human representative may be better suited to help with your question."
        )

    final_response = "\n\n".join(p for p in parts if p)

    # Append agent response to chat_history
    chat_history.append(f"ACA: {final_response}")

    # Guardrails: decide if we should flag for human handoff
    handoff = should_handoff(
        {
            "user_query": user_query,
            "tool_output": tool_output,
        }
    )

    return {
        "chat_history": chat_history,
        "final_response": final_response,
        "requires_human_handoff": handoff,
    }
