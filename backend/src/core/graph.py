from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, START, END

from core.state import ACAState
from agents.router import router_node
from agents.retrieve import retrieve_node
from agents.transactional import transactional_node
from agents.conversational import conversational_node
from agents.orchestrator import orchestrator_node


def route_from_router(state: ACAState) -> Literal["retrieve", "transactional", "conversational", "human_handoff"]:
    decision = (state.get("router_decision") or "Conversational").lower()

    if decision in ("rag", "policy", "general"):
        return "retrieve"
    if decision in ("transactional", "patient_data", "scheduling"):
        return "transactional"
    if decision in ("human", "off_topic", "guardrail"):
        return "human_handoff"
    return "conversational"


def build_graph():
    """
    Build and compile the LangGraph StateGraph for the ACA.
    """
    builder = StateGraph(ACAState)

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("transactional", transactional_node)
    builder.add_node("conversational", conversational_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("human_handoff", human_handoff_node)

    # Entry: START -> router
    builder.add_edge(START, "router")

    # Router -> conditional branches
    builder.add_conditional_edges(
        "router",
        route_from_router,
        {
            "retrieve": "retrieve",
            "transactional": "transactional",
            "conversational": "conversational",
            "human_handoff": "human_handoff",
        },
    )

    # After each “work” node, go to orchestrator for synthesis
    builder.add_edge("retrieve", "orchestrator")
    builder.add_edge("transactional", "orchestrator")
    builder.add_edge("conversational", "orchestrator")

    # Orchestrator is terminal (for now)
    builder.add_edge("orchestrator", END)

    # Human handoff is also terminal
    builder.add_edge("human_handoff", END)

    graph = builder.compile()
    return graph


def human_handoff_node(state: ACAState):
    """
    Node that represents routing to a human agent / Tier 2.

    Here we just set a canned response and flip the flag.
    """
    response = (
        "Your question requires a licensed representative to review your account "
        "or medical details. I'm transferring you to a human agent who can help you "
        "safely and in compliance with our policies."
    )
    history = list(state.get("chat_history", []))
    history.append(f"ACA: {response}")

    return {
        "final_response": response,
        "chat_history": history,
        "requires_human_handoff": True,
    }
