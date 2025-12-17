from __future__ import annotations

from typing import Dict, Any

from core.state import ACAState
from tools.apis import (
    fetch_billing_info,
    check_plan_coverage,
    schedule_appointment,
)


def transactional_node(state: ACAState) -> Dict[str, Any]:
    """
    Transactional Agent.

    For this prototype, we assume:
      - member_id and plan_id are already known (e.g., via auth context)
      - visit_date and procedure_code may or may not be present
    """
    user_query = (state.get("user_query") or "").lower()
    member_id = state.get("member_id") or "MBR156655633"
    plan_id = state.get("plan_id") or "EPO_CORE"

    updates: Dict[str, Any] = {}
    tool_output: Dict[str, Any] = {"source": "Transactional"}

    # Example 1: Co-pay for specific visit
    if "co-pay" in user_query or "copay" in user_query:
        # In real life you'd parse the exact date; here we pick the last visit in DB.
        visit_date = state.get("visit_date") or "2025-11-10"
        billing = fetch_billing_info(member_id, visit_date)
        if billing:
            tool_output["billing_info"] = billing.__dict__
        else:
            tool_output["billing_info_error"] = "Visit not found for given member_id/date."

    # Example 2: Plan coverage for physical therapy
    if "physical therapy" in user_query or "pt" in user_query:
        procedure_code = state.get("procedure_code") or "PT_GENERIC"
        coverage = check_plan_coverage(plan_id, procedure_code)
        if coverage:
            tool_output["coverage_info"] = coverage.__dict__
        else:
            tool_output["coverage_info_error"] = "Coverage not found for this plan/procedure."

    # Example 3: Schedule follow-up appointment
    if "schedule" in user_query or "follow-up" in user_query or "follow up" in user_query:
        # We pretend doctor is "Dr. Chen" from the example
        confirmation = schedule_appointment(
            member_id=member_id,
            doctor="Dr. Chen",
            reason="Follow-up regarding medication discussed in last visit.",
        )
        tool_output["appointment_confirmation"] = confirmation.__dict__

    updates["tool_output"] = tool_output
    return updates
