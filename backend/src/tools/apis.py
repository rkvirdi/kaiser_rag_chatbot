from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

from backend.data_ingestion.retriever import retrieve_context
from .db_connector import load_mock_patient_db
from ...data_ingestion.generator import generate_answer
from ...data_ingestion.retriever import retrieve_context


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class BillingInfo:
    member_id: str
    visit_date: str
    doctor: str
    copay: float
    deductible_status: str
    outstanding_balance: float


@dataclass
class CoverageInfo:
    plan_id: str
    procedure_code: str
    covered: bool
    details: str


@dataclass
class AppointmentConfirmation:
    member_id: str
    doctor: str
    reason: str
    scheduled_at: str  # ISO timestamp string (mocked)


# -----------------------------
# Mock API implementations
# -----------------------------


def fetch_billing_info(member_id: str, visit_date: str) -> Optional[BillingInfo]:
    """
    Mock billing lookup from JSON DB.

    Mock internal Billing service call.
    """
    db = load_mock_patient_db()
    for member in db.get("members", []):
        if member.get("member_id") != member_id:
            continue
        for visit in member.get("visits", []):
            if visit.get("visit_date") == visit_date:
                return BillingInfo(
                    member_id=member_id,
                    visit_date=visit_date,
                    doctor=visit.get("doctor", "Unknown"),
                    copay=float(visit.get("copay", 0)),
                    deductible_status=visit.get("deductible_status", "unknown"),
                    outstanding_balance=float(visit.get("outstanding_balance", 0)),
                )
    return None


def check_plan_coverage(plan_id: str, procedure_code: str) -> Optional[CoverageInfo]:
    """
    Mock coverage lookup.
    """
    db = load_mock_patient_db()
    for plan in db.get("plans", []):
        if plan.get("plan_id") != plan_id:
            continue
        procedures = plan.get("covered_procedures", {})
        if procedure_code in procedures:
            entry = procedures[procedure_code]
            return CoverageInfo(
                plan_id=plan_id,
                procedure_code=procedure_code,
                covered=bool(entry.get("covered", False)),
                details=entry.get("details", ""),
            )
    return None


def schedule_appointment(member_id: str, doctor: str, reason: str) -> AppointmentConfirmation:
    """
    Mock appointment scheduling.

    Here we just return a fake confirmation. In reality this would call
    a scheduling microservice.
    """
   
    return AppointmentConfirmation(
        member_id=member_id,
        doctor=doctor,
        reason=reason,
        scheduled_at=datetime.now().isoformat(),
    )



def RAG_Search_Tool(query: str, top_k: int = 2, include_context: bool = True) -> Dict[str, Any]:
    """
    Run retrieval + generation and return answer plus optional context.
    """
    context = None
    if include_context:
        try:
            context = retrieve_context(query, top_k=top_k)
        except Exception as e:
            context = {"error": str(e)}

    answer = generate_answer(query, top_k=top_k)
    return {"query": query, "answer": answer, "context": context}
