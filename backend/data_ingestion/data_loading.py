"""
Data loading module for Kaiser RAG system.
Handles loading and processing of patient data, compliance documents, and policies.

Architecture:
- Patient Data (Transactional): Stored in structured format (CSV/JSON)
- Compliance/Policies (Knowledge): Prepared for vector DB ingestion
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Import utilities
from src.utils.data_utils import (
    load_csv_file,
    load_json_file,
    find_record_by_field,
    filter_records_by_field,
)
from src.utils.pdf_processor import (
    extract_text_from_pdf,
    list_pdf_files,
)

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage data from various sources."""

    def __init__(self, data_dir: str = None):
        """
        Initialize DataLoader with data directory path.
        
        Args:
            data_dir: Path to the data folder. Defaults to backend/data/
        """
        if data_dir is None:
            # Get relative path from this file to the data folder
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir / "data"
        
        self.data_dir = Path(data_dir)
        self.patient_data_dir = self.data_dir / "patient_data"
        self.compliance_dir = self.data_dir / "compliance"
        self.policies_dir = self.data_dir / "policies"
        self.guidelines_dir = self.data_dir / "guidelines"

    def load_patient_members(self) -> List[Dict[str, Any]]:
        """
        Load patient members data from CSV.
        
        Returns:
            List of dictionaries containing patient member information.
        """
        csv_path = self.patient_data_dir / "patient_members.csv"
        return load_csv_file(csv_path)

    def load_patient_mock_db(self) -> Dict[str, Any]:
        """
        Load comprehensive patient mock database from JSON.
        
        Returns:
            Dictionary containing plans, members, visits, and procedures.
        """
        json_path = self.patient_data_dir / "patient_mock_db.json"
        return load_json_file(json_path)

    def load_patient_visits(self) -> List[Dict[str, Any]]:
        """
        Load patient visit history from CSV.
        
        Returns:
            List of dictionaries containing visit information.
        """
        csv_path = self.patient_data_dir / "patient_visits.csv"
        return load_csv_file(csv_path)

    def load_plan_coverage(self) -> List[Dict[str, Any]]:
        """
        Load covered procedures and plan coverage information from CSV.
        
        Returns:
            List of dictionaries containing procedure coverage details.
        """
        csv_path = self.patient_data_dir / "plan_covered_procedures.csv"
        return load_csv_file(csv_path)

    def load_all_patient_data(self) -> Dict[str, Any]:
        """
        Load all patient-related data.
        
        Returns:
            Dictionary containing members, visits, and coverage information.
        """
        return {
            "members": self.load_patient_members(),
            "visits": self.load_patient_visits(),
            "coverage": self.load_plan_coverage(),
            "mock_db": self.load_patient_mock_db()
        }

    def get_member_by_id(self, member_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific member's information by member ID.
        
        Args:
            member_id: The member ID to look up.
        
        Returns:
            Dictionary with member details or None if not found.
        """
        members = self.load_patient_members()
        return find_record_by_field(members, 'member_id', member_id)

    def get_member_plan(self, member_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the plan information for a specific member.
        
        Args:
            member_id: The member ID to look up.
        
        Returns:
            Dictionary with plan details or None if not found.
        """
        member = self.get_member_by_id(member_id)
        if not member:
            return None
        
        plan_id = member.get('plan_id')
        mock_db = self.load_patient_mock_db()
        
        plans = mock_db.get('plans', [])
        for plan in plans:
            if plan.get('plan_id') == plan_id:
                return plan
        
        logger.warning(f"Plan not found for member {member_id}: {plan_id}")
        return None

    def get_member_visits(self, member_id: str) -> List[Dict[str, Any]]:
        """
        Get all visits for a specific member.
        
        Args:
            member_id: The member ID to look up.
        
        Returns:
            List of visit records for the member.
        """
        visits = self.load_patient_visits()
        return filter_records_by_field(visits, 'member_id', member_id)


class ComplianceDocumentProcessor:
    """Process compliance, policy, and guideline documents for vector DB ingestion."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize document processor.
        
        Args:
            data_dir: Path to the data folder.
        """
        if data_dir is None:
            current_dir = Path(__file__).parent.parent
            data_dir = current_dir / "data"
        
        self.data_dir = Path(data_dir)
        self.compliance_dir = self.data_dir / "compliance"
        self.policies_dir = self.data_dir / "policies"
        self.guidelines_dir = self.data_dir / "guidelines"

    def load_compliance_documents(self) -> List[Dict[str, Any]]:
        """
        Load and process compliance documents.
        
        Returns:
            List of documents with metadata for vector DB ingestion.
        """
        from .chunking import create_chunks_with_metadata
        
        documents = []
        
        if not self.compliance_dir.exists():
            logger.warning(f"Compliance directory not found: {self.compliance_dir}")
            return documents
        
        pdf_files = list_pdf_files(self.compliance_dir)
        for file_path in pdf_files:
            text = extract_text_from_pdf(file_path)
            chunks = create_chunks_with_metadata(
                text=text,
                source=f"compliance/{file_path.name}",
                doc_type="compliance"
            )
            documents.extend(chunks)
        
        logger.info(f"Loaded {len(documents)} compliance document chunks")
        return documents

    def load_policy_documents(self) -> List[Dict[str, Any]]:
        """
        Load and process policy documents.
        
        Returns:
            List of documents with metadata for vector DB ingestion.
        """
        from .chunking import create_chunks_with_metadata
        
        documents = []
        
        if not self.policies_dir.exists():
            logger.warning(f"Policies directory not found: {self.policies_dir}")
            return documents
        
        pdf_files = list_pdf_files(self.policies_dir)
        for file_path in pdf_files:
            text = extract_text_from_pdf(file_path)
            chunks = create_chunks_with_metadata(
                text=text,
                source=f"policies/{file_path.name}",
                doc_type="policy"
            )
            documents.extend(chunks)
        
        logger.info(f"Loaded {len(documents)} policy document chunks")
        return documents

    def load_guideline_documents(self) -> List[Dict[str, Any]]:
        """
        Load and process guideline documents.
        
        Returns:
            List of documents with metadata for vector DB ingestion.
        """
        from .chunking import create_chunks_with_metadata
        
        documents = []
        
        if not self.guidelines_dir.exists():
            logger.warning(f"Guidelines directory not found: {self.guidelines_dir}")
            return documents
        
        pdf_files = list_pdf_files(self.guidelines_dir)
        for file_path in pdf_files:
            text = extract_text_from_pdf(file_path)
            chunks = create_chunks_with_metadata(
                text=text,
                source=f"guidelines/{file_path.name}",
                doc_type="guideline"
            )
            documents.extend(chunks)
        
        logger.info(f"Loaded {len(documents)} guideline document chunks")
        return documents

    def load_all_knowledge_documents(self) -> List[Dict[str, Any]]:
        """
        Load all knowledge documents (compliance, policies, guidelines).
        
        Returns:
            Combined list of all knowledge documents ready for vector DB.
        """
        all_docs = []
        all_docs.extend(self.load_compliance_documents())
        all_docs.extend(self.load_policy_documents())
        all_docs.extend(self.load_guideline_documents())
        
        logger.info(f"Total knowledge documents loaded: {len(all_docs)}")
        return all_docs


# Convenience functions for quick access
def get_data_loader(data_dir: str = None) -> DataLoader:
    """Create and return a DataLoader instance."""
    return DataLoader(data_dir)


def get_document_processor(data_dir: str = None) -> ComplianceDocumentProcessor:
    """Create and return a ComplianceDocumentProcessor instance."""
    return ComplianceDocumentProcessor(data_dir)


def load_knowledge_documents(data_dir: str = None) -> List[Dict[str, Any]]:
    """Load all knowledge documents for vector DB ingestion."""
    processor = ComplianceDocumentProcessor(data_dir)
    return processor.load_all_knowledge_documents()


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("KAISER RAG DATA LOADER - DEMO")
    print("=" * 60)
    
    # Initialize loaders
    data_loader = DataLoader()
    doc_processor = ComplianceDocumentProcessor()
    
    # Load patient data
    print("\n[TRANSACTIONAL DATA - Patient Database]")
    print("-" * 60)
    members = data_loader.load_patient_members()
    visits = data_loader.load_patient_visits()
    coverage = data_loader.load_plan_coverage()
    print(f"Total Members: {len(members)}")
    print(f"Total Visits: {len(visits)}")
    print(f"Total Covered Procedures: {len(coverage)}")
    
    # Example member lookup
    print("\n[EXAMPLE MEMBER LOOKUP]")
    print("-" * 60)
    if members:
        first_member_id = members[0]['member_id']
        member_info = data_loader.get_member_by_id(first_member_id)
        member_plan = data_loader.get_member_plan(first_member_id)
        member_visits = data_loader.get_member_visits(first_member_id)
        print(f"Member ID: {first_member_id}")
        print(f"Member Info: {json.dumps(member_info, indent=2)}")
        print(f"Member Visits: {len(member_visits)}")
        if member_plan:
            print(f"Plan: {member_plan.get('plan_name')}")
    
    # Load knowledge documents
    print("\n[KNOWLEDGE DATA - For Vector DB Ingestion]")
    print("-" * 60)
    knowledge_docs = doc_processor.load_all_knowledge_documents()
    print(f"Total Knowledge Document Chunks: {len(knowledge_docs)}")
    
    # Display sample documents
    if knowledge_docs:
        print("\nFirst 2 document chunks:")
        for doc in knowledge_docs[:2]:
            print(f"\nID: {doc['id']}")
            print(f"Type: {doc['type']}")
            print(f"Content Type: {doc.get('content_type', 'text')}")
            print(f"Source: {doc['source']}")
            print(f"Content Preview: {doc['content'][:200]}...")
    else:
        print("No knowledge documents found. Add PDF files to compliance/, policies/, and guidelines/ directories.")
    
    print("\n" + "=" * 60)
