from typing import Optional
from datetime import datetime
from backend.app.stage2.models import Provenance, ExtractionMethod, ExtractionStatus

class ProvenanceManager:
    @staticmethod
    def create_provenance(
        source_type: str,
        source_reference: str,
        extraction_method: ExtractionMethod,
        extraction_status: ExtractionStatus,
        evidence_text: Optional[str] = None
    ) -> Provenance:
        """
        Helper to construct a Provenance record explicitly ensuring all fields.
        """
        return Provenance(
            source_type=source_type,
            source_reference=source_reference,
            extraction_method=extraction_method,
            extraction_status=extraction_status,
            evidence_text=evidence_text,
            retrieval_date=datetime.now().isoformat()
        )
