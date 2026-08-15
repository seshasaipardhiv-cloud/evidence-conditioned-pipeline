import pytest
from backend.app.stage2.provenance import ProvenanceManager
from backend.app.stage2.models import ExtractionMethod, ExtractionStatus

def test_metadata_provenance():
    prov = ProvenanceManager.create_provenance(
        source_type="test",
        source_reference="123",
        extraction_method=ExtractionMethod.manual,
        extraction_status=ExtractionStatus.explicit
    )
    assert prov.source_type == "test"
    assert prov.source_reference == "123"

def test_evidence_provenance():
    prov = ProvenanceManager.create_provenance(
        source_type="test",
        source_reference="123",
        extraction_method=ExtractionMethod.regex_based,
        extraction_status=ExtractionStatus.structured,
        evidence_text="Accuracy increased by 5%"
    )
    assert prov.evidence_text == "Accuracy increased by 5%"
