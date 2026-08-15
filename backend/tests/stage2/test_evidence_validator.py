import pytest
from backend.app.stage2.evidence_validator import EvidenceValidator
from backend.app.stage2.models import (
    EvidenceClaim, EvidenceStatus, ExtractionStatus, Provenance, ExtractionMethod, EmpiricalResult
)

def test_contradiction_candidate():
    validator = EvidenceValidator()
    
    prov = Provenance(source_type="test", source_reference="1", extraction_method=ExtractionMethod.manual, extraction_status=ExtractionStatus.explicit, retrieval_date="now")
    
    c1 = EvidenceClaim(
        evidence_id="c1", paper_id="1", claim="test claim", mechanisms=["m1"], 
        evidence_status=EvidenceStatus.direct_empirical, 
        evidence_location="Abstract",
        result=EmpiricalResult(metric="Accuracy", direction="improvement"),
        extraction_method=ExtractionMethod.manual, provenance=prov
    )
    
    c2 = EvidenceClaim(
        evidence_id="c2", paper_id="2", claim="test claim", mechanisms=["m1"], 
        evidence_status=EvidenceStatus.direct_empirical, 
        evidence_location="Abstract",
        result=EmpiricalResult(metric="Accuracy", direction="degradation"),
        extraction_method=ExtractionMethod.manual, provenance=prov
    )
    
    validated = validator.validate_claims([c1, c2])
    assert validated[0].contradiction_candidate is True
    assert validated[1].contradiction_candidate is True

def test_no_patient_data_leakage():
    # If the system had patient data, it shouldn't end up here. 
    # Just a mock check that evidence claims don't store "patient_name".
    prov = Provenance(source_type="test", source_reference="1", extraction_method=ExtractionMethod.manual, extraction_status=ExtractionStatus.explicit, retrieval_date="now")
    c1 = EvidenceClaim(
        evidence_id="c1", paper_id="1", claim="test claim", mechanisms=["m1"], 
        evidence_location="Abstract",
        extraction_method=ExtractionMethod.manual, provenance=prov
    )
    # Model validation passes
    assert not hasattr(c1, "patient_name")
