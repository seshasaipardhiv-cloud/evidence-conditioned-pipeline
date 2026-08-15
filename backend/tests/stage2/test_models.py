import pytest
from backend.app.stage2.models import (
    EvidenceClaim, EmpiricalResult, EvidenceStatus, MechanismCategory, Mechanism, Provenance, ExtractionMethod, ExtractionStatus
)

def test_negative_evidence():
    result = EmpiricalResult(metric="Accuracy", direction="degradation")
    assert result.direction == "degradation"

def test_null_metric_in_result():
    result = EmpiricalResult(metric=None, direction="qualitative")
    assert result.metric is None

def test_quantitative_result():
    result = EmpiricalResult(metric="AUROC", baseline_value=0.81, method_value=0.84, delta=0.03, direction="improvement")
    assert result.delta == 0.03
    
def test_qualitative_result():
    result = EmpiricalResult(metric="Visual Quality", direction="qualitative")
    assert result.direction == "qualitative"
    
def test_unknown_task_handling():
    prov = Provenance(
        source_type="test", source_reference="test",
        extraction_method=ExtractionMethod.manual,
        extraction_status=ExtractionStatus.structured,
        retrieval_date="now"
    )
    claim = EvidenceClaim(
        evidence_id="1", paper_id="p1", claim="test", mechanisms=[], 
        evidence_location="Abstract",
        extraction_method=ExtractionMethod.manual, provenance=prov
    )
    # Task defaults to None (unknown)
    assert claim.task is None
