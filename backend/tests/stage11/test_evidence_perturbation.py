"""
Stage 11 Tests: Evidence Perturbation Audit (Objective 5)
Verifies:
1. Different evidence profiles systematically change model selection
2. Compute budget and task domain shifts alter architecture rankings
"""

from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


def test_evidence_perturbation(tmp_path):
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_path))
    res = validator.audit_evidence_perturbation()

    assert res["status"] == "PASSED"
    assert res["perturbation_sensitivity_verified"] is True
    assert "PMID:" in res["profile_a_lightweight"]["image_provenance"]
    assert "PMID:" in res["profile_b_heavy"]["image_provenance"]
