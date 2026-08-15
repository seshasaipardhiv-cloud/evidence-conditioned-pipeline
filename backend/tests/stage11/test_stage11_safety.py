"""
Stage 11 Tests: Safety Audit and Claim Boundary Matrix (Objective 11 & 12)
Verifies:
1. 14 Comprehensive safety gates across transfer datasets
2. Formal Claim Boundary Matrix classifications
"""

import json
from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


@pytest.fixture(scope="module")
def safety_fixture(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("safe_test")
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_p))
    validator.run_all_cross_dataset_experiments()
    return tmp_p


def test_stage11_safety_audit_and_claim_matrix(safety_fixture):
    safe_p = safety_fixture / "stage11_safety_audit.json"
    claim_p = safety_fixture / "stage11_claim_boundary.json"

    assert safe_p.exists()
    assert claim_p.exists()

    with open(safe_p, "r", encoding="utf-8") as f:
        safe_data = json.load(f)
    assert safe_data["overall_status"] == "PASSED"

    with open(claim_p, "r", encoding="utf-8") as f:
        claim_data = json.load(f)

    assert claim_data["Claim 1: The framework transfers across different dataset schemas."]["verdict"] == "SUPPORTED"
    assert claim_data["Claim 2: The framework adapts to different modality combinations."]["verdict"] == "SUPPORTED"
    assert claim_data["Claim 3: Evidence changes model selection."]["verdict"] == "SUPPORTED"
    assert claim_data["Claim 4: Evidence-conditioned selection consistently improves predictive performance."]["verdict"] == "PARTIALLY_SUPPORTED"
    assert claim_data["Claim 6: The framework generalizes clinically."]["verdict"] == "NOT_SUPPORTED"
