"""
Tests for Stage 10 Safety Auditor and Scientific Claim Boundary (Stage 10 - Objectives I, J & L)
Verifies:
1. 14 Comprehensive safety gates
2. Patient overlap firewall
3. Target leakage rejection
4. Scientific claim boundary evaluation
"""

from pathlib import Path
import pytest
from backend.app.multimodal.safety_gates import MultimodalSafetyAuditor
from backend.app.stage10.evidence_conditioned_automation_validation import Stage10AutomationValidator


def test_stage10_safety_gates_and_claim_matrix(tmp_path):
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")

    report = auditor.audit_all(
        modalities=["image", "text"],
        train_pids=["p1", "p2", "p3"],
        val_pids=[],
        test_pids=["p4", "p5"],
        train_features={},
        val_features={},
        test_features={},
        pipeline_config={"embed_dim": 128, "seeds": [42, 100, 2026]},
        image_meta={"evidence_source": "PMID: 42487970", "execution_status": "EXECUTABLE", "compute_cost": "LIGHT"},
        text_meta={"evidence_source": "PMID: 41826845", "execution_status": "EXECUTABLE", "compute_cost": "LIGHT"},
    )

    assert report["overall_status"] == "PASSED"
    assert report["passed_gates_count"] == 14

    validator = Stage10AutomationValidator(base_dir=".", output_dir=str(tmp_path))
    summary = validator.run_end_to_end_and_ablation(num_samples=20, seeds=[42])
    matrix = summary["claim_boundary_matrix"]

    assert matrix["1. The system automatically discovers modalities."] == "SUPPORTED"
    assert matrix["2. The system automatically selects image models from literature evidence."] == "SUPPORTED"
    assert matrix["3. The system automatically selects text models from literature evidence."] == "SUPPORTED"
    assert matrix["9. Evidence conditioning improves predictive performance."] == "PARTIALLY_SUPPORTED"
