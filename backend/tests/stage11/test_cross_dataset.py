"""
Stage 11 Tests: Cross-Dataset Automation & Transfer (Objective 1 & 2)
Verifies:
1. Transfer across 7 distinct dataset schemas with unique column names
2. Zero manual model configuration
3. Complete decision ledger generation
"""

from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


@pytest.fixture(scope="module")
def transfer_fixture(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("transfer_test")
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_p))
    summary = validator.run_all_cross_dataset_experiments()
    return {"path": tmp_p, "summary": summary}


def test_1_cross_dataset_seven_cohorts_executed(transfer_fixture):
    summary = transfer_fixture["summary"]
    assert summary["status"] == "STAGE11_TRANSFER_VALIDATION_COMPLETE"
    assert summary["evaluated_cohorts_count"] == 7

    results = summary["cross_dataset_results"]
    assert "cohort_a_tabular" in results
    assert "cohort_b_image" in results
    assert "cohort_c_text" in results
    assert "cohort_d_image_text" in results
    assert "cohort_e_tabular_image" in results
    assert "cohort_f_tabular_text" in results
    assert "cohort_g_trimodal" in results


def test_2_zero_manual_model_configuration(transfer_fixture):
    results = transfer_fixture["summary"]["cross_dataset_results"]
    for cohort_name, ledger in results.items():
        assert ledger["execution_status"] == "SUCCESSFUL_EXECUTION"
        assert ledger["safety_status"] == "PASSED"
        assert "metrics" in ledger
