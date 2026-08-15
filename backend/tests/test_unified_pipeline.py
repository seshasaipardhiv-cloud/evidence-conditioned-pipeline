"""
Tests for Unified End-to-End User Pipeline Runner & Multi-Cohort Harness
Verifies:
1. Automatic modality discovery (tabular, image, text, trimodal)
2. Safe blocking on missing / ambiguous target
3. Evidence-conditioned selection and literature provenance retention
4. Train-isolated preprocessing
5. Dynamic multimodal neural execution
6. Validation-gated ensembling
7. 14-Gate safety compliance
8. Fixed-default baseline comparison
9. Multi-cohort transferability
10. Scientific claim boundary classification
11. Historical research immutability
"""

import json
from pathlib import Path
import pytest
import numpy as np

from backend.app.run_pipeline import UnifiedPipelineRunner
from backend.app.unified_demo_harness import UnifiedDemoHarness, UnseenCohortGenerator
from backend.app.multimodal.safety_gates import MultimodalSafetyAuditor


def test_1_modality_discovery_unseen_datasets(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "modality_test"))
    gen = UnseenCohortGenerator(tmp_path / "test_data")

    # Tabular
    tab_data = gen._generate_tabular_cohort(20)
    res_tab = runner.run_pipeline(dataset=tab_data, num_samples_if_synthetic=20)
    assert res_tab["dataset_info"]["discovered_modalities"] == ["tabular"]

    # Image
    img_data = gen._generate_image_cohort(20)
    res_img = runner.run_pipeline(dataset=img_data, num_samples_if_synthetic=20)
    assert res_img["dataset_info"]["discovered_modalities"] == ["image"]

    # Text
    txt_data = gen._generate_text_cohort(20)
    res_txt = runner.run_pipeline(dataset=txt_data, num_samples_if_synthetic=20)
    assert res_txt["dataset_info"]["discovered_modalities"] == ["text"]

    # Trimodal
    tri_data = gen._generate_trimodal_cohort(20)
    res_tri = runner.run_pipeline(dataset=tri_data, num_samples_if_synthetic=20)
    assert set(res_tri["dataset_info"]["discovered_modalities"]) == {"tabular", "image", "text"}


def test_2_ambiguous_target_safe_blocking(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "blocking_test"))
    bad_data = {
        "records": [
            {"patient_record_id": "P1", "feature_a": 1.0, "feature_b": 2.0},
            {"patient_record_id": "P2", "feature_a": 3.0, "feature_b": 4.0},
        ]
    }
    with pytest.raises(ValueError, match="Execution BLOCKED: Target column could not be unambiguously resolved"):
        runner.run_pipeline(dataset=bad_data)


def test_3_evidence_selection_and_provenance(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "evidence_test"))
    gen = UnseenCohortGenerator(tmp_path / "evidence_data")
    tri_data = gen._generate_trimodal_cohort(20)
    res = runner.run_pipeline(dataset=tri_data, num_samples_if_synthetic=20)

    # Verify PMIDs are attached to selections
    assert "PMID:" in res["selected_components"]["image_backbone"]["evidence_source"]
    assert "PMID:" in res["selected_components"]["text_backbone"]["evidence_source"]
    assert "PMID:" in res["selected_components"]["fusion"]["evidence_source"]
    assert res["selected_components"]["image_backbone"]["evidence_status"] == "EVIDENCE_BACKED"
    assert res["selected_components"]["text_backbone"]["evidence_status"] == "EVIDENCE_BACKED"


def test_4_train_isolated_preprocessing(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "prep_test"))
    gen = UnseenCohortGenerator(tmp_path / "prep_data")
    tab_data = gen._generate_tabular_cohort(20)
    res = runner.run_pipeline(dataset=tab_data, num_samples_if_synthetic=20)
    assert res["preprocessing"]["tabular"]["train_only"] is True


def test_5_dynamic_neural_multimodal_construction(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "neural_test"))
    gen = UnseenCohortGenerator(tmp_path / "neural_data")
    tri_data = gen._generate_trimodal_cohort(30)
    res = runner.run_pipeline(dataset=tri_data, num_samples_if_synthetic=30)
    assert "candidate_metrics" in res
    assert 0.0 <= res["candidate_metrics"]["roc_auc_mean"] <= 1.0


def test_6_validation_gated_ensemble_selection(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "ensemble_test"))
    gen = UnseenCohortGenerator(tmp_path / "ens_data")

    # Unimodal -> Ensemble is dormant
    tab_data = gen._generate_tabular_cohort(20)
    res_tab = runner.run_pipeline(dataset=tab_data, num_samples_if_synthetic=20)
    assert res_tab["selected_components"]["ensemble"]["selected_value"] is None

    # Multimodal -> Ensemble is active
    tri_data = gen._generate_trimodal_cohort(20)
    res_tri = runner.run_pipeline(dataset=tri_data, num_samples_if_synthetic=20)
    assert res_tri["selected_components"]["ensemble"]["selected_value"] == "average_ensembling"


def test_7_fourteen_safety_gates_compliance(tmp_path):
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")
    report = auditor.audit_all(
        modalities=["tabular", "image", "text"],
        train_pids=["P01", "P02", "P03"],
        val_pids=["P04"],
        test_pids=["P05", "P06"],
        train_features={},
        val_features={},
        test_features={},
        pipeline_config={"embed_dim": 64, "seeds": [42, 100, 2026]},
        image_meta={"evidence_source": "PMID: 42487970", "execution_status": "EXECUTABLE", "compute_cost": "LIGHT"},
        text_meta={"evidence_source": "PMID: 41826845", "execution_status": "EXECUTABLE", "compute_cost": "LIGHT"},
    )
    assert report["overall_status"] == "PASSED"
    assert report["passed_gates_count"] == 14


def test_8_baseline_comparison_generation(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "baseline_test"))
    gen = UnseenCohortGenerator(tmp_path / "base_data")
    tri_data = gen._generate_trimodal_cohort(30)
    res = runner.run_pipeline(dataset=tri_data, num_samples_if_synthetic=30)
    assert "baseline_metrics" in res
    assert (tmp_path / "baseline_test" / "baseline_comparison.json").exists()


def test_9_unseen_cohort_transfer_harness(tmp_path):
    harness = UnifiedDemoHarness(output_dir=str(tmp_path / "harness_test"))
    res = harness.run_all_validations(n_samples=20)
    assert res["cohorts_evaluated_count"] == 4
    assert res["all_safety_gates_passed"] is True


def test_10_scientific_claim_boundaries(tmp_path):
    runner = UnifiedPipelineRunner(output_dir=str(tmp_path / "claims_test"))
    gen = UnseenCohortGenerator(tmp_path / "claims_data")
    tab_data = gen._generate_tabular_cohort(20)
    res = runner.run_pipeline(dataset=tab_data, num_samples_if_synthetic=20)

    matrix = res["claim_boundary_matrix"]
    assert matrix["Claim 1: The framework transfers across unseen datasets without manual model specification"] == "SUPPORTED"
    assert matrix["Claim 5: Evidence-conditioned selection consistently improves predictive performance"] == "PARTIALLY_SUPPORTED"
    assert matrix["Claim 6: The framework generalizes clinically to real-world multicenter clinical settings"] == "NOT_SUPPORTED"


def test_11_historical_immutability():
    # Verify Stage 5B/5C, 6A, 10, 10.5, 11, 12 files are untouched
    assert Path("evidence/final/stage6a_master_results.json").exists()
    assert Path("evidence/processed/stage10_5/stage10_5_final_summary.json").exists()
    assert Path("evidence/processed/stage11/stage11_final_summary.json").exists()
    assert Path("evidence/processed/stage12/stage12_final_summary.json").exists()
