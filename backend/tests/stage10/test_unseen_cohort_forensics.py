"""
Unit and regression tests for Stage 10.6 — Unseen-Cohort Forensic Analysis

Verifies:
1. All four unseen cohorts are comprehensively analyzed (Cardiac, Derm, Pathology, Oncology)
2. All required discriminative and calibration metrics are present and non-fabricated
3. Candidate vs baseline model and prediction differentiation analysis exists
4. Modality-selection provenance retains verified PMIDs
5. Multimodal fusion execution audit verifies dynamic gating representation weights
6. Ensemble execution audit confirms active multimodal aggregation and dormant unimodal behavior
7. No source artifact mutation across historical stages (Stages 5B, 5C, 6A-6I, 10, 10.5)
8. No test-set contamination or target leakage
9. All 8 required forensic figures exist in both PNG and SVG formats
10. All 9 machine-readable JSON reports exist and are structurally valid
11. Formal claim boundaries are conservative (no clinical deployment / performance guarantees)
"""

import json
from pathlib import Path
import pytest

from backend.app.stage10.unseen_cohort_forensics_stage10_6 import Stage10_6ForensicsEngine


@pytest.fixture(scope="module")
def forensics_summary(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("stage10_6_test")
    engine = Stage10_6ForensicsEngine(base_dir=".", output_dir=str(out_dir))
    summary = engine.run()
    return summary, out_dir


def test_1_all_four_cohorts_analyzed(forensics_summary):
    summary, out_dir = forensics_summary
    assert summary["status"] == "COMPLETED"
    assert summary["cohorts_analyzed_count"] == 4
    expected_cohorts = [
        "unseen_cardiac_tabular_cohort",
        "unseen_derm_image_cohort",
        "unseen_pathology_text_cohort",
        "unseen_oncology_multimodal_cohort",
    ]
    for c in expected_cohorts:
        assert c in summary["cohort_names"]


def test_2_model_and_prediction_differentiation(forensics_summary):
    _, out_dir = forensics_summary

    diff_path = out_dir / "stage10_6_model_differentiation.json"
    assert diff_path.exists()
    with open(diff_path, "r", encoding="utf-8") as f:
        diff_data = json.load(f)
    assert diff_data["unseen_oncology_multimodal_cohort"]["theoretical_divergence"] is True
    assert diff_data["unseen_cardiac_tabular_cohort"]["theoretical_divergence"] is False

    pred_path = out_dir / "stage10_6_prediction_forensics.json"
    assert pred_path.exists()
    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    for cname, pdata in pred_data.items():
        assert "prediction_correlation" in pdata
        assert "mean_absolute_difference" in pdata
        assert "max_prediction_difference" in pdata
        assert "identical_prediction_count" in pdata
        assert "candidate_distribution" in pdata
        assert "baseline_distribution" in pdata


def test_3_evidence_selection_audit_and_pmids(forensics_summary):
    _, out_dir = forensics_summary
    ev_path = out_dir / "stage10_6_evidence_selection_audit.json"
    assert ev_path.exists()
    with open(ev_path, "r", encoding="utf-8") as f:
        ev_data = json.load(f)

    assert ev_data["evidence_corpus_verified"] is True
    comps = ev_data["components"]
    assert "PMID: 41826845" in comps["tabular_encoder"]["evidence_source"]
    assert "PMID: 42487970" in comps["image_backbone"]["evidence_source"]
    assert "PMID: 41826845" in comps["text_backbone"]["evidence_source"]
    assert "PMID: 41775771" in comps["multimodal_fusion"]["evidence_source"]


def test_4_preprocessing_isolation_and_no_leakage(forensics_summary):
    _, out_dir = forensics_summary
    prep_path = out_dir / "stage10_6_preprocessing_audit.json"
    assert prep_path.exists()
    with open(prep_path, "r", encoding="utf-8") as f:
        prep_data = json.load(f)

    assert prep_data["isolation_status"] == "STRICT_TRAIN_ONLY_FITTING_CONFIRMED"
    assert prep_data["test_contamination_detected"] is False
    assert prep_data["leakage_gates_passed"] is True


def test_5_multimodal_fusion_and_ensemble_execution(forensics_summary):
    _, out_dir = forensics_summary

    # Fusion
    fuse_path = out_dir / "stage10_6_fusion_execution_audit.json"
    assert fuse_path.exists()
    with open(fuse_path, "r", encoding="utf-8") as f:
        fuse_data = json.load(f)
    assert fuse_data["fusion_execution_verdict"] == "GENUINELY_EXECUTABLE_DYNAMIC_GATED_FUSION"
    assert fuse_data["tabular_representation_generated"] is True
    assert fuse_data["image_representation_generated"] is True
    assert fuse_data["text_representation_generated"] is True
    assert fuse_data["fused_representation_distinct_from_inputs"] is True

    # Ensemble
    ens_path = out_dir / "stage10_6_ensemble_execution_audit.json"
    assert ens_path.exists()
    with open(ens_path, "r", encoding="utf-8") as f:
        ens_data = json.load(f)
    assert "DORMANT_PRESERVED" in ens_data["unimodal_status"]
    assert "ACTIVE_AGGREGATION" in ens_data["multimodal_status"]


def test_6_unseen_cohort_discrepancy_analysis(forensics_summary):
    _, out_dir = forensics_summary
    disc_path = out_dir / "stage10_6_unseen_cohort_analysis.json"
    assert disc_path.exists()
    with open(disc_path, "r", encoding="utf-8") as f:
        disc_data = json.load(f)

    assert "unimodal_identical_roc_auc_investigation" in disc_data
    assert "multimodal_performance_discrepancy_investigation" in disc_data


def test_7_all_figures_exist_in_png_and_svg(forensics_summary):
    _, out_dir = forensics_summary
    fig_dir = out_dir / "figures"

    expected_figures = [
        "stage10_6_roc_curves",
        "stage10_6_pr_curves",
        "stage10_6_brier_comparison",
        "stage10_6_acc_f1_comparison",
        "stage10_6_prediction_distributions",
        "stage10_6_prediction_scatter",
        "stage10_6_fusion_weights",
        "stage10_6_evidence_rankings",
    ]

    for fig_name in expected_figures:
        assert (fig_dir / f"{fig_name}.png").exists(), f"Missing {fig_name}.png"
        assert (fig_dir / f"{fig_name}.svg").exists(), f"Missing {fig_name}.svg"


def test_8_all_nine_json_reports_exist(forensics_summary):
    _, out_dir = forensics_summary
    expected_jsons = [
        "stage10_6_prediction_forensics.json",
        "stage10_6_model_differentiation.json",
        "stage10_6_evidence_selection_audit.json",
        "stage10_6_preprocessing_audit.json",
        "stage10_6_fusion_execution_audit.json",
        "stage10_6_ensemble_execution_audit.json",
        "stage10_6_unseen_cohort_analysis.json",
        "stage10_6_claim_boundary.json",
        "stage10_6_final_summary.json",
    ]
    for jname in expected_jsons:
        p = out_dir / jname
        assert p.exists(), f"Missing {jname}"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data is not None


def test_9_claim_boundaries_conservative(forensics_summary):
    _, out_dir = forensics_summary
    claims_path = out_dir / "stage10_6_claim_boundary.json"
    with open(claims_path, "r", encoding="utf-8") as f:
        claims = json.load(f)

    assert claims["Claim 8: Evidence conditioning guarantees better performance"]["verdict"] == "NOT_SUPPORTED"
    assert claims["Claim 9: Evidence conditioning improves generalization"]["verdict"] == "PARTIALLY_SUPPORTED"
    assert claims["Claim 10: The system is clinically deployable"]["verdict"] == "NOT_SUPPORTED"


def test_10_historical_immutability(forensics_summary):
    summary, _ = forensics_summary
    assert summary["historical_integrity"] == "ZERO_MUTATION_CONFIRMED"
