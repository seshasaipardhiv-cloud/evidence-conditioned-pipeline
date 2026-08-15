"""
Unit and regression tests for Stage 6G: Scientific Implementation–Manuscript Reconciliation Audit

Tests:
1. All 7 reconciliation JSON artifacts are generated and non-empty
2. Executed pipeline forensics identifies exactly 6 executed and 2 dormant components
3. Imputation mismatch is formally classified as PIPELINE_IMPLEMENTATION_MISMATCH
4. Multimodal cross-attention and ensembling are classified as EVIDENCE_BACKED_BUT_DORMANT
5. Temporal audit identifies progress_1 and classifies prediction epoch
6. Baseline fairness audit identifies Simple MLP as undertrained (max_iter=10)
7. Manuscript claim audit identifies the 4 core claims requiring reconciliation
8. Immutability audit passes with zero mutation across Stage 5B/5C/6A artifacts
9. Final reconciliation decision is MANUSCRIPT_ONLY_RECONCILIATION
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.scientific_reconciliation_stage6g import Stage6GReconciliationAuditor


def _setup_auditor():
    return Stage6GReconciliationAuditor(base_dir=".")


def test_1_all_reconciliation_artifacts_generated():
    auditor = _setup_auditor()
    auditor.run()

    recon_dir = Path("evidence/final/reconciliation")
    expected_files = [
        "stage6g_execution_reconciliation.json",
        "stage6g_component_discrepancy_audit.json",
        "stage6g_temporal_feature_audit.json",
        "stage6g_claim_audit.json",
        "stage6g_baseline_fairness_audit.json",
        "stage6g_immutability_audit.json",
        "stage6g_final_summary.json",
    ]
    for ef in expected_files:
        p = recon_dir / ef
        assert p.exists(), f"Missing reconciliation file: {ef}"
        assert p.stat().st_size > 50, f"File too small: {ef}"


def test_2_executed_pipeline_forensics_counts():
    auditor = _setup_auditor()
    recon = auditor.audit_executed_pipeline()

    assert recon["total_configured_components"] == 8
    assert recon["actually_executed_count"] == 6
    assert recon["dormant_component_count"] == 2
    assert recon["mismatched_component_count"] == 1

    # Check dormant
    assert recon["components"]["modality_fusion"]["actually_executed"] is False
    assert recon["components"]["ensembling"]["actually_executed"] is False

    # Check executed
    assert recon["components"]["base_learner"]["actually_executed"] is True
    assert recon["components"]["imbalance_handling"]["actually_executed"] is True
    assert recon["components"]["categorical_encoding"]["actually_executed"] is True
    assert recon["components"]["loss_function"]["actually_executed"] is True


def test_3_imputation_mismatch_classification():
    auditor = _setup_auditor()
    comp_audit = auditor.audit_component_discrepancies()

    imp = comp_audit["imputation_audit"]
    assert imp["classification"] == "PIPELINE_IMPLEMENTATION_MISMATCH"
    assert "SimpleImputer" in imp["executed_code"]
    assert imp["reconciliation_action"] == "MANUSCRIPT_CORRECTION_REQUIRED"


def test_4_multimodal_dormancy_classification():
    auditor = _setup_auditor()
    comp_audit = auditor.audit_component_discrepancies()

    cross_att = comp_audit["multimodal_dormancy_audit"]["cross_attention"]
    assert cross_att["classification"] == "EVIDENCE_BACKED_BUT_DORMANT"
    assert cross_att["executed"] is False

    ens = comp_audit["multimodal_dormancy_audit"]["average_ensembling"]
    assert ens["classification"] == "EVIDENCE_BACKED_BUT_DORMANT"
    assert ens["executed"] is False


def test_5_temporal_feature_audit_and_epoch():
    auditor = _setup_auditor()
    temp_audit = auditor.audit_temporal_features()

    assert temp_audit["temporal_contract_status"] == "TEMPORAL_CONTRACT_UNSPECIFIED"
    assert "Post-Adjuvant" in temp_audit["supported_clinical_prediction_epoch"]

    # Check progress_1 is flagged
    fb = temp_audit["feature_breakdown"]
    assert "progress_1" in fb
    assert fb["progress_1"]["category"] == "OUTCOME_PROXY"
    assert fb["progress_1"]["status"] == "SUBTLE_LEAKAGE_RISK"


def test_6_baseline_fairness_mlp_undertrained():
    auditor = _setup_auditor()
    base_audit = auditor.audit_baseline_fairness()

    mlp = base_audit["baselines_evaluated"]["simple_mlp"]
    assert mlp["fairness_rating"] == "UNDER_TRAINED_STRAWMAN"
    assert "max_iter=10" in mlp["config"]


def test_7_manuscript_claim_audit():
    auditor = _setup_auditor()
    claim_audit = auditor.audit_manuscript_claims()

    assert claim_audit["total_problematic_claims"] >= 4
    problems = [c["problem"] for c in claim_audit["claims"]]
    assert any("SimpleImputer" in p for p in problems)
    assert any("dormant" in p for p in problems)
    assert any("adjuvant" in p for p in problems)
    assert any("max_iter=10" in p for p in problems)


def test_8_immutability_audit_passes():
    auditor = _setup_auditor()
    immut = auditor.audit_immutability()

    assert immut["immutability_verified"] is True
    assert immut["mismatch_count"] == 0


def test_9_final_reconciliation_decision():
    auditor = _setup_auditor()
    summary = auditor.run()

    assert summary["overall_reconciliation_decision"] == "MANUSCRIPT_ONLY_RECONCILIATION"
    assert len(summary["exact_action_plan"]) == 5
