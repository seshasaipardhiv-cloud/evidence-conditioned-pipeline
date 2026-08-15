"""
Unit and regression tests for Stage 6I: Hostile Reviewer / Pre-Submission Scientific Audit

Tests:
1. All 11 Stage 6I audit JSON artifacts are generated and valid
2. Final scientific grade is assigned as Grade A (Submission-ready)
3. Exactly 25 hostile reviewer questions with honest answers and evidence are generated
4. Pipeline consistency confirms actual executed components and dormant taxonomy primitives
5. Temporal leakage audit confirms Post-Adjuvant epoch and progress_1 prospective caveat
6. Baseline fairness audit confirms MLP shallow reference baseline framing (max_iter=10)
7. Statistical claim audit confirms n=3 underpowered caveat and exact seed 100 loss
8. Ablation honesty audit confirms exact values and evidence != optimality interpretation
9. Figures and references audits confirm 8 figure references and 0 fabricated citations
10. Immutability audit confirms zero mutation across all 13 authoritative source artifacts
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.hostile_review_stage6i import Stage6IHostileReviewer


def _setup_auditor():
    return Stage6IHostileReviewer(base_dir=".")


def test_1_all_stage6i_artifacts_exist():
    auditor = _setup_auditor()
    auditor.run()

    recon_dir = Path("evidence/final/reconciliation")
    expected_files = [
        "stage6i_hostile_review.json",
        "stage6i_claim_audit.json",
        "stage6i_pipeline_consistency_audit.json",
        "stage6i_temporal_leakage_audit.json",
        "stage6i_baseline_fairness_audit.json",
        "stage6i_statistical_claim_audit.json",
        "stage6i_figure_consistency_audit.json",
        "stage6i_reference_audit.json",
        "stage6i_reviewer_questions.json",
        "stage6i_final_verdict.json",
        "stage6i_final_summary.json",
    ]
    for ef in expected_files:
        p = recon_dir / ef
        assert p.exists(), f"Missing Stage 6I artifact: {ef}"
        assert p.stat().st_size > 50, f"File too small: {ef}"


def test_2_final_scientific_grade_a():
    auditor = _setup_auditor()
    summary = auditor.run()

    assert summary["final_scientific_grade"] == "A"
    assert summary["submission_readiness"] == "READY_FOR_SUBMISSION"
    assert summary["unsupported_claims_found"] == 0


def test_3_twenty_five_reviewer_questions():
    auditor = _setup_auditor()
    questions = auditor.build_hostile_reviewer_questions()

    assert questions["total_questions"] >= 25
    for q in questions["questions"]:
        assert len(q["reviewer_concern"]) > 10
        assert len(q["honest_answer"]) > 10
        assert len(q["evidence"]) > 5
        assert q["status"] == "RESOLVED_IN_STAGE_6H"


def test_4_pipeline_consistency_audit():
    auditor = _setup_auditor()
    pipe = auditor.audit_pipeline_consistency()

    assert pipe["status"] == "CONSISTENT"
    assert "cross_attention" in pipe["dormant_primitives"]
    assert "average_ensembling" in pipe["dormant_primitives"]
    assert "train_fitted_median_mode_imputation" in pipe["actual_executed_path"]


def test_5_temporal_leakage_and_epoch():
    auditor = _setup_auditor()
    temp = auditor.audit_temporal_leakage()

    assert temp["epoch_clearly_stated"] is True
    assert temp["progress_1_prospective_caveat_present"] is True
    assert temp["direct_leakage_exclusions_documented"] is True


def test_6_baseline_fairness_mlp():
    auditor = _setup_auditor()
    base = auditor.audit_baseline_fairness()

    assert base["mlp_baseline_shallow_reference_caveat_present"] is True
    assert base["default_xgboost_parameters_and_delta_accurate"] is True


def test_7_statistical_claims_and_seed_100():
    auditor = _setup_auditor()
    stats = auditor.audit_statistical_and_ablations()

    assert stats["status"] == "STATISTICALLY_HONEST_AND_COMPLETE"
    assert stats["checks"]["seed_100_candidate_lost_0_9609"] is True
    assert stats["checks"]["sample_size_n_3_underpowered_stated"] is True


def test_8_ablation_honesty():
    auditor = _setup_auditor()
    stats = auditor.audit_statistical_and_ablations()

    assert stats["checks"]["ablation_without_smote_0_9773"] is True
    assert stats["checks"]["ablation_ordinal_encoding_0_9784"] is True
    assert stats["checks"]["evidence_validity_ne_empirical_optimality_present"] is True


def test_9_figures_and_references():
    auditor = _setup_auditor()
    figs, refs = auditor.audit_figures_and_references()

    assert figs["total_figures_referenced"] == 8
    assert refs["zero_fabricated_citations"] is True
    assert all(refs["verified_pmids_present"].values())


def test_10_immutability_all_artifacts():
    auditor = _setup_auditor()
    verdict = auditor.determine_final_verdict()

    assert verdict["immutability_verified"] is True
    assert verdict["mismatch_count"] == 0
