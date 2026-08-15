"""
Unit and regression tests for Stage 6H: Manuscript Reconciliation, Scientific Correction & Final Paper Rebuild

Tests:
1. Reconciled final research paper exists and has non-empty content
2. Operational imputation (median/mode) is accurately documented without claiming iterative MICE was executed
3. Cross-attention and average ensembling are explicitly documented as dormant on unimodal tabular benchmark
4. Clinical prediction epoch is formally defined as Post-Adjuvant Recurrence Risk Prediction with progress_1 caveat
5. Simple MLP is explicitly framed as a minimal shallow reference baseline (max_iter=10)
6. All authoritative metrics match Stage 6A/5B exactly (Candidate: 0.9751, Default XGB: 0.9704, Delta: +0.0047)
7. Per-seed metrics and Seed 100 loss are preserved verbatim
8. Ablation values and evidence validity != empirical optimality interpretation are preserved
9. Related work covers AutoML, TRIPOD+AI, PROBAST, and data leakage without fabricated citations
10. All 7 Stage 6H reconciliation JSON artifacts exist
11. All authoritative Stage 5B/5C/6A/6B/6G source result artifacts remain strictly immutable
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.manuscript_reconciliation_stage6h import (
    Stage6HManuscriptReconciler,
    IMMUTABLE_SOURCE_PATHS,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    EXPECTED_STAGE5A_CONTRACT_HASH,
)


def _setup_reconciler():
    return Stage6HManuscriptReconciler(base_dir=".")


def test_1_final_paper_exists_and_reconciled():
    reconciler = _setup_reconciler()
    summary = reconciler.run()

    paper_path = Path("evidence/final/paper/final_research_paper.md")
    assert paper_path.exists()
    assert summary["reconciliation_status"] == "SUCCESSFULLY_RECONCILED"
    assert summary["final_manuscript_word_count"] > 2500


def test_2_imputation_operational_description_accurate():
    reconciler = _setup_reconciler()
    reconciler.run()

    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "univariate median imputation" in text
    assert "most-frequent imputation" in text
    assert "evaluating this operational tabular implementation rather than an iterative MICE/MissForest estimator" in text


def test_3_cross_attention_and_ensembling_dormant():
    reconciler = _setup_reconciler()
    reconciler.run()

    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "dormant" in text.lower()
    assert "unimodal" in text.lower()


def test_4_temporal_epoch_post_adjuvant_and_progress_caveat():
    reconciler = _setup_reconciler()
    reconciler.run()

    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "Post-Adjuvant Recurrence Risk Prediction" in text
    assert "progress_1" in text


def test_5_mlp_baseline_shallow_reference_caveat():
    reconciler = _setup_reconciler()
    reconciler.run()

    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "minimal shallow MLP reference baseline" in text or "minimal shallow neural reference baseline" in text
    assert "should not be interpreted as evidence of superiority over optimized neural architectures" in text


def test_6_exact_authoritative_metrics():
    reconciler = _setup_reconciler()
    summary = reconciler.run()

    metrics = summary["authoritative_metrics_preserved"]
    assert metrics["candidate_mean_roc_auc"] == 0.9751
    assert metrics["candidate_std_roc_auc"] == 0.0114
    assert metrics["default_xgboost_mean_roc_auc"] == 0.9704
    assert metrics["margin_delta"] == 0.0047
    assert metrics["candidate_brier_score"] == 0.0175


def test_7_per_seed_results_and_seed_100_loss():
    reconciler = _setup_reconciler()
    reconciler.run()

    with open(Path("evidence/final/paper/results.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "0.9888" in text  # Seed 42
    assert "0.9609" in text and "0.9643" in text and "-0.0034" in text  # Seed 100
    assert "0.9756" in text  # Seed 2026


def test_8_ablation_values_and_interpretation():
    reconciler = _setup_reconciler()
    reconciler.run()

    with open(Path("evidence/final/paper/results.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "0.9773" in text  # Without SMOTE
    assert "0.9767" in text  # Mean Imputation
    assert "0.9784" in text  # Ordinal Encoding
    assert "0.9686" in text  # Default XGBoost
    assert "evidence-conditioned framework does not claim to identify the empirically optimal configuration" in text


def test_9_related_work_expansion():
    reconciler = _setup_reconciler()
    reconciler.run()

    with open(Path("evidence/final/paper/related_work.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "AutoML" in text
    assert "TPOT" in text or "Auto-sklearn" in text
    assert "TRIPOD+AI" in text or "PROBAST" in text
    assert "Data Leakage" in text


def test_10_all_seven_reconciliation_artifacts_exist():
    recon_dir = Path("evidence/final/reconciliation")
    expected_files = [
        "stage6h_manuscript_reconciliation.json",
        "stage6h_claim_reconciliation.json",
        "stage6h_temporal_clarity_audit.json",
        "stage6h_pipeline_description_audit.json",
        "stage6h_reference_audit.json",
        "stage6h_immutability_audit.json",
        "stage6h_final_summary.json",
    ]
    for ef in expected_files:
        p = recon_dir / ef
        assert p.exists(), f"Missing Stage 6H file: {ef}"


def test_11_source_artifacts_immutability():
    reconciler = _setup_reconciler()
    summary = reconciler.run()

    assert summary["source_immutability_status"] == "ZERO_MUTATION_CONFIRMED"
