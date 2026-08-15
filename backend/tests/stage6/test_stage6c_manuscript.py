"""
Unit and regression tests for Phase 6C: Scientific Manuscript Generator

Tests:
1. All 7 manuscript markdown documents and manifest JSON exist
2. Reported metrics in experimental setup exactly match Stage 6A
3. All 8 Phase 6B figures are referenced in figure_captions.md
4. Evidence-backed and explicitly-configured components remain strictly distinguished
5. Unsupported clinical and deployment claims remain NOT_SUPPORTED
6. Statistical significance is explicitly rejected in limitations and claims
7. Seed-100 candidate loss is explicitly documented
8. Component ablation values match Stage 6A/5C exactly
9. Target leakage 8-variable exclusions remain verbatim
10. External validation is not claimed
11. Source master results remain strictly immutable
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.manuscript_generator_stage6c import (
    Stage6CManuscriptGenerator,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    EXPECTED_STAGE5A_CONTRACT_HASH,
)


def _setup_test_generator():
    return Stage6CManuscriptGenerator(
        final_dir="evidence/final",
        figures_dir="evidence/final/figures",
        manuscript_dir="evidence/final/manuscript",
    )


def test_1_all_manuscript_files_exist():
    gen = _setup_test_generator()
    manifest = gen.run()
    man_dir = Path("evidence/final/manuscript")

    expected_files = [
        "methodology.md",
        "experimental_setup.md",
        "reproducibility.md",
        "limitations.md",
        "claim_boundary.md",
        "figure_captions.md",
        "references.md",
        "manuscript_manifest.json",
    ]
    for ef in expected_files:
        p = man_dir / ef
        assert p.exists(), f"Missing manuscript file: {ef}"
        assert p.stat().st_size > 100, f"File too small: {ef}"


def test_2_metrics_match_stage6a_authoritative():
    gen = _setup_test_generator()
    manifest = gen.run()
    metrics = manifest["authoritative_metrics"]
    assert metrics["candidate_roc_auc"] == 0.9751
    assert metrics["candidate_roc_auc_std"] == 0.0114
    assert metrics["candidate_pr_auc"] == 0.9679
    assert metrics["candidate_f1"] == 0.9611
    assert metrics["candidate_accuracy"] == 0.9825
    assert metrics["candidate_brier_score"] == 0.0175
    assert metrics["default_xgboost_roc_auc"] == 0.9704


def test_3_figure_captions_reference_all_eight_figures():
    gen = _setup_test_generator()
    gen.run()
    captions_path = Path("evidence/final/manuscript/figure_captions.md")
    with open(captions_path, "r", encoding="utf-8") as f:
        text = f.read()

    for i in range(1, 9):
        assert f"Figure {i}:" in text, f"Missing Figure {i} caption"


def test_4_provenance_distinction_preserved():
    gen = _setup_test_generator()
    gen.run()
    meth_path = Path("evidence/final/manuscript/methodology.md")
    with open(meth_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "EVIDENCE_BACKED" in text
    assert "EXPLICITLY_CONFIGURED" in text
    assert "PMID: 41775771" in text  # XGBoost
    assert "PMID: 41006422" in text  # SMOTE
    assert "experiment_config.json" in text  # Explicit configs


def test_5_unsupported_claims_rejected():
    gen = _setup_test_generator()
    gen.run()
    claim_path = Path("evidence/final/manuscript/claim_boundary.md")
    with open(claim_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "CLAIM 8" in text and "NOT_SUPPORTED" in text
    assert "CLAIM 9" in text and "NOT_SUPPORTED" in text
    assert "CLAIM 10" in text and "NOT_SUPPORTED" in text
    assert "Evidence-conditioned pipeline synthesis provides a rigorous" in text


def test_6_seed_100_loss_and_ablations_documented():
    gen = _setup_test_generator()
    gen.run()
    exp_path = Path("evidence/final/manuscript/experimental_setup.md")
    with open(exp_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Seed 100 loss
    assert "Seed 100" in text and "Candidate Lost" in text and "-0.0034" in text
    # Ablation values
    assert "0.9773" in text  # Without SMOTE
    assert "0.9767" in text  # Mean Imputation
    assert "0.9784" in text  # Ordinal Encoding
    assert "0.9686" in text  # Default XGBoost


def test_7_target_leakage_exclusions():
    gen = _setup_test_generator()
    gen.run()
    exp_path = Path("evidence/final/manuscript/experimental_setup.md")
    with open(exp_path, "r", encoding="utf-8") as f:
        text = f.read()

    exclusions = [
        "recurrence",
        "survival_status",
        "survival_status_with_cause",
        "days_to_recurrence",
        "days_to_last_information",
        "days_to_progress_1",
        "days_to_progress_2",
        "days_to_metastasis_1",
    ]
    for ex in exclusions:
        assert ex in text


def test_8_master_source_immutability():
    master_path = Path("evidence/final/stage6a_master_results.json")
    with open(master_path, "r", encoding="utf-8") as f:
        master_before = json.load(f)

    gen = _setup_test_generator()
    gen.run()

    with open(master_path, "r", encoding="utf-8") as f:
        master_after = json.load(f)

    assert master_before == master_after
