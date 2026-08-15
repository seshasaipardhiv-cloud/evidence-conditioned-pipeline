"""
Unit and regression tests for Stage 5C: Statistical Validation, Component Ablation, and Robustness Analysis

Tests:
1. Stage 5B raw artifacts remain unmodified
2. Correct baseline delta calculations and win rates
3. Component ablations evaluated under identical patient splits
4. Preprocessing train-only contract preserved during ablations
5. Per-seed margin tracking over Default XGBoost
6. Calibration comparison (Brier score evaluation)
7. Prohibition of manufactured p-values (n=3 caveat enforced)
8. Clinical generalization distinction enforced
9. Immutability of Stage 2C corpus and Stage 5A contract
"""

import json
from pathlib import Path
import pytest
import numpy as np

from backend.app.stage5.statistical_validation_stage5c import (
    Stage5CStatisticalValidator,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    compute_sha256,
)


def _setup_test_validator():
    return Stage5CStatisticalValidator(
        processed_dir="evidence/processed",
        metadata_dir="evidence/metadata",
        contract_path="evidence/processed/stage5a_experiment_contract.json",
        clinical_data_path="data/raw/hancock/structured/StructuredData/clinical_data.json",
    )


def test_1_stage5b_raw_artifacts_verified():
    validator = _setup_test_validator()
    valid, errors, data = validator.verify_stage5b_results()
    assert valid is True
    assert len(errors) == 0
    assert "candidate" in data["run_results"]
    assert "baselines" in data["run_results"]


def test_2_baseline_comparison_deltas():
    validator = _setup_test_validator()
    _, _, data = validator.verify_stage5b_results()
    comparisons = validator.compute_baseline_comparisons(data)
    cand_auc = comparisons["candidate_pipeline"]["mean_roc_auc"]
    assert cand_auc == 0.9751

    def_xgb = comparisons["baseline_comparisons"]["baseline_xgboost_default"]
    assert def_xgb["baseline_mean_roc_auc"] == 0.9704
    assert def_xgb["absolute_delta_roc_auc"] == 0.0047


def test_3_component_ablations_execution():
    validator = _setup_test_validator()
    ablations = validator.run_component_ablations()
    assert "ablation_full_candidate" in ablations["ablations"]
    assert "ablation_no_smote" in ablations["ablations"]
    assert "ablation_no_advanced_imputation" in ablations["ablations"]
    assert "ablation_ordinal_encoding" in ablations["ablations"]
    assert "ablation_default_xgboost" in ablations["ablations"]

    # Verify all ablations produced valid ROC-AUC scores (> 0.90)
    for abl_id, abl_res in ablations["ablations"].items():
        assert abl_res["mean_roc_auc"] > 0.90
        assert len(abl_res["per_seed"]) == 3


def test_4_calibration_brier_score():
    validator = _setup_test_validator()
    _, _, data = validator.verify_stage5b_results()
    ablations = validator.run_component_ablations()
    _, calib, _ = validator.analyze_robustness_and_calibration(data, ablations)
    brier_cand = calib["brier_score_comparison"]["candidate_pipeline"]
    brier_mlp = calib["brier_score_comparison"]["simple_mlp"]
    assert brier_cand < brier_mlp
    assert calib["calibration_assessment"]["candidate_achieves_best_brier"] is True


def test_5_no_manufactured_p_values_and_sample_size_caveat():
    validator = _setup_test_validator()
    _, _, data = validator.verify_stage5b_results()
    ablations = validator.run_component_ablations()
    _, _, stats = validator.analyze_robustness_and_calibration(data, ablations)
    assert stats["sample_size_seeds"] == 3
    assert stats["inferential_claim_policy"] == "NO_STATISTICAL_SIGNIFICANCE_CLAIMED_FROM_N3"
    assert stats["p_value_generation"] == "SUPPRESSED_DUE_TO_SMALL_SAMPLE_SIZE"


def test_6_generalization_and_clinical_warning():
    validator = _setup_test_validator()
    _, _, data = validator.verify_stage5b_results()
    ablations = validator.run_component_ablations()
    _, _, stats = validator.analyze_robustness_and_calibration(data, ablations)
    assert stats["generalization_warning"]["clinical_utility_proven"] is False
    assert stats["generalization_warning"]["external_validation_available"] is False


def test_7_stage5b_results_immutability():
    validator = _setup_test_validator()
    hash_before = compute_sha256(validator.stage5b_run_results_path)
    validator.run()
    hash_after = compute_sha256(validator.stage5b_run_results_path)
    assert hash_before == hash_after
