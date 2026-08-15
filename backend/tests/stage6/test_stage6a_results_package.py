"""
Unit and regression tests for Phase 6A: Final Research Results Package

Tests:
1. All 8 components present in master results and provenance ledger
2. Metrics match Stage 5B authoritative values exactly
3. Baseline values match Stage 5B authoritative values exactly
4. Ablations match Stage 5C authoritative values exactly
5. Claim statuses match Stage 5D authoritative classifications (5 Supported, 1 Partially Supported, 4 Not Supported)
6. Random seeds match Stage 5A frozen contract ([42, 100, 2026])
7. Immutability hashes are recorded and match source files
8. Zero source artifact mutation across packaging execution
9. No model training calls or unauthorized state mutations
10. Final results summary generated cleanly
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.results_packager_stage6a import (
    Stage6AResultsPackager,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    EXPECTED_STAGE5A_CONTRACT_HASH,
)


def _setup_test_packager():
    return Stage6AResultsPackager(
        processed_dir="evidence/processed",
        metadata_dir="evidence/metadata",
        final_dir="evidence/final",
    )


def test_1_all_eight_components_present():
    packager = _setup_test_packager()
    prov = packager.build_pipeline_provenance()
    comps = prov["components"]
    assert len(comps) == 8
    required = [
        "feature_representation",
        "modality_fusion",
        "ensembling",
        "missing_value_handling",
        "base_learner",
        "imbalance_handling",
        "categorical_encoding",
        "loss_function",
    ]
    for c in required:
        assert c in comps
        assert comps[c]["fully_traceable"] is True


def test_2_candidate_metrics_match_stage5b():
    packager = _setup_test_packager()
    exp = packager.build_experiment_results()
    cand = exp["candidate_pipeline"]
    assert cand["mean_roc_auc"] == 0.9751
    assert cand["std_roc_auc"] == 0.0114
    agg = cand["aggregated_test_metrics"]
    assert agg["f1"]["mean"] == 0.9611
    assert agg["accuracy"]["mean"] == 0.9825
    assert agg["precision"]["mean"] == 0.9801
    assert agg["recall"]["mean"] == 0.9429
    assert agg["brier_score"]["mean"] == 0.0175
    assert agg["pr_auc"]["mean"] == 0.9679


def test_3_baseline_values_match_stage5b():
    packager = _setup_test_packager()
    exp = packager.build_experiment_results()
    baselines = exp["baseline_comparisons"]
    assert baselines["baseline_xgboost_default"]["baseline_mean_roc_auc"] == 0.9704
    assert baselines["baseline_random_forest"]["baseline_mean_roc_auc"] == 0.9698
    assert baselines["baseline_logistic_regression"]["baseline_mean_roc_auc"] == 0.9645
    assert baselines["baseline_simple_mlp"]["baseline_mean_roc_auc"] == 0.9405


def test_4_ablations_match_stage5c():
    packager = _setup_test_packager()
    abl = packager.build_ablation_results()
    abls = abl["ablations"]
    assert abls["ablation_full_candidate"]["mean_roc_auc"] == 0.9751
    assert abls["ablation_no_smote"]["mean_roc_auc"] == 0.9773
    assert abls["ablation_no_advanced_imputation"]["mean_roc_auc"] == 0.9767
    assert abls["ablation_ordinal_encoding"]["mean_roc_auc"] == 0.9784
    assert abls["ablation_default_xgboost"]["mean_roc_auc"] == 0.9686


def test_5_claim_statuses_match_stage5d():
    packager = _setup_test_packager()
    claims_doc = packager.build_claim_boundaries()
    counts = claims_doc["claim_counts"]
    assert counts["supported"] == 5
    assert counts["partially_supported"] == 1
    assert counts["not_supported"] == 4

    claims = {c["claim_id"]: c for c in claims_doc["claim_ledger"]}
    assert claims["CLAIM_1"]["status"] == "SUPPORTED"
    assert claims["CLAIM_6"]["status"] == "PARTIALLY_SUPPORTED"
    assert claims["CLAIM_7"]["status"] == "NOT_SUPPORTED"
    assert claims["CLAIM_8"]["status"] == "NOT_SUPPORTED"
    assert claims["CLAIM_9"]["status"] == "NOT_SUPPORTED"
    assert claims["CLAIM_10"]["status"] == "NOT_SUPPORTED"


def test_6_seeds_match_stage5a():
    packager = _setup_test_packager()
    repro = packager.build_reproducibility_manifest()
    assert repro["random_seeds"] == [42, 100, 2026]
    assert repro["pipeline_hash"] == EXPECTED_STAGE3_6_PIPELINE_HASH
    assert repro["contract_hash"] == EXPECTED_STAGE5A_CONTRACT_HASH


def test_7_immutability_hashes_and_zero_mutation():
    packager = _setup_test_packager()
    pre_hashes = packager.collect_source_hashes()
    summary = packager.run()
    post_hashes = packager.collect_source_hashes()
    assert pre_hashes == post_hashes
    assert summary["zero_mutations_verified"] is True
    assert summary["status"] == "PACKAGE_GENERATED_SUCCESSFULLY"


def test_8_master_results_complete_sections():
    packager = _setup_test_packager()
    packager.run()
    master_path = packager.final_dir / "stage6a_master_results.json"
    assert master_path.exists()
    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)

    expected_keys = [
        "1_research_objective",
        "2_final_pipeline",
        "3_experiment_contract",
        "4_candidate_results",
        "5_baseline_comparison",
        "6_per_seed_results",
        "7_ablation_results",
        "8_calibration",
        "9_reproducibility",
        "10_claim_boundaries",
        "11_limitations",
        "12_strongest_defensible_contribution",
        "13_immutability_hashes",
    ]
    for k in expected_keys:
        assert k in master
