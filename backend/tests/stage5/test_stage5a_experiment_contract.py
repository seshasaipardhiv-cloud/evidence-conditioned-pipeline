"""
Unit and regression tests for Stage 5A: Controlled Experimental Execution Contract

Tests:
1. Contract structure and mandatory top-level sections
2. Exact pipeline specification and SHA-256 hash frozen
3. Exact dataset/subset identity and split ratios
4. Exact random seeds [42, 100, 2026]
5. Target definition and target leakage exclusion list
6. Preprocessing sequence and train-only fit scope
7. Model architecture and component implementation mappings
8. Compute budget constraints
9. Baselines to be evaluated
10. Evaluation metrics (primary and secondary)
11. Reproducibility mandates and zero silent fallback
12. Failure and abort triggers
13. Provenance distinction (EVIDENCE_BACKED vs EXPLICITLY_CONFIGURED)
14. Zero model training calls across execution path
15. training_allowed remains false during contract stage
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage5.contract_stage5a import (
    Stage5AExperimentContract,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    TARGET_LEAKAGE_EXCLUSIONS,
    CONFIGURED_SEEDS,
    BASELINES_TO_EVALUATE,
    EVALUATION_METRICS,
    COMPUTE_BUDGET,
    ABORT_CONDITIONS,
    compute_sha256,
)


def _setup_mock_environment(tmpdir):
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    data_config_dir = Path(tmpdir) / "data" / "config"
    data_metadata_dir = Path(tmpdir) / "data" / "metadata" / "hancock"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    data_config_dir.mkdir(parents=True, exist_ok=True)
    data_metadata_dir.mkdir(parents=True, exist_ok=True)

    papers = [
        {"paper_id": f"paper_{i}", "doi": f"10.1000/p_{i}", "pmid": f"10000{i}", "title": f"Paper {i}", "publication_year": 2024}
        for i in range(30)
    ]
    exps = [{"experiment_id": f"exp_{i}", "paper_id": f"paper_{i}"} for i in range(30)]
    mechs = [
        {"mechanism_id": "mech_cross_attention", "canonical_name": "cross-attention", "category": "Attention", "mapping_status": "MAPPED"},
        {"mechanism_id": "mech_clinical_tabular_representation", "canonical_name": "clinical_tabular_representation", "category": "Representation", "mapping_status": "MAPPED"},
    ]

    with open(processed_dir / "papers.jsonl", "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")
    with open(processed_dir / "experiments.jsonl", "w", encoding="utf-8") as f:
        for e in exps:
            f.write(json.dumps(e) + "\n")
    with open(processed_dir / "evidence_claims.jsonl", "w", encoding="utf-8") as f:
        f.write("")
    with open(processed_dir / "mechanisms.jsonl", "w", encoding="utf-8") as f:
        for m in mechs:
            f.write(json.dumps(m) + "\n")

    default_pipe = {
        "specification_version": "3.6",
        "pipeline_hash": EXPECTED_STAGE3_6_PIPELINE_HASH,
        "target_task": "recurrence_classification",
        "primary_metric": "roc_auc",
        "status": "CONFIGURATION_COMPLETE",
    }
    with open(processed_dir / "stage3_6_configured_pipeline.json", "w", encoding="utf-8") as f:
        json.dump(default_pipe, f)

    return Stage5AExperimentContract(
        metadata_dir=str(metadata_dir),
        processed_dir=str(processed_dir),
        data_config_dir=str(data_config_dir),
        data_metadata_dir=str(data_metadata_dir),
    )


def test_1_contract_structure_and_mandatory_sections(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    assert "pipeline_identity" in contract
    assert "dataset_cohort" in contract
    assert "target_isolation_firewall" in contract
    assert "preprocessing_sequence" in contract
    assert "model_architecture" in contract
    assert "compute_budget" in contract
    assert "baselines_to_evaluate" in contract
    assert "evaluation_metrics" in contract
    assert "reproducibility_mandate" in contract
    assert "abort_conditions" in contract


def test_2_exact_pipeline_hash_frozen(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    assert contract["pipeline_identity"]["pipeline_hash"] == EXPECTED_STAGE3_6_PIPELINE_HASH


def test_3_dataset_identity_and_split_ratios(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    cohort = contract["dataset_cohort"]
    assert cohort["modality"] == "clinical_tabular"
    assert cohort["split_ratios"]["train"] == 0.65
    assert cohort["split_ratios"]["validation"] == 0.15
    assert cohort["split_ratios"]["test"] == 0.20


def test_4_exact_random_seeds_frozen(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    assert contract["dataset_cohort"]["random_seeds"] == [42, 100, 2026]


def test_5_target_definition_and_exclusion_list(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    assert contract["dataset_cohort"]["target_variable"] == "recurrence"
    exclusions = contract["target_isolation_firewall"]["excluded_outcome_fields"]
    for field in ["recurrence", "survival_status", "days_to_recurrence", "days_to_last_information"]:
        assert field in exclusions


def test_6_preprocessing_sequence_and_train_only_fit(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    steps = contract["preprocessing_sequence"]
    assert len(steps) == 3
    assert steps[0]["component"] == "missing_value_handling"
    assert steps[0]["fit_scope"] == "TRAIN_ONLY"
    assert steps[1]["component"] == "categorical_encoding"
    assert steps[1]["fit_scope"] == "TRAIN_ONLY"
    assert steps[2]["component"] == "imbalance_handling"
    assert steps[2]["fit_scope"] == "TRAIN_ONLY"


def test_7_model_architecture_and_mappings(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    arch = contract["model_architecture"]
    assert arch["feature_representation"]["mechanism"] == "clinical_tabular_representation"
    assert arch["modality_fusion"]["mechanism"] == "cross_attention"
    assert arch["base_learner"]["mechanism"] == "XGBoost"
    assert arch["loss_function"]["mechanism"] == "binary_logistic"
    assert arch["ensembling"]["mechanism"] == "average_ensembling"


def test_8_compute_budget_constraints(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    budget = contract["compute_budget"]
    assert budget["max_memory_gb"] == 4
    assert budget["max_training_time_minutes"] == 15
    assert budget["device"] == "cpu"


def test_9_baselines_to_evaluate(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    baselines = contract["baselines_to_evaluate"]
    assert len(baselines) == 4
    ids = [b["baseline_id"] for b in baselines]
    assert "baseline_logistic_regression" in ids
    assert "baseline_random_forest" in ids
    assert "baseline_simple_mlp" in ids
    assert "baseline_xgboost_default" in ids


def test_10_evaluation_metrics(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    metrics = contract["evaluation_metrics"]
    assert metrics["primary"] == "roc_auc"
    assert "f1" in metrics["secondary"]
    assert "brier_score" in metrics["secondary"]


def test_11_reproducibility_mandate_and_no_silent_fallback(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, repro, _ = contract_agent.build_contract()
    assert contract["reproducibility_mandate"]["zero_silent_fallback"] is True
    assert repro["test_set_isolation_guarantee"] is not None


def test_12_abort_conditions_defined(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    assert len(contract["abort_conditions"]) == 7


def test_13_provenance_distinction_preserved(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    contract, _, _ = contract_agent.build_contract()
    assert contract["preprocessing_sequence"][0]["classification"] == "EVIDENCE_BACKED"
    assert contract["preprocessing_sequence"][1]["classification"] == "EXPLICITLY_CONFIGURED"
    assert contract["model_architecture"]["base_learner"]["classification"] == "EVIDENCE_BACKED"
    assert contract["model_architecture"]["loss_function"]["classification"] == "EXPLICITLY_CONFIGURED"


def test_14_zero_training_calls_across_execution_path():
    source = inspect.getsource(Stage5AExperimentContract)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_15_training_allowed_remains_false_during_contract_stage(tmpdir):
    contract_agent = _setup_mock_environment(tmpdir)
    summary = contract_agent.run()
    assert summary["training_allowed"] is False
    assert summary["contract_status"] == "CONTRACT_FROZEN"
