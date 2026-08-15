"""
Unit and regression tests for Stage 3.5: Explicit Experiment Decision Gate

Tests:
1. missing configuration is blocked
2. explicit categorical encoding is accepted
3. explicit loss function is accepted
4. values are not inferred
5. XGBoost cannot determine loss automatically
6. data types cannot determine encoding automatically
7. explicit configuration is distinguished from evidence
8. incompatible encoding is rejected
9. incompatible loss is rejected
10. target leakage remains blocked
11. preprocessing remains train-only
12. original evidence remains immutable
13. deterministic configuration
14. no model training
15. training_allowed remains false unless all required components are explicitly resolved and validated
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage3.decision_gate_stage3_5 import (
    Stage3_5DecisionGate,
    TARGET_LEAKAGE_COLUMNS,
    compute_sha256,
)


def _setup_mock_environment(tmpdir, config_dict=None):
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

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

    comp_res_3_4 = {
        "components": {
            "feature_representation": {"selected_value": "clinical_tabular_representation", "execution_status": "READY_WITH_EVIDENCE"},
            "modality_fusion": {"selected_value": "cross_attention", "execution_status": "READY_WITH_EVIDENCE"},
            "ensembling": {"selected_value": "average_ensembling", "execution_status": "READY_WITH_EVIDENCE"},
            "missing_value_handling": {"selected_value": "MissForest / MICE", "execution_status": "READY_WITH_EVIDENCE"},
            "base_learner": {"selected_value": "XGBoost", "execution_status": "READY_WITH_EVIDENCE"},
            "imbalance_handling": {"selected_value": "SMOTE", "execution_status": "READY_WITH_EVIDENCE"},
        }
    }

    with open(metadata_dir / "stage3_4_component_resolution.json", "w", encoding="utf-8") as f:
        json.dump(comp_res_3_4, f)
    with open(processed_dir / "stage3_4_resolved_pipeline.json", "w", encoding="utf-8") as f:
        json.dump({"status": "BLOCKED_MISSING_COMPONENTS"}, f)

    config_path = Path(tmpdir) / "experiment_config.json"
    if config_dict is not None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

    return Stage3_5DecisionGate(
        metadata_dir=str(metadata_dir),
        processed_dir=str(processed_dir),
        config_path=str(config_path) if config_dict is not None else str(Path(tmpdir) / "non_existent.json"),
    )


def test_1_missing_configuration_is_blocked(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    summary = gate.run()
    assert summary["final_decision"] == "CONFIGURATION_REQUIRED"
    assert "categorical_encoding" in summary["unresolved_components"]
    assert "loss_function" in summary["unresolved_components"]


def test_2_explicit_categorical_encoding_is_accepted(tmpdir):
    config = {
        "categorical_encoding": {
            "value": "one_hot_encoding",
            "rationale": "One-hot encoding of discrete clinical categorical variables.",
        }
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, decisions, validations = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["categorical_encoding"]["selected_value"] == "one_hot_encoding"
    assert decisions["categorical_encoding"]["execution_status"] == "READY_WITH_EXPLICIT_CONFIG"
    assert validations["categorical_encoding"]["is_valid"] is True


def test_3_explicit_loss_function_is_accepted(tmpdir):
    config = {
        "loss_function": {
            "value": "binary_logistic",
            "rationale": "Explicitly selected for binary recurrence classification with XGBoost.",
        }
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, decisions, validations = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["loss_function"]["selected_value"] == "binary_logistic"
    assert decisions["loss_function"]["execution_status"] == "READY_WITH_EXPLICIT_CONFIG"
    assert validations["loss_function"]["is_valid"] is True


def test_4_values_are_not_inferred(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, decisions, _ = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["categorical_encoding"]["selected_value"] is None
    assert decisions["loss_function"]["selected_value"] is None


def test_5_xgboost_cannot_determine_loss_automatically(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, decisions, _ = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["loss_function"]["execution_status"] == "BLOCKED"


def test_6_data_types_cannot_determine_encoding_automatically(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, decisions, _ = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["categorical_encoding"]["execution_status"] == "BLOCKED"


def test_7_explicit_configuration_is_distinguished_from_evidence(tmpdir):
    config = {"categorical_encoding": "one_hot_encoding", "loss_function": "binary_logistic"}
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, decisions, _ = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["categorical_encoding"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert decisions["categorical_encoding"]["evidence_status"] == "EXPLICITLY_CONFIGURED"
    assert decisions["loss_function"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert decisions["loss_function"]["evidence_status"] == "EXPLICITLY_CONFIGURED"


def test_8_incompatible_encoding_is_rejected(tmpdir):
    config = {"categorical_encoding": "unrecognized_fancy_encoder"}
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, decisions, validations = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["categorical_encoding"]["execution_status"] == "BLOCKED"
    assert validations["categorical_encoding"]["is_valid"] is False


def test_9_incompatible_loss_is_rejected(tmpdir):
    config = {"loss_function": "continuous_mse"}
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, decisions, validations = gate.evaluate_decisions(gate.load_explicit_decisions())
    assert decisions["loss_function"]["execution_status"] == "BLOCKED"
    assert validations["loss_function"]["is_valid"] is False


def test_10_target_leakage_remains_blocked():
    for target in ["recurrence", "survival_status", "days_to_recurrence"]:
        assert target in TARGET_LEAKAGE_COLUMNS


def test_11_preprocessing_remains_train_only(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    summary = gate.run()
    assert summary["safety_firewalls"]["preprocessing_train_only"] is True


def test_12_original_evidence_remains_immutable(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(gate.papers_path)
    exps_before = compute_sha256(gate.experiments_path)
    mechs_before = compute_sha256(gate.mechanisms_path)
    gate.run()
    assert compute_sha256(gate.papers_path) == papers_before
    assert compute_sha256(gate.experiments_path) == exps_before
    assert compute_sha256(gate.mechanisms_path) == mechs_before


def test_13_deterministic_configuration(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    sum1 = gate.run()
    sum2 = gate.run()
    assert sum1["final_decision"] == sum2["final_decision"]
    assert sum1["unresolved_components"] == sum2["unresolved_components"]


def test_14_no_model_training():
    source = inspect.getsource(Stage3_5DecisionGate)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_15_training_allowed_remains_false_unless_all_required_components_validated(tmpdir):
    config = {
        "categorical_encoding": "one_hot_encoding",
        "loss_function": "binary_logistic",
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    summary = gate.run()
    # In Stage 3.5, training_allowed remains false; downstream Stage 4 authorizes training
    assert summary["training_allowed"] is False
    assert summary["final_decision"] == "CONFIGURATION_VALIDATED"
