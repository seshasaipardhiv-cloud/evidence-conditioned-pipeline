"""
Unit and regression tests for Stage 3.6: Configured Pipeline Specification & Explicit Configuration Gate

Tests:
1. missing configuration produces NO_GO and BLOCKED
2. valid explicit configuration is accepted
3. invalid categorical encoding is rejected
4. invalid loss function is rejected
5. values are not inferred from learner or data types
6. evidence provenance is strictly distinguished from explicit project configuration
7. target leakage remains blocked
8. preprocessing contract remains train-only
9. original Stage 2 and Stage 3 evidence remains unchanged
10. configuration hash is deterministic
11. no model fitting or training calls
12. training_allowed remains false
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage3.configured_pipeline_stage3_6 import (
    Stage3_6ConfiguredPipelineGate,
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

    config_path = Path(tmpdir) / "experiment_config.json"
    if config_dict is not None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

    return Stage3_6ConfiguredPipelineGate(
        metadata_dir=str(metadata_dir),
        processed_dir=str(processed_dir),
        config_path=str(config_path) if config_dict is not None else str(Path(tmpdir) / "non_existent.json"),
    )


def test_1_missing_configuration_produces_no_go(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    summary = gate.run()
    assert summary["final_decision"] == "NO_GO"
    assert "categorical_encoding" in summary["unresolved_components"]
    assert "loss_function" in summary["unresolved_components"]


def test_2_valid_explicit_configuration_accepted(tmpdir):
    config = {
        "categorical_encoding": {
            "value": "one_hot_encoding",
            "rationale": "Explicitly selected for categorical clinical variables.",
        },
        "loss_function": {
            "value": "binary_logistic",
            "rationale": "Explicitly selected for binary recurrence classification.",
        },
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    summary = gate.run()
    assert summary["final_decision"] == "CONFIGURATION_COMPLETE"
    assert summary["categorical_encoding"]["selected_value"] == "one_hot_encoding"
    assert summary["categorical_encoding"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert summary["loss_function"]["selected_value"] == "binary_logistic"
    assert summary["loss_function"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert len(summary["unresolved_components"]) == 0


def test_3_invalid_categorical_encoding_rejected(tmpdir):
    config = {
        "categorical_encoding": {"value": "invalid_encoding_xyz"},
        "loss_function": {"value": "binary_logistic"},
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    summary = gate.run()
    assert summary["final_decision"] == "NO_GO"
    assert summary["categorical_encoding"]["execution_status"] == "BLOCKED"


def test_4_invalid_loss_function_rejected(tmpdir):
    config = {
        "categorical_encoding": {"value": "one_hot_encoding"},
        "loss_function": {"value": "invalid_loss_xyz"},
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    summary = gate.run()
    assert summary["final_decision"] == "NO_GO"
    assert summary["loss_function"]["execution_status"] == "BLOCKED"


def test_5_no_inference_from_learner_or_data_types(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, _, ledger = gate.build_configured_pipeline()
    assert ledger["base_learner"]["selected_value"] == "XGBoost"
    assert ledger["loss_function"]["selected_value"] is None
    assert ledger["categorical_encoding"]["selected_value"] is None


def test_6_provenance_distinction_preserved(tmpdir):
    config = {
        "categorical_encoding": {"value": "one_hot_encoding"},
        "loss_function": {"value": "binary_logistic"},
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, _, ledger = gate.build_configured_pipeline()
    assert ledger["base_learner"]["classification"] == "EVIDENCE_BACKED"
    assert ledger["categorical_encoding"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert ledger["loss_function"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert ledger["categorical_encoding"]["provenance"]["literature_claim"] is False
    assert ledger["loss_function"]["provenance"]["literature_claim"] is False


def test_7_target_leakage_remains_blocked():
    for target in ["recurrence", "survival_status", "days_to_recurrence", "days_to_last_information"]:
        assert target in TARGET_LEAKAGE_COLUMNS


def test_8_preprocessing_remains_train_only(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    summary = gate.run()
    assert summary["safety_firewalls"]["preprocessing_train_only"] is True


def test_9_original_evidence_remains_unchanged(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(gate.papers_path)
    exps_before = compute_sha256(gate.experiments_path)
    mechs_before = compute_sha256(gate.mechanisms_path)
    gate.run()
    assert compute_sha256(gate.papers_path) == papers_before
    assert compute_sha256(gate.experiments_path) == exps_before
    assert compute_sha256(gate.mechanisms_path) == mechs_before


def test_10_configuration_hash_deterministic(tmpdir):
    config = {
        "categorical_encoding": {"value": "one_hot_encoding"},
        "loss_function": {"value": "binary_logistic"},
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    sum1 = gate.run()
    sum2 = gate.run()
    assert sum1["pipeline_hash"] == sum2["pipeline_hash"]
    assert len(sum1["pipeline_hash"]) == 64


def test_11_no_model_fitting():
    source = inspect.getsource(Stage3_6ConfiguredPipelineGate)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_12_training_allowed_remains_false(tmpdir):
    config = {
        "categorical_encoding": {"value": "one_hot_encoding"},
        "loss_function": {"value": "binary_logistic"},
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    summary = gate.run()
    assert summary["training_allowed"] is False
