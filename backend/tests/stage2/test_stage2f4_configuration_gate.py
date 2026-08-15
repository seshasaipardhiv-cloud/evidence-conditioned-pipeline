"""
Unit and regression tests for Stage 2F-4: Explicit Primitive Configuration Gate

Tests:
1. missing configuration remains blocked
2. explicit configuration is accepted
3. evidence-backed values remain evidence-backed
4. explicit configuration is not mislabeled as literature evidence
5. loss cannot be inferred from learner
6. encoding cannot be inferred from data type
7. incompatible values are rejected
8. leakage remains blocked
9. no training occurs
10. training_allowed remains false unless all required gates are satisfied
11. preprocessing remains train-only
12. production corpus remains unchanged
13. deterministic output
14. no provenance fabrication
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage2.configuration_gate_stage2f4 import (
    Stage2F4ConfigurationGate,
    ALL_PRIMITIVES,
    compute_sha256,
)


def _setup_mock_environment(tmpdir, prov_details=None, config_dict=None):
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

    if prov_details is None:
        prov_details = [
            {
                "primitive": "missing_value_handling",
                "pmid": "41826845",
                "doi": "10.1186/s12874-026-02805-4",
                "title": "Imputation study",
                "classification": "EXPLICIT_SUPPORTED",
                "source_sentences": ["Missing values were imputed using MissForest."],
                "provenance_complete": True,
                "full_text_status": "accessible",
            },
            {
                "primitive": "base_learner",
                "pmid": "41775771",
                "doi": "10.1038/s41598-026-39104-3",
                "title": "XGBoost study",
                "classification": "EXPLICIT_SUPPORTED",
                "source_sentences": ["XGBoost classifier was trained on clinical features."],
                "provenance_complete": True,
                "full_text_status": "accessible",
            },
            {
                "primitive": "imbalance_handling",
                "pmid": "41006422",
                "doi": "10.1038/s41598-025-16790-z",
                "title": "SMOTE study",
                "classification": "EXPLICIT_SUPPORTED",
                "source_sentences": ["SMOTE was applied to address class imbalance."],
                "provenance_complete": True,
                "full_text_status": "accessible",
            },
        ]

    with open(metadata_dir / "stage2f1_provenance_audit.json", "w", encoding="utf-8") as f:
        json.dump({"details": prov_details}, f)

    config_path = Path(tmpdir) / "experiment_config.json"
    if config_dict is not None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

    return Stage2F4ConfigurationGate(
        metadata_dir=str(metadata_dir),
        processed_dir=str(processed_dir),
        config_path=str(config_path) if config_dict is not None else str(Path(tmpdir) / "non_existent.json"),
    )


def test_1_missing_configuration_remains_blocked(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    summary = gate.run()
    assert summary["primitive_resolutions"]["categorical_encoding"] == "BLOCKED"
    assert summary["primitive_resolutions"]["loss_function"] == "BLOCKED"
    assert summary["overall_gate_status"] == "GATE_BLOCKED"


def test_2_explicit_configuration_is_accepted(tmpdir):
    config = {
        "categorical_encoding": "one_hot",
        "loss_function": "binary_cross_entropy",
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    summary = gate.run()
    assert summary["primitive_resolutions"]["categorical_encoding"] == "READY_WITH_EXPLICIT_CONFIG"
    assert summary["primitive_resolutions"]["loss_function"] == "READY_WITH_EXPLICIT_CONFIG"
    assert summary["overall_gate_status"] == "GATE_OPEN"


def test_3_evidence_backed_values_remain_evidence_backed(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, configs = gate.evaluate_gate()
    assert configs["missing_value_handling"]["classification"] == "EVIDENCE_BACKED"
    assert configs["base_learner"]["classification"] == "EVIDENCE_BACKED"
    assert configs["imbalance_handling"]["classification"] == "EVIDENCE_BACKED"


def test_4_explicit_configuration_not_mislabeled_as_literature_evidence(tmpdir):
    config = {"categorical_encoding": "one_hot"}
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, configs = gate.evaluate_gate()
    assert configs["categorical_encoding"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert configs["categorical_encoding"]["evidence_status"] == "UNSUPPORTED"
    assert configs["categorical_encoding"]["configuration_source"] == "explicit_project_configuration"
    assert configs["categorical_encoding"]["provenance"]["literature_claim"] is False


def test_5_loss_cannot_be_inferred_from_learner(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, configs = gate.evaluate_gate()
    assert configs["base_learner"]["selected_value"] == "XGBoost"
    assert configs["loss_function"]["selected_value"] is None
    assert configs["loss_function"]["execution_status"] == "BLOCKED"


def test_6_encoding_cannot_be_inferred_from_data_type(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, configs = gate.evaluate_gate()
    assert configs["categorical_encoding"]["selected_value"] is None
    assert configs["categorical_encoding"]["execution_status"] == "BLOCKED"


def test_7_incompatible_values_are_rejected(tmpdir):
    config = {
        "categorical_encoding": "invalid_magic_encoding",
        "loss_function": "unsupported_loss_xyz",
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    _, configs = gate.evaluate_gate()
    assert configs["categorical_encoding"]["execution_status"] == "BLOCKED"
    assert configs["categorical_encoding"]["compatibility_status"] == "INCOMPATIBLE"
    assert configs["loss_function"]["execution_status"] == "BLOCKED"
    assert configs["loss_function"]["compatibility_status"] == "INCOMPATIBLE"


def test_8_leakage_remains_blocked():
    from backend.app.stage2.configuration_gate_stage2f4 import TARGET_LEAKAGE_COLUMNS
    assert "recurrence" in TARGET_LEAKAGE_COLUMNS
    assert "survival_status" in TARGET_LEAKAGE_COLUMNS
    assert "days_to_recurrence" in TARGET_LEAKAGE_COLUMNS


def test_9_no_training_occurs():
    source = inspect.getsource(Stage2F4ConfigurationGate)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_10_training_allowed_remains_false_unless_all_required_gates_satisfied(tmpdir):
    # Even if all primitive configurations pass, training_allowed in Stage 2F-4 remains false
    config = {
        "categorical_encoding": "one_hot",
        "loss_function": "binary_cross_entropy",
    }
    gate = _setup_mock_environment(tmpdir, config_dict=config)
    summary = gate.run()
    assert summary["training_allowed"] is False


def test_11_preprocessing_remains_train_only(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    gate_audit, _ = gate.evaluate_gate()
    assert gate_audit["safety_firewalls"]["preprocessing_train_only"] is True


def test_12_production_corpus_remains_unchanged(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(gate.papers_path)
    exps_before = compute_sha256(gate.experiments_path)
    mechs_before = compute_sha256(gate.mechanisms_path)
    gate.run()
    assert compute_sha256(gate.papers_path) == papers_before
    assert compute_sha256(gate.experiments_path) == exps_before
    assert compute_sha256(gate.mechanisms_path) == mechs_before


def test_13_deterministic_output(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    sum1 = gate.run()
    sum2 = gate.run()
    assert sum1["final_decision"] == sum2["final_decision"]
    assert sum1["primitive_resolutions"] == sum2["primitive_resolutions"]


def test_14_no_provenance_fabrication(tmpdir):
    gate = _setup_mock_environment(tmpdir)
    _, configs = gate.evaluate_gate()
    assert configs["categorical_encoding"]["provenance"] is None
    assert configs["loss_function"]["provenance"] is None
