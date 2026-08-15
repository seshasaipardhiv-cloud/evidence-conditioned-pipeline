"""
Unit and regression tests for Stage 3.4: Controlled Resolution of Remaining Pipeline Components

Tests:
1. explicit configuration is accepted
2. absent configuration remains blocked
3. no automatic encoding inference
4. no automatic loss inference
5. library defaults are rejected as configuration
6. evidence provenance is preserved
7. explicit configuration is distinguished from evidence
8. incompatible encoding is rejected
9. incompatible loss is rejected
10. target leakage remains blocked
11. preprocessing remains train-only
12. original evidence remains unchanged
13. original Stage 3 artifacts remain unchanged
14. deterministic resolution
15. no model fitting
16. training_allowed remains false unless every required component is independently validated
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage3.component_resolver_stage3_4 import (
    Stage3_4ComponentResolver,
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

    prov_3_3 = {
        "feature_representation": {
            "selected_value": "clinical_tabular_representation",
            "provenance": {"paper_id": "paper_42487970"},
        },
        "modality_fusion": {
            "selected_value": "cross_attention",
            "provenance": {"canonical_name": "cross-attention"},
        },
        "ensembling": {
            "selected_value": "average_ensembling",
            "provenance": {"canonical_name": "average_ensembling"},
        },
        "missing_value_handling": {
            "selected_value": "MissForest / MICE",
            "provenance": {"pmid": "41826845"},
        },
        "base_learner": {
            "selected_value": "XGBoost",
            "provenance": {"pmid": "41775771"},
        },
        "imbalance_handling": {
            "selected_value": "SMOTE",
            "provenance": {"pmid": "41006422"},
        },
        "categorical_encoding": {
            "selected_value": None,
            "provenance": None,
        },
        "loss_function": {
            "selected_value": None,
            "provenance": None,
        },
    }

    with open(metadata_dir / "stage3_3_component_provenance.json", "w", encoding="utf-8") as f:
        json.dump({"components": prov_3_3}, f)
    with open(processed_dir / "stage3_3_final_candidate_pipeline.json", "w", encoding="utf-8") as f:
        json.dump({"status": "BLOCKED_MISSING_COMPONENTS"}, f)

    config_path = Path(tmpdir) / "experiment_config.json"
    if config_dict is not None:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

    return Stage3_4ComponentResolver(
        metadata_dir=str(metadata_dir),
        processed_dir=str(processed_dir),
        config_path=str(config_path) if config_dict is not None else str(Path(tmpdir) / "non_existent.json"),
    )


def test_1_explicit_configuration_is_accepted(tmpdir):
    config = {
        "categorical_encoding": "one_hot",
        "loss_function": "binary_cross_entropy",
    }
    resolver = _setup_mock_environment(tmpdir, config_dict=config)
    pipeline, records = resolver.resolve_components(resolver.audit_configuration())
    assert pipeline["categorical_encoding"] == "one_hot"
    assert pipeline["loss_function"] == "binary_cross_entropy"
    assert records["categorical_encoding"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert records["loss_function"]["classification"] == "EXPLICITLY_CONFIGURED"


def test_2_absent_configuration_remains_blocked(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    pipeline, records = resolver.resolve_components(resolver.audit_configuration())
    assert pipeline["categorical_encoding"] is None
    assert pipeline["loss_function"] is None
    assert records["categorical_encoding"]["execution_status"] == "BLOCKED"
    assert records["loss_function"]["execution_status"] == "BLOCKED"


def test_3_no_automatic_encoding_inference(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    pipeline, _ = resolver.resolve_components(resolver.audit_configuration())
    assert pipeline["categorical_encoding"] is None


def test_4_no_automatic_loss_inference(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    pipeline, _ = resolver.resolve_components(resolver.audit_configuration())
    assert pipeline["base_learner"] == "XGBoost"
    assert pipeline["loss_function"] is None


def test_5_library_defaults_rejected_as_configuration(tmpdir):
    # Without explicit file entries, resolver does not inject library defaults
    resolver = _setup_mock_environment(tmpdir)
    summary = resolver.run()
    assert summary["final_decision"] == "CONFIGURATION_REQUIRED"


def test_6_evidence_provenance_is_preserved(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    _, records = resolver.resolve_components(resolver.audit_configuration())
    assert records["base_learner"]["provenance"]["pmid"] == "41775771"
    assert records["missing_value_handling"]["provenance"]["pmid"] == "41826845"
    assert records["imbalance_handling"]["provenance"]["pmid"] == "41006422"


def test_7_explicit_configuration_is_distinguished_from_evidence(tmpdir):
    config = {"categorical_encoding": "one_hot"}
    resolver = _setup_mock_environment(tmpdir, config_dict=config)
    _, records = resolver.resolve_components(resolver.audit_configuration())
    assert records["categorical_encoding"]["classification"] == "EXPLICITLY_CONFIGURED"
    assert records["categorical_encoding"]["evidence_status"] == "UNSUPPORTED"
    assert records["categorical_encoding"]["configuration_source"] == "explicit_project_configuration"
    assert records["categorical_encoding"]["provenance"]["literature_claim"] is False


def test_8_incompatible_encoding_is_rejected(tmpdir):
    config = {"categorical_encoding": "arbitrary_custom_encoding"}
    resolver = _setup_mock_environment(tmpdir, config_dict=config)
    _, records = resolver.resolve_components(resolver.audit_configuration())
    assert records["categorical_encoding"]["execution_status"] == "BLOCKED"
    assert records["categorical_encoding"]["compatibility_status"] == "INCOMPATIBLE"


def test_9_incompatible_loss_is_rejected(tmpdir):
    config = {"loss_function": "mse_continuous_loss"}
    resolver = _setup_mock_environment(tmpdir, config_dict=config)
    _, records = resolver.resolve_components(resolver.audit_configuration())
    assert records["loss_function"]["execution_status"] == "BLOCKED"
    assert records["loss_function"]["compatibility_status"] == "INCOMPATIBLE"


def test_10_target_leakage_remains_blocked():
    for target in ["recurrence", "survival_status", "days_to_recurrence"]:
        assert target in TARGET_LEAKAGE_COLUMNS


def test_11_preprocessing_remains_train_only(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    summary = resolver.run()
    assert summary["safety_firewalls"]["preprocessing_train_only"] is True


def test_12_original_evidence_remains_unchanged(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(resolver.papers_path)
    exps_before = compute_sha256(resolver.experiments_path)
    resolver.run()
    assert compute_sha256(resolver.papers_path) == papers_before
    assert compute_sha256(resolver.experiments_path) == exps_before


def test_13_original_stage3_artifacts_remain_unchanged(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    mechs_before = compute_sha256(resolver.mechanisms_path)
    resolver.run()
    assert compute_sha256(resolver.mechanisms_path) == mechs_before


def test_14_deterministic_resolution(tmpdir):
    resolver = _setup_mock_environment(tmpdir)
    sum1 = resolver.run()
    sum2 = resolver.run()
    assert sum1["final_decision"] == sum2["final_decision"]
    assert sum1["unresolved_components"] == sum2["unresolved_components"]


def test_15_no_model_fitting():
    source = inspect.getsource(Stage3_4ComponentResolver)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_16_training_allowed_remains_false_unless_independently_validated(tmpdir):
    config = {
        "categorical_encoding": "one_hot",
        "loss_function": "binary_cross_entropy",
    }
    resolver = _setup_mock_environment(tmpdir, config_dict=config)
    summary = resolver.run()
    assert summary["training_allowed"] is False
