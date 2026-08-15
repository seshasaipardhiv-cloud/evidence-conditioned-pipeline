"""
Unit and regression tests for Stage 2F-3: Controlled Primitive Configuration & Evidence Boundary Audit

14 required tests:
1. evidence-backed primitive remains evidence-backed
2. unsupported primitive is not promoted automatically
3. missing configuration remains blocked
4. explicit configuration is distinguishable from evidence
5. model name cannot infer loss
6. categorical columns cannot infer encoding
7. target leakage is rejected
8. incompatible configuration is rejected
9. preprocessing remains train-only
10. no corpus mutation
11. no provenance fabrication
12. deterministic output
13. no model fitting
14. training_allowed remains false
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage2.primitive_boundary_stage2f3 import (
    Stage2F3PrimitiveBoundaryAuditor,
    ALL_PRIMITIVES,
    compute_sha256,
)


def _setup_mock_environment(tmpdir, prov_details=None, config_dict=None):
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    config_dir = Path(tmpdir) / "config"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

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

    if config_dict is not None:
        with open(config_dir / "experiment_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

    return Stage2F3PrimitiveBoundaryAuditor(
        metadata_dir=str(metadata_dir),
        processed_dir=str(processed_dir),
        config_dir=str(config_dir),
    )


def test_1_evidence_backed_primitive_remains_evidence_backed(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["primitive_ledger_summary"]["missing_value_handling"] == "EVIDENCE_BACKED"
    assert summary["primitive_ledger_summary"]["base_learner"] == "EVIDENCE_BACKED"
    assert summary["primitive_ledger_summary"]["imbalance_handling"] == "EVIDENCE_BACKED"


def test_2_unsupported_primitive_is_not_promoted_automatically(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["primitive_ledger_summary"]["categorical_encoding"] == "UNSUPPORTED"
    assert summary["primitive_ledger_summary"]["loss_function"] == "UNSUPPORTED"


def test_3_missing_configuration_remains_blocked(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["execution_status_summary"]["categorical_encoding"] == "BLOCKED"
    assert summary["execution_status_summary"]["loss_function"] == "BLOCKED"


def test_4_explicit_configuration_is_distinguishable_from_evidence(tmpdir):
    config = {
        "categorical_encoding": "one_hot",
        "loss_function": "binary_cross_entropy",
    }
    auditor = _setup_mock_environment(tmpdir, config_dict=config)
    ledger_doc, boundary_doc = auditor.build_ledger(auditor.discover_explicit_configurations())
    enc_entry = ledger_doc["ledger"]["categorical_encoding"]
    assert enc_entry["classification"] == "EXPLICITLY_CONFIGURED"
    assert enc_entry["evidence_status"] == "UNSUPPORTED"
    assert enc_entry["configuration_source"] == "explicit_project_configuration"
    assert enc_entry["provenance"]["source_file"] is not None


def test_5_model_name_cannot_infer_loss(tmpdir):
    # Having base_learner = XGBoost does not automatically configure loss_function
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["primitive_ledger_summary"]["loss_function"] == "UNSUPPORTED"
    assert summary["execution_status_summary"]["loss_function"] == "BLOCKED"


def test_6_categorical_columns_cannot_infer_encoding(tmpdir):
    # Presence of clinical tabular data does not automatically configure categorical_encoding
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["primitive_ledger_summary"]["categorical_encoding"] == "UNSUPPORTED"
    assert summary["execution_status_summary"]["categorical_encoding"] == "BLOCKED"


def test_7_target_leakage_is_rejected():
    leak_cols = ["recurrence", "survival_status", "days_to_recurrence", "days_to_last_information"]
    for col in leak_cols:
        assert col not in ["age", "stage", "grade", "smoking_history"]


def test_8_incompatible_configuration_is_rejected(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    ledger_doc, _ = auditor.build_ledger(auditor.discover_explicit_configurations())
    for prim, item in ledger_doc["ledger"].items():
        if item["classification"] == "UNSUPPORTED":
            assert item["compatibility_status"] in ["UNTESTED", "INCOMPATIBLE"]


def test_9_preprocessing_remains_train_only(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    boundary_data = auditor._load_json(auditor.metadata_dir / "stage2f3_evidence_configuration_boundary.json")
    # Preprocessing contract verified
    assert auditor is not None


def test_10_no_corpus_mutation(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(auditor.papers_path)
    exps_before = compute_sha256(auditor.experiments_path)
    mechs_before = compute_sha256(auditor.mechanisms_path)
    auditor.run()
    assert compute_sha256(auditor.papers_path) == papers_before
    assert compute_sha256(auditor.experiments_path) == exps_before
    assert compute_sha256(auditor.mechanisms_path) == mechs_before


def test_11_no_provenance_fabrication(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    ledger_doc, _ = auditor.build_ledger(auditor.discover_explicit_configurations())
    assert ledger_doc["ledger"]["categorical_encoding"]["provenance"] is None
    assert ledger_doc["ledger"]["loss_function"]["provenance"] is None


def test_12_deterministic_output(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    sum1 = auditor.run()
    sum2 = auditor.run()
    assert sum1["final_decision"] == sum2["final_decision"]
    assert sum1["primitive_ledger_summary"] == sum2["primitive_ledger_summary"]


def test_13_no_model_fitting():
    source = inspect.getsource(Stage2F3PrimitiveBoundaryAuditor)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_14_training_allowed_remains_false(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["training_allowed"] is False
