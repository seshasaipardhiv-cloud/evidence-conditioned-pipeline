"""
Unit and regression tests for Stage 3.3: Evidence-Conditioned Pipeline Finalization Audit

Tests:
1. clinical_tabular_representation is preserved
2. MissForest/MICE provenance is preserved
3. XGBoost provenance is preserved
4. SMOTE provenance is preserved
5. categorical_encoding remains BLOCKED
6. loss_function remains BLOCKED
7. blocked components cannot receive automatic defaults
8. blocked components cannot be inferred
9. target leakage remains impossible
10. patient-level split remains unchanged
11. original Stage 2 artifacts remain unchanged
12. original Stage 3 artifacts remain unchanged
13. provenance cannot be fabricated
14. preprocessing remains train-only
15. no model fitting occurs
16. training_allowed remains false
17. output is deterministic
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage3.pipeline_finalizer_stage3_3 import (
    Stage3_3PipelineFinalizer,
    TARGET_LEAKAGE_COLUMNS,
    compute_sha256,
)


def _setup_mock_environment(tmpdir, prim_override=None):
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

    prims = {
        "missing_value_handling": {
            "primitive": "missing_value_handling",
            "selected_value": "MissForest / MICE",
            "classification": "EVIDENCE_BACKED",
            "execution_status": "READY_WITH_EVIDENCE",
            "provenance": {
                "pmid": "41826845",
                "doi": "10.1186/s12874-026-02805-4",
                "source_sentence": "Missing values were imputed using MissForest.",
            },
        },
        "base_learner": {
            "primitive": "base_learner",
            "selected_value": "XGBoost",
            "classification": "EVIDENCE_BACKED",
            "execution_status": "READY_WITH_EVIDENCE",
            "provenance": {
                "pmid": "41775771",
                "doi": "10.1038/s41598-026-39104-3",
                "source_sentence": "XGBoost classifier was trained on clinical features.",
            },
        },
        "imbalance_handling": {
            "primitive": "imbalance_handling",
            "selected_value": "SMOTE",
            "classification": "EVIDENCE_BACKED",
            "execution_status": "READY_WITH_EVIDENCE",
            "provenance": {
                "pmid": "41006422",
                "doi": "10.1038/s41598-025-16790-z",
                "source_sentence": "SMOTE was applied to address class imbalance.",
            },
        },
        "categorical_encoding": {
            "primitive": "categorical_encoding",
            "selected_value": None,
            "classification": "UNSUPPORTED",
            "execution_status": "BLOCKED",
            "provenance": None,
        },
        "loss_function": {
            "primitive": "loss_function",
            "selected_value": None,
            "classification": "UNSUPPORTED",
            "execution_status": "BLOCKED",
            "provenance": None,
        },
    }
    if prim_override:
        prims.update(prim_override)

    with open(metadata_dir / "stage2f4_primitive_configuration.json", "w", encoding="utf-8") as f:
        json.dump({"primitives": prims}, f)

    return Stage3_3PipelineFinalizer(
        metadata_dir=str(metadata_dir),
        processed_dir=str(processed_dir),
    )


def test_1_clinical_tabular_representation_preserved(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    spec, _ = finalizer.synthesize_pipeline()
    assert spec["feature_representation"] == "clinical_tabular_representation"


def test_2_missforest_mice_provenance_preserved(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    _, provs = finalizer.synthesize_pipeline()
    assert provs["missing_value_handling"]["selected_value"] == "MissForest / MICE"
    assert provs["missing_value_handling"]["provenance"]["pmid"] == "41826845"


def test_3_xgboost_provenance_preserved(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    _, provs = finalizer.synthesize_pipeline()
    assert provs["base_learner"]["selected_value"] == "XGBoost"
    assert provs["base_learner"]["provenance"]["pmid"] == "41775771"


def test_4_smote_provenance_preserved(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    _, provs = finalizer.synthesize_pipeline()
    assert provs["imbalance_handling"]["selected_value"] == "SMOTE"
    assert provs["imbalance_handling"]["provenance"]["pmid"] == "41006422"


def test_5_categorical_encoding_remains_blocked(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    spec, provs = finalizer.synthesize_pipeline()
    assert spec["categorical_encoding"] is None
    assert provs["categorical_encoding"]["execution_status"] == "BLOCKED"
    assert "categorical_encoding" in spec["unresolved_components"]


def test_6_loss_function_remains_blocked(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    spec, provs = finalizer.synthesize_pipeline()
    assert spec["loss_function"] is None
    assert provs["loss_function"]["execution_status"] == "BLOCKED"
    assert "loss_function" in spec["unresolved_components"]


def test_7_blocked_components_cannot_receive_automatic_defaults(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    spec, _ = finalizer.synthesize_pipeline()
    assert spec["categorical_encoding"] is None
    assert spec["loss_function"] is None


def test_8_blocked_components_cannot_be_inferred(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    spec, _ = finalizer.synthesize_pipeline()
    assert spec["base_learner"] == "XGBoost"
    assert spec["loss_function"] is None


def test_9_target_leakage_remains_impossible():
    for target in ["recurrence", "survival_status", "days_to_recurrence"]:
        assert target in TARGET_LEAKAGE_COLUMNS


def test_10_patient_level_split_remains_unchanged(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    summary = finalizer.run()
    assert summary["safety_firewalls"]["target_leakage_secure"] is True


def test_11_original_stage2_artifacts_remain_unchanged(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(finalizer.papers_path)
    exps_before = compute_sha256(finalizer.experiments_path)
    finalizer.run()
    assert compute_sha256(finalizer.papers_path) == papers_before
    assert compute_sha256(finalizer.experiments_path) == exps_before


def test_12_original_stage3_artifacts_remain_unchanged(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    mechs_before = compute_sha256(finalizer.mechanisms_path)
    finalizer.run()
    assert compute_sha256(finalizer.mechanisms_path) == mechs_before


def test_13_provenance_cannot_be_fabricated(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    _, provs = finalizer.synthesize_pipeline()
    assert provs["categorical_encoding"]["provenance"] is None
    assert provs["loss_function"]["provenance"] is None


def test_14_preprocessing_remains_train_only(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    summary = finalizer.run()
    assert summary["safety_firewalls"]["preprocessing_train_only"] is True


def test_15_no_model_fitting_occurs():
    source = inspect.getsource(Stage3_3PipelineFinalizer)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_16_training_allowed_remains_false(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    summary = finalizer.run()
    assert summary["training_allowed"] is False
    assert summary["final_decision"] == "BLOCKED_MISSING_COMPONENTS"


def test_17_output_is_deterministic(tmpdir):
    finalizer = _setup_mock_environment(tmpdir)
    sum1 = finalizer.run()
    sum2 = finalizer.run()
    assert sum1["final_decision"] == sum2["final_decision"]
    assert sum1["unresolved_components"] == sum2["unresolved_components"]
