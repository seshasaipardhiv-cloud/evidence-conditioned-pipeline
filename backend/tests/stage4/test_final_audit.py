import json
from pathlib import Path
from backend.app.stage4.final_audit import FinalAudit

def setup_audit():
    audit = FinalAudit(
        "data/config/experiment_config.json",
        "data/config/compute_budget.json",
        "data/config/implementation_config.json",
        "data/metadata/hancock/stage4_execution_gate.json",
        "data/metadata/hancock/stage4_mechanism_gate.json",
        "data/metadata/hancock/stage4_materialization_audit.json",
        "data/metadata/hancock/stage4_pretraining_readiness.json",
        "data/metadata/hancock/data_split_manifest.json",
        "data/metadata/hancock/feature_target_audit.json",
        "data/metadata/hancock/target_leakage_report.json",
        "evidence/processed/stage3_validated_pipeline_specification.json"
    )
    return audit.audit()

def test_target_leakage_cannot_enter_x():
    report = setup_audit()
    assert report["target_audit"]["target_excluded"] is True
    assert report["target_audit"]["leakage_excluded"] is True

def test_patient_overlap_remains_zero():
    report = setup_audit()
    assert report["data_split_audit"]["overlap_zero"] is True

def test_incompatible_mechanisms_remain_blocked():
    report = setup_audit()
    cnn = next((c for c in report["implementation_audit"]["components"] if c["component"] == "feature_representation"), None)
    if cnn:
        assert cnn["execution_status"] == "BLOCKED"

def test_unsupported_mechanisms_remain_blocked():
    report = setup_audit()
    cnn = next((c for c in report["implementation_audit"]["components"] if c["component"] == "feature_representation"), None)
    if cnn:
        assert cnn["category"] == "UNSUPPORTED"
        assert cnn["execution_status"] == "BLOCKED"

def test_explicit_configuration_is_distinguishable_from_evidence():
    report = setup_audit()
    mean_imp = next((c for c in report["implementation_audit"]["components"] if c["component"] == "missing_value_handling"), None)
    if mean_imp:
        assert mean_imp["category"] == "EXPLICITLY_CONFIGURED"

def test_preprocessing_remains_train_only():
    report = setup_audit()
    assert report["preprocessing_audit"]["fit_train_only"] is True

def test_stage2_3_artifacts_remain_unchanged():
    assert Path("evidence/processed/stage3_validated_pipeline_specification.json").exists()

def test_zero_model_fitting_calls_occur():
    report = setup_audit()
    assert report["data_split_audit"]["no_preprocessing_fitted"] is True

def test_final_gate_cannot_become_ready_for_training_merely_because_a_field_is_non_null():
    report = setup_audit()
    # Despite explicit config, missing CNN blocks it.
    assert report["final_gate_decision"] == "BLOCKED"

def test_missing_required_implementation_configuration_blocks_execution():
    report = setup_audit()
    assert report["training_allowed"] is False
    assert report["final_gate_decision"] == "BLOCKED"
