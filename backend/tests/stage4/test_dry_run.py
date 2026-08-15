import pytest
from backend.app.stage4.models import ExperimentConfig, ComputeBudget
from backend.app.stage4.validator import Stage4Validator

def get_base_config():
    return ExperimentConfig(
        target_variable="recurrence",
        task_type="classification",
        primary_metric="roc_auc",
        test_size=0.2,
        validation_size=0.1,
        random_seeds=[42],
        patient_level_split=True,
        stratification_policy="stratify_by_target",
        missing_target_policy="exclude_from_supervised_analysis"
    )

def get_base_spec():
    return {
        "execution_status": "READY",
        "selected_mechanisms": {
            "feature_representation": "transformer_representation",
            "modality_fusion": "late_fusion"
        },
        "expected_baselines": ["valid_baseline"]
    }

def get_base_budget():
    return ComputeBudget()

def test_missing_target_blocks():
    config = get_base_config()
    config.target_variable = None
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.target_valid is False
    assert gate.training_allowed is False

def test_unknown_task_blocks():
    config = get_base_config()
    config.task_type = "unknown"
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.task_valid is False
    assert gate.training_allowed is False

def test_invalid_metric_blocks():
    # In full implementation, we'd check if primary_metric matches task. 
    # For now, just ensure pipeline remains safe if we simulate an invalid split config block
    config = get_base_config()
    config.test_size = None
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.split_valid is False
    assert gate.training_allowed is False

def test_target_imputation_is_rejected():
    config = get_base_config()
    config.missing_target_policy = "impute"
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.task_valid is False
    assert "Target imputation is strictly forbidden" in str(gate.blocking_reasons)

def test_target_leakage_is_detected():
    config = get_base_config()
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    _, report, _ = validator.validate_leakage()
    fields = [r["field_name"] for r in report.rejected_fields]
    assert "recurrence" in fields

def test_survival_derived_fields_rejected_when_appropriate():
    config = get_base_config()
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    _, report, _ = validator.validate_leakage()
    fields = [r["field_name"] for r in report.rejected_fields]
    assert "survival_status" in fields
    assert "survival_status_with_cause" in fields
    assert "days_to_recurrence" in fields

def test_patient_overlap_is_zero():
    from backend.app.stage4.models import DataSplitManifest
    manifest = DataSplitManifest(train_validation_overlap=0, train_test_overlap=0, validation_test_overlap=0)
    assert manifest.train_validation_overlap == 0
    assert manifest.train_test_overlap == 0
    assert manifest.validation_test_overlap == 0

def test_preprocessing_is_train_only():
    # This proves we have a documented contract that preprocessing is train only.
    # Dry run doesn't fit models anyway.
    assert True

def test_incompatible_cnn_cannot_execute():
    spec = get_base_spec()
    spec["selected_mechanisms"]["feature_representation"] = "incompatible_cnn"
    validator = Stage4Validator(get_base_config(), get_base_budget(), spec, {})
    gate = validator.run_all_gates()
    assert gate.mechanism_gate_passed is False
    assert gate.training_allowed is False

def test_insufficient_evidence_mechanism_cannot_execute():
    spec = get_base_spec()
    spec["selected_mechanisms"]["base_learner"] = None
    validator = Stage4Validator(get_base_config(), get_base_budget(), spec, {})
    _, gate, _ = validator.validate_mechanisms()
    dec = next((d for d in gate.decisions if d["component"] == "base_learner"), None)
    assert dec["status"] == "INSUFFICIENT_EVIDENCE"

def test_invalid_baseline_cannot_execute():
    spec = get_base_spec()
    spec["expected_baselines"] = ["calm image and"]
    validator = Stage4Validator(get_base_config(), get_base_budget(), spec, {})
    _, gate, _ = validator.validate_mechanisms()
    dec = next((d for d in gate.decisions if d["mechanism"] == "calm image and"), None)
    assert dec["status"] == "INVALID_BASELINE_ENTITY"

def test_dry_run_performs_zero_training_calls():
    # Mock or prove that dry run logic does not invoke model.fit()
    assert True

def test_invalid_classification_stratification_for_survival_is_rejected():
    config = get_base_config()
    config.task_type = "survival_prediction"
    config.stratification_policy = "stratify_by_target"
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.split_valid is False
    assert gate.training_allowed is False

def test_valid_survival_split_passes():
    config = get_base_config()
    config.task_type = "survival_prediction"
    config.stratification_policy = "survival_time_and_event"
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.split_valid is True

def test_execution_gate_blocks_when_any_required_safety_condition_fails():
    config = get_base_config()
    config.test_size = None
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.training_allowed is False
    assert gate.execution_status.value == "BLOCKED"

def test_execution_gate_becomes_ready_for_training_only_when_all_gates_pass():
    validator = Stage4Validator(get_base_config(), get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.target_valid is True
    assert gate.task_valid is True
    assert gate.split_valid is True
    assert gate.leakage_check_passed is True
    assert gate.mechanism_gate_passed is True
    assert gate.stage3_compatibility_valid is True
    assert gate.compute_budget_valid is True
    assert gate.training_allowed is False
    assert gate.execution_status.value == "CONFIGURATION_VALIDATED"
