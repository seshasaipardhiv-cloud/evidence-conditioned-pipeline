import pytest
from backend.app.stage4.models import ExperimentConfig, ComputeBudget
from backend.app.stage4.validator import Stage4Validator

def get_base_config():
    return ExperimentConfig(
        target_variable="recurrence",
        task_type="classification",
        primary_metric="roc_auc",
        secondary_metrics=["f1", "accuracy", "precision", "recall"],
        test_size=0.2,
        validation_size=0.15,
        random_seeds=[42, 100, 2026],
        patient_level_split=True,
        stratification_policy="stratified",
        missing_target_policy="exclude_from_supervised_analysis"
    )

def get_base_spec():
    return {
        "execution_status": "READY",
        "selected_mechanisms": {
            "feature_representation": "cnn_representation",
            "modality_fusion": "late_fusion"
        },
        "expected_baselines": ["single-modality PET"]
    }

def get_base_budget():
    return ComputeBudget()

def test_recurrence_is_accepted_as_explicitly_configured_target():
    config = get_base_config()
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.target_valid is True

def test_classification_is_accepted():
    config = get_base_config()
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.task_valid is True

def test_roc_auc_is_accepted():
    config = get_base_config()
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    valid, msg = validator.validate_metrics()
    assert valid is True

def test_missing_target_policy_is_correct():
    config = get_base_config()
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.task_valid is True
    
    config.missing_target_policy = "impute"
    validator2 = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    assert validator2.run_all_gates().task_valid is False

def test_recurrence_cannot_appear_in_feature_columns():
    validator = Stage4Validator(get_base_config(), get_base_budget(), get_base_spec(), {})
    _, report, _ = validator.validate_leakage()
    rejected = [r["field_name"] for r in report.rejected_fields]
    assert "recurrence" in rejected

def test_survival_status_cannot_appear_in_feature_columns():
    validator = Stage4Validator(get_base_config(), get_base_budget(), get_base_spec(), {})
    _, report, _ = validator.validate_leakage()
    rejected = [r["field_name"] for r in report.rejected_fields]
    assert "survival_status" in rejected

def test_survival_status_with_cause_cannot_appear_in_feature_columns():
    validator = Stage4Validator(get_base_config(), get_base_budget(), get_base_spec(), {})
    _, report, _ = validator.validate_leakage()
    rejected = [r["field_name"] for r in report.rejected_fields]
    assert "survival_status_with_cause" in rejected

def test_days_to_recurrence_cannot_appear_in_feature_columns():
    validator = Stage4Validator(get_base_config(), get_base_budget(), get_base_spec(), {})
    _, report, _ = validator.validate_leakage()
    rejected = [r["field_name"] for r in report.rejected_fields]
    assert "days_to_recurrence" in rejected

def test_post_outcome_fields_cannot_appear_in_features():
    validator = Stage4Validator(get_base_config(), get_base_budget(), get_base_spec(), {})
    _, report, _ = validator.validate_leakage()
    rejected = [r["field_name"] for r in report.rejected_fields]
    assert "days_to_progress_1" in rejected
    assert "days_to_metastasis_1" in rejected

def test_invalid_target_task_combinations_are_rejected():
    config = get_base_config()
    config.target_variable = "recurrence"
    config.task_type = "survival_prediction"
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.task_valid is False

def test_invalid_classification_metrics_are_rejected():
    config = get_base_config()
    config.primary_metric = "c_index" # Not valid for classification
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.task_valid is False

def test_invalid_split_sizes_are_rejected():
    config = get_base_config()
    config.test_size = 0.6
    config.validation_size = 0.5
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.split_valid is False

def test_non_patient_level_splitting_is_rejected():
    config = get_base_config()
    config.patient_level_split = False
    validator = Stage4Validator(config, get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.split_valid is False

def test_stage_3_1_incompatible_mechanisms_remain_blocked():
    spec = get_base_spec()
    spec["selected_mechanisms"]["feature_representation"] = "incompatible_cnn"
    validator = Stage4Validator(get_base_config(), get_base_budget(), spec, {})
    gate = validator.run_all_gates()
    assert gate.mechanism_gate_passed is False

def test_no_model_training_call_occurs():
    # Proof by invariant that training_allowed is strictly false in 4B-1 execution gate
    validator = Stage4Validator(get_base_config(), get_base_budget(), get_base_spec(), {})
    gate = validator.run_all_gates()
    assert gate.training_allowed is False
    assert gate.execution_status.value == "CONFIGURATION_VALIDATED"
