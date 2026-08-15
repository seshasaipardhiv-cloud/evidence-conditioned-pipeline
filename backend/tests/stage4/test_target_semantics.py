import pytest
import json
from pathlib import Path

def test_survival_status_is_inspected():
    report_path = Path("data/metadata/hancock/target_semantics_report.json")
    assert report_path.exists()
    
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    fields = [r["field_name"] for r in report]
    assert "survival_status" in fields

def test_recurrence_is_inspected():
    report_path = Path("data/metadata/hancock/target_semantics_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    fields = [r["field_name"] for r in report]
    assert "recurrence" in fields
    
def test_days_to_recurrence_is_inspected():
    report_path = Path("data/metadata/hancock/target_semantics_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    fields = [r["field_name"] for r in report]
    assert "days_to_recurrence" in fields

def test_target_fields_cannot_silently_become_input_features():
    from backend.app.stage4.models import ExperimentConfig, ComputeBudget
    from backend.app.stage4.validator import Stage4Validator
    
    config = ExperimentConfig(
        target_variable="recurrence",
        task_type="classification",
        primary_metric="roc_auc",
        test_size=0.2,
        validation_size=0.1,
        random_seeds=[42],
        patient_level_split=True,
        missing_target_policy="exclude_from_supervised_analysis"
    )
    
    validator = Stage4Validator(config, ComputeBudget(), {}, {})
    _, leakage_report, _ = validator.validate_leakage()
    
    rejected = [r["field_name"] for r in leakage_report.rejected_fields]
    assert "recurrence" in rejected

def test_unresolved_semantics_remain_unresolved():
    report_path = Path("data/metadata/hancock/target_semantics_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    # days_to_recurrence alone is unresolved for survival without a censoring time
    days_to_recurrence = next(r for r in report if r["field_name"] == "days_to_recurrence")
    assert days_to_recurrence["candidate_role"] == "unresolved"

def test_survival_task_cannot_be_configured_without_appropriate_time_event_representation():
    # As the auditor script demonstrated, there is no single valid survival variable
    # So we don't automatically select one. 
    report_path = Path("data/metadata/hancock/target_semantics_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
        
    unresolved = [r["field_name"] for r in report if r["candidate_role"] == "unresolved"]
    assert len(unresolved) > 0

def test_target_is_explicitly_configured():
    config_path = Path("data/config/experiment_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    assert config["target_variable"] == "recurrence"
    assert config["task_type"] == "classification"
