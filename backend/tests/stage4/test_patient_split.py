import json
from pathlib import Path
from backend.app.stage4.splitter import PatientSplitter
from backend.app.stage4.feature_auditor import FeatureAuditor

config_path = "data/config/experiment_config.json"
clinical_path = "data/raw/hancock/structured/StructuredData/clinical_data.json"

def get_manifest():
    splitter = PatientSplitter(config_path, clinical_path)
    return splitter.run_splits()

def get_audit():
    auditor = FeatureAuditor(config_path, clinical_path)
    return auditor.audit()

def test_all_three_seeds_processed():
    manifest = get_manifest()
    seeds = [s["seed"] for s in manifest["splits"]]
    assert set(seeds) == {42, 100, 2026}

def test_patient_overlap_is_zero():
    manifest = get_manifest()
    for s in manifest["splits"]:
        assert s["overlap_counts"]["train_validation"] == 0
        assert s["overlap_counts"]["train_test"] == 0
        assert s["overlap_counts"]["validation_test"] == 0

def test_split_is_deterministic():
    manifest1 = get_manifest()
    manifest2 = get_manifest()
    
    for s1, s2 in zip(manifest1["splits"], manifest2["splits"]):
        assert s1["train_patient_hash"] == s2["train_patient_hash"]
        assert s1["validation_patient_hash"] == s2["validation_patient_hash"]
        assert s1["test_patient_hash"] == s2["test_patient_hash"]

def test_recurrence_is_stratified():
    manifest = get_manifest()
    for s in manifest["splits"]:
        tr_rate = s["target_distribution"]["train"]["recurrence_rate"]
        te_rate = s["target_distribution"]["test"]["recurrence_rate"]
        # They should be roughly similar (stratified)
        assert abs(tr_rate - te_rate) < 0.1

def test_missing_target_policy_is_enforced():
    manifest = get_manifest()
    for s in manifest["splits"]:
        assert s["target_distribution"]["train"]["missing"] == 0
        assert s["target_distribution"]["validation"]["missing"] == 0
        assert s["target_distribution"]["test"]["missing"] == 0

def test_recurrence_is_not_in_x():
    audit = get_audit()
    assert "recurrence" in audit["excluded_fields"]
    assert audit["target_not_in_features"] is True

def test_survival_status_is_not_in_x():
    audit = get_audit()
    assert "survival_status" in audit["excluded_fields"]
    assert audit["outcome_fields_not_in_features"] is True

def test_survival_status_with_cause_is_not_in_x():
    audit = get_audit()
    assert "survival_status_with_cause" in audit["excluded_fields"]

def test_days_to_recurrence_is_not_in_x():
    audit = get_audit()
    assert "days_to_recurrence" in audit["excluded_fields"]
    assert audit["post_outcome_fields_not_in_features"] is True

def test_days_to_last_information_is_not_in_x():
    audit = get_audit()
    assert "days_to_last_information" in audit["excluded_fields"]

def test_progress_fields_are_excluded():
    audit = get_audit()
    assert "days_to_progress_1" in audit["excluded_fields"]
    assert "days_to_progress_2" in audit["excluded_fields"]

def test_metastasis_fields_are_excluded():
    audit = get_audit()
    assert "days_to_metastasis_1" in audit["excluded_fields"]

def test_no_preprocessing_fit_occurs():
    audit = get_audit()
    assert audit["preprocessing_fit_calls"] == 0

def test_stage_3_1_decisions_are_unchanged():
    with open("evidence/processed/stage3_validated_pipeline_specification.json", "r", encoding="utf-8") as f:
        spec = json.load(f)
    assert spec["selected_mechanisms"].get("missing_value_handling") is None
    assert spec["selected_mechanisms"].get("categorical_encoding") is None
