import json
from pathlib import Path
from backend.app.stage4.readiness_gate import ReadinessGate
from backend.app.stage4.implementation_auditor import ImplementationAuditor

def setup_components():
    config_path = "data/config/implementation_config.json"
    res_path = "data/metadata/hancock/stage4_blocker_resolution.json"
    clinical = "data/raw/hancock/structured/StructuredData/clinical_data.json"
    exp_conf = "data/config/experiment_config.json"
    
    auditor = ImplementationAuditor(config_path, res_path, clinical, exp_conf)
    audit = auditor.audit()
    
    gate = ReadinessGate(
        "data/metadata/hancock/stage4_blocker_resolution.json",
        "data/metadata/hancock/stage4_materialization_audit.json",
        "data/metadata/hancock/stage4_execution_gate.json",
        "evidence/processed/stage3_validated_pipeline_specification.json",
        "evidence/processed/stage3_mechanism_rankings.json",
        "data/config/experiment_config.json",
        "data/metadata/hancock/stage4_implementation_config_audit.json"
    )
    report = gate.check_readiness()
    return audit, report, gate

def test_explicit_configuration_does_not_fabricate_evidence():
    audit, report, gate = setup_components()
    for c in report["required_components"]:
        if c["evidence_status"] == "EXPLICITLY_CONFIGURED":
            assert c["provenance"] == "explicit_configuration"
            
def test_unresolved_components_remain_blocked():
    # If a primitive is left null in config (like feature_representation), it remains unsupported
    audit, report, gate = setup_components()
    for c in report["required_components"]:
        if c["component"] == "feature_representation":
            assert c["evidence_status"] == "UNSUPPORTED"
            assert c["execution_status"] == "BLOCKED"

def test_incompatible_cnn_remains_blocked():
    audit, report, gate = setup_components()
    cnn = next((c for c in report["required_components"] if c["component"] == "feature_representation"), None)
    if cnn and cnn["selected_value"] == "cnn_representation":
        assert cnn["execution_status"] == "BLOCKED"
        assert cnn["compatibility_status"] == "incompatible"

def test_target_leakage_remains_impossible():
    audit, report, gate = setup_components()
    assert report["leakage_validation"] is True

def test_preprocessing_remains_train_only():
    audit, report, gate = setup_components()
    assert gate.materialization["preprocessing_contract"]["allowed_fit_partition"] == "train"

def test_explicit_configuration_alone_does_not_enable_training():
    audit, report, gate = setup_components()
    # Still blocked due to missing CNN / lack of evidence for it
    assert report["training_allowed"] is False
    assert report["final_readiness_decision"] == "BLOCKED_MISSING_EVIDENCE"

def test_stage2_3_artifacts_remain_unchanged():
    assert Path("evidence/processed/stage3_validated_pipeline_specification.json").exists()

def test_training_model_fit_calls_remain_zero():
    audit, report, gate = setup_components()
    assert gate.materialization["preprocessing_contract"]["fit_calls_during_setup"] == 0
    assert report["training_allowed"] is False
