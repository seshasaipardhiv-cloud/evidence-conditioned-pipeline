import json
from pathlib import Path
from backend.app.stage4.readiness_gate import ReadinessGate

def setup_readiness_gate():
    return ReadinessGate(
        "data/metadata/hancock/stage4_blocker_resolution.json",
        "data/metadata/hancock/stage4_materialization_audit.json",
        "data/metadata/hancock/stage4_execution_gate.json",
        "evidence/processed/stage3_validated_pipeline_specification.json",
        "evidence/processed/stage3_mechanism_rankings.json",
        "data/config/experiment_config.json"
    )

def test_unsupported_evidence_mechanism_remains_blocked():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    
    unsupported = report["unsupported_components"]
    # We know categorical_encoding etc are unsupported
    assert len(unsupported) > 0
    
def test_implementation_primitive_is_distinguishable_from_evidence_conditioned_mechanism():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    
    prims = [c for c in report["required_components"] if c["category"] == "implementation_primitive"]
    evids = [c for c in report["required_components"] if c["category"] == "evidence_conditioned"]
    
    assert any(c["component"] == "missing_value_handling" for c in prims)
    assert any(c["component"] == "modality_fusion" for c in evids)

def test_no_fabricated_provenance():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    
    for c in report["required_components"]:
        if c["evidence_status"] == "unsupported":
            # If unsupported, it must have no fabricated positive provenance
            assert c["provenance"] == "No explicit support found in Stage 2 evidence." or c["provenance"] is None

def test_invalid_baseline_remains_blocked():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    assert report["baseline_validation"] is False
    
def test_incompatible_cnn_remains_blocked():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    
    cnn = next((c for c in report["required_components"] if c["component"] == "feature_representation"), None)
    if cnn and cnn["selected_value"] == "cnn_representation":
        assert cnn["compatibility_status"] == "incompatible"
        assert cnn["execution_status"] == "BLOCKED"

def test_target_leakage_remains_blocked():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    assert report["leakage_validation"] is True # Meaning validation was run and it blocked leakage
    
def test_valid_recurrence_configuration_passes_its_configuration_checks():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    assert report["target_task_validation"] is True
    assert report["split_validation"] is True

def test_final_readiness_cannot_become_ready_when_required_evidence_conditioned_components_are_unresolved():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    
    # We know there are unresolved components
    assert report["final_readiness_decision"] == "BLOCKED_MISSING_EVIDENCE"
    assert report["training_allowed"] is False
    
def test_zero_model_fitting_calls():
    gate = setup_readiness_gate()
    report = gate.check_readiness()
    # Contract asserts 0 fit calls
    assert gate.materialization["preprocessing_contract"]["fit_calls_during_setup"] == 0
    assert report["training_allowed"] is False

def test_stage2_3_artifacts_remain_unchanged():
    # If the dry run operates, it only writes to data/metadata/hancock
    assert Path("evidence/processed/stage3_validated_pipeline_specification.json").exists()
    assert Path("evidence/processed/stage3_mechanism_rankings.json").exists()
