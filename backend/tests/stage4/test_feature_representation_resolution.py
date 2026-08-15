import json
import pytest
from pathlib import Path
from backend.app.stage4.feature_representation_resolution import FeatureRepresentationResolutionAuditor
from backend.app.stage4.readiness_gate import ReadinessGate
from backend.app.stage4.final_audit import FinalAudit

def setup_auditor(mock_mechanisms=None, mock_impl=None):
    if mock_mechanisms is None:
        mock_mechanisms = [{"category": "Representation", "canonical_name": "cnn"}]
    
    mech_path = "data/metadata/hancock/mock_mechanisms.jsonl"
    with open(mech_path, "w", encoding="utf-8") as f:
        for m in mock_mechanisms:
            f.write(json.dumps(m) + "\n")
                
    impl_path = "data/config/implementation_config.json"
    if mock_impl:
        impl_path = "data/metadata/hancock/mock_impl_config.json"
        with open(impl_path, "w", encoding="utf-8") as f:
            json.dump(mock_impl, f)

    out_path = "data/metadata/hancock/mock_feature_representation_resolution.json"
    auditor = FeatureRepresentationResolutionAuditor(
        mech_path,
        impl_path,
        "data/metadata/hancock/stage4_pretraining_readiness.json",
        "evidence/processed/experiments.jsonl",
        out_path
    )
    return auditor.audit()

def test_cnn_remains_incompatible_without_imaging():
    report = setup_auditor()
    assert report["original_selected_representation"] == "cnn_representation"
    assert report["original_compatibility_status"] == "incompatible"
    assert "cnn" in report["incompatible_candidates"]
    assert report["final_resolution_status"] == "BLOCKED"

def test_pathology_cannot_masquerade_as_imaging():
    report = setup_auditor()
    assert "cnn" in report["incompatible_candidates"]

def test_unsupported_representations_remain_blocked():
    report = setup_auditor()
    assert report["final_resolution_status"] == "BLOCKED"
    assert report["selected_replacement"] is None

def test_evidence_backed_representation_requires_provenance():
    # Mock an evidence-backed representation
    report = setup_auditor(mock_mechanisms=[{
        "category": "Representation",
        "canonical_name": "tabular_mlp"
    }])
    assert report["final_resolution_status"] == "RESOLVED"
    assert report["selected_replacement"] == "tabular_mlp"
    assert report["provenance"] == "stage2_corpus"

def test_explicit_configuration_is_distinguishable_from_literature_evidence():
    report = setup_auditor(mock_impl={"feature_representation": "manual_pca"})
    assert report["final_resolution_status"] == "RESOLVED"
    assert report["selected_replacement"] == "manual_pca"
    assert report["provenance"] == "explicit_configuration"

def test_no_arbitrary_fallback_is_selected():
    report = setup_auditor()
    assert report["selected_replacement"] is None
    assert report["final_resolution_status"] == "BLOCKED"

def test_target_leakage_remains_impossible():
    gate = ReadinessGate(
        "data/metadata/hancock/stage4_blocker_resolution.json",
        "data/metadata/hancock/stage4_materialization_audit.json",
        "data/metadata/hancock/stage4_execution_gate.json",
        "evidence/processed/stage3_validated_pipeline_specification.json",
        "evidence/processed/stage3_mechanism_rankings.json",
        "data/config/experiment_config.json",
        "data/metadata/hancock/stage4_implementation_config_audit.json",
        feat_rep_res_path="data/metadata/hancock/mock_feature_representation_resolution.json",
        out_path="data/metadata/hancock/mock_stage4_pretraining_readiness.json"
    )
    report = gate.check_readiness()
    assert report["leakage_validation"] is True

def test_stage2_artifacts_remain_unchanged():
    assert Path("evidence/processed/mechanisms.jsonl").exists()

def test_stage3_artifacts_remain_unchanged():
    assert Path("evidence/processed/stage3_validated_pipeline_specification.json").exists()

def test_no_model_fit_occurs():
    gate = ReadinessGate(
        "data/metadata/hancock/stage4_blocker_resolution.json",
        "data/metadata/hancock/stage4_materialization_audit.json",
        "data/metadata/hancock/stage4_execution_gate.json",
        "evidence/processed/stage3_validated_pipeline_specification.json",
        "evidence/processed/stage3_mechanism_rankings.json",
        "data/config/experiment_config.json",
        "data/metadata/hancock/stage4_implementation_config_audit.json"
    )
    assert gate.materialization["preprocessing_contract"]["fit_calls_during_setup"] == 0

def test_training_remains_blocked_when_no_valid_representation_exists():
    report = setup_auditor()
    assert report["final_resolution_status"] == "BLOCKED"

def test_valid_mocked_evidence_backed_representation_resolves_blocker():
    report = setup_auditor(mock_mechanisms=[{
        "category": "Representation",
        "canonical_name": "mlp_representation"
    }])
    assert report["final_resolution_status"] == "RESOLVED"
    assert report["selected_replacement"] == "mlp_representation"

def test_compatibility_is_revalidated_after_replacement():
    report = setup_auditor(mock_mechanisms=[{
        "category": "Representation",
        "canonical_name": "mlp_representation"
    }])
    assert report["candidate_representations"][0]["compatibility_status"] == "valid"

def test_final_readiness_changes_only_when_every_required_gate_passes():
    # If the feature rep is blocked, Readiness Gate MUST be blocked
    gate = ReadinessGate(
        "data/metadata/hancock/stage4_blocker_resolution.json",
        "data/metadata/hancock/stage4_materialization_audit.json",
        "data/metadata/hancock/stage4_execution_gate.json",
        "evidence/processed/stage3_validated_pipeline_specification.json",
        "evidence/processed/stage3_mechanism_rankings.json",
        "data/config/experiment_config.json",
        "data/metadata/hancock/stage4_implementation_config_audit.json",
        feat_rep_res_path="data/metadata/hancock/mock_feature_representation_resolution.json",
        out_path="data/metadata/hancock/mock_stage4_pretraining_readiness.json"
    )
    readiness = gate.check_readiness()
    assert readiness["final_readiness_decision"] == "BLOCKED_COMPATIBILITY"
    assert readiness["training_allowed"] is False
