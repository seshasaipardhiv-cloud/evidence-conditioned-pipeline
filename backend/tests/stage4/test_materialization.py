import json
from pathlib import Path
from backend.app.stage4.materializer import Materializer
import copy
import hashlib

def get_base_spec():
    with open("evidence/processed/stage3_validated_pipeline_specification.json", "r", encoding="utf-8") as f:
        return json.load(f)

def run_materializer(spec_override=None, data_override=None):
    # Setup temp paths
    spec_path = "evidence/processed/test_spec.json"
    data_path = "data/raw/hancock/structured/StructuredData/test_clinical_data.json"
    
    spec = get_base_spec()
    if spec_override:
        spec.update(spec_override)
        
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f)
        
    if data_override is not None:
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data_override, f)
    else:
        # Load real data
        with open("data/raw/hancock/structured/StructuredData/clinical_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
            
    mat = Materializer(
        spec_path,
        "evidence/metadata/stage3_compatibility_audit.json",
        data_path
    )
    
    audit, manifest = mat.audit_materialization()
    return audit, manifest

def test_stage_3_1_is_read_only():
    def get_hash():
        h = hashlib.sha256()
        with open("evidence/processed/stage3_validated_pipeline_specification.json", "rb") as f:
            h.update(f.read())
        return h.hexdigest()
        
    h1 = get_hash()
    run_materializer()
    h2 = get_hash()
    assert h1 == h2

def test_incompatible_mechanisms_cannot_materialize():
    spec = get_base_spec()
    spec["selected_mechanisms"]["feature_representation"] = "incompatible_cnn"
    audit, manifest = run_materializer(spec_override=spec)
    comp = next(c for c in audit["components"] if c["component"] == "feature_representation")
    assert comp["materialization_status"] == "BLOCKED"
    assert comp["Stage3_status"] == "INCOMPATIBLE"

def test_insufficient_evidence_mechanisms_cannot_materialize():
    spec = get_base_spec()
    spec["selected_mechanisms"]["missing_value_handling"] = None
    audit, manifest = run_materializer(spec_override=spec)
    comp = next(c for c in audit["components"] if c["component"] == "missing_value_handling")
    assert comp["materialization_status"] == "BLOCKED"
    assert comp["Stage3_status"] == "INSUFFICIENT_EVIDENCE"

def test_unsupported_implementations_cannot_materialize():
    spec = get_base_spec()
    spec["selected_mechanisms"]["base_learner"] = "quantum_neural_net"
    audit, manifest = run_materializer(spec_override=spec)
    comp = next(c for c in audit["components"] if c["component"] == "base_learner")
    assert comp["materialization_status"] == "BLOCKED"
    assert comp["implementation_available"] is False

def test_target_fields_cannot_enter_components():
    audit, manifest = run_materializer()
    assert audit["target_firewall"]["enforced"] is True
    assert "recurrence" in audit["target_firewall"]["excluded_fields"]
    assert "survival_status" in audit["target_firewall"]["excluded_fields"]

def test_preprocessing_fit_is_train_only():
    audit, manifest = run_materializer()
    assert audit["preprocessing_contract"]["enforced"] is True
    assert audit["preprocessing_contract"]["allowed_fit_partition"] == "train"
    assert "validation" in audit["preprocessing_contract"]["allowed_transform_partitions"]
    assert audit["preprocessing_contract"]["fit_calls_during_setup"] == 0

def test_malformed_baselines_cannot_materialize():
    spec = get_base_spec()
    spec["expected_baselines"] = ["calm image and", "unimodal models across", "valid_baseline"]
    audit, manifest = run_materializer(spec_override=spec)
    b_mat = audit["baseline_materialization"]
    assert b_mat["calm image and"]["materialization_status"] == "BLOCKED"
    assert b_mat["unimodal models across"]["materialization_status"] == "BLOCKED"
    # Even valid baselines are blocked currently because no implementations exist for them directly.
    assert b_mat["valid_baseline"]["materialization_status"] == "BLOCKED"

def test_missing_modality_cannot_enable_cnn():
    spec = get_base_spec()
    spec["selected_mechanisms"]["feature_representation"] = "cnn_representation"
    
    # Override data to explicitly have NO imaging modality
    data = [{"patient_id": "P1", "clinical_var": 1.0}] # No "imaging"
    
    audit, manifest = run_materializer(spec_override=spec, data_override=data)
    comp = next(c for c in audit["components"] if c["component"] == "feature_representation")
    assert comp["dataset_compatibility"] is False
    assert comp["materialization_status"] == "BLOCKED"

def test_no_mechanism_substitution_occurs():
    spec = get_base_spec()
    spec["selected_mechanisms"]["categorical_encoding"] = None
    audit, manifest = run_materializer(spec_override=spec)
    comp = next(c for c in audit["components"] if c["component"] == "categorical_encoding")
    assert comp["selected_mechanism"] is None # It stays None, not substituted

def test_no_model_fit_occurs():
    audit, manifest = run_materializer()
    # Contract asserts 0 fit calls
    assert audit["preprocessing_contract"]["fit_calls_during_setup"] == 0

def test_no_training_occurs():
    audit, manifest = run_materializer()
    assert audit["training_allowed"] is False

def test_final_pipeline_remains_blocked_if_any_required_component_is_unresolved():
    audit, manifest = run_materializer()
    assert audit["pipeline_materializable"] is False
    assert audit["execution_status"] == "CONFIGURATION_VALIDATED" # Stays validated, but NOT READY_FOR_TRAINING
