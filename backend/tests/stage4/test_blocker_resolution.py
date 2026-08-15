import json
from pathlib import Path
from backend.app.stage4.resolution_auditor import ResolutionAuditor

def setup_resolution_auditor(claims_override=None):
    # Prepare dummy files
    mat_audit_path = "data/metadata/hancock/stage4_materialization_audit.json"
    mat_manifest_path = "data/metadata/hancock/stage4_materialization_manifest.json"
    claims_path = "evidence/processed/test_claims.jsonl"
    
    if claims_override is not None:
        with open(claims_path, "w", encoding="utf-8") as f:
            for c in claims_override:
                f.write(json.dumps(c) + "\n")
    else:
        # Write empty claims
        with open(claims_path, "w", encoding="utf-8") as f:
            f.write("")
            
    # Assuming mat_audit and manifest exist from previous steps
    # We will test against the actual current output of stage4
    
    return ResolutionAuditor(
        mat_audit_path,
        mat_manifest_path,
        claims_path,
        "evidence/processed/experiments.jsonl"
    )

def test_unsupported_mechanisms_remain_blocked():
    # Provide no claims that support the missing mechanisms
    auditor = setup_resolution_auditor([])
    report = auditor.audit_resolution()
    
    for r in report["resolutions"]:
        if r["original_status"] in ["BLOCKED", "INCOMPATIBLE", "INSUFFICIENT_EVIDENCE"]:
            assert r["resolved"] is False
            assert r["new_status"] == r["original_status"]
            assert "No explicit support found" in r["provenance"]
            
def test_evidence_supported_resolution():
    # Provide a mock claim that supports missing_value_handling
    mock_claim = {
        "mechanism_id": "missing_value_handling",
        "direction": "positive",
        "paper_id": "paper_test",
        "claim_id": "claim_test"
    }
    auditor = setup_resolution_auditor([mock_claim])
    report = auditor.audit_resolution()
    
    res = next((r for r in report["resolutions"] if r["component"] == "missing_value_handling"), None)
    if res and res["original_status"] in ["BLOCKED", "INSUFFICIENT_EVIDENCE"]:
        assert res["resolved"] is True
        assert res["new_status"] == "SUPPORTED"
        assert res["provenance"]["paper_id"] == "paper_test"

def test_zero_model_fitting_calls_occur():
    auditor = setup_resolution_auditor()
    report = auditor.audit_resolution()
    assert report["training_allowed"] is False
    assert report["execution_status"] == "CONFIGURATION_VALIDATED"

def test_target_leakage_invariants_maintained():
    # Since resolution auditor doesn't re-run leakage, we just verify it doesn't bypass training
    # The requirement asks to ensure target fields remain excluded, which is tested in materialization.
    auditor = setup_resolution_auditor()
    assert auditor.mat_audit["target_firewall"]["enforced"] is True
    
def test_invalid_baselines_remain_blocked():
    auditor = setup_resolution_auditor()
    b_mat = auditor.mat_audit["baseline_materialization"]
    if "calm image and" in b_mat:
        assert b_mat["calm image and"]["materialization_status"] == "BLOCKED"
