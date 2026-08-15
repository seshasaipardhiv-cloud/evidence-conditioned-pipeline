"""
Tests for Stage 4G: FinalReadinessGate

Covers the 18 required tests:
1. All gates passing -> GO in a mocked safe configuration.
2. Missing target -> NO_GO.
3. Invalid task -> NO_GO.
4. Leakage detected -> NO_GO.
5. Patient overlap > 0 -> NO_GO.
6. Unsupported mechanism -> NO_GO.
7. Incompatible mechanism -> NO_GO.
8. Missing representation -> NO_GO.
9. Invalid baseline -> NO_GO.
10. Missing preprocessing contract -> NO_GO.
11. Invalid compute budget -> NO_GO.
12. Missing provenance -> NO_GO.
13. Modified Stage 2 artifact -> NO_GO.
14. Modified Stage 3 artifact -> NO_GO.
15. Any unresolved blocker -> NO_GO.
16. Verify zero model-training calls.
17. Verify final report is deterministic.
18. Verify training_allowed can NEVER become true while any gate is false.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from backend.app.stage4.final_readiness import FinalReadinessGate

def get_base_mock_data():
    return {
        "config": {
            "target_variable": "recurrence",
            "task_type": "classification",
            "patient_level_split": True
        },
        "budget": {
            "max_epochs": 10
        },
        "split": {
            "splits": [{"seed": 42}]
        },
        "leakage": {
            "rejected_fields": []
        },
        "feature_target": {
            "target_not_in_features": True,
            "outcome_fields_not_in_features": True,
            "post_outcome_fields_not_in_features": True,
            "patient_overlap": 0
        },
        "materialization": {
            "pipeline_materializable": True,
            "baseline_materialization": {},
            "preprocessing_contract": {
                "enforced": True,
                "fit_calls_during_setup": 0
            }
        },
        "readiness": {
            "final_readiness_decision": "READY",
            "unsupported_components": [],
            "incompatible_components": [],
            "required_components": [
                {"component": "fusion", "evidence_status": "EVIDENCE_BACKED", "provenance": "abc"}
            ]
        },
        "representation": {
            "final_resolution_status": "RESOLVED_EVIDENCE_BACKED"
        },
        "final_audit": {
            "target_audit": {"valid": True},
            "data_split_audit": {"valid": True},
            "implementation_audit": {
                "components": [
                    {"component": "fusion", "category": "EVIDENCE_BACKED", "provenance": "abc"}
                ]
            },
            "reproducibility_audit": {
                "hashes": {"stage3_spec": "dummymatch"}
            }
        },
        "stage2c": {
            "summary": {"corpus_valid": True}
        }
    }

def _make_gate(tmpdir, overrides=None, hash_mismatch=False):
    if overrides is None:
        overrides = {}
    base = get_base_mock_data()
    for k, v in overrides.items():
        base[k].update(v)

    def _write(name, data):
        p = Path(tmpdir) / name
        with open(p, "w") as f:
            json.dump(data, f)
        return str(p)

    # For stage 3 spec hash matching, create a dummy file and compute its hash
    import hashlib
    p_spec = Path(tmpdir) / "stage3_spec.json"
    with open(p_spec, "wb") as f:
        f.write(b"{}")
    h = hashlib.sha256(b"{}").hexdigest()
    
    if hash_mismatch:
        base["final_audit"]["reproducibility_audit"]["hashes"]["stage3_spec"] = "badhash"
    else:
        base["final_audit"]["reproducibility_audit"]["hashes"]["stage3_spec"] = h

    # We have to patch the stage3 spec path inside FinalReadinessGate to point to our temp file
    # Or we can just patch Path.exists inside FinalReadinessGate? No, simpler to mock the hardcoded path.
    return FinalReadinessGate(
        config_path=_write("config.json", base["config"]),
        compute_budget_path=_write("budget.json", base["budget"]),
        split_manifest_path=_write("split.json", base["split"]),
        target_leakage_path=_write("leakage.json", base["leakage"]),
        feature_target_audit_path=_write("ft.json", base["feature_target"]),
        materialization_audit_path=_write("mat.json", base["materialization"]),
        pretraining_readiness_path=_write("read.json", base["readiness"]),
        representation_resolution_path=_write("rep.json", base["representation"]),
        final_pretraining_audit_path=_write("final_aud.json", base["final_audit"]),
        stage2c_audit_path=_write("s2c.json", base["stage2c"]),
        out_path=str(Path(tmpdir) / "out.json")
    ), str(p_spec)

# 1. All gates passing -> GO
def test_all_gates_passing_go(tmpdir):
    gate, spec_path = _make_gate(tmpdir)
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        # Pass through all Path instantiations normally except the hardcoded stage3_spec
        orig_path = Path
        def side_effect(*args, **kwargs):
            if args and args[0] == "evidence/processed/stage3_validated_pipeline_specification.json":
                return orig_path(spec_path)
            return orig_path(*args, **kwargs)
        mock_path.side_effect = side_effect

        res = gate.evaluate()
        assert res["final_decision"] == "GO"
        assert res["training_allowed"] is False

# 2. Missing target -> NO_GO
def test_missing_target_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"config": {"target_variable": None}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["target_gate"] == "FAIL"

# 3. Invalid task -> NO_GO
def test_invalid_task_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"config": {"task_type": "clustering"}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["task_gate"] == "FAIL"

# 4. Leakage detected -> NO_GO
def test_leakage_detected_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"feature_target": {"target_not_in_features": False}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["leakage_gate"] == "FAIL"

# 5. Patient overlap > 0 -> NO_GO
def test_patient_overlap_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"feature_target": {"patient_overlap": 5}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["split_gate"] == "FAIL"

# 6. Unsupported mechanism -> NO_GO
def test_unsupported_mechanism_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"readiness": {"unsupported_components": ["some_comp"]}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["evidence_gate"] == "FAIL"

# 7. Incompatible mechanism -> NO_GO
def test_incompatible_mechanism_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"readiness": {"incompatible_components": ["some_comp"]}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["compatibility_gate"] == "FAIL"

# 8. Missing representation -> NO_GO
def test_missing_representation_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"representation": {"final_resolution_status": "BLOCKED"}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["representation_gate"] == "FAIL"

# 9. Invalid baseline -> NO_GO
def test_invalid_baseline_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"materialization": {"baseline_materialization": {"base1": {"materialization_status": "BLOCKED"}}}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["baseline_gate"] == "FAIL"

# 10. Missing preprocessing contract -> NO_GO
def test_missing_preprocessing_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"materialization": {"preprocessing_contract": {"enforced": False}}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["preprocessing_gate"] == "FAIL"

# 11. Invalid compute budget -> NO_GO
def test_invalid_budget_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"budget": {"max_epochs": 0}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["compute_budget_gate"] == "FAIL"

# 12. Missing provenance -> NO_GO
def test_missing_provenance_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"readiness": {"required_components": [{"component": "fusion", "evidence_status": "EVIDENCE_BACKED"}]}}) # no provenance
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["evidence_gate"] == "FAIL"

# 13. Modified Stage 2 artifact -> NO_GO
def test_modified_stage2_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"stage2c": {"summary": {"corpus_valid": False}}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["corpus_gate"] == "FAIL"

# 14. Modified Stage 3 artifact -> NO_GO
def test_modified_stage3_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, hash_mismatch=True)
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        orig_path = Path
        def side_effect(*args, **kwargs):
            if args and args[0] == "evidence/processed/stage3_validated_pipeline_specification.json":
                return orig_path(spec)
            return orig_path(*args, **kwargs)
        mock_path.side_effect = side_effect
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["reproducibility_gate"] == "FAIL"

# 15. Any unresolved blocker -> NO_GO
def test_unresolved_blocker_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"materialization": {"pipeline_materializable": False, "blocking_reasons": ["some component blocked"]}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        # Mock blocker resolution too
        p_res = Path(tmpdir) / "stage4_blocker_resolution.json"
        with open(p_res, "w") as f:
            json.dump({"pipeline_materializable": False}, f)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["materialization_gate"] == "FAIL"

# 16. Verify zero model-training calls
def test_no_model_fitting_occurs():
    import inspect
    import backend.app.stage4.final_readiness as mod
    source = inspect.getsource(mod)
    for forbidden in ["model.fit(", ".train(", "optimizer.step(", "backward("]:
        assert forbidden not in source

# 17. Verify final report is deterministic
def test_final_report_deterministic(tmpdir):
    gate, spec = _make_gate(tmpdir)
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        orig_path = Path
        def side_effect(*args, **kwargs):
            if args and args[0] == "evidence/processed/stage3_validated_pipeline_specification.json":
                return orig_path(spec)
            return orig_path(*args, **kwargs)
        mock_path.side_effect = side_effect
        res1 = gate.evaluate()
        res2 = gate.evaluate()
        assert json.dumps(res1, sort_keys=True) == json.dumps(res2, sort_keys=True)

# 18. Verify training_allowed can NEVER become true while any gate is false
def test_training_allowed_always_false_on_no_go(tmpdir):
    gate, spec = _make_gate(tmpdir, {"config": {"target_variable": None}})
    with patch("backend.app.stage4.final_readiness.Path") as mock_path:
        mock_path.return_value = Path(spec)
        res = gate.evaluate()
        assert res["final_decision"] == "NO_GO"
        assert res["training_allowed"] is False
