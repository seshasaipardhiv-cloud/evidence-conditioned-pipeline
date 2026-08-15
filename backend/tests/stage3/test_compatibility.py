import pytest
from backend.app.stage3.compatibility import CompatibilityAuditor
from backend.app.stage3.compatibility_models import CompatibilityStatus

def get_base_context():
    return {
        "problem": {
            "task_type": {"value": "classification"},
            "target_variable": {"value": "recurrence"}
        },
        "modalities": {
            "imaging": True
        }
    }

def get_base_spec():
    return {
        "selected_mechanisms": {
            "feature_representation": "cnn_representation",
            "modality_fusion": "cross_attention"
        },
        "mechanism_scores": {
            "cnn_representation": {"posterior_mean": 0.8},
            "cross_attention": {"posterior_mean": 0.9}
        },
        "expected_baselines": ["single-modality PET"]
    }

def test_unknown_task_blocks_execution():
    ctx = get_base_context()
    ctx["problem"]["task_type"]["value"] = "unknown"
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    assert report.target_gate.blocked is True
    assert report.status == "BLOCKED_BY_TASK"
    assert spec["execution_status"] == "BLOCKED"

def test_ambiguous_target_blocks_execution():
    ctx = get_base_context()
    ctx["problem"]["target_variable"]["value"] = "ambiguous"
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    assert report.target_gate.blocked is True
    assert report.status == "BLOCKED_BY_TASK"
    assert spec["execution_status"] == "BLOCKED"

def test_cnn_rejected_when_image_representation_not_established():
    ctx = get_base_context()
    ctx["modalities"]["imaging"] = False
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    dec = next((m for m in report.mechanism_decisions if m.mechanism == "cnn_representation"), None)
    assert dec is not None
    assert dec.decision == CompatibilityStatus.INCOMPATIBLE
    assert "cnn_representation requires an image representation" in dec.reason

def test_cross_attention_evaluated_against_actual_context():
    ctx = get_base_context()
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    dec = next((m for m in report.mechanism_decisions if m.mechanism == "cross_attention"), None)
    assert dec is not None
    assert dec.decision == CompatibilityStatus.SUPPORTED

def test_posterior_mean_not_described_as_probability():
    ctx = get_base_context()
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    dec = next((m for m in report.mechanism_decisions if m.mechanism == "cross_attention"), None)
    assert dec.posterior_mean == 0.9
    # Must not contain word probability in reason
    assert "probability" not in dec.reason.lower()

def test_malformed_baseline_calm_image_rejected():
    ctx = get_base_context()
    spec = get_base_spec()
    spec["expected_baselines"].append("calm image and")
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    dec = next((b for b in report.baseline_decisions if b.baseline == "calm image and"), None)
    assert dec.decision == CompatibilityStatus.INVALID_BASELINE_ENTITY

def test_malformed_baseline_unimodal_models_rejected():
    ctx = get_base_context()
    spec = get_base_spec()
    spec["expected_baselines"].append("unimodal models across")
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    dec = next((b for b in report.baseline_decisions if b.baseline == "unimodal models across"), None)
    assert dec.decision == CompatibilityStatus.INVALID_BASELINE_ENTITY

def test_unsupported_mechanisms_remain_insufficient_evidence():
    ctx = get_base_context()
    spec = get_base_spec()
    spec["selected_mechanisms"]["base_learner"] = None
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    dec = next((m for m in report.mechanism_decisions if m.component == "base_learner"), None)
    assert dec.decision == CompatibilityStatus.INSUFFICIENT_EVIDENCE

def test_stage2_evidence_never_modified():
    # Since we don't modify evidence passed in
    ctx = get_base_context()
    spec = get_base_spec()
    experiments = [{"id": "exp1"}]
    auditor = CompatibilityAuditor(ctx, spec, {}, [], experiments)
    _ = auditor.audit()
    assert experiments == [{"id": "exp1"}]

def test_every_accepted_mechanism_has_provenance():
    ctx = get_base_context()
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    assert report.provenance_checks["all_mechanisms_have_provenance"] is True

def test_no_arbitrary_mechanism_introduced():
    ctx = get_base_context()
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    report = auditor.audit()
    mechs = [m.mechanism for m in report.mechanism_decisions if m.mechanism != "None"]
    assert set(mechs) == {"cnn_representation", "cross_attention"}

def test_validated_pipeline_cannot_become_executable_while_target_blocked():
    ctx = get_base_context()
    ctx["problem"]["target_variable"]["value"] = "ambiguous"
    spec = get_base_spec()
    auditor = CompatibilityAuditor(ctx, spec, {}, [], [])
    _ = auditor.audit()
    assert spec["execution_status"] == "BLOCKED"
