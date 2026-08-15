"""Tests for Stage 1 orchestrator (end-to-end unit tests with no real dataset files)."""
import pytest
from backend.app.stage1.orchestrator import run_stage1
from backend.app.stage1.models import TaskType, TargetStatus, CompatibilityStatus


MULTIMODAL_STMT = (
    "We want to develop a multimodal machine learning system for cancer research "
    "using clinical, pathology, blood and text data. "
    "The system should identify an appropriate predictive learning pipeline based on scientific evidence."
)

CLASSIFICATION_WITH_TARGET_STMT = (
    "Classify cancer patients for recurrence detection using clinical and blood data. "
    "Optimise using AUROC."
)

REGRESSION_STMT = "Predict continuous tumour size values from pathology and clinical measurements."
UNKNOWN_STMT = "Develop a system for cancer research."


def test_full_orchestration_unknown_task():
    rep = run_stage1(UNKNOWN_STMT, write_outputs=False)
    assert rep.task.value == TaskType.unknown.value
    assert rep.compatibility.status == CompatibilityStatus.insufficient_information
    assert len(rep.warnings) > 0


def test_full_orchestration_classification():
    rep = run_stage1(CLASSIFICATION_WITH_TARGET_STMT, write_outputs=False)
    assert rep.task.value == TaskType.classification.value
    assert rep.problem.evaluation_metric.value == "AUROC"


def test_full_orchestration_regression():
    rep = run_stage1(REGRESSION_STMT, write_outputs=False)
    assert rep.task.value == TaskType.regression.value


def test_target_never_fabricated():
    rep = run_stage1(MULTIMODAL_STMT, write_outputs=False)
    # No target is stated in this problem statement — must not be fabricated
    assert rep.target_information.stated_target_variable is None
    assert rep.target_information.target_status != TargetStatus.DEFINED


def test_modalities_from_problem_statement():
    rep = run_stage1(MULTIMODAL_STMT, write_outputs=False)
    mod_req = rep.problem.modality_requirements.value
    assert mod_req is not None
    assert "clinical" in mod_req
    assert "blood" in mod_req
    assert "text" in mod_req


def test_provenance_everywhere():
    rep = run_stage1(MULTIMODAL_STMT, write_outputs=False)
    assert rep.provenance is not None
    assert rep.problem.task_type.provenance is not None
    assert rep.target_information.provenance is not None
    assert rep.compatibility.provenance is not None
    assert rep.dataset_features.provenance is not None


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        run_stage1("Hi", write_outputs=False)


def test_no_raw_text_in_output():
    rep = run_stage1(MULTIMODAL_STMT, write_outputs=False)
    rep_str = str(rep.model_dump())
    assert "patient history:" not in rep_str.lower()
    assert "surgical report:" not in rep_str.lower()


def test_warnings_populated_for_missing_info():
    rep = run_stage1(UNKNOWN_STMT, write_outputs=False)
    assert any("Task type" in w or "target" in w.lower() for w in rep.warnings)
