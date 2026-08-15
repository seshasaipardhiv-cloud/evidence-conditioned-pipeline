"""Tests for Stage 1 problem parser."""
import pytest
from backend.app.stage1.problem_parser import parse_problem_statement
from backend.app.stage1.models import TaskType, ConfidenceLevel


def test_explicit_classification():
    stmt = "We want to classify whether patients have cancer recurrence using clinical data."
    result = parse_problem_statement(stmt)
    assert result.task_type.value == TaskType.classification.value
    assert result.task_type.provenance.confidence == ConfidenceLevel.explicit


def test_explicit_regression():
    stmt = "Predict the continuous value of tumour size from blood biomarker data."
    result = parse_problem_statement(stmt)
    assert result.task_type.value == TaskType.regression.value
    assert result.task_type.provenance.confidence == ConfidenceLevel.explicit


def test_explicit_survival_analysis():
    stmt = "We want to perform survival analysis to model time-to-event outcomes for cancer patients."
    result = parse_problem_statement(stmt)
    assert result.task_type.value == TaskType.survival_analysis.value


def test_unknown_task():
    stmt = "We want to develop a multimodal machine learning system for cancer research."
    result = parse_problem_statement(stmt)
    assert result.task_type.value == TaskType.unknown.value
    assert result.task_type.provenance.confidence == ConfidenceLevel.unknown


def test_missing_target():
    stmt = "We want to classify cancer patients using clinical features."
    result = parse_problem_statement(stmt)
    # Target variable must not be fabricated — parser never infers it
    assert result.target_variable.value is None


def test_missing_metric():
    stmt = "Perform classification on cancer data."
    result = parse_problem_statement(stmt)
    assert result.evaluation_metric.value is None


def test_explicit_metric():
    stmt = "Optimise the model using AUROC as the evaluation metric."
    result = parse_problem_statement(stmt)
    assert result.evaluation_metric.value == "AUROC"
    assert result.evaluation_metric.provenance.confidence == ConfidenceLevel.explicit


def test_multimodal_requirement():
    stmt = "Build a system using clinical, pathology, blood and text data for cancer research."
    result = parse_problem_statement(stmt)
    mods = result.modality_requirements.value
    assert mods is not None
    assert "clinical" in mods
    assert "blood" in mods
    assert "text" in mods


def test_explicit_exclusion():
    stmt = "Build a classifier. Do not use imaging data."
    result = parse_problem_statement(stmt)
    excl = result.explicit_exclusions.value
    assert excl is not None
    assert any("imaging" in e for e in excl)


def test_interpretability_detected():
    stmt = "The model must be interpretable and explainable for clinicians."
    result = parse_problem_statement(stmt)
    assert result.interpretability_requirements.value is not None


def test_raw_statement_preserved():
    stmt = "Classify cancer patients."
    result = parse_problem_statement(stmt)
    assert result.raw_problem_statement == stmt


def test_provenance_fields_present():
    stmt = "Classify cancer patients using clinical data."
    result = parse_problem_statement(stmt)
    assert result.task_type.provenance.source_type is not None
    assert result.task_type.provenance.extraction_method is not None
    assert result.domain.provenance.extraction_method is not None
