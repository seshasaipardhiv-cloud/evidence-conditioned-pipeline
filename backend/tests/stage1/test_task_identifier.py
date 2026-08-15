"""Tests for Stage 1 task identifier."""
import pytest
from backend.app.stage1.task_identifier import identify_target, check_task_dataset_compatibility
from backend.app.stage1.models import (
    TargetStatus, CompatibilityStatus, TaskType,
    DatasetStructuralFeatures, Provenance, SourceType, ConfidenceLevel
)
from backend.app.stage1.problem_parser import parse_problem_statement

_PROV = Provenance(
    source_type=SourceType.derived,
    source_reference="test",
    extraction_method="test",
    confidence=ConfidenceLevel.explicit,
)

def _features(n_patients=10, modalities=None) -> DatasetStructuralFeatures:
    if modalities is None:
        modalities = {"clinical": True, "pathology": True, "blood": True, "text": True}
    return DatasetStructuralFeatures(
        number_of_patients=n_patients,
        number_of_modalities=sum(modalities.values()),
        modality_availability=modalities,
        provenance=_PROV,
    )


def test_target_not_defined_when_absent():
    parsed = parse_problem_statement("Build a classifier for cancer using clinical data.")
    target = identify_target(parsed, [])
    assert target.target_status == TargetStatus.NOT_DEFINED
    assert target.stated_target_variable is None


def test_target_ambiguous_with_candidate_fields():
    parsed = parse_problem_statement("Build a model for cancer patients.")
    target = identify_target(parsed, ["survival_months", "recurrence_flag", "age"])
    assert target.target_status == TargetStatus.AMBIGUOUS
    assert "survival_months" in target.candidate_fields
    assert "recurrence_flag" in target.candidate_fields
    assert "age" not in target.candidate_fields  # not a target keyword


def test_target_safety_never_fabricated():
    # No target keyword in the problem and no suspicious fields
    parsed = parse_problem_statement("Cluster patients by clinical profiles.")
    target = identify_target(parsed, ["age", "gender", "stage"])
    assert target.target_status == TargetStatus.NOT_DEFINED
    assert target.stated_target_variable is None


def test_classification_compatible():
    parsed = parse_problem_statement("Classify cancer recurrence using clinical data.")
    features = _features()
    # Inject a target to make it defined — simulating explicit statement
    from backend.app.stage1.models import TargetInformation
    target_info = TargetInformation(
        target_status=TargetStatus.DEFINED,
        stated_target_variable="recurrence",
        provenance=_PROV,
    )
    compat = check_task_dataset_compatibility(
        task_type=TaskType.classification.value,
        target_info=target_info,
        features=features,
        modality_availability=features.modality_availability,
    )
    assert compat.status == CompatibilityStatus.compatible


def test_classification_incompatible_no_target():
    features = _features()
    from backend.app.stage1.models import TargetInformation
    target_info = TargetInformation(
        target_status=TargetStatus.NOT_DEFINED,
        stated_target_variable=None,
        provenance=_PROV,
    )
    compat = check_task_dataset_compatibility(
        task_type=TaskType.classification.value,
        target_info=target_info,
        features=features,
        modality_availability=features.modality_availability,
    )
    assert compat.status == CompatibilityStatus.partially_compatible
    assert "target_variable" in compat.missing_for_full_compatibility


def test_unknown_task_insufficient_info():
    features = _features()
    from backend.app.stage1.models import TargetInformation
    target_info = TargetInformation(
        target_status=TargetStatus.NOT_DEFINED,
        stated_target_variable=None,
        provenance=_PROV,
    )
    compat = check_task_dataset_compatibility(
        task_type=TaskType.unknown.value,
        target_info=target_info,
        features=features,
        modality_availability=features.modality_availability,
    )
    assert compat.status == CompatibilityStatus.insufficient_information


def test_missing_text_modality_flagged():
    mods = {"clinical": True, "pathology": True, "blood": True, "text": False}
    features = _features(modalities=mods)
    from backend.app.stage1.models import TargetInformation
    target_info = TargetInformation(
        target_status=TargetStatus.NOT_DEFINED,
        stated_target_variable=None,
        provenance=_PROV,
    )
    compat = check_task_dataset_compatibility(
        task_type=TaskType.classification.value,
        target_info=target_info,
        features=features,
        modality_availability=mods,
    )
    assert "text_modality" in compat.missing_for_full_compatibility


def test_no_patients_incompatible():
    features = _features(n_patients=0)
    from backend.app.stage1.models import TargetInformation
    target_info = TargetInformation(
        target_status=TargetStatus.DEFINED,
        stated_target_variable="recurrence",
        provenance=_PROV,
    )
    compat = check_task_dataset_compatibility(
        task_type=TaskType.classification.value,
        target_info=target_info,
        features=features,
        modality_availability=features.modality_availability,
    )
    assert compat.status == CompatibilityStatus.incompatible
