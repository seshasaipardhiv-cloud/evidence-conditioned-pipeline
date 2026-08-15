"""Tests for Stage 1 feature extractor."""
import pytest
from backend.app.stage1.dataset_profiler import (
    profile_clinical, profile_pathology, profile_blood, profile_text
)
from backend.app.stage1.feature_extractor import extract_structural_features
from backend.app.stage1.models import DatasetProfile, Provenance, SourceType, ConfidenceLevel

_PROV = Provenance(
    source_type=SourceType.dataset,
    source_reference="test",
    extraction_method="test",
    confidence=ConfidenceLevel.explicit,
)


def _make_profile(with_text=True, with_blood=True) -> DatasetProfile:
    clin = profile_clinical([
        {"patient_id": "P001", "age": 60, "gender": "M"},
        {"patient_id": "P002", "age": 70, "gender": "F"},
    ])
    path = profile_pathology([
        {"patient_id": "P001", "grade": "III", "size_cm": 3.1},
    ])
    blood = profile_blood([
        {"patient_id": "P001", "analyte_name": "WBC", "value": 8.5, "unit": "10^9/L", "days_before_first_treatment": -10},
        {"patient_id": "P001", "analyte_name": "WBC", "value": 9.0, "unit": "10^9/L", "days_before_first_treatment": -5},
    ]) if with_blood else None
    text = profile_text([
        {"patient_id": "P001", "history": "history text", "report": "report text", "description": ""},
    ]) if with_text else None

    return DatasetProfile(
        dataset_name="TEST",
        total_patients=2,
        clinical=clin,
        pathology=path,
        blood=blood,
        text=text,
        provenance=_PROV,
    )


def test_feature_extraction_full():
    profile = _make_profile()
    features = extract_structural_features(profile)

    assert features.number_of_patients == 2
    assert features.number_of_modalities == 4
    assert features.modality_availability["clinical"] is True
    assert features.modality_availability["pathology"] is True
    assert features.modality_availability["blood"] is True
    assert features.modality_availability["text"] is True
    assert features.numerical_feature_count > 0
    assert features.categorical_feature_count > 0


def test_missing_modality_reflected():
    profile = _make_profile(with_text=False, with_blood=False)
    features = extract_structural_features(profile)
    assert features.modality_availability["text"] is False
    assert features.modality_availability["blood"] is False
    assert features.number_of_modalities == 2


def test_repeated_measurement_blood():
    profile = _make_profile(with_blood=True)
    features = extract_structural_features(profile)
    assert "blood" in features.repeated_measurement_modalities


def test_temporal_blood():
    profile = _make_profile(with_blood=True)
    features = extract_structural_features(profile)
    assert "blood" in features.temporal_modalities


def test_provenance_present():
    profile = _make_profile()
    features = extract_structural_features(profile)
    assert features.provenance is not None
    assert features.provenance.source_type == SourceType.derived
