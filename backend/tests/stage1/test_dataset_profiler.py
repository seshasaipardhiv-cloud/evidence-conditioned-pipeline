"""Tests for Stage 1 dataset profiler using synthetic fixtures."""
import pytest
from backend.app.stage1.dataset_profiler import (
    profile_clinical,
    profile_pathology,
    profile_blood,
    profile_text,
)

CLINICAL_RECORDS = [
    {"patient_id": "P001", "age": 65, "gender": "M", "stage": "III", "is_smoker": True},
    {"patient_id": "P002", "age": 72, "gender": "F", "stage": "II",  "is_smoker": False},
    {"patient_id": "P003", "age": 58, "gender": "M", "stage": None,  "is_smoker": None},
]

PATHOLOGY_RECORDS = [
    {"patient_id": "P001", "differentiation": "poor",  "lymph_node_status": "positive", "tumour_size_cm": 4.2},
    {"patient_id": "P002", "differentiation": "moderate", "lymph_node_status": "negative", "tumour_size_cm": 2.1},
]

BLOOD_OBSERVATIONS = [
    {"patient_id": "P001", "analyte_name": "WBC",  "value": 8.5,  "unit": "10^9/L", "days_before_first_treatment": -30},
    {"patient_id": "P001", "analyte_name": "WBC",  "value": 9.0,  "unit": "10^9/L", "days_before_first_treatment": -15},
    {"patient_id": "P002", "analyte_name": "HGB",  "value": 11.2, "unit": "g/dL",   "days_before_first_treatment": None},
    {"patient_id": "P003", "analyte_name": "PLT",  "value": None, "unit": "10^9/L", "days_before_first_treatment": -7},
]

TEXT_RECORDS = [
    {"patient_id": "P001", "history": "Patient has a long history of smoking.", "report": "Biopsy confirmed carcinoma.", "description": ""},
    {"patient_id": "P002", "history": "",   "report": "Margins clear.", "description": ""},
    {"patient_id": "P003", "history": None, "report": None,             "description": None},
]


def test_clinical_profiling():
    profile = profile_clinical(CLINICAL_RECORDS)
    assert profile.patient_count == 3
    assert profile.record_count == 3
    assert "age" in profile.numerical_fields
    assert "gender" in profile.categorical_fields
    assert "is_smoker" in profile.boolean_fields
    # stage has a missing value for P003
    assert "stage" in profile.missing_value_counts


def test_pathology_profiling():
    profile = profile_pathology(PATHOLOGY_RECORDS)
    assert profile.patient_count == 2
    assert profile.record_count == 2
    assert "tumour_size_cm" in profile.numerical_fields
    assert "differentiation" in profile.categorical_fields


def test_blood_profiling():
    profile = profile_blood(BLOOD_OBSERVATIONS)
    assert profile.patient_count == 3
    assert profile.observation_count == 4
    assert profile.analyte_count == 3
    assert "WBC" in profile.analyte_names
    assert profile.missing_value_count == 1   # PLT has None value
    assert profile.repeated_measurements_detected  # P001 has 2 WBC readings
    assert profile.temporal_available             # days_before_first_treatment present


def test_text_profiling_no_raw_text():
    profile = profile_text(TEXT_RECORDS)
    # Raw text must NEVER appear in profile output
    profile_dict = profile.model_dump()
    profile_str = str(profile_dict)
    assert "smoking" not in profile_str
    assert "carcinoma" not in profile_str
    assert "Biopsy" not in profile_str

    assert profile.patient_count == 3
    assert profile.history_available_count == 1
    assert profile.report_available_count == 2
    # P003 has no text at all
    assert profile.empty_text_rate > 0.0


def test_missing_text():
    all_empty = [{"patient_id": "P001", "history": None, "report": None, "description": None}]
    profile = profile_text(all_empty)
    assert profile.empty_text_rate == 1.0
    assert profile.history_available_count == 0
