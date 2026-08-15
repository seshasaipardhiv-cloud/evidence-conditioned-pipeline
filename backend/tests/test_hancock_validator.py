import pytest
from backend.app.data.hancock.validator import normalize_patient_id, check_target_leakage, validate_json_schema
from pydantic import ValidationError

def test_normalize_patient_id():
    assert normalize_patient_id(" 123 ") == "123"
    assert normalize_patient_id("00123") == "00123"  # Preserves leading zeros
    assert normalize_patient_id("P-01") == "P-01"    # Preserves casing/punctuation
    assert normalize_patient_id(None) == ""

def test_check_target_leakage():
    leakage, msg = check_target_leakage({"survival_months": 12, "age": 50})
    assert leakage is True
    assert "survival_months" in msg
    
    leakage, msg = check_target_leakage({"age": 50, "gender": "F"})
    assert leakage is False

def test_validate_single_clinical_object():
    data = {"patient_id": "001", "age": 60}
    assert validate_json_schema(data, "clinical_data.json") == "clinical"

def test_validate_list_of_clinical_objects():
    data = [
        {"patient_id": "001", "age": 60},
        {"patient_id": "002", "age": 70}
    ]
    assert validate_json_schema(data, "clinical_data.json") == "clinical"

def test_validate_pathology_list():
    data = [{"patient_id": "001", "grading": "G2"}]
    assert validate_json_schema(data, "pathological_data.json") == "pathology"

def test_validate_blood_list():
    data = [{"patient_id": "001", "analyte_name": "WBC", "value": 5.5}]
    assert validate_json_schema(data, "blood_data.json") == "blood"

def test_validate_text_list():
    data = [{"patient_id": "001", "report": "All clear."}]
    assert validate_json_schema(data, "TextData.json") == "text"

def test_malformed_clinical_record():
    # Schema requires patient_id string
    data = {"age": 60}
    with pytest.raises(ValueError, match="Schema validation failed for clinical in clinical_data.json"):
        validate_json_schema(data, "clinical_data.json")

def test_list_with_one_malformed_record():
    data = [
        {"patient_id": "001", "age": 60},
        {"age": 70} # Missing patient_id
    ]
    with pytest.raises(ValueError):
        validate_json_schema(data, "clinical_data.json")

def test_unknown_filename():
    data = {"patient_id": "001"}
    assert validate_json_schema(data, "random_file.json") == "unknown"

def test_reference_ranges_returns_metadata():
    assert validate_json_schema([], "blood_data_reference_ranges.json") == "metadata"
