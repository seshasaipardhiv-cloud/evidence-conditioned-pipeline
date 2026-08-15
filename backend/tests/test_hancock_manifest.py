import pytest
from backend.app.data.hancock.manifest import ManifestBuilder

def test_manifest_builder():
    builder = ManifestBuilder()
    
    builder.add_patient_modality("P001", "clinical", "clin1.json")
    builder.add_patient_modality("P001", "pathology", "path1.json")
    builder.add_patient_modality("P002", "clinical", "clin2.json")
    
    # Duplicate same modality
    builder.add_patient_modality("P001", "clinical", "clin3.json")
    
    # Collision (same canonical ID 'P001' but different raw ID ' P001 ')
    # Wait, raw_id ' P001 ' strips to 'P001' which collides with raw_id 'P001'
    builder.add_patient_modality(" P001 ", "text", "text1.json")
    
    assert "P001" in builder.patient_indices
    assert "P002" in builder.patient_indices
    
    p1 = builder.patient_indices["P001"]
    assert p1.clinical_available
    assert p1.pathology_available
    # text_available should be False because it collided and wasn't merged
    assert not p1.text_available
    assert not p1.blood_available
    
    assert "P001" in builder.duplicate_ids
    assert len(builder.validation_warnings) == 1
    assert any("Collision detected" in e for e in builder.validation_errors)
    
    report = builder.generate_report("TEST", "SRC", {})
    assert report.number_of_patients == 2
    assert report.missing_modality_counts["missing_blood"] == 2
