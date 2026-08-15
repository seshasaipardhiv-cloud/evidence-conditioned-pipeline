"""
Stage 1 — Feature Extractor

Computes deterministic structural features from the DatasetProfile.
No ML preprocessing. No imputation. No normalization. No model training.
"""
from __future__ import annotations

from backend.app.stage1.models import (
    ConfidenceLevel,
    DatasetProfile,
    DatasetStructuralFeatures,
    Provenance,
    SourceType,
)


_FEATURE_PROV = Provenance(
    source_type=SourceType.derived,
    source_reference="dataset_profiler output",
    extraction_method="structural_feature_extractor",
    confidence=ConfidenceLevel.explicit,
)


def extract_structural_features(profile: DatasetProfile) -> DatasetStructuralFeatures:
    modality_availability = {
        "clinical": profile.clinical is not None,
        "pathology": profile.pathology is not None,
        "blood": profile.blood is not None,
        "text": profile.text is not None,
    }
    num_modalities = sum(modality_availability.values())

    numerical_count = 0
    categorical_count = 0
    binary_count = 0
    missingness_rates: dict = {}
    constant_count = 0
    sparse_modalities = []
    repeated_measurement_modalities = []
    temporal_modalities = []
    text_length_stats: dict = {}

    # Clinical
    if profile.clinical:
        c = profile.clinical
        numerical_count += len(c.numerical_fields)
        categorical_count += len(c.categorical_fields)
        binary_count += len(c.boolean_fields)
        constant_count += len(c.constant_fields)
        total_fields = len(c.numerical_fields) + len(c.categorical_fields) + len(c.boolean_fields)
        if total_fields > 0 and c.record_count > 0:
            total_missing = sum(c.missing_value_counts.values())
            total_possible = total_fields * c.record_count
            missingness_rates["clinical"] = round(total_missing / total_possible, 4)
        if total_fields > 0 and sum(c.missing_value_counts.values()) / max(c.record_count * total_fields, 1) > 0.5:
            sparse_modalities.append("clinical")

    # Pathology
    if profile.pathology:
        p = profile.pathology
        numerical_count += len(p.numerical_fields)
        categorical_count += len(p.categorical_fields)
        if p.record_count > 0:
            total_fields = len(p.categorical_fields) + len(p.numerical_fields)
            if total_fields > 0:
                total_missing = sum(p.missing_value_counts.values())
                total_possible = total_fields * p.record_count
                missingness_rates["pathology"] = round(total_missing / total_possible, 4)

    # Blood
    if profile.blood:
        b = profile.blood
        numerical_count += 1  # value field
        if b.observation_count > 0:
            missingness_rates["blood"] = round(b.missing_value_count / b.observation_count, 4)
        if b.repeated_measurements_detected:
            repeated_measurement_modalities.append("blood")
        if b.temporal_available:
            temporal_modalities.append("blood")

    # Text
    if profile.text:
        t = profile.text
        if t.patient_count > 0 and t.empty_text_rate > 0.5:
            sparse_modalities.append("text")
        # Aggregate text length stats (no raw text)
        all_stats: dict = {}
        for subfield, stats_dict in [
            ("history", t.history_char_count_stats),
            ("report", t.report_char_count_stats),
            ("description", t.description_char_count_stats),
        ]:
            if stats_dict:
                all_stats[subfield] = stats_dict
        text_length_stats = all_stats

    return DatasetStructuralFeatures(
        number_of_patients=profile.total_patients,
        number_of_modalities=num_modalities,
        modality_availability=modality_availability,
        numerical_feature_count=numerical_count,
        categorical_feature_count=categorical_count,
        binary_feature_count=binary_count,
        missingness_rates=missingness_rates,
        duplicate_patient_count=0,  # populated from ingestion report by orchestrator
        constant_feature_count=constant_count,
        sparse_modalities=sparse_modalities,
        text_length_stats=text_length_stats,
        repeated_measurement_modalities=repeated_measurement_modalities,
        temporal_modalities=temporal_modalities,
        provenance=_FEATURE_PROV,
    )
