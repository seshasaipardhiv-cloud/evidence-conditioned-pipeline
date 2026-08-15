"""
Stage 1 Pydantic models for the Structured Problem Representation.

Every field carries a provenance sub-object to support later evidence grounding.
No inferences are silently promoted to facts.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    classification = "classification"
    regression = "regression"
    survival_analysis = "survival_analysis"
    ranking = "ranking"
    clustering = "clustering"
    anomaly_detection = "anomaly_detection"
    generation = "generation"
    segmentation = "segmentation"
    unknown = "unknown"


class ConfidenceLevel(str, Enum):
    explicit = "explicit"
    inferred = "inferred"
    unknown = "unknown"


class TargetStatus(str, Enum):
    DEFINED = "DEFINED"
    NOT_DEFINED = "NOT_DEFINED"
    AMBIGUOUS = "AMBIGUOUS"


class CompatibilityStatus(str, Enum):
    compatible = "compatible"
    partially_compatible = "partially_compatible"
    incompatible = "incompatible"
    insufficient_information = "insufficient_information"


class SourceType(str, Enum):
    user_input = "user_input"
    dataset = "dataset"
    derived = "derived"


# ──────────────────────────────────────────────────────────────────────────────
# Provenance
# ──────────────────────────────────────────────────────────────────────────────

class Provenance(BaseModel):
    source_type: SourceType
    source_reference: str = Field(..., description="e.g. 'user_problem_statement' or 'ingestion_report'")
    extraction_method: str = Field(..., description="e.g. 'deterministic_parser' or 'schema_based_profiler'")
    confidence: ConfidenceLevel = ConfidenceLevel.unknown
    evidence_text: Optional[str] = Field(None, description="Original text span or structured evidence used for extraction")


# ──────────────────────────────────────────────────────────────────────────────
# Problem Statement Fields
# ──────────────────────────────────────────────────────────────────────────────

class ExtractedField(BaseModel):
    value: Optional[Any] = None
    provenance: Provenance


class ParsedProblem(BaseModel):
    domain: ExtractedField
    application_area: ExtractedField
    task_type: ExtractedField  # value is TaskType or "unknown"
    prediction_objective: ExtractedField
    target_variable: ExtractedField  # value is None when not stated
    desired_output: ExtractedField
    evaluation_metric: ExtractedField  # value is None when not stated
    computational_constraints: ExtractedField
    latency_constraints: ExtractedField
    interpretability_requirements: ExtractedField
    modality_requirements: ExtractedField  # List[str] of required modalities
    explicit_exclusions: ExtractedField  # List[str]
    raw_problem_statement: str


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Profile Per Modality
# ──────────────────────────────────────────────────────────────────────────────

class ClinicalProfile(BaseModel):
    patient_count: int = 0
    record_count: int = 0
    numerical_fields: List[str] = []
    categorical_fields: List[str] = []
    boolean_fields: List[str] = []
    missing_value_counts: Dict[str, int] = {}
    unique_value_counts: Dict[str, int] = {}
    constant_fields: List[str] = []
    suspicious_identifier_fields: List[str] = []
    provenance: Provenance


class PathologyProfile(BaseModel):
    patient_count: int = 0
    record_count: int = 0
    categorical_fields: List[str] = []
    numerical_fields: List[str] = []
    missing_value_counts: Dict[str, int] = {}
    unique_value_counts: Dict[str, int] = {}
    provenance: Provenance


class BloodProfile(BaseModel):
    patient_count: int = 0
    observation_count: int = 0
    analyte_count: int = 0
    analyte_names: List[str] = []
    units_observed: List[str] = []
    missing_value_count: int = 0
    repeated_measurements_detected: bool = False
    temporal_available: bool = False
    provenance: Provenance


class TextProfile(BaseModel):
    patient_count: int = 0
    history_available_count: int = 0
    report_available_count: int = 0
    description_available_count: int = 0
    # Aggregate statistics only — no raw text stored
    history_char_count_stats: Dict[str, float] = {}
    report_char_count_stats: Dict[str, float] = {}
    description_char_count_stats: Dict[str, float] = {}
    history_word_count_stats: Dict[str, float] = {}
    report_word_count_stats: Dict[str, float] = {}
    description_word_count_stats: Dict[str, float] = {}
    empty_text_rate: float = 0.0
    provenance: Provenance


class DatasetProfile(BaseModel):
    dataset_name: str
    total_patients: int = 0
    clinical: Optional[ClinicalProfile] = None
    pathology: Optional[PathologyProfile] = None
    blood: Optional[BloodProfile] = None
    text: Optional[TextProfile] = None
    provenance: Provenance


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Structural Features
# ──────────────────────────────────────────────────────────────────────────────

class DatasetStructuralFeatures(BaseModel):
    number_of_patients: int
    number_of_modalities: int
    modality_availability: Dict[str, bool]  # {"clinical": True, "text": False, ...}
    numerical_feature_count: int = 0
    categorical_feature_count: int = 0
    binary_feature_count: int = 0
    missingness_rates: Dict[str, float] = {}  # per modality
    duplicate_patient_count: int = 0
    constant_feature_count: int = 0
    sparse_modalities: List[str] = []
    text_length_stats: Dict[str, Any] = {}  # nested per sub-field (history/report/description)
    repeated_measurement_modalities: List[str] = []
    temporal_modalities: List[str] = []
    provenance: Provenance


# ──────────────────────────────────────────────────────────────────────────────
# Target Information
# ──────────────────────────────────────────────────────────────────────────────

class TargetInformation(BaseModel):
    target_status: TargetStatus
    stated_target_variable: Optional[str] = None      # from problem statement, never fabricated
    candidate_fields: Optional[List[str]] = None      # only populated when AMBIGUOUS
    ambiguity_reason: Optional[str] = None
    provenance: Provenance


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility
# ──────────────────────────────────────────────────────────────────────────────

class CompatibilityCheck(BaseModel):
    status: CompatibilityStatus
    reasons: List[str] = []
    missing_for_full_compatibility: List[str] = []
    provenance: Provenance


# ──────────────────────────────────────────────────────────────────────────────
# Top-Level Structured Problem Representation
# ──────────────────────────────────────────────────────────────────────────────

class StructuredProblemRepresentation(BaseModel):
    problem: ParsedProblem
    dataset: DatasetProfile
    task: ExtractedField               # TaskType value
    modalities: Dict[str, bool]        # availability map
    dataset_features: DatasetStructuralFeatures
    missingness: Dict[str, int]        # missing modality counts from ingestion report
    constraints: Dict[str, Optional[Any]]
    target_information: TargetInformation
    compatibility: CompatibilityCheck
    warnings: List[str]
    provenance: Provenance             # top-level provenance for the whole document


# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────

class Stage1Request(BaseModel):
    problem_statement: str = Field(..., min_length=10, description="Natural-language description of the ML problem")


class Stage1Response(BaseModel):
    status: str = "ok"
    representation: StructuredProblemRepresentation
