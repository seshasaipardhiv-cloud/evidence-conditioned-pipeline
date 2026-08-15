"""
Stage 1 — Task Identifier

Determines target status and validates task-dataset consistency.
Never fabricates a target. Never infers survival or recurrence from data.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from backend.app.stage1.models import (
    CompatibilityCheck,
    CompatibilityStatus,
    ConfidenceLevel,
    DatasetStructuralFeatures,
    ExtractedField,
    ParsedProblem,
    Provenance,
    SourceType,
    TargetInformation,
    TargetStatus,
    TaskType,
)

_TASK_PROV = Provenance(
    source_type=SourceType.derived,
    source_reference="parsed_problem + dataset_features",
    extraction_method="deterministic_task_identifier",
    confidence=ConfidenceLevel.explicit,
)

# Fields that are potentially target-related — we flag but never remove
_POTENTIAL_TARGET_FIELDS = {
    "survival", "recurrence", "death", "event", "status", "outcome",
    "label", "target", "progression", "response", "class",
}


def identify_target(
    parsed_problem: ParsedProblem,
    available_fields: List[str],
) -> TargetInformation:
    """
    Determine target status from the problem statement ONLY.
    Never infer a target from dataset content.
    """
    stated = parsed_problem.target_variable.value  # always None from parser unless explicitly stated

    if stated is not None:
        return TargetInformation(
            target_status=TargetStatus.DEFINED,
            stated_target_variable=stated,
            candidate_fields=None,
            ambiguity_reason=None,
            provenance=Provenance(
                source_type=SourceType.user_input,
                source_reference="user_problem_statement",
                extraction_method="deterministic_parser",
                confidence=ConfidenceLevel.explicit,
                evidence_text=None,
            ),
        )

    # Check if any available dataset fields *look* target-related — flag as AMBIGUOUS
    candidates = [
        f for f in available_fields
        if any(t in f.lower() for t in _POTENTIAL_TARGET_FIELDS)
    ]

    if candidates:
        return TargetInformation(
            target_status=TargetStatus.AMBIGUOUS,
            stated_target_variable=None,
            candidate_fields=candidates,
            ambiguity_reason=(
                "Target variable not stated in problem statement. "
                "The following dataset fields may be target-related and require human confirmation."
            ),
            provenance=Provenance(
                source_type=SourceType.dataset,
                source_reference="dataset field names",
                extraction_method="field_name_heuristic",
                confidence=ConfidenceLevel.inferred,
                evidence_text=f"Candidate fields: {candidates}",
            ),
        )

    return TargetInformation(
        target_status=TargetStatus.NOT_DEFINED,
        stated_target_variable=None,
        candidate_fields=None,
        ambiguity_reason=None,
        provenance=Provenance(
            source_type=SourceType.user_input,
            source_reference="user_problem_statement",
            extraction_method="deterministic_parser",
            confidence=ConfidenceLevel.explicit,
            evidence_text="No target variable was stated in the problem statement.",
        ),
    )


def check_task_dataset_compatibility(
    task_type: str,
    target_info: TargetInformation,
    features: DatasetStructuralFeatures,
    modality_availability: Dict[str, bool],
) -> CompatibilityCheck:
    """
    Evaluate whether the requested task is compatible with the available dataset.
    Never recommends models or pipelines — only reports structural compatibility.
    """
    reasons: List[str] = []
    missing_for_full: List[str] = []

    task = TaskType(task_type) if task_type in TaskType.__members__.values() else TaskType.unknown

    if task == TaskType.unknown:
        return CompatibilityCheck(
            status=CompatibilityStatus.insufficient_information,
            reasons=["Task type is unknown. Cannot evaluate compatibility until task is specified."],
            missing_for_full_compatibility=["task_type"],
            provenance=_TASK_PROV,
        )

    # Classification
    if task == TaskType.classification:
        if target_info.target_status == TargetStatus.NOT_DEFINED:
            reasons.append("Classification requires a defined target variable, but none was stated.")
            missing_for_full.append("target_variable")
        elif target_info.target_status == TargetStatus.AMBIGUOUS:
            reasons.append("Classification target is ambiguous — human confirmation required.")

    # Regression
    elif task == TaskType.regression:
        if target_info.target_status == TargetStatus.NOT_DEFINED:
            reasons.append("Regression requires a numerical target variable, but none was stated.")
            missing_for_full.append("target_variable")
        elif target_info.target_status == TargetStatus.AMBIGUOUS:
            reasons.append("Regression target is ambiguous — requires confirmation of numerical nature.")

    # Survival analysis
    elif task == TaskType.survival_analysis:
        if target_info.target_status == TargetStatus.NOT_DEFINED:
            reasons.append("Survival analysis requires explicit survival time and event indicator — not stated.")
            missing_for_full.extend(["survival_time_field", "event_indicator_field"])

    # Clustering — target not required
    elif task == TaskType.clustering:
        if target_info.target_status == TargetStatus.DEFINED:
            reasons.append("Note: clustering is unsupervised; a stated target will not be used for training.")

    # Text classification — needs text
    if "text" in task_type.lower() or task == TaskType.classification:
        if not modality_availability.get("text", False):
            reasons.append("Task may require text modality, but text is not available in this dataset.")
            missing_for_full.append("text_modality")

    # Check clinical / pathology requirements
    if not modality_availability.get("clinical", False):
        reasons.append("Clinical modality is unavailable.")
        missing_for_full.append("clinical_modality")

    if features.number_of_patients == 0:
        reasons.append("No patients detected in dataset.")
        missing_for_full.append("patient_records")
        return CompatibilityCheck(
            status=CompatibilityStatus.incompatible,
            reasons=reasons,
            missing_for_full_compatibility=missing_for_full,
            provenance=_TASK_PROV,
        )

    if missing_for_full:
        return CompatibilityCheck(
            status=CompatibilityStatus.partially_compatible,
            reasons=reasons,
            missing_for_full_compatibility=missing_for_full,
            provenance=_TASK_PROV,
        )

    return CompatibilityCheck(
        status=CompatibilityStatus.compatible,
        reasons=reasons if reasons else ["All checked structural requirements are met."],
        missing_for_full_compatibility=[],
        provenance=_TASK_PROV,
    )
