"""
Stage 1 — Orchestrator

Coordinates: problem parsing → dataset profiling → feature extraction →
             task identification → compatibility check → safety validation →
             file output.

No LLM calls. No model training. No pipeline generation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from backend.app.stage1.models import (
    ConfidenceLevel,
    ExtractedField,
    Provenance,
    SourceType,
    StructuredProblemRepresentation,
    TaskType,
)
from backend.app.stage1.problem_parser import parse_problem_statement
from backend.app.stage1.dataset_profiler import profile_dataset_from_files
from backend.app.stage1.feature_extractor import extract_structural_features
from backend.app.stage1.task_identifier import identify_target, check_task_dataset_compatibility
from backend.app.stage1.validator import validate_problem_statement, validate_representation_safety

logger = logging.getLogger(__name__)

# Default paths (relative to project root)
_STRUCTURED_DIR = Path("data/raw/hancock/structured")
_TEXT_DIR = Path("data/raw/hancock/text")
_INGESTION_REPORT = Path("data/metadata/hancock/ingestion_report.json")
_STAGE1_OUTPUT = Path("data/processed/hancock/stage1_problem_representation.json")
_STAGE1_PROFILE = Path("data/metadata/hancock/stage1_profile_report.json")

_ORCH_PROV = Provenance(
    source_type=SourceType.derived,
    source_reference="stage1_orchestrator",
    extraction_method="stage1_pipeline",
    confidence=ConfidenceLevel.explicit,
)


def run_stage1(
    problem_statement: str,
    structured_dir: Optional[Path] = None,
    text_dir: Optional[Path] = None,
    ingestion_report_path: Optional[Path] = None,
    write_outputs: bool = True,
) -> StructuredProblemRepresentation:
    """
    Full Stage 1 pipeline.
    Raises ValueError for invalid input.
    Raises RuntimeError if safety validation fails.
    """
    structured_dir = structured_dir or _STRUCTURED_DIR
    text_dir = text_dir or _TEXT_DIR
    ingestion_report_path = ingestion_report_path or _INGESTION_REPORT

    # ── 1. Validate input ────────────────────────────────────────────────────
    is_valid, input_errors = validate_problem_statement(problem_statement)
    if not is_valid:
        raise ValueError(f"Invalid problem statement: {input_errors}")

    logger.info("Stage 1: parsing problem statement")

    # ── 2. Parse problem statement ───────────────────────────────────────────
    parsed = parse_problem_statement(problem_statement)

    # ── 3. Profile dataset ───────────────────────────────────────────────────
    logger.info("Stage 1: profiling dataset")
    dataset_profile = profile_dataset_from_files(
        structured_dir=structured_dir,
        text_dir=text_dir,
        ingestion_report_path=ingestion_report_path,
    )

    # ── 4. Extract structural features ──────────────────────────────────────
    features = extract_structural_features(dataset_profile)

    # Enrich duplicate count from ingestion report
    if ingestion_report_path.exists():
        with open(ingestion_report_path, "r", encoding="utf-8") as f:
            ir = json.load(f)
        features.duplicate_patient_count = ir.get("duplicate_id_counts", 0)
        missingness = ir.get("missing_modality_counts", {})
    else:
        missingness = {}

    # ── 5. Identify target ───────────────────────────────────────────────────
    # Collect all known field names from profile for candidate detection
    available_fields: List[str] = []
    if dataset_profile.clinical:
        available_fields += (
            dataset_profile.clinical.numerical_fields
            + dataset_profile.clinical.categorical_fields
            + dataset_profile.clinical.boolean_fields
        )
    if dataset_profile.pathology:
        available_fields += (
            dataset_profile.pathology.categorical_fields
            + dataset_profile.pathology.numerical_fields
        )

    target_info = identify_target(parsed, available_fields)

    # ── 6. Modality availability ──────────────────────────────────────────────
    modality_availability = features.modality_availability

    # ── 7. Task consistency check ────────────────────────────────────────────
    task_type_value = parsed.task_type.value or TaskType.unknown.value
    compatibility = check_task_dataset_compatibility(
        task_type=task_type_value,
        target_info=target_info,
        features=features,
        modality_availability=modality_availability,
    )

    # ── 8. Collect warnings ──────────────────────────────────────────────────
    warnings: List[str] = []
    if task_type_value == TaskType.unknown.value:
        warnings.append("Task type could not be determined from the problem statement.")
    if target_info.target_status.value != "DEFINED":
        warnings.append(f"Prediction target status: {target_info.target_status.value}")
    if compatibility.missing_for_full_compatibility:
        warnings.append(
            f"Missing for full compatibility: {compatibility.missing_for_full_compatibility}"
        )
    for mod, available in modality_availability.items():
        if not available:
            warnings.append(f"Modality '{mod}' is not available in the dataset.")

    # ── 9. Constraints summary ───────────────────────────────────────────────
    constraints: Dict = {
        "computational": parsed.computational_constraints.value,
        "latency": parsed.latency_constraints.value,
        "interpretability": parsed.interpretability_requirements.value,
        "explicit_exclusions": parsed.explicit_exclusions.value,
    }

    # ── 10. Task ExtractedField ──────────────────────────────────────────────
    task_field = ExtractedField(
        value=task_type_value,
        provenance=parsed.task_type.provenance,
    )

    # ── 11. Build representation ──────────────────────────────────────────────
    representation = StructuredProblemRepresentation(
        problem=parsed,
        dataset=dataset_profile,
        task=task_field,
        modalities=modality_availability,
        dataset_features=features,
        missingness=missingness,
        constraints=constraints,
        target_information=target_info,
        compatibility=compatibility,
        warnings=warnings,
        provenance=_ORCH_PROV,
    )

    # ── 12. Safety validation ────────────────────────────────────────────────
    is_safe, violations = validate_representation_safety(representation)
    if not is_safe:
        raise RuntimeError(f"Stage 1 safety validation failed: {violations}")

    # ── 13. Write outputs ────────────────────────────────────────────────────
    if write_outputs:
        _STAGE1_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(_STAGE1_OUTPUT, "w", encoding="utf-8") as f:
            json.dump(representation.model_dump(), f, indent=2, default=str)
        logger.info(f"Stage 1 representation written to {_STAGE1_OUTPUT}")

        # Profile report — aggregate stats only
        _STAGE1_PROFILE.parent.mkdir(parents=True, exist_ok=True)
        profile_dict = dataset_profile.model_dump()
        profile_dict["generated_at"] = datetime.utcnow().isoformat()
        with open(_STAGE1_PROFILE, "w", encoding="utf-8") as f:
            json.dump(profile_dict, f, indent=2, default=str)
        logger.info(f"Stage 1 profile report written to {_STAGE1_PROFILE}")

    return representation
