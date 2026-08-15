"""
Stage 1 — Input and Output Validator

Validates Stage 1 request inputs and final representation for
internal consistency and safety constraints.
"""
from __future__ import annotations

from typing import List, Tuple

from backend.app.stage1.models import (
    StructuredProblemRepresentation,
    TargetStatus,
    TaskType,
)

# Fields that must NEVER appear in any string output of Stage 1
# (raw clinical text indicators — we check report strings, not actual data)
_FORBIDDEN_OUTPUT_PATTERNS = [
    "patient history:",
    "surgical report:",
    "diagnosis report:",
    "clinical note:",
]


def validate_problem_statement(problem_statement: str) -> Tuple[bool, List[str]]:
    """
    Validate the raw problem statement before processing.
    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    if not problem_statement or not problem_statement.strip():
        errors.append("Problem statement must not be empty.")
        return False, errors

    if len(problem_statement.strip()) < 10:
        errors.append("Problem statement is too short (minimum 10 characters).")
        return False, errors

    if len(problem_statement) > 10_000:
        errors.append("Problem statement exceeds maximum allowed length (10,000 characters).")
        return False, errors

    return True, []


def validate_representation_safety(
    representation: StructuredProblemRepresentation,
) -> Tuple[bool, List[str]]:
    """
    Post-generation safety check:
    1. No raw clinical text in any output string.
    2. Target was not fabricated.
    3. Task type is a known enum value.
    4. Provenance fields are populated on every key sub-model.
    """
    violations: List[str] = []

    # Serialize to dict for scanning
    rep_dict = representation.model_dump()
    rep_str = str(rep_dict).lower()

    for pattern in _FORBIDDEN_OUTPUT_PATTERNS:
        if pattern.lower() in rep_str:
            violations.append(f"Forbidden clinical text pattern found in output: '{pattern}'")

    # Target safety
    ti = representation.target_information
    if ti.target_status == TargetStatus.DEFINED and ti.stated_target_variable is None:
        violations.append(
            "Target status is DEFINED but stated_target_variable is None — this is a fabrication."
        )

    # Task type validity
    task_val = representation.task.value
    valid_task_values = {t.value for t in TaskType}
    if task_val not in valid_task_values:
        violations.append(f"Task type '{task_val}' is not a recognised TaskType.")

    # Provenance spot checks
    if representation.problem.task_type.provenance is None:
        violations.append("Missing provenance on task_type field.")
    if representation.target_information.provenance is None:
        violations.append("Missing provenance on target_information.")
    if representation.compatibility.provenance is None:
        violations.append("Missing provenance on compatibility.")

    return len(violations) == 0, violations
