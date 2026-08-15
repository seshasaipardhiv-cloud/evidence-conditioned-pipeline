import logging
from typing import Dict, Any, Tuple, List
from pydantic import ValidationError, TypeAdapter
from backend.app.data.hancock.schemas import ClinicalData, PathologyData, TextData, BloodDataObservation

logger = logging.getLogger(__name__)

def normalize_patient_id(raw_id: Any) -> str:
    if raw_id is None:
        return ""
    return str(raw_id).strip()

def check_target_leakage(data: Dict[str, Any]) -> Tuple[bool, str]:
    leakage_fields = {"survival", "recurrence", "target", "label", "outcome", "death"}
    found_leakage = []
    for key in data.keys():
        key_lower = key.lower()
        if any(leakage_term in key_lower for leakage_term in leakage_fields):
            found_leakage.append(key)
    
    if found_leakage:
        return True, f"Potential target leakage fields detected: {found_leakage}"
    return False, ""

def validate_json_schema(data: Any, filename: str) -> str:
    """
    Validate dataset against official HANCOTHON schema using Pydantic models.
    Supports both individual objects and arrays of objects.
    Returns the modality name string.
    Raises ValueError on validation failure.
    """
    modality = "unknown"
    model_class = None

    if "clinical" in filename.lower():
        modality = "clinical"
        model_class = ClinicalData
    elif "patho" in filename.lower():
        modality = "pathology"
        model_class = PathologyData
    elif "blood" in filename.lower() or "lab" in filename.lower():
        # skip reference ranges file for now
        if "reference" in filename.lower():
            return "metadata"
        modality = "blood"
        model_class = BloodDataObservation
    elif "text" in filename.lower():
        modality = "text"
        model_class = TextData

    if model_class:
        try:
            if isinstance(data, list):
                adapter = TypeAdapter(List[model_class])
                adapter.validate_python(data)
            elif isinstance(data, dict):
                model_class.model_validate(data)
            else:
                raise ValueError(f"Expected JSON object or array, got {type(data).__name__}")
        except ValidationError as e:
            # We raise ValueError with minimal sensitive info (pydantic handles this nicely)
            logger.error(f"Validation failed for {filename}: {str(e)[:500]}")
            raise ValueError(f"Schema validation failed for {modality} in {filename}")
        
        return modality

    return "unknown"
