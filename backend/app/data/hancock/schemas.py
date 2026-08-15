from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any

class ClinicalData(BaseModel):
    patient_id: str
    model_config = ConfigDict(extra='allow')

class PathologyData(BaseModel):
    patient_id: str
    model_config = ConfigDict(extra='allow')

class BloodDataObservation(BaseModel):
    patient_id: str
    analyte_name: Optional[str] = None
    LOINC_code: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    group: Optional[str] = None
    days_before_first_treatment: Optional[int] = None
    model_config = ConfigDict(extra='allow')

class BloodData(BaseModel):
    observations: List[BloodDataObservation]

class TextData(BaseModel):
    patient_id: str
    history: Optional[str] = None
    report: Optional[str] = None
    description: Optional[str] = None
    model_config = ConfigDict(extra='allow')

class ModalityAvailability(BaseModel):
    patient_id: str
    clinical_available: bool = False
    pathology_available: bool = False
    blood_available: bool = False
    text_available: bool = False
    history_available: bool = False
    report_available: bool = False
    description_available: bool = False

class FileManifestEntry(BaseModel):
    relative_path: str
    file_type: str
    size_bytes: int
    sha256: str
    modality: str
    status: str

class IngestionReport(BaseModel):
    dataset_name: str
    source: str
    download_files: List[str]
    file_sizes: Dict[str, int]
    sha256_checksums: Dict[str, str]
    number_of_patients: int
    number_of_clinical_records: int
    number_of_pathology_records: int
    number_of_blood_records: int
    number_of_text_records: int
    text_file_count: int = 0
    text_patient_count: int = 0
    history_count: int = 0
    report_count: int = 0
    description_count: int = 0
    empty_text_count: int = 0
    unmapped_text_files: int = 0
    text_validation_errors: List[str] = []
    missing_modality_counts: Dict[str, int]
    duplicate_id_counts: int
    validation_errors: List[str]
    validation_warnings: List[str]
    ingestion_timestamp: str
