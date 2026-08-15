import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from backend.app.data.hancock.schemas import (
    ModalityAvailability, 
    FileManifestEntry, 
    IngestionReport
)

logger = logging.getLogger(__name__)

class ManifestBuilder:
    def __init__(self):
        self.patient_indices: Dict[str, ModalityAvailability] = {}
        self.canonical_to_raw: Dict[str, set] = {}
        self.file_manifest: List[FileManifestEntry] = []
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        self.duplicate_ids = set()
        
        self.stats = {
            "clinical": 0,
            "pathology": 0,
            "blood": 0,
            "text": 0
        }
        self.text_stats = {
            "text_file_count": 0,
            "history_count": 0,
            "report_count": 0,
            "description_count": 0,
            "empty_text_count": 0,
            "unmapped_text_files": 0
        }
        
    def add_patient_modality(self, raw_patient_id: Any, modality: str, file_path: str):
        if not raw_patient_id:
            self.validation_errors.append(f"Missing patient_id in {file_path}")
            return
            
        raw_id_str = str(raw_patient_id)
        canonical_id = raw_id_str.strip()
        
        if canonical_id not in self.canonical_to_raw:
            self.canonical_to_raw[canonical_id] = set()
            
        self.canonical_to_raw[canonical_id].add(raw_id_str)
        
        if len(self.canonical_to_raw[canonical_id]) > 1:
            self.validation_errors.append(f"Collision detected for canonical ID '{canonical_id}': multiple raw IDs {self.canonical_to_raw[canonical_id]} in {file_path}")
            # Do not merge this record if it collides differently
            return
        
        if canonical_id not in self.patient_indices:
            self.patient_indices[canonical_id] = ModalityAvailability(patient_id=canonical_id)
            
        patient = self.patient_indices[canonical_id]
        
        if modality == "clinical" and patient.clinical_available:
            self.duplicate_ids.add(canonical_id)
            self.validation_warnings.append(f"Duplicate clinical record for patient {canonical_id} in {file_path}")
        elif modality == "pathology" and patient.pathology_available:
            self.duplicate_ids.add(canonical_id)
            self.validation_warnings.append(f"Duplicate pathology record for patient {canonical_id} in {file_path}")
        elif modality == "text" and patient.text_available:
            self.duplicate_ids.add(canonical_id)
            self.validation_warnings.append(f"Duplicate text record for patient {canonical_id} in {file_path}")
            
        if modality == "clinical":
            patient.clinical_available = True
            self.stats["clinical"] += 1
        elif modality == "pathology":
            patient.pathology_available = True
            self.stats["pathology"] += 1
        elif modality == "blood":
            patient.blood_available = True
            self.stats["blood"] += 1
        elif modality == "text":
            patient.text_available = True
            self.stats["text"] += 1

    def add_text_record(self, record_dict: Dict[str, Any]):
        canonical_id = record_dict["patient_id"]
        if canonical_id not in self.patient_indices:
            self.patient_indices[canonical_id] = ModalityAvailability(patient_id=canonical_id)
        
        patient = self.patient_indices[canonical_id]
        
        has_history = bool(record_dict.get("history"))
        has_report = bool(record_dict.get("report"))
        has_description = bool(record_dict.get("description"))
        
        patient.history_available = has_history
        patient.report_available = has_report
        patient.description_available = has_description
        patient.text_available = has_history or has_report or has_description
        
        if patient.text_available:
            self.stats["text"] += 1

    def generate_report(self, dataset_name: str, source: str, files_info: Dict[str, Dict[str, Any]]) -> IngestionReport:
        download_files = list(files_info.keys())
        file_sizes = {f: info["size"] for f, info in files_info.items()}
        sha256_checksums = {f: info["sha256"] for f, info in files_info.items()}
        
        missing_modality_counts = {
            "missing_clinical": sum(1 for p in self.patient_indices.values() if not p.clinical_available),
            "missing_pathology": sum(1 for p in self.patient_indices.values() if not p.pathology_available),
            "missing_blood": sum(1 for p in self.patient_indices.values() if not p.blood_available),
            "missing_text": sum(1 for p in self.patient_indices.values() if not p.text_available)
        }
        
        return IngestionReport(
            dataset_name=dataset_name,
            source=source,
            download_files=download_files,
            file_sizes=file_sizes,
            sha256_checksums=sha256_checksums,
            number_of_patients=len(self.patient_indices),
            number_of_clinical_records=self.stats["clinical"],
            number_of_pathology_records=self.stats["pathology"],
            number_of_blood_records=self.stats["blood"],
            number_of_text_records=self.stats["text"],
            text_file_count=self.text_stats["text_file_count"],
            text_patient_count=self.stats["text"],
            history_count=self.text_stats["history_count"],
            report_count=self.text_stats["report_count"],
            description_count=self.text_stats["description_count"],
            empty_text_count=self.text_stats["empty_text_count"],
            unmapped_text_files=self.text_stats["unmapped_text_files"],
            missing_modality_counts=missing_modality_counts,
            duplicate_id_counts=len(self.duplicate_ids),
            validation_errors=self.validation_errors,
            validation_warnings=self.validation_warnings,
            ingestion_timestamp=datetime.utcnow().isoformat()
        )
