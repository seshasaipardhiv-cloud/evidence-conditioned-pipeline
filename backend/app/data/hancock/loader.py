import json
import logging
from pathlib import Path
from typing import Dict, Any

from backend.app.data.hancock.config import (
    STRUCTURED_ZIP, TEXT_ZIP, STRUCTURED_EXTRACT_DIR, TEXT_EXTRACT_DIR,
    FILE_MANIFEST_PATH, INGESTION_REPORT_PATH
)
from backend.app.data.hancock.extractor import calculate_sha256, safe_extract_zip
from backend.app.data.hancock.validator import validate_json_schema, check_target_leakage
from backend.app.data.hancock.manifest import ManifestBuilder
from backend.app.data.hancock.schemas import FileManifestEntry
from backend.app.data.hancock.text_extractor import HancockTextExtractor

logger = logging.getLogger(__name__)

def ingest_hancock_dataset():
    logger.info("Starting HANCOCK dataset ingestion")
    
    files_info = {}
    
    for zip_file, extract_dir in [(STRUCTURED_ZIP, STRUCTURED_EXTRACT_DIR), (TEXT_ZIP, TEXT_EXTRACT_DIR)]:
        if zip_file.exists():
            size = zip_file.stat().st_size
            sha256 = calculate_sha256(zip_file)
            files_info[zip_file.name] = {"size": size, "sha256": sha256}
            
            logger.info(f"Extracting {zip_file.name}...")
            success, msg = safe_extract_zip(zip_file, extract_dir, overwrite=True)
            if not success:
                logger.error(f"Extraction failed for {zip_file.name}: {msg}")
                raise RuntimeError(f"Extraction failed: {msg}")
        else:
            logger.warning(f"File not found: {zip_file}")
            
    if not files_info:
        logger.warning("No HANCOCK dataset ZIP files found. Skipping ingestion.")
        return False
        
    builder = ManifestBuilder()
    
    for extract_dir in [STRUCTURED_EXTRACT_DIR, TEXT_EXTRACT_DIR]:
        if not extract_dir.exists():
            continue
            
        if extract_dir == TEXT_EXTRACT_DIR:
            text_extractor = HancockTextExtractor()
            text_records = text_extractor.process_directory(extract_dir)
            
            # Write text records out (you can change this to a processed folder if you have one, 
            # but usually it goes to processed layer)
            text_processed_path = FILE_MANIFEST_PATH.parent.parent / "processed" / "hancock" / "text_records.json"
            text_processed_path.parent.mkdir(parents=True, exist_ok=True)
            with open(text_processed_path, "w", encoding="utf-8") as f:
                json.dump(text_records, f, indent=2)
            
            builder.text_stats = text_extractor.stats
            builder.text_stats["unmapped_text_files"] = len(text_extractor.unmapped_files)
            builder.validation_errors.extend(text_extractor.errors)
            
            for rec in text_records:
                builder.add_text_record(rec)
            
            continue
            
        for json_path in extract_dir.rglob("*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                builder.validation_errors.append(f"Malformed JSON in {json_path.name}")
                continue

            if isinstance(data, list):
                modality = validate_json_schema(data, json_path.name)
                has_leakage, leak_msg = False, ""
                for obs in data:
                    leak, lmsg = check_target_leakage(obs)
                    if leak:
                        has_leakage, leak_msg = True, lmsg
                        break
            else:
                modality = validate_json_schema(data, json_path.name)
                has_leakage, leak_msg = check_target_leakage(data)
                
            if has_leakage:
                builder.validation_warnings.append(f"{json_path.name}: {leak_msg}")
                
            if modality == "unknown":
                builder.validation_warnings.append(f"Unknown schema for {json_path.name}")
                continue
                
            if isinstance(data, list):
                unique_patients = set(obs.get('patient_id') for obs in data if isinstance(obs, dict) and obs.get('patient_id'))
                if not unique_patients:
                    builder.validation_errors.append(f"Missing patient_id in {json_path.name}")
                for pid in unique_patients:
                    builder.add_patient_modality(pid, modality, json_path.name)
            else:
                patient_id = data.get('patient_id') if isinstance(data, dict) else None
                builder.add_patient_modality(patient_id, modality, json_path.name)
            
            file_sha256 = calculate_sha256(json_path)
            builder.file_manifest.append(FileManifestEntry(
                relative_path=str(json_path.relative_to(extract_dir.parent.parent)),
                file_type="json",
                size_bytes=json_path.stat().st_size,
                sha256=file_sha256,
                modality=modality,
                status="validated"
            ))

    FILE_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    with open(FILE_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump([entry.model_dump() for entry in builder.file_manifest], f, indent=2)
        
    report = builder.generate_report("HANCOCK", "TCIA/HANCOTHON", files_info)
    with open(INGESTION_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
        
    logger.info("HANCOCK ingestion completed successfully.")
    return True
