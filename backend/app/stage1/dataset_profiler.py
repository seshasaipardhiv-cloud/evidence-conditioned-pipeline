"""
Stage 1 — Dataset Profiler

Loads and profiles HANCOCK modality data from the validated ingestion outputs.
Produces ONLY aggregate statistics. Never stores or returns raw clinical text.
"""
from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.stage1.models import (
    BloodProfile,
    ClinicalProfile,
    DatasetProfile,
    PathologyProfile,
    Provenance,
    SourceType,
    ConfidenceLevel,
    TextProfile,
)

logger = logging.getLogger(__name__)

_PROFILER_PROV = Provenance(
    source_type=SourceType.dataset,
    source_reference="ingestion_report + extracted json files",
    extraction_method="schema_based_profiler",
    confidence=ConfidenceLevel.explicit,
)

_SUSPECT_ID_PATTERNS = {"id", "patient", "subject", "case", "code", "number", "pid"}


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_boolean(value: Any) -> bool:
    return isinstance(value, bool)


def _classify_fields(records: List[Dict[str, Any]], skip_fields: set) -> Dict[str, List[str]]:
    """Classify field names into numerical, categorical, and boolean lists."""
    type_map: Dict[str, str] = {}
    for record in records:
        for key, val in record.items():
            if key in skip_fields:
                continue
            if key in type_map:
                continue
            if _is_boolean(val):
                type_map[key] = "boolean"
            elif _is_numeric(val):
                type_map[key] = "numerical"
            elif val is None:
                type_map[key] = "unknown"
            else:
                type_map[key] = "categorical"

    numerical = [k for k, v in type_map.items() if v == "numerical"]
    categorical = [k for k, v in type_map.items() if v == "categorical"]
    boolean = [k for k, v in type_map.items() if v == "boolean"]
    return {"numerical": numerical, "categorical": categorical, "boolean": boolean}


def _count_missing(records: List[Dict[str, Any]], skip_fields: set) -> Dict[str, int]:
    missing: Dict[str, int] = {}
    for record in records:
        for key, val in record.items():
            if key in skip_fields:
                continue
            if val is None or val == "" or val != val:  # nan check
                missing[key] = missing.get(key, 0) + 1
    return {k: v for k, v in missing.items() if v > 0}


def _count_unique(records: List[Dict[str, Any]], skip_fields: set) -> Dict[str, int]:
    seen: Dict[str, set] = {}
    for record in records:
        for key, val in record.items():
            if key in skip_fields:
                continue
            seen.setdefault(key, set()).add(str(val))
    return {k: len(v) for k, v in seen.items()}


def _detect_constant_fields(records: List[Dict[str, Any]], unique_counts: Dict[str, int]) -> List[str]:
    return [k for k, cnt in unique_counts.items() if cnt == 1 and len(records) > 1]


def _detect_suspicious_id_fields(fields: List[str]) -> List[str]:
    suspicious = []
    for f in fields:
        lower = f.lower()
        if any(pat in lower for pat in _SUSPECT_ID_PATTERNS) and lower != "patient_id":
            suspicious.append(f)
    return suspicious


def _text_stats(texts: List[str]) -> Dict[str, float]:
    if not texts:
        return {}
    char_counts = [len(t) for t in texts]
    word_counts = [len(t.split()) for t in texts]
    return {
        "char_min": float(min(char_counts)),
        "char_max": float(max(char_counts)),
        "char_mean": statistics.mean(char_counts),
        "char_median": statistics.median(char_counts),
        "word_min": float(min(word_counts)),
        "word_max": float(max(word_counts)),
        "word_mean": statistics.mean(word_counts),
        "word_median": statistics.median(word_counts),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-modality profilers
# ──────────────────────────────────────────────────────────────────────────────

def profile_clinical(records: List[Dict[str, Any]]) -> ClinicalProfile:
    skip = {"patient_id"}
    fields = _classify_fields(records, skip)
    unique_counts = _count_unique(records, skip)
    all_non_id_fields = fields["numerical"] + fields["categorical"] + fields["boolean"]
    return ClinicalProfile(
        patient_count=len({r.get("patient_id") for r in records if r.get("patient_id")}),
        record_count=len(records),
        numerical_fields=fields["numerical"],
        categorical_fields=fields["categorical"],
        boolean_fields=fields["boolean"],
        missing_value_counts=_count_missing(records, skip),
        unique_value_counts=unique_counts,
        constant_fields=_detect_constant_fields(records, unique_counts),
        suspicious_identifier_fields=_detect_suspicious_id_fields(all_non_id_fields),
        provenance=_PROFILER_PROV,
    )


def profile_pathology(records: List[Dict[str, Any]]) -> PathologyProfile:
    skip = {"patient_id"}
    fields = _classify_fields(records, skip)
    unique_counts = _count_unique(records, skip)
    return PathologyProfile(
        patient_count=len({r.get("patient_id") for r in records if r.get("patient_id")}),
        record_count=len(records),
        categorical_fields=fields["categorical"],
        numerical_fields=fields["numerical"],
        missing_value_counts=_count_missing(records, skip),
        unique_value_counts=unique_counts,
        provenance=_PROFILER_PROV,
    )


def profile_blood(observations: List[Dict[str, Any]]) -> BloodProfile:
    patients = {obs.get("patient_id") for obs in observations if obs.get("patient_id")}
    analytes = {obs.get("analyte_name") for obs in observations if obs.get("analyte_name")}
    units = {obs.get("unit") for obs in observations if obs.get("unit")}

    missing_count = sum(
        1 for obs in observations
        if obs.get("value") is None
    )

    # Detect repeated measurements: same patient + same analyte > 1 time
    from collections import Counter
    combo_counts = Counter(
        (obs.get("patient_id"), obs.get("analyte_name"))
        for obs in observations
    )
    repeated = any(v > 1 for v in combo_counts.values())

    temporal = any(obs.get("days_before_first_treatment") is not None for obs in observations)

    return BloodProfile(
        patient_count=len(patients),
        observation_count=len(observations),
        analyte_count=len(analytes),
        analyte_names=sorted(a for a in analytes if a),
        units_observed=sorted(u for u in units if u),
        missing_value_count=missing_count,
        repeated_measurements_detected=repeated,
        temporal_available=temporal,
        provenance=_PROFILER_PROV,
    )


def profile_text(records: List[Dict[str, Any]]) -> TextProfile:
    """
    Profile text modality. Stores only aggregate statistics.
    Raw text fields (history, report, description) are NEVER stored or returned.
    """
    history_lengths: List[str] = []
    report_lengths: List[str] = []
    desc_lengths: List[str] = []
    empty_count = 0
    total = len(records)

    history_count = 0
    report_count = 0
    desc_count = 0

    for record in records:
        has_any = False
        
        # history might be a list of dicts (new format) or a string (old format)
        h_data = record.get("history")
        r_data = record.get("report")
        d_data = record.get("description")
        
        if isinstance(h_data, list):
            h = " ".join(item.get("text", "") for item in h_data)
        else:
            h = h_data or ""
            
        if isinstance(r_data, list):
            r = " ".join(item.get("text", "") for item in r_data)
        else:
            r = r_data or ""
            
        if isinstance(d_data, list):
            d = " ".join(item.get("text", "") for item in d_data)
        else:
            d = d_data or ""

        if h.strip():
            history_lengths.append(h)
            history_count += 1
            has_any = True
        if r.strip():
            report_lengths.append(r)
            report_count += 1
            has_any = True
        if d.strip():
            desc_lengths.append(d)
            desc_count += 1
            has_any = True

        if not has_any:
            empty_count += 1

    def split_stats(texts: List[str]) -> tuple:
        if not texts:
            return {}, {}
        chars = [len(t) for t in texts]
        words = [len(t.split()) for t in texts]
        char_stats = {
            "min": float(min(chars)), "max": float(max(chars)),
            "mean": statistics.mean(chars), "median": statistics.median(chars),
        }
        word_stats = {
            "min": float(min(words)), "max": float(max(words)),
            "mean": statistics.mean(words), "median": statistics.median(words),
        }
        return char_stats, word_stats

    h_char, h_word = split_stats(history_lengths)
    r_char, r_word = split_stats(report_lengths)
    d_char, d_word = split_stats(desc_lengths)

    return TextProfile(
        patient_count=total,
        history_available_count=history_count,
        report_available_count=report_count,
        description_available_count=desc_count,
        history_char_count_stats=h_char,
        report_char_count_stats=r_char,
        description_char_count_stats=d_char,
        history_word_count_stats=h_word,
        report_word_count_stats=r_word,
        description_word_count_stats=d_word,
        empty_text_rate=empty_count / total if total > 0 else 0.0,
        provenance=_PROFILER_PROV,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrated dataset profiler
# ──────────────────────────────────────────────────────────────────────────────

def profile_dataset_from_files(
    structured_dir: Path,
    text_dir: Path,
    ingestion_report_path: Path,
) -> DatasetProfile:
    """
    Load validated JSON files from the ingestion outputs and profile each modality.
    Falls back gracefully when files are absent.
    """
    clinical_records: List[Dict] = []
    pathology_records: List[Dict] = []
    blood_observations: List[Dict] = []
    text_records: List[Dict] = []

    # Load ingestion report for patient counts
    total_patients = 0
    if ingestion_report_path.exists():
        with open(ingestion_report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)
        total_patients = report_data.get("number_of_patients", 0)

    def safe_load_json(path: Path) -> Any:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {path.name}: {e}")
            return None

    # Scan structured directory
    if structured_dir.exists():
        for json_file in structured_dir.rglob("*.json"):
            data = safe_load_json(json_file)
            if data is None:
                continue
            name = json_file.name.lower()
            if isinstance(data, list):
                if "blood" in name or "lab" in name:
                    blood_observations.extend([r for r in data if isinstance(r, dict)])
                elif "clin" in name or "clinical" in name:
                    clinical_records.extend([r for r in data if isinstance(r, dict)])
                elif "patho" in name:
                    pathology_records.extend([r for r in data if isinstance(r, dict)])
            elif isinstance(data, dict):
                if "blood" in name or "lab" in name:
                    blood_observations.append(data)
                elif "clin" in name or "clinical" in name:
                    clinical_records.append(data)
                elif "patho" in name:
                    pathology_records.append(data)

    # Load processed text records if available
    # Assuming structured_dir is data/raw/hancock/structured, so processed is data/processed/hancock
    processed_text_path = structured_dir.parent.parent.parent / "processed" / "hancock" / "text_records.json"
    if processed_text_path.exists():
        data = safe_load_json(processed_text_path)
        if data and isinstance(data, list):
            text_records.extend([r for r in data if isinstance(r, dict)])
    elif text_dir.exists():
        for json_file in text_dir.rglob("*.json"):
            data = safe_load_json(json_file)
            if data is None:
                continue
            if isinstance(data, list):
                text_records.extend([r for r in data if isinstance(r, dict)])
            elif isinstance(data, dict) and ("history" in data or "report" in data):
                text_records.append(data)

    return DatasetProfile(
        dataset_name="HANCOCK",
        total_patients=total_patients,
        clinical=profile_clinical(clinical_records) if clinical_records else None,
        pathology=profile_pathology(pathology_records) if pathology_records else None,
        blood=profile_blood(blood_observations) if blood_observations else None,
        text=profile_text(text_records) if text_records else None,
        provenance=_PROFILER_PROV,
    )
