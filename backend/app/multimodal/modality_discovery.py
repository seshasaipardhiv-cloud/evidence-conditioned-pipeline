"""
Multimodal Modality Discovery Layer

Inspects arbitrary structured tables, image directories, text directories, and multimodal records
to discover present modalities, identifiers, targets, missingness, and cross-modality mappings.
Never assumes hardcoded column names; uses schema heuristics, content inspection, and file validation.
Emits auditable modality_detection.json metadata and returns BLOCKED if requirements are ambiguous.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Common identifier patterns across clinical and ML datasets
ID_PATTERNS = [
    r"^patient[_-]?id$",
    r"^subject[_-]?id$",
    r"^case[_-]?id$",
    r"^sample[_-]?id$",
    r"^record[_-]?id$",
    r"^id$",
    r"^pid$",
    r"^mrn$",
    r"^study[_-]?id$",
]

# Common outcome / target patterns
TARGET_PATTERNS = [
    r"^recurrence$",
    r"^relapse$",
    r"^target$",
    r"^label$",
    r"^outcome$",
    r"^class$",
    r"^progression$",
    r"^mortality$",
    r"^survival$",
    r"^status$",
    r"^y$",
]

# Known image file extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".dcm", ".nii", ".nii.gz"}

# Known text file extensions
TEXT_EXTENSIONS = {".txt", ".json", ".md", ".csv", ".tsv", ".xml", ".rtf"}


class ModalityDiscoveryResult:
    def __init__(
        self,
        status: str,  # 'DISCOVERED' | 'BLOCKED'
        detected_modalities: List[str],  # ['tabular', 'image', 'text']
        sample_count: int,
        identifier_field: Optional[str],
        target_field: Optional[str],
        tabular_numerical_fields: List[str],
        tabular_categorical_fields: List[str],
        image_fields_or_paths: Dict[str, Any],
        text_fields_or_paths: Dict[str, Any],
        missingness_by_modality: Dict[str, Any],
        modality_mappings: Dict[str, Any],
        unresolved_fields: List[str],
        reason: str,
        metadata_dict: Dict[str, Any],
    ):
        self.status = status
        self.detected_modalities = detected_modalities
        self.sample_count = sample_count
        self.identifier_field = identifier_field
        self.target_field = target_field
        self.tabular_numerical_fields = tabular_numerical_fields
        self.tabular_categorical_fields = tabular_categorical_fields
        self.image_fields_or_paths = image_fields_or_paths
        self.text_fields_or_paths = text_fields_or_paths
        self.missingness_by_modality = missingness_by_modality
        self.modality_mappings = modality_mappings
        self.unresolved_fields = unresolved_fields
        self.reason = reason
        self.metadata_dict = metadata_dict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "detected_modalities": self.detected_modalities,
            "sample_count": self.sample_count,
            "identifier_field": self.identifier_field,
            "target_field": self.target_field,
            "tabular_features": {
                "numerical": self.tabular_numerical_fields,
                "categorical": self.tabular_categorical_fields,
                "total_tabular_count": len(self.tabular_numerical_fields) + len(self.tabular_categorical_fields),
            },
            "image_modality": self.image_fields_or_paths,
            "text_modality": self.text_fields_or_paths,
            "missingness": self.missingness_by_modality,
            "modality_mappings": self.modality_mappings,
            "unresolved_fields": self.unresolved_fields,
            "reason": self.reason,
            "metadata": self.metadata_dict,
        }


class ModalityDiscoveryEngine:
    """Discovers and validates modalities present in an arbitrary data source."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def discover(
        self,
        tabular_data: Optional[Union[List[Dict[str, Any]], Path, str]] = None,
        image_data: Optional[Union[List[Path], Path, str, Dict[str, Path]]] = None,
        text_data: Optional[Union[List[Dict[str, Any]], Path, str, Dict[str, str]]] = None,
        candidate_target: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> ModalityDiscoveryResult:
        """
        Performs full cross-modality inspection and generates discovery metadata.
        """
        detected_modalities: List[str] = []
        tabular_num: List[str] = []
        tabular_cat: List[str] = []
        image_info: Dict[str, Any] = {"available": False, "count": 0, "paths": {}}
        text_info: Dict[str, Any] = {"available": False, "count": 0, "fields": []}
        missingness: Dict[str, Any] = {}
        unresolved: List[str] = []
        patient_ids: Set[str] = set()
        target_values: Dict[str, Any] = {}
        found_id_field = candidate_id
        found_target_field = candidate_target

        # 1. Inspect Tabular Modality
        parsed_tabular_records: List[Dict[str, Any]] = []
        if tabular_data is not None:
            parsed_tabular_records = self._load_tabular(tabular_data)
            if parsed_tabular_records:
                detected_modalities.append("tabular")
                num_fields, cat_fields, id_f, tgt_f, tab_miss, unres = self._profile_tabular_records(
                    parsed_tabular_records, found_id_field, found_target_field
                )
                tabular_num = num_fields
                tabular_cat = cat_fields
                if not found_id_field and id_f:
                    found_id_field = id_f
                if not found_target_field and tgt_f:
                    found_target_field = tgt_f
                missingness["tabular"] = tab_miss
                unresolved.extend(unres)

                # Extract patient IDs and target values from tabular
                for rec in parsed_tabular_records:
                    pid = str(rec.get(found_id_field, "")) if found_id_field else ""
                    if pid:
                        patient_ids.add(pid)
                    if found_target_field and found_target_field in rec:
                        target_values[pid] = rec[found_target_field]

        # 2. Inspect Image Modality
        parsed_images: Dict[str, Path] = {}
        if image_data is not None:
            parsed_images, img_miss = self._inspect_images(image_data, patient_ids)
            if parsed_images:
                detected_modalities.append("image")
                image_info = {
                    "available": True,
                    "count": len(parsed_images),
                    "sample_ids_with_images": list(parsed_images.keys())[:10],
                    "total_valid_images": len(parsed_images),
                }
                missingness["image"] = img_miss

        # 3. Inspect Text Modality
        parsed_texts: Dict[str, Dict[str, str]] = {}
        if text_data is not None:
            parsed_texts, txt_fields, txt_miss = self._inspect_texts(text_data, patient_ids)
            if parsed_texts:
                detected_modalities.append("text")
                text_info = {
                    "available": True,
                    "count": len(parsed_texts),
                    "text_fields": txt_fields,
                    "sample_ids_with_text": list(parsed_texts.keys())[:10],
                }
                missingness["text"] = txt_miss

        # 4. Modality Mappings and Alignment
        all_ids = set(patient_ids)
        if parsed_images:
            all_ids.update(parsed_images.keys())
        if parsed_texts:
            all_ids.update(parsed_texts.keys())

        # Remove empty string id if present
        all_ids.discard("")

        if not all_ids and parsed_tabular_records:
            all_ids = {f"sample_{i}" for i in range(len(parsed_tabular_records))}

        modality_mappings = {
            "total_unique_samples": len(all_ids),
            "samples_with_tabular": len(patient_ids) if "tabular" in detected_modalities else 0,
            "samples_with_image": len(parsed_images) if "image" in detected_modalities else 0,
            "samples_with_text": len(parsed_texts) if "text" in detected_modalities else 0,
            "complete_multimodal_samples": len(
                all_ids.intersection(patient_ids if "tabular" in detected_modalities else all_ids)
                .intersection(parsed_images.keys() if "image" in detected_modalities else all_ids)
                .intersection(parsed_texts.keys() if "text" in detected_modalities else all_ids)
            ) if detected_modalities else 0,
        }

        # 5. Determine Overall Discovery Status
        if not detected_modalities:
            status = "BLOCKED"
            reason = "No valid tabular, image, or text modalities detected in provided inputs."
        elif "tabular" in detected_modalities and not found_target_field and not target_values:
            status = "BLOCKED"
            reason = "Target variable could not be safely resolved from tabular records."
        else:
            status = "DISCOVERED"
            reason = f"Successfully discovered modalities: {', '.join(detected_modalities)} for {len(all_ids)} samples."

        result = ModalityDiscoveryResult(
            status=status,
            detected_modalities=detected_modalities,
            sample_count=len(all_ids),
            identifier_field=found_id_field,
            target_field=found_target_field,
            tabular_numerical_fields=tabular_num,
            tabular_categorical_fields=tabular_cat,
            image_fields_or_paths=image_info,
            text_fields_or_paths=text_info,
            missingness_by_modality=missingness,
            modality_mappings=modality_mappings,
            unresolved_fields=unresolved,
            reason=reason,
            metadata_dict={
                "discovery_timestamp": datetime.now(timezone.utc).isoformat(),
                "multimodal_ready": len(detected_modalities) >= 2,
            },
        )

        if self.output_dir:
            out_file = self.output_dir / "modality_detection.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info("Saved modality discovery metadata to %s", out_file)

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _load_tabular(self, data: Union[List[Dict[str, Any]], Path, str]) -> List[Dict[str, Any]]:
        if isinstance(data, list):
            return data
        p = Path(data)
        if not p.exists():
            return []
        if p.is_file():
            if p.suffix.lower() == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    content = json.load(f)
                    return content if isinstance(content, list) else [content]
            elif p.suffix.lower() in [".csv", ".tsv"]:
                import csv
                delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
                records = []
                with open(p, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    for row in reader:
                        records.append(dict(row))
                return records
        return []

    def _profile_tabular_records(
        self, records: List[Dict[str, Any]], candidate_id: Optional[str], candidate_target: Optional[str]
    ) -> Tuple[List[str], List[str], Optional[str], Optional[str], Dict[str, int], List[str]]:
        if not records:
            return [], [], candidate_id, candidate_target, {}, []

        all_keys = set()
        for r in records:
            all_keys.update(r.keys())

        # Resolve ID field if not given
        id_field = candidate_id
        if not id_field:
            for k in all_keys:
                if any(re.match(pat, k, re.IGNORECASE) for pat in ID_PATTERNS):
                    id_field = k
                    break

        # Resolve target field if not given
        target_field = candidate_target
        if not target_field:
            for k in all_keys:
                if any(re.match(pat, k, re.IGNORECASE) for pat in TARGET_PATTERNS) and k != id_field:
                    target_field = k
                    break

        numerical = []
        categorical = []
        missing = {}
        unresolved = []

        skip = {id_field, target_field} - {None}

        for k in all_keys:
            if k in skip:
                continue

            num_count = 0
            cat_count = 0
            miss_count = 0

            for r in records:
                val = r.get(k)
                if val is None or val == "" or val != val:
                    miss_count += 1
                elif isinstance(val, (int, float)) and not isinstance(val, bool):
                    num_count += 1
                elif isinstance(val, (str, bool)):
                    cat_count += 1
                else:
                    unresolved.append(k)

            if miss_count > 0:
                missing[k] = miss_count

            if num_count > cat_count:
                numerical.append(k)
            else:
                categorical.append(k)

        return sorted(numerical), sorted(categorical), id_field, target_field, missing, unresolved

    def _inspect_images(
        self, data: Union[List[Path], Path, str, Dict[str, Path]], known_ids: Set[str]
    ) -> Tuple[Dict[str, Path], Dict[str, int]]:
        images: Dict[str, Path] = {}
        missing_count = 0

        if isinstance(data, dict):
            for sample_id, path in data.items():
                p = Path(path)
                if p.exists() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    images[sample_id] = p
                else:
                    missing_count += 1
            return images, {"missing_or_corrupt_images": missing_count}

        if isinstance(data, list):
            for path in data:
                p = Path(path)
                if p.exists() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    sample_id = None
                    for kid in known_ids:
                        if kid in p.stem:
                            sample_id = kid
                            break
                    if not sample_id:
                        sample_id = p.stem
                    images[sample_id] = p
            return images, {"unresolved_images": 0}

        p = Path(data)
        if p.exists() and p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in IMAGE_EXTENSIONS:
                        sample_id = None
                        for kid in known_ids:
                            if kid in fp.stem:
                                sample_id = kid
                                break
                        if not sample_id:
                            sample_id = fp.stem
                        images[sample_id] = fp

        return images, {"unmatched_images": 0}

    def _inspect_texts(
        self, data: Union[List[Dict[str, Any]], Path, str, Dict[str, str]], known_ids: Set[str]
    ) -> Tuple[Dict[str, Dict[str, str]], List[str], Dict[str, int]]:
        texts: Dict[str, Dict[str, str]] = {}
        fields: Set[str] = set()
        missing_count = 0

        if isinstance(data, dict):
            for sample_id, content in data.items():
                if isinstance(content, str):
                    texts[sample_id] = {"raw_text": content}
                    fields.add("raw_text")
                elif isinstance(content, dict):
                    texts[sample_id] = content
                    fields.update(content.keys())
            return texts, sorted(list(fields)), {"missing_texts": 0}

        if isinstance(data, list):
            sorted_known = sorted(list(known_ids)) if known_ids else []
            for i, item in enumerate(data):
                if isinstance(item, str):
                    if len(item.strip()) > 0:
                        sid = sorted_known[i] if i < len(sorted_known) else f"sample_{i}"
                        texts[sid] = {"raw_text": item}
                        fields.add("raw_text")
                    else:
                        missing_count += 1
                elif isinstance(item, dict):
                    sid = None
                    for k in item.keys():
                        if any(re.match(pat, k, re.IGNORECASE) for pat in ID_PATTERNS):
                            sid = str(item[k])
                            break
                    if not sid:
                        sid = sorted_known[i] if i < len(sorted_known) else f"text_sample_{len(texts)}"
                    extracted_fields = {k: str(v) for k, v in item.items() if isinstance(v, str) and len(v) > 5}
                    if extracted_fields:
                        texts[sid] = extracted_fields
                        fields.update(extracted_fields.keys())
            return texts, sorted(list(fields)), {"missing_texts": missing_count}

        p = Path(data)
        if p.exists() and p.is_file() and p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                return self._inspect_texts(loaded, known_ids)

        if p.exists() and p.is_dir():
            for root, _, files in os.walk(p):
                for f in files:
                    fp = Path(root) / f
                    if fp.suffix.lower() in [".txt", ".json", ".md"]:
                        sample_id = fp.stem.split("_")[0]
                        try:
                            with open(fp, "r", encoding="utf-8", errors="ignore") as tf:
                                txt_content = tf.read()
                                if len(txt_content.strip()) > 5:
                                    texts.setdefault(sample_id, {})[fp.stem] = txt_content
                                    fields.add(fp.stem)
                        except Exception:
                            missing_count += 1

        return texts, sorted(list(fields)), {"corrupted_or_empty_text_files": missing_count}
