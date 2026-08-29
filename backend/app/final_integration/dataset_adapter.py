"""
dataset_adapter.py

Stage 2D Dataset Auto-Discovery & Adaptation Engine

Accepts unseen clinical datasets and automatically infers:
  - Modalities present (Tabular, Image, Text, Trimodal)
  - Target variable and task type (binary classification)
  - Missingness and class imbalance
  - Patient/sample identifiers
  - Numerical and categorical feature splits.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DatasetAdapter:
    """
    Auto-discovers schema, modalities, and characteristics of unseen datasets.
    """

    ID_PATTERNS = [r"id\b", r"patient", r"record_id", r"subject", r"case_id", r"sample_id"]
    TARGET_PATTERNS = [r"target", r"label", r"outcome", r"event", r"malignancy", r"recurrence", r"progression", r"dysplasia", r"status"]

    def adapt_dataset(
        self,
        raw_records: Union[List[Dict[str, Any]], Dict[str, Any]],
        target_override: Optional[str] = None,
        id_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Infers modalities, extracts features/targets/identifiers, and computes cohort profile.
        """
        if isinstance(raw_records, dict) and "records" in raw_records:
            records = raw_records["records"]
        elif isinstance(raw_records, list):
            records = raw_records
        else:
            raise ValueError("Invalid dataset input: must be list of dicts or dict with 'records' key.")

        n_samples = len(records)
        if n_samples == 0:
            raise ValueError("Dataset is empty.")

        first_rec = records[0]
        keys = list(first_rec.keys())

        # 1. Identify ID column
        id_col = id_override
        if not id_col:
            for k in keys:
                if any(re.search(pat, k, re.I) for pat in self.ID_PATTERNS):
                    id_col = k
                    break
        if not id_col:
            id_col = "sample_index"

        # 2. Identify Target column
        target_col = target_override
        if not target_col:
            for k in keys:
                if k != id_col and any(re.search(pat, k, re.I) for pat in self.TARGET_PATTERNS):
                    target_col = k
                    break
        if not target_col:
            # Default to the first binary/integer column
            for k in keys:
                if k != id_col:
                    vals = [r[k] for r in records if r.get(k) is not None]
                    if all(v in [0, 1, True, False] for v in vals):
                        target_col = k
                        break

        if not target_col:
            target_col = keys[-1]

        # 3. Discover Modalities
        has_tabular = False
        has_image = False
        has_text = False

        tabular_cols = []
        image_col = None
        text_col = None

        for k in keys:
            if k in [id_col, target_col]:
                continue
            val = first_rec.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                tabular_cols.append(k)
                has_tabular = True
            elif isinstance(val, str):
                if any(ext in val.lower() for ext in [".png", ".jpg", ".jpeg", ".tiff", ".dcm"]):
                    image_col = k
                    has_image = True
                elif len(val.split()) > 3:
                    text_col = k
                    has_text = True
                else:
                    # Categorical feature
                    tabular_cols.append(k)
                    has_tabular = True

        discovered_modalities = []
        if has_tabular:
            discovered_modalities.append("tabular")
        if has_image:
            discovered_modalities.append("image")
        if has_text:
            discovered_modalities.append("text")

        # 4. Extract samples, targets, and features
        sample_ids = [str(r.get(id_col, f"S_{i:04d}")) for i, r in enumerate(records)]
        targets = np.array([int(r.get(target_col, 0)) for r in records], dtype=int)

        # Tabular matrix
        X_tab = None
        has_missing = False
        if has_tabular and tabular_cols:
            tab_matrix = []
            for r in records:
                row = []
                for col in tabular_cols:
                    v = r.get(col)
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        has_missing = True
                        row.append(0.0)
                    elif isinstance(v, (int, float)):
                        row.append(float(v))
                    else:
                        # Simple categorical hashing
                        row.append(float(hash(str(v)) % 100))
                tab_matrix.append(row)
            X_tab = np.array(tab_matrix, dtype=np.float32)

        # Image paths & Text notes
        image_paths = [str(r[image_col]) for r in records] if image_col else []
        text_notes = [str(r[text_col]) for r in records] if text_col else []

        # Class imbalance ratio
        pos_count = int(np.sum(targets == 1))
        neg_count = int(np.sum(targets == 0))
        imbalance_ratio = round(pos_count / max(1, neg_count), 3)
        has_imbalance = (imbalance_ratio < 0.40 or imbalance_ratio > 2.50)

        return {
            "discovered_modalities": discovered_modalities,
            "sample_count": n_samples,
            "id_column": id_col,
            "target_column": target_col,
            "sample_ids": sample_ids,
            "targets": targets,
            "tabular_features": X_tab,
            "tabular_feature_names": tabular_cols,
            "image_paths": image_paths,
            "text_notes": text_notes,
            "has_missing": has_missing,
            "has_imbalance": has_imbalance,
            "positive_cases": pos_count,
            "negative_cases": neg_count,
            "imbalance_ratio": imbalance_ratio,
        }
