"""
mcr_sl_adapter.py

MCR-SL (Multimodal Context-Rich Skin Lesion) Dataset Adapter & Manifest Generator.

Parses subject.xlsx, lesion.xlsx, and image.xlsx to create a canonical multimodal dataset.
Serializes structured pre-diagnostic clinical context into standardized text for PubMedBERT.
Implements strict subject-level group stratified splitting to guarantee zero patient/lesion leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

# Strict allowlist of SAFE pre-diagnostic clinical context fields
ALLOWLIST_SUBJECT_FIELDS = [
    "age", "sex", "height", "weight", "natural_hair_color",
    "skin_reaction_to_sun", "moles_body_18", "moles_bigger_5mm",
    "moles_bigger_20cm", "moles_body", "sunburn_number_group",
    "sunbed", "h_cancer", "h_skin_cancer", "h_skin_cancer_relatives",
    "organ_transplant", "immunosuppresion",
]

ALLOWLIST_LESION_FIELDS = [
    "location_group", "location", "diameter",
]

# Explicit blacklist of post-diagnostic / target-leaking fields
FORBIDDEN_LEAKING_FIELDS = [
    "malignancy", "lesion_diagnosis", "unified_diagnosis",
    "histopathology_diagnosis", "dermatology_diagnosis",
    "referral_diagnosis", "tumor_thickness", "procedure",
    "lesion_status_when_captured", "diagnosis_image_id",
]


class MCRSLDatasetAdapter:
    """
    Adapter for the real MCR-SL dataset.
    Constructs clean paired multimodal samples and manages subject-isolated splits.
    """

    def __init__(self, base_dir: str = "data/real/mcr_sl"):
        self.base_dir = Path(base_dir)
        self.dermoscopic_dir = self.base_dir / "dermoscopic"
        self.clinical_dir = self.base_dir / "clinical"

    def load_raw_tables(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_sub = pd.read_excel(self.base_dir / "subject.xlsx")
        df_les = pd.read_excel(self.base_dir / "lesion.xlsx")
        df_img = pd.read_excel(self.base_dir / "image.xlsx")
        return df_sub, df_les, df_img

    def serialize_clinical_context(self, row: pd.Series) -> str:
        """
        Serializes safe pre-diagnostic structured clinical context into text.
        Guarantees zero target-derived words or diagnostic labels appear in the text.
        """
        def _clean(k: str) -> Optional[str]:
            v = row.get(k)
            if pd.isna(v):
                return None
            s = str(v).strip()
            if s.lower() in ("unknown", "unk", "none", "nan", ""):
                return None
            return s

        parts = []

        # 1. Demographics & Body Characteristics
        demo_parts = []
        age = _clean("age")
        if age:
            demo_parts.append(f"Age {age}")
        sex = _clean("sex")
        if sex:
            demo_parts.append(f"Sex {sex}")
        height = _clean("height")
        if height:
            demo_parts.append(f"Height {height} cm")
        weight = _clean("weight")
        if weight:
            demo_parts.append(f"Weight {weight} kg")
        hair = _clean("natural_hair_color")
        if hair:
            demo_parts.append(f"Natural hair color {hair}")
        if demo_parts:
            parts.append(f"Patient Demographics: {', '.join(demo_parts)}.")

        # 2. Sun Exposure & Skin History
        sun_parts = []
        sun_react = _clean("skin_reaction_to_sun")
        if sun_react:
            sun_parts.append(f"Skin reaction to sun {sun_react}")
        moles_18 = _clean("moles_body_18")
        if moles_18:
            sun_parts.append(f"Moles at age 18: {moles_18}")
        moles_5 = _clean("moles_bigger_5mm")
        if moles_5:
            sun_parts.append(f"Moles larger than 5mm: {moles_5}")
        moles_20 = _clean("moles_bigger_20cm")
        if moles_20:
            sun_parts.append(f"Moles larger than 20cm: {moles_20}")
        moles_tot = _clean("moles_body")
        if moles_tot:
            sun_parts.append(f"Total body moles count: {moles_tot}")
        sunburn = _clean("sunburn_number_group")
        if sunburn:
            sun_parts.append(f"Sunburn episodes: {sunburn}")
        sunbed = _clean("sunbed")
        if sunbed:
            sun_parts.append(f"Sunbed exposure: {sunbed}")
        if sun_parts:
            parts.append(f"Sun & Skin Profile: {', '.join(sun_parts)}.")

        # 3. Medical & Family Risk History
        hist_parts = []
        h_cancer = _clean("h_cancer")
        if h_cancer:
            hist_parts.append(f"Personal history of cancer: {h_cancer}")
        h_skin_cancer = _clean("h_skin_cancer")
        if h_skin_cancer:
            hist_parts.append(f"Personal history of skin cancer: {h_skin_cancer}")
        h_rel = _clean("h_skin_cancer_relatives")
        if h_rel:
            hist_parts.append(f"Family history of skin cancer in relatives: {h_rel}")
        organ = _clean("organ_transplant")
        if organ:
            hist_parts.append(f"Organ transplant history: {organ}")
        immuno = _clean("immunosuppresion")
        if immuno:
            hist_parts.append(f"Immunosuppression status: {immuno}")
        if hist_parts:
            parts.append(f"Medical History: {', '.join(hist_parts)}.")

        # 4. Lesion Presentation
        les_parts = []
        loc_grp = _clean("location_group")
        if loc_grp:
            les_parts.append(f"Anatomical region {loc_grp}")
        loc = _clean("location")
        if loc:
            les_parts.append(f"Specific site {loc}")
        diam = _clean("diameter")
        if diam:
            les_parts.append(f"Clinical diameter {diam} mm")
        if les_parts:
            parts.append(f"Lesion Presentation: {', '.join(les_parts)}.")

        return " ".join(parts)

    def build_manifest(self, seeds: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Builds the canonical paired manifest with subject-level stratified splits.
        """
        seeds = seeds or [42, 100, 2026]
        df_sub, df_les, df_img = self.load_raw_tables()

        # Merge subject metadata into lesion table
        df_merged = pd.merge(df_les, df_sub, on="subject_id", how="left", suffixes=("", "_sub"))

        # Filter out unknown malignancy labels
        df_valid = df_merged[df_merged["malignancy"].isin(["Malignant", "Non-malignant"])].copy()

        # Define binary target: 1 = Malignant, 0 = Non-malignant
        df_valid["target"] = (df_valid["malignancy"] == "Malignant").astype(int)

        # Map primary image path for each lesion
        dermo_files = {f.stem: f for f in self.dermoscopic_dir.glob("*.png")}
        clin_files = {f.stem: f for f in self.clinical_dir.glob("*.png")}

        image_paths = []
        image_ids = []
        image_types = []

        for _, row in df_valid.iterrows():
            diag_img_id = str(row.get("diagnosis_image_id", ""))
            les_id = str(row["lesion_id"])

            if diag_img_id in dermo_files:
                image_ids.append(diag_img_id)
                image_paths.append(str(dermo_files[diag_img_id].resolve()))
                image_types.append("dermoscopy")
            elif diag_img_id in clin_files:
                image_ids.append(diag_img_id)
                image_paths.append(str(clin_files[diag_img_id].resolve()))
                image_types.append("clinical")
            else:
                # Find any dermoscopic image for this lesion
                les_imgs = df_img[df_img["lesion_id"] == les_id]
                dermo_matches = [img_id for img_id in les_imgs["image_id"] if img_id in dermo_files]
                if dermo_matches:
                    chosen = dermo_matches[0]
                    image_ids.append(chosen)
                    image_paths.append(str(dermo_files[chosen].resolve()))
                    image_types.append("dermoscopy")
                else:
                    clin_matches = [img_id for img_id in les_imgs["image_id"] if img_id in clin_files]
                    if clin_matches:
                        chosen = clin_matches[0]
                        image_ids.append(chosen)
                        image_paths.append(str(clin_files[chosen].resolve()))
                        image_types.append("clinical")
                    else:
                        image_ids.append("MISSING")
                        image_paths.append("MISSING")
                        image_types.append("MISSING")

        df_valid["image_id"] = image_ids
        df_valid["image_path"] = image_paths
        df_valid["image_type"] = image_types

        # Filter out any rows where image was missing
        df_valid = df_valid[df_valid["image_path"] != "MISSING"].copy()

        # Serialize structured clinical context
        clinical_texts = [self.serialize_clinical_context(row) for _, row in df_valid.iterrows()]
        df_valid["clinical_text"] = clinical_texts

        # Perform Subject-Level Stratified Group Splitting across all seeds
        # We assign 70% Train, 30% Test at the SUBJECT level
        for seed in seeds:
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            # Use fold 0 and 1 as test (40%) or fold 0 as test (20%) + fold 1 as val (20%) + rest train (60%)
            # Standard 70% train, 30% test split by subject:
            # We assign fold 0 and 1 (approx 30% subjects) as test, folds 2,3,4 as train
            splits_col = [None] * len(df_valid)
            subjects = df_valid["subject_id"].values
            y = df_valid["target"].values
            groups = subjects

            for fold, (train_idx, test_idx) in enumerate(sgkf.split(df_valid, y, groups)):
                if fold in (0, 1):
                    for idx in test_idx:
                        splits_col[idx] = "test"
                else:
                    for idx in train_idx:
                        if splits_col[idx] is None:
                            splits_col[idx] = "train"

            # Fill any remaining None with train
            splits_col = ["train" if s is None else s for s in splits_col]
            df_valid[f"split_seed_{seed}"] = splits_col

        # Default split is seed 42
        df_valid["split"] = df_valid["split_seed_42"]

        # Select canonical columns for manifest
        manifest_cols = [
            "subject_id", "lesion_id", "image_id", "image_path", "image_type",
            "clinical_text", "target", "malignancy", "location_group", "diameter",
            "split",
        ] + [f"split_seed_{s}" for s in seeds]

        manifest_df = df_valid[manifest_cols].copy()
        manifest_path = self.base_dir / "mcr_sl_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"MCR-SL canonical manifest saved to {manifest_path} ({len(manifest_df)} rows).")

        return manifest_df

    def get_dataset_for_evaluation(self, seed: int = 42) -> Dict[str, Any]:
        """
        Returns the formatted dataset dictionary ready for CohortBenchmarkEvaluator.
        """
        manifest_path = self.base_dir / "mcr_sl_manifest.csv"
        if not manifest_path.exists():
            df = self.build_manifest(seeds=[42, 100, 2026])
        else:
            df = pd.read_csv(manifest_path)

        records = []
        for _, row in df.iterrows():
            records.append({
                "subject_id": str(row["subject_id"]),
                "lesion_id": str(row["lesion_id"]),
                "image_id": str(row["image_id"]),
                "image_path": str(row["image_path"]),
                "image_file": str(row["image_path"]),
                "clinical_text": str(row["clinical_text"]),
                "biopsy_report": str(row["clinical_text"]),  # Alias for text modality adapter
                "clinical_narrative": str(row["clinical_text"]),
                "mcr_sl_malignancy": int(row["target"]),
                "malignancy_flag": int(row["target"]),  # Alias for target adapter
                "target": int(row["target"]),
                "split": str(row.get(f"split_seed_{seed}", row.get("split", "train"))),
            })

        return {
            "records": records,
            "dataset_status": "REAL_MCR_SL_MULTIMODAL_EXPERIMENT",
            "description": (
                "Real MCR-SL (Multimodal, Context-Rich Skin Lesion) Dataset. "
                "Paired real dermoscopic images + serialized pre-diagnostic structured clinical context. "
                "Binary target: mcr_sl_malignancy (0 = Non-malignant, 1 = Malignant). "
                "Subject-level stratified group splitting ensures 0 patient and 0 lesion overlap."
            ),
        }
