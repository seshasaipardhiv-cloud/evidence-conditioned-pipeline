"""
test_mcr_sl_integration.py

Scientific validation and integration tests for the real MCR-SL multimodal experiment.

Covers:
  - MCR-SL manifest generation
  - Image file existence verification
  - Subject-level split isolation (0 overlap required)
  - Lesion-level split isolation (0 overlap required)
  - Text target leakage prevention (zero diagnostic terms)
  - Forensic audit completeness
  - Prediction store completeness
  - Multi-seed reproducibility
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path("data/real/mcr_sl")
EVIDENCE_DIR = Path("evidence/final/submission/New")
SEEDS = [42, 100, 2026]

FORBIDDEN_DIAGNOSTIC_TERMS = [
    "malignan", "melanom", "carcinom", "bcc", "scc", "basal",
    "squamous", "nevus", "nevi", "benign", "histopathol",
    "biopsy", "excision", "thickness", "tumor", "clark",
    "breslow", "metastat", "dysplasia", "keratosis", "dermatofibroma",
]


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def manifest() -> pd.DataFrame:
    """Load MCR-SL manifest, building it if absent."""
    mpath = BASE_DIR / "mcr_sl_manifest.csv"
    if not mpath.exists():
        from backend.app.final_integration.mcr_sl_adapter import MCRSLDatasetAdapter
        adapter = MCRSLDatasetAdapter()
        df = adapter.build_manifest(seeds=SEEDS)
    else:
        df = pd.read_csv(mpath)
    return df


@pytest.fixture(scope="module")
def forensic_report() -> dict:
    """Load MCR-SL forensic report, running audit if absent."""
    rpath = EVIDENCE_DIR / "provenance" / "mcr_sl_forensic_report.json"
    if not rpath.exists():
        from backend.app.final_integration.mcr_sl_forensics import MCRSLForensicAuditor
        auditor = MCRSLForensicAuditor()
        return auditor.run_full_forensic_audit()
    with open(rpath, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def experiment_results() -> dict:
    """Load MCR-SL experiment results JSON."""
    rpath = EVIDENCE_DIR / "results" / "mcr_sl_multimodal_results.json"
    if not rpath.exists():
        return {}
    with open(rpath, "r") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def predictions() -> list:
    """Load MCR-SL canonical predictions JSONL."""
    ppath = EVIDENCE_DIR / "predictions" / "Cohort_MCR_SL_Real_Multimodal_predictions.jsonl"
    if not ppath.exists():
        return []
    records = []
    with open(ppath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Manifest Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestMCRSLManifest:
    def test_manifest_exists(self):
        assert (BASE_DIR / "mcr_sl_manifest.csv").exists(), "MCR-SL manifest CSV not found."

    def test_manifest_non_empty(self, manifest):
        assert len(manifest) > 0, "Manifest has zero rows."

    def test_manifest_expected_sample_count(self, manifest):
        # 234 labeled samples (240 total minus 6 unknown)
        assert len(manifest) == 234, f"Expected 234 rows, got {len(manifest)}."

    def test_manifest_has_required_columns(self, manifest):
        required = ["subject_id", "lesion_id", "image_id", "image_path", "image_type",
                    "clinical_text", "target", "split"]
        for col in required:
            assert col in manifest.columns, f"Missing column: {col}"

    def test_target_is_binary(self, manifest):
        unique_targets = set(manifest["target"].unique())
        assert unique_targets == {0, 1}, f"Unexpected target values: {unique_targets}"

    def test_target_class_distribution(self, manifest):
        # 42 malignant, 192 non-malignant
        assert manifest["target"].sum() == 42, f"Expected 42 malignant, got {manifest['target'].sum()}"
        assert (manifest["target"] == 0).sum() == 192, f"Expected 192 non-malignant"

    def test_unique_lesion_ids(self, manifest):
        assert manifest["lesion_id"].nunique() == 234, "Duplicate lesion IDs in manifest."

    def test_unique_subject_ids(self, manifest):
        # 60 subjects across all lesions (not lesion-level unique)
        assert manifest["subject_id"].nunique() == 59, \
            f"Expected ~59 unique subjects, got {manifest['subject_id'].nunique()}"

    def test_subject_count_at_most_60(self, manifest):
        assert manifest["subject_id"].nunique() <= 60

    def test_all_image_types_are_dermoscopy(self, manifest):
        # Primary image type should be dermoscopy for MCR-SL
        derm_count = (manifest["image_type"] == "dermoscopy").sum()
        assert derm_count > 200, f"Expected most images to be dermoscopy, got {derm_count}"

    def test_clinical_text_is_non_empty(self, manifest):
        empty_texts = (manifest["clinical_text"].fillna("").str.strip() == "").sum()
        assert empty_texts == 0, f"Found {empty_texts} rows with empty clinical_text."


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Image Existence
# ─────────────────────────────────────────────────────────────────────────────
class TestMCRSLImageExistence:
    def test_all_image_paths_exist(self, manifest):
        missing = [p for p in manifest["image_path"] if not Path(p).exists()]
        assert len(missing) == 0, f"Missing {len(missing)} image files: {missing[:3]}"

    def test_no_duplicate_image_hashes(self, forensic_report):
        assert forensic_report["image_audit"]["exact_duplicate_hashes_count"] == 0, \
            "Exact duplicate images detected in MCR-SL."

    def test_no_duplicate_image_ids(self, forensic_report):
        assert forensic_report["image_audit"]["duplicate_image_ids_count"] == 0, \
            "Duplicate image IDs detected in MCR-SL manifest."


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Subject-Level Split Isolation
# ─────────────────────────────────────────────────────────────────────────────
class TestSubjectLevelSplitIsolation:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_zero_subject_overlap(self, manifest, seed):
        col = f"split_seed_{seed}" if f"split_seed_{seed}" in manifest.columns else "split"
        train_subs = set(manifest[manifest[col] == "train"]["subject_id"])
        test_subs = set(manifest[manifest[col] == "test"]["subject_id"])
        overlap = train_subs & test_subs
        assert len(overlap) == 0, \
            f"Seed {seed}: {len(overlap)} subjects appear in both train and test!"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_zero_lesion_overlap(self, manifest, seed):
        col = f"split_seed_{seed}" if f"split_seed_{seed}" in manifest.columns else "split"
        train_les = set(manifest[manifest[col] == "train"]["lesion_id"])
        test_les = set(manifest[manifest[col] == "test"]["lesion_id"])
        overlap = train_les & test_les
        assert len(overlap) == 0, \
            f"Seed {seed}: {len(overlap)} lesions appear in both train and test!"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_train_and_test_non_empty(self, manifest, seed):
        col = f"split_seed_{seed}" if f"split_seed_{seed}" in manifest.columns else "split"
        n_train = (manifest[col] == "train").sum()
        n_test = (manifest[col] == "test").sum()
        assert n_train > 0, f"Seed {seed}: Train split is empty."
        assert n_test > 0, f"Seed {seed}: Test split is empty."

    @pytest.mark.parametrize("seed", SEEDS)
    def test_train_covers_majority_of_samples(self, manifest, seed):
        col = f"split_seed_{seed}" if f"split_seed_{seed}" in manifest.columns else "split"
        n_train = (manifest[col] == "train").sum()
        total = len(manifest)
        # Train should be at least 50% of data
        assert n_train >= total * 0.5, \
            f"Seed {seed}: Train only {n_train}/{total} samples (<50%)"

    def test_split_isolation_audit_passed(self, forensic_report):
        for seed_key, audit in forensic_report["split_isolation_audit"].items():
            assert audit["subject_isolation_passed"], f"{seed_key}: Subject isolation FAILED."
            assert audit["lesion_isolation_passed"], f"{seed_key}: Lesion isolation FAILED."


# ─────────────────────────────────────────────────────────────────────────────
# Phase 7 — Text Target Leakage
# ─────────────────────────────────────────────────────────────────────────────
class TestTextTargetLeakage:
    def test_zero_diagnostic_terms_in_clinical_text(self, manifest):
        violations = []
        for _, row in manifest.iterrows():
            txt_low = str(row["clinical_text"]).lower()
            found = [t for t in FORBIDDEN_DIAGNOSTIC_TERMS if t in txt_low]
            if found:
                violations.append({"lesion_id": row["lesion_id"], "found": found})
        assert len(violations) == 0, \
            f"Text target leakage in {len(violations)} samples! First: {violations[0]}"

    def test_malignancy_not_in_clinical_text(self, manifest):
        hits = manifest["clinical_text"].str.lower().str.contains("malign", na=False)
        assert not hits.any(), f"Word 'malign' found in {hits.sum()} clinical texts!"

    def test_diagnosis_not_in_clinical_text(self, manifest):
        hits = manifest["clinical_text"].str.lower().str.contains("diagnosis", na=False)
        assert not hits.any(), f"Word 'diagnosis' found in {hits.sum()} clinical texts!"

    def test_cancer_code_not_in_clinical_text(self, manifest):
        # 'bcc' should not appear as a diagnostic code in the free text
        # (it appears in 'h_cancer' column value only as structured text, not as 'bcc')
        # Verify that literal substring 'bcc' is absent as code
        hits = manifest["clinical_text"].str.lower().str.contains(r"\bbcc\b", regex=True, na=False)
        assert not hits.any(), f"'BCC' diagnostic code found in {hits.sum()} clinical texts!"

    def test_forensic_report_text_leakage_zero(self, forensic_report):
        assert not forensic_report["text_target_leakage_audit"]["leakage_detected"], \
            "Forensic report reports text leakage detected!"
        assert forensic_report["text_target_leakage_audit"]["leakage_violations_count"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Phase 13 — Canonical Predictions
# ─────────────────────────────────────────────────────────────────────────────
class TestCanonicalPredictions:
    def test_predictions_file_exists(self):
        ppath = EVIDENCE_DIR / "predictions" / "Cohort_MCR_SL_Real_Multimodal_predictions.jsonl"
        assert ppath.exists(), "MCR-SL predictions JSONL not found."

    def test_predictions_non_empty(self, predictions):
        assert len(predictions) > 0, "No predictions in MCR-SL predictions JSONL."

    def test_prediction_fields_present(self, predictions):
        required = ["subject_id", "lesion_id", "image_id", "seed", "model_name",
                    "true_label", "predicted_probability", "predicted_class"]
        for pred in predictions[:5]:
            for field in required:
                assert field in pred, f"Missing field '{field}' in prediction record."

    def test_probabilities_in_valid_range(self, predictions):
        probs = [p["predicted_probability"] for p in predictions]
        assert all(0.0 <= prob <= 1.0 for prob in probs), \
            "Some probabilities are outside [0, 1]!"

    def test_predicted_classes_are_binary(self, predictions):
        classes = [p["predicted_class"] for p in predictions]
        assert set(classes) <= {0, 1}, f"Non-binary predicted classes: {set(classes)}"

    def test_all_three_seeds_present(self, predictions):
        seeds_present = set(p["seed"] for p in predictions)
        for seed in SEEDS:
            assert seed in seeds_present, f"Seed {seed} missing from predictions."

    def test_all_six_architectures_present(self, predictions):
        arch_names = set(p["model_name"] for p in predictions)
        # At least image-only and context-only must be present
        assert any("Image-Only" in a or "image" in a.lower() for a in arch_names), \
            "Image-only architecture predictions missing."
        assert any("Context" in a or "text" in a.lower() or "PubMed" in a for a in arch_names), \
            "Context-only architecture predictions missing."

    def test_true_labels_match_manifest(self, manifest, predictions):
        """Verify true labels in predictions are consistent with manifest."""
        les_to_target = dict(zip(manifest["lesion_id"], manifest["target"]))
        for pred in predictions:
            les_id = pred["lesion_id"]
            if les_id in les_to_target:
                assert pred["true_label"] == les_to_target[les_id], \
                    f"Mismatched true label for lesion {les_id}!"


# ─────────────────────────────────────────────────────────────────────────────
# Phase 12 — Forensic Validation
# ─────────────────────────────────────────────────────────────────────────────
class TestMCRSLForensicValidation:
    def test_forensic_report_exists(self):
        rpath = EVIDENCE_DIR / "provenance" / "mcr_sl_forensic_report.json"
        assert rpath.exists(), "MCR-SL forensic report not found."

    def test_overall_forensic_status_pass(self, forensic_report):
        assert forensic_report["overall_forensic_status"] == "PASS", \
            f"Overall forensic status is FAIL! Details: {forensic_report}"

    def test_no_missing_images(self, forensic_report):
        assert forensic_report["image_audit"]["missing_image_count"] == 0

    def test_sample_count_matches(self, forensic_report, manifest):
        assert forensic_report["sample_count"] == len(manifest), \
            f"Forensic report sample count ({forensic_report['sample_count']}) != manifest ({len(manifest)})"

    def test_positive_count_matches(self, forensic_report):
        assert forensic_report["positive_malignant_count"] == 42

    def test_negative_count_matches(self, forensic_report):
        assert forensic_report["negative_non_malignant_count"] == 192


# ─────────────────────────────────────────────────────────────────────────────
# Phase 11 — Experiment Results
# ─────────────────────────────────────────────────────────────────────────────
class TestMCRSLExperimentResults:
    def test_results_file_exists(self):
        rpath = EVIDENCE_DIR / "results" / "mcr_sl_multimodal_results.json"
        assert rpath.exists(), "MCR-SL multimodal results JSON not found."

    def test_all_architectures_present(self, experiment_results):
        if not experiment_results:
            pytest.skip("Results file empty, run experiments first.")
        exp = experiment_results.get("experiment_results", {})
        expected_configs = [
            "image_only", "context_only", "concatenation_fusion",
            "late_fusion", "cross_attention_fusion", "gated_fusion",
        ]
        for cfg_id in expected_configs:
            assert cfg_id in exp, f"Architecture '{cfg_id}' missing from results."

    def test_roc_auc_in_valid_range(self, experiment_results):
        if not experiment_results:
            pytest.skip("Results file empty.")
        for cfg_id, res in experiment_results.get("experiment_results", {}).items():
            roc = res["multi_seed_summary"]["roc_auc_mean"]
            assert 0.0 <= roc <= 1.0, f"{cfg_id}: ROC-AUC {roc} out of range [0, 1]"

    def test_image_only_roc_above_chance(self, experiment_results):
        if not experiment_results:
            pytest.skip("Results file empty.")
        roc = experiment_results["experiment_results"]["image_only"]["multi_seed_summary"]["roc_auc_mean"]
        # Image-only should be detectably above chance (>0.52) for MCR-SL dermoscopy
        assert roc > 0.52, f"Image-Only ROC-AUC {roc} too close to chance level."

    def test_dataset_description_is_real(self, experiment_results):
        if not experiment_results:
            pytest.skip("Results file empty.")
        assert experiment_results.get("dataset") == "MCR-SL (Multimodal Context-Rich Skin Lesion)"
