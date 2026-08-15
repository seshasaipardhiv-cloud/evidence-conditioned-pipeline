"""
Stage 12 Tests: Safety Audits, Firewalls, and Missing Data Handling (Tests 10 - 16)
Verifies:
1. TEST 10: Missing image safe handling
2. TEST 11: Corrupted image safe handling
3. TEST 12: Missing text safe handling
4. TEST 13: Patient overlap rejection
5. TEST 14: Target leakage rejection
6. TEST 15: Temporal leakage rejection
7. TEST 16: Duplicate records across partitions rejection
"""

from pathlib import Path
import numpy as np
import pytest

from backend.app.multimodal.image_preprocessing import ImagePreprocessor
from backend.app.multimodal.text_preprocessing import TextPreprocessor
from backend.app.multimodal.safety_gates import MultimodalSafetyAuditor


def test_10_missing_image_safe_handling(tmp_path):
    prep = ImagePreprocessor(target_size=(32, 32))
    prep.fit([str(tmp_path / "non_existent.png")])
    out = prep.transform([str(tmp_path / "non_existent.png")], is_training=False)
    assert out.shape == (1, 3, 32, 32)
    assert np.all(np.isfinite(out))


def test_11_corrupted_image_safe_handling(tmp_path):
    bad_img = tmp_path / "corrupt.png"
    with open(bad_img, "wb") as f:
        f.write(b"not a valid png file")

    prep = ImagePreprocessor(target_size=(32, 32))
    prep.fit([str(bad_img)])
    out = prep.transform([str(bad_img)], is_training=False)
    assert out.shape == (1, 3, 32, 32)
    assert np.all(np.isfinite(out))


def test_12_missing_text_safe_handling():
    prep = TextPreprocessor(max_seq_length=32)
    prep.fit(["valid report", None, ""])
    ids, masks = prep.transform([None, ""], is_training=False)
    assert ids.shape == (2, 32)
    assert masks.shape == (2, 32)
    assert np.all(np.isfinite(ids))


def test_13_patient_overlap_rejection():
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")
    res = auditor.audit_all(
        modalities=["tabular"],
        train_pids=["P01", "P02", "P03"],
        val_pids=[],
        test_pids=["P03", "P04"],  # Overlap on P03
        train_features={},
        val_features={},
        test_features={},
        pipeline_config={},
    )
    assert res["overall_status"] in ["BLOCKED", "FAILED"]
    assert res["gate_results"]["gate_3_patient_overlap_firewall"]["passed"] is False


def test_14_target_leakage_rejection():
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")
    res = auditor.audit_all(
        modalities=["tabular"],
        train_pids=["P01", "P02"],
        val_pids=[],
        test_pids=["P03", "P04"],
        train_features={"tabular": np.zeros((2, 2)), "five_year_recurrence_flag": np.array([1, 0])},  # Target in features
        val_features={},
        test_features={},
        pipeline_config={},
    )
    assert res["overall_status"] in ["BLOCKED", "FAILED"]
    assert res["gate_results"]["gate_4_target_leakage"]["passed"] is False


def test_15_temporal_leakage_rejection():
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")
    res = auditor.audit_all(
        modalities=["tabular"],
        train_pids=["P01", "P02"],
        val_pids=[],
        test_pids=["P03", "P04"],
        train_features={"tabular": np.zeros((2, 2)), "post_recurrence_chemo_dose": np.zeros((2,))},
        val_features={},
        test_features={},
        pipeline_config={},
    )
    assert res["overall_status"] in ["BLOCKED", "FAILED"]
    assert res["gate_results"]["gate_5_temporal_leakage_post_adjuvant"]["passed"] is False


def test_16_duplicate_records_rejection():
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")
    # Duplicate image hashes across train and test
    res = auditor.audit_all(
        modalities=["image"],
        train_pids=["P01", "P02"],
        val_pids=[],
        test_pids=["P03", "P04"],
        train_features={"image_hashes": {"P01": "hash_alpha", "P02": "hash_beta"}},
        val_features={},
        test_features={"image_hashes": {"P03": "hash_alpha", "P04": "hash_gamma"}},  # Duplicate hash_alpha in test
        pipeline_config={},
    )
    assert res["overall_status"] in ["BLOCKED", "FAILED"]
    assert res["gate_results"]["gate_3_patient_overlap_firewall"]["passed"] is False
