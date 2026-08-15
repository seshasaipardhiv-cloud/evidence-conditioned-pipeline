"""
Stage 11 Tests: Missing Modality Edge Cases (Objective 7)
Verifies:
1. Safe handling of missing image files
2. Corruption detection for malformed images
3. Safe fallback for empty and None text inputs
4. Rejection of overlapping patient IDs across partitions
"""

from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


def test_missing_and_corrupt_modality_handling(tmp_path):
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_path))
    res = validator.audit_missing_modalities()

    assert res["overall_status"] == "PASSED"
    assert res["scenarios"]["missing_image_handled"]["passed"] is True
    assert res["scenarios"]["corrupt_image_handled"]["passed"] is True
    assert res["scenarios"]["empty_and_none_text_handled"]["passed"] is True
    assert res["scenarios"]["patient_overlap_firewall"]["passed"] is True
