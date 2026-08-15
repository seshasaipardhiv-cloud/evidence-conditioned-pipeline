"""
Stage 12 Tests: Evidence Profile and Compute Budget Shifts (Tests 8 & 9)
Verifies:
1. TEST 8: Changed literature evidence profile changes model rankings
2. TEST 9: Changed compute budget tier changes selected architectures appropriately
"""

from pathlib import Path
import pytest
from backend.app.multimodal.image_selector import ImageModelSelector
from backend.app.multimodal.text_selector import TextModelSelector
from backend.app.multimodal.fusion_selector import FusionSelector


def test_8_changed_evidence_profile_shifts_rankings():
    img_sel = ImageModelSelector()
    txt_sel = TextModelSelector()

    # Profile A: Standard binary classification (lightweight)
    res_img_a = img_sel.select(task_type="binary_classification", compute_budget="LIGHT")
    res_txt_a = txt_sel.select(task_type="binary_classification", domain_type="biomedical", compute_budget="LIGHT")

    # Profile B: Task-specific domain (tumor grading, clinical EHR discharge notes)
    res_img_b = img_sel.select(task_type="tumor_grading", modality_subtypes=["histopathology"], compute_budget="LIGHT")
    res_txt_b = txt_sel.select(task_type="recurrence_prediction", domain_type="clinical_notes", compute_budget="MEDIUM")

    assert res_img_a["name"] != res_img_b["name"] or res_img_a["score"] != res_img_b["score"]
    assert res_txt_a["name"] != res_txt_b["name"] or res_txt_a["score"] != res_txt_b["score"]
    assert "PMID:" in res_img_a["evidence_source"]
    assert "PMID:" in res_txt_a["evidence_source"]


def test_9_changed_compute_budget_shifts_architecture():
    img_sel = ImageModelSelector()

    # Light Budget
    light_img = img_sel.select(task_type="binary_classification", compute_budget="LIGHT")
    assert light_img["compute_cost"] == "LIGHT"

    # Heavy Budget
    heavy_img = img_sel.select(task_type="binary_classification", modality_subtypes=["histopathology"], compute_budget="HEAVY", sample_count=200)
    assert heavy_img["compute_cost"] in ["MEDIUM", "HEAVY"]
