"""
Stage 10.5 Tests: Multimodal Result Forensic Validation
Verifies:
1. Independent ROC-AUC recomputation using sklearn.metrics.roc_auc_score
2. Independent PR-AUC and Brier score loss calculation
3. Confusion matrix metrics (TP, TN, FP, FN, Accuracy, Precision, Recall, F1)
4. Patient-level firewall (0 patient overlap across all folds)
5. Target leakage absence
6. True neural model execution & gradient flow
7. Multi-seed reproduction across seeds [42, 100, 2026]
8. Scientific verdict classification and conservative claim boundary
"""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.app.stage10.multimodal_result_forensic_validation import MultimodalResultForensicValidator


@pytest.fixture(scope="module")
def forensic_output(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("forensics")
    validator = MultimodalResultForensicValidator(base_dir=".", stage10_dir="evidence/processed/stage10", output_dir=str(tmp_p))
    summary = validator.run_all_forensics()
    return {"path": tmp_p, "summary": summary}


def test_1_roc_auc_independent_recomputation(forensic_output):
    tmp_path = forensic_output["path"]
    roc_p = tmp_path / "roc_auc_forensic.json"
    assert roc_p.exists()

    with open(roc_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["metric"] == "ROC-AUC"
    assert np.isclose(data["independent_recomputed_mean"], 1.0000)
    assert data["exact_match"] is True


def test_2_pr_auc_and_brier_independent_recomputation(forensic_output):
    tmp_path = forensic_output["path"]
    pr_p = tmp_path / "pr_auc_forensic.json"
    brier_p = tmp_path / "brier_forensic.json"
    assert pr_p.exists()
    assert brier_p.exists()

    with open(pr_p, "r", encoding="utf-8") as f:
        pr_data = json.load(f)
    assert pr_data["pr_auc_vs_roc_auc_distinction_verified"] is True

    with open(brier_p, "r", encoding="utf-8") as f:
        brier_data = json.load(f)
    assert brier_data["probability_range_valid"] is True


def test_3_confusion_matrix_and_threshold_metrics(forensic_output):
    tmp_path = forensic_output["path"]
    cm_p = tmp_path / "confusion_matrix_forensic.json"
    assert cm_p.exists()

    with open(cm_p, "r", encoding="utf-8") as f:
        cm_data = json.load(f)

    assert "42" in cm_data["per_seed_confusion_matrices"]
    assert "100" in cm_data["per_seed_confusion_matrices"]
    assert "2026" in cm_data["per_seed_confusion_matrices"]
    assert "threshold_vs_ranking_explanation" in cm_data


def test_4_patient_overlap_firewall(forensic_output):
    tmp_path = forensic_output["path"]
    patient_p = tmp_path / "patient_integrity_audit.json"
    assert patient_p.exists()

    with open(patient_p, "r", encoding="utf-8") as f:
        p_data = json.load(f)

    assert p_data["patient_overlap_detected"] is False
    assert "PASSED" in p_data["patient_overlap_firewall_status"]


def test_5_target_leakage_forensics(forensic_output):
    tmp_path = forensic_output["path"]
    leakage_p = tmp_path / "target_leakage_audit.json"
    assert leakage_p.exists()

    with open(leakage_p, "r", encoding="utf-8") as f:
        leak_data = json.load(f)

    assert leak_data["target_leakage_verdict"] == "ZERO_LEAKAGE_CONFIRMED"
    assert leak_data["target_in_image_tensors"] is False
    assert leak_data["target_in_text_embeddings"] is False


def test_6_model_execution_forensics(forensic_output):
    tmp_path = forensic_output["path"]
    model_p = tmp_path / "model_execution_audit.json"
    assert model_p.exists()

    with open(model_p, "r", encoding="utf-8") as f:
        m_data = json.load(f)

    assert m_data["resnet18_execution_verified"] is True
    assert m_data["pubmedbert_execution_verified"] is True
    assert m_data["cross_attention_forward_and_backward_verified"] is True


def test_7_seed_reproduction_and_ablation(forensic_output):
    tmp_path = forensic_output["path"]
    reprod_p = tmp_path / "reproduction_results.json"
    ablation_p = tmp_path / "ablation_integrity_audit.json"
    assert reprod_p.exists()
    assert ablation_p.exists()

    with open(reprod_p, "r", encoding="utf-8") as f:
        r_data = json.load(f)
    assert r_data["exact_match_with_stage10"] is True

    with open(ablation_p, "r", encoding="utf-8") as f:
        a_data = json.load(f)
    assert a_data["identical_patient_splits"] is True
    assert a_data["identical_seeds"] is True


def test_8_scientific_verdict_and_conservative_claim(forensic_output):
    summary = forensic_output["summary"]
    assert summary["scientific_verdict"] == "VALID BUT UNDERPOWERED"
    assert "conservative_claim_boundary" in summary
    assert "small cohort size" in summary["conservative_claim_boundary"]
