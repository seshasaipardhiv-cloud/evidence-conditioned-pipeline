"""
Tests for Dataset-Adaptive Modality Discovery (Stage 10 - Objective B)
Verifies discovery across 10 controlled scenarios:
1. Tabular only
2. Image only
3. Text only
4. Image + Text
5. Tabular + Image
6. Tabular + Text
7. Tabular + Image + Text
8. Missing modality handling
9. Malformed modality handling
10. Mismatched patient overlap rejection
"""

from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from backend.app.multimodal.modality_discovery import ModalityDiscoveryEngine
from backend.app.multimodal.safety_gates import MultimodalSafetyAuditor


def test_modality_adaptation_ten_scenarios(tmp_path):
    engine = ModalityDiscoveryEngine(output_dir=str(tmp_path))
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")

    # Sample images
    img1 = tmp_path / "p1_scan.png"
    img2 = tmp_path / "p2_scan.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img1)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img2)

    # 1. Tabular Only
    res1 = engine.discover(
        tabular_data=[{"patient_id": "p1", "age": 60, "recurrence": 1}, {"patient_id": "p2", "age": 55, "recurrence": 0}],
        candidate_target="recurrence",
        candidate_id="patient_id",
    )
    assert res1.detected_modalities == ["tabular"]

    # 2. Image Only
    res2 = engine.discover(image_data=[str(img1), str(img2)])
    assert res2.detected_modalities == ["image"]

    # 3. Text Only
    res3 = engine.discover(text_data={"p1": "Patient report clear.", "p2": "High grade tumor."})
    assert res3.detected_modalities == ["text"]

    # 4. Image + Text
    res4 = engine.discover(image_data=[str(img1), str(img2)], text_data={"p1": "report 1", "p2": "report 2"})
    assert set(res4.detected_modalities) == {"image", "text"}

    # 5. Tabular + Image
    res5 = engine.discover(
        tabular_data=[{"patient_id": "p1", "recurrence": 1}, {"patient_id": "p2", "recurrence": 0}],
        image_data=[str(img1), str(img2)],
        candidate_target="recurrence",
    )
    assert set(res5.detected_modalities) == {"tabular", "image"}

    # 6. Tabular + Text
    res6 = engine.discover(
        tabular_data=[{"patient_id": "p1", "recurrence": 1}, {"patient_id": "p2", "recurrence": 0}],
        text_data={"p1": "report 1", "p2": "report 2"},
        candidate_target="recurrence",
    )
    assert set(res6.detected_modalities) == {"tabular", "text"}

    # 7. Tabular + Image + Text
    res7 = engine.discover(
        tabular_data=[{"patient_id": "p1", "recurrence": 1}, {"patient_id": "p2", "recurrence": 0}],
        image_data=[str(img1), str(img2)],
        text_data={"p1": "report 1", "p2": "report 2"},
        candidate_target="recurrence",
    )
    assert set(res7.detected_modalities) == {"tabular", "image", "text"}

    # 8. Missing Modality (Empty input)
    res8 = engine.discover()
    assert res8.status == "BLOCKED"

    # 9. Malformed Modality (No target)
    res9 = engine.discover(tabular_data=[{"feat1": 1, "feat2": 2}])
    assert res9.status == "BLOCKED"

    # 10. Mismatched Patient Overlap Rejection
    audit = auditor.audit_all(
        modalities=["tabular", "image"],
        train_pids=["p1", "p2"],
        val_pids=[],
        test_pids=["p2", "p3"],  # Overlap on p2!
        train_features={},
        val_features={},
        test_features={},
        pipeline_config={"embed_dim": 64},
    )
    assert audit["gate_results"]["gate_3_patient_overlap_firewall"]["passed"] is False
