"""
Stage 11 Tests: Dataset Adaptation & Decision Traceability (Objective 3 & 4)
Verifies:
1. Decision ledger captures modality discovery and candidate rankings
2. Architecture and fusion selections adapt based on modality input
"""

import json
from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


@pytest.fixture(scope="module")
def adaptation_fixture(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("adapt_test")
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_p))
    validator.run_all_cross_dataset_experiments()
    return tmp_p


def test_dataset_adaptation_audit_file(adaptation_fixture):
    audit_p = adaptation_fixture / "dataset_adaptation_audit.json"
    assert audit_p.exists()

    with open(audit_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "cohorts" in data
    assert len(data["cohorts"]) == 7

    for cohort in data["cohorts"]:
        mods = cohort["discovered_modalities"]
        comps = cohort["selected_components"]
        if "image" in mods:
            assert comps["image_model"] is not None
        if "text" in mods:
            assert comps["text_model"] is not None
        if len(mods) >= 2:
            assert comps["fusion_mechanism"] != "UNIMODAL_HEAD"
