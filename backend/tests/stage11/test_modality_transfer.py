"""
Stage 11 Tests: Modality Transfer & Pipeline Functionality (Objective 6)
Verifies:
1. Valid pipeline execution for Tabular, Image, Text, and all combinations
2. Functionality of neural heads and fusion layers across all 7 modalities
"""

import json
from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


@pytest.fixture(scope="module")
def modality_fixture(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("mod_test")
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_p))
    validator.run_all_cross_dataset_experiments()
    return tmp_p


def test_modality_transfer_file_and_metrics(modality_fixture):
    mod_p = modality_fixture / "modality_transfer_results.json"
    assert mod_p.exists()

    with open(mod_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["successful_transfers_count"] == 7
    assert data["transfer_status"] == "ALL_MODALITIES_FUNCTIONAL"
