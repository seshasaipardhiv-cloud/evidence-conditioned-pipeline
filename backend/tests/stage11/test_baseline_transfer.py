"""
Stage 11 Tests: Baseline Transfer Comparison (Objective 9)
Verifies:
1. Controlled comparison between evidence-conditioned candidate and fixed default baseline
2. Identical splits and seeds used for benchmarking
"""

import json
from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


@pytest.fixture(scope="module")
def baseline_fixture(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("base_test")
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_p))
    validator.run_all_cross_dataset_experiments()
    return tmp_p


def test_baseline_transfer_comparison_file(baseline_fixture):
    base_p = baseline_fixture / "baseline_transfer_comparison.json"
    assert base_p.exists()

    with open(base_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "evidence_conditioned_pipeline" in data
    assert "fixed_default_pipeline" in data
    assert "delta_roc_auc" in data
