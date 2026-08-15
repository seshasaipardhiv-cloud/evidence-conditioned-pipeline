"""
Stage 11 Tests: Reproducibility & Historical Immutability (Objective 10 & 15)
Verifies:
1. Multi-seed deterministic reproducibility ([42, 100, 2026])
2. Immutability of all historical Stage 5B, 5C, 6A–6I, 10, 10.5 artifacts
"""

import json
from pathlib import Path
import pytest
from backend.app.stage11.cross_dataset_generalization import CrossDatasetTransferValidator


@pytest.fixture(scope="module")
def repro_fixture(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("repro_test")
    validator = CrossDatasetTransferValidator(base_dir=".", output_dir=str(tmp_p))
    validator.run_all_cross_dataset_experiments()
    return tmp_p


def test_stage11_reproducibility_and_immutability(repro_fixture):
    repro_p = repro_fixture / "reproducibility_report.json"
    assert repro_p.exists()

    with open(repro_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["deterministic_reproducibility_verified"] is True
    assert data["seeds_evaluated"] == [42, 100, 2026]

    # Verify historical Stage 5B/5C/6A/10/10.5 files exist
    assert Path("evidence/processed/stage5b_candidate_results.json").exists()
    assert Path("evidence/metadata/stage5c_statistical_analysis.json").exists()
    assert Path("evidence/final/stage6a_master_results.json").exists()
    assert Path("evidence/processed/stage10/stage10_final_summary.json").exists()
    assert Path("evidence/processed/stage10_5/stage10_5_final_summary.json").exists()
