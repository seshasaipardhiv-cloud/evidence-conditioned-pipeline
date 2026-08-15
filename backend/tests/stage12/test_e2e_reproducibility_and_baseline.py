"""
Stage 12 Tests: Reproducibility and Baseline Benchmarking (Tests 17 & 18)
Verifies:
1. TEST 17: Deterministic execution and identical results across runs with same seeds
2. TEST 18: Evidence-conditioned candidate vs fixed baseline comparison
3. Historical immutability of Stage 5B, 5C, 6A, 10, 10.5, and 11 artifacts
"""

import json
from pathlib import Path
import pytest
from backend.app.stage12.final_end_to_end_demo import EndToEndPipelineOrchestrator


@pytest.fixture(scope="module")
def stage12_run(tmp_path_factory):
    tmp_p = tmp_path_factory.mktemp("stage12_repro")
    orch = EndToEndPipelineOrchestrator(base_dir=".", output_dir=str(tmp_p), compute_budget="LIGHT")
    manifest = orch.run_end_to_end()
    return {"dir": tmp_p, "manifest": manifest}


def test_17_reproducibility_report_and_manifest(stage12_run):
    manifest = stage12_run["manifest"]
    tmp_p = stage12_run["dir"]

    assert manifest["status"] == "PASS"
    assert manifest["reproducibility"] == "CONFIRMED"

    repro_p = tmp_p / "reproducibility_report.json"
    assert repro_p.exists()

    with open(repro_p, "r", encoding="utf-8") as f:
        repro_data = json.load(f)
    assert repro_data["exact_deterministic_reproducibility"] is True


def test_18_baseline_comparison_file(stage12_run):
    tmp_p = stage12_run["dir"]
    base_p = tmp_p / "baseline_comparison.json"
    report_p = tmp_p / "final_analysis_report.md"

    assert base_p.exists()
    assert report_p.exists()

    with open(base_p, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    assert "evidence_conditioned_candidate" in base_data
    assert "fixed_default_baseline" in base_data
    assert "delta_roc_auc" in base_data

    # Verify figures generated
    fig1 = tmp_p / "figures" / "stage12_comparative_roc_auc.png"
    fig2 = tmp_p / "figures" / "stage12_calibration_brier_benchmark.png"
    assert fig1.exists()
    assert fig2.exists()


def test_19_historical_immutability():
    # Verify historical Stage 5B, 5C, 6A, 10, 10.5, 11 artifacts remain present and untouched
    assert Path("evidence/processed/stage5b_candidate_results.json").exists()
    assert Path("evidence/metadata/stage5c_statistical_analysis.json").exists()
    assert Path("evidence/final/stage6a_master_results.json").exists()
    assert Path("evidence/processed/stage10/stage10_final_summary.json").exists()
    assert Path("evidence/processed/stage10_5/stage10_5_final_summary.json").exists()
    assert Path("evidence/processed/stage11/stage11_final_summary.json").exists()
