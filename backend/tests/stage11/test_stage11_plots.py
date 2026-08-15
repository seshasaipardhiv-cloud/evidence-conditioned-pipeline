"""
Unit tests verifying generation and existence of all 10 publication-quality figures
"""

import json
from pathlib import Path
import pytest


def test_all_ten_figures_exist_in_png_and_svg():
    fig_dir = Path("evidence/processed/stage11/figures")
    assert fig_dir.exists()

    expected_figures = [
        "figure1_individual_roc_auc",
        "figure2_individual_pr_auc",
        "figure3_individual_brier_score",
        "figure4_individual_f1_score",
        "figure5_candidate_vs_ensemble_roc",
        "figure6_candidate_vs_ensemble_pr",
        "figure7_per_seed_stability",
        "figure8_roc_curves",
        "figure9_pr_curves",
        "figure10_calibration_confusion",
    ]

    for fig_name in expected_figures:
        png_path = fig_dir / f"{fig_name}.png"
        svg_path = fig_dir / f"{fig_name}.svg"
        assert png_path.exists(), f"Missing {png_path}"
        assert svg_path.exists(), f"Missing {svg_path}"
        assert png_path.stat().st_size > 1000
        assert svg_path.stat().st_size > 1000


def test_plot_manifest_valid():
    manifest_path = Path("evidence/processed/stage11/stage11_plot_manifest.json")
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["figures_generated_count"] >= 20
    assert len(manifest["figures"]) >= 10
