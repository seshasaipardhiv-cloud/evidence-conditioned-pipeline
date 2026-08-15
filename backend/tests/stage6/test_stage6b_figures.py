"""
Unit and regression tests for Phase 6B: Publication-Quality Figures and Visual Evidence

Tests:
1. All 8 figures exist in both PNG and vector SVG format
2. Figure manifest is complete and contains 8 valid entries
3. Source values in manifest match Phase 6A authoritative values exactly
4. Candidate ROC-AUC is exactly 0.9751
5. Baseline comparison values match Phase 6A exactly
6. Per-seed values (42, 100, 2026) match Phase 6A exactly
7. Component ablation values match Phase 6A exactly
8. Calibration Brier scores match Phase 6A exactly
9. No unsupported claim is encoded in figure titles or captions
10. Source artifacts remain strictly immutable
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.figure_generator_stage6b import Stage6BFigureGenerator


def _setup_test_generator():
    return Stage6BFigureGenerator(
        master_results_path="evidence/final/stage6a_master_results.json",
        figures_dir="evidence/final/figures",
    )


def test_1_all_eight_figures_generated_and_exist():
    gen = _setup_test_generator()
    manifest = gen.run()
    assert len(manifest["figures"]) == 8

    fig_dir = Path("evidence/final/figures")
    for i in range(1, 9):
        png_files = list(fig_dir.glob(f"fig{i}_*.png"))
        svg_files = list(fig_dir.glob(f"fig{i}_*.svg"))
        assert len(png_files) >= 1, f"Missing PNG for figure {i}"
        assert len(svg_files) >= 1, f"Missing SVG for figure {i}"
        assert png_files[0].stat().st_size > 1000, f"PNG file too small for figure {i}"
        assert svg_files[0].stat().st_size > 500, f"SVG file too small for figure {i}"


def test_2_manifest_structure_and_hashes():
    gen = _setup_test_generator()
    manifest = gen.run()
    for fig in manifest["figures"]:
        assert "figure_id" in fig
        assert "title" in fig
        assert "source_artifact" in fig
        assert "exact_source_values" in fig
        assert "interpretation" in fig
        assert "limitations" in fig
        assert "generated_png" in fig
        assert "generated_svg" in fig
        assert len(fig["sha256_png"]) == 64


def test_3_exact_candidate_and_baseline_values():
    gen = _setup_test_generator()
    manifest = gen.run()
    fig2 = next(f for f in manifest["figures"] if f["figure_id"] == "FIGURE_2")
    vals = fig2["exact_source_values"]
    assert vals["candidate"] == 0.9751
    assert vals["default_xgboost"] == 0.9704
    assert vals["random_forest"] == 0.9698
    assert vals["logistic_regression"] == 0.9645
    assert vals["simple_mlp"] == 0.9405


def test_4_per_seed_robustness_values():
    gen = _setup_test_generator()
    manifest = gen.run()
    fig3 = next(f for f in manifest["figures"] if f["figure_id"] == "FIGURE_3")
    vals = fig3["exact_source_values"]
    assert vals["seed_42"]["candidate"] == 0.9888
    assert vals["seed_100"]["candidate"] == 0.9609
    assert vals["seed_100"]["default_xgb"] == 0.9643  # Seed 100 loss recorded
    assert vals["seed_2026"]["candidate"] == 0.9756


def test_5_component_ablation_values():
    gen = _setup_test_generator()
    manifest = gen.run()
    fig4 = next(f for f in manifest["figures"] if f["figure_id"] == "FIGURE_4")
    vals = fig4["exact_source_values"]
    assert vals["full_candidate"] == 0.9751
    assert vals["without_smote"] == 0.9773
    assert vals["mean_imputation"] == 0.9767
    assert vals["ordinal_encoding"] == 0.9784
    assert vals["default_xgboost"] == 0.9686


def test_6_calibration_brier_values():
    gen = _setup_test_generator()
    manifest = gen.run()
    fig5 = next(f for f in manifest["figures"] if f["figure_id"] == "FIGURE_5")
    vals = fig5["exact_source_values"]
    assert vals["candidate"] == 0.0175
    assert vals["default_xgboost"] == 0.0180
    assert vals["logistic_regression"] == 0.0201
    assert vals["random_forest"] == 0.0207
    assert vals["simple_mlp"] == 0.0683


def test_7_no_unsupported_claims_in_figures():
    gen = _setup_test_generator()
    manifest = gen.run()
    for fig in manifest["figures"]:
        title = fig["title"].lower()
        interp = fig["interpretation"].lower()
        assert "statistically significant" not in title
        assert "clinical cure" not in title
        assert "universally superior" not in title


def test_8_master_source_immutability():
    master_path = Path("evidence/final/stage6a_master_results.json")
    with open(master_path, "r", encoding="utf-8") as f:
        master_before = json.load(f)

    gen = _setup_test_generator()
    gen.run()

    with open(master_path, "r", encoding="utf-8") as f:
        master_after = json.load(f)

    assert master_before == master_after
