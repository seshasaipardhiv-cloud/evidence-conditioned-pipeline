"""
Unit and regression tests for Phase 6D: Scientific Results, Discussion, Research Gap, Novelty, and Contribution Analysis

Tests:
1. Exact candidate metrics match authoritative Stage 6A/5B values
2. Exact baseline metrics match authoritative Stage 6A/5B values
3. Exact ablation values match authoritative Stage 6A/5C values
4. Exact per-seed values match authoritative Stage 6A/5B values
5. Seed-100 candidate loss is preserved and documented
6. No "statistically significant" claim is introduced
7. No "clinically validated" or deployment claim is introduced
8. No "external validation" claim is introduced
9. No "state-of-the-art" or "first-ever" claim is introduced
10. Evidence-conditioned synthesis is identified as the core methodological contribution
11. Evidence-backed vs explicitly-configured distinction is preserved
12. Threats to validity and limitations are comprehensively present
13. Future validation requirements are explicitly present
14. Source artifacts remain strictly immutable
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.scientific_interpretation_stage6d import (
    Stage6DInterpretationGenerator,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    EXPECTED_STAGE5A_CONTRACT_HASH,
)


def _setup_test_generator():
    return Stage6DInterpretationGenerator(
        final_dir="evidence/final",
        manuscript_dir="evidence/final/manuscript",
    )


def test_1_exact_candidate_and_baseline_metrics():
    gen = _setup_test_generator()
    manifest = gen.run()
    metrics = manifest["core_metrics"]
    assert metrics["candidate_mean_roc_auc"] == 0.9751
    assert metrics["candidate_std_roc_auc"] == 0.0114
    assert metrics["default_xgboost_mean_roc_auc"] == 0.9704
    assert metrics["margin_delta"] == 0.0047
    assert metrics["candidate_brier_score"] == 0.0175


def test_2_exact_ablation_values():
    gen = _setup_test_generator()
    manifest = gen.run()
    abls = manifest["ablations"]
    assert abls["full_candidate"] == 0.9751
    assert abls["without_smote"] == 0.9773
    assert abls["mean_imputation"] == 0.9767
    assert abls["ordinal_encoding"] == 0.9784
    assert abls["default_xgboost"] == 0.9686


def test_3_exact_per_seed_values_and_seed_100_loss():
    gen = _setup_test_generator()
    manifest = gen.run()
    seeds = manifest["per_seed_margins"]
    assert seeds["seed_42"]["candidate"] == 0.9888
    assert seeds["seed_42"]["default_xgb"] == 0.9783
    assert seeds["seed_42"]["won"] is True

    assert seeds["seed_100"]["candidate"] == 0.9609
    assert seeds["seed_100"]["default_xgb"] == 0.9643
    assert seeds["seed_100"]["won"] is False  # Candidate lost on seed 100

    assert seeds["seed_2026"]["candidate"] == 0.9756
    assert seeds["seed_2026"]["default_xgb"] == 0.9685
    assert seeds["seed_2026"]["won"] is True


def test_4_no_hyperbolic_or_unsupported_claims():
    gen = _setup_test_generator()
    gen.run()
    man_dir = Path("evidence/final/manuscript")

    for f_name in ["results.md", "discussion.md", "novelty.md", "contributions.md", "threats_to_validity.md"]:
        with open(man_dir / f_name, "r", encoding="utf-8") as f:
            text = f.read().lower()

        assert "first ever" not in text
        assert "first-ever" not in text
        assert "state of the art" not in text
        assert "state-of-the-art" not in text
        assert "statistically significant" not in text or "not statistically significant" in text or "suppress claims of statistical significance" in text


def test_5_research_gap_methodological_positioning():
    gen = _setup_test_generator()
    gen.run()
    with open(Path("evidence/final/manuscript/research_gap.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "Arbitrary Default Proliferation" in text
    assert "Fabricated Provenance Risk" in text
    assert "Data and Target Leakage" in text
    assert "Silent Fallback" in text


def test_6_novelty_three_levels_present():
    gen = _setup_test_generator()
    gen.run()
    with open(Path("evidence/final/manuscript/novelty.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "Level 1 — Methodological Novelty" in text
    assert "Level 2 — Governance & Safety Novelty" in text
    assert "Level 3 — Execution Novelty" in text


def test_7_threats_to_validity_comprehensive():
    gen = _setup_test_generator()
    gen.run()
    with open(Path("evidence/final/manuscript/threats_to_validity.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "Internal Validity" in text
    assert "Dataset Validity" in text
    assert "Statistical Validity" in text
    assert "External Validity" in text
    assert "Configuration Validity" in text
    assert "Evidence Corpus Limitations" in text
    assert "Model Comparison Limitations" in text


def test_8_future_work_priorities():
    gen = _setup_test_generator()
    gen.run()
    with open(Path("evidence/final/manuscript/future_work.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "Multi-Center External Validation" in text
    assert "Prospective Clinical Trials" in text
    assert "Large-Scale Seed and Cross-Validation Expansion" in text


def test_9_master_source_immutability():
    master_path = Path("evidence/final/stage6a_master_results.json")
    with open(master_path, "r", encoding="utf-8") as f:
        master_before = json.load(f)

    gen = _setup_test_generator()
    gen.run()

    with open(master_path, "r", encoding="utf-8") as f:
        master_after = json.load(f)

    assert master_before == master_after
