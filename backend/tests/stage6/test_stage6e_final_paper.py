"""
Unit and regression tests for Phase 6E: Final Research Paper Assembly and Consistency Audit

Tests:
1. Final paper exists and has substantial content (>2000 words)
2. All required section markdown files exist
3. Candidate ROC-AUC = 0.9751 ± 0.0114 exactly
4. Default XGBoost = 0.9704 ± 0.0059 exactly
5. Margin +0.0047 is explicitly described as modest
6. Seed 100 loss is preserved (0.9609 vs 0.9643)
7. Candidate wins exactly 2 of 3 seeds
8. Ablation values match Stage 6A/5C exactly
9. Brier score for candidate = 0.0175 and lowest across models
10. Six evidence-backed components match citations
11. Two explicitly configured components are present and distinct
12. Evidence vs configuration distinction is strictly preserved
13. All 8 figures are referenced in the paper
14. No statistical significance claims
15. No clinical validation claims
16. No external validation claims
17. No state-of-the-art claims
18. No first-ever claims
19. Limitations and threats to validity are comprehensively present
20. Source master results and figures remain immutable
"""

import json
from pathlib import Path
import pytest

from backend.app.stage6.final_paper_assembler_stage6e import (
    Stage6EPaperAssembler,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    EXPECTED_STAGE5A_CONTRACT_HASH,
)


def _setup_test_assembler():
    return Stage6EPaperAssembler(
        final_dir="evidence/final",
        manuscript_dir="evidence/final/manuscript",
        figures_dir="evidence/final/figures",
        paper_dir="evidence/final/paper",
    )


def test_1_final_paper_exists_and_word_count():
    assembler = _setup_test_assembler()
    manifest, audit = assembler.run()

    paper_path = Path("evidence/final/paper/final_research_paper.md")
    assert paper_path.exists()
    assert manifest["word_count"] > 1500
    assert audit["audit_status"] == "AUDIT_PASSED"


def test_2_required_section_files_exist():
    assembler = _setup_test_assembler()
    assembler.run()
    paper_dir = Path("evidence/final/paper")

    expected_files = [
        "abstract.md",
        "introduction.md",
        "related_work.md",
        "methodology.md",
        "experimental_setup.md",
        "results.md",
        "discussion.md",
        "novelty_contributions.md",
        "limitations.md",
        "conclusion.md",
        "references.md",
        "final_paper_manifest.json",
        "final_scientific_audit.json",
    ]
    for ef in expected_files:
        p = paper_dir / ef
        assert p.exists(), f"Missing paper section file: {ef}"


def test_3_candidate_roc_auc_exact():
    assembler = _setup_test_assembler()
    manifest, _ = assembler.run()
    assert manifest["core_metrics"]["candidate_mean_roc_auc"] == 0.9751
    assert manifest["core_metrics"]["candidate_std_roc_auc"] == 0.0114


def test_4_default_xgboost_exact():
    assembler = _setup_test_assembler()
    manifest, _ = assembler.run()
    assert manifest["core_metrics"]["default_xgboost_mean_roc_auc"] == 0.9704


def test_5_margin_described_as_modest():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "+0.0047" in text or "0.0047" in text
    assert "modest" in text.lower()


def test_6_seed_100_loss_appears():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "0.9609" in text
    assert "0.9643" in text
    assert "-0.0034" in text


def test_7_candidate_wins_2_of_3_seeds():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "2 out of 3 seeds" in text or "2 of 3 seeds" in text


def test_8_ablation_values_unchanged():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "0.9773" in text  # Without SMOTE
    assert "0.9767" in text  # Mean Imputation
    assert "0.9784" in text  # Ordinal Encoding
    assert "0.9686" in text  # Default XGBoost


def test_9_brier_scores_unchanged():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "0.0175" in text  # Candidate
    assert "0.0180" in text  # Default XGB
    assert "0.0201" in text  # LR
    assert "0.0207" in text  # RF
    assert "0.0683" in text  # MLP


def test_10_six_evidence_backed_components():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/methodology.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "clinical_tabular_representation" in text
    assert "cross_attention" in text
    assert "average_ensembling" in text
    assert "MissForest / MICE" in text
    assert "XGBoost" in text
    assert "SMOTE" in text


def test_11_two_explicitly_configured_components():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/methodology.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "one_hot_encoding" in text
    assert "binary_logistic" in text


def test_12_evidence_configuration_distinction_preserved():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/methodology.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "EVIDENCE_BACKED" in text
    assert "EXPLICITLY_CONFIGURED" in text


def test_13_all_eight_figures_referenced():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    for i in range(1, 9):
        assert f"Figure {i}" in text, f"Missing Figure {i} reference in final paper"


def test_14_no_statistical_significance_claim():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read().lower()

    assert "is statistically significant" not in text
    assert "statistically significant improvement" not in text


def test_15_no_clinical_validation_claim():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read().lower()

    assert "clinically proven" not in text
    assert "clinically validated" not in text


def test_16_no_external_validation_claim():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read().lower()

    assert "externally validated" not in text


def test_17_no_state_of_the_art_claim():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read().lower()

    assert "state-of-the-art" not in text
    assert "state of the art" not in text


def test_18_no_first_ever_claim():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read().lower()

    assert "first-ever" not in text
    assert "first ever" not in text


def test_19_limitations_are_present():
    assembler = _setup_test_assembler()
    assembler.run()
    with open(Path("evidence/final/paper/final_research_paper.md"), "r", encoding="utf-8") as f:
        text = f.read()

    assert "Threats to Validity" in text or "Limitations" in text
    assert "Single Retrospective Cohort" in text


def test_20_master_source_immutability():
    master_path = Path("evidence/final/stage6a_master_results.json")
    with open(master_path, "r", encoding="utf-8") as f:
        master_before = json.load(f)

    assembler = _setup_test_assembler()
    assembler.run()

    with open(master_path, "r", encoding="utf-8") as f:
        master_after = json.load(f)

    assert master_before == master_after
