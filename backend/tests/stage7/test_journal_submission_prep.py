"""
Unit and regression tests for Stage 7: Final Journal Submission Preparation

Tests:
1. PDF Quality Assurance passes with complete page-by-page integrity and no clipping
2. Submission metadata is generated with structured abstract, MeSH keywords, and placeholders
3. Cover letter is generated with conservative framing and zero hyperbolic claims
4. Journal targeting document evaluates JBI and JAMIA
5. Submission checklist contains all required phases and verification items
6. Cryptographic integrity check confirms zero mutation of authoritative empirical sources
7. Stage 7 final summary recommends submission to primary target
"""

import json
from pathlib import Path
import pytest

from backend.app.stage7.journal_submission_prep import Stage7JournalSubmissionPrep


def _setup_prep():
    return Stage7JournalSubmissionPrep(base_dir=".")


def test_1_pdf_qa_passes():
    prep = _setup_prep()
    qa = prep.perform_pdf_qa()

    assert qa["qa_status"] == "PDF_QA_PASSED"
    assert qa["all_qa_checks_passed"] is True
    assert qa["page_count"] == 15
    assert len(qa["clipping_or_blank_issues"]) == 0


def test_2_submission_metadata():
    prep = _setup_prep()
    meta = prep.create_submission_metadata()

    sub_dir = Path("evidence/final/submission")
    assert (sub_dir / "submission_metadata.json").exists()
    assert (sub_dir / "submission_metadata.md").exists()

    assert len(meta["mesh_keywords"]) >= 6
    assert "AUTHOR_NAME_PLACEHOLDER" in meta["authorship_metadata"]["authors"][0]["name"]
    assert "data_availability_statement" in meta["mandatory_declarations"]
    assert "code_availability_statement" in meta["mandatory_declarations"]


def test_3_cover_letter():
    prep = _setup_prep()
    content = prep.create_cover_letter()

    assert (Path("evidence/final/submission/cover_letter.md")).exists()
    assert "Evidence-Conditioned Compositional Pipeline Synthesis" in content
    assert "0.9751" in content
    assert "0.9704" in content
    assert "modest" in content
    assert "Seed 100" in content
    assert "state-of-the-art" not in content.lower()


def test_4_journal_targeting():
    prep = _setup_prep()
    targeting = prep.create_journal_targeting()

    assert (Path("evidence/final/submission/journal_targeting.md")).exists()
    assert (Path("evidence/final/submission/journal_targeting.json")).exists()
    assert len(targeting["target_venues"]) == 2
    assert "Journal of Biomedical Informatics (JBI)" in targeting["recommendation"]["primary_target"]


def test_5_submission_checklist():
    prep = _setup_prep()
    qa = prep.perform_pdf_qa()
    checklist_path = prep.create_submission_checklist(qa)

    assert checklist_path.exists()
    with open(checklist_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "[x]" in text
    assert "Manuscript PDF" in text
    assert "Cover Letter" in text
    assert "Submission Metadata" in text
    assert "PROCEED TO JOURNAL SUBMISSION" in text


def test_6_final_integrity_verification():
    prep = _setup_prep()
    integrity = prep.verify_final_integrity()

    assert integrity["overall_integrity_status"] == "ZERO_MUTATION_CONFIRMED"
    assert integrity["immutability_verified"] is True
    assert integrity["mismatch_count"] == 0


def test_7_stage7_final_summary():
    prep = _setup_prep()
    summary = prep.run()

    assert summary["pdf_qa_status"] == "PDF_QA_PASSED"
    assert summary["integrity_status"] == "ZERO_MUTATION_CONFIRMED"
    assert summary["final_recommendation"] == "SUBMIT_TO_JOURNAL_OF_BIOMEDICAL_INFORMATICS"
