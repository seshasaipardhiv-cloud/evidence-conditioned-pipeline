"""
Unit and regression tests for Stage 9: Post-Submission / Peer-Review Readiness

Tests:
1. Review Response Template exists with structured point-by-point response schema
2. Reviewer Issue Tracker contains all 10 mandatory fields and pre-populated defense items
3. Submission Record contains complete submission placeholders and PDF SHA-256
4. Revision Policy charter enforces immutable baseline invariants and versioning rules
5. Stage 9 manifest and summary confirm PEER_REVIEW_INFRASTRUCTURE_READY
6. Cryptographic integrity check confirms zero mutation of authoritative empirical sources
"""

import json
from pathlib import Path
import pytest

from backend.app.stage9.peer_review_readiness import Stage9PeerReviewReadiness


def _setup_prep():
    return Stage9PeerReviewReadiness(base_dir=".")


def test_1_review_response_template():
    prep = _setup_prep()
    path = prep.create_response_template()

    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "Reviewer Comment" in text
    assert "Author Response" in text
    assert "Action Taken" in text
    assert "Exact Manuscript Location" in text
    assert "Evidence / Source" in text
    assert "Verification Status" in text


def test_2_reviewer_issue_tracker():
    prep = _setup_prep()
    path = prep.create_issue_tracker()

    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    required_headers = [
        "Reviewer",
        "Comment ID",
        "Exact Reviewer Comment",
        "Manuscript Section",
        "Requested Change",
        "Scientific Impact",
        "Evidence Required",
        "Response Drafted",
        "Manuscript Change Made",
        "Verification Status",
    ]
    for h in required_headers:
        assert h in text, f"Missing header in issue tracker: {h}"

    # Check that pre-populated items exist
    assert "Q01" in text
    assert "Q25" in text


def test_3_submission_record():
    prep = _setup_prep()
    path = prep.create_submission_record()

    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "Journal of Biomedical Informatics" in text
    assert "MANUSCRIPT_ID_PENDING_SUBMISSION" in text
    assert "SUBMISSION_DATE_PENDING" in text
    assert "final_research_paper.pdf" in text
    assert "0.9751" in text
    assert "0.9704" in text


def test_4_revision_policy_charter():
    prep = _setup_prep()
    path = prep.create_revision_policy()

    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "Stage 5B Raw Results" in text
    assert "Seed 100" in text
    assert "+0.0047" in text
    assert "n=3" in text
    assert "HANCOCK" in text
    assert "cross_attention" in text
    assert "Post-Adjuvant" in text
    assert "Separate Stage Versioning" in text


def test_5_stage9_final_summary_and_readiness():
    prep = _setup_prep()
    summary = prep.run()

    assert summary["status"] == "PEER_REVIEW_INFRASTRUCTURE_READY"
    assert summary["integrity_status"] == "ZERO_MUTATION_CONFIRMED"
    assert summary["pre_populated_defense_items_count"] == 25


def test_6_immutability_verification():
    prep = _setup_prep()
    integrity = prep.verify_integrity()

    assert integrity["immutability_verified"] is True
    assert integrity["mismatch_count"] == 0
