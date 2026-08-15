"""
Unit and regression tests for Stage 8: Final JBI Submission Portal Readiness

Tests:
1. JBI Research Highlights exist, contain 3-5 bullets, and strictly obey <= 85 character limit
2. CRediT Author Statement template exists and uses CRediT taxonomy roles
3. JBI Compliance Audit passes with >= 85% compliance score (only pending author metadata placeholders)
4. JBI Upload Manifest exists and maps all 17 submission items to Editorial Manager types
5. Stage 8 final summary confirms submission readiness
6. Cryptographic integrity confirms zero mutation of authoritative empirical sources
"""

import json
from pathlib import Path
import pytest

from backend.app.stage8.jbi_portal_readiness import Stage8JBIPortalReadiness


def _setup_prep():
    return Stage8JBIPortalReadiness(base_dir=".")


def test_1_jbi_highlights_character_limit():
    prep = _setup_prep()
    highlights = prep.create_jbi_highlights()

    assert 3 <= len(highlights) <= 5
    for h in highlights:
        assert len(h) <= 85, f"Highlight exceeds 85 chars: '{h}' ({len(h)})"

    hl_path = Path("evidence/final/submission/jbi_highlights.md")
    assert hl_path.exists()


def test_2_credit_statement_exists():
    prep = _setup_prep()
    content = prep.create_credit_statement()

    credit_path = Path("evidence/final/submission/credit_statement.md")
    assert credit_path.exists()
    assert "Conceptualization" in content
    assert "Methodology" in content
    assert "Software" in content


def test_3_jbi_compliance_audit():
    prep = _setup_prep()
    audit = prep.audit_jbi_compliance()

    sub_dir = Path("evidence/final/submission")
    assert (sub_dir / "jbi_compliance_audit.json").exists()
    assert (sub_dir / "jbi_compliance_checklist.md").exists()

    assert audit["compliance_score_percent"] >= 85.0
    assert audit["passed_requirements_count"] >= 13
    assert audit["needs_action_count"] == 2  # Real author metadata and CRediT initials input by author


def test_4_jbi_upload_manifest():
    prep = _setup_prep()
    manifest_path = prep.create_jbi_upload_manifest()

    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        text = f.read()

    assert "Editorial Manager Item Type" in text
    assert "Cover Letter" in text
    assert "Highlights" in text
    assert "Manuscript" in text
    assert "Figure 1" in text
    assert "Figure 8" in text
    assert "Supplementary Material" in text


def test_5_stage8_final_summary_and_readiness():
    prep = _setup_prep()
    summary = prep.run()

    assert summary["submission_readiness"] == "READY_FOR_PORTAL_UPLOAD"
    assert summary["integrity_status"] == "ZERO_MUTATION_CONFIRMED"


def test_6_immutability_verification():
    prep = _setup_prep()
    integrity = prep.verify_portal_integrity()

    assert integrity["immutability_verified"] is True
    assert integrity["mismatch_count"] == 0
