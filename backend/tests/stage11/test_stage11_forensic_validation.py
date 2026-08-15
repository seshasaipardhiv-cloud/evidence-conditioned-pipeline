"""
Unit tests for Stage 11 Forensic Validation Engine
"""

import json
from pathlib import Path
import pytest
from backend.app.stage11.stage11_forensic_validation import Stage11ForensicValidator


def test_stage11_forensic_audit_passes():
    validator = Stage11ForensicValidator(base_dir=".")
    results = validator.run_forensic_audit()

    assert results["overall_status"] == "PASSED"
    assert len(results["checks"]) == 14
    for check_id, check_data in results["checks"].items():
        assert check_data["status"] == "PASSED", f"Check {check_id} failed: {check_data}"


def test_stage11_forensic_audit_json_valid():
    path = Path("evidence/processed/stage11/stage11_forensic_audit.json")
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["overall_status"] == "PASSED"
