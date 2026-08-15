"""Tests for Stage 1 validator."""
import pytest
from backend.app.stage1.validator import validate_problem_statement, validate_representation_safety
from backend.app.stage1.orchestrator import run_stage1


def test_empty_problem_statement():
    valid, errors = validate_problem_statement("")
    assert not valid
    assert any("empty" in e.lower() for e in errors)


def test_too_short_problem_statement():
    valid, errors = validate_problem_statement("Hi")
    assert not valid


def test_too_long_problem_statement():
    valid, errors = validate_problem_statement("x" * 10_001)
    assert not valid
    assert any("length" in e.lower() for e in errors)


def test_valid_problem_statement():
    valid, errors = validate_problem_statement("Classify cancer using clinical data.")
    assert valid
    assert errors == []


def test_representation_safety_no_raw_text():
    """End-to-end: ensure no raw clinical text appears in Stage 1 output."""
    stmt = "Build a multimodal classification system for cancer using clinical, pathology, blood and text data."
    rep = run_stage1(stmt, write_outputs=False)
    rep_str = str(rep.model_dump())
    # These raw clinical phrases must never appear
    assert "patient history:" not in rep_str.lower()
    assert "surgical report:" not in rep_str.lower()


def test_safety_validation_passes_for_valid_rep():
    stmt = "Classify cancer using clinical data."
    rep = run_stage1(stmt, write_outputs=False)
    is_safe, violations = validate_representation_safety(rep)
    assert is_safe, f"Safety violations: {violations}"
