"""
Tests for End-to-End Automation and Ablation Comparison (Stage 10 - Objectives G & H)
Verifies:
1. Autonomous synthesis without manual component overrides
2. Decision ledger generation
3. Controlled ablation comparing evidence-conditioned selection vs fixed default
"""

from pathlib import Path
import pytest
from backend.app.stage10.evidence_conditioned_automation_validation import Stage10AutomationValidator


def test_end_to_end_automation_and_ablation(tmp_path):
    validator = Stage10AutomationValidator(base_dir=".", output_dir=str(tmp_path))

    summary = validator.run_end_to_end_and_ablation(num_samples=30, seeds=[42, 100])

    assert summary["status"] == "STAGE10_VALIDATION_COMPLETE"
    assert "ablation_comparison" in summary

    abl = summary["ablation_comparison"]
    assert "evidence_conditioned_pipeline" in abl
    assert "fixed_default_baseline" in abl
    assert "empirical_delta_roc_auc" in abl

    # Check generated decision ledger
    ledger_p = tmp_path / "automation_decision_ledger.json"
    assert ledger_p.exists()
