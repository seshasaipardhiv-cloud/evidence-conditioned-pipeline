"""
Unit tests confirming historical Stage 5B/6A/10/10.5 empirical immutability
"""

import json
from pathlib import Path
import pytest


def test_stage5b_candidate_roc_auc_immutable():
    p = Path("evidence/processed/stage5b_candidate_results.json")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    rocs = [m["roc_auc"] for m in data["test_metrics"]]
    assert round(sum(rocs)/len(rocs), 4) == 0.9751
    assert rocs == [0.9888, 0.9609, 0.9756]


def test_stage5b_default_xgboost_roc_auc_immutable():
    p = Path("evidence/processed/stage5b_baseline_results.json")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    xgb_metrics = data["baseline_xgboost_default"]["test_metrics"]
    rocs = [m["roc_auc"] for m in xgb_metrics]
    assert round(sum(rocs)/len(rocs), 4) == 0.9704


def test_stage5c_ablation_values_immutable():
    p = Path("evidence/metadata/stage5c_ablation_results.json")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["ablations"]["ablation_full_candidate"]["mean_roc_auc"] == 0.9751
    assert data["ablations"]["ablation_no_smote"]["mean_roc_auc"] == 0.9773
    assert data["ablations"]["ablation_no_advanced_imputation"]["mean_roc_auc"] == 0.9767
    assert data["ablations"]["ablation_ordinal_encoding"]["mean_roc_auc"] == 0.9784
