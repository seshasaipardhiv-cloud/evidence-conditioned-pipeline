"""
Unit tests for Stage 11 Prediction Storage and Integrity
"""

import json
from pathlib import Path
import pytest
import numpy as np


def test_prediction_files_exist_and_format():
    pred_dir = Path("evidence/processed/stage11/predictions")
    assert pred_dir.exists()
    pred_files = list(pred_dir.glob("*.json"))
    assert len(pred_files) >= 15

    for pf in pred_files[:5]:
        with open(pf, "r", encoding="utf-8") as f:
            records = json.load(f)
        assert len(records) > 0
        r0 = records[0]
        assert "patient_id" in r0
        assert "true_label" in r0
        assert "predicted_probability" in r0
        assert "predicted_class" in r0
        assert "model_name" in r0
        assert "seed" in r0
        assert "split" in r0
        assert r0["split"] == "test"
        assert 0.0 <= r0["predicted_probability"] <= 1.0
        assert r0["true_label"] in [0, 1]


def test_prediction_manifest_exists():
    path = Path("evidence/processed/stage11/stage11_predictions_manifest.json")
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["prediction_files_count"] >= 15
