"""
Unit tests for Stage 11 Ensemble Benchmark
"""

import json
from pathlib import Path
import pytest
import numpy as np
from backend.app.stage11.ensemble_engine import EnsembleEngine


def test_soft_voting():
    engine = EnsembleEngine(random_seed=42)
    members = ["m1", "m2"]
    v_p = [np.array([0.2, 0.8]), np.array([0.4, 0.6])]
    t_p = [np.array([0.1, 0.9]), np.array([0.3, 0.7])]

    val_ens, test_ens, w = engine.soft_voting(members, v_p, t_p)
    assert np.allclose(val_ens, [0.3, 0.7])
    assert np.allclose(test_ens, [0.2, 0.8])
    assert w["m1"] == 0.5 and w["m2"] == 0.5


def test_val_performance_weighted_voting():
    engine = EnsembleEngine(random_seed=42)
    members = ["m1", "m2"]
    val_scores = [0.95, 0.80]
    v_p = [np.array([0.2, 0.8]), np.array([0.4, 0.6])]
    t_p = [np.array([0.1, 0.9]), np.array([0.3, 0.7])]

    val_ens, test_ens, w = engine.val_performance_weighted_voting(members, val_scores, v_p, t_p, temperature=1.0)
    assert w["m1"] > w["m2"]
    assert np.isclose(sum(w.values()), 1.0, atol=1e-3)


def test_rank_averaging():
    engine = EnsembleEngine(random_seed=42)
    members = ["m1", "m2"]
    v_p = [np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.4, 0.8])]
    t_p = [np.array([0.05, 0.55, 0.95]), np.array([0.15, 0.45, 0.85])]

    val_ens, test_ens, w = engine.rank_averaging(members, v_p, t_p)
    assert val_ens[0] < val_ens[1] < val_ens[2]
    assert test_ens[0] < test_ens[1] < test_ens[2]


def test_stacking_meta_model():
    engine = EnsembleEngine(random_seed=42)
    members = ["m1", "m2"]
    v_p = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.15, 0.25, 0.75, 0.85])]
    y_val = np.array([0, 0, 1, 1])
    t_p = [np.array([0.1, 0.85]), np.array([0.2, 0.8])]

    val_ens, test_ens, w, meta = engine.stacking_ensemble(members, v_p, y_val, t_p)
    assert len(test_ens) == 2
    assert test_ens[0] < test_ens[1]
    assert meta is not None


def test_ensemble_comparison_json_exists():
    path = Path("evidence/processed/stage11/stage11_ensemble_comparison.json")
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "VALIDATED"
    assert "ensembles" in data
    assert len(data["ensembles"]) >= 4
