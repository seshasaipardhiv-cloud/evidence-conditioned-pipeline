"""
Unit tests for Stage 11 Model Registry
"""

import pytest
from backend.app.stage11.model_registry import ModelRegistry, ModelSpec


def test_registry_initialization():
    registry = ModelRegistry()
    available = registry.list_available_models()
    assert len(available) >= 9

    names = [m.model_name for m in available]
    assert "candidate_pipeline" in names
    assert "xgboost_default" in names
    assert "random_forest" in names
    assert "logistic_regression" in names
    assert "extra_trees" in names
    assert "hist_gradient_boosting" in names
    assert "svm" in names
    assert "knn" in names
    assert "decision_tree" in names


def test_candidate_pipeline_spec():
    registry = ModelRegistry()
    cand = registry.get_model("candidate_pipeline")
    assert cand is not None
    assert cand.evidence_status == "EVIDENCE_BACKED"
    assert cand.citation_pmid is not None
    assert "41826845" in cand.citation_pmid
    assert cand.compute_tier == "LIGHT"
    assert cand.hyperparameters["n_estimators"] == 100


def test_baseline_specs():
    registry = ModelRegistry()
    lr = registry.get_model("logistic_regression")
    assert lr.evidence_status == "BASELINE"
    assert lr.requires_scaling is True

    rf = registry.get_model("random_forest")
    assert rf.evidence_status == "BASELINE"
    assert rf.requires_scaling is False


def test_optional_models_graceful_availability():
    registry = ModelRegistry()
    lgb = registry.get_model("lightgbm")
    cb = registry.get_model("catboost")
    assert lgb is not None
    assert cb is not None
    # Availability reflects environment without throwing exceptions
    assert isinstance(lgb.is_available, bool)
    assert isinstance(cb.is_available, bool)


def test_model_factory_returns_classifier():
    registry = ModelRegistry()
    for m in registry.list_available_models():
        clf = m.factory(42)
        assert clf is not None
        assert hasattr(clf, "fit")
        assert hasattr(clf, "predict_proba")
