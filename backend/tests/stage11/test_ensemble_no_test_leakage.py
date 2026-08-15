"""
Unit tests confirming zero test-set leakage in ensemble weighting and meta-learning
"""

import numpy as np
import pytest
from backend.app.stage11.ensemble_engine import EnsembleEngine


def test_ensemble_weights_derived_from_val_only():
    engine = EnsembleEngine(random_seed=42)
    members = ["m1", "m2"]
    val_scores = [0.98, 0.70]
    val_probs = [np.array([0.1, 0.9]), np.array([0.4, 0.6])]
    test_probs = [np.array([0.2, 0.8]), np.array([0.3, 0.7])]

    # Modify test_probs arbitrarily; ensure weights remain unchanged
    _, _, w1 = engine.val_performance_weighted_voting(members, val_scores, val_probs, test_probs)

    corrupted_test_probs = [np.array([0.9, 0.1]), np.array([0.8, 0.2])]
    _, _, w2 = engine.val_performance_weighted_voting(members, val_scores, val_probs, corrupted_test_probs)

    assert w1 == w2, "Ensemble weights mutated based on test probabilities!"


def test_stacking_meta_model_isolated_from_test():
    engine = EnsembleEngine(random_seed=42)
    members = ["m1", "m2"]
    val_probs = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.15, 0.25, 0.75, 0.85])]
    y_val = np.array([0, 0, 1, 1])

    test_probs_1 = [np.array([0.1, 0.85]), np.array([0.2, 0.8])]
    test_probs_2 = [np.array([0.99, 0.01]), np.array([0.95, 0.05])]

    _, _, _, meta1 = engine.stacking_ensemble(members, val_probs, y_val, test_probs_1)
    _, _, _, meta2 = engine.stacking_ensemble(members, val_probs, y_val, test_probs_2)

    assert np.allclose(meta1.coef_, meta2.coef_), "Stacking meta-model weights mutated based on test data!"
    assert np.isclose(meta1.intercept_, meta2.intercept_), "Stacking intercept mutated based on test data!"
