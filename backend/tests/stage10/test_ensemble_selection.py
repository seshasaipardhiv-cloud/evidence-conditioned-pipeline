"""
Tests for Ensemble Selection (Stage 10 - Objective F)
Verifies:
1. Average Ensembling probability aggregation
2. Weighted Ensembling response to validation performance
3. Single model ensemble dormancy
4. Multi-model ensemble activation
"""

import numpy as np
import pytest
from backend.app.multimodal.ensemble_selector import EnsembleSelector
from backend.app.multimodal.neural_components import AverageEnsemble, WeightedEnsemble


def test_ensemble_selection_and_execution():
    selector = EnsembleSelector()

    # Single candidate model must keep ensemble dormant
    sel_single = selector.select(candidate_count=1)
    assert sel_single["execution_status"] == "DORMANT"

    # Multi-model selection activates ensemble
    sel_multi = selector.select(candidate_count=2, validation_scores=[0.95, 0.82])
    assert sel_multi["execution_status"] == "EXECUTABLE"

    # Execution
    p1 = np.array([0.9, 0.1, 0.8])
    p2 = np.array([0.7, 0.3, 0.6])

    avg_ens = AverageEnsemble()
    p_avg = avg_ens.predict_proba([p1, p2])
    assert np.allclose(p_avg, np.array([0.8, 0.2, 0.7]))

    wt_ens = WeightedEnsemble(validation_scores=[0.95, 0.82])
    p_wt = wt_ens.predict_proba([p1, p2])
    assert len(p_wt) == 3
    # Model 1 has higher weight, so p_wt[0] should be closer to 0.9 than 0.7
    assert p_wt[0] > 0.8
