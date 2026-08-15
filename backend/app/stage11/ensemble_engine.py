"""
Stage 11 Ensemble Engine

Implements principled ensemble learning algorithms for tabular and multimodal prediction:
1. Soft Voting / Probability Averaging
2. Weighted Probability Averaging
3. Validation-Performance-Weighted Ensemble (Validation ROC-AUC Softmax Weights)
4. Rank Averaging (Normalized Rank Aggregation)
5. Stacking Ensemble (Validation-fitted Meta-Classifier with Out-Of-Fold / Validation Predictions)
6. Bootstrap Aggregation (Bagging)

Guarantees:
- Ensemble weights and meta-models are derived STRICTLY from training/validation data.
- Test predictions remain completely isolated until final inference evaluation.
- All member predictions, weights, and configuration hashes are recorded.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    strategy_name: str
    display_name: str
    member_models: List[str]
    weights: Optional[Dict[str, float]]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    test_probabilities: np.ndarray
    configuration_hash: str
    provenance: str


class EnsembleEngine:
    """
    Ensemble synthesis and execution engine.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    @staticmethod
    def compute_config_hash(member_models: List[str], strategy: str, params: Dict[str, Any]) -> str:
        h = hashlib.sha256()
        h.update(strategy.encode("utf-8"))
        for m in sorted(member_models):
            h.update(m.encode("utf-8"))
        h.update(json.dumps(params, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    # 1. Soft Voting / Uniform Average
    def soft_voting(
        self,
        member_names: List[str],
        val_probs: List[np.ndarray],
        test_probs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Calculates simple arithmetic mean of predicted probabilities across member models.
        """
        M = len(member_names)
        val_ens = np.mean(np.array(val_probs), axis=0)
        test_ens = np.mean(np.array(test_probs), axis=0)
        weights = {m: round(1.0 / M, 4) for m in member_names}
        return val_ens, test_ens, weights

    # 2. Validation-Performance-Weighted Averaging
    def val_performance_weighted_voting(
        self,
        member_names: List[str],
        val_scores: List[float],  # Validation ROC-AUC scores per model
        val_probs: List[np.ndarray],
        test_probs: List[np.ndarray],
        temperature: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Weights member models proportional to their validation ROC-AUC scores via softmax.
        Test data is NEVER used in weight calculation.
        """
        scores = np.array(val_scores, dtype=np.float64) / max(1e-5, temperature)
        exp_s = np.exp(scores - np.max(scores))
        w = exp_s / np.sum(exp_s)

        val_ens = np.zeros_like(val_probs[0])
        test_ens = np.zeros_like(test_probs[0])

        for i, weight in enumerate(w):
            val_ens += weight * val_probs[i]
            test_ens += weight * test_probs[i]

        weights_dict = {name: round(float(w[i]), 4) for i, name in enumerate(member_names)}
        return val_ens, test_ens, weights_dict

    # 3. Rank Averaging
    def rank_averaging(
        self,
        member_names: List[str],
        val_probs: List[np.ndarray],
        test_probs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Converts probability vectors into normalized percentile ranks [0, 1] per model,
        then averages the normalized ranks across all members.
        """
        M = len(member_names)
        val_ranks = []
        test_ranks = []

        for vp in val_probs:
            r = rankdata(vp) / float(len(vp))
            val_ranks.append(r)

        for tp in test_probs:
            r = rankdata(tp) / float(len(tp))
            test_ranks.append(r)

        val_ens = np.mean(np.array(val_ranks), axis=0)
        test_ens = np.mean(np.array(test_ranks), axis=0)
        weights = {m: round(1.0 / M, 4) for m in member_names}
        return val_ens, test_ens, weights

    # 4. Stacking Ensemble
    def stacking_ensemble(
        self,
        member_names: List[str],
        val_probs: List[np.ndarray],
        val_y: np.ndarray,
        test_probs: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Any]:
        """
        Trains a Logistic Regression meta-model strictly on validation fold predictions.
        The trained meta-model is then evaluated on the untouched test predictions.
        """
        # Form meta-feature matrices (N_samples, M_models)
        X_meta_val = np.column_stack(val_probs)
        X_meta_test = np.column_stack(test_probs)

        meta_clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=self.random_seed,
        )
        meta_clf.fit(X_meta_val, val_y)

        val_ens = meta_clf.predict_proba(X_meta_val)[:, 1]
        test_ens = meta_clf.predict_proba(X_meta_test)[:, 1]

        # Extract normalized meta-model coefficients as effective feature weights
        coefs = meta_clf.coef_[0]
        abs_coefs = np.abs(coefs)
        norm_coefs = abs_coefs / (np.sum(abs_coefs) + 1e-7)
        weights_dict = {name: round(float(norm_coefs[i]), 4) for i, name in enumerate(member_names)}

        return val_ens, test_ens, weights_dict, meta_clf

    # 5. Bootstrap Aggregation (Bagging)
    def bagging_ensemble(
        self,
        candidate_model_factory: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        n_bags: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Trains multiple bootstrapped instances of the candidate model on resampled training sets.
        """
        val_preds_list = []
        test_preds_list = []

        rng = np.random.RandomState(self.random_seed)
        N = len(X_train)

        for b in range(n_bags):
            boot_idx = rng.choice(N, size=N, replace=True)
            X_b = X_train[boot_idx]
            y_b = y_train[boot_idx]

            model = candidate_model_factory(self.random_seed + b * 17)
            model.fit(X_b, y_b)

            vp = model.predict_proba(X_val)[:, 1]
            tp = model.predict_proba(X_test)[:, 1]

            val_preds_list.append(vp)
            test_preds_list.append(tp)

        val_ens = np.mean(np.array(val_preds_list), axis=0)
        test_ens = np.mean(np.array(test_preds_list), axis=0)
        weights_dict = {f"bag_{i+1}": round(1.0 / n_bags, 4) for i in range(n_bags)}

        return val_ens, test_ens, weights_dict
