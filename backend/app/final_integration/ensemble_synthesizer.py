"""
ensemble_synthesizer.py

Stage 2D Explicit-Member Ensemble Synthesizer

Combines individual baseline models into principled ensembles using validation-derived weights:
  1. Individual baseline training (XGBoost, Random Forest, Logistic Regression)
  2. Validation-performance weighting (Softmax over validation ROC-AUC)
  3. Strict isolation of test set (zero test leakage)
  4. Explicit tracking and reporting of all constituent ensemble members.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ExplicitEnsembleSynthesizer:
    """
    Synthesizes and evaluates ensembles with explicit model membership.
    """

    def synthesize_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        member_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Trains constituent models, computes validation weights, and evaluates individual vs ensemble performance.
        """
        if member_names is None:
            member_names = ["XGBoost", "Random Forest", "Logistic Regression"]

        # 60% Train, 20% Validation, 20% Test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.20, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val if len(np.unique(y_train_val)) > 1 else None
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        X_test_sc = scaler.transform(X_test)

        individual_results: Dict[str, Dict[str, Any]] = {}
        val_probs_list = []
        test_probs_list = []
        val_scores = []

        for name in member_names:
            clf = self._instantiate_model(name, seed)
            clf.fit(X_train_sc, y_train)

            v_prob = clf.predict_proba(X_val_sc)[:, 1]
            t_prob = clf.predict_proba(X_test_sc)[:, 1]

            try:
                v_roc = float(roc_auc_score(y_val, v_prob)) if len(np.unique(y_val)) > 1 else 0.5
            except Exception:
                v_roc = 0.5

            t_pred = (t_prob >= 0.5).astype(int)
            t_metrics = self._calc_metrics(y_test, t_prob, t_pred)

            individual_results[name] = {
                "val_roc_auc": round(v_roc, 4),
                "test_metrics": t_metrics,
                "test_probs": t_prob.tolist(),
            }

            val_probs_list.append(v_prob)
            test_probs_list.append(t_prob)
            val_scores.append(v_roc)

        # Compute Validation Softmax Weights
        scores_arr = np.array(val_scores)
        exp_s = np.exp(scores_arr - np.max(scores_arr))
        weights = exp_s / np.sum(exp_s)
        weights_dict = {name: round(float(w), 4) for name, w in zip(member_names, weights)}

        # Weighted test ensemble
        ens_test_prob = np.zeros_like(test_probs_list[0])
        for w, tp in zip(weights, test_probs_list):
            ens_test_prob += w * tp

        ens_test_pred = (ens_test_prob >= 0.5).astype(int)
        ens_metrics = self._calc_metrics(y_test, ens_test_prob, ens_test_pred)

        ensemble_label = f"Ensemble: {' + '.join(member_names)}"

        return {
            "ensemble_label": ensemble_label,
            "ensemble_method": "Validation-Performance-Weighted Averaging",
            "member_models": member_names,
            "member_weights": weights_dict,
            "individual_results": individual_results,
            "ensemble_metrics": ens_metrics,
            "ensemble_test_probs": ens_test_prob.tolist(),
            "y_test": y_test.tolist(),
            "seed": seed,
        }

    def _instantiate_model(self, name: str, seed: int):
        m_key = name.lower()
        if "xgboost" in m_key or "gradient" in m_key:
            return GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=seed)
        elif "random forest" in m_key:
            return RandomForestClassifier(n_estimators=40, max_depth=4, random_state=seed)
        elif "logistic" in m_key:
            return LogisticRegression(C=1.0, max_iter=200, random_state=seed)
        return GradientBoostingClassifier(n_estimators=30, random_state=seed)

    def _calc_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        try:
            roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            roc = 0.5
        try:
            pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            pr = 0.5
        brier = float(brier_score_loss(y_true, y_prob))
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        return {
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "brier_score": round(brier, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
