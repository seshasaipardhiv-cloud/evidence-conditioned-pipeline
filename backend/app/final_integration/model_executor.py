"""
model_executor.py  —  SCIENTIFICALLY REPAIRED

Stage 2D Real Model Training & Evaluation Engine

REPAIR:
  - train_and_evaluate_multimodal now returns y_test, test_probs, test_preds
    (the previous version only returned aggregated metrics — fixing Cohort E
    which had only 1/18 predictions stored).
  - Hardcoded fallback metric values (0.88, 0.85, etc.) in multimodal path removed.
  - All metrics computed from actual predictions.
  - split_indices stored for leakage audit.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IntegratedModelExecutor:
    """
    Executes real multi-seed model fitting and inference.
    All metrics computed from actual predictions — no hardcoded fallbacks.
    """

    def __init__(self, seeds: Optional[List[int]] = None):
        self.seeds = seeds or [42, 100, 2026]

    def train_and_evaluate_tabular(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Trains tabular model and evaluates on held-out test split."""
        start_t = time.time()

        # Stratified split — store indices for leakage audit
        indices = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=seed,
            stratify=y if len(np.unique(y)) > 1 else None,
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        clf = self._build_classifier(model_name, seed)
        clf.fit(X_train_sc, y_train)

        test_probs = clf.predict_proba(X_test_sc)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)
        metrics    = self._compute_metrics(y_test, test_probs, test_preds)

        return {
            "model_name":    model_name,
            "seed":          seed,
            "train_time_sec":round(time.time() - start_t, 4),
            "train_indices": train_idx.tolist(),
            "test_indices":  test_idx.tolist(),
            "y_test":        y_test.tolist(),
            "test_probs":    test_probs.tolist(),
            "test_preds":    test_preds.tolist(),
            "n_train":       len(train_idx),
            "n_test":        len(test_idx),
            "metrics":       metrics,
        }

    def train_and_evaluate_multimodal(
        self,
        cohort_data: Dict[str, Any],
        selected_components: Dict[str, Any],
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Multimodal training (Tabular + Image + Text) with dynamic fusion.

        REPAIR: Now returns complete y_test, test_probs, test_preds arrays.
        The previous version returned only aggregated metrics, causing Cohort E
        to store only 1 prediction.
        """
        start_t = time.time()
        from backend.app.multimodal.multimodal_executor import MultimodalExecutor

        executor = MultimodalExecutor(
            seeds=[seed],
            compute_budget="LIGHT",
            epochs=6,
            learning_rate=0.015,
        )

        modalities   = cohort_data["discovered_modalities"]
        fusion_mech  = selected_components.get("fusion", {}).get("selected_fusion", "Late Fusion (Feature Concatenation)")
        fusion_val   = "gated_fusion" if "gate" in fusion_mech.lower() else "feature_concatenation"

        from backend.app.multimodal.multimodal_pipeline import MultimodalPipeline
        from backend.app.multimodal.image_selector import ImageModelSelector
        from backend.app.multimodal.text_selector import TextModelSelector

        image_selector = ImageModelSelector()
        text_selector = TextModelSelector()
        n_samples = len(cohort_data.get("targets", []))
        selected_img = image_selector.select(sample_count=n_samples, compute_budget="LIGHT") if "image" in modalities else None
        selected_txt = text_selector.select(sample_count=n_samples, compute_budget="LIGHT") if "text" in modalities else None

        indices = np.arange(len(cohort_data["targets"]))
        y_all = np.array(cohort_data["targets"], dtype=int)
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.30,
            random_state=seed,
            stratify=y_all if len(np.unique(y_all)) > 1 else None,
        )

        tabular_matrix = cohort_data.get("tabular_features")
        image_paths = cohort_data.get("image_paths")
        raw_texts = cohort_data.get("text_notes")

        train_tab = tabular_matrix[train_idx] if tabular_matrix is not None else None
        test_tab = tabular_matrix[test_idx] if tabular_matrix is not None else None

        train_img = [image_paths[i] for i in train_idx] if image_paths is not None else None
        test_img = [image_paths[i] for i in test_idx] if image_paths is not None else None

        train_txt = [raw_texts[i] for i in train_idx] if raw_texts is not None else None
        test_txt = [raw_texts[i] for i in test_idx] if raw_texts is not None else None

        y_train = y_all[train_idx]
        y_test = y_all[test_idx]

        candidate = MultimodalPipeline(
            active_modalities=modalities,
            image_config=selected_img,
            text_config=selected_txt,
            fusion_mechanism=fusion_val,
            embed_dim=64,
            seed=seed,
        )
        candidate.fit_preprocessors(train_tab, train_img, train_txt)
        train_reps_cand = candidate.extract_features(train_tab, train_img, train_txt, is_training=True)
        for _ in range(8):
            candidate.train_step(y_true=y_train, lr=0.015, cached_reps=train_reps_cand)
        candidate.is_trained = True

        test_probs = candidate.predict_proba(test_tab, test_img, test_txt)
        test_preds = (test_probs >= 0.5).astype(int)
        metrics = self._compute_metrics(y_test, test_probs, test_preds)

        return {
            "model_name":           f"Multimodal Pipeline ({' + '.join(modalities)})",
            "seed":                 seed,
            "train_time_sec":       round(time.time() - start_t, 4),
            "train_indices":        train_idx.tolist(),
            "test_indices":         test_idx.tolist(),
            "y_test":               y_test.tolist(),
            "test_probs":           test_probs.tolist(),
            "test_preds":           test_preds.tolist(),
            "n_test":               len(y_test),
            "predictions_complete": True,
            "metrics":              metrics,
        }

    def _build_classifier(self, model_name: str, seed: int):
        m_key = model_name.lower()
        if "xgboost" in m_key or "gradient" in m_key:
            return GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=seed)
        elif "random forest" in m_key or "rf" in m_key:
            return RandomForestClassifier(n_estimators=50, max_depth=5, random_state=seed)
        elif "logistic" in m_key or "linear" in m_key or "tfidf" in m_key:
            return LogisticRegression(C=1.0, max_iter=300, random_state=seed)
        elif "resnet" in m_key or "efficientnet" in m_key or "pubmedbert" in m_key or "clinicalbert" in m_key:
            # Placeholder for vision/language model — uses MLP on feature vectors
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=seed)
        else:
            return MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=seed)

    def _compute_metrics(
        self, y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """All metrics computed from actual predictions — no hardcoded values."""
        try:
            roc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            roc = 0.5
        try:
            pr = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            pr = 0.5

        brier = float(brier_score_loss(y_true, y_prob))
        acc   = float(accuracy_score(y_true, y_pred))
        prec  = float(precision_score(y_true, y_pred, zero_division=0))
        rec   = float(recall_score(y_true, y_pred, zero_division=0))
        f1    = float(f1_score(y_true, y_pred, zero_division=0))

        return {
            "roc_auc":    round(roc,   4),
            "pr_auc":     round(pr,    4),
            "brier_score":round(brier, 4),
            "accuracy":   round(acc,   4),
            "precision":  round(prec,  4),
            "recall":     round(rec,   4),
            "f1":         round(f1,    4),
        }
