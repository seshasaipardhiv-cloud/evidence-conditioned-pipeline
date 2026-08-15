"""
Stage 11 Model Registry

Defines and registers individual candidate and alternative baseline machine learning models
for fair benchmarking on tabular and multimodal clinical datasets.

Classifies models into:
- EVIDENCE_BACKED: Literature-grounded pipeline architectures with explicit PMIDs
- BASELINE: Standard default benchmark models
- OPTIONAL_ALTERNATIVE: General machine learning classifiers
- EXPLICITLY_CONFIGURED: User-requested or experimental architectures

Includes graceful availability detection for optional dependencies (e.g. LightGBM, CatBoost).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    model_name: str
    display_name: str
    model_family: str
    evidence_status: str  # EVIDENCE_BACKED, BASELINE, OPTIONAL_ALTERNATIVE, EXPLICITLY_CONFIGURED
    provenance: str
    compute_tier: str  # LIGHT, MEDIUM, HIGH
    hyperparameters: Dict[str, Any]
    factory: Callable[[int], Any]
    is_available: bool = True
    requires_scaling: bool = False
    enabled: bool = True
    citation_pmid: Optional[str] = None


class ModelRegistry:
    """
    Central registry of individual alternative models and candidate pipelines.
    """

    def __init__(self):
        self._registry: Dict[str, ModelSpec] = {}
        self._register_default_models()

    def _register_default_models(self):
        # 1. Evidence-Conditioned Candidate Pipeline (Stage 5B Configured Model)
        self.register(
            ModelSpec(
                model_name="candidate_pipeline",
                display_name="Evidence-Conditioned Candidate",
                model_family="gradient_boosting",
                evidence_status="EVIDENCE_BACKED",
                provenance="Literature-derived pipeline with median/mode imputation, one-hot encoding, SMOTE, and tuned XGBoost.",
                compute_tier="LIGHT",
                citation_pmid="PMID: 41826845 / PMC Biomarkers 2026",
                hyperparameters={
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "eval_metric": "logloss",
                    "objective": "binary:logistic",
                },
                factory=lambda seed: XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    random_state=seed,
                    n_jobs=1,
                ),
                requires_scaling=False,
            )
        )

        # 2. Baseline: Default XGBoost
        self.register(
            ModelSpec(
                model_name="xgboost_default",
                display_name="Default XGBoost",
                model_family="gradient_boosting",
                evidence_status="BASELINE",
                provenance="Standard untuned XGBoost default baseline.",
                compute_tier="LIGHT",
                hyperparameters={"n_estimators": 50, "eval_metric": "logloss"},
                factory=lambda seed: XGBClassifier(
                    n_estimators=50,
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=1,
                ),
                requires_scaling=False,
            )
        )

        # 3. Baseline: Random Forest
        self.register(
            ModelSpec(
                model_name="random_forest",
                display_name="Random Forest",
                model_family="ensemble_tree",
                evidence_status="BASELINE",
                provenance="Standard Breiman Random Forest ensemble baseline.",
                compute_tier="LIGHT",
                hyperparameters={"n_estimators": 100, "max_depth": None},
                factory=lambda seed: RandomForestClassifier(
                    n_estimators=100,
                    random_state=seed,
                    n_jobs=1,
                ),
                requires_scaling=False,
            )
        )

        # 4. Baseline: Logistic Regression (Scaled)
        self.register(
            ModelSpec(
                model_name="logistic_regression",
                display_name="Logistic Regression",
                model_family="linear",
                evidence_status="BASELINE",
                provenance="L2-regularized logistic regression baseline with feature standardization.",
                compute_tier="LIGHT",
                hyperparameters={"penalty": "l2", "max_iter": 1000},
                factory=lambda seed: LogisticRegression(
                    penalty="l2",
                    max_iter=1000,
                    random_state=seed,
                ),
                requires_scaling=True,
            )
        )

        # 5. Alternative: Extra Trees
        self.register(
            ModelSpec(
                model_name="extra_trees",
                display_name="Extra Trees",
                model_family="ensemble_tree",
                evidence_status="OPTIONAL_ALTERNATIVE",
                provenance="Extremely Randomized Trees ensemble baseline.",
                compute_tier="LIGHT",
                hyperparameters={"n_estimators": 100},
                factory=lambda seed: ExtraTreesClassifier(
                    n_estimators=100,
                    random_state=seed,
                    n_jobs=1,
                ),
                requires_scaling=False,
            )
        )

        # 6. Alternative: HistGradientBoosting
        self.register(
            ModelSpec(
                model_name="hist_gradient_boosting",
                display_name="HistGradientBoosting",
                model_family="gradient_boosting",
                evidence_status="OPTIONAL_ALTERNATIVE",
                provenance="Histogram-based gradient boosting classifier baseline.",
                compute_tier="LIGHT",
                hyperparameters={"max_iter": 100},
                factory=lambda seed: HistGradientBoostingClassifier(
                    max_iter=100,
                    random_state=seed,
                ),
                requires_scaling=False,
            )
        )

        # 7. Alternative: Support Vector Machine (RBF kernel, probability enabled)
        self.register(
            ModelSpec(
                model_name="svm",
                display_name="Support Vector Machine",
                model_family="kernel_svm",
                evidence_status="OPTIONAL_ALTERNATIVE",
                provenance="Support Vector Classifier with RBF kernel and Platt scaling probabilities.",
                compute_tier="LIGHT",
                hyperparameters={"kernel": "rbf", "probability": True},
                factory=lambda seed: SVC(
                    kernel="rbf",
                    probability=True,
                    random_state=seed,
                ),
                requires_scaling=True,
            )
        )

        # 8. Alternative: K-Nearest Neighbors
        self.register(
            ModelSpec(
                model_name="knn",
                display_name="K-Nearest Neighbors",
                model_family="instance_based",
                evidence_status="OPTIONAL_ALTERNATIVE",
                provenance="Standard k-Nearest Neighbors classifier (k=5).",
                compute_tier="LIGHT",
                hyperparameters={"n_neighbors": 5},
                factory=lambda seed: KNeighborsClassifier(
                    n_neighbors=5,
                ),
                requires_scaling=True,
            )
        )

        # 9. Alternative: Decision Tree
        self.register(
            ModelSpec(
                model_name="decision_tree",
                display_name="Decision Tree",
                model_family="tree",
                evidence_status="OPTIONAL_ALTERNATIVE",
                provenance="Standard single CART decision tree with max depth 5.",
                compute_tier="LIGHT",
                hyperparameters={"max_depth": 5},
                factory=lambda seed: DecisionTreeClassifier(
                    max_depth=5,
                    random_state=seed,
                ),
                requires_scaling=False,
            )
        )

        # 10. Baseline: Simple MLP (Reference Baseline)
        self.register(
            ModelSpec(
                model_name="simple_mlp",
                display_name="Simple MLP",
                model_family="neural_network",
                evidence_status="BASELINE",
                provenance="Shallow 2-layer Multilayer Perceptron baseline.",
                compute_tier="LIGHT",
                hyperparameters={"hidden_layer_sizes": (64, 32), "max_iter": 50},
                factory=lambda seed: MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=50,
                    random_state=seed,
                ),
                requires_scaling=True,
            )
        )

        # 11. Optional: LightGBM (Graceful detection)
        try:
            import lightgbm as lgb
            self.register(
                ModelSpec(
                    model_name="lightgbm",
                    display_name="LightGBM",
                    model_family="gradient_boosting",
                    evidence_status="OPTIONAL_ALTERNATIVE",
                    provenance="LightGBM gradient boosting framework.",
                    compute_tier="LIGHT",
                    hyperparameters={"n_estimators": 100, "learning_rate": 0.05},
                    factory=lambda seed: lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.05,
                        random_state=seed,
                        verbose=-1,
                        n_jobs=1,
                    ),
                    is_available=True,
                    requires_scaling=False,
                )
            )
        except ImportError:
            self.register(
                ModelSpec(
                    model_name="lightgbm",
                    display_name="LightGBM",
                    model_family="gradient_boosting",
                    evidence_status="OPTIONAL_ALTERNATIVE",
                    provenance="LightGBM gradient boosting framework (Not installed).",
                    compute_tier="LIGHT",
                    hyperparameters={},
                    factory=lambda seed: None,
                    is_available=False,
                    enabled=False,
                )
            )

        # 12. Optional: CatBoost (Graceful detection)
        try:
            import catboost as cb
            self.register(
                ModelSpec(
                    model_name="catboost",
                    display_name="CatBoost",
                    model_family="gradient_boosting",
                    evidence_status="OPTIONAL_ALTERNATIVE",
                    provenance="CatBoost categorical gradient boosting framework.",
                    compute_tier="LIGHT",
                    hyperparameters={"iterations": 100, "learning_rate": 0.05},
                    factory=lambda seed: cb.CatBoostClassifier(
                        iterations=100,
                        learning_rate=0.05,
                        random_seed=seed,
                        verbose=0,
                        thread_count=1,
                    ),
                    is_available=True,
                    requires_scaling=False,
                )
            )
        except ImportError:
            self.register(
                ModelSpec(
                    model_name="catboost",
                    display_name="CatBoost",
                    model_family="gradient_boosting",
                    evidence_status="OPTIONAL_ALTERNATIVE",
                    provenance="CatBoost categorical gradient boosting framework (Not installed).",
                    compute_tier="LIGHT",
                    hyperparameters={},
                    factory=lambda seed: None,
                    is_available=False,
                    enabled=False,
                )
            )

    def register(self, spec: ModelSpec):
        self._registry[spec.model_name] = spec

    def get_model(self, name: str) -> Optional[ModelSpec]:
        return self._registry.get(name)

    def list_available_models(self) -> List[ModelSpec]:
        return [m for m in self._registry.values() if m.is_available and m.enabled]

    def list_all_model_specs(self) -> List[Dict[str, Any]]:
        specs = []
        for m in self._registry.values():
            specs.append({
                "model_name": m.model_name,
                "display_name": m.display_name,
                "model_family": m.model_family,
                "evidence_status": m.evidence_status,
                "provenance": m.provenance,
                "citation_pmid": m.citation_pmid,
                "compute_tier": m.compute_tier,
                "hyperparameters": m.hyperparameters,
                "is_available": m.is_available,
                "requires_scaling": m.requires_scaling,
                "enabled": m.enabled,
            })
        return specs
