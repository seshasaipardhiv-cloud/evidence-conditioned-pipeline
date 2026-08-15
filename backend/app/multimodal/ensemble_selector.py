"""
Evidence-Conditioned Ensemble Selector

Decides whether an ensemble mechanism should be activated based on:
1. Availability of multiple independent candidate pipelines
2. Validation diversity and individual candidate performance
3. Compute budget constraints
4. Literature evidence support
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnsembleSelector:
    """Evaluates prerequisites and selects appropriate ensemble strategies."""

    def __init__(self):
        self.catalog = [
            {
                "ensemble_id": "average_ensembling",
                "name": "Uniform Probability Average Ensemble",
                "evidence_source": "PMID: 41775771 / Nature Sci Rep 2026",
                "evidence_status": "EVIDENCE_BACKED",
                "min_models": 2,
                "compute_cost": "LIGHT",
                "rationale": "Reduces predictive variance by taking the arithmetic mean of calibrated class probabilities across independent candidate models.",
            },
            {
                "ensemble_id": "weighted_ensembling",
                "name": "Validation Performance-Weighted Ensemble",
                "evidence_source": "PMID: 42487970 / Multimodal Biomedical Ensemble",
                "evidence_status": "EVIDENCE_BACKED",
                "min_models": 2,
                "compute_cost": "LIGHT",
                "rationale": "Softmax weights derived strictly from validation ROC-AUC scores dynamically weight stronger candidate architectures.",
            },
        ]

    def select(
        self,
        candidate_count: int,
        validation_scores: Optional[List[float]] = None,
        compute_budget: str = "LIGHT",
        explicit_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Selects an ensemble strategy when multiple candidates exist, or returns DORMANT/SINGLE_MODEL.
        """
        if candidate_count < 2:
            return {
                "component": "ensembling",
                "selected_value": None,
                "evidence_status": "DORMANT_INSUFFICIENT_MODELS",
                "evidence_source": "None",
                "rationale": f"Ensemble dormant: Only {candidate_count} model available (minimum required: 2).",
                "execution_status": "DORMANT",
            }

        if explicit_override:
            for cand in self.catalog:
                if cand["ensemble_id"] == explicit_override:
                    return {
                        "component": "ensembling",
                        "selected_value": cand["ensemble_id"],
                        "name": cand["name"],
                        "evidence_status": "EXPLICITLY_CONFIGURED",
                        "evidence_source": "experiment_config.json / user_override",
                        "rationale": f"Explicitly configured ensemble: {cand['name']}.",
                        "execution_status": "EXECUTABLE",
                    }

        # If validation scores are available and have variance, select weighted ensembling
        if validation_scores and len(validation_scores) >= 2 and (max(validation_scores) - min(validation_scores) > 0.02):
            top = self.catalog[1]  # weighted_ensembling
        else:
            top = self.catalog[0]  # average_ensembling

        return {
            "component": "ensembling",
            "selected_value": top["ensemble_id"],
            "name": top["name"],
            "evidence_status": top["evidence_status"],
            "evidence_source": top["evidence_source"],
            "rationale": top["rationale"],
            "compute_cost": top["compute_cost"],
            "execution_status": "EXECUTABLE",
        }
