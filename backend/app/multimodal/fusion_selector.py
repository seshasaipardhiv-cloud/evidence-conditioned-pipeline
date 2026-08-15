"""
Evidence-Conditioned Multimodal Fusion Selector

Ranks and selects candidate multimodal fusion architectures based on literature evidence,
modality compatibility, sample size, alignment requirements, and compute tiers.
Maintains auditable provenance for every fusion decision.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Literature-grounded candidate fusion mechanisms with publication provenance
FUSION_CANDIDATE_CATALOG: List[Dict[str, Any]] = [
    {
        "fusion_id": "cross_attention",
        "name": "Bi-directional Multi-Head Cross-Attention Fusion",
        "evidence_source": "PMID: 42487970 / Multimodal Biomedical Deep Learning 2026",
        "evidence_status": "EVIDENCE_BACKED",
        "supported_modalities": [["image", "text"], ["tabular", "text"], ["tabular", "image"]],
        "min_samples": 50,
        "compute_cost": "LIGHT",
        "alignment_type": "dense_cross_attention",
        "rationale": "Enables asymmetric query-key-value cross-attention between imaging tokens and pathology text narratives with residual gating.",
    },
    {
        "fusion_id": "gated_fusion",
        "name": "Learned Dynamic Gated Multimodal Fusion",
        "evidence_source": "PMID: 41775771 / Nature Sci Rep 2026",
        "evidence_status": "EVIDENCE_BACKED",
        "supported_modalities": [
            ["image", "text"],
            ["tabular", "image"],
            ["tabular", "text"],
            ["tabular", "image", "text"],
        ],
        "min_samples": 40,
        "compute_cost": "LIGHT",
        "alignment_type": "dynamic_weighting",
        "rationale": "Learns input-dependent gating coefficients dynamically balancing representation contributions across arbitrary number of modalities.",
    },
    {
        "fusion_id": "feature_concatenation",
        "name": "Feature Concatenation with Subspace Projection",
        "evidence_source": "PMID: 41826845 / PMC Biomarkers 2026",
        "evidence_status": "EVIDENCE_BACKED",
        "supported_modalities": [
            ["image", "text"],
            ["tabular", "image"],
            ["tabular", "text"],
            ["tabular", "image", "text"],
        ],
        "min_samples": 20,
        "compute_cost": "LIGHT",
        "alignment_type": "joint_projection",
        "rationale": "Concatenates all modality feature vectors into a single unified space followed by joint linear transformation and non-linearity.",
    },
    {
        "fusion_id": "late_fusion",
        "name": "Late Probability Blending Fusion",
        "evidence_source": "Standard Minimal Reference Baseline",
        "evidence_status": "EXPLICITLY_CONFIGURED",
        "supported_modalities": [
            ["image", "text"],
            ["tabular", "image"],
            ["tabular", "text"],
            ["tabular", "image", "text"],
        ],
        "min_samples": 10,
        "compute_cost": "LIGHT",
        "alignment_type": "decision_level",
        "rationale": "Trains separate modality-specific task heads and aggregates final output probabilities.",
    },
    {
        "fusion_id": "early_fusion",
        "name": "Early Feature Stacking Fusion",
        "evidence_source": "General Multimodal Architecture Taxonomy",
        "evidence_status": "EXPLICITLY_CONFIGURED",
        "supported_modalities": [["tabular", "image"], ["tabular", "text"]],
        "min_samples": 30,
        "compute_cost": "MEDIUM",
        "alignment_type": "input_level",
        "rationale": "Combines raw or early-stage feature embeddings at the input layer before deep encoding.",
    },
]


class FusionSelector:
    """Selects and ranks multimodal fusion mechanisms based on evidence and modality characteristics."""

    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None):
        self.catalog = catalog or FUSION_CANDIDATE_CATALOG

    def select(
        self,
        active_modalities: List[str],
        sample_count: int = 100,
        compute_budget: str = "LIGHT",
        explicit_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ranks and selects candidate fusion mechanisms based on evidence, sample size, and modality fit.
        """
        if len(active_modalities) < 2:
            return {
                "component": "multimodal_fusion",
                "selected_value": None,
                "evidence_status": "NOT_APPLICABLE_UNIMODAL",
                "evidence_source": "None (Single Modality)",
                "rationale": f"Unimodal pipeline ({active_modalities}) does not require multimodal fusion.",
                "execution_status": "UNIMODAL",
                "selection_rankings": [],
            }

        # 1. Check explicit override
        if explicit_override:
            for cand in self.catalog:
                if cand["fusion_id"] == explicit_override:
                    return {
                        "component": "multimodal_fusion",
                        "selected_value": cand["fusion_id"],
                        "name": cand["name"],
                        "evidence_status": "EXPLICITLY_CONFIGURED",
                        "evidence_source": "experiment_config.json / user_override",
                        "rationale": f"Explicitly configured fusion mechanism: {cand['name']}.",
                        "compute_cost": cand["compute_cost"],
                        "execution_status": "EXECUTABLE",
                        "selection_rankings": [],
                    }

        # 2. Score candidates
        scored_candidates = []
        budget_rank = {"LIGHT": 1, "MEDIUM": 2, "HEAVY": 3}
        allowed_budget_val = budget_rank.get(compute_budget.upper(), 1)
        mod_set = set(active_modalities)

        for cand in self.catalog:
            cand_budget_val = budget_rank.get(cand["compute_cost"], 2)

            if cand_budget_val > allowed_budget_val:
                scored_candidates.append({
                    "fusion_id": cand["fusion_id"],
                    "name": cand["name"],
                    "score": 0.0,
                    "status": "REJECTED_COMPUTE_BUDGET",
                    "reason": f"Compute cost ({cand['compute_cost']}) exceeds budget ({compute_budget}).",
                })
                continue

            # Modality compatibility check
            is_compat = any(set(supp) == mod_set for supp in cand["supported_modalities"])
            if not is_compat:
                scored_candidates.append({
                    "fusion_id": cand["fusion_id"],
                    "name": cand["name"],
                    "score": 0.0,
                    "status": "REJECTED_MODALITY_INCOMPATIBLE",
                    "reason": f"Active modalities {active_modalities} not supported by {cand['name']}.",
                })
                continue

            # Sample size check
            if sample_count < cand["min_samples"]:
                scored_candidates.append({
                    "fusion_id": cand["fusion_id"],
                    "name": cand["name"],
                    "score": 0.2,
                    "status": "REJECTED_SAMPLE_SIZE",
                    "reason": f"Sample count ({sample_count}) below minimum ({cand['min_samples']}).",
                })
                continue

            score = 1.0
            if cand["evidence_status"] == "EVIDENCE_BACKED":
                score += 1.5
            else:
                score += 0.5

            # Alignment bonuses
            if mod_set == {"image", "text"} and cand["fusion_id"] == "cross_attention":
                score += 1.2
            elif len(active_modalities) == 3 and cand["fusion_id"] == "gated_fusion":
                score += 1.2
            elif cand["fusion_id"] == "feature_concatenation":
                score += 0.8

            scored_candidates.append({
                "fusion_id": cand["fusion_id"],
                "name": cand["name"],
                "score": round(score, 3),
                "status": "ADMISSIBLE",
                "evidence_source": cand["evidence_source"],
                "evidence_status": cand["evidence_status"],
                "compute_cost": cand["compute_cost"],
                "rationale": cand["rationale"],
                "cand_ref": cand,
            })

        admissible = [c for c in scored_candidates if c["status"] == "ADMISSIBLE"]
        admissible.sort(key=lambda x: x["score"], reverse=True)

        if not admissible:
            return {
                "component": "multimodal_fusion",
                "selected_value": "feature_concatenation",
                "name": "Feature Concatenation Fallback",
                "evidence_status": "EXPLICITLY_CONFIGURED",
                "evidence_source": "Robust Default Fallback",
                "rationale": "Fallback to feature concatenation due to constraints.",
                "execution_status": "EXECUTABLE",
                "selection_rankings": scored_candidates,
            }

        top = admissible[0]
        return {
            "component": "multimodal_fusion",
            "selected_value": top["fusion_id"],
            "name": top["name"],
            "evidence_status": top["evidence_status"],
            "evidence_source": top["evidence_source"],
            "rationale": top["rationale"],
            "compute_cost": top["compute_cost"],
            "score": top["score"],
            "execution_status": "EXECUTABLE",
            "selection_rankings": scored_candidates,
        }
