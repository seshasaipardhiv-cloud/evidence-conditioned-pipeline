"""
pipeline_selector.py

Stage 2C — Automatic Dataset-Conditioned Component Selector

Translates extracted SciBERT evidence scores and dataset characteristics
into an optimal, evidence-grounded machine learning pipeline specification.

Features:
  - NO hardcoded choices (e.g. no "always use XGBoost" or "always use ResNet-18").
  - Dynamically scores candidate backbones per modality based on:
      Score = Evidence_Score * Modality_Task_Weight * Sample_Budget_Multiplier
  - Automatically selects:
      1. Tabular Model
      2. Image Model
      3. Text Model
      4. Preprocessing Strategy
      5. Sampling Strategy
      6. Multimodal Fusion Operator
      7. Loss Function
      8. Optimizer
      9. Ensemble Strategy
  - Maintains complete provenance: why each component won, evidence score,
    supporting PMIDs/DOIs, and alternative candidate rankings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.stage2.evidence_scoring import EvidenceScoreRecord

logger = logging.getLogger(__name__)


class SelectedComponentRecord(BaseModel):
    """Full decision and provenance record for an automatically selected component."""
    component_type: str                  # e.g. "image_model", "tabular_model", "sampling"
    selected_name: str                   # e.g. "ResNet-18", "XGBoost", "SMOTE"
    canonical_id: str                    # e.g. "resnet18", "xgboost", "smote"
    winning_score: float
    evidence_score: float
    supporting_pmids: List[str] = Field(default_factory=list)
    supporting_dois: List[str] = Field(default_factory=list)
    candidate_rankings: List[Dict[str, Any]] = Field(default_factory=list)
    decision_rationale: str
    provenance_trace: str


class PipelineSpecification(BaseModel):
    """Complete synthesized pipeline configuration synthesized from research evidence."""
    pipeline_id: str
    target_modalities: List[str]
    sample_count: int
    task_type: str
    compute_budget: str
    selected_components: Dict[str, SelectedComponentRecord]
    total_evidence_score: float
    synthesis_timestamp: str


class AutomaticPipelineSelector:
    """
    Synthesizes optimal pipeline components conditioned on extracted research evidence
    and target dataset characteristics.
    """

    # Catalog of available candidates for each component slot
    CANDIDATE_CATALOG = {
        "tabular_model": [
            {"id": "xgboost", "name": "XGBoost", "aliases": ["xgboost", "xgb", "gradient boosting"], "min_n": 20, "light_budget": True},
            {"id": "random_forest", "name": "Random Forest", "aliases": ["random forest", "rf"], "min_n": 15, "light_budget": True},
            {"id": "logistic_regression", "name": "Logistic Regression", "aliases": ["logistic regression", "linear model"], "min_n": 10, "light_budget": True},
            {"id": "mlp_tabular", "name": "Tabular MLP", "aliases": ["mlp", "neural network", "deep tabular"], "min_n": 50, "light_budget": False},
        ],
        "image_model": [
            {"id": "resnet18", "name": "ResNet-18", "aliases": ["resnet-18", "resnet18", "resnet"], "min_n": 20, "light_budget": True},
            {"id": "resnet50", "name": "ResNet-50", "aliases": ["resnet-50", "resnet50"], "min_n": 60, "light_budget": False},
            {"id": "efficientnet_b0", "name": "EfficientNet-B0", "aliases": ["efficientnet", "efficientnet-b0"], "min_n": 25, "light_budget": True},
            {"id": "vit_small", "name": "Vision Transformer (ViT)", "aliases": ["vit", "vision transformer", "transformer"], "min_n": 80, "light_budget": False},
        ],
        "text_model": [
            {"id": "pubmedbert", "name": "PubMedBERT", "aliases": ["pubmedbert", "bert", "biomedical bert"], "min_n": 20, "light_budget": True},
            {"id": "clinicalbert", "name": "ClinicalBERT", "aliases": ["clinicalbert", "clinical bert"], "min_n": 30, "light_budget": True},
            {"id": "tfidf_mlp", "name": "TF-IDF + Dense MLP", "aliases": ["tf-idf", "tfidf", "bow"], "min_n": 10, "light_budget": True},
        ],
        "preprocessing": [
            {"id": "mice_imputation", "name": "MICE Imputation", "aliases": ["mice", "imputation", "iterative imputer"], "light_budget": True},
            {"id": "standard_scaler", "name": "Standard Scaling", "aliases": ["standardization", "z-score", "normalization"], "light_budget": True},
            {"id": "one_hot_encoding", "name": "One-Hot Encoding", "aliases": ["one-hot encoding", "one-hot", "ohe"], "light_budget": True},
        ],
        "sampling": [
            {"id": "smote", "name": "SMOTE", "aliases": ["smote", "oversampling", "synthetic minority"], "light_budget": True},
            {"id": "adasyn", "name": "ADASYN", "aliases": ["adasyn", "adaptive synthetic"], "light_budget": True},
            {"id": "random_oversample", "name": "Random Oversampling", "aliases": ["random oversampling", "bootstrap sampling"], "light_budget": True},
        ],
        "fusion": [
            {"id": "late_fusion", "name": "Late Fusion (Concatenation)", "aliases": ["late fusion", "concatenation", "feature fusion"], "light_budget": True},
            {"id": "cross_attention", "name": "Cross-Modal Attention", "aliases": ["cross-attention", "cross attention", "attention"], "light_budget": False},
            {"id": "gated_fusion", "name": "Gated Multimodal Fusion", "aliases": ["gated fusion", "gate"], "light_budget": True},
        ],
        "loss": [
            {"id": "binary_cross_entropy", "name": "Binary Cross-Entropy", "aliases": ["binary cross-entropy", "cross entropy", "cross-entropy", "bce"], "light_budget": True},
            {"id": "focal_loss", "name": "Focal Loss", "aliases": ["focal loss", "focal"], "light_budget": True},
        ],
        "optimizer": [
            {"id": "adamw", "name": "AdamW", "aliases": ["adamw", "adam with weight decay"], "light_budget": True},
            {"id": "adam", "name": "Adam", "aliases": ["adam", "adaptive moment estimation"], "light_budget": True},
            {"id": "sgd", "name": "SGD with Momentum", "aliases": ["sgd", "stochastic gradient descent", "sgd with momentum", "momentum"], "light_budget": True},
        ],
        "ensemble": [
            {"id": "gated_blend", "name": "Validation-Gated Blend", "aliases": ["gated blend", "blending", "weighted average"], "light_budget": True},
            {"id": "stacking", "name": "Super Learner Stacking", "aliases": ["stacking", "meta-learner"], "light_budget": True},
            {"id": "soft_voting", "name": "Soft Voting Ensemble", "aliases": ["bagging", "voting", "averaging"], "light_budget": True},
        ],
    }

    def select_pipeline(
        self,
        scored_evidence: Dict[str, EvidenceScoreRecord],
        modalities: List[str],
        sample_count: int = 50,
        class_imbalance: bool = True,
        task_type: str = "binary_classification",
        compute_budget: str = "LIGHT",
    ) -> PipelineSpecification:
        """
        Synthesizes the complete pipeline specification dynamically from evidence and dataset.
        """
        selected_components: Dict[str, SelectedComponentRecord] = {}

        # 1. Select Tabular Model if tabular modality is present
        if "tabular" in modalities:
            selected_components["tabular_model"] = self._select_best_candidate(
                component_type="tabular_model",
                candidates=self.CANDIDATE_CATALOG["tabular_model"],
                evidence=scored_evidence,
                sample_count=sample_count,
                compute_budget=compute_budget,
            )

        # 2. Select Image Model if image modality is present
        if "image" in modalities:
            selected_components["image_model"] = self._select_best_candidate(
                component_type="image_model",
                candidates=self.CANDIDATE_CATALOG["image_model"],
                evidence=scored_evidence,
                sample_count=sample_count,
                compute_budget=compute_budget,
            )

        # 3. Select Text Model if text modality is present
        if "text" in modalities:
            selected_components["text_model"] = self._select_best_candidate(
                component_type="text_model",
                candidates=self.CANDIDATE_CATALOG["text_model"],
                evidence=scored_evidence,
                sample_count=sample_count,
                compute_budget=compute_budget,
            )

        # 4. Preprocessing
        selected_components["preprocessing"] = self._select_best_candidate(
            component_type="preprocessing",
            candidates=self.CANDIDATE_CATALOG["preprocessing"],
            evidence=scored_evidence,
            sample_count=sample_count,
            compute_budget=compute_budget,
        )

        # 5. Sampling (conditioned on class_imbalance)
        if class_imbalance:
            selected_components["sampling"] = self._select_best_candidate(
                component_type="sampling",
                candidates=self.CANDIDATE_CATALOG["sampling"],
                evidence=scored_evidence,
                sample_count=sample_count,
                compute_budget=compute_budget,
            )

        # 6. Multimodal Fusion (if >= 2 modalities)
        if len(modalities) >= 2:
            selected_components["fusion"] = self._select_best_candidate(
                component_type="fusion",
                candidates=self.CANDIDATE_CATALOG["fusion"],
                evidence=scored_evidence,
                sample_count=sample_count,
                compute_budget=compute_budget,
            )

        # 7. Loss function (conditioned on imbalance)
        selected_components["loss"] = self._select_best_candidate(
            component_type="loss",
            candidates=self.CANDIDATE_CATALOG["loss"],
            evidence=scored_evidence,
            sample_count=sample_count,
            compute_budget=compute_budget,
            imbalance_preference="focal_loss" if class_imbalance else None,
        )

        # 8. Optimizer
        selected_components["optimizer"] = self._select_best_candidate(
            component_type="optimizer",
            candidates=self.CANDIDATE_CATALOG["optimizer"],
            evidence=scored_evidence,
            sample_count=sample_count,
            compute_budget=compute_budget,
        )

        # 9. Ensemble
        selected_components["ensemble"] = self._select_best_candidate(
            component_type="ensemble",
            candidates=self.CANDIDATE_CATALOG["ensemble"],
            evidence=scored_evidence,
            sample_count=sample_count,
            compute_budget=compute_budget,
        )

        total_ev_score = round(
            sum(c.evidence_score for c in selected_components.values()) / max(1, len(selected_components)), 4
        )

        from datetime import datetime, timezone
        return PipelineSpecification(
            pipeline_id=f"synth_pipeline_{abs(hash(str(modalities) + str(sample_count))) % 100000:05d}",
            target_modalities=modalities,
            sample_count=sample_count,
            task_type=task_type,
            compute_budget=compute_budget,
            selected_components=selected_components,
            total_evidence_score=total_ev_score,
            synthesis_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _select_best_candidate(
        self,
        component_type: str,
        candidates: List[Dict[str, Any]],
        evidence: Dict[str, EvidenceScoreRecord],
        sample_count: int,
        compute_budget: str,
        imbalance_preference: Optional[str] = None,
    ) -> SelectedComponentRecord:
        """
        Ranks all candidates for a component slot and returns the winning record with full audit trace.
        """
        ranked = []

        for cand in candidates:
            cand_id = cand["id"]
            cand_name = cand["name"]
            aliases = cand["aliases"]

            # 1. Base Evidence Score from SciBERT extraction
            ev_record = None
            for alias in aliases:
                if alias.lower() in evidence:
                    ev_record = evidence[alias.lower()]
                    break

            base_ev_score = ev_record.composite_score if ev_record else 0.45
            pmids = ev_record.supporting_pmids if ev_record else ["PMID:LiteratureBaseline"]
            dois = ev_record.supporting_dois if ev_record else []

            # 2. Dataset conditioning multipliers
            multiplier = 1.0

            # Sample count constraint
            min_n = cand.get("min_n", 0)
            if sample_count < min_n:
                multiplier *= 0.70  # penalty for data-hungry models on small cohort
            elif sample_count >= 100 and not cand.get("light_budget", True):
                multiplier *= 1.15  # reward capable architectures on larger cohorts

            # Compute budget constraint
            is_light = cand.get("light_budget", True)
            if compute_budget.upper() == "LIGHT" and not is_light:
                multiplier *= 0.65
            elif compute_budget.upper() in ["MEDIUM", "HEAVY"] and not is_light:
                multiplier *= 1.10

            # Imbalance bonus
            if imbalance_preference and cand_id == imbalance_preference:
                multiplier *= 1.25

            final_score = round(base_ev_score * multiplier, 4)

            ranked.append({
                "canonical_id": cand_id,
                "name": cand_name,
                "base_evidence_score": base_ev_score,
                "dataset_multiplier": round(multiplier, 3),
                "final_score": final_score,
                "supporting_pmids": pmids,
                "supporting_dois": dois,
                "evidence_status": "LITERATURE_GROUNDED" if ev_record else "BASELINE_GROUNDED",
            })

        # Sort descending by final score
        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        winner = ranked[0]

        rationale = (
            f"Selected '{winner['name']}' with evidence score {winner['base_evidence_score']:.3f} "
            f"and dataset conditioning factor {winner['dataset_multiplier']:.2f} (final score: {winner['final_score']:.4f}). "
            f"Outranked {len(ranked)-1} alternative candidate(s)."
        )

        trace = f"Evidence: {', '.join(winner['supporting_pmids'])} | Score: {winner['final_score']}"

        return SelectedComponentRecord(
            component_type=component_type,
            selected_name=winner["name"],
            canonical_id=winner["canonical_id"],
            winning_score=winner["final_score"],
            evidence_score=winner["base_evidence_score"],
            supporting_pmids=winner["supporting_pmids"],
            supporting_dois=winner["supporting_dois"],
            candidate_rankings=ranked,
            decision_rationale=rationale,
            provenance_trace=trace,
        )
