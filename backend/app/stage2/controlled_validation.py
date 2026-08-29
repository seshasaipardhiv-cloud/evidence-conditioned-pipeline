"""
controlled_validation.py

Stage 2C — Controlled Evidence-Switching Validation Harness

Empirically validates that changes in the scientific evidence corpus directly
and automatically change the synthesized pipeline decisions.

Tests two controlled evidence corpora:
  - Scenario A: Literature corpus favoring ResNet-18, XGBoost, SMOTE, Binary Cross-Entropy, AdamW.
  - Scenario B: Literature corpus favoring EfficientNet-B0, Random Forest, ADASYN, Focal Loss, SGD.

Demonstrates:
  Evidence Profile A ──▶ Pipeline Decision A
  Evidence Profile B ──▶ Pipeline Decision B (Dynamic Automatic Switching)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from backend.app.stage2.evidence_scoring import EvidenceScoringEngine
from backend.app.stage2.models import ExtractionMethod, NEREntity, PaperRecord, RelationRecord
from backend.app.stage2.pipeline_selector import AutomaticPipelineSelector, PipelineSpecification

logger = logging.getLogger(__name__)


class ControlledEvidenceValidator:
    """
    Executes controlled experiments to prove evidence-driven decision switching.
    """

    def run_evidence_switching_experiment(self) -> Dict[str, Any]:
        """
        Runs both Scenario A and Scenario B and compares resulting pipeline specifications.
        """
        logger.info("Executing Controlled Evidence-Switching Experiment (Scenario A vs B)...")

        # -------------------------------------------------------------
        # Scenario A: Corpus favoring ResNet-18, XGBoost, SMOTE
        # -------------------------------------------------------------
        papers_a, entities_a, relations_a = self._build_scenario_a()
        scoring_engine = EvidenceScoringEngine()
        evidence_scores_a = scoring_engine.score_corpus_evidence(
            entities=entities_a,
            relations=relations_a,
            papers=papers_a,
        )
        selector = AutomaticPipelineSelector()
        spec_a = selector.select_pipeline(
            scored_evidence=evidence_scores_a,
            modalities=["tabular", "image"],
            sample_count=50,
            compute_budget="LIGHT",
        )

        # -------------------------------------------------------------
        # Scenario B: Corpus favoring EfficientNet-B0, Random Forest, ADASYN
        # -------------------------------------------------------------
        papers_b, entities_b, relations_b = self._build_scenario_b()
        evidence_scores_b = scoring_engine.score_corpus_evidence(
            entities=entities_b,
            relations=relations_b,
            papers=papers_b,
        )
        spec_b = selector.select_pipeline(
            scored_evidence=evidence_scores_b,
            modalities=["tabular", "image"],
            sample_count=50,
            compute_budget="LIGHT",
        )

        # Compare decisions
        comparisons = []
        for comp_key in ["tabular_model", "image_model", "sampling", "loss", "optimizer"]:
            sel_a = spec_a.selected_components.get(comp_key)
            sel_b = spec_b.selected_components.get(comp_key)

            name_a = sel_a.selected_name if sel_a else "N/A"
            name_b = sel_b.selected_name if sel_b else "N/A"
            score_a = sel_a.winning_score if sel_a else 0.0
            score_b = sel_b.winning_score if sel_b else 0.0

            switched = (name_a != name_b)
            comparisons.append({
                "component_slot": comp_key,
                "scenario_a_selection": name_a,
                "scenario_a_score": score_a,
                "scenario_b_selection": name_b,
                "scenario_b_score": score_b,
                "decision_switched": switched,
            })

        all_switched = all(c["decision_switched"] for c in comparisons)

        report = {
            "validation_name": "Controlled Evidence-Switching Validation",
            "scenario_a_title": "Literature Favoring ResNet-18 / XGBoost / SMOTE",
            "scenario_b_title": "Literature Favoring EfficientNet-B0 / Random Forest / ADASYN",
            "all_decisions_switched_dynamically": all_switched,
            "component_comparisons": comparisons,
            "scenario_a_spec": spec_a.model_dump(),
            "scenario_b_spec": spec_b.model_dump(),
        }

        logger.info(f"Evidence switching experiment completed. All decisions switched dynamically: {all_switched}")
        return report

    def _build_scenario_a(self) -> Tuple[List[PaperRecord], List[NEREntity], List[RelationRecord]]:
        """Constructs synthetic evidence state strongly supporting Suite A."""
        papers = [
            PaperRecord(
                paper_id="paper_scen_a_1",
                title="Residual Architectures and Gradient Boosting in Clinical Diagnostics",
                authors=["Smith et al."],
                publication_year=2026,
                source="PMC",
                pmid="42487970",
                full_text_available=True,
                retrieval_date="2026-08-29",
            ),
            PaperRecord(
                paper_id="paper_scen_a_2",
                title="SMOTE and Binary Cross-Entropy Optimization for Tabular Oncological Classification",
                authors=["Johnson et al."],
                publication_year=2026,
                source="PMC",
                pmid="38396486",
                full_text_available=True,
                retrieval_date="2026-08-29",
            ),
        ]

        entities = [
            # High evidence for ResNet-18
            self._mock_entity("ResNet-18", "MODEL_ARCH", "Representation", 0.95, "paper_scen_a_1", "42487970"),
            self._mock_entity("ResNet-18", "MODEL_ARCH", "Representation", 0.92, "paper_scen_a_2", "38396486"),
            # High evidence for XGBoost
            self._mock_entity("XGBoost", "MODEL_ARCH", "Representation", 0.94, "paper_scen_a_1", "42487970"),
            self._mock_entity("XGBoost", "MODEL_ARCH", "Representation", 0.91, "paper_scen_a_2", "38396486"),
            # High evidence for SMOTE
            self._mock_entity("SMOTE", "SAMPLING", "Sampling", 0.93, "paper_scen_a_2", "38396486"),
            # High evidence for Binary Cross-Entropy
            self._mock_entity("binary cross-entropy", "LOSS", "Loss", 0.94, "paper_scen_a_2", "38396486"),
            # High evidence for AdamW
            self._mock_entity("AdamW", "OPTIMIZATION", "Regularization", 0.91, "paper_scen_a_1", "42487970"),
        ]

        relations = [
            RelationRecord(
                relation_id="rel_a_1",
                entity_a_id="e1", entity_a_text="ResNet-18", entity_a_type="MODEL_ARCH",
                entity_b_id="e2", entity_b_text="binary cross-entropy", entity_b_type="LOSS",
                relation_type="HAS_LOSS", confidence=0.92, source_paper_id="paper_scen_a_1",
                source_sentence="We trained ResNet-18 with binary cross-entropy.",
            ),
            RelationRecord(
                relation_id="rel_a_2",
                entity_a_id="e3", entity_a_text="XGBoost", entity_a_type="MODEL_ARCH",
                entity_b_id="e4", entity_b_text="SMOTE", entity_b_type="SAMPLING",
                relation_type="HAS_SAMPLING", confidence=0.90, source_paper_id="paper_scen_a_2",
                source_sentence="XGBoost was applied after SMOTE preprocessing.",
            ),
        ]

        return papers, entities, relations

    def _build_scenario_b(self) -> Tuple[List[PaperRecord], List[NEREntity], List[RelationRecord]]:
        """Constructs synthetic evidence state strongly supporting Suite B."""
        papers = [
            PaperRecord(
                paper_id="paper_scen_b_1",
                title="Efficient Convolutional Architectures and Random Forests in Medical AI",
                authors=["Zhang et al."],
                publication_year=2026,
                source="PMC",
                pmid="49991111",
                full_text_available=True,
                retrieval_date="2026-08-29",
            ),
            PaperRecord(
                paper_id="paper_scen_b_2",
                title="ADASYN and Focal Loss Frameworks for Highly Imbalanced Diagnostics",
                authors=["Miller et al."],
                publication_year=2026,
                source="PMC",
                pmid="49992222",
                full_text_available=True,
                retrieval_date="2026-08-29",
            ),
        ]

        entities = [
            # High evidence for EfficientNet-B0
            self._mock_entity("EfficientNet-B0", "MODEL_ARCH", "Representation", 0.96, "paper_scen_b_1", "49991111"),
            self._mock_entity("EfficientNet-B0", "MODEL_ARCH", "Representation", 0.94, "paper_scen_b_2", "49992222"),
            # High evidence for Random Forest
            self._mock_entity("Random Forest", "MODEL_ARCH", "Representation", 0.95, "paper_scen_b_1", "49991111"),
            self._mock_entity("Random Forest", "MODEL_ARCH", "Representation", 0.93, "paper_scen_b_2", "49992222"),
            # High evidence for ADASYN
            self._mock_entity("ADASYN", "SAMPLING", "Sampling", 0.92, "paper_scen_b_2", "49992222"),
            # High evidence for Focal Loss
            self._mock_entity("Focal Loss", "LOSS", "Loss", 0.95, "paper_scen_b_2", "49992222"),
            # High evidence for SGD
            self._mock_entity("SGD with Momentum", "OPTIMIZATION", "Regularization", 0.90, "paper_scen_b_1", "49991111"),
        ]

        relations = [
            RelationRecord(
                relation_id="rel_b_1",
                entity_a_id="eb1", entity_a_text="EfficientNet-B0", entity_a_type="MODEL_ARCH",
                entity_b_id="eb2", entity_b_text="Focal Loss", entity_b_type="LOSS",
                relation_type="HAS_LOSS", confidence=0.93, source_paper_id="paper_scen_b_1",
                source_sentence="EfficientNet-B0 trained with Focal Loss.",
            ),
            RelationRecord(
                relation_id="rel_b_2",
                entity_a_id="eb3", entity_a_text="Random Forest", entity_a_type="MODEL_ARCH",
                entity_b_id="eb4", entity_b_text="ADASYN", entity_b_type="SAMPLING",
                relation_type="HAS_SAMPLING", confidence=0.91, source_paper_id="paper_scen_b_2",
                source_sentence="Random Forest trained after ADASYN sampling.",
            ),
        ]

        return papers, entities, relations

    def _mock_entity(self, text: str, entity_type: str, mech_cat: str, conf: float, paper_id: str, pmid: str) -> NEREntity:
        import uuid
        return NEREntity(
            entity_id=str(uuid.uuid4()),
            text=text,
            entity_type=entity_type,
            mechanism_category=mech_cat,
            start_char=0,
            end_char=len(text),
            source_text=f"We applied {text} in our method.",
            source_paper_id=paper_id,
            source_pmid=pmid,
            confidence=conf,
            confidence_level="HIGH" if conf >= 0.80 else "MEDIUM",
            review_flag=False,
            extraction_method=ExtractionMethod.transformer_ner,
            model_version="allenai/scibert_scivocab_uncased",
            bio_tag=f"B-{entity_type}",
            confidence_status="explicit",
            is_bootstrap=False,
        )
