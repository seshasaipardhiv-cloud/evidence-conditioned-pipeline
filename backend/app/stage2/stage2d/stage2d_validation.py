"""
stage2d_validation.py

Stage 2D — 5-Scenario Controlled Evidence-Switching Validation Harness

Validates across five distinct controlled evidence literature profiles:
  - Scenario A: Corpus strongly favors XGBoost
  - Scenario B: Corpus strongly favors Random Forest
  - Scenario C: Corpus strongly favors Logistic Regression
  - Scenario D: Corpus strongly favors ResNet-18
  - Scenario E: Corpus strongly favors EfficientNet-B0 / ViT

Proves that:
  RESEARCH LITERATURE PROFILE ──▶ SCIENTIFIC EVIDENCE EXTRACTION ──▶ COMPONENT SELECTION
is genuinely and dynamically connected with zero hardcoded model preferences.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from backend.app.stage2.models import ExtractionMethod, NEREntity, PaperRecord, RelationRecord
from backend.app.stage2.pipeline_selector import AutomaticPipelineSelector
from backend.app.stage2.stage2d.section_evidence_scorer import SectionAwareEvidenceScorer

logger = logging.getLogger(__name__)


class Stage2DControlledValidator:
    """
    Executes the 5-scenario controlled evidence-switching experiment for Stage 2D.
    """

    def run_5_scenario_validation(self) -> Dict[str, Any]:
        """
        Runs all 5 scenarios and verifies that each selected candidate dynamically matches the evidence profile.
        """
        logger.info("Executing Stage 2D 5-Scenario Controlled Evidence Validation...")

        scorer = SectionAwareEvidenceScorer()
        selector = AutomaticPipelineSelector()

        scenarios = {
            "Scenario_A": ("tabular_model", "XGBoost", self._build_scenario_a()),
            "Scenario_B": ("tabular_model", "Random Forest", self._build_scenario_b()),
            "Scenario_C": ("tabular_model", "Logistic Regression", self._build_scenario_c()),
            "Scenario_D": ("image_model", "ResNet-18", self._build_scenario_d()),
            "Scenario_E": ("image_model", "EfficientNet-B0", self._build_scenario_e()),
        }

        results = []
        all_passed = True

        for scen_id, (slot, expected_winner, (papers, entities, relations)) in scenarios.items():
            # Score evidence with section awareness
            scored = scorer.score_evidence(entities=entities, relations=relations, papers=papers)

            # Map to EvidenceScoreRecord format expected by selector
            from backend.app.stage2.evidence_scoring import EvidenceScoreRecord
            adapted_scores = {}
            for k, v in scored.items():
                adapted_scores[k] = EvidenceScoreRecord(
                    canonical_name=v.canonical_name,
                    entity_type=v.entity_type,
                    mechanism_category=v.mechanism_category,
                    composite_score=v.composite_score,
                    ner_confidence_score=v.ner_confidence_score,
                    relation_confidence_score=v.relation_confidence_score,
                    full_text_score=v.full_text_score,
                    provenance_score=v.provenance_score,
                    paper_support_score=v.paper_support_score,
                    task_modality_match_score=v.task_modality_match_score,
                    consistency_score=0.90,
                    supporting_paper_count=v.supporting_paper_count,
                    supporting_paper_ids=v.supporting_paper_ids,
                    supporting_pmids=v.supporting_pmids,
                    supporting_dois=v.supporting_dois,
                    total_mention_count=v.total_mention_count,
                    participating_relation_count=v.participating_relation_count,
                    selection_rationale=v.selection_rationale,
                )

            spec = selector.select_pipeline(
                scored_evidence=adapted_scores,
                modalities=["tabular", "image"],
                sample_count=50,
                compute_budget="LIGHT",
            )

            actual_winner = spec.selected_components[slot].selected_name
            winning_score = spec.selected_components[slot].winning_score
            passed = (actual_winner == expected_winner)

            if not passed:
                all_passed = False

            results.append({
                "scenario_id": scen_id,
                "target_slot": slot,
                "expected_winner": expected_winner,
                "actual_winner": actual_winner,
                "winning_score": winning_score,
                "passed": passed,
            })

        report = {
            "experiment_name": "Stage 2D 5-Scenario Controlled Evidence-Switching Validation",
            "all_scenarios_passed": all_passed,
            "scenario_results": results,
        }

        logger.info(f"Stage 2D 5-scenario validation complete. All scenarios switched dynamically: {all_passed}")
        return report

    def _mock_ent(self, text: str, etype: str, conf: float, pmid: str, section: str = "methods") -> NEREntity:
        import uuid
        return NEREntity(
            entity_id=str(uuid.uuid4()),
            text=text,
            entity_type=etype,
            mechanism_category="Representation",
            start_char=0,
            end_char=len(text),
            source_text=f"We applied {text} in our experimental method.",
            source_section=section,
            source_paper_id=f"paper_{pmid}",
            source_pmid=pmid,
            confidence=conf,
            confidence_level="HIGH" if conf >= 0.80 else "MEDIUM",
            review_flag=False,
            extraction_method=ExtractionMethod.transformer_ner,
            model_version="allenai/scibert_scivocab_uncased",
            bio_tag=f"B-{etype}",
            confidence_status="explicit",
            is_bootstrap=False,
        )

    def _build_scenario_a(self):
        papers = [PaperRecord(paper_id="paper_a", title="XGBoost Study", authors=["A"], publication_year=2026, source="PMC", pmid="38396486", full_text_available=True, retrieval_date="2026-08-29")]
        ents = [self._mock_ent("XGBoost", "MODEL_ARCH", 0.98, "38396486", "methods")]
        return papers, ents, []

    def _build_scenario_b(self):
        papers = [PaperRecord(paper_id="paper_b", title="Random Forest Study", authors=["B"], publication_year=2026, source="PMC", pmid="49991111", full_text_available=True, retrieval_date="2026-08-29")]
        ents = [self._mock_ent("Random Forest", "MODEL_ARCH", 0.98, "49991111", "methods")]
        return papers, ents, []

    def _build_scenario_c(self):
        papers = [PaperRecord(paper_id="paper_c", title="Logistic Regression Study", authors=["C"], publication_year=2026, source="PMC", pmid="49992222", full_text_available=True, retrieval_date="2026-08-29")]
        ents = [self._mock_ent("Logistic Regression", "MODEL_ARCH", 0.98, "49992222", "methods")]
        return papers, ents, []

    def _build_scenario_d(self):
        papers = [PaperRecord(paper_id="paper_d", title="ResNet-18 Study", authors=["D"], publication_year=2026, source="PMC", pmid="42487970", full_text_available=True, retrieval_date="2026-08-29")]
        ents = [self._mock_ent("ResNet-18", "MODEL_ARCH", 0.98, "42487970", "methods")]
        return papers, ents, []

    def _build_scenario_e(self):
        papers = [PaperRecord(paper_id="paper_e", title="EfficientNet Study", authors=["E"], publication_year=2026, source="PMC", pmid="41006422", full_text_available=True, retrieval_date="2026-08-29")]
        ents = [self._mock_ent("EfficientNet-B0", "MODEL_ARCH", 0.98, "41006422", "methods")]
        return papers, ents, []
