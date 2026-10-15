"""
test_evidence_pipeline_integration.py

Comprehensive integration test suite verifying the evidence-conditioned pipeline:
  Test A: Evidence changes the selection
  Test B: Selected model reaches the executor
  Test C: Execution and reported identity agree
  Test D: Evidence integrity (missing, contradictory, and insufficient evidence)
  Test E: Function-contract integrity (coverage_2b, ranker return)
  Test F: Context propagation (real task, modalities, sample count)
  Test G: Fallback behavior (unavailable model, invalid config, explicit fallback marker)
  Test H: Reproducibility (deterministic pipeline IDs)
  Test I: Full end-to-end path (evidence extraction -> scoring -> selection -> model build -> execution)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.stage2.evidence_scoring import EvidenceScoreRecord, EvidenceScoringEngine
from backend.app.stage2.models import (
    ExtractionMethod,
    NEREntity,
    PaperRecord,
    RelationRecord,
    SourceScope,
)
from backend.app.stage2.pipeline_selector import AutomaticPipelineSelector, PipelineSpecification
from backend.app.stage3.models import Component, Mechanism, Stage3Context
from backend.app.stage3.ranker import RankerComposer
from backend.app.stage3.belief_updater import BeliefUpdater
from backend.app.stage5.executor_stage5b import Stage5BExecutor


# ---------------------------------------------------------------------------
# Test A: Evidence changes the selection
# ---------------------------------------------------------------------------

class TestEvidenceChangesSelection:
    """Verify that different evidence sets produce different pipeline selections."""

    def test_high_evidence_xgboost_wins_over_low_evidence_rf(self):
        """When XGBoost has high evidence and Random Forest has low evidence, XGBoost is chosen."""
        xgb_record = EvidenceScoreRecord(
            canonical_name="xgboost",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            composite_score=0.92,
            ner_confidence_score=0.93,
            relation_confidence_score=0.85,
            full_text_score=0.95,
            provenance_score=1.0,
            paper_support_score=0.90,
            task_modality_match_score=0.85,
            consistency_score=0.85,
            supporting_paper_count=4,
            supporting_pmids=["11111111", "22222222"],
            total_mention_count=6,
            participating_relation_count=3,
            selection_rationale="High confidence XGBoost evidence",
        )
        rf_record = EvidenceScoreRecord(
            canonical_name="random forest",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            composite_score=0.55,
            ner_confidence_score=0.55,
            relation_confidence_score=0.50,
            full_text_score=0.60,
            provenance_score=0.50,
            paper_support_score=0.38,
            task_modality_match_score=0.60,
            consistency_score=0.70,
            supporting_paper_count=1,
            supporting_pmids=["33333333"],
            total_mention_count=1,
            participating_relation_count=0,
            selection_rationale="Low evidence RF",
        )

        evidence = {"xgboost": xgb_record, "random forest": rf_record}
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=evidence,
            modalities=["tabular"],
            sample_count=80,
            compute_budget="LIGHT",
        )
        assert spec.selected_components["tabular_model"].canonical_id == "xgboost"

    def test_high_evidence_rf_beats_baseline_xgboost(self):
        """When Random Forest has strong evidence and XGBoost has zero evidence, RF wins."""
        rf_record = EvidenceScoreRecord(
            canonical_name="random forest",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            composite_score=0.91,
            ner_confidence_score=0.92,
            relation_confidence_score=0.88,
            full_text_score=0.95,
            provenance_score=1.0,
            paper_support_score=0.90,
            task_modality_match_score=0.90,
            consistency_score=0.85,
            supporting_paper_count=5,
            supporting_pmids=["55555555"],
            total_mention_count=8,
            participating_relation_count=4,
            selection_rationale="Strong RF evidence across multiple papers",
        )
        # XGBoost has no evidence in this corpus
        evidence = {"random forest": rf_record}
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=evidence,
            modalities=["tabular"],
            sample_count=80,
            compute_budget="LIGHT",
        )
        assert spec.selected_components["tabular_model"].canonical_id == "random_forest"

    def test_high_evidence_logistic_regression_beats_other_models(self):
        """When Logistic Regression has overwhelming literature support, it is selected."""
        lr_record = EvidenceScoreRecord(
            canonical_name="logistic regression",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            composite_score=0.95,
            ner_confidence_score=0.95,
            relation_confidence_score=0.90,
            full_text_score=1.0,
            provenance_score=1.0,
            paper_support_score=0.95,
            task_modality_match_score=0.90,
            consistency_score=0.85,
            supporting_paper_count=6,
            supporting_pmids=["77777777"],
            total_mention_count=10,
            participating_relation_count=5,
            selection_rationale="Extensive empirical support for logistic regression",
        )
        evidence = {"logistic regression": lr_record}
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=evidence,
            modalities=["tabular"],
            sample_count=50,
            compute_budget="LIGHT",
        )
        assert spec.selected_components["tabular_model"].canonical_id == "logistic_regression"


# ---------------------------------------------------------------------------
# Test B: Selected model reaches the executor
# ---------------------------------------------------------------------------

class TestSelectedModelReachesExecutor:
    """Verify that the selected model reaches the actual execution factory."""

    def test_random_forest_reaches_executor(self):
        """When the pipeline spec specifies Random Forest, a RandomForestClassifier is instantiated."""
        from sklearn.ensemble import RandomForestClassifier

        spec = {
            "pipeline_id": "synth_pipeline_rf_test",
            "selected_components": {
                "tabular_model": {
                    "canonical_id": "random_forest",
                    "selected_name": "Random Forest",
                    "evidence_score": 0.89,
                }
            },
        }
        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(spec, seed=42)
        assert isinstance(model, RandomForestClassifier)
        assert "evidence_selected" in identity
        assert "RandomForest" in identity

    def test_logistic_regression_reaches_executor(self):
        """When the pipeline spec specifies Logistic Regression, LogisticRegression is instantiated."""
        from sklearn.linear_model import LogisticRegression

        spec = {
            "pipeline_id": "synth_pipeline_lr_test",
            "selected_components": {
                "tabular_model": {
                    "canonical_id": "logistic_regression",
                    "selected_name": "Logistic Regression",
                    "evidence_score": 0.92,
                }
            },
        }
        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(spec, seed=42)
        assert isinstance(model, LogisticRegression)
        assert "evidence_selected" in identity
        assert "LogisticRegression" in identity

    def test_mlp_tabular_reaches_executor(self):
        """When the pipeline spec specifies Tabular MLP, MLPClassifier is instantiated."""
        from sklearn.neural_network import MLPClassifier

        spec = {
            "pipeline_id": "synth_pipeline_mlp_test",
            "selected_components": {
                "tabular_model": {
                    "canonical_id": "mlp_tabular",
                    "selected_name": "Tabular MLP",
                    "evidence_score": 0.85,
                }
            },
        }
        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(spec, seed=42)
        assert isinstance(model, MLPClassifier)
        assert "evidence_selected" in identity


# ---------------------------------------------------------------------------
# Test C: Execution and reported identity agree
# ---------------------------------------------------------------------------

class TestExecutionIdentityAgreement:
    """Verify that the reported identity matches what was executed."""

    def test_reported_identity_matches_executed_rf(self):
        """Candidate identity includes model name and pipeline ID."""
        spec = {
            "pipeline_id": "synth_rf_verify_123",
            "selected_components": {
                "tabular_model": {
                    "canonical_id": "random_forest",
                    "selected_name": "Random Forest",
                    "evidence_score": 0.88,
                }
            },
        }
        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(spec, seed=42)
        assert "RandomForest" in identity
        assert "synth_rf_verify_123" in identity
        assert "evidence_selected" in identity

    def test_fallback_identity_is_explicitly_flagged(self):
        """Fallback identity explicitly labels fallback and cannot masquerade as evidence-selected."""
        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(None, seed=42)
        assert "FALLBACK" in identity
        assert "evidence_selected" not in identity


# ---------------------------------------------------------------------------
# Test D: Evidence integrity
# ---------------------------------------------------------------------------

class TestEvidenceIntegrity:
    """Verify handling of missing, contradictory, and single-source evidence."""

    def test_empty_evidence_produces_empty_dict(self):
        engine = EvidenceScoringEngine()
        scores = engine.score_corpus_evidence(entities=[], relations=[], papers=[])
        assert scores == {}

    def test_single_paper_gets_unverified_consistency(self):
        """A mechanism from only 1 paper receives consistency score 0.70 (unverified)."""
        engine = EvidenceScoringEngine()
        entity = NEREntity(
            entity_id="e001",
            text="XGBoost",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            start_char=0,
            end_char=7,
            source_text="We used XGBoost.",
            source_section="abstract",
            source_paper_id="p001",
            source_pmid="12345678",
            source_doi=None,
            confidence=0.88,
            confidence_level="HIGH",
            review_flag=False,
            extraction_method=ExtractionMethod.transformer_ner,
            model_version="scibert",
            bio_tag="B-MODEL_ARCH",
            confidence_status="explicit",
            is_bootstrap=False,
        )
        paper = PaperRecord(
            paper_id="p001", title="Study 1", authors=["A"],
            publication_year=2024, source="PubMed", retrieval_date="2024-01-01",
            pmid="12345678", abstract="We used XGBoost.",
            abstract_available=True, full_text_available=False,
        )
        scores = engine.score_corpus_evidence(entities=[entity], relations=[], papers=[paper])
        assert "xgboost" in scores
        assert scores["xgboost"].consistency_score == 0.70

    def test_contradiction_penalizes_score(self):
        """Mechanisms with contradiction candidates receive penalized consistency score 0.50."""
        engine = EvidenceScoringEngine()
        entity = NEREntity(
            entity_id="e_contra",
            text="random forest",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            start_char=0,
            end_char=13,
            source_text="Random forest degraded performance.",
            source_section="abstract",
            source_paper_id="p002",
            source_pmid="99999999",
            source_doi=None,
            confidence=0.80,
            confidence_level="HIGH",
            review_flag=False,
            extraction_method=ExtractionMethod.transformer_ner,
            model_version="scibert",
            bio_tag="B-MODEL_ARCH",
            confidence_status="explicit",
            is_bootstrap=False,
        )
        paper = PaperRecord(
            paper_id="p002", title="Study 2", authors=["B"],
            publication_year=2024, source="PubMed", retrieval_date="2024-01-01",
            pmid="99999999", abstract="Random forest degraded performance.",
            abstract_available=True, full_text_available=False,
        )
        scores = engine.score_corpus_evidence(
            entities=[entity], relations=[], papers=[paper],
            contradiction_mechanism_ids={"random forest"},
        )
        assert "random forest" in scores
        assert scores["random forest"].consistency_score == 0.50


# ---------------------------------------------------------------------------
# Test E: Function-contract integrity
# ---------------------------------------------------------------------------

class TestFunctionContractIntegrity:
    """Verify function return types and contracts."""

    def test_build_coverage_2b_returns_list(self):
        """_build_coverage_2b returns a non-empty list of dicts with expected keys."""
        from backend.app.stage2.orchestrator import Stage2Orchestrator

        orch = Stage2Orchestrator()
        paper = PaperRecord(
            paper_id="p1", title="Paper 1", authors=["A"],
            publication_year=2024, source="PubMed", retrieval_date="2024-01-01",
            pmid="11111111", abstract="Abstract.",
            abstract_available=True, full_text_available=True,
        )
        cov = orch._build_coverage_2b([paper], [], [], [])
        assert isinstance(cov, list)
        assert len(cov) == 1
        assert cov[0]["paper_id"] == "p1"
        assert cov[0]["evidence_coverage"] == "full"
        assert "experiments_extracted" in cov[0]
        assert "ablations_extracted" in cov[0]

    def test_ranker_returns_tuple_of_spec_and_rankings(self):
        """rank_and_compose returns (PipelineSpecification, Dict[Component, MechanismRanking])."""
        ranker = RankerComposer(evidence_threshold=0.1)
        updater = BeliefUpdater()
        beliefs = updater.update_beliefs([])
        context = Stage3Context(task="binary_classification", modalities=["tabular"])

        res = ranker.rank_and_compose(context, beliefs)
        assert isinstance(res, tuple)
        assert len(res) == 2
        spec, rankings = res
        assert hasattr(spec, "selected_mechanisms")
        assert isinstance(rankings, dict)


# ---------------------------------------------------------------------------
# Test F: Context propagation
# ---------------------------------------------------------------------------

class TestContextPropagation:
    """Verify that dataset context is passed accurately without hardcoded substitution."""

    def test_selector_respects_single_modality(self):
        """When only 'tabular' is provided, image and text models are not selected."""
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence={},
            modalities=["tabular"],
            sample_count=50,
            compute_budget="LIGHT",
        )
        assert "tabular_model" in spec.selected_components
        assert "image_model" not in spec.selected_components
        assert "text_model" not in spec.selected_components
        assert "fusion" not in spec.selected_components  # Fusion requires >= 2 modalities

    def test_selector_penalizes_data_hungry_models_on_small_cohorts(self):
        """With small sample count (n=10), models with high min_n (like XGBoost) are penalized."""
        base_record = EvidenceScoreRecord(
            canonical_name="test",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            composite_score=0.80,
            ner_confidence_score=0.80,
            relation_confidence_score=0.80,
            full_text_score=0.80,
            provenance_score=0.80,
            paper_support_score=0.80,
            task_modality_match_score=0.80,
            consistency_score=0.80,
            supporting_paper_count=3,
            supporting_pmids=["11111"],
            total_mention_count=3,
            participating_relation_count=1,
            selection_rationale="test",
        )
        evidence = {
            "xgboost": base_record,
            "logistic regression": base_record,
        }
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=evidence,
            modalities=["tabular"],
            sample_count=10,  # min_n for XGBoost is 20 (penalized), min_n for LR is 10 (not penalized)
            compute_budget="LIGHT",
        )
        assert spec.selected_components["tabular_model"].canonical_id == "logistic_regression"


# ---------------------------------------------------------------------------
# Test G: Fallback behavior
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    """Verify explicit fallback handling."""

    def test_missing_spec_triggers_fallback_identity(self):
        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(None, seed=42)
        assert "FALLBACK" in identity

    def test_unmapped_model_triggers_fallback_with_details(self):
        executor = Stage5BExecutor()
        spec = {
            "pipeline_id": "synth_exotic",
            "selected_components": {
                "tabular_model": {
                    "canonical_id": "quantum_neural_network_v9",
                    "selected_name": "Quantum NN",
                    "evidence_score": 0.99,
                }
            },
        }
        model, identity = executor._build_candidate_from_spec(spec, seed=42)
        assert "FALLBACK" in identity
        assert "quantum_neural_network_v9" in identity


# ---------------------------------------------------------------------------
# Test H: Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Verify deterministic pipeline IDs."""

    def test_pipeline_id_is_deterministic(self):
        selector = AutomaticPipelineSelector()
        spec1 = selector.select_pipeline({}, ["tabular"], 50, True, "binary_classification", "LIGHT")
        spec2 = selector.select_pipeline({}, ["tabular"], 50, True, "binary_classification", "LIGHT")
        assert spec1.pipeline_id == spec2.pipeline_id

    def test_pipeline_id_changes_with_parameters(self):
        selector = AutomaticPipelineSelector()
        spec1 = selector.select_pipeline({}, ["tabular"], 50, True, "binary_classification", "LIGHT")
        spec2 = selector.select_pipeline({}, ["tabular", "image"], 50, True, "binary_classification", "LIGHT")
        assert spec1.pipeline_id != spec2.pipeline_id

    def test_pipeline_id_uses_sha256_hex(self):
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline({}, ["tabular"], 50, True, "binary_classification", "LIGHT")
        pid = spec.pipeline_id
        assert pid.startswith("synth_pipeline_")
        hex_suffix = pid.replace("synth_pipeline_", "")
        assert all(c in "0123456789abcdef" for c in hex_suffix)


# ---------------------------------------------------------------------------
# Test I: Full end-to-end path
# ---------------------------------------------------------------------------

class TestEndToEndPath:
    """Verify evidence -> scoring -> selection -> model construction -> execution -> prediction."""

    def test_complete_evidence_to_execution_chain(self):
        # 1. Literature Entity
        entity = NEREntity(
            entity_id="e_synth_lr",
            text="logistic regression",
            entity_type="MODEL_ARCH",
            mechanism_category="model_architecture",
            start_char=0,
            end_char=19,
            source_text="Logistic regression showed superior calibration.",
            source_section="results",
            source_paper_id="paper_synth_01",
            source_pmid="99887766",
            source_doi="10.1000/synth01",
            confidence=0.94,
            confidence_level="HIGH",
            review_flag=False,
            extraction_method=ExtractionMethod.transformer_ner,
            model_version="scibert_v1",
            bio_tag="B-MODEL_ARCH",
            confidence_status="explicit",
            is_bootstrap=False,
        )
        paper = PaperRecord(
            paper_id="paper_synth_01", title="Calibration Study",
            authors=["Smith et al."], publication_year=2024, source="PMC",
            retrieval_date="2024-01-01", pmid="99887766", doi="10.1000/synth01",
            abstract="Logistic regression showed superior calibration.",
            abstract_available=True, full_text_available=True,
        )

        # 2. Score Evidence
        scoring_engine = EvidenceScoringEngine()
        evidence_scores = scoring_engine.score_corpus_evidence(
            entities=[entity], relations=[], papers=[paper]
        )
        assert "logistic regression" in evidence_scores
        lr_score = evidence_scores["logistic regression"]
        assert lr_score.composite_score > 0.60
        assert "99887766" in lr_score.supporting_pmids

        # 3. Select Pipeline
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=evidence_scores,
            modalities=["tabular"],
            sample_count=40,
            compute_budget="LIGHT",
        )
        assert spec.selected_components["tabular_model"].canonical_id == "logistic_regression"

        # 4. Resolve and build model in Executor
        executor = Stage5BExecutor()
        spec_dict = spec.model_dump()
        model, identity = executor._build_candidate_from_spec(spec_dict, seed=42)
        assert "LogisticRegression" in identity
        assert "evidence_selected" in identity

        # 5. Fit model on synthetic data and predict
        rng = np.random.RandomState(42)
        X_train = rng.randn(40, 4)
        y_train = rng.randint(0, 2, 40)
        X_test = rng.randn(10, 4)

        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        assert len(probs) == 10
        assert all(0.0 <= p <= 1.0 for p in probs)

        # 6. Verify full provenance chain
        provenance = spec.selected_components["tabular_model"].provenance_trace
        assert "99887766" in provenance
