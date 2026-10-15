"""
test_result_aware_selection.py

Comprehensive tests verifying the scientific result-aware evidence comparison
and pipeline selection engine:
  - Test A: Metric direction (higher is better for ROC-AUC/F1; lower is better for RMSE/loss)
  - Test B: Reported performance vs. mentions (superior empirical results beat high-mention models)
  - Test C: Dataset mismatch (penalizes unrelated domains like ImageNet vs. target domain)
  - Test D: Metric incompatibility (incompatible metrics isolated, flagged in notes, never aggregated)
  - Test E: Missing and ambiguous values (gracefully dropped without fabrication)
  - Test F: Conflicting studies (preserved, uncertainty flagged as HIGH)
  - Test G: Repeated evidence (same-paper runs de-duplicated into single study comparison)
  - Test H: Selection-to-execution integration (different evidence triggers different executor models)
  - Test I: Existing contract behavior (backwards compatibility when no experimental results exist)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

from backend.app.stage2.evidence_scoring import EvidenceScoreRecord
from backend.app.stage2.experiment_extractor import ExperimentExtractor
from backend.app.stage2.models import PublishedExperimentResult
from backend.app.stage2.pipeline_selector import AutomaticPipelineSelector, SelectedComponentRecord
from backend.app.stage2.result_evidence import (
    METRIC_DIRECTION_MAP,
    ResultAwareEvidenceComparator,
    are_metrics_compatible,
    canonicalize_metric,
    canonicalize_model_id,
    load_published_experiments,
    normalize_metric_name,
    normalize_protocol,
    save_published_experiments,
)
from backend.app.stage5.executor_stage5b import Stage5BExecutor


# ---------------------------------------------------------------------------
# Test Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_result(
    result_id: str,
    paper_id: str,
    model_name: str,
    canonical_id: str,
    metric_name: str,
    reported_value: float | None,
    higher_is_better: bool = True,
    dataset_name: str = "TCGA_Clinical",
    task_type: str = "binary_classification",
    validation_protocol: str | None = None,
    is_ambiguous: bool = False,
    pmid: str = "12345678",
) -> PublishedExperimentResult:
    return PublishedExperimentResult(
        result_id=result_id,
        paper_id=paper_id,
        pmid=pmid,
        doi=f"10.1000/{paper_id}",
        citation=f"Author et al., 2024 ({paper_id})",
        model_name=model_name,
        canonical_id=canonical_id,
        dataset_name=dataset_name,
        task_type=task_type,
        metric_name=metric_name,
        reported_value=reported_value,
        higher_is_better=higher_is_better,
        validation_protocol=validation_protocol,
        is_ambiguous=is_ambiguous,
    )


def _make_score_record(canonical_name: str, composite_score: float, mentions: int) -> EvidenceScoreRecord:
    return EvidenceScoreRecord(
        canonical_name=canonical_name,
        entity_type="MODEL_ARCH",
        mechanism_category="model_architecture",
        composite_score=composite_score,
        ner_confidence_score=composite_score,
        relation_confidence_score=0.85,
        full_text_score=0.90,
        provenance_score=1.0,
        paper_support_score=0.85,
        task_modality_match_score=0.85,
        consistency_score=0.85,
        supporting_paper_count=max(1, mentions // 2),
        supporting_pmids=["PMID:11111111"],
        total_mention_count=mentions,
        participating_relation_count=mentions // 2,
        selection_rationale=f"Score record for {canonical_name}",
    )


# ---------------------------------------------------------------------------
# Test A: Metric Direction
# ---------------------------------------------------------------------------

class TestMetricDirection:
    """Verify that metric directionality is strictly respected in head-to-head evaluations."""

    def test_higher_is_better_for_auc_f1_accuracy(self):
        """Higher values for AUC, F1, and accuracy must win head-to-head comparisons."""
        comparator = ResultAwareEvidenceComparator()
        candidate_ids = {"xgboost", "random_forest"}

        # Paper 1: AUC (higher is better) -> XGBoost (0.86) beats Random Forest (0.78)
        results = [
            _make_result("r1", "p1", "XGBoost", "xgboost", "ROC_AUC", 0.86),
            _make_result("r2", "p1", "Random Forest", "random_forest", "ROC_AUC", 0.78),
        ]
        comps = comparator.extract_head_to_head_comparisons(results, candidate_ids)
        assert len(comps) == 1
        comp = comps[0]
        assert comp.higher_is_better is True
        assert comp.winner_id == "xgboost"
        assert comp.delta == pytest.approx(0.08, abs=1e-4)

    def test_lower_is_better_for_rmse_mae_loss(self):
        """Lower values for RMSE, MAE, and loss must win head-to-head comparisons."""
        comparator = ResultAwareEvidenceComparator()
        candidate_ids = {"xgboost", "logistic_regression"}

        # Paper 2: RMSE (lower is better) -> Logistic Regression (0.62) beats XGBoost (1.15)
        results = [
            _make_result("r1", "p2", "XGBoost", "xgboost", "RMSE", 1.15, higher_is_better=False),
            _make_result("r2", "p2", "Logistic Regression", "logistic_regression", "RMSE", 0.62, higher_is_better=False),
        ]
        comps = comparator.extract_head_to_head_comparisons(results, candidate_ids)
        assert len(comps) == 1
        comp = comps[0]
        assert comp.higher_is_better is False
        assert comp.winner_id == "logistic_regression"
        assert comp.delta == pytest.approx(0.53, abs=1e-4)


# ---------------------------------------------------------------------------
# Test B: Reported Performance vs Mentions
# ---------------------------------------------------------------------------

class TestPerformanceVsMentions:
    """Verify that a model with superior reported empirical performance beats a high-mention model."""

    def test_superior_reported_results_beat_high_mention_model(self):
        """
        Random Forest: 20 mentions, high NER confidence, composite_score = 0.92.
        Logistic Regression: 2 mentions, low NER confidence, composite_score = 0.55.
        BUT: Direct head-to-head published study shows Logistic Regression (AUC 0.88)
        decisively beats Random Forest (AUC 0.68) on the target problem.
        The selector must choose Logistic Regression based on experimental evidence.
        """
        scored_evidence = {
            "random forest": _make_score_record("random forest", 0.92, 20),
            "logistic regression": _make_score_record("logistic regression", 0.55, 2),
        }
        experimental_results = [
            _make_result("r1", "paper_h2h", "Logistic Regression", "logistic_regression", "ROC_AUC", 0.88),
            _make_result("r2", "paper_h2h", "Random Forest", "random_forest", "ROC_AUC", 0.68),
        ]

        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=scored_evidence,
            modalities=["tabular"],
            sample_count=60,
            compute_budget="LIGHT",
            experimental_results=experimental_results,
            target_dataset="TCGA_Clinical",
        )

        selected_tab = spec.selected_components["tabular_model"]
        assert selected_tab.canonical_id == "logistic_regression", (
            f"Expected Logistic Regression to win based on experimental evidence, but got {selected_tab.canonical_id}"
        )
        assert "experimental performance" in selected_tab.decision_rationale.lower()
        assert selected_tab.reported_performance_score is not None
        assert selected_tab.reported_performance_score > 0.70


# ---------------------------------------------------------------------------
# Test C: Dataset Mismatch
# ---------------------------------------------------------------------------

class TestDatasetMismatch:
    """Verify that models evaluated on unrelated domains (e.g. ImageNet) receive severe applicability penalties."""

    def test_unrelated_dataset_is_penalized(self):
        """
        Model A (XGBoost) evaluated on 'ImageNet' with AUC 0.95.
        Model B (Random Forest) evaluated on 'TCGA_Clinical' with AUC 0.82.
        Target dataset is 'TCGA_Clinical'.
        XGBoost's applicability must be penalized, allowing Random Forest to win.
        """
        scored_evidence = {
            "xgboost": _make_score_record("xgboost", 0.70, 5),
            "random forest": _make_score_record("random forest", 0.70, 5),
        }
        experimental_results = [
            _make_result("r1", "p_img", "XGBoost", "xgboost", "ROC_AUC", 0.95, dataset_name="ImageNet"),
            _make_result("r2", "p_cli", "Random Forest", "random_forest", "ROC_AUC", 0.82, dataset_name="TCGA_Clinical"),
        ]

        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=scored_evidence,
            modalities=["tabular"],
            sample_count=60,
            compute_budget="LIGHT",
            experimental_results=experimental_results,
            target_dataset="TCGA_Clinical",
        )

        selected_tab = spec.selected_components["tabular_model"]
        assert selected_tab.canonical_id == "random_forest"
        # Verify XGBoost applicability was penalized
        xgb_rank = [c for c in selected_tab.candidate_rankings if c["canonical_id"] == "xgboost"][0]
        rf_rank = [c for c in selected_tab.candidate_rankings if c["canonical_id"] == "random_forest"][0]
        assert xgb_rank["applicability_score"] < rf_rank["applicability_score"]


# ---------------------------------------------------------------------------
# Test D: Metric Incompatibility
# ---------------------------------------------------------------------------

class TestMetricIncompatibility:
    """Verify that incompatible metrics are isolated, flagged, and never compared head-to-head."""

    def test_incompatible_metrics_not_compared_head_to_head(self):
        """ROC-AUC and RMSE measure different dimensions and cannot be compared directly."""
        assert are_metrics_compatible("roc_auc", "auc") is True
        assert are_metrics_compatible("roc_auc", "rmse") is False
        assert are_metrics_compatible("f1", "mae") is False

        comparator = ResultAwareEvidenceComparator()
        candidate_ids = {"xgboost", "random_forest"}

        # One model evaluated on ROC_AUC, the other on RMSE in the same paper
        results = [
            _make_result("r1", "p1", "XGBoost", "xgboost", "ROC_AUC", 0.85),
            _make_result("r2", "p1", "Random Forest", "random_forest", "RMSE", 0.45, higher_is_better=False),
        ]
        # Incompatible metric families must produce ZERO head-to-head comparisons
        comps = comparator.extract_head_to_head_comparisons(results, candidate_ids)
        assert len(comps) == 0

    def test_incompatibility_flagged_in_notes(self):
        """Candidates with incompatible metrics record notes in candidate evaluation."""
        comparator = ResultAwareEvidenceComparator()
        candidates = [
            {"id": "xgboost", "name": "XGBoost", "aliases": ["xgboost"]},
            {"id": "random_forest", "name": "Random Forest", "aliases": ["random forest"]},
        ]
        results = [
            _make_result("r1", "p1", "XGBoost", "xgboost", "ROC_AUC", 0.85),
            _make_result("r2", "p1", "Random Forest", "random_forest", "RMSE", 0.45, higher_is_better=False),
        ]
        evals = comparator.evaluate_candidates(candidates, results)
        assert len(evals["xgboost"].incomparable_metrics_notes) > 0
        assert "incompatible" in evals["xgboost"].incomparable_metrics_notes[0].lower()


# ---------------------------------------------------------------------------
# Test E: Missing and Ambiguous Values
# ---------------------------------------------------------------------------

class TestMissingAndAmbiguousValues:
    """Verify that missing values and ambiguous records are rejected without fabricating scores."""

    def test_missing_and_ambiguous_results_filtered_out(self):
        comparator = ResultAwareEvidenceComparator()
        results = [
            _make_result("r1", "p1", "XGBoost", "xgboost", "ROC_AUC", None), # Missing value
            _make_result("r2", "p2", "Random Forest", "random_forest", "ROC_AUC", 0.85, is_ambiguous=True), # Ambiguous
            _make_result("r3", "p3", "Logistic Regression", "logistic_regression", "ROC_AUC", 0.80), # Valid
        ]
        filtered = comparator.filter_relevant_results(results)
        assert len(filtered) == 1
        assert filtered[0].result_id == "r3"
        assert filtered[0].canonical_id == "logistic_regression"


# ---------------------------------------------------------------------------
# Test F: Conflicting Studies
# ---------------------------------------------------------------------------

class TestConflictingStudies:
    """Verify that contradictory findings are preserved and flagged with HIGH uncertainty."""

    def test_conflicting_studies_produce_high_uncertainty(self):
        """
        Study 1: XGBoost (0.85) beats Random Forest (0.75).
        Study 2: Random Forest (0.88) beats XGBoost (0.78).
        Both comparisons must be preserved, and uncertainty set to HIGH.
        """
        comparator = ResultAwareEvidenceComparator()
        candidates = [
            {"id": "xgboost", "name": "XGBoost", "aliases": ["xgboost"]},
            {"id": "random_forest", "name": "Random Forest", "aliases": ["random forest"]},
        ]
        results = [
            _make_result("r1", "study_1", "XGBoost", "xgboost", "ROC_AUC", 0.85),
            _make_result("r2", "study_1", "Random Forest", "random_forest", "ROC_AUC", 0.75),
            _make_result("r3", "study_2", "XGBoost", "xgboost", "ROC_AUC", 0.78),
            _make_result("r4", "study_2", "Random Forest", "random_forest", "ROC_AUC", 0.88),
        ]

        evals = comparator.evaluate_candidates(candidates, results)
        assert evals["xgboost"].uncertainty_level == "HIGH"
        assert evals["random_forest"].uncertainty_level == "HIGH"
        assert len(evals["xgboost"].supporting_comparisons) == 1
        assert len(evals["xgboost"].conflicting_comparisons) == 1


# ---------------------------------------------------------------------------
# Test G: Repeated Evidence
# ---------------------------------------------------------------------------

class TestRepeatedEvidence:
    """Verify that multiple runs or CV folds from the same paper are de-duplicated into 1 study."""

    def test_same_paper_multiple_runs_deduplicated(self):
        """
        Paper 1 reports 5 cross-validation fold results for XGBoost and Random Forest.
        This must produce exactly 1 study comparison, not 5 or 25 artificial comparisons.
        """
        comparator = ResultAwareEvidenceComparator()
        candidate_ids = {"xgboost", "random_forest"}

        results = [
            _make_result(f"r_xgb_{fold}", "paper_cv", "XGBoost", "xgboost", "ROC_AUC", 0.80 + 0.01 * fold)
            for fold in range(5)
        ] + [
            _make_result(f"r_rf_{fold}", "paper_cv", "Random Forest", "random_forest", "ROC_AUC", 0.75 + 0.01 * fold)
            for fold in range(5)
        ]

        comps = comparator.extract_head_to_head_comparisons(results, candidate_ids)
        assert len(comps) == 1, f"Expected 1 de-duplicated comparison, got {len(comps)}"
        comp = comps[0]
        # Mean XGBoost: 0.82; Mean RF: 0.77 -> delta = 0.05
        assert comp.winner_id == "xgboost"
        assert comp.delta == pytest.approx(0.05, abs=1e-3)


# ---------------------------------------------------------------------------
# Test H: Selection-to-Execution Integration
# ---------------------------------------------------------------------------

class TestSelectionToExecutionIntegration:
    """Verify that different published evidence sets control what model executes in Stage 5B."""

    def test_evidence_favors_rf_executes_rf(self, tmp_path):
        """Evidence set favoring Random Forest results in Stage5BExecutor running RandomForestClassifier."""
        from sklearn.ensemble import RandomForestClassifier

        scored_evidence = {
            "random forest": _make_score_record("random forest", 0.80, 5),
            "xgboost": _make_score_record("xgboost", 0.80, 5),
        }
        # Published paper: RF (0.91) beats XGBoost (0.70)
        exp_results = [
            _make_result("r1", "p_rf_win", "Random Forest", "random_forest", "ROC_AUC", 0.91),
            _make_result("r2", "p_rf_win", "XGBoost", "xgboost", "ROC_AUC", 0.70),
        ]

        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=scored_evidence,
            modalities=["tabular"],
            sample_count=60,
            compute_budget="LIGHT",
            experimental_results=exp_results,
        )

        assert spec.selected_components["tabular_model"].canonical_id == "random_forest"

        # Pass spec to Stage5BExecutor
        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(spec.model_dump(), seed=42)

        assert isinstance(model, RandomForestClassifier)
        assert "RandomForest[evidence_selected]" in identity
        assert "synth_pipeline_" in identity
        assert "FALLBACK" not in identity

    def test_evidence_favors_lr_executes_lr(self, tmp_path):
        """Evidence set favoring Logistic Regression results in Stage5BExecutor running LogisticRegression."""
        from sklearn.linear_model import LogisticRegression

        scored_evidence = {
            "logistic regression": _make_score_record("logistic regression", 0.80, 5),
            "xgboost": _make_score_record("xgboost", 0.80, 5),
        }
        # Published paper: LR (0.89) beats XGBoost (0.69)
        exp_results = [
            _make_result("r1", "p_lr_win", "Logistic Regression", "logistic_regression", "ROC_AUC", 0.89),
            _make_result("r2", "p_lr_win", "XGBoost", "xgboost", "ROC_AUC", 0.69),
        ]

        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=scored_evidence,
            modalities=["tabular"],
            sample_count=60,
            compute_budget="LIGHT",
            experimental_results=exp_results,
        )

        assert spec.selected_components["tabular_model"].canonical_id == "logistic_regression"

        executor = Stage5BExecutor()
        model, identity = executor._build_candidate_from_spec(spec.model_dump(), seed=42)

        assert isinstance(model, LogisticRegression)
        assert "LogisticRegression[evidence_selected]" in identity
        assert "synth_pipeline_" in identity
        assert "FALLBACK" not in identity


# ---------------------------------------------------------------------------
# Test I: Existing Contract Behavior & Utilities
# ---------------------------------------------------------------------------

class TestExistingContractBehavior:
    """Verify backwards compatibility and persistence utilities."""

    def test_fallback_when_experimental_results_none(self):
        """When experimental_results is None, selector behaves identically to baseline."""
        scored_evidence = {
            "xgboost": _make_score_record("xgboost", 0.88, 10),
            "random forest": _make_score_record("random forest", 0.65, 3),
        }
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=scored_evidence,
            modalities=["tabular"],
            sample_count=50,
            compute_budget="LIGHT",
            experimental_results=None,
        )
        assert spec.selected_components["tabular_model"].canonical_id == "xgboost"
        assert spec.pipeline_id.startswith("synth_pipeline_")

    def test_load_and_save_published_experiments_roundtrip(self, tmp_path):
        """Verify save_published_experiments and load_published_experiments roundtrip cleanly."""
        results = [
            _make_result("r1", "p1", "XGBoost", "xgboost", "ROC_AUC", 0.86),
            _make_result("r2", "p1", "Random Forest", "random_forest", "ROC_AUC", 0.78),
        ]
        out_file = tmp_path / "test_experiments.jsonl"
        save_published_experiments(results, out_file)
        assert out_file.exists()

        loaded = load_published_experiments(out_file)
        assert len(loaded) == 2
        assert loaded[0].result_id == "r1"
        assert loaded[0].reported_value == 0.86
        assert loaded[1].result_id == "r2"
        assert loaded[1].reported_value == 0.78


# ---------------------------------------------------------------------------
# Test J: Metric Comparability Strictness
# ---------------------------------------------------------------------------

class TestMetricComparabilityStrictness:
    """Verify strict canonical metric matching rules."""

    def test_accuracy_not_compatible_with_f1(self):
        """Accuracy and F1 measure fundamentally different dimensions and must be incompatible."""
        assert are_metrics_compatible("accuracy", "f1") is False
        assert are_metrics_compatible("acc", "f1-score") is False

    def test_roc_auc_not_compatible_with_pr_auc(self):
        """ROC-AUC and PR-AUC are not equivalent under severe imbalance and must be incompatible."""
        assert are_metrics_compatible("roc_auc", "pr_auc") is False
        assert are_metrics_compatible("auc", "average_precision") is False

    def test_rmse_not_compatible_with_mae(self):
        """RMSE (L2 norm) and MAE (L1 norm) cannot be compared numerically."""
        assert are_metrics_compatible("rmse", "mae") is False
        assert are_metrics_compatible("root_mean_squared_error", "mean_absolute_error") is False

    def test_rmse_not_compatible_with_mse(self):
        """RMSE (square root scale) and MSE (squared error scale) are incompatible."""
        assert are_metrics_compatible("rmse", "mse") is False

    def test_roc_auc_compatible_with_auc_and_auroc(self):
        """ROC-AUC, AUC, and AUROC are verified synonymous representations."""
        assert are_metrics_compatible("roc_auc", "auc") is True
        assert are_metrics_compatible("roc-auc", "auroc") is True


# ---------------------------------------------------------------------------
# Test K: Evaluation Protocol Isolation
# ---------------------------------------------------------------------------

class TestProtocolIsolation:
    """Verify that different evaluation protocols are never mixed in head-to-head comparisons."""

    def test_cv_and_external_validation_isolated_in_same_paper(self):
        """
        Paper reports 5-fold cross-validation for XGBoost (0.88)
        and external holdout validation for Random Forest (0.80).
        Because the protocols differ, they MUST NOT produce a direct head-to-head comparison.
        """
        comparator = ResultAwareEvidenceComparator()
        candidate_ids = {"xgboost", "random_forest"}

        results = [
            _make_result("r1", "p_proto", "XGBoost", "xgboost", "ROC_AUC", 0.88, validation_protocol="5_fold_cv"),
            _make_result("r2", "p_proto", "Random Forest", "random_forest", "ROC_AUC", 0.80, validation_protocol="external_validation"),
        ]
        comps = comparator.extract_head_to_head_comparisons(results, candidate_ids)
        assert len(comps) == 0, "CV and external validation must not be compared head-to-head"


# ---------------------------------------------------------------------------
# Test L: Score Range, Unverified Prior, and Decision Mode
# ---------------------------------------------------------------------------

class TestScoreRangeAndUnverifiedPrior:
    """Verify calibrated score range [0.0, 1.0], unverified priors, and transparent decision modes."""

    def test_score_strictly_bounded_zero_to_one(self):
        """Verify that integrated score is mathematically bounded in [0.0, 1.0] even with maximum inputs."""
        scored_evidence = {
            "xgboost": _make_score_record("xgboost", 1.0, 50),
            "random forest": _make_score_record("random forest", 1.0, 50),
        }
        experimental_results = [
            _make_result("r1", "p_max", "XGBoost", "xgboost", "ROC_AUC", 1.0),
            _make_result("r2", "p_max", "Random Forest", "random_forest", "ROC_AUC", 0.99),
        ]
        selector = AutomaticPipelineSelector()
        spec = selector.select_pipeline(
            scored_evidence=scored_evidence,
            modalities=["tabular"],
            sample_count=200,
            compute_budget="HEAVY",
            experimental_results=experimental_results,
            target_dataset="TCGA_Clinical",
        )
        for comp_name, comp in spec.selected_components.items():
            for rank in comp.candidate_rankings:
                assert 0.0 <= rank["final_score"] <= 1.0, f"{rank['name']} score {rank['final_score']} out of bounds"

    def test_unverified_prior_vs_demonstrated_loser(self):
        """
        Candidate A: Unverified model (no experimental results, neutral prior 0.40, UNVERIFIED).
        Candidate B: Demonstrated loser (lost 3/3 head-to-head comparisons, S_perf ~ 0.15).
        Candidate C: Demonstrated winner (won 3/3 head-to-head comparisons, S_perf ~ 0.95).

        Verified winner (C) > Unverified prior (A) > Demonstrated loser (B).
        """
        comparator = ResultAwareEvidenceComparator()
        candidates = [
            {"id": "xgboost", "name": "XGBoost", "aliases": ["xgboost"]},
            {"id": "random_forest", "name": "Random Forest", "aliases": ["random forest"]},
            {"id": "logistic_regression", "name": "Logistic Regression", "aliases": ["logistic regression"]},
        ]
        # XGBoost beats Random Forest in 3 studies. Logistic Regression has no results.
        results = [
            _make_result("r1", "s1", "XGBoost", "xgboost", "ROC_AUC", 0.88),
            _make_result("r2", "s1", "Random Forest", "random_forest", "ROC_AUC", 0.75),
            _make_result("r3", "s2", "XGBoost", "xgboost", "ROC_AUC", 0.85),
            _make_result("r4", "s2", "Random Forest", "random_forest", "ROC_AUC", 0.72),
            _make_result("r5", "s3", "XGBoost", "xgboost", "ROC_AUC", 0.90),
            _make_result("r6", "s3", "Random Forest", "random_forest", "ROC_AUC", 0.78),
        ]
        evals = comparator.evaluate_candidates(candidates, results)

        # Candidate A (Logistic Regression): Unverified
        assert evals["logistic_regression"].reported_performance_score is None
        assert evals["logistic_regression"].uncertainty_level == "UNVERIFIED"

        # Candidate B (Random Forest): Demonstrated loser (0 wins, 3 losses)
        assert evals["random_forest"].head_to_head_losses == 3
        assert evals["random_forest"].reported_performance_score < 0.25

        # Candidate C (XGBoost): Demonstrated winner (3 wins, 0 losses)
        assert evals["xgboost"].head_to_head_wins == 3
        assert evals["xgboost"].reported_performance_score > 0.80

    def test_decision_mode_transparency(self):
        """Verify explicit reporting of RESULT_AWARE_EXPERIMENTAL vs HEURISTIC_LITERATURE_FALLBACK."""
        selector = AutomaticPipelineSelector()

        # Case 1: With experimental results -> RESULT_AWARE_EXPERIMENTAL
        spec_exp = selector.select_pipeline(
            scored_evidence={"xgboost": _make_score_record("xgboost", 0.85, 10)},
            modalities=["tabular"],
            experimental_results=[_make_result("r1", "p1", "XGBoost", "xgboost", "ROC_AUC", 0.88)],
        )
        assert spec_exp.overall_decision_mode == "RESULT_AWARE_EXPERIMENTAL"
        assert spec_exp.selected_components["tabular_model"].decision_mode == "RESULT_AWARE_EXPERIMENTAL"

        # Case 2: Without experimental results -> HEURISTIC_LITERATURE_FALLBACK
        spec_fallback = selector.select_pipeline(
            scored_evidence={"xgboost": _make_score_record("xgboost", 0.85, 10)},
            modalities=["tabular"],
            experimental_results=None,
        )
        assert spec_fallback.overall_decision_mode == "HEURISTIC_LITERATURE_FALLBACK"
        assert spec_fallback.selected_components["tabular_model"].decision_mode == "HEURISTIC_LITERATURE_FALLBACK"


# ---------------------------------------------------------------------------
# Test M: Live Research Paper Extraction
# ---------------------------------------------------------------------------

class TestLivePaperExtraction:
    """Verify live extraction of PublishedExperimentResult records from empirical text without hallucination."""

    def test_extract_published_results_from_text(self):
        """Tests live extraction from sentence with model names and metric values."""
        extractor = ExperimentExtractor()
        text = (
            "In our 5-fold cross-validation on the TCGA cohort, "
            "XGBoost achieved an AUC of 0.88, while Random Forest achieved an AUC of 0.82."
        )
        results = extractor.extract_published_results(
            paper_id="paper_test_01",
            text=text,
            pmid="12345678",
            doi="10.1000/test01",
            dataset="TCGA",
            task="cancer_classification",
        )
        assert len(results) == 2
        models = {r.canonical_id: r for r in results}
        assert "xgboost" in models
        assert "random_forest" in models
        assert models["xgboost"].reported_value == 0.88
        assert models["xgboost"].validation_protocol == "5_fold_cv"
        assert models["xgboost"].is_ambiguous is False
        assert models["random_forest"].reported_value == 0.82
        assert models["random_forest"].validation_protocol == "5_fold_cv"
        assert models["random_forest"].is_ambiguous is False

    def test_extract_published_results_ambiguity_flag(self):
        """When multiple models are mentioned with a single metric value, is_ambiguous must be True."""
        extractor = ExperimentExtractor()
        text = "Both XGBoost and Random Forest were evaluated for mortality prediction, achieving an AUC of 0.85."
        results = extractor.extract_published_results(
            paper_id="paper_test_02",
            text=text,
            dataset="MIMIC",
        )
        assert len(results) == 2
        for r in results:
            assert r.is_ambiguous is True
            assert r.reported_value == 0.85

