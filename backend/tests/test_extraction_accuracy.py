"""
test_extraction_accuracy.py

Regression test suite validating research evidence extraction accuracy
from published paper excerpts.

Covers:
  1. Multi-model comparisons with 'respectively' patterns.
  2. Paired parentheticals and inline model-score bindings.
  3. Head-to-head 'X vs Y' comparative structures.
  4. Contrastive clauses ('whereas', 'while', 'compared to').
  5. Metric propagation across comparative clauses.
  6. Strict percentage vs decimal scaling for all bounded metrics.
  7. Unbounded metric scale preservation (RMSE, MAE).
  8. False-positive rejection (CI ranges, p-values, scanner specs, patient counts).
  9. Non-ML baseline isolation (radiologists/clinicians).
  10. Strict ambiguity flagging (is_ambiguous=True).
  11. Sentence-level dataset attribution.
  12. Sentence-level validation protocol detection.
  13. Expanded model alias canonicalization (RSF, GBDT, CNN, ANN, SVM).
  14. Benchmark regression assertion (Strict Precision >= 0.85, Recall >= 0.85).
"""

import json
from pathlib import Path
import pytest

from backend.app.stage2.experiment_extractor import ExperimentExtractor
from backend.app.stage2.models import SourceScope, PublishedExperimentResult


@pytest.fixture
def extractor():
    return ExperimentExtractor()


class TestMultiModelExtraction:
    """Verify correct attribution in complex multi-model sentences."""

    def test_respectively_two_models(self, extractor):
        text = "Random Forest and Logistic Regression achieved AUCs of 0.88 and 0.82, respectively, on the clinical test split."
        res = extractor.extract_published_results("paper_01", text)
        assert len(res) == 2
        rf = next(r for r in res if r.canonical_id == "random_forest")
        lr = next(r for r in res if r.canonical_id == "logistic_regression")
        assert rf.reported_value == 0.88
        assert not rf.is_ambiguous
        assert lr.reported_value == 0.82
        assert not lr.is_ambiguous

    def test_respectively_three_models(self, extractor):
        text = "XGBoost, Random Forest, and Logistic Regression achieved AUCs of 0.89, 0.84, and 0.76, respectively."
        res = extractor.extract_published_results("paper_02", text)
        assert len(res) == 3
        xgb = next(r for r in res if r.canonical_id == "xgboost")
        rf = next(r for r in res if r.canonical_id == "random_forest")
        lr = next(r for r in res if r.canonical_id == "logistic_regression")
        assert xgb.reported_value == 0.89
        assert rf.reported_value == 0.84
        assert lr.reported_value == 0.76
        assert all(not r.is_ambiguous for r in res)

    def test_paired_parentheticals(self, extractor):
        text = "XGBoost (AUC: 0.88) outperformed Random Forest (AUC: 0.82) across all folds."
        res = extractor.extract_published_results("paper_03", text)
        assert len(res) == 2
        xgb = next(r for r in res if r.canonical_id == "xgboost")
        rf = next(r for r in res if r.canonical_id == "random_forest")
        assert xgb.reported_value == 0.88
        assert rf.reported_value == 0.82
        assert not xgb.is_ambiguous
        assert not rf.is_ambiguous

    def test_head_to_head_vs(self, extractor):
        text = "In predicting recurrence, XGBoost outperformed Random Forest (0.745 vs 0.680 AUC)."
        res = extractor.extract_published_results("paper_04", text)
        assert len(res) == 2
        xgb = next(r for r in res if r.canonical_id == "xgboost")
        rf = next(r for r in res if r.canonical_id == "random_forest")
        assert xgb.reported_value == 0.745
        assert rf.reported_value == 0.680
        assert not xgb.is_ambiguous

    def test_contrastive_clause_without_comma(self, extractor):
        text = "XGBoost achieved an AUC of 0.85 whereas Random Forest reached an AUC of 0.79."
        res = extractor.extract_published_results("paper_05", text)
        assert len(res) == 2
        xgb = next(r for r in res if r.canonical_id == "xgboost")
        rf = next(r for r in res if r.canonical_id == "random_forest")
        assert xgb.reported_value == 0.85
        assert rf.reported_value == 0.79

    def test_metric_propagation_across_comparison_clauses(self, extractor):
        text = "XGBoost reached an AUC of 0.86 compared to the baseline Logistic Regression of 0.72."
        res = extractor.extract_published_results("paper_06", text)
        assert len(res) == 2
        xgb = next(r for r in res if r.canonical_id == "xgboost")
        lr = next(r for r in res if r.canonical_id == "logistic_regression")
        assert xgb.reported_value == 0.86
        assert lr.reported_value == 0.72
        assert lr.metric_name == "AUC"


class TestMetricNormalizationAndPreservation:
    """Verify bounded metrics are scaled and unbounded metrics are preserved."""

    def test_percentage_scaling_accuracy_and_recall(self, extractor):
        text = "Random Forest achieved 94.5% recall and 91.2% precision on the test cohort."
        res = extractor.extract_published_results("paper_07", text)
        assert len(res) == 2
        recall_res = next(r for r in res if "recall" in r.metric_name.lower())
        prec_res = next(r for r in res if "precision" in r.metric_name.lower())
        assert abs(recall_res.reported_value - 0.945) < 1e-4
        assert abs(prec_res.reported_value - 0.912) < 1e-4

    def test_non_breaking_space_percentage(self, extractor):
        text = "The CNN model achieved a weighted accuracy of 92\xa0%."
        res = extractor.extract_published_results("paper_08", text)
        assert len(res) == 1
        assert res[0].canonical_id == "resnet18"
        assert res[0].reported_value == 0.92

    def test_unbounded_metric_preservation(self, extractor):
        text = "Tabular MLP achieved an RMSE of 1.45 and MAE of 1.12."
        res = extractor.extract_published_results("paper_09", text)
        assert len(res) == 2
        rmse_res = next(r for r in res if r.metric_name.upper() == "RMSE")
        mae_res = next(r for r in res if r.metric_name.upper() == "MAE")
        assert rmse_res.reported_value == 1.45
        assert mae_res.reported_value == 1.12


class TestFalsePositiveRejection:
    """Verify non-metric numerical mentions are never extracted as scores."""

    def test_confidence_interval_not_extracted_as_score(self, extractor):
        text = "Logistic Regression yielded an AUC of 0.78 (95% CI: 0.72-0.84)."
        res = extractor.extract_published_results("paper_10", text)
        assert len(res) == 1
        assert res[0].reported_value == 0.78
        # Ensure 0.72 and 0.84 were not extracted as separate results
        assert not any(abs(r.reported_value - 0.72) < 1e-3 for r in res)
        assert not any(abs(r.reported_value - 0.84) < 1e-3 for r in res)

    def test_scanner_parameters_rejected(self, extractor):
        text = "CT scans were obtained with a tube voltage of 120 kV, tube current of 250 mA, and slice thickness of 3.0 mm across 240 patients."
        res = extractor.extract_published_results("paper_11", text)
        assert len(res) == 0

    def test_p_values_and_patient_counts_rejected(self, extractor):
        text = "The cohort of 999 patients showed significant difference (P < 0.001 and P = 0.024)."
        res = extractor.extract_published_results("paper_12", text)
        assert len(res) == 0

    def test_radiologist_score_not_assigned_to_ml_model(self, extractor):
        text = "The deep learning fusion model achieved an AUC of 0.814, outperforming junior radiologists (AUC: 0.782)."
        res = extractor.extract_published_results("paper_13", text)
        assert len(res) == 1
        assert res[0].canonical_id == "mlp_tabular"
        assert res[0].reported_value == 0.814


class TestContextAttributionAndAmbiguity:
    """Verify dataset, protocol, and ambiguity flags."""

    def test_sentence_level_dataset_attribution(self, extractor):
        text = "XGBoost achieved an AUC of 0.694 on the TCGA dataset."
        res = extractor.extract_published_results("paper_14", text)
        assert len(res) == 1
        assert res[0].dataset_name == "TCGA"

    def test_protocol_detection(self, extractor):
        text = "EfficientNet-B0 achieved an AUC of 0.852 in 10-fold cross-validation."
        res = extractor.extract_published_results("paper_15", text)
        assert len(res) == 1
        assert res[0].validation_protocol == "10_fold_cv"

    def test_genuine_ambiguity_flagged(self, extractor):
        text = "Both XGBoost and Random Forest demonstrated competitive diagnostic utility (AUC: 0.85)."
        res = extractor.extract_published_results("paper_16", text)
        assert len(res) == 2
        assert all(r.is_ambiguous for r in res)
        assert all(r.extraction_confidence <= 0.50 for r in res)

    def test_model_alias_canonicalization(self, extractor):
        text_rsf = "The RSF model demonstrated superior predictive performance, yielding a C-index of 0.91."
        res_rsf = extractor.extract_published_results("paper_17", text_rsf)
        assert len(res_rsf) == 1
        assert res_rsf[0].canonical_id == "random_forest"

        text_gbdt = "Among baselines, GBDT obtained an AUC of 0.924, outperforming Support Vector Machine (AUC: 0.881)."
        res_gbdt = extractor.extract_published_results("paper_18", text_gbdt)
        assert len(res_gbdt) == 2
        xgb = next(r for r in res_gbdt if r.canonical_id == "xgboost")
        svm = next(r for r in res_gbdt if r.canonical_id == "svm")
        assert xgb.reported_value == 0.924
        assert svm.reported_value == 0.881


class TestBenchmarkAccuracyRegression:
    """Run full 30-statement real-world evaluation benchmark."""

    def test_benchmark_precision_and_recall_thresholds(self, extractor):
        benchmark_path = Path("evidence/metadata/extraction_evaluation_benchmark.json")
        assert benchmark_path.exists(), "Benchmark dataset must exist"
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark = json.load(f)

        total_expected = 0
        total_extracted = 0
        strict_tp = 0

        for item in benchmark:
            p_id = item["paper_id"]
            text = item["source_text"]
            expected = item.get("expected_extractions", [])
            total_expected += len(expected)

            extracted = extractor.extract_published_results(
                paper_id=p_id,
                text=text,
                citation=f"Benchmark {p_id}",
            )
            total_extracted += len(extracted)

            matched_expected = set()
            for ext in extracted:
                for idx, exp in enumerate(expected):
                    if idx in matched_expected:
                        continue
                    m_ok = (ext.canonical_id == exp["canonical_id"])
                    v_ok = (abs(ext.reported_value - exp["reported_value"]) <= 0.005)
                    amb_ok = (ext.is_ambiguous == exp.get("is_ambiguous", False))
                    ds_ok = True
                    if exp.get("dataset_name"):
                        ds_ok = (ext.dataset_name is not None and exp["dataset_name"].lower() in ext.dataset_name.lower())
                    proto_ok = True
                    if exp.get("validation_protocol"):
                        proto_ok = (ext.validation_protocol == exp["validation_protocol"])

                    if m_ok and v_ok and amb_ok and ds_ok and proto_ok:
                        matched_expected.add(idx)
                        strict_tp += 1
                        break

        precision = strict_tp / total_extracted if total_extracted > 0 else 0.0
        recall = strict_tp / total_expected if total_expected > 0 else 0.0

        assert precision >= 0.85, f"Expected Strict Precision >= 0.85, got {precision:.4f}"
        assert recall >= 0.85, f"Expected Strict Recall >= 0.85, got {recall:.4f}"
