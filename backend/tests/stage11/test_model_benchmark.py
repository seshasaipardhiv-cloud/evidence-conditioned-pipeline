"""
Unit tests for Stage 11 Model Benchmark Runner
"""

import json
from pathlib import Path
import pytest
from backend.app.stage11.model_benchmark import Stage11ModelBenchmark


@pytest.fixture(scope="module")
def benchmark_results():
    bench = Stage11ModelBenchmark(seeds=[42, 100, 2026])
    res = bench.run_benchmark(selected_models=["candidate_pipeline", "xgboost_default", "random_forest"])
    return res


def test_benchmark_execution(benchmark_results):
    assert benchmark_results["status"] == "COMPLETED"
    assert benchmark_results["models_evaluated_count"] >= 3
    assert benchmark_results["candidate_roc_auc"] > 0.90


def test_master_comparison_json_exists():
    path = Path("evidence/processed/stage11/stage11_model_comparison.json")
    assert path.exists()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "VALIDATED"
    assert "rankings" in data
    assert "best_roc_auc" in data["rankings"]
    assert "best_pr_auc" in data["rankings"]
    assert "best_brier_score" in data["rankings"]


def test_master_comparison_markdown_exists():
    path = Path("evidence/processed/stage11/stage11_final_report.md")
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Authoritative Final Performance Comparison Table" in content
    assert "Scientific Ensemble Interpretation" in content
