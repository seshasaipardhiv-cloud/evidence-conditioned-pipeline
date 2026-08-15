from backend.app.stage3.similarity import (
    calculate_modality_jaccard,
    calculate_sample_size_similarity,
    calculate_missingness_similarity,
    calculate_task_similarity,
    calculate_context_similarity
)
from backend.app.stage3.models import Stage3Context

def test_modality_similarity():
    assert calculate_modality_jaccard(["clinical", "text"], ["clinical", "text"]) == 1.0
    assert calculate_modality_jaccard(["clinical", "text"], ["clinical"]) == 0.5
    assert calculate_modality_jaccard(["clinical"], ["imaging"]) == 0.0
    assert calculate_modality_jaccard([], []) == 1.0

def test_task_similarity():
    assert calculate_task_similarity("survival_prediction", "survival_prediction") == 1.0
    assert calculate_task_similarity("survival_prediction", "diagnosis") == 0.0
    assert calculate_task_similarity("unknown", "survival_prediction") == 0.5
    assert calculate_task_similarity("survival_prediction", "unknown") == 0.5

def test_missingness_similarity():
    assert abs(calculate_missingness_similarity(0.1, 0.1) - 1.0) < 1e-5
    assert abs(calculate_missingness_similarity(0.1, 0.9) - 0.2) < 1e-5
    assert abs(calculate_missingness_similarity(None, 0.1) - 0.5) < 1e-5

def test_sample_size_similarity():
    assert calculate_sample_size_similarity(100, 100) == 1.0
    assert calculate_sample_size_similarity(10, 1000) == 0.0 # diff is 2 log steps

def test_context_similarity_computation():
    ctx = Stage3Context(task="survival_prediction", modalities=["clinical"], sample_size=100)
    exp_perfect = {"task": "survival_prediction", "modalities": ["clinical"], "sample_count": 100}
    
    sim = calculate_context_similarity(ctx, exp_perfect)
    # weights: task 0.4, mod 0.4, size 0.1, miss 0.05, imb 0.05
    # perfect match -> 0.4 + 0.4 + 0.1 + (0.5 * 0.05) + (0.5 * 0.05) = 0.9 + 0.025 + 0.025 = 0.95
    assert abs(sim - 0.95) < 1e-5
