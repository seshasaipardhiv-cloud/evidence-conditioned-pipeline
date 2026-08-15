"""
Unit and regression tests for Stage 2F-2: Targeted Evidence Expansion for Remaining Primitive Gaps

15 required tests:
1. real paper identity
2. synthetic evidence rejection
3. duplicate rejection
4. categorical encoding requires explicit procedure
5. loss requires explicit training objective
6. keyword-only evidence rejected
7. inferred loss rejected
8. inferred encoding rejected
9. missing provenance rejected
10. target leakage rejected
11. incompatible modality rejected
12. deterministic results
13. production corpus unchanged
14. no model fitting
15. training_allowed remains false
"""

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.stage2.real_primitive_retrieval_stage2f2 import (
    Stage2F2RemainingPrimitiveRetriever,
    PRIMITIVES_STAGE2F2,
    compute_sha256,
)
from backend.app.stage2.models import FullTextAccessStatus


def _setup_mock_environment(tmpdir, papers=None, exps=None, mechs=None):
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if papers is None:
        papers = [
            {"paper_id": f"paper_{i}", "doi": f"10.1000/p_{i}", "pmid": f"10000{i}", "title": f"Paper {i}", "publication_year": 2024}
            for i in range(30)
        ]

    if exps is None:
        exps = [
            {"experiment_id": f"exp_{p.get('paper_id')}", "paper_id": p.get("paper_id")} for p in papers
        ]

    if mechs is None:
        mechs = [
            {"mechanism_id": "mech_cross_attention", "canonical_name": "cross-attention", "category": "Attention", "mapping_status": "MAPPED"},
            {"mechanism_id": "mech_clinical_tabular_representation", "canonical_name": "clinical_tabular_representation", "category": "Representation", "mapping_status": "MAPPED"},
        ]

    with open(processed_dir / "papers.jsonl", "w", encoding="utf-8") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")

    with open(processed_dir / "experiments.jsonl", "w", encoding="utf-8") as f:
        for e in exps:
            f.write(json.dumps(e) + "\n")

    with open(processed_dir / "evidence_claims.jsonl", "w", encoding="utf-8") as f:
        f.write("")

    with open(processed_dir / "mechanisms.jsonl", "w", encoding="utf-8") as f:
        for m in mechs:
            f.write(json.dumps(m) + "\n")

    mock_fetcher = MagicMock()
    mock_fetcher.fetch.side_effect = lambda paper: (None, paper.model_copy(update={"full_text_access_status": FullTextAccessStatus.abstract_only}))

    return Stage2F2RemainingPrimitiveRetriever(metadata_dir=str(metadata_dir), processed_dir=str(processed_dir), full_text_fetcher=mock_fetcher)


def test_1_real_paper_identity(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand_no_id = {"title": "Paper Without ID", "pmid": None, "doi": None}
    res = retriever.evaluate_candidate(cand_no_id, "categorical_encoding", "Some text")
    assert res["classification"] == "MISSING_PROVENANCE"


def test_2_synthetic_evidence_rejection(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand_fake = {"title": "Fake Paper", "pmid": "sim_98765", "doi": "10.0000/sim_98765"}
    res = retriever.evaluate_candidate(cand_fake, "loss_function", "The model was trained with binary cross-entropy.")
    assert res["classification"] == "NOT_EVIDENCE"


def test_3_duplicate_rejection(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [
        {"pmid": "100000", "title": "Paper 0", "doi": "10.1000/p_0"},
        {"pmid": "888888", "title": "Brand New Paper", "doi": "10.8888/new"},
    ]
    unique, duplicates = retriever.deduplicate_candidates(candidates)
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert duplicates[0]["pmid"] == "100000"


def test_4_categorical_encoding_requires_explicit_procedure(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Encoding Study"}
    text = "Categorical variables were encoded using one-hot encoding for all baseline features."
    res = retriever.evaluate_candidate(cand, "categorical_encoding", text)
    assert res["classification"] == "EXPLICIT_SUPPORTED"


def test_5_loss_requires_explicit_training_objective(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Objective Study"}
    text = "The model was trained using binary cross-entropy as the loss function."
    res = retriever.evaluate_candidate(cand, "loss_function", text)
    assert res["classification"] == "EXPLICIT_SUPPORTED"


def test_6_keyword_only_evidence_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Overview Study"}
    text = "We analyzed categorical clinical factors and evaluated classification loss."
    res_enc = retriever.evaluate_candidate(cand, "categorical_encoding", text)
    res_loss = retriever.evaluate_candidate(cand, "loss_function", text)
    assert res_enc["classification"] == "INDIRECT_INSUFFICIENT"
    assert res_loss["classification"] == "INDIRECT_INSUFFICIENT"


def test_7_inferred_loss_rejected(tmpdir):
    # A paper mentioning XGBoost or Neural Network without stating the loss cannot resolve loss_function
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "XGBoost Classifier Study"}
    text = "An XGBoost classifier was trained to predict recurrence."
    res = retriever.evaluate_candidate(cand, "loss_function", text)
    assert res["classification"] == "INDIRECT_INSUFFICIENT"


def test_8_inferred_encoding_rejected(tmpdir):
    # A paper mentioning that the dataset has categorical variables without stating encoding
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Dataset Study"}
    text = "The clinical dataset contains 15 categorical variables and 10 numerical features."
    res = retriever.evaluate_candidate(cand, "categorical_encoding", text)
    assert res["classification"] == "INDIRECT_INSUFFICIENT"


def test_9_missing_provenance_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": None, "doi": None, "title": "No Provenance"}
    res = retriever.evaluate_candidate(cand, "categorical_encoding", "Categorical variables were encoded using one-hot encoding.")
    assert res["classification"] == "MISSING_PROVENANCE"


def test_10_target_leakage_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Leakage Study"}
    text = "Features including recurrence and days_to_recurrence were one-hot encoded."
    res = retriever.evaluate_candidate(cand, "categorical_encoding", text)
    assert res["classification"] == "LEAKAGE_RISK"


def test_11_incompatible_modality_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "MRI Study"}
    text = "MRI imaging modality was required and indispensable for classification with focal loss."
    res = retriever.evaluate_candidate(cand, "loss_function", text)
    assert res["classification"] == "INCOMPATIBLE_MODALITY"


def test_12_deterministic_results(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Determinism"}
    text = "Binary cross-entropy loss function was adopted."
    res1 = retriever.evaluate_candidate(cand, "loss_function", text)
    res2 = retriever.evaluate_candidate(cand, "loss_function", text)
    assert res1["classification"] == res2["classification"]
    assert res1["score"] == res2["score"]


def test_13_production_corpus_unchanged(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(retriever.papers_path)
    exps_before = compute_sha256(retriever.experiments_path)
    mechs_before = compute_sha256(retriever.mechanisms_path)
    with patch.object(retriever, "search_pubmed", return_value=[]):
        retriever.run()
    assert compute_sha256(retriever.papers_path) == papers_before
    assert compute_sha256(retriever.experiments_path) == exps_before
    assert compute_sha256(retriever.mechanisms_path) == mechs_before


def test_14_no_model_fitting():
    source = inspect.getsource(Stage2F2RemainingPrimitiveRetriever)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_15_training_allowed_remains_false(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    with patch.object(retriever, "search_pubmed", return_value=[]):
        summary = retriever.run()
    assert summary["training_allowed"] is False
