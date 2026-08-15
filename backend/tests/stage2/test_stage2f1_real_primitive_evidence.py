"""
Unit and regression tests for Stage 2F-1: Targeted Real-Literature Expansion for Implementation Primitives

16 required tests:
1. real paper identity is required
2. fake/synthetic identifiers are rejected
3. duplicate papers are rejected
4. all five primitives are searched
5. keyword-only mentions do not count
6. explicit procedural evidence is accepted
7. missing provenance is rejected
8. target leakage is rejected
9. incompatible modality is rejected
10. primitive evidence cannot be transferred between components
11. abstract-only evidence is handled conservatively
12. deterministic scoring
13. existing corpus remains unchanged
14. no model training occurs
15. training_allowed remains false
16. taxonomy is not silently modified
"""

import inspect
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from backend.app.stage2.real_primitive_retrieval_stage2f1 import Stage2F1PrimitiveRetriever, PRIMITIVES, compute_sha256
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

    return Stage2F1PrimitiveRetriever(metadata_dir=str(metadata_dir), processed_dir=str(processed_dir), full_text_fetcher=mock_fetcher)


def test_1_real_paper_identity_is_required(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand_no_id = {"title": "No ID paper", "pmid": None, "doi": None}
    res = retriever.evaluate_candidate(cand_no_id, "missing_value_handling", "some text")
    assert res["classification"] == "MISSING_PROVENANCE"


def test_2_fake_synthetic_identifiers_are_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand_fake = {"title": "Fake paper", "pmid": "sim_12345", "doi": "10.0000/sim_12345"}
    res = retriever.evaluate_candidate(cand_fake, "missing_value_handling", "some text")
    assert res["classification"] == "NOT_EVIDENCE"


def test_3_duplicate_papers_are_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [
        {"pmid": "100000", "title": "Paper 0", "doi": "10.1000/p_0"},
        {"pmid": "999999", "title": "Unique New Paper", "doi": "10.9999/unique"},
    ]
    unique, duplicates = retriever.deduplicate_candidates(candidates)
    assert len(unique) == 1
    assert len(duplicates) == 1
    assert duplicates[0]["pmid"] == "100000"


def test_4_all_five_primitives_are_searched(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    pmid_counter = [200000]

    def mock_search(query, max_results=4):
        pmid_counter[0] += 1
        return [str(pmid_counter[0])]

    def mock_summaries(pmids):
        return [{"pmid": p, "title": f"Real Title {p}", "doi": f"10.1111/{p}", "publication_year": 2024} for p in pmids]

    with patch.object(retriever, "search_pubmed", side_effect=mock_search), \
         patch.object(retriever, "fetch_summaries", side_effect=mock_summaries), \
         patch.object(retriever, "fetch_abstract", return_value="Abstract text"), \
         patch("time.sleep"):
        summary = retriever.run()
        assert set(summary["evaluations_by_primitive"].keys()) == set(PRIMITIVES)
        assert summary["queries_executed"] == 18
        for p in PRIMITIVES:
            assert summary["evaluations_by_primitive"][p] > 0


def test_5_keyword_only_mentions_do_not_count(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Discussion of ML"}
    text = "We discussed missing data and class imbalance in future work."
    res = retriever.evaluate_candidate(cand, "missing_value_handling", text)
    assert res["classification"] == "INDIRECT_INSUFFICIENT"


def test_6_explicit_procedural_evidence_is_accepted(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Imputation Study"}
    text = "We applied mean imputation to handle missing baseline clinical variables."
    res = retriever.evaluate_candidate(cand, "missing_value_handling", text)
    assert res["classification"] == "EXPLICIT_SUPPORTED"


def test_7_missing_provenance_is_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": None, "doi": None, "title": "Unidentified"}
    res = retriever.evaluate_candidate(cand, "categorical_encoding", "Categorical variables were encoded using one-hot encoding.")
    assert res["classification"] == "MISSING_PROVENANCE"


def test_8_target_leakage_is_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Leakage"}
    text = "Predictor variables including recurrence and survival_status were used."
    res = retriever.evaluate_candidate(cand, "base_learner", text)
    assert res["classification"] == "LEAKAGE_RISK"


def test_9_incompatible_modality_is_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "CT Required"}
    text = "CT imaging modality was required and indispensable for classification."
    res = retriever.evaluate_candidate(cand, "loss_function", text)
    assert res["classification"] == "INCOMPATIBLE_MODALITY"


def test_10_primitive_evidence_cannot_be_transferred_between_components(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Loss Study"}
    text = "The model was optimized using binary cross-entropy loss."
    res_loss = retriever.evaluate_candidate(cand, "loss_function", text)
    res_imb = retriever.evaluate_candidate(cand, "imbalance_handling", text)
    assert res_loss["classification"] == "EXPLICIT_SUPPORTED"
    assert res_imb["classification"] == "INDIRECT_INSUFFICIENT"


def test_11_abstract_only_evidence_handled_conservatively(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Abstract only"}
    text = "Machine learning models were applied to cancer prediction."
    res = retriever.evaluate_candidate(cand, "base_learner", text)
    assert res["classification"] != "EXPLICIT_SUPPORTED"


def test_12_deterministic_scoring(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = {"pmid": "123456", "doi": "10.1000/123456", "title": "Study"}
    text = "We applied SMOTE to handle class imbalance."
    res1 = retriever.evaluate_candidate(cand, "imbalance_handling", text)
    res2 = retriever.evaluate_candidate(cand, "imbalance_handling", text)
    assert res1["classification"] == res2["classification"]
    assert res1["score"] == res2["score"]


def test_13_existing_corpus_remains_unchanged(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(retriever.papers_path)
    exps_before = compute_sha256(retriever.experiments_path)
    mechs_before = compute_sha256(retriever.mechanisms_path)
    with patch.object(retriever, "search_pubmed", return_value=[]):
        retriever.run()
    assert compute_sha256(retriever.papers_path) == papers_before
    assert compute_sha256(retriever.experiments_path) == exps_before
    assert compute_sha256(retriever.mechanisms_path) == mechs_before


def test_14_no_model_training_occurs():
    source = inspect.getsource(Stage2F1PrimitiveRetriever)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_15_training_allowed_remains_false(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    with patch.object(retriever, "search_pubmed", return_value=[]):
        summary = retriever.run()
    assert summary["training_allowed"] is False


def test_16_taxonomy_is_not_silently_modified(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    mechs_before = retriever._load_jsonl(retriever.mechanisms_path)
    with patch.object(retriever, "search_pubmed", return_value=[]):
        retriever.run()
    mechs_after = retriever._load_jsonl(retriever.mechanisms_path)
    assert len(mechs_before) == len(mechs_after)
