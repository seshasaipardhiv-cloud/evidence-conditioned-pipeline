"""
Unit and regression tests for Stage 2E: Evidence Sufficiency and Representation Decision Audit

13 required tests:
1. explicit tabular representation is recognized
2. descriptive clinical mention is rejected
3. cohort-only clinical data is rejected
4. outcome-derived representation is rejected
5. imaging representation is rejected
6. missing provenance is rejected
7. missing source sentence is rejected
8. taxonomy gap is distinguished from evidence gap
9. genuine Stage 2C corpus remains unchanged
10. deterministic decision
11. no synthetic evidence
12. no model training
13. training_allowed remains false
"""

import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.app.stage2.representation_audit_stage2e import Stage2ERepresentationAuditor


def _setup_mock_environment(tmpdir, papers=None, exps=None, mechs=None, summary=None):
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if summary is None:
        summary = {
            "corpus_valid": True,
            "critical_errors": 0,
            "warnings": 0,
            "corpus_counts": {"total_papers": 30, "seed_papers": 8, "new_papers": 22},
            "duplicate_count": {"duplicate_papers": 0},
            "provenance_coverage": {"provenance_coverage_percent": 100.0},
        }

    if papers is None:
        papers = [{"paper_id": f"paper_{i}", "doi": f"10.1000/p_{i}", "pmid": f"10000{i}", "title": f"Paper {i}", "publication_year": 2024, "abstract": ""} for i in range(30)]

    if exps is None:
        exps = [{"experiment_id": f"exp_{i}", "paper_id": f"paper_{i}"} for i in range(len(papers))]

    if mechs is None:
        mechs = [
            {"mechanism_id": "mech_cnn", "canonical_name": "cnn", "category": "Representation", "mapping_status": "MAPPED"},
            {"mechanism_id": "mech_cross_attention", "canonical_name": "cross-attention", "category": "Attention", "mapping_status": "MAPPED"},
            {"mechanism_id": "mech_unmapped_1", "canonical_name": "UNMAPPED", "category": "UNMAPPED", "mapping_status": "UNMAPPED"},
        ]

    with open(metadata_dir / "stage2c_final_integrity_summary.json", "w") as f:
        json.dump(summary, f)

    with open(processed_dir / "papers.jsonl", "w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")

    with open(processed_dir / "experiments.jsonl", "w") as f:
        for e in exps:
            f.write(json.dumps(e) + "\n")

    with open(processed_dir / "evidence_claims.jsonl", "w") as f:
        f.write("")

    with open(processed_dir / "mechanisms.jsonl", "w") as f:
        for m in mechs:
            f.write(json.dumps(m) + "\n")

    return Stage2ERepresentationAuditor(metadata_dir=str(metadata_dir), processed_dir=str(processed_dir))


def test_1_explicit_tabular_representation_recognized(tmpdir):
    papers = [{
        "paper_id": "paper_exp_tab",
        "title": "Clinical tabular ML",
        "abstract": "Clinical tabular features were encoded and used as input for cancer recurrence classification.",
        "publication_year": 2024,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers)
    inventory, counts = retriever.audit_corpus()
    assert counts["explicit_supported"] == 1
    assert inventory[0]["classification"] == "EXPLICIT_SUPPORTED"


def test_2_descriptive_clinical_mention_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_desc",
        "title": "Descriptive study",
        "abstract": "Clinical data were collected and patient characteristics were reported.",
        "publication_year": 2024,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers)
    inventory, counts = retriever.audit_corpus()
    assert counts["explicit_supported"] == 0
    assert inventory[0]["classification"] != "EXPLICIT_SUPPORTED"


def test_3_cohort_only_clinical_data_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_cohort",
        "title": "Cohort analysis",
        "abstract": "Baseline clinical demographics were summarized for 100 patients.",
        "publication_year": 2024,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers)
    inventory, counts = retriever.audit_corpus()
    assert counts["explicit_supported"] == 0
    assert inventory[0]["classification"] != "EXPLICIT_SUPPORTED"


def test_4_outcome_derived_representation_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_leak",
        "title": "Leakage paper",
        "abstract": "Clinical tabular features including recurrence and survival_status were used as input for prediction.",
        "publication_year": 2024,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers)
    inventory, _ = retriever.audit_corpus()
    assert inventory[0]["leakage_safe"] is False


def test_5_imaging_representation_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_img",
        "title": "Imaging study",
        "abstract": "CT imaging modality was essential and required for recurrence prediction.",
        "publication_year": 2024,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers)
    inventory, _ = retriever.audit_corpus()
    assert inventory[0]["hancock_compatible"] is False


def test_6_missing_provenance_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_no_prov",
        "title": "No provenance",
        "abstract": "",
        "publication_year": 2024,
        "abstract_available": False,
        "full_text_available": False,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers, exps=[{"experiment_id": "e1", "paper_id": "paper_no_prov"}])
    inventory, counts = retriever.audit_corpus()
    assert counts["explicit_supported"] == 0


def test_7_missing_source_sentence_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_no_sentence",
        "title": "Generic title without sentences",
        "abstract": "",
        "publication_year": 2024,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers)
    inventory, counts = retriever.audit_corpus()
    assert counts["explicit_supported"] == 0
    assert len(inventory[0]["source_sentences"]) == 0


def test_8_taxonomy_gap_distinguished_from_evidence_gap(tmpdir):
    papers_with_evidence = [{
        "paper_id": "paper_with_ev",
        "title": "Tabular ML",
        "abstract": "Clinical tabular features were encoded and fed as input for recurrence classification.",
        "publication_year": 2024,
    }]
    retriever = _setup_mock_environment(tmpdir, papers=papers_with_evidence)
    summary = retriever.run()
    assert summary["taxonomy_gap_status"] == "TAXONOMY_GAP"
    assert summary["final_representation_decision"] == "TAXONOMY_GAP"

    papers_without_evidence = [{
        "paper_id": "paper_no_ev",
        "title": "Pure imaging study",
        "abstract": "CT scans were analyzed.",
        "publication_year": 2024,
    }]
    mechs_sufficient = [
        {"mechanism_id": "mech_tab", "canonical_name": "tabular_representation", "category": "Representation", "mapping_status": "MAPPED"}
    ]
    retriever2 = _setup_mock_environment(tmpdir, papers=papers_without_evidence, mechs=mechs_sufficient)
    summary2 = retriever2.run()
    assert summary2["taxonomy_gap_status"] == "TAXONOMY_SUFFICIENT"
    assert summary2["final_representation_decision"] == "EVIDENCE_GAP_CONFIRMED"


def test_9_genuine_stage2c_corpus_remains_unchanged(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    papers_before = retriever._load_jsonl(retriever.papers_path)
    summary = retriever.run()
    papers_after = retriever._load_jsonl(retriever.papers_path)
    assert len(papers_before) == len(papers_after) == 30
    assert summary["total_genuine_papers_examined"] == 30


def test_10_deterministic_decision(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    summary1 = retriever.run()
    summary2 = retriever.run()
    assert summary1["final_representation_decision"] == summary2["final_representation_decision"]
    assert summary1["explicit_supported_candidates"] == summary2["explicit_supported_candidates"]


def test_11_no_synthetic_evidence(tmpdir):
    # Ensure auditor reads genuine corpus without adding synthetic fixtures to production
    retriever = _setup_mock_environment(tmpdir)
    inventory, _ = retriever.audit_corpus()
    for item in inventory:
        assert not str(item["paper_id"]).startswith("paper_sim")


def test_12_no_model_training():
    source = inspect.getsource(Stage2ERepresentationAuditor)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_13_training_allowed_remains_false(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    summary = retriever.run()
    assert summary["training_allowed"] is False
