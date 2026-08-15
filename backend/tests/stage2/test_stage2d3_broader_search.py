"""
Unit and regression tests for Stage 2D-3: Broader Real-Literature Evidence Expansion

20 required tests:
1. real identifier required
2. fake identifier rejected
3. duplicate PMID rejected
4. duplicate DOI rejected
5. duplicate title rejected
6. clinical descriptive statement rejected
7. explicit tabular representation accepted
8. imaging-only representation rejected
9. text-only representation rejected
10. target leakage rejected
11. missing provenance rejected
12. missing source sentence rejected
13. abstract-only evidence handled correctly
14. deterministic scoring
15. Stage 2C records unchanged when no candidate passes
16. authenticity gate mandatory
17. no synthetic candidates
18. no model training
19. training_allowed remains false
20. Stage 3.2 cannot consume unauthenticated evidence
"""

import inspect
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from backend.app.stage2.broader_retrieval_stage2d3 import Stage2D3BroaderRetriever, normalize_title
from backend.app.stage2.authenticity_audit_stage2d1 import EvidenceAuthenticityAuditor


def _setup_mock_environment(tmpdir, papers=None, exps=None, summary=None):
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
            "suspicious_entity_count": 0,
            "numerical_evidence_errors": 0,
            "entity_confusion_errors": 0,
        }

    known_seed_ids = [
        "paper_38396486", "paper_39074400", "paper_40325104", "paper_40449048",
        "paper_41131352", "paper_41353186", "paper_10.1038_s42256-023-00633-5",
        "paper_10.3390_bioengineering11010013"
    ]
    if papers is None:
        papers = []
        for sid in known_seed_ids:
            if "10.1038" in sid:
                doi = "10.1038/s42256-023-00633-5"
            elif "10.3390" in sid:
                doi = "10.3390/bioengineering11010013"
            else:
                doi = f"10.1000/{sid}"
            papers.append({
                "paper_id": sid,
                "doi": doi,
                "pmid": sid.replace("paper_", "") if not sid.startswith("paper_10.") else None,
                "title": f"Title {sid}",
                "publication_year": 2024
            })
        for i in range(22):
            papers.append({"paper_id": f"paper_new_{i}", "doi": f"10.2000/new_{i}", "pmid": f"90000{i}", "title": f"New Paper {i}", "publication_year": 2024})

    if exps is None:
        exps = [{"experiment_id": f"exp_{i}", "paper_id": papers[i]["paper_id"]} for i in range(len(papers))]

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

    return Stage2D3BroaderRetriever(metadata_dir=str(metadata_dir), processed_dir=str(processed_dir))


def test_1_real_identifier_required(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990011",
        "doi": "10.1016/j.compbiomed.2025.110198",
        "title": "A genuine oncology paper",
        "publication_year": 2025,
        "abstract": "We studied clinical features for classification.",
    }]
    unique, _ = retriever.deduplicate_candidates(candidates)
    assert len(unique) == 1
    assert unique[0]["pmid"] == "88990011"


def test_2_fake_identifier_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "paper_sim_fake_1",
        "doi": "10.0000/sim.valid.3",
        "title": "Fake paper",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED_SYNTHETIC"


def test_3_duplicate_pmid_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "38396486",
        "doi": "10.9999/different_doi",
        "title": "Different Title",
        "publication_year": 2024,
    }]
    unique, duplicates = retriever.deduplicate_candidates(candidates)
    assert len(unique) == 0
    assert len(duplicates) == 1
    assert "PMID 38396486 already in Stage 2C" in duplicates[0]["rejection_reason"]


def test_4_duplicate_doi_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "99999999",
        "doi": "10.1038/s42256-023-00633-5",
        "title": "Unique Title",
        "publication_year": 2024,
    }]
    unique, duplicates = retriever.deduplicate_candidates(candidates)
    assert len(unique) == 0
    assert len(duplicates) == 1
    assert "DOI" in duplicates[0]["rejection_reason"]


def test_5_duplicate_title_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "99999998",
        "doi": "10.8888/unique_doi",
        "title": "Title paper_38396486",
        "publication_year": 2024,
    }]
    unique, duplicates = retriever.deduplicate_candidates(candidates)
    assert len(unique) == 0
    assert len(duplicates) == 1
    assert "Title" in duplicates[0]["rejection_reason"]


def test_6_clinical_descriptive_statement_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990012",
        "doi": "10.1000/desc_paper",
        "title": "Descriptive patient statistics",
        "publication_year": 2024,
        "abstract": "Clinical data were collected and patient characteristics were reported for recurrence prediction.",
        "full_text": "Baseline clinical demographics were summarized.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED"
    assert "descriptive" in scored[0]["rejection_reason"].lower()


def test_7_explicit_tabular_representation_accepted(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990013",
        "doi": "10.1000/tabular_explicit",
        "title": "Structured clinical feature representation for cancer recurrence classification",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded and used as input for cancer recurrence classification.",
        "full_text": "We represented clinical tabular features as input for binary recurrence prediction and classification.",
        "full_text_available": True,
        "full_text_class": "FULL_TEXT_VERIFIED",
    }]
    scored, accepted, rep_audits, prov_audits = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 1
    assert scored[0]["status"] == "ACCEPTED"
    assert scored[0]["score"] == 100
    assert rep_audits[0]["explicitly_demonstrated"] is True
    assert prov_audits[0]["provenance_complete"] is True


def test_8_imaging_only_representation_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990014",
        "doi": "10.1000/imaging_paper",
        "title": "CT imaging features in recurrence",
        "publication_year": 2024,
        "abstract": "CT imaging features was required for recurrence classification. Clinical tabular features were encoded.",
    }]
    scored, accepted, rep_audits, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert rep_audits[0]["modality_compatible_hancock"] is False


def test_9_text_only_representation_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990015",
        "doi": "10.1000/text_paper",
        "title": "NLP for recurrence classification",
        "publication_year": 2024,
        "abstract": "Unstructured text modality was required for recurrence classification.",
    }]
    scored, accepted, rep_audits, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert rep_audits[0]["modality_compatible_hancock"] is False


def test_10_target_leakage_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990016",
        "doi": "10.1000/leak_paper",
        "title": "Predictor leakage in recurrence",
        "publication_year": 2024,
        "abstract": "Clinical tabular features including recurrence and days_to_recurrence were encoded as input features for recurrence classification.",
    }]
    scored, accepted, rep_audits, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert rep_audits[0]["target_leakage_free"] is False
    assert "leakage" in scored[0]["rejection_reason"].lower()


def test_11_missing_provenance_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990017",
        "doi": "10.1000/no_prov_paper",
        "title": "No text paper",
        "publication_year": 2024,
        "abstract": "",
        "full_text": "",
    }]
    _, accepted, _, prov_audits = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert prov_audits[0]["provenance_complete"] is False


def test_12_missing_source_sentence_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990018",
        "doi": "10.1000/no_sent_paper",
        "title": "Generic cancer statistics",
        "publication_year": 2024,
        "abstract": "Survival analysis of patient cohort was completed.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["source_sentence"] is None


def test_13_abstract_only_evidence_handled_correctly(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "88990019",
        "doi": "10.1000/abstract_paper",
        "title": "Structured clinical feature representation for cancer recurrence classification",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded and used as input for cancer recurrence classification.",
        "full_text": None,
        "full_text_available": False,
        "full_text_class": "ABSTRACT_ONLY",
    }]
    scored, accepted, rep_audits, prov_audits = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 1
    assert scored[0]["full_text_class"] == "ABSTRACT_ONLY"
    assert rep_audits[0]["full_text_class"] == "ABSTRACT_ONLY"
    assert prov_audits[0]["source_section"] == "Abstract"


def test_14_deterministic_scoring(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = [{
        "pmid": "88990020",
        "doi": "10.1000/det_score",
        "title": "Deterministic scoring test",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded and used as input for recurrence classification.",
    }]
    scored1, _, _, _ = retriever.extract_and_score(cand, [])
    scored2, _, _, _ = retriever.extract_and_score(cand, [])
    assert scored1[0]["score"] == scored2[0]["score"]
    assert scored1[0]["status"] == scored2[0]["status"]
    assert scored1[0]["rejection_reason"] == scored2[0]["rejection_reason"]


def test_15_stage2c_records_unchanged_when_no_candidate_passes(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    # Ensure baseline is exactly 30 papers
    papers = retriever._load_jsonl(retriever.papers_path)
    assert len(papers) == 30

    # Execute empty run
    with patch.object(retriever, "search_broader_pubmed", return_value=[]):
        summary = retriever.run()
        assert summary["final_decision"] == "NO_VALID_CANDIDATE"
        assert summary["papers_added"] == 0

    papers_after = retriever._load_jsonl(retriever.papers_path)
    assert len(papers_after) == 30


def test_16_authenticity_gate_mandatory(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    accepted = [{
        "pmid": "paper_sim_888",
        "doi": "10.1234/sim.valid.888",
        "title": "Synthetic mock paper",
        "representation_method": "clinical_tabular_representation",
        "evidence_sentence": "Clinical tabular features were encoded.",
        "source_section": "Results",
        "full_text_available": True,
    }]
    auth_status, auth_result, _ = retriever.run_authenticity_and_integrity(accepted)
    assert auth_status == "INVALID_SIMULATED_EVIDENCE"
    assert "simulated" in auth_result["reason"].lower()


def test_17_no_synthetic_candidates(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = [{
        "pmid": "paper_sim_fake_2",
        "doi": "10.1234/sim.valid.99",
        "title": "Synthetic fixture paper",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(cand, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED_SYNTHETIC"


def test_18_no_model_training():
    source = inspect.getsource(Stage2D3BroaderRetriever)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_19_training_allowed_remains_false(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    with patch.object(retriever, "search_broader_pubmed", return_value=[]):
        summary = retriever.run()
        assert summary["training_allowed"] is False


def test_20_stage3_2_cannot_consume_unauthenticated_evidence(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    accepted = [{
        "pmid": "paper_sim_unauth",
        "doi": "10.1234/sim.valid.unauth",
        "title": "Unauthenticated paper",
        "representation_method": "clinical_tabular_representation",
        "evidence_sentence": "Clinical tabular features were encoded.",
        "source_section": "Results",
        "full_text_available": True,
    }]
    auth_status, _, _ = retriever.run_authenticity_and_integrity(accepted)
    assert auth_status != "AUTHENTIC"
