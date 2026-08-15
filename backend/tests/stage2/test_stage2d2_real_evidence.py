"""
Unit and regression tests for Stage 2D-2: Real Literature Retrieval and Evidence Expansion

20 required tests:
1. simulated paper rejected
2. fake PMID rejected
3. fake DOI rejected
4. hard-coded candidate rejected
5. duplicate paper rejected
6. missing source sentence rejected
7. missing provenance rejected
8. clinical-only descriptive statement rejected
9. actual clinical/tabular representation accepted
10. imaging-dependent representation rejected
11. text-dependent representation rejected
12. incompatible task rejected
13. target leakage rejected
14. Stage 2C records remain unchanged
15. deterministic scoring
16. real paper metadata required
17. Stage 2D-1 must pass before acceptance
18. Stage 3.2 cannot consume unauthenticated evidence
19. no model fitting
20. training_allowed remains false
"""

import inspect
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from backend.app.stage2.real_retrieval_stage2d2 import Stage2D2RealRetriever, normalize_title
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
        papers = [{"paper_id": sid, "doi": f"10.1000/{sid}", "pmid": sid.replace("paper_", ""), "title": f"Title {sid}", "publication_year": 2024} for sid in known_seed_ids]
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

    return Stage2D2RealRetriever(metadata_dir=str(metadata_dir), processed_dir=str(processed_dir))


def test_1_simulated_paper_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "paper_sim_123",
        "doi": "10.1234/sim.valid.3",
        "title": "Simulated synthetic paper",
        "publication_year": 2024,
        "abstract": "We encoded clinical tabular features.",
        "full_text": "We encoded clinical tabular features.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED_SYNTHETIC"


def test_2_fake_pmid_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "paper_sim_fake_pmid",
        "doi": "10.1000/real_doi",
        "title": "Real sounding title",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded for classification.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert "SYNTHETIC" in scored[0]["status"] or scored[0]["score"] == 0


def test_3_fake_doi_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "12345678",
        "doi": "10.0000/sim.valid.fake",
        "title": "Some paper",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were used as input for recurrence classification.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED_SYNTHETIC"


def test_4_hard_coded_candidate_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "paper_sim_valid_3",
        "doi": "10.1234/sim.valid.3",
        "title": "Synthetic fixture candidate",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were used for recurrence prediction.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED_SYNTHETIC"


def test_5_duplicate_paper_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    # 38396486 is one of the 8 seed papers
    candidates = [{
        "pmid": "38396486",
        "doi": "10.1038/s42256-023-00633-5",
        "title": "Title paper_38396486",
        "publication_year": 2024,
    }]
    unique, duplicates = retriever.deduplicate_candidates(candidates)
    assert len(unique) == 0
    assert len(duplicates) == 1
    assert "already in Stage 2C" in duplicates[0]["rejection_reason"]


def test_6_missing_source_sentence_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910111",
        "doi": "10.1000/some_doi",
        "title": "A cancer study",
        "publication_year": 2024,
        "abstract": "We studied cancer recurrence and outcomes in patients.",
        "full_text": "We analyzed survival rates without stating any feature encoding.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED"


def test_7_missing_provenance_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910112",
        "doi": "10.1000/another_doi",
        "title": "Oncology classification",
        "publication_year": 2024,
        "abstract": "",
        "full_text": "",
    }]
    _, accepted, _, prov_audits = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert prov_audits[0]["provenance_complete"] is False


def test_8_clinical_only_descriptive_statement_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910113",
        "doi": "10.1000/desc_doi",
        "title": "Descriptive study on recurrence",
        "publication_year": 2024,
        "abstract": "Clinical data were collected and patient characteristics were reported. Clinical variables were analyzed descriptively for recurrence prediction.",
        "full_text": "Baseline clinical demographics were summarized.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED"
    assert "descriptive" in scored[0]["rejection_reason"].lower()


def test_9_actual_clinical_tabular_representation_accepted(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910114",
        "doi": "10.1000/tabular_doi",
        "title": "Structured clinical feature representation for cancer recurrence classification",
        "publication_year": 2024,
        "abstract": "We represented clinical tabular features as input for binary recurrence prediction and classification.",
        "full_text": "Clinical tabular features were encoded and fed into the classifier for recurrence classification.",
        "full_text_available": True,
    }]
    scored, accepted, rep_audits, prov_audits = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 1
    assert scored[0]["status"] == "ACCEPTED"
    assert scored[0]["score"] == 100
    assert rep_audits[0]["explicitly_demonstrated"] is True
    assert prov_audits[0]["provenance_complete"] is True


def test_10_imaging_dependent_representation_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910115",
        "doi": "10.1000/imaging_doi",
        "title": "Multimodal imaging and clinical representation",
        "publication_year": 2024,
        "abstract": "CT imaging features was required for recurrence classification. Clinical tabular features were encoded.",
        "full_text": "PET imaging modality was indispensable alongside clinical features for recurrence prediction.",
    }]
    scored, accepted, rep_audits, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert rep_audits[0]["modality_compatible_hancock"] is False


def test_11_text_dependent_representation_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910116",
        "doi": "10.1000/text_doi",
        "title": "Text reports for recurrence",
        "publication_year": 2024,
        "abstract": "Unstructured text modality was required for recurrence classification.",
    }]
    _, accepted, _, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0


def test_12_incompatible_task_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910117",
        "doi": "10.1000/task_doi",
        "title": "Descriptive statistics of clinical cohort",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded, but task was purely unsupervised clustering without classification.",
    }]
    scored, accepted, rep_audits, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert rep_audits[0]["task_compatible_recurrence_classification"] is False


def test_13_target_leakage_rejected(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    candidates = [{
        "pmid": "78910118",
        "doi": "10.1000/leak_doi",
        "title": "Target leakage in features",
        "publication_year": 2024,
        "abstract": "Clinical tabular features including recurrence and survival_status were encoded and fed as input for recurrence classification.",
    }]
    scored, accepted, rep_audits, _ = retriever.extract_and_score(candidates, [])
    assert len(accepted) == 0
    assert rep_audits[0]["target_leakage_free"] is False
    assert "leakage" in scored[0]["rejection_reason"].lower()


def test_14_stage2c_records_remain_unchanged(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    pre = retriever.verify_pre_search_integrity()
    assert pre["corpus_valid"] is True
    assert pre["actual_papers_count"] == 30

    # Ensure no write operations modify the original 30 papers unless authentic candidate passed
    papers = retriever._load_jsonl(retriever.papers_path)
    assert len(papers) == 30


def test_15_deterministic_scoring(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = [{
        "pmid": "78910119",
        "doi": "10.1000/det_doi",
        "title": "Deterministic test",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded and used as input for recurrence classification.",
    }]
    scored1, _, _, _ = retriever.extract_and_score(cand, [])
    scored2, _, _, _ = retriever.extract_and_score(cand, [])
    assert scored1[0]["score"] == scored2[0]["score"]
    assert scored1[0]["status"] == scored2[0]["status"]


def test_16_real_paper_metadata_required(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    cand = [{
        "pmid": "paper_sim_fake",
        "doi": "10.1234/sim.valid.3",
        "title": "Simulated candidate",
        "publication_year": 2024,
        "abstract": "Clinical tabular features were encoded.",
    }]
    scored, accepted, _, _ = retriever.extract_and_score(cand, [])
    assert len(accepted) == 0
    assert scored[0]["status"] == "REJECTED_SYNTHETIC"


def test_17_stage2d1_must_pass_before_acceptance(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    # Synthetic candidate
    accepted = [{
        "pmid": "paper_sim_999",
        "doi": "10.1234/sim.valid.999",
        "title": "Mocked paper",
        "representation_method": "clinical_tabular_representation",
        "evidence_sentence": "Clinical tabular features were encoded.",
        "source_section": "Results",
        "full_text_available": True,
    }]
    auth_status, _, _ = retriever.run_authenticity_and_integrity(accepted)
    assert auth_status == "INVALID_SIMULATED_EVIDENCE"


def test_18_stage3_2_cannot_consume_unauthenticated_evidence(tmpdir):
    # If authenticity fails, Stage 2D-2 returns NO_VALID_CANDIDATE and does not unblock Stage 3.2
    retriever = _setup_mock_environment(tmpdir)
    accepted = [{
        "pmid": "paper_sim_999",
        "doi": "10.1234/sim.valid.999",
        "title": "Mocked paper",
        "representation_method": "clinical_tabular_representation",
        "evidence_sentence": "Clinical tabular features were encoded.",
        "source_section": "Results",
        "full_text_available": True,
    }]
    auth_status, _, _ = retriever.run_authenticity_and_integrity(accepted)
    assert auth_status != "AUTHENTIC"


def test_19_no_model_fitting():
    source = inspect.getsource(Stage2D2RealRetriever)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_20_training_allowed_remains_false(tmpdir):
    retriever = _setup_mock_environment(tmpdir)
    with patch.object(retriever, "search_pubmed", return_value=[]):
        summary = retriever.run()
        assert summary["training_allowed"] is False
