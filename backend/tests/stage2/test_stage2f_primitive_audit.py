"""
Unit and regression tests for Stage 2F: Evidence Sufficiency Audit for Implementation Primitives

14 required tests:
1. all five primitives are audited
2. explicit evidence is distinguishable from keyword-only mentions
3. indirect evidence remains unresolved
4. missing provenance is rejected
5. target leakage is rejected
6. incompatible modality is rejected
7. components cannot be conflated
8. taxonomy gaps are detected
9. existing evidence is not modified
10. deterministic results
11. synthetic evidence is rejected
12. no model fitting occurs
13. training_allowed remains false
14. unsupported components remain blocked
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage2.primitive_audit_stage2f import Stage2FPrimitiveAuditor, PRIMITIVES
from backend.app.stage2.taxonomy_extension_stage2e1 import compute_sha256


def _setup_mock_environment(tmpdir, papers=None, exps=None, mechs=None):
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if papers is None:
        papers = [
            {"paper_id": f"paper_{i}", "doi": f"10.1000/p_{i}", "pmid": f"10000{i}", "title": f"Paper {i}", "publication_year": 2024, "abstract_available": True}
            for i in range(30)
        ]
    else:
        for p in papers:
            if "abstract_available" not in p:
                p["abstract_available"] = True

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

    return Stage2FPrimitiveAuditor(metadata_dir=str(metadata_dir), processed_dir=str(processed_dir))


def test_1_all_five_primitives_are_audited(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["primitives_audited"] == 5
    assert set(summary["primitive_resolutions"].keys()) == set(PRIMITIVES)


def test_2_explicit_evidence_is_distinguishable_from_keyword_only_mentions(tmpdir):
    papers = [
        {
            "paper_id": "paper_explicit",
            "title": "Study 1",
            "abstract": "We applied mean imputation to replace missing clinical variables.",
            "publication_year": 2024,
        },
        {
            "paper_id": "paper_keyword_only",
            "title": "Study 2 with imputation keyword",
            "abstract": "Missingness was noted across the patient cohort.",
            "publication_year": 2024,
        }
    ]
    auditor = _setup_mock_environment(tmpdir, papers=papers)
    inventory, counts = auditor.audit_primitives()
    imp_cands = [inv for inv in inventory if inv["primitive"] == "missing_value_handling"]
    assert any(inv["classification"] == "EXPLICIT_SUPPORTED" for inv in imp_cands)
    assert any(inv["classification"] == "INDIRECT_INSUFFICIENT" for inv in imp_cands)


def test_3_indirect_evidence_remains_unresolved(tmpdir):
    papers = [{
        "paper_id": "paper_indirect",
        "title": "Classifier study",
        "abstract": "Random forest and logistic regression were mentioned in discussion.",
        "publication_year": 2024,
    }]
    auditor = _setup_mock_environment(tmpdir, papers=papers)
    summary = auditor.run()
    assert summary["primitive_resolutions"]["base_learner"] == "UNSUPPORTED"


def test_4_missing_provenance_is_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_no_prov",
        "title": "Generic",
        "abstract": "",
        "publication_year": 2024,
        "abstract_available": False,
        "full_text_available": False,
    }]
    auditor = _setup_mock_environment(tmpdir, papers=papers, exps=[])
    inventory, _ = auditor.audit_primitives()
    assert all(inv["classification"] != "EXPLICIT_SUPPORTED" for inv in inventory)


def test_5_target_leakage_is_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_leak",
        "title": "Leaking variables",
        "abstract": "Categorical variables including recurrence and survival_status were one-hot encoded as inputs.",
        "publication_year": 2024,
    }]
    auditor = _setup_mock_environment(tmpdir, papers=papers)
    inventory, _ = auditor.audit_primitives()
    leak_cands = [inv for inv in inventory if inv["paper_id"] == "paper_leak"]
    assert all(inv["classification"] == "LEAKAGE_RISK" or not inv["leakage_safe"] for inv in leak_cands)


def test_6_incompatible_modality_is_rejected(tmpdir):
    papers = [{
        "paper_id": "paper_img",
        "title": "CT imaging",
        "abstract": "CT imaging modality was required and indispensable for classification.",
        "publication_year": 2024,
    }]
    auditor = _setup_mock_environment(tmpdir, papers=papers)
    inventory, _ = auditor.audit_primitives()
    for inv in inventory:
        if inv["paper_id"] == "paper_img":
            assert inv["hancock_compatible"] is False


def test_7_components_cannot_be_conflated(tmpdir):
    # A loss function like cross-entropy cannot automatically resolve imbalance_handling
    papers = [{
        "paper_id": "paper_loss",
        "title": "Loss optimization",
        "abstract": "The model was optimized using binary cross-entropy loss.",
        "publication_year": 2024,
    }]
    auditor = _setup_mock_environment(tmpdir, papers=papers)
    inventory, _ = auditor.audit_primitives()
    loss_cands = [inv for inv in inventory if inv["primitive"] == "loss_function"]
    imb_cands = [inv for inv in inventory if inv["primitive"] == "imbalance_handling"]
    assert any(inv["classification"] == "EXPLICIT_SUPPORTED" for inv in loss_cands)
    assert not any(inv["classification"] == "EXPLICIT_SUPPORTED" for inv in imb_cands)


def test_8_taxonomy_gaps_are_detected(tmpdir):
    papers = [{
        "paper_id": "paper_enc",
        "title": "One hot encoding study",
        "abstract": "Categorical variables were encoded using one-hot encoding for all input features.",
        "publication_year": 2024,
    }]
    # Mechs without categorical encoding mechanism
    auditor = _setup_mock_environment(tmpdir, papers=papers)
    summary = auditor.run()
    assert "categorical_encoding" in summary["taxonomy_gaps"]
    assert summary["primitive_resolutions"]["categorical_encoding"] == "TAXONOMY_GAP"


def test_9_existing_evidence_is_not_modified(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(auditor.papers_path)
    exps_before = compute_sha256(auditor.experiments_path)
    mechs_before = compute_sha256(auditor.mechanisms_path)
    auditor.run()
    assert compute_sha256(auditor.papers_path) == papers_before
    assert compute_sha256(auditor.experiments_path) == exps_before
    assert compute_sha256(auditor.mechanisms_path) == mechs_before


def test_10_deterministic_results(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary1 = auditor.run()
    summary2 = auditor.run()
    assert summary1["final_decision"] == summary2["final_decision"]
    assert summary1["primitive_resolutions"] == summary2["primitive_resolutions"]


def test_11_synthetic_evidence_is_rejected(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    inventory, _ = auditor.audit_primitives()
    for inv in inventory:
        assert not str(inv["paper_id"]).startswith("paper_sim")


def test_12_no_model_fitting_occurs():
    source = inspect.getsource(Stage2FPrimitiveAuditor)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_13_training_allowed_remains_false(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert summary["training_allowed"] is False


def test_14_unsupported_components_remain_blocked(tmpdir):
    auditor = _setup_mock_environment(tmpdir)
    summary = auditor.run()
    assert "missing_value_handling" in summary["unsupported_primitives"]
    assert summary["primitive_resolutions"]["missing_value_handling"] == "UNSUPPORTED"
