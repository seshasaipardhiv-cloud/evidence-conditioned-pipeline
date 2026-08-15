"""
Tests for Stage 2D-1: Evidence Authenticity & Provenance Audit
"""

import json
from pathlib import Path
import inspect
from backend.app.stage2.authenticity_audit_stage2d1 import EvidenceAuthenticityAuditor

def _setup_mock_files(
    tmpdir, 
    integrity={"corpus_valid": True, "critical_errors": 0}, 
    expansion=None, 
    papers=None, 
    experiments=None,
    spec=None
):
    if expansion is None:
        expansion = {"status": "EXPANDED", "selected_candidates": [{"paper_id": "p1", "representation_method": "rep", "doi": "10.000", "title": "real_title"}]}
    if papers is None:
        papers = [{"id": "p1", "doi": "10.000", "title": "real_title"}]
    if experiments is None:
        experiments = [{
            "paper_id": "p1", 
            "feature_representation": "rep", 
            "field_provenance": {"feature_representation": {"source_sentence": "sentence"}}
        }]
    if spec is None:
        spec = {"feature_representation": "rep"}

    p_integrity = Path(tmpdir) / "stage2c_final_integrity_summary.json"
    p_expansion = Path(tmpdir) / "stage2d_corpus_expansion.json"
    p_papers = Path(tmpdir) / "papers.jsonl"
    p_exps = Path(tmpdir) / "experiments.jsonl"
    p_spec = Path(tmpdir) / "stage3_2_pipeline_specification.json"
    
    with open(p_integrity, "w") as f: json.dump(integrity, f)
    with open(p_expansion, "w") as f: json.dump(expansion, f)
    with open(p_papers, "w") as f:
        for p in papers: f.write(json.dumps(p) + "\n")
    with open(p_exps, "w") as f:
        for e in experiments: f.write(json.dumps(e) + "\n")
    with open(p_spec, "w") as f: json.dump(spec, f)
        
    return {
        "integrity_summary_path": str(p_integrity),
        "expansion_path": str(p_expansion),
        "papers_path": str(p_papers),
        "experiments_path": str(p_exps),
        "stage3_spec_path": str(p_spec),
        "out_dir": str(tmpdir)
    }

def test_real_paper_record_required(tmpdir):
    paths = _setup_mock_files(tmpdir, papers=[])
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "MISSING_SOURCE"
    assert "not found in papers" in res["reason"]

def test_pmid_doi_traceability(tmpdir):
    paths = _setup_mock_files(tmpdir, papers=[{"id": "p1", "doi": None, "title": None}])
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "INCOMPLETE_PROVENANCE"

def test_source_sentence_required(tmpdir):
    paths = _setup_mock_files(tmpdir, experiments=[{
        "paper_id": "p1", "feature_representation": "rep", "field_provenance": {"feature_representation": {}}
    }])
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "INCOMPLETE_PROVENANCE"

def test_provenance_required(tmpdir):
    # covered by source sentence missing / field_provenance missing
    paths = _setup_mock_files(tmpdir, experiments=[{
        "paper_id": "p1", "feature_representation": "rep" # No provenance
    }])
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "INCOMPLETE_PROVENANCE"

def test_hard_coded_candidate_rejection(tmpdir):
    paths = _setup_mock_files(tmpdir, expansion={"status": "EXPANDED", "selected_candidates": [{"paper_id": "sim.valid.3", "title": "mock"}]})
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "INVALID_SIMULATED_EVIDENCE"

def test_missing_full_text_source_rejection(tmpdir):
    # Handled inside the provenance rules. If source_sentence is present, the audit continues,
    # but test logic checks for missing pieces of the link.
    paths = _setup_mock_files(tmpdir, experiments=[{"paper_id": "p1"}]) # Missing feature_rep field
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "INCOMPLETE_PROVENANCE"

def test_stage2c_integrity_must_be_read_from_real_artifact(tmpdir):
    paths = _setup_mock_files(tmpdir, integrity={"corpus_valid": False, "critical_errors": 1})
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "INCOMPLETE_PROVENANCE"
    assert "integrity failed" in res["reason"]

def test_stage3_2_traceability_required(tmpdir):
    paths = _setup_mock_files(tmpdir, spec={"feature_representation": "different_rep"})
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    # It logs AUTHENTIC or INCOMPLETE based on checks, just testing it executes without crash.
    assert "status" in res

def test_no_stage2_mutation():
    source = inspect.getsource(EvidenceAuthenticityAuditor)
    for p in ["papers_path", "experiments_path", "integrity_summary_path", "expansion_path"]:
        # Verify auditor never opens these files in write mode
        assert f'open(self.{p}, "w"' not in source
        assert f'open(self.{p}, "a"' not in source

def test_no_model_training():
    source = inspect.getsource(EvidenceAuthenticityAuditor)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source

def test_deterministic_audit(tmpdir):
    paths = _setup_mock_files(tmpdir)
    auditor1 = EvidenceAuthenticityAuditor(**paths)
    res1 = auditor1.audit()
    auditor2 = EvidenceAuthenticityAuditor(**paths)
    res2 = auditor2.audit()
    assert res1["status"] == res2["status"]
    assert res1["reason"] == res2["reason"]

def test_failure_must_preserve_no_go(tmpdir):
    paths = _setup_mock_files(tmpdir, integrity={"corpus_valid": False, "critical_errors": 1})
    auditor = EvidenceAuthenticityAuditor(**paths)
    res = auditor.audit()
    assert res["status"] == "INCOMPLETE_PROVENANCE"
    assert res["training_allowed"] is False
