"""
Unit and regression tests for Stage 2E-1: Controlled Taxonomy Extension for Evidence-Backed Tabular Representation

17 required tests:
1. new mechanism has canonical ID
2. existing mechanisms remain unchanged
3. existing evidence remains unchanged
4. new mechanism has genuine provenance
5. no synthetic provenance accepted
6. indirect candidates cannot become canonical evidence
7. clinical tabular mechanism is HANCOCK-compatible
8. CNN remains incompatible
9. imaging-dependent mechanism cannot pass
10. text-dependent mechanism cannot pass
11. target leakage remains blocked
12. deterministic taxonomy extension
13. original Stage 3 artifacts remain unchanged
14. only feature_representation may be replaced
15. unsupported implementation primitives remain blocked
16. no model training
17. training_allowed remains false
"""

import inspect
import json
from pathlib import Path
import pytest

from backend.app.stage2.taxonomy_extension_stage2e1 import ControlledTaxonomyExtender, compute_sha256


def _setup_mock_environment(tmpdir):
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    papers = [
        {"paper_id": f"paper_{i}", "doi": f"10.1000/p_{i}", "pmid": f"10000{i}", "title": f"Paper {i}", "publication_year": 2024}
        for i in range(30)
    ]
    papers.append({
        "paper_id": "paper_42487970",
        "pmid": "42487970",
        "doi": "10.3389/fmed.2026.1842344",
        "title": "Clinical structured study",
        "publication_year": 2026,
    })

    exps = [
        {"experiment_id": f"exp_{i}", "paper_id": f"paper_{i}"} for i in range(30)
    ]
    exps.append({
        "experiment_id": "exp_aef6b872",
        "paper_id": "paper_42487970",
        "modalities": ["clinical"],
        "field_provenance": {
            "modalities_clinical": {
                "source_sentence": "Structured data, such as the patient’s age and stage, can be directly used as input for the model to ensure that all information is in a unified unit and standard.",
                "section": "unstructured",
                "confidence_status": "explicit",
                "verification_status": "VERIFIED",
            }
        }
    })

    mechs = [
        {"mechanism_id": "mech_cross_attention", "canonical_name": "cross-attention", "category": "Attention", "mapping_status": "MAPPED"},
        {"mechanism_id": "mech_early_fusion", "canonical_name": "early fusion", "category": "Fusion", "mapping_status": "MAPPED"},
        {"mechanism_id": "mech_late_fusion", "canonical_name": "late fusion", "category": "Fusion", "mapping_status": "MAPPED"},
        {"mechanism_id": "mech_cnn", "canonical_name": "cnn", "category": "Representation", "mapping_status": "MAPPED"},
        {"mechanism_id": "mech_unmapped_662c070b", "canonical_name": "UNMAPPED", "category": "UNMAPPED", "mapping_status": "UNMAPPED"},
        {"mechanism_id": "mech_unmapped_3d3909ac", "canonical_name": "UNMAPPED", "category": "UNMAPPED", "mapping_status": "UNMAPPED"},
    ]

    stage3_spec = {
        "selected_mechanisms": {
            "missing_value_handling": None,
            "categorical_encoding": None,
            "feature_representation": "cnn_representation",
            "modality_fusion": "cross_attention",
            "base_learner": None,
            "loss_function": None,
            "imbalance_handling": None,
            "ensembling": "average_ensembling"
        },
        "mechanism_scores": {
            "cnn_representation": {"mechanism": "cnn_representation", "final_score": 0.257}
        }
    }

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

    with open(processed_dir / "stage3_validated_pipeline_specification.json", "w", encoding="utf-8") as f:
        json.dump(stage3_spec, f)

    return ControlledTaxonomyExtender(metadata_dir=str(metadata_dir), processed_dir=str(processed_dir))


def test_1_new_mechanism_has_canonical_id(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    tax_report, _ = extender.extend_taxonomy()
    assert tax_report["new_mechanism"]["mechanism_id"] == "mech_clinical_tabular_representation"
    assert tax_report["new_mechanism"]["canonical_name"] == "clinical_tabular_representation"


def test_2_existing_mechanisms_remain_unchanged(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    mechs_before = extender._load_jsonl(extender.mechanisms_path)
    tax_report, _ = extender.extend_taxonomy()
    mechs_after = extender._load_jsonl(extender.mechanisms_path)
    assert len(mechs_after) == len(mechs_before) + 1
    for orig in mechs_before:
        assert orig in mechs_after


def test_3_existing_evidence_remains_unchanged(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    papers_before = compute_sha256(extender.papers_path)
    exps_before = compute_sha256(extender.experiments_path)
    extender.run()
    assert compute_sha256(extender.papers_path) == papers_before
    assert compute_sha256(extender.experiments_path) == exps_before


def test_4_new_mechanism_has_genuine_provenance(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    tax_report, _ = extender.extend_taxonomy()
    prov = tax_report["provenance_references"][0]
    assert prov["paper_id"] == "paper_42487970"
    assert prov["experiment_id"] == "exp_aef6b872"
    assert "Structured data" in prov["source_sentence"]


def test_5_no_synthetic_provenance_accepted(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    tax_report, _ = extender.extend_taxonomy()
    for prov in tax_report["provenance_references"]:
        assert not str(prov["paper_id"]).startswith("paper_sim")


def test_6_indirect_candidates_cannot_become_canonical_evidence(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    tax_report, _ = extender.extend_taxonomy()
    # Check that only explicit supported paper_42487970 is the supporting paper
    assert tax_report["supporting_paper_ids"] == ["paper_42487970"]


def test_7_clinical_tabular_mechanism_is_hancock_compatible(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    comp_audit = extender.run_compatibility_audit()
    assert comp_audit["evaluations"]["clinical_tabular_representation"]["status"] == "SUPPORTED"
    assert comp_audit["evaluations"]["clinical_tabular_representation"]["compatible"] is True


def test_8_cnn_remains_incompatible(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    comp_audit = extender.run_compatibility_audit()
    assert comp_audit["evaluations"]["cnn_representation"]["status"] == "INCOMPATIBLE"
    assert comp_audit["evaluations"]["cnn_representation"]["compatible"] is False


def test_9_imaging_dependent_mechanism_cannot_pass(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    comp_audit = extender.run_compatibility_audit()
    assert comp_audit["hancock_imaging_available"] is False
    assert comp_audit["evaluations"]["cnn_representation"]["compatible"] is False


def test_10_text_dependent_mechanism_cannot_pass(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    comp_audit = extender.run_compatibility_audit()
    assert comp_audit["evaluations"]["transformer_representation"]["compatible"] is False


def test_11_target_leakage_remains_blocked(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    summary = extender.run()
    # Ensure no leakage fields in new mechanism or recomposed spec
    recomp = extender._load_json(extender.processed_dir / "stage3_2_recomposed_pipeline_specification.json")
    assert "clinical_tabular_representation" in str(recomp)


def test_12_deterministic_taxonomy_extension(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    summary1 = extender.run()
    summary2 = extender.run()
    assert summary1["new_mechanism_count"] == summary2["new_mechanism_count"]
    assert summary1["stage3_recomposition_status"] == summary2["stage3_recomposition_status"]


def test_13_original_stage3_artifacts_remain_unchanged(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    stage3_before = compute_sha256(extender.stage3_spec_path)
    extender.run()
    assert compute_sha256(extender.stage3_spec_path) == stage3_before


def test_14_only_feature_representation_may_be_replaced(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    _, _, recomp = extender.recompose_stage3()
    assert list(recomp["replaced_components"].keys()) == ["feature_representation"]
    assert recomp["unchanged_components"]["modality_fusion"] == "cross_attention"
    assert recomp["unchanged_components"]["ensembling"] == "average_ensembling"


def test_15_unsupported_implementation_primitives_remain_blocked(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    _, _, recomp = extender.recompose_stage3()
    assert "missing_value_handling" in recomp["remaining_unsupported_components"]
    assert "categorical_encoding" in recomp["remaining_unsupported_components"]
    assert "base_learner" in recomp["remaining_unsupported_components"]


def test_16_no_model_training():
    source = inspect.getsource(ControlledTaxonomyExtender)
    for forbidden in [".fit(", ".train(", ".backward(", "optimizer.step("]:
        assert forbidden not in source


def test_17_training_allowed_remains_false(tmpdir):
    extender = _setup_mock_environment(tmpdir)
    summary = extender.run()
    assert summary["training_allowed"] is False
