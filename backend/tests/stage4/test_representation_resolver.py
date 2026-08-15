"""
Tests for Stage 4F: RepresentationResolver

Covers all 13 required safety assertions:
 1. CNN remains rejected without imaging.
 2. Pathology is never treated as imaging.
 3. Unsupported representations remain blocked.
 4. Evidence-backed representation requires provenance.
 5. Explicit configuration is distinguishable from literature evidence.
 6. No arbitrary fallback is selected.
 7. Stage 2 artifacts remain unchanged.
 8. Stage 3 artifacts remain unchanged.
 9. Target leakage remains impossible.
10. No model fitting occurs.
11. A compatible mocked representation can resolve the blocker without training.
12. Compatibility is revalidated after replacement.
13. Final readiness changes only when every required gate passes.
"""

from __future__ import annotations

import json
import os
import copy
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch, MagicMock

import pytest

from backend.app.stage4.representation_resolver import (
    RepresentationResolver,
    IMAGING_ONLY_REPRESENTATIONS,
    PATHOLOGY_SLIDE_REPRESENTATIONS,
    HANCOCK_NON_IMAGING_MODALITIES,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

REAL_STAGE2_EXPERIMENTS = "evidence/processed/experiments.jsonl"
REAL_STAGE2_MECHANISMS = "evidence/processed/mechanisms.jsonl"
REAL_STAGE3_SPEC = "evidence/processed/stage3_validated_pipeline_specification.json"
REAL_STAGE3_RANKINGS = "evidence/processed/stage3_mechanism_rankings.json"
REAL_STAGE1_PROFILE = "data/metadata/hancock/stage1_profile_report.json"
REAL_EXISTING_RESOLUTION = "data/metadata/hancock/stage4_feature_representation_resolution.json"
REAL_IMPL_CONFIG = "data/config/implementation_config.json"


def _make_resolver(
    stage2_experiments: Optional[List[Dict]] = None,
    stage2_mechanisms: Optional[List[Dict]] = None,
    stage3_rankings: Optional[Dict] = None,
    stage1_profile: Optional[Dict] = None,
    impl_config: Optional[Dict] = None,
    tmpdir: Optional[str] = None,
) -> tuple[RepresentationResolver, str]:
    """
    Build a RepresentationResolver backed by temp files when mock data
    is provided; otherwise fall through to real files.
    """
    td = tmpdir or tempfile.mkdtemp()

    def _write(filename: str, data: Any, jsonl: bool = False) -> str:
        path = os.path.join(td, filename)
        with open(path, "w", encoding="utf-8") as f:
            if jsonl:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            else:
                json.dump(data, f)
        return path

    exp_path = (
        _write("experiments.jsonl", stage2_experiments, jsonl=True)
        if stage2_experiments is not None
        else REAL_STAGE2_EXPERIMENTS
    )
    mech_path = (
        _write("mechanisms.jsonl", stage2_mechanisms, jsonl=True)
        if stage2_mechanisms is not None
        else REAL_STAGE2_MECHANISMS
    )
    rankings_path = (
        _write("rankings.json", stage3_rankings)
        if stage3_rankings is not None
        else REAL_STAGE3_RANKINGS
    )
    profile_path = (
        _write("profile.json", stage1_profile)
        if stage1_profile is not None
        else REAL_STAGE1_PROFILE
    )
    config_path = (
        _write("impl_config.json", impl_config)
        if impl_config is not None
        else REAL_IMPL_CONFIG
    )

    # Existing resolution — always use real file if available
    existing_res_path = REAL_EXISTING_RESOLUTION

    out_path = os.path.join(td, "stage4_representation_resolution.json")

    resolver = RepresentationResolver(
        stage2_experiments_path=exp_path,
        stage2_mechanisms_path=mech_path,
        stage3_spec_path=REAL_STAGE3_SPEC,
        stage3_rankings_path=rankings_path,
        stage1_profile_path=profile_path,
        existing_resolution_path=existing_res_path,
        impl_config_path=config_path,
        out_path=out_path,
    )
    return resolver, out_path


# ---------------------------------------------------------------------------
# Test 1: CNN remains rejected without imaging
# ---------------------------------------------------------------------------

def test_cnn_remains_rejected_without_imaging():
    """
    CNN-based representation must always appear in incompatible_candidates
    when HANCOCK provides no imaging modality.
    """
    resolver, _ = _make_resolver()
    report = resolver.resolve()

    assert "cnn_representation" in report["incompatible_candidates"], (
        "cnn_representation must be in incompatible_candidates when no imaging present"
    )
    assert report["selected_replacement"] != "cnn_representation"
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 2: Pathology is never treated as imaging
# ---------------------------------------------------------------------------

def test_pathology_not_treated_as_imaging():
    """
    HANCOCK pathology modality is structured tabular staging data.
    It must never unlock imaging-only representations.
    """
    # Provide a profile where pathology is available but imaging is not
    profile = {
        "pathology": {
            "patient_count": 763,
            "categorical_fields": ["primary_tumor_site", "pT_stage"],
        }
    }
    resolver, _ = _make_resolver(stage1_profile=profile)
    mods = resolver._get_hancock_available_modalities()

    assert mods["imaging"] is False, "imaging must be False when only pathology is present"
    assert mods["pathology_tabular"] is True

    # Resolve — CNN must still be incompatible
    report = resolver.resolve()
    assert "cnn_representation" in report["incompatible_candidates"]
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 3: Unsupported representations remain blocked
# ---------------------------------------------------------------------------

def test_unsupported_representations_remain_blocked():
    """
    A mechanism with zero Stage 3 evidence and not in the explicit config
    must remain in INSUFFICIENT_EVIDENCE / not selected.
    """
    rankings = {
        "feature_representation": {
            "winner": "cnn_representation",
            "alternatives": [
                {
                    "mechanism": "transformer_representation",
                    "evidence_count": 0,
                    "support_count": 0,
                    "contradiction_count": 0,
                    "context_similarity_sum": 0.0,
                    "evidence_quality_sum": 0.0,
                    "final_score": 0.0,
                }
            ],
        }
    }
    resolver, _ = _make_resolver(stage3_rankings=rankings)
    report = resolver.resolve()

    # transformer_representation has 0 evidence → must not be selected
    assert report["selected_replacement"] != "transformer_representation"
    assert report["final_resolution_status"] == "BLOCKED"
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 4: Evidence-backed representation requires provenance
# ---------------------------------------------------------------------------

def test_evidence_backed_requires_provenance():
    """
    A candidate recorded as evidence-backed must originate from an actual
    Stage 2 experiment entry.  A mocked experiment that provides an explicit
    feature_representation for non-imaging modalities should appear as an
    evidence-backed candidate with a stage2 source.
    """
    mock_exp = [
        {
            "experiment_id": "exp_mock_tabular",
            "paper_id": "paper_mock_01",
            "dataset": "SYNTHETIC",
            "task": "classification",
            "modalities": ["clinical", "blood"],
            "feature_representation": "tabular_mlp",
            "fusion_strategy": None,
            "field_provenance": {
                "feature_representation": {
                    "source_sentence": "We use an MLP to encode tabular clinical features.",
                    "section": "methods",
                    "extraction_method": "regex_based",
                    "confidence_status": "explicit",
                }
            },
        }
    ]
    resolver, _ = _make_resolver(stage2_experiments=mock_exp)
    report = resolver.resolve()

    # Check provenance chain
    assert "tabular_mlp" in report["evidence_backed_candidates"], (
        "tabular_mlp from mocked Stage 2 experiment should be evidence-backed"
    )
    found = next(
        c for c in report["candidate_representations"]
        if c["mechanism"] == "tabular_mlp"
    )
    assert found["source"] == "stage2_corpus"
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 5: Explicit configuration is distinguishable from literature evidence
# ---------------------------------------------------------------------------

def test_explicit_config_distinct_from_literature():
    """
    A value set in implementation_config.json must appear in
    explicitly_configured_candidates, NOT in evidence_backed_candidates.
    """
    config = {
        "feature_representation": "gradient_boosting_features",
        "missing_value_handling": "mean_imputation",
    }
    resolver, _ = _make_resolver(impl_config=config)
    report = resolver.resolve()

    assert "gradient_boosting_features" in report["explicitly_configured_candidates"]
    assert "gradient_boosting_features" not in report["evidence_backed_candidates"]
    assert report["provenance"] == "explicit_configuration"
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 6: No arbitrary fallback is selected
# ---------------------------------------------------------------------------

def test_no_arbitrary_fallback_selected():
    """
    When no valid candidate exists (empty Stage 2, zero-evidence Stage 3,
    null impl_config), the resolver must NOT choose an arbitrary default.
    """
    empty_rankings = {"feature_representation": {"winner": "cnn_representation", "alternatives": []}}
    resolver, _ = _make_resolver(
        stage2_experiments=[],
        stage3_rankings=empty_rankings,
        impl_config={"feature_representation": None, "missing_value_handling": "mean_imputation"},
    )
    report = resolver.resolve()

    assert report["selected_replacement"] is None
    assert report["final_resolution_status"] == "BLOCKED"
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 7: Stage 2 artifacts remain unchanged after resolution
# ---------------------------------------------------------------------------

def test_stage2_artifacts_unchanged():
    """
    Resolve must not mutate any Stage 2 artifacts on disk.
    """
    # Read before
    mtime_before = Path(REAL_STAGE2_EXPERIMENTS).stat().st_mtime if Path(REAL_STAGE2_EXPERIMENTS).exists() else None

    resolver, _ = _make_resolver()
    resolver.resolve()

    mtime_after = Path(REAL_STAGE2_EXPERIMENTS).stat().st_mtime if Path(REAL_STAGE2_EXPERIMENTS).exists() else None
    assert mtime_before == mtime_after, "Stage 2 experiments.jsonl was modified during resolve()"


# ---------------------------------------------------------------------------
# Test 8: Stage 3 artifacts remain unchanged after resolution
# ---------------------------------------------------------------------------

def test_stage3_artifacts_unchanged():
    """
    Resolve must not mutate any Stage 3 artifacts on disk.
    """
    mtime_before_spec = Path(REAL_STAGE3_SPEC).stat().st_mtime if Path(REAL_STAGE3_SPEC).exists() else None
    mtime_before_rank = Path(REAL_STAGE3_RANKINGS).stat().st_mtime if Path(REAL_STAGE3_RANKINGS).exists() else None

    resolver, _ = _make_resolver()
    resolver.resolve()

    mtime_after_spec = Path(REAL_STAGE3_SPEC).stat().st_mtime if Path(REAL_STAGE3_SPEC).exists() else None
    mtime_after_rank = Path(REAL_STAGE3_RANKINGS).stat().st_mtime if Path(REAL_STAGE3_RANKINGS).exists() else None

    assert mtime_before_spec == mtime_after_spec, "Stage 3 spec was modified"
    assert mtime_before_rank == mtime_after_rank, "Stage 3 rankings were modified"


# ---------------------------------------------------------------------------
# Test 9: Target leakage remains impossible
# ---------------------------------------------------------------------------

def test_target_leakage_impossible():
    """
    The resolution report must not include target-derived fields
    (recurrence, survival_status, days_to_recurrence, etc.) as
    candidate mechanisms or selected representations.
    """
    TARGET_DERIVED = {
        "recurrence", "survival_status", "survival_status_with_cause",
        "days_to_recurrence", "days_to_last_information",
        "days_to_progress_1", "days_to_progress_2", "days_to_metastasis_1",
    }
    resolver, _ = _make_resolver()
    report = resolver.resolve()

    all_mechs = [c["mechanism"] for c in report["candidate_representations"]]
    for mech in all_mechs:
        assert mech not in TARGET_DERIVED, f"Target-derived field '{mech}' found in candidates"

    assert report.get("selected_replacement") not in TARGET_DERIVED


# ---------------------------------------------------------------------------
# Test 10: No model fitting occurs
# ---------------------------------------------------------------------------

def test_no_model_fitting_occurs():
    """
    The resolver must never call fit(), train(), or any optimizer step.
    Verified by confirming the resolver class has no fit/train references
    and that resolve() completes without training_allowed flipping to True.
    """
    import inspect
    import backend.app.stage4.representation_resolver as mod
    source = inspect.getsource(mod)

    # The module must not contain any fit() or train() calls
    for forbidden in ["model.fit(", ".train(", "optimizer.step(", "backward("]:
        assert forbidden not in source, (
            f"Forbidden training call '{forbidden}' found in representation_resolver.py"
        )

    # Resolving with real data must still have training_allowed=False
    resolver, _ = _make_resolver()
    report = resolver.resolve()
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 11: A compatible mocked representation can resolve the blocker
#          without enabling training
# ---------------------------------------------------------------------------

def test_mocked_evidence_backed_representation_resolves_blocker():
    """
    When Stage 2 contains an explicit, non-imaging feature_representation
    compatible with HANCOCK's modalities, it should resolve the blocker.
    training_allowed must remain False.
    """
    mock_exp = [
        {
            "experiment_id": "exp_mock_clinical",
            "paper_id": "paper_mock_02",
            "dataset": "SYNTHETIC_CLINICAL",
            "task": "classification",
            "modalities": ["clinical"],
            "feature_representation": "gradient_boosting_leaf_embedding",
            "fusion_strategy": None,
            "field_provenance": {
                "feature_representation": {
                    "source_sentence": "Gradient boosting leaf embeddings encode clinical features.",
                    "extraction_method": "regex_based",
                    "confidence_status": "explicit",
                }
            },
        }
    ]
    resolver, out_path = _make_resolver(stage2_experiments=mock_exp)
    report = resolver.resolve()

    assert report["selected_replacement"] == "gradient_boosting_leaf_embedding"
    assert "gradient_boosting_leaf_embedding" in report["evidence_backed_candidates"]
    assert report["final_resolution_status"] in (
        "RESOLVED_EVIDENCE_BACKED", "RESOLVED_EXPLICIT"
    )
    assert report["training_allowed"] is False  # HARD RULE


# ---------------------------------------------------------------------------
# Test 12: Compatibility is revalidated after replacement
# ---------------------------------------------------------------------------

def test_compatibility_revalidated_after_replacement():
    """
    When a mocked replacement is selected, the candidate entry must record
    a compatibility_status of POTENTIALLY_COMPATIBLE (not INCOMPATIBLE).
    The resolver must not skip the modality check.
    """
    mock_exp = [
        {
            "experiment_id": "exp_tabular_only",
            "paper_id": "paper_mock_03",
            "dataset": "CLINICAL_DS",
            "task": "classification",
            "modalities": ["clinical", "blood"],
            "feature_representation": "mlp_encoder",
            "fusion_strategy": None,
            "field_provenance": {},
        }
    ]
    resolver, _ = _make_resolver(stage2_experiments=mock_exp)
    report = resolver.resolve()

    mlp_candidates = [
        c for c in report["candidate_representations"]
        if c["mechanism"] == "mlp_encoder"
    ]
    assert len(mlp_candidates) == 1
    assert mlp_candidates[0]["compatibility_status"] == "POTENTIALLY_COMPATIBLE", (
        f"Expected POTENTIALLY_COMPATIBLE but got {mlp_candidates[0]['compatibility_status']}"
    )
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Test 13: Final readiness changes only when every gate passes
# ---------------------------------------------------------------------------

def test_final_readiness_blocked_without_valid_representation():
    """
    With no valid evidence-backed or explicitly-configured candidate,
    the overall resolution status must be BLOCKED and training_allowed False.
    The real HANCOCK dataset/evidence has no valid representation → BLOCKED.
    """
    resolver, _ = _make_resolver()
    report = resolver.resolve()

    # Using real files: no valid candidate exists
    assert report["final_resolution_status"] == "BLOCKED"
    assert report["selected_replacement"] is None
    assert report["training_allowed"] is False


# ---------------------------------------------------------------------------
# Additional: verify output artifact is written to disk
# ---------------------------------------------------------------------------

def test_output_artifact_is_written():
    resolver, out_path = _make_resolver()
    resolver.resolve()
    assert Path(out_path).exists(), "Resolution artifact was not written to disk"
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "training_allowed" in data
    assert data["training_allowed"] is False


# ---------------------------------------------------------------------------
# Additional: imaging-only constants are correctly classified
# ---------------------------------------------------------------------------

def test_imaging_only_constant_set():
    assert "cnn_representation" in IMAGING_ONLY_REPRESENTATIONS
    assert "cnn" in IMAGING_ONLY_REPRESENTATIONS
    assert "wsi_representation" in PATHOLOGY_SLIDE_REPRESENTATIONS


def test_hancock_modalities_never_include_imaging():
    resolver, _ = _make_resolver()
    mods = resolver._get_hancock_available_modalities()
    assert mods["imaging"] is False
    assert mods["clinical"] is True
    assert mods["pathology_tabular"] is True
    assert mods["blood"] is True
    assert mods["text"] is True
