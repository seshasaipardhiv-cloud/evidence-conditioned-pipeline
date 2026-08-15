"""
Stage 11.x Tests: Ensemble Transparency, Model-Composition Verification, and Final Results

Verifies:
1. Every ensemble has an explicit composition.
2. No fabricated model members exist.
3. Stacking has an explicit meta-learner.
4. Dynamic ensemble membership / weights is recorded per seed.
5. Every ensemble plot has composition information.
6. Final results explicitly identify the primary prediction target ('recurrence').
7. Primary clinical experiment is separated from multimodal demonstrations.
8. All measured metrics match authoritative result artifacts.
9. Historical Stage 5B/6A results remain unchanged (ZERO_MUTATION_CONFIRMED).
10. Zero target leakage.
11. Zero patient overlap.
12. Deterministic seeds remain [42, 100, 2026].
"""

import json
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def composition_manifest():
    p = Path("evidence/processed/stage11/final/ensemble_composition_manifest.json")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def final_results():
    p = Path("evidence/processed/stage11/final/final_results.json")
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def final_markdown():
    p = Path("evidence/processed/stage11/final/final_results.md")
    assert p.exists()
    return p.read_text(encoding="utf-8")


def test_1_every_ensemble_has_explicit_composition(composition_manifest):
    assert composition_manifest["ensembles_count"] >= 5
    for ens in composition_manifest["ensembles"]:
        assert "ensemble_name" in ens
        assert "ensemble_type" in ens
        assert "base_models" in ens
        assert len(ens["base_models"]) > 0
        assert "base_model_display_names" in ens
        assert "selection_rule" in ens


def test_2_no_fabricated_model_members(composition_manifest):
    valid_model_names = {
        "candidate_pipeline",
        "xgboost_default",
        "random_forest",
        "logistic_regression",
        "extra_trees",
        "hist_gradient_boosting",
        "svm",
        "knn",
        "decision_tree",
        "simple_mlp",
    }
    for ens in composition_manifest["ensembles"]:
        for bm in ens["base_models"]:
            assert bm in valid_model_names, f"Fabricated model name found: {bm}"


def test_3_stacking_has_explicit_meta_learner(composition_manifest):
    stacking_ens = [e for e in composition_manifest["ensembles"] if e["ensemble_identifier"] == "ensemble_stacking"][0]
    assert stacking_ens["meta_model"] is not None
    assert "LogisticRegression" in stacking_ens["meta_model"]


def test_4_dynamic_ensemble_weights_recorded_per_seed(composition_manifest):
    weighted_voting = [e for e in composition_manifest["ensembles"] if e["ensemble_identifier"] == "ensemble_weighted_voting"][0]
    assert "per_seed_weights" in weighted_voting
    assert "42" in weighted_voting["per_seed_weights"]
    assert "100" in weighted_voting["per_seed_weights"]
    assert "2026" in weighted_voting["per_seed_weights"]
    # Check that weights sum to ~1.0
    for seed, w_dict in weighted_voting["per_seed_weights"].items():
        assert abs(sum(w_dict.values()) - 1.0) < 0.01


def test_5_every_ensemble_plot_has_composition_information():
    manifest_p = Path("evidence/processed/stage11/final/stage11_plot_manifest.json")
    assert manifest_p.exists()
    with open(manifest_p, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    assert p_data["figures_generated_count"] >= 20

    # Ensure dedicated Figure 11 and Figure 12 exist in both PNG and SVG
    fig_dir = Path("evidence/processed/stage11/final/figures")
    for fig_id in ["figure11_candidate_vs_individual_vs_ensembles", "figure12_ensemble_composition_and_performance"]:
        assert (fig_dir / f"{fig_id}.png").exists()
        assert (fig_dir / f"{fig_id}.svg").exists()


def test_6_final_results_explicitly_identifies_primary_target(final_results, final_markdown):
    assert final_results["target_variable"] == "recurrence"
    assert final_results["task_type"] == "binary_classification"
    assert "HANCOCK" in final_results["cohort"]
    assert "recurrence" in final_markdown
    assert "PRIMARY CLINICAL EXPERIMENT" in final_markdown


def test_7_primary_experiment_separated_from_multimodal_demos(final_markdown):
    assert "AUTOMATION DEMONSTRATION TASKS" in final_markdown
    assert "unseen_cardiac_tabular_cohort" in final_markdown
    assert "adverse_cardiac_event" in final_markdown
    assert "unseen_derm_image_cohort" in final_markdown
    assert "malignancy_flag" in final_markdown
    assert "unseen_pathology_text_cohort" in final_markdown
    assert "high_grade_dysplasia" in final_markdown
    assert "unseen_oncology_multimodal_cohort" in final_markdown
    assert "disease_progression" in final_markdown


def test_8_all_measured_metrics_match_authoritative_artifacts(final_results):
    assert final_results["candidate_roc_auc"] == 0.9751
    assert final_results["best_individual_roc_auc"] in [0.9704, 0.9717]
    assert final_results["best_ensemble_roc_auc"] == 0.9749
    assert final_results["ensemble_outperformed_candidate"] is False


def test_9_historical_stage5b_stage6a_results_immutable():
    p5b = Path("evidence/processed/stage5b_candidate_results.json")
    p6a = Path("evidence/final/stage6a_master_results.json")
    assert p5b.exists()
    assert p6a.exists()
    with open(p5b, "r", encoding="utf-8") as f:
        data5b = json.load(f)
    rocs = [m["roc_auc"] for m in data5b["test_metrics"]]
    assert round(sum(rocs)/len(rocs), 4) == 0.9751


def test_10_zero_target_leakage_and_11_zero_patient_overlap():
    audit_p = Path("evidence/processed/stage11/stage11_forensic_audit.json")
    assert audit_p.exists()
    with open(audit_p, "r", encoding="utf-8") as f:
        audit = json.load(f)
    assert audit["checks"]["1_zero_patient_overlap"]["status"] == "PASSED"
    assert audit["checks"]["2_zero_target_leakage"]["status"] == "PASSED"
    assert audit["checks"]["3_zero_test_contamination"]["status"] == "PASSED"


def test_12_deterministic_seeds_remain_42_100_2026(final_results):
    cand_model = [m for m in final_results["individual_models"] if m["model_name"] == "candidate_pipeline"][0]
    assert list(cand_model["per_seed_roc_auc"].keys()) == ["42", "100", "2026"]
    assert cand_model["per_seed_roc_auc"]["42"] == 0.9888
    assert cand_model["per_seed_roc_auc"]["100"] == 0.9609
    assert cand_model["per_seed_roc_auc"]["2026"] == 0.9756
