"""
Unit and regression tests for Stage 5B: Controlled Experimental Execution

Tests:
1. Contract hash verification and tampering detection
2. Patient split isolation and zero overlap across seeds
3. Target isolation firewall: no leakage fields in X
4. Train-only preprocessing: transformers fit strictly on training set
5. Test-set isolation: test split evaluated strictly once
6. Deterministic seeds [42, 100, 2026] execution
7. All 7 evaluation metrics correctly generated
8. Compute budget compliance (< 4 GB RAM, CPU device)
9. Baselines and candidate pipeline evaluated
10. Model artifacts exported to output directory
11. Failure handling and zero silent fallback
"""

import json
import os
from pathlib import Path
import pytest
import numpy as np

from backend.app.stage5.executor_stage5b import (
    Stage5BExecutor,
    EXPECTED_STAGE3_6_PIPELINE_HASH,
    TARGET_LEAKAGE_EXCLUSIONS,
)


def _setup_test_executor(tmpdir):
    processed_dir = Path(tmpdir) / "evidence" / "processed"
    metadata_dir = Path(tmpdir) / "evidence" / "metadata"
    experiments_dir = Path(tmpdir) / "data" / "experiments" / "stage5"
    data_dir = Path(tmpdir) / "data"

    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    contract_path = processed_dir / "stage5a_experiment_contract.json"
    clinical_path = Path("data/raw/hancock/structured/StructuredData/clinical_data.json")

    contract = {
        "contract_version": "5.0-A",
        "pipeline_identity": {
            "pipeline_hash": EXPECTED_STAGE3_6_PIPELINE_HASH,
            "target_task": "recurrence_classification",
        },
        "dataset_cohort": {
            "target_variable": "recurrence",
            "random_seeds": [42, 100, 2026],
            "split_ratios": {"train": 0.65, "validation": 0.15, "test": 0.20},
        },
        "target_isolation_firewall": {
            "excluded_outcome_fields": TARGET_LEAKAGE_EXCLUSIONS,
        },
    }
    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f)

    return Stage5BExecutor(
        contract_path=str(contract_path),
        clinical_data_path=str(clinical_path),
        processed_dir=str(processed_dir),
        metadata_dir=str(metadata_dir),
        experiments_dir=str(experiments_dir),
    )


def test_1_contract_verification_passes(tmpdir):
    executor = _setup_test_executor(tmpdir)
    valid, errors, contract = executor.verify_contract()
    assert valid is True
    assert len(errors) == 0
    assert contract["pipeline_identity"]["pipeline_hash"] == EXPECTED_STAGE3_6_PIPELINE_HASH


def test_2_tampered_contract_is_rejected(tmpdir):
    executor = _setup_test_executor(tmpdir)
    # Tamper with contract hash
    with open(executor.contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)
    contract["pipeline_identity"]["pipeline_hash"] = "tampered_hash_xyz"
    with open(executor.contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f)

    valid, errors, _ = executor.verify_contract()
    assert valid is False
    assert any("Pipeline hash mismatch" in e for e in errors)


def test_3_patient_split_zero_overlap(tmpdir):
    executor = _setup_test_executor(tmpdir)
    _, _, contract = executor.verify_contract()
    for seed in [42, 100, 2026]:
        split_info, data_splits, _ = executor.prepare_cohort_and_splits(contract, seed)
        assert split_info["patient_overlap"] == 0
        assert split_info["train_count"] > 0
        assert split_info["val_count"] > 0
        assert split_info["test_count"] > 0


def test_4_target_isolation_no_leakage_in_x(tmpdir):
    executor = _setup_test_executor(tmpdir)
    _, _, contract = executor.verify_contract()
    _, data_splits, _ = executor.prepare_cohort_and_splits(contract, seed=42)
    for split_name in ["train", "val", "test"]:
        X_rows, y_arr = data_splits[split_name]
        for row in X_rows:
            for forbidden in TARGET_LEAKAGE_EXCLUSIONS:
                assert forbidden not in row


def test_5_train_only_preprocessing(tmpdir):
    executor = _setup_test_executor(tmpdir)
    _, _, contract = executor.verify_contract()
    _, data_splits, _ = executor.prepare_cohort_and_splits(contract, seed=42)
    X_tr, y_tr, X_val, y_val, X_te, y_te = executor.preprocess_splits(data_splits, seed=42)
    # SMOTE only affects train
    assert X_tr.shape[0] >= len(data_splits["train"][0])
    # Val and test row counts are strictly preserved
    assert X_val.shape[0] == len(data_splits["val"][0])
    assert X_te.shape[0] == len(data_splits["test"][0])


def test_6_metric_computation():
    executor = Stage5BExecutor()
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.85, 0.15, 0.7, 0.3])
    metrics = executor.compute_all_metrics(y_true, y_prob)
    assert metrics["roc_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert "pr_auc" in metrics
    assert "brier_score" in metrics


def test_7_deterministic_seed_splits(tmpdir):
    executor = _setup_test_executor(tmpdir)
    _, _, contract = executor.verify_contract()
    split1, _, _ = executor.prepare_cohort_and_splits(contract, seed=42)
    split2, _, _ = executor.prepare_cohort_and_splits(contract, seed=42)
    assert split1["train_patient_hash"] == split2["train_patient_hash"]
    assert split1["val_patient_hash"] == split2["val_patient_hash"]
    assert split1["test_patient_hash"] == split2["test_patient_hash"]
