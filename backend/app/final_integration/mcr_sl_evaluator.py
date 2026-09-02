"""
mcr_sl_evaluator.py

MCR-SL Real Multimodal Experiment Engine.

Executes real deep learning training and evaluation across 6 architecture configurations:
  1. Image Only (ResNet-18 on dermoscopic images)
  2. Clinical-Context Only (PubMedBERT on serialized context)
  3. Feature Concatenation Fusion (ResNet-18 + PubMedBERT)
  4. Late Fusion (Probability-level weighted fusion)
  5. Cross-Attention Fusion (Candidate evidence-selected fusion)
  6. Gated Multimodal Fusion (Adaptive gating network)

Enforces strict subject-level isolation across seeds [42, 100, 2026].
Collects sample-level predictions and multi-seed statistical summaries.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.app.final_integration.mcr_sl_adapter import MCRSLDatasetAdapter
from backend.app.multimodal.image_selector import ImageModelSelector
from backend.app.multimodal.multimodal_pipeline import MultimodalPipeline
from backend.app.multimodal.text_selector import TextModelSelector

logger = logging.getLogger(__name__)


def _compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true = np.array(y_true, dtype=int)
    y_prob = np.clip(np.array(y_prob, dtype=float), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)

    if len(np.unique(y_true)) > 1:
        try:
            roc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc = 0.5
        try:
            pr = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr = float(np.mean(y_true))
    else:
        roc = 0.5
        pr = float(np.mean(y_true))

    brier = float(brier_score_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
        "brier_score": round(brier, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm,
    }


class MCRSLExperimentRunner:
    """
    Executes controlled real multimodal benchmark on MCR-SL across all fusion paradigms.
    """

    def __init__(
        self,
        base_dir: str = "data/real/mcr_sl",
        seeds: Optional[List[int]] = None,
        epochs: int = 15,
        learning_rate: float = 0.015,
    ):
        self.base_dir = Path(base_dir)
        self.adapter = MCRSLDatasetAdapter(base_dir=str(self.base_dir))
        self.seeds = seeds or [42, 100, 2026]
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.image_selector = ImageModelSelector()
        self.text_selector = TextModelSelector()

    def run_all_experiments(self) -> Dict[str, Any]:
        logger.info("================================================================================")
        logger.info("STARTING REAL MCR-SL MULTIMODAL BENCHMARK (IMAGE + STRUCTURED CLINICAL CONTEXT)")
        logger.info("================================================================================")

        manifest_path = self.base_dir / "mcr_sl_manifest.csv"
        if not manifest_path.exists():
            df_manifest = self.adapter.build_manifest(seeds=self.seeds)
        else:
            df_manifest = pd.read_csv(manifest_path)

        # Architectural configurations to benchmark
        pipeline_configs = [
            {
                "config_id": "image_only",
                "name": "Image-Only (ResNet-18)",
                "modalities": ["image"],
                "fusion": "none",
                "is_multimodal": False,
            },
            {
                "config_id": "context_only",
                "name": "Clinical-Context-Only (PubMedBERT)",
                "modalities": ["text"],
                "fusion": "none",
                "is_multimodal": False,
            },
            {
                "config_id": "concatenation_fusion",
                "name": "Feature Concatenation Fusion (ResNet-18 + PubMedBERT)",
                "modalities": ["image", "text"],
                "fusion": "feature_concatenation",
                "is_multimodal": True,
            },
            {
                "config_id": "late_fusion",
                "name": "Late Fusion (Probability-Weighted)",
                "modalities": ["image", "text"],
                "fusion": "late_fusion",
                "is_multimodal": True,
            },
            {
                "config_id": "cross_attention_fusion",
                "name": "Cross-Attention Fusion (Candidate Selected)",
                "modalities": ["image", "text"],
                "fusion": "cross_attention",
                "is_multimodal": True,
            },
            {
                "config_id": "gated_fusion",
                "name": "Gated Multimodal Fusion",
                "modalities": ["image", "text"],
                "fusion": "gated_fusion",
                "is_multimodal": True,
            },
        ]

        all_predictions: List[Dict[str, Any]] = []
        experiment_results: Dict[str, Any] = {}

        for p_cfg in pipeline_configs:
            cfg_id = p_cfg["config_id"]
            cfg_name = p_cfg["name"]
            logger.info(f"Running real benchmark for architecture: {cfg_name}...")

            seed_runs = []
            for seed in self.seeds:
                split_col = f"split_seed_{seed}" if f"split_seed_{seed}" in df_manifest.columns else "split"
                train_mask = df_manifest[split_col] == "train"
                test_mask = df_manifest[split_col] == "test"

                train_df = df_manifest[train_mask].copy()
                test_df = df_manifest[test_mask].copy()

                # Verify Subject Isolation
                train_subs = set(train_df["subject_id"])
                test_subs = set(test_df["subject_id"])
                assert len(train_subs & test_subs) == 0, f"LEAKAGE: Subject overlap in seed {seed}!"

                train_images = train_df["image_path"].tolist()
                test_images = test_df["image_path"].tolist()

                train_texts = train_df["clinical_text"].tolist()
                test_texts = test_df["clinical_text"].tolist()

                y_train = train_df["target"].values.astype(int)
                y_test = test_df["target"].values.astype(int)

                # Instantiate Model
                selected_img = self.image_selector.select(sample_count=len(train_df), compute_budget="LIGHT") if "image" in p_cfg["modalities"] else None
                selected_txt = self.text_selector.select(sample_count=len(train_df), compute_budget="LIGHT") if "text" in p_cfg["modalities"] else None

                pipeline = MultimodalPipeline(
                    active_modalities=p_cfg["modalities"],
                    image_config=selected_img,
                    text_config=selected_txt,
                    fusion_mechanism=p_cfg["fusion"],
                    embed_dim=64,
                    seed=seed,
                )

                # Fit preprocessors strictly on training fold
                pipeline.fit_preprocessors(
                    image_paths=train_images if "image" in p_cfg["modalities"] else None,
                    raw_texts=train_texts if "text" in p_cfg["modalities"] else None,
                )

                # Extract and cache training features
                cached_train_reps = pipeline.extract_features(
                    image_paths=train_images if "image" in p_cfg["modalities"] else None,
                    raw_texts=train_texts if "text" in p_cfg["modalities"] else None,
                    is_training=True,
                )

                # Execute Real Training Steps
                loss_history = []
                for ep in range(self.epochs):
                    loss = pipeline.train_step(
                        y_true=y_train,
                        lr=self.learning_rate,
                        cached_reps=cached_train_reps,
                    )
                    loss_history.append(loss)
                pipeline.is_trained = True

                # Predict on Isolated Test Fold
                test_probs = pipeline.predict_proba(
                    image_paths=test_images if "image" in p_cfg["modalities"] else None,
                    raw_texts=test_texts if "text" in p_cfg["modalities"] else None,
                )
                test_preds = (test_probs >= 0.5).astype(int)

                metrics = _compute_binary_metrics(y_test, test_probs)
                metrics["seed"] = seed
                metrics["loss_final"] = round(loss_history[-1], 4)

                seed_runs.append({
                    "seed": seed,
                    "metrics": metrics,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "train_subjects_count": len(train_subs),
                    "test_subjects_count": len(test_subs),
                    "y_test": y_test.tolist(),
                    "test_probs": test_probs.tolist(),
                    "test_preds": test_preds.tolist(),
                })

                # Store Canonical Sample-Level Prediction Records
                for idx_sample, (_, s_row) in enumerate(test_df.iterrows()):
                    all_predictions.append({
                        "dataset": "MCR-SL",
                        "cohort": "Cohort_MCR_SL_Real_Multimodal",
                        "subject_id": str(s_row["subject_id"]),
                        "lesion_id": str(s_row["lesion_id"]),
                        "image_id": str(s_row["image_id"]),
                        "split": "test",
                        "seed": seed,
                        "model_name": cfg_name,
                        "fusion_name": p_cfg["fusion"],
                        "true_label": int(y_test[idx_sample]),
                        "predicted_probability": round(float(test_probs[idx_sample]), 4),
                        "predicted_class": int(test_preds[idx_sample]),
                    })

            # Multi-seed Aggregation
            rocs = [r["metrics"]["roc_auc"] for r in seed_runs]
            prs = [r["metrics"]["pr_auc"] for r in seed_runs]
            briers = [r["metrics"]["brier_score"] for r in seed_runs]
            accs = [r["metrics"]["accuracy"] for r in seed_runs]
            precs = [r["metrics"]["precision"] for r in seed_runs]
            recs = [r["metrics"]["recall"] for r in seed_runs]
            f1s = [r["metrics"]["f1"] for r in seed_runs]

            experiment_results[cfg_id] = {
                "architecture_name": cfg_name,
                "modalities": p_cfg["modalities"],
                "fusion_mechanism": p_cfg["fusion"],
                "seed_runs": seed_runs,
                "multi_seed_summary": {
                    "roc_auc_mean": round(float(np.mean(rocs)), 4),
                    "roc_auc_std": round(float(np.std(rocs)), 4),
                    "pr_auc_mean": round(float(np.mean(prs)), 4),
                    "pr_auc_std": round(float(np.std(prs)), 4),
                    "brier_score_mean": round(float(np.mean(briers)), 4),
                    "accuracy_mean": round(float(np.mean(accs)), 4),
                    "precision_mean": round(float(np.mean(precs)), 4),
                    "recall_mean": round(float(np.mean(recs)), 4),
                    "f1_mean": round(float(np.mean(f1s)), 4),
                    "f1_std": round(float(np.std(f1s)), 4),
                },
            }

        # Save Canonical MCR-SL Results
        out_results_dir = Path("evidence/final/submission/New/results")
        out_results_dir.mkdir(parents=True, exist_ok=True)
        mcr_results_file = out_results_dir / "mcr_sl_multimodal_results.json"
        with open(mcr_results_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": "MCR-SL (Multimodal Context-Rich Skin Lesion)",
                "sample_count": len(df_manifest),
                "positive_count": int(df_manifest["target"].sum()),
                "negative_count": int(len(df_manifest) - df_manifest["target"].sum()),
                "seeds_evaluated": self.seeds,
                "experiment_results": experiment_results,
            }, f, indent=2)

        # Save MCR-SL Predictions
        mcr_preds_file = Path("evidence/final/submission/New/predictions/Cohort_MCR_SL_Real_Multimodal_predictions.jsonl")
        mcr_preds_file.parent.mkdir(parents=True, exist_ok=True)
        with open(mcr_preds_file, "w", encoding="utf-8") as f:
            for pred in all_predictions:
                f.write(json.dumps(pred) + "\n")

        logger.info(f"MCR-SL experiment complete. Results: {mcr_results_file}, Predictions: {mcr_preds_file}")
        return {
            "experiment_results": experiment_results,
            "all_predictions": all_predictions,
            "manifest_sample_count": len(df_manifest),
        }
