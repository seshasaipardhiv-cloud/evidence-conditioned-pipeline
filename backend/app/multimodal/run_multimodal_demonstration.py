"""
End-to-End Multimodal Execution & Demonstration Runner

Executes the complete evidence-conditioned multimodal synthesis and evaluation workflow:
1. Discovers modalities from dataset sources.
2. Selects image and text architectures grounded in biomedical literature evidence.
3. Preprocesses image and clinical text data with strict train-only isolation.
4. Executes 14 safety gates.
5. Trains and evaluates candidate multimodal pipeline with Bi-directional Cross-Attention Fusion.
6. Benchmarks against unimodal baselines (Image-Only, Text-Only, Late Fusion, Concatenation Ablation).
7. Generates 7 machine-readable JSON artifacts and 8 scientific visualizations under evidence/processed/multimodal/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from backend.app.multimodal.image_selector import ImageModelSelector
from backend.app.multimodal.modality_discovery import ModalityDiscoveryEngine
from backend.app.multimodal.multimodal_executor import MultimodalExecutor
from backend.app.multimodal.multimodal_results_package import MultimodalResultsPackager
from backend.app.multimodal.text_selector import TextModelSelector

logger = logging.getLogger(__name__)


def generate_demonstration_data(
    num_samples: int = 150,
    base_dir: str = "data/interim/multimodal_demo",
) -> Dict[str, Any]:
    """
    Constructs a reproducible multimodal dataset (clinical tabular + imaging + pathology text).
    """
    out_dir = Path(base_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)
    pids = [f"HANCOCK_PT_{i:04d}" for i in range(num_samples)]

    # Generate synthetic ground-truth recurrence labels with realistic disease prevalence (~30%)
    labels = []
    image_paths = []
    text_records = []
    tabular_matrix = np.zeros((num_samples, 8), dtype=np.float32)

    cancer_types = ["Adenocarcinoma", "Squamous Cell Carcinoma", "Large Cell Carcinoma"]
    t_stages = ["T1a", "T1b", "T2a", "T2b", "T3"]

    for i, pid in enumerate(pids):
        # Latent risk score driving recurrence outcome
        age = float(rng.randint(45, 80))
        tumor_size = float(rng.uniform(1.2, 6.5))
        lymph_nodes = int(rng.poisson(1.5))
        grade = int(rng.choice([1, 2, 3], p=[0.25, 0.50, 0.25]))

        latent_risk = (
            0.02 * (age - 60)
            + 0.35 * (tumor_size - 3.0)
            + 0.40 * lymph_nodes
            + 0.50 * (grade - 2)
            + rng.randn() * 0.5
        )
        prob = 1.0 / (1.0 + np.exp(-latent_risk))
        y = 1 if prob > 0.45 else 0
        labels.append(y)

        # Tabular Features
        tabular_matrix[i, :] = [
            age / 100.0,
            tumor_size / 10.0,
            float(lymph_nodes) / 10.0,
            float(grade) / 3.0,
            float(rng.uniform(0.1, 1.0)),
            float(rng.uniform(0.1, 1.0)),
            float(rng.uniform(0.1, 1.0)),
            float(rng.uniform(0.1, 1.0)),
        ]

        # Clinical Text Narrative
        c_type = cancer_types[i % len(cancer_types)]
        t_st = t_stages[i % len(t_stages)]
        if y == 1:
            txt = (
                f"Surgical pathology for {pid}: High-grade {c_type}, pathological stage {t_st} with "
                f"microvascular invasion and {lymph_nodes} positive lymph nodes. "
                f"Post-resection adjuvant platinum-based chemotherapy administered. Elevated recurrence risk flagged."
            )
        else:
            txt = (
                f"Surgical pathology for {pid}: Well-differentiated {c_type}, pathological stage {t_st}. "
                f"Surgical margins negative with zero nodal metastasis ({lymph_nodes} nodes resected). "
                f"Patient completed standard adjuvant protocol with excellent functional recovery."
            )
        text_records.append(txt)

        # Synthetic Histopathology / Clinical Image Tile (128x128 RGB)
        img_path = img_dir / f"{pid}_scan.png"
        if y == 1:
            # High cellular density texture for recurrence positive
            img_arr = rng.randint(80, 240, (128, 128, 3), dtype=np.uint8)
        else:
            # Normal stroma pattern
            img_arr = rng.randint(20, 180, (128, 128, 3), dtype=np.uint8)

        Image.fromarray(img_arr).save(img_path)
        image_paths.append(str(img_path))

    return {
        "pids": pids,
        "labels": labels,
        "tabular_matrix": tabular_matrix,
        "image_paths": image_paths,
        "text_records": text_records,
    }


def run_demonstration():
    print("=================================================================", flush=True)
    print("STAGE: EVIDENCE-CONDITIONED MULTIMODAL EXECUTION DEMONSTRATION", flush=True)
    print("=================================================================", flush=True)

    # 1. Generate Multimodal Dataset
    print("\n[Step 1] Initializing multimodal cohort data...", flush=True)
    ds = generate_demonstration_data(num_samples=100)
    print(f"Generated {len(ds['pids'])} patient records (Tabular + Imaging + Pathology Text).", flush=True)

    # 2. Modality Discovery
    print("\n[Step 2] Executing Modality Discovery Engine...", flush=True)
    discovery_engine = ModalityDiscoveryEngine(output_dir="evidence/processed/multimodal")
    disc_res = discovery_engine.discover(
        tabular_data=[{"patient_id": pid, "recurrence": ds["labels"][i]} for i, pid in enumerate(ds["pids"])],
        image_data=ds["image_paths"],
        text_data=ds["text_records"],
        candidate_target="recurrence",
        candidate_id="patient_id",
    )
    print(f"Modality Discovery Status: {disc_res.status}", flush=True)
    print(f"Detected Modalities: {disc_res.detected_modalities}", flush=True)

    # 3. Evidence-Conditioned Model Selection
    print("\n[Step 3] Selecting Evidence-Conditioned Image & Text Models...", flush=True)
    img_selector = ImageModelSelector()
    txt_selector = TextModelSelector()

    sel_img = img_selector.select(task_type="binary_classification", compute_budget="LIGHT")
    sel_txt = txt_selector.select(task_type="binary_classification", domain_type="clinical", compute_budget="LIGHT")

    print(f"Selected Image Architecture: {sel_img['name']} ({sel_img['evidence_source']})", flush=True)
    print(f"Selected Text Architecture:  {sel_txt['name']} ({sel_txt['evidence_source']})", flush=True)

    # 4. Multimodal Execution & Baseline Benchmarking
    print("\n[Step 4] Running Multi-Seed Multimodal Training & Baseline Benchmarking...", flush=True)
    executor = MultimodalExecutor(seeds=[42, 100, 2026], compute_budget="LIGHT", epochs=5, learning_rate=0.02)

    results = executor.run_experiment(
        patient_ids=ds["pids"],
        labels=ds["labels"],
        tabular_matrix=ds["tabular_matrix"],
        image_paths=ds["image_paths"],
        raw_texts=ds["text_records"],
        active_modalities=["image", "text"],
        fusion_mechanism="cross_attention",
        embed_dim=128,
    )

    print("\n--- Empirical Benchmark Results (Mean ± Std over n=3 seeds) ---", flush=True)
    for name, m in results["summary_metrics"].items():
        print(f"  {name:25s}: ROC-AUC = {m['mean_roc_auc']:.4f} ± {m['std_roc_auc']:.4f} | Brier = {m['mean_brier_score']:.4f}", flush=True)

    # 5. Results Packaging & Visualizations
    print("\n[Step 5] Packaging Machine-Readable Results & Visualizations...", flush=True)
    packager = MultimodalResultsPackager(output_dir="evidence/processed/multimodal")
    saved_files = packager.package_results(results)

    print("Created artifacts:", flush=True)
    for k, v in saved_files.items():
        print(f"  - {k}: {v}", flush=True)

    print("\n[Step 6] Execution Status: MULTIMODAL_EXECUTION_COMPLETE", flush=True)
    return results


if __name__ == "__main__":
    run_demonstration()
