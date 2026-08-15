"""
Comprehensive Functionality & Integration Tests for Multimodal Pipeline Engine

Tests:
1. Modality Discovery Layer (tabular, image, text, multi-modal mappings, modality_detection.json)
2. Evidence-Conditioned Image Model Selector (ranking, compute tiers, provenance)
3. Evidence-Conditioned Text Model Selector (ranking, biomedical domain, provenance)
4. Image Preprocessing Engine (loading, corrupt handling, train-only augmentation isolation)
5. Text Preprocessing Engine (tokenization, padding, attention masks, train-only vocab fitting)
6. Neural Backbones (CNN, ViT, Biomedical Transformer, Tabular Dense Encoder)
7. Multimodal Fusion Mechanisms (Cross-Attention, Concatenation, Late Fusion, Gated Fusion)
8. Ensembling Mechanisms (Average Ensembling, Weighted Ensembling)
9. Multimodal Pipeline Construction & Forward/Backward Training Steps
10. 14 Multimodal Safety Gates (patient overlap, target leakage, temporal leakage, compute budget)
11. Multimodal Executor (multi-seed execution, baseline benchmarking, ablation studies)
12. Multimodal Results Packaging & Figure Generation
13. Historical Empirical Artifacts Immutability Verification
"""

import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from backend.app.multimodal.image_preprocessing import ImagePreprocessor
from backend.app.multimodal.image_selector import ImageModelSelector
from backend.app.multimodal.modality_discovery import ModalityDiscoveryEngine
from backend.app.multimodal.multimodal_executor import MultimodalExecutor, compute_binary_metrics
from backend.app.multimodal.multimodal_pipeline import MultimodalPipeline
from backend.app.multimodal.multimodal_results_package import MultimodalResultsPackager
from backend.app.multimodal.neural_components import (
    AverageEnsemble,
    BiomedicalTextTransformer,
    CNNBackbone,
    CrossAttentionFusion,
    FeatureConcatenationFusion,
    GatedMultimodalFusion,
    LateFusion,
    TabularDenseEncoder,
    VisionTransformerBackbone,
    WeightedEnsemble,
)
from backend.app.multimodal.safety_gates import MultimodalSafetyAuditor
from backend.app.multimodal.text_preprocessing import TextPreprocessor
from backend.app.multimodal.text_selector import TextModelSelector


@pytest.fixture
def temp_multimodal_dataset(tmp_path):
    """Creates a synthetic multi-modal dataset with tabular, image, and text records."""
    img_dir = tmp_path / "images"
    txt_dir = tmp_path / "texts"
    img_dir.mkdir()
    txt_dir.mkdir()

    pids = [f"patient_{i:03d}" for i in range(20)]
    labels = [1 if i % 2 == 0 else 0 for i in range(20)]

    image_paths = []
    text_records = []
    tabular_records = []

    for i, pid in enumerate(pids):
        # 1. Create Synthetic Image (64x64 RGB PNG)
        img_p = img_dir / f"{pid}_scan.png"
        img_arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img_arr).save(img_p)
        image_paths.append(str(img_p))

        # 2. Create Clinical Text
        txt_content = (
            f"Patient {pid} presented with clinical stage T{1 + (i % 3)} N{i % 2} M0 tumor. "
            f"Histopathology confirmed adenocarcinoma with clear margins. "
            f"Adjuvant chemotherapy initiated with good tolerance."
        )
        text_records.append(txt_content)

        # 3. Create Tabular Record
        tabular_records.append({
            "patient_id": pid,
            "age": 55 + i,
            "tumor_size": 2.5 + (i * 0.1),
            "grade": f"G{1 + (i % 3)}",
            "recurrence": labels[i],
        })

    return {
        "pids": pids,
        "labels": labels,
        "image_paths": image_paths,
        "text_records": text_records,
        "tabular_records": tabular_records,
        "tmp_path": tmp_path,
    }


def test_1_modality_discovery(temp_multimodal_dataset):
    ds = temp_multimodal_dataset
    out_dir = ds["tmp_path"] / "discovery_out"
    engine = ModalityDiscoveryEngine(output_dir=str(out_dir))

    res = engine.discover(
        tabular_data=ds["tabular_records"],
        image_data=ds["image_paths"],
        text_data=ds["text_records"],
        candidate_target="recurrence",
        candidate_id="patient_id",
    )

    assert res.status == "DISCOVERED"
    assert "tabular" in res.detected_modalities
    assert "image" in res.detected_modalities
    assert "text" in res.detected_modalities
    assert res.sample_count == 20
    assert res.target_field == "recurrence"
    assert res.identifier_field == "patient_id"

    # Check generated metadata file
    out_json = out_dir / "modality_detection.json"
    assert out_json.exists()
    with open(out_json, "r") as f:
        data = json.load(f)
    assert data["status"] == "DISCOVERED"


def test_2_image_model_evidence_selection():
    selector = ImageModelSelector()

    # Test Light Budget Selection
    light_res = selector.select(task_type="binary_classification", compute_budget="LIGHT")
    assert light_res["execution_status"] == "EXECUTABLE"
    assert light_res["compute_cost"] == "LIGHT"
    assert light_res["evidence_status"] == "EVIDENCE_BACKED"
    assert "PMID:" in light_res["evidence_source"]

    # Test Heavy Budget Model Ranking
    heavy_res = selector.select(task_type="binary_classification", compute_budget="HEAVY")
    assert heavy_res["execution_status"] == "EXECUTABLE"


def test_3_text_model_evidence_selection():
    selector = TextModelSelector()

    res = selector.select(task_type="binary_classification", domain_type="clinical", compute_budget="LIGHT")
    assert res["execution_status"] == "EXECUTABLE"
    assert res["selected_value"] in ["pubmedbert", "biobert", "tfidf_linear"]
    assert res["evidence_status"] == "EVIDENCE_BACKED"
    assert "PMID:" in res["evidence_source"]


def test_4_image_preprocessing_train_only_isolation(temp_multimodal_dataset):
    ds = temp_multimodal_dataset
    preprocessor = ImagePreprocessor(target_size=(32, 32), augment_train=True)

    # Split images into train and test
    train_imgs = ds["image_paths"][:15]
    test_imgs = ds["image_paths"][15:]

    # Fit only on train
    preprocessor.fit(train_imgs)
    assert preprocessor.is_fitted

    # Transform train (with augmentations enabled)
    train_tensors = preprocessor.transform(train_imgs, is_training=True)
    assert train_tensors.shape == (15, 3, 32, 32)

    # Transform test (with augmentations strictly disabled)
    test_tensors = preprocessor.transform(test_imgs, is_training=False)
    assert test_tensors.shape == (5, 3, 32, 32)

    # Test missing/corrupt image handling
    corrupt_res = preprocessor.transform(["non_existent_file.png"], is_training=False)
    assert corrupt_res.shape == (1, 3, 32, 32)


def test_5_text_preprocessing_train_only_isolation(temp_multimodal_dataset):
    ds = temp_multimodal_dataset
    preprocessor = TextPreprocessor(max_seq_length=64, vocab_size=500, use_tfidf=True)

    train_txt = ds["text_records"][:15]
    test_txt = ds["text_records"][15:]

    # Fit strictly on train
    preprocessor.fit(train_txt)
    assert preprocessor.is_fitted
    assert len(preprocessor.vocab) > 10

    # Transform token IDs and attention masks
    input_ids, masks = preprocessor.transform(test_txt, is_training=False)
    assert input_ids.shape == (5, 64)
    assert masks.shape == (5, 64)

    # Transform TF-IDF
    tfidf_mat = preprocessor.transform_tfidf(test_txt)
    assert tfidf_mat.shape == (5, len(preprocessor.vocab))


def test_6_neural_backbones():
    # 1. CNN Backbone
    cnn = CNNBackbone(in_channels=3, embed_dim=64, seed=42)
    fake_imgs = np.random.randn(4, 3, 32, 32).astype(np.float32)
    cnn_out = cnn.forward(fake_imgs)
    assert cnn_out.shape == (4, 64)

    # 2. Vision Transformer Backbone
    vit = VisionTransformerBackbone(image_size=(32, 32), patch_size=8, embed_dim=64, seed=42)
    vit_out = vit.forward(fake_imgs)
    assert vit_out.shape == (4, 64)

    # 3. Biomedical Text Transformer
    txt_tf = BiomedicalTextTransformer(vocab_size=100, max_seq_len=32, embed_dim=64, seed=42)
    fake_toks = np.random.randint(0, 100, (4, 32))
    fake_mask = np.ones((4, 32), dtype=np.float32)
    txt_out = txt_tf.forward(fake_toks, attention_mask=fake_mask)
    assert txt_out.shape == (4, 64)

    # 4. Tabular Dense Encoder
    tab_enc = TabularDenseEncoder(in_features=10, embed_dim=64, seed=42)
    fake_tab = np.random.randn(4, 10).astype(np.float32)
    tab_out = tab_enc.forward(fake_tab)
    assert tab_out.shape == (4, 64)


def test_7_multimodal_fusion_mechanisms():
    feat_a = np.random.randn(4, 64).astype(np.float32)
    feat_b = np.random.randn(4, 64).astype(np.float32)

    # 1. Cross-Attention Fusion
    cross_attn = CrossAttentionFusion(dim_a=64, dim_b=64, out_dim=64, seed=42)
    fused_ca = cross_attn.forward(feat_a, feat_b)
    assert fused_ca.shape == (4, 64)

    # 2. Feature Concatenation Fusion
    concat_f = FeatureConcatenationFusion(in_dims=[64, 64], out_dim=64, seed=42)
    fused_cc = concat_f.forward([feat_a, feat_b])
    assert fused_cc.shape == (4, 64)

    # 3. Gated Multimodal Fusion
    gated_f = GatedMultimodalFusion(in_dims=[64, 64], out_dim=64, seed=42)
    fused_gt = gated_f.forward([feat_a, feat_b])
    assert fused_gt.shape == (4, 64)

    # 4. Late Fusion
    late_f = LateFusion(num_modalities=2)
    p_a = np.array([0.7, 0.2, 0.9, 0.4])
    p_b = np.array([0.8, 0.1, 0.85, 0.5])
    fused_lt = late_f.forward([p_a, p_b])
    assert fused_lt.shape == (4,)


def test_8_ensembling_mechanisms():
    p1 = np.array([0.8, 0.2, 0.9])
    p2 = np.array([0.6, 0.4, 0.7])

    # 1. Average Ensemble
    avg_ens = AverageEnsemble()
    p_avg = avg_ens.predict_proba([p1, p2])
    assert np.allclose(p_avg, np.array([0.7, 0.3, 0.8]))

    # 2. Weighted Ensemble
    wt_ens = WeightedEnsemble(validation_scores=[0.9, 0.7])
    p_wt = wt_ens.predict_proba([p1, p2])
    assert len(p_wt) == 3


def test_9_multimodal_pipeline_training_and_inference(temp_multimodal_dataset):
    ds = temp_multimodal_dataset
    pipeline = MultimodalPipeline(
        active_modalities=["image", "text"],
        fusion_mechanism="cross_attention",
        embed_dim=64,
        seed=42,
    )

    # Fit preprocessors
    pipeline.fit_preprocessors(image_paths=ds["image_paths"], raw_texts=ds["text_records"])

    # Forward pass
    probs, fused = pipeline.forward(image_paths=ds["image_paths"], raw_texts=ds["text_records"], is_training=False)
    assert len(probs) == 20
    assert fused.shape == (20, 64)

    # Train Step
    y_true = np.array(ds["labels"])
    loss = pipeline.train_step(None, ds["image_paths"], ds["text_records"], y_true, lr=0.01)
    assert loss > 0.0

    # Predict
    preds = pipeline.predict(None, ds["image_paths"], ds["text_records"])
    assert len(preds) == 20
    assert set(preds).issubset({0, 1})


def test_10_fourteen_safety_gates():
    auditor = MultimodalSafetyAuditor(compute_budget="LIGHT")

    report = auditor.audit_all(
        modalities=["image", "text"],
        train_pids=["p1", "p2", "p3"],
        val_pids=["p4"],
        test_pids=["p5", "p6"],
        train_features={},
        val_features={},
        test_features={},
        pipeline_config={"embed_dim": 256, "seeds": [42, 100, 2026]},
        image_meta={"evidence_source": "PMID: 41775771", "execution_status": "EXECUTABLE", "compute_cost": "LIGHT"},
        text_meta={"evidence_source": "PMID: 41826845", "execution_status": "EXECUTABLE", "compute_cost": "LIGHT"},
    )

    assert report["overall_status"] == "PASSED"
    assert report["passed_gates_count"] == 14
    assert report["total_gates_count"] == 14


def test_11_multimodal_executor_and_baselines(temp_multimodal_dataset):
    ds = temp_multimodal_dataset
    executor = MultimodalExecutor(seeds=[42, 100], compute_budget="LIGHT", epochs=3)

    results = executor.run_experiment(
        patient_ids=ds["pids"],
        labels=ds["labels"],
        image_paths=ds["image_paths"],
        raw_texts=ds["text_records"],
        active_modalities=["image", "text"],
        fusion_mechanism="cross_attention",
        embed_dim=64,
    )

    assert results["status"] == "COMPLETED"
    assert "multimodal_candidate" in results["summary_metrics"]
    assert "image_only_baseline" in results["summary_metrics"]
    assert "text_only_baseline" in results["summary_metrics"]
    assert "late_fusion_baseline" in results["summary_metrics"]
    assert "concat_fusion_ablation" in results["summary_metrics"]

    # Check metrics existence
    cand_m = results["summary_metrics"]["multimodal_candidate"]
    assert "mean_roc_auc" in cand_m
    assert "mean_brier_score" in cand_m


def test_12_results_packaging_and_figures(temp_multimodal_dataset, tmp_path):
    ds = temp_multimodal_dataset
    executor = MultimodalExecutor(seeds=[42], compute_budget="LIGHT", epochs=2)
    exp_res = executor.run_experiment(
        patient_ids=ds["pids"],
        labels=ds["labels"],
        image_paths=ds["image_paths"],
        raw_texts=ds["text_records"],
        active_modalities=["image", "text"],
        embed_dim=64,
    )

    out_dir = tmp_path / "mm_results"
    packager = MultimodalResultsPackager(output_dir=str(out_dir))
    saved = packager.package_results(exp_res)

    assert Path(saved["manifest"]).exists()
    assert Path(saved["pipeline"]).exists()
    assert Path(saved["results"]).exists()
    assert Path(saved["baselines"]).exists()
    assert Path(saved["provenance"]).exists()
    assert Path(saved["safety_audit"]).exists()
    assert Path(saved["summary"]).exists()

    # Check figures
    for i in range(1, 9):
        fig_files = list((out_dir / "figures").glob(f"fig{i}_*.png"))
        assert len(fig_files) == 1, f"Missing figure {i}"


def test_13_historical_empirical_artifacts_immutability():
    """Confirms that historical Stage 5B/5C/6A/6B/6H/6I empirical artifacts are untouched."""
    cand_path = Path("evidence/processed/stage5b_candidate_results.json")
    base_path = Path("evidence/processed/stage5b_baseline_results.json")
    if cand_path.exists():
        with open(cand_path, "r") as f:
            data = json.load(f)
        assert data["aggregated_test_metrics"]["roc_auc"]["mean"] == 0.9751

    if base_path.exists():
        with open(base_path, "r") as f:
            data = json.load(f)
        assert data["baseline_xgboost_default"]["aggregated_test_metrics"]["roc_auc"]["mean"] == 0.9704

    cleanup_path = Path("repository_cleanup_manifest.json")
    assert cleanup_path.exists()
