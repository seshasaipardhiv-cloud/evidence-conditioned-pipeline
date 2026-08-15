"""
Tests for Automatic Fusion Selection (Stage 10 - Objective E)
Verifies:
1. Evidence-conditioned fusion ranking
2. Cross-Attention forward pass and dimensions
3. Feature Concatenation forward pass and dimensions
4. Gated Multimodal Fusion forward pass and dimensions
5. Late Fusion probability blending
"""

import numpy as np
import pytest
from backend.app.multimodal.fusion_selector import FusionSelector
from backend.app.multimodal.neural_components import (
    CrossAttentionFusion,
    FeatureConcatenationFusion,
    GatedMultimodalFusion,
    LateFusion,
)


def test_fusion_selection_and_execution():
    selector = FusionSelector()

    # Ranking check for bimodal image + text
    sel_img_txt = selector.select(active_modalities=["image", "text"], compute_budget="LIGHT")
    assert sel_img_txt["selected_value"] == "cross_attention"
    assert "PMID:" in sel_img_txt["evidence_source"]

    # Ranking check for trimodal tabular + image + text
    sel_tri = selector.select(active_modalities=["tabular", "image", "text"], compute_budget="LIGHT")
    assert sel_tri["selected_value"] in ["gated_fusion", "feature_concatenation"]

    # Forward Passes
    feat_a = np.random.randn(4, 64).astype(np.float32)
    feat_b = np.random.randn(4, 64).astype(np.float32)
    feat_c = np.random.randn(4, 64).astype(np.float32)

    # 1. Cross-Attention
    ca = CrossAttentionFusion(dim_a=64, dim_b=64, out_dim=64, seed=42)
    out_ca = ca.forward(feat_a, feat_b)
    assert out_ca.shape == (4, 64)

    # 2. Feature Concatenation
    fc = FeatureConcatenationFusion(in_dims=[64, 64, 64], out_dim=64, seed=42)
    out_fc = fc.forward([feat_a, feat_b, feat_c])
    assert out_fc.shape == (4, 64)

    # 3. Gated Fusion
    gf = GatedMultimodalFusion(in_dims=[64, 64, 64], out_dim=64, seed=42)
    out_gf = gf.forward([feat_a, feat_b, feat_c])
    assert out_gf.shape == (4, 64)

    # 4. Late Fusion
    lf = LateFusion(num_modalities=2)
    out_lf = lf.forward([np.array([0.9, 0.1]), np.array([0.8, 0.2])])
    assert out_lf.shape == (2,)
