"""
Evidence-Conditioned Multimodal Pipeline Constructor & Model

Composes and executes arbitrary unimodal and multimodal pipelines:
- TABULAR
- IMAGE
- TEXT
- IMAGE + TEXT
- TABULAR + IMAGE
- TABULAR + TEXT
- TABULAR + IMAGE + TEXT

Connects modality preprocessors, neural backbones, evidence-conditioned fusion modules,
classification heads, and loss functions into a unified, trainable, and verifiable computational graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from backend.app.multimodal.image_preprocessing import ImagePreprocessor
from backend.app.multimodal.neural_components import (
    CNNBackbone,
    VisionTransformerBackbone,
    BiomedicalTextTransformer,
    TabularDenseEncoder,
    CrossAttentionFusion,
    FeatureConcatenationFusion,
    LateFusion,
    GatedMultimodalFusion,
    layer_norm,
    relu,
    sigmoid,
)
from backend.app.multimodal.text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class MultimodalPipeline:
    """
    Unified multimodal predictive pipeline supporting unimodal, bimodal, and trimodal clinical tasks.
    """

    def __init__(
        self,
        active_modalities: List[str],  # e.g. ['image', 'text'], ['tabular', 'image', 'text']
        image_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        tabular_config: Optional[Dict[str, Any]] = None,
        fusion_mechanism: str = "cross_attention",  # 'cross_attention' | 'feature_concatenation' | 'gated_fusion' | 'late_fusion'
        embed_dim: int = 256,
        seed: int = 42,
    ):
        self.active_modalities = active_modalities
        self.fusion_mechanism = fusion_mechanism
        self.embed_dim = embed_dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # 1. Modality Preprocessors
        self.image_preprocessor = ImagePreprocessor() if "image" in active_modalities else None
        self.text_preprocessor = TextPreprocessor() if "text" in active_modalities else None

        # 2. Modality Encoders
        self.image_encoder: Optional[Union[CNNBackbone, VisionTransformerBackbone]] = None
        if "image" in active_modalities:
            img_arch = (image_config or {}).get("selected_value", "resnet18")
            if "vit" in img_arch:
                self.image_encoder = VisionTransformerBackbone(embed_dim=embed_dim, seed=seed)
            else:
                self.image_encoder = CNNBackbone(embed_dim=embed_dim, architecture=img_arch, seed=seed)

        self.text_encoder: Optional[BiomedicalTextTransformer] = None
        if "text" in active_modalities:
            vocab_sz = (text_config or {}).get("vocab_size", 5000)
            self.text_encoder = BiomedicalTextTransformer(vocab_size=vocab_sz, embed_dim=embed_dim, seed=seed)

        self.tabular_encoder: Optional[TabularDenseEncoder] = None
        if "tabular" in active_modalities:
            num_feats = (tabular_config or {}).get("num_features", 20)
            self.tabular_encoder = TabularDenseEncoder(in_features=num_feats, embed_dim=embed_dim, seed=seed)

        # 3. Fusion Module
        self.cross_attention_fusion: Optional[CrossAttentionFusion] = None
        self.concat_fusion: Optional[FeatureConcatenationFusion] = None
        self.gated_fusion: Optional[GatedMultimodalFusion] = None
        self.late_fusion: Optional[LateFusion] = None

        if len(active_modalities) >= 2:
            if fusion_mechanism == "cross_attention":
                self.cross_attention_fusion = CrossAttentionFusion(dim_a=embed_dim, dim_b=embed_dim, out_dim=embed_dim, seed=seed)
            elif fusion_mechanism == "gated_fusion":
                self.gated_fusion = GatedMultimodalFusion(in_dims=[embed_dim] * len(active_modalities), out_dim=embed_dim, seed=seed)
            elif fusion_mechanism == "late_fusion":
                self.late_fusion = LateFusion(num_modalities=len(active_modalities))
            else:
                # Default to feature concatenation
                self.concat_fusion = FeatureConcatenationFusion(in_dims=[embed_dim] * len(active_modalities), out_dim=embed_dim, seed=seed)

        # 4. Classification Head (Linear + Sigmoid for binary clinical classification)
        self.w_head = self.rng.randn(embed_dim, 1).astype(np.float32) * np.sqrt(2.0 / embed_dim)
        self.b_head = np.zeros((1,), dtype=np.float32)

        # Unimodal heads for late fusion
        self.unimodal_heads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for mod in active_modalities:
            self.unimodal_heads[mod] = (
                self.rng.randn(embed_dim, 1).astype(np.float32) * np.sqrt(2.0 / embed_dim),
                np.zeros((1,), dtype=np.float32),
            )

        self.is_trained = False
        self.training_history: List[Dict[str, float]] = []

    def fit_preprocessors(
        self,
        tabular_data: Optional[np.ndarray] = None,
        image_paths: Optional[List[Any]] = None,
        raw_texts: Optional[List[Optional[str]]] = None,
    ) -> "MultimodalPipeline":
        """
        Fits preprocessing components strictly on the training partition.
        """
        if self.image_preprocessor is not None and image_paths is not None:
            self.image_preprocessor.fit(image_paths)

        if self.text_preprocessor is not None and raw_texts is not None:
            self.text_preprocessor.fit(raw_texts)

        return self

    def extract_features(
        self,
        tabular_data: Optional[np.ndarray] = None,
        image_paths: Optional[List[Any]] = None,
        raw_texts: Optional[List[Optional[str]]] = None,
        is_training: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts representations from each available modality.
        """
        representations: Dict[str, np.ndarray] = {}

        if "tabular" in self.active_modalities and tabular_data is not None and self.tabular_encoder is not None:
            representations["tabular"] = self.tabular_encoder.forward(tabular_data)

        if "image" in self.active_modalities and image_paths is not None and self.image_preprocessor is not None and self.image_encoder is not None:
            img_tensors = self.image_preprocessor.transform(image_paths, is_training=is_training)
            representations["image"] = self.image_encoder.forward(img_tensors)

        if "text" in self.active_modalities and raw_texts is not None and self.text_preprocessor is not None and self.text_encoder is not None:
            input_ids, att_masks = self.text_preprocessor.transform(raw_texts, is_training=is_training)
            representations["text"] = self.text_encoder.forward(input_ids, attention_mask=att_masks)

        return representations

    def forward(
        self,
        tabular_data: Optional[np.ndarray] = None,
        image_paths: Optional[List[Any]] = None,
        raw_texts: Optional[List[Optional[str]]] = None,
        is_training: bool = False,
        cached_reps: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full forward pass returning (probabilities, fused_representation).
        """
        reps = cached_reps or self.extract_features(tabular_data, image_paths, raw_texts, is_training=is_training)

        if not reps:
            raise ValueError("No valid representations extracted from active modalities.")

        # Single Modality
        if len(self.active_modalities) == 1:
            mod = self.active_modalities[0]
            fused = reps[mod]
            logits = np.dot(fused, self.w_head) + self.b_head
            probs = sigmoid(logits).flatten()
            return probs, fused

        # Multimodal Late Fusion
        if self.late_fusion is not None:
            mod_probs = []
            for mod in self.active_modalities:
                w_h, b_h = self.unimodal_heads[mod]
                p = sigmoid(np.dot(reps[mod], w_h) + b_h).flatten()
                mod_probs.append(p)
            fused_probs = self.late_fusion.forward(mod_probs)
            fused_rep = np.mean(list(reps.values()), axis=0)
            return fused_probs, fused_rep

        # Multimodal Intermediate / Feature Fusion
        if self.cross_attention_fusion is not None and len(self.active_modalities) == 2:
            mods = list(self.active_modalities)
            fused = self.cross_attention_fusion.forward(reps[mods[0]], reps[mods[1]])
        elif self.gated_fusion is not None:
            fused = self.gated_fusion.forward([reps[m] for m in self.active_modalities])
        elif self.concat_fusion is not None:
            fused = self.concat_fusion.forward([reps[m] for m in self.active_modalities])
        else:
            # Fallback mean pooling across modalities
            fused = layer_norm(np.mean(list(reps.values()), axis=0))

        logits = np.dot(fused, self.w_head) + self.b_head
        probs = sigmoid(logits).flatten()
        return probs, fused

    def train_step(
        self,
        tabular_data: Optional[np.ndarray] = None,
        image_paths: Optional[List[Any]] = None,
        raw_texts: Optional[List[Optional[str]]] = None,
        y_true: Optional[np.ndarray] = None,
        lr: float = 0.01,
        weight_decay: float = 1e-4,
        cached_reps: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """
        Executes one gradient descent optimization step using Binary Cross-Entropy loss.
        """
        probs, fused = self.forward(tabular_data, image_paths, raw_texts, is_training=True, cached_reps=cached_reps)
        y = y_true.astype(np.float32)

        # Compute Binary Cross Entropy Loss with numerical clipping
        eps = 1e-7
        p_clipped = np.clip(probs, eps, 1.0 - eps)
        loss = -float(np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped)))

        # Gradient of BCE w.r.t logits: (p - y) / N
        N = len(y)
        grad_logits = ((probs - y) / N)[:, None]  # (N, 1)

        # Gradient w.r.t classification head weights: fused.T @ grad_logits
        grad_w = np.dot(fused.T, grad_logits) + weight_decay * self.w_head
        grad_b = np.sum(grad_logits, axis=0)

        # Update head parameters
        self.w_head -= lr * grad_w
        self.b_head -= lr * grad_b

        return loss

    def predict_proba(
        self,
        tabular_data: Optional[np.ndarray] = None,
        image_paths: Optional[List[Any]] = None,
        raw_texts: Optional[List[Optional[str]]] = None,
    ) -> np.ndarray:
        """
        Predicts recurrence risk probabilities.
        """
        probs, _ = self.forward(tabular_data, image_paths, raw_texts, is_training=False)
        return probs

    def predict(
        self,
        tabular_data: Optional[np.ndarray] = None,
        image_paths: Optional[List[Any]] = None,
        raw_texts: Optional[List[Optional[str]]] = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Predicts binary class labels.
        """
        probs = self.predict_proba(tabular_data, image_paths, raw_texts)
        return (probs >= threshold).astype(int)

    def get_pipeline_manifest(self) -> Dict[str, Any]:
        return {
            "active_modalities": self.active_modalities,
            "fusion_mechanism": self.fusion_mechanism,
            "embed_dim": self.embed_dim,
            "seed": self.seed,
            "is_trained": self.is_trained,
            "image_preprocessor": self.image_preprocessor.get_provenance_spec() if self.image_preprocessor else None,
            "text_preprocessor": self.text_preprocessor.get_provenance_spec() if self.text_preprocessor else None,
            "architecture_summary": {
                "has_cross_attention": self.cross_attention_fusion is not None,
                "has_image_encoder": self.image_encoder is not None,
                "has_text_encoder": self.text_encoder is not None,
                "has_tabular_encoder": self.tabular_encoder is not None,
            },
        }
