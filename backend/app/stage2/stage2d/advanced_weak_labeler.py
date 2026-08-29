"""
advanced_weak_labeler.py

Stage 2D — Context-Aware Weak-Supervision Label Generator

Generates high-precision weakly supervised BIO labels for training SciBERT
by combining:
  1. Expanded scientific methodology terminology (synonyms, abbreviations, aliases)
  2. Syntactic context analysis (distinguishing active methodology vs background citation vs future work)
  3. Section-aware weighting (Methods > Results > Abstract > Introduction)
  4. Per-span confidence calibration and noise-filtering thresholds.

Outputs are explicitly tagged as:
  - extraction_method: bootstrap_weak
  - supervision_type: WEAKLY_SUPERVISED
  - is_bootstrap: True
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend.app.stage2.models import ExtractionMethod, MechanismCategory, NEREntity
from backend.app.stage2.ner_entity_types import (
    BIO_LABELS, CANONICAL_EXAMPLES, ENTITY_TO_MECHANISM, ID2LABEL, LABEL2ID,
    NEREntityType,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Terminology & Abbreviation Lexicon
# ─────────────────────────────────────────────────────────────────────────────

EXTENDED_METHODOLOGY_LEXICON: Dict[NEREntityType, List[Dict[str, Any]]] = {
    NEREntityType.MODEL_ARCH: [
        {"term": "XGBoost", "aliases": ["xgboost", "xgb", "extreme gradient boosting", "gradient boosting trees"], "weight": 0.95},
        {"term": "Random Forest", "aliases": ["random forest", "random forests", "rf", "random forest classifier"], "weight": 0.95},
        {"term": "ResNet-18", "aliases": ["resnet-18", "resnet18", "18-layer resnet", "residual network 18"], "weight": 0.95},
        {"term": "ResNet-50", "aliases": ["resnet-50", "resnet50", "50-layer resnet"], "weight": 0.95},
        {"term": "EfficientNet-B0", "aliases": ["efficientnet-b0", "efficientnet", "efficientnet_b0"], "weight": 0.95},
        {"term": "Vision Transformer", "aliases": ["vision transformer", "vit", "vit-small", "vit-base"], "weight": 0.95},
        {"term": "Swin Transformer", "aliases": ["swin transformer", "swin-t", "swin-b"], "weight": 0.95},
        {"term": "Logistic Regression", "aliases": ["logistic regression", "multinomial logistic regression", "l2-regularized logistic regression"], "weight": 0.95},
        {"term": "Support Vector Machine", "aliases": ["support vector machine", "svm", "svc", "support vector classifier"], "weight": 0.95},
        {"term": "Multilayer Perceptron", "aliases": ["multilayer perceptron", "mlp", "feedforward neural network", "deep neural network"], "weight": 0.90},
        {"term": "PubMedBERT", "aliases": ["pubmedbert", "biomedical bert", "pubmed-bert"], "weight": 0.95},
        {"term": "ClinicalBERT", "aliases": ["clinicalbert", "clinical-bert"], "weight": 0.95},
    ],
    NEREntityType.PREPROCESSING: [
        {"term": "MICE Imputation", "aliases": ["mice", "multiple imputation by chained equations", "iterative imputer", "mice imputation"], "weight": 0.95},
        {"term": "Standard Scaling", "aliases": ["standard scaling", "standard scaler", "z-score normalization", "standardization"], "weight": 0.95},
        {"term": "Min-Max Normalization", "aliases": ["min-max normalization", "min-max scaling", "minmax scaler"], "weight": 0.95},
        {"term": "One-Hot Encoding", "aliases": ["one-hot encoding", "one-hot", "ohe", "dummy encoding"], "weight": 0.95},
        {"term": "Median Imputation", "aliases": ["median imputation", "simple imputer median", "missing value imputation"], "weight": 0.90},
    ],
    NEREntityType.SAMPLING: [
        {"term": "SMOTE", "aliases": ["smote", "synthetic minority oversampling technique", "smote oversampling"], "weight": 0.95},
        {"term": "ADASYN", "aliases": ["adasyn", "adaptive synthetic sampling", "adasyn oversampling"], "weight": 0.95},
        {"term": "Random Oversampling", "aliases": ["random oversampling", "ros", "naive oversampling"], "weight": 0.90},
        {"term": "Random Undersampling", "aliases": ["random undersampling", "rus"], "weight": 0.90},
        {"term": "Stratified K-Fold", "aliases": ["stratified k-fold", "stratified cross-validation", "stratified split"], "weight": 0.95},
    ],
    NEREntityType.FEATURE_REPR: [
        {"term": "Principal Component Analysis", "aliases": ["principal component analysis", "pca", "pca feature extraction"], "weight": 0.95},
        {"term": "t-SNE", "aliases": ["t-sne", "t-distributed stochastic neighbor embedding"], "weight": 0.90},
        {"term": "Word Embeddings", "aliases": ["word embeddings", "word2vec", "fasttext embeddings"], "weight": 0.90},
        {"term": "Convolutional Features", "aliases": ["convolutional features", "cnn embeddings", "latent representations"], "weight": 0.90},
    ],
    NEREntityType.FUSION: [
        {"term": "Late Fusion", "aliases": ["late fusion", "decision-level fusion", "feature concatenation fusion", "post-fusion"], "weight": 0.95},
        {"term": "Cross-Attention", "aliases": ["cross-attention", "cross attention", "cross-modal attention", "inter-modality attention"], "weight": 0.95},
        {"term": "Early Fusion", "aliases": ["early fusion", "input-level fusion", "raw feature concatenation"], "weight": 0.90},
        {"term": "Gated Fusion", "aliases": ["gated fusion", "gated multimodal fusion", "gating network"], "weight": 0.95},
    ],
    NEREntityType.LOSS: [
        {"term": "Binary Cross-Entropy", "aliases": ["binary cross-entropy", "binary cross entropy", "bce loss", "bce", "log loss"], "weight": 0.95},
        {"term": "Cross-Entropy Loss", "aliases": ["cross-entropy loss", "cross entropy loss", "categorical cross-entropy"], "weight": 0.95},
        {"term": "Focal Loss", "aliases": ["focal loss", "focal loss objective"], "weight": 0.95},
        {"term": "Dice Loss", "aliases": ["dice loss", "soft dice loss"], "weight": 0.90},
        {"term": "Mean Squared Error", "aliases": ["mean squared error", "mse loss", "l2 loss"], "weight": 0.90},
    ],
    NEREntityType.OPTIMIZATION: [
        {"term": "AdamW", "aliases": ["adamw", "adam with decoupled weight decay", "adamw optimizer"], "weight": 0.95},
        {"term": "Adam", "aliases": ["adam optimizer", "adam", "adaptive moment estimation"], "weight": 0.95},
        {"term": "SGD with Momentum", "aliases": ["sgd with momentum", "stochastic gradient descent", "sgd", "momentum sgd"], "weight": 0.95},
        {"term": "Cosine Annealing", "aliases": ["cosine annealing", "cosine learning rate schedule", "cosine decay"], "weight": 0.90},
    ],
    NEREntityType.REGULARIZATION: [
        {"term": "Dropout", "aliases": ["dropout", "spatial dropout", "mc dropout"], "weight": 0.95},
        {"term": "L2 Regularization", "aliases": ["l2 regularization", "weight decay", "ridge penalty"], "weight": 0.95},
        {"term": "L1 Regularization", "aliases": ["l1 regularization", "lasso penalty"], "weight": 0.95},
        {"term": "Early Stopping", "aliases": ["early stopping", "patience-based stopping"], "weight": 0.95},
        {"term": "Batch Normalization", "aliases": ["batch normalization", "batchnorm", "bn layers"], "weight": 0.90},
    ],
    NEREntityType.EVALUATION: [
        {"term": "ROC-AUC", "aliases": ["roc-auc", "auroc", "area under the roc curve", "c-statistic", "c-index"], "weight": 0.95},
        {"term": "PR-AUC", "aliases": ["pr-auc", "auprc", "average precision", "area under precision-recall"], "weight": 0.95},
        {"term": "F1-Score", "aliases": ["f1-score", "f1 score", "macro f1", "micro f1"], "weight": 0.95},
        {"term": "Brier Score", "aliases": ["brier score", "calibration brier score"], "weight": 0.95},
        {"term": "Expected Calibration Error", "aliases": ["expected calibration error", "ece"], "weight": 0.90},
    ],
    NEREntityType.DATASET: [
        {"term": "TCGA", "aliases": ["tcga", "the cancer genome atlas", "tcga cohort"], "weight": 0.95},
        {"term": "MIMIC-III", "aliases": ["mimic-iii", "mimic database", "mimic-iv"], "weight": 0.95},
        {"term": "Hancock Cohort", "aliases": ["hancock", "hancock cohort", "hancock clinical trial"], "weight": 0.95},
        {"term": "CheXpert", "aliases": ["chexpert", "chexpert dataset"], "weight": 0.95},
    ],
    NEREntityType.HYPERPARAMETER: [
        {"term": "Learning Rate", "aliases": ["learning rate", "initial learning rate", "lr"], "weight": 0.90},
        {"term": "Batch Size", "aliases": ["batch size", "mini-batch size"], "weight": 0.90},
        {"term": "Number of Epochs", "aliases": ["epochs", "training epochs", "number of epochs"], "weight": 0.90},
        {"term": "Embedding Dimension", "aliases": ["embedding dimension", "hidden size", "feature dimension"], "weight": 0.90},
    ],
}


# Syntactic Usage Patterns
_ACTIVE_USAGE_VERBS = re.compile(
    r"\b(?:we\s+(?:use|used|apply|applied|employ|employed|implement|implemented|adopt|adopted|train|trained|evaluated|developed)"
    r"|(?:our|the proposed)\s+(?:model|method|framework|approach|network|architecture|pipeline)"
    r"|was\s+(?:trained|applied|used|implemented|optimized|evaluated)\s+(?:with|using|via|on)"
    r"|using\s+[a-z\s]{0,25}?(?=\bxgb\b|\bresnet\b|\bsmote\b|\bmice\b|\bbert\b|\badamw\b|\bbce\b|\broc-auc\b))\b",
    re.IGNORECASE,
)

_BACKGROUND_CITATION_VERBS = re.compile(
    r"\b(?:prior|previous|earlier|other)\s+(?:work|studies|authors|literature|methods)"
    r"|(?:et\s+al\.|demonstrated\s+by|proposed\s+by|introduced\s+by|reported\s+in)\b",
    re.IGNORECASE,
)

_FUTURE_WORK_VERBS = re.compile(
    r"\b(?:future\s+work|could\s+be|may\s+be\s+extended|planned\s+for|promising\s+direction)\b",
    re.IGNORECASE,
)


class AdvancedWeakLabeler:
    """
    Context-aware weak-supervision generator for Stage 2D SciBERT fine-tuning.
    """

    def __init__(self):
        self._compiled_patterns: List[Tuple[re.Pattern, NEREntityType, str, float]] = []
        for entity_type, term_entries in EXTENDED_METHODOLOGY_LEXICON.items():
            for entry in term_entries:
                canonical = entry["term"]
                base_w = entry.get("weight", 0.90)
                for alias in entry["aliases"]:
                    pattern = re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE)
                    self._compiled_patterns.append((pattern, entity_type, canonical, base_w))

        # Sort patterns by alias length descending so multi-word phrases match first
        self._compiled_patterns.sort(key=lambda p: len(p[2]), reverse=True)

    def extract_weak_labels(
        self,
        text: str,
        paper_id: str,
        section_name: Optional[str] = None,
        pmid: Optional[str] = None,
        doi: Optional[str] = None,
    ) -> List[NEREntity]:
        """
        Extracts weakly supervised NER entities with context-aware confidence calibration.
        """
        if not text or not text.strip():
            return []

        section_weight = self._get_section_weight(section_name)
        entities: List[NEREntity] = []
        matched_ranges: List[Tuple[int, int]] = []

        # Split into sentences to check local syntactic context
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sent_offset = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if sent_len < 5:
                sent_offset += sent_len + 1
                continue

            context_multiplier, intent_tag = self._analyze_sentence_intent(sentence)

            for pattern, entity_type, canonical, base_w in self._compiled_patterns:
                for match in pattern.finditer(sentence):
                    start_local, end_local = match.start(), match.end()
                    start_global = sent_offset + start_local
                    end_global = sent_offset + end_local

                    # Avoid overlapping spans
                    if any(not (end_global <= ms or start_global >= me) for ms, me in matched_ranges):
                        continue

                    matched_ranges.append((start_global, end_global))
                    span_text = sentence[start_local:end_local]

                    # Calibrate confidence score
                    calibrated_conf = round(base_w * section_weight * context_multiplier, 4)
                    calibrated_conf = min(0.98, max(0.40, calibrated_conf))

                    conf_level = "HIGH" if calibrated_conf >= 0.80 else ("MEDIUM" if calibrated_conf >= 0.60 else "LOW")
                    review = calibrated_conf < 0.60

                    mech_cat = ENTITY_TO_MECHANISM.get(entity_type, MechanismCategory.unmapped)

                    entities.append(NEREntity(
                        entity_id=str(uuid.uuid4()),
                        text=span_text,
                        entity_type=entity_type.value,
                        mechanism_category=mech_cat.value,
                        start_char=start_global,
                        end_char=end_global,
                        source_text=sentence,
                        source_section=section_name or "abstract",
                        source_paper_id=paper_id,
                        source_pmid=pmid,
                        source_doi=doi,
                        confidence=calibrated_conf,
                        confidence_level=conf_level,
                        review_flag=review,
                        extraction_method=ExtractionMethod.bootstrap_weak,
                        model_version="stage2d_advanced_weak_supervision_v2.0",
                        bio_tag=f"B-{entity_type.value}",
                        confidence_status="unresolved" if review else "explicit",
                        is_bootstrap=True,
                    ))

            sent_offset += sent_len + 1

        return entities

    def _get_section_weight(self, section: Optional[str]) -> float:
        if not section:
            return 0.80
        s = section.lower()
        if any(k in s for k in ["method", "material", "experiment", "model", "training", "implementation", "data", "preprocess"]):
            return 1.00
        elif any(k in s for k in ["result", "finding", "performance", "ablation"]):
            return 0.90
        elif any(k in s for k in ["abstract", "summary"]):
            return 0.80
        elif any(k in s for k in ["intro", "background", "related", "discussion", "limit"]):
            return 0.50
        return 0.70

    def _analyze_sentence_intent(self, sentence: str) -> Tuple[float, str]:
        """Classifies sentence into ACTIVE_METHOD (1.0), BACKGROUND (0.55), or FUTURE_WORK (0.45)."""
        if _ACTIVE_USAGE_VERBS.search(sentence):
            return 1.00, "ACTIVE_METHOD"
        if _BACKGROUND_CITATION_VERBS.search(sentence):
            return 0.55, "BACKGROUND_CITATION"
        if _FUTURE_WORK_VERBS.search(sentence):
            return 0.45, "FUTURE_WORK"
        return 0.75, "NEUTRAL_MENTION"
