"""
ner_entity_types.py

Stage 2C — Entity Ontology for Transformer-Based NER Extraction

Defines 11 research-methodology entity types extracted by the SciBERT NER
pipeline.  Each entity type maps to a MechanismCategory for downstream
compatibility with the existing evidence graph.

Entity Ontology (adapted for research-methodology/evidence-extraction):
  - Inspired by Shetty et al. biomedical NER pipeline concept
  - Entity types are specific to ML research papers, not generic clinical NER

BIO Tagging Schema:
  B-<TYPE>  — Beginning of an entity span
  I-<TYPE>  — Inside (continuation) of an entity span
  O         — Outside (no entity)

Confidence Thresholds:
  HIGH_CONFIDENCE  : score >= 0.80  → accepted, no review flag
  MEDIUM_CONFIDENCE: score >= 0.60  → accepted, review_flag = False
  LOW_CONFIDENCE   : score <  0.60  → accepted but review_flag = True, marked UNCERTAIN

Scientific rule:
  Entities below LOW_CONFIDENCE_THRESHOLD are retained in output but
  explicitly flagged with review_flag=True and confidence_status=unresolved.
  They are NEVER silently upgraded to regex extractions.
"""

from __future__ import annotations
from enum import Enum
from typing import Dict, List, Optional

from backend.app.stage2.models import MechanismCategory


# ─────────────────────────────────────────────────────────────────────────────
# Entity type enum
# ─────────────────────────────────────────────────────────────────────────────

class NEREntityType(str, Enum):
    """11-class entity ontology for research-methodology NER."""
    MODEL_ARCH    = "MODEL_ARCH"       # e.g. ResNet-18, XGBoost, LSTM, Transformer
    PREPROCESSING = "PREPROCESSING"   # e.g. normalization, one-hot encoding, MICE
    SAMPLING      = "SAMPLING"         # e.g. SMOTE, oversampling, stratified split
    FEATURE_REPR  = "FEATURE_REPR"     # e.g. PCA, embeddings, HOG features
    FUSION        = "FUSION"           # e.g. late fusion, cross-attention, concatenation
    LOSS          = "LOSS"             # e.g. binary cross-entropy, focal loss, MSE
    OPTIMIZATION  = "OPTIMIZATION"    # e.g. Adam, SGD, learning rate schedule
    REGULARIZATION = "REGULARIZATION" # e.g. dropout, L2 weight decay, early stopping
    EVALUATION    = "EVALUATION"       # e.g. ROC-AUC, F1-score, AUROC, C-index
    DATASET       = "DATASET"          # e.g. TCGA, MIMIC-III, ImageNet
    HYPERPARAMETER = "HYPERPARAMETER"  # e.g. batch size 32, learning rate 1e-4
    O             = "O"               # Outside (no entity) — used in BIO schema only


# ─────────────────────────────────────────────────────────────────────────────
# BIO label inventory
# ─────────────────────────────────────────────────────────────────────────────

def _build_bio_labels() -> List[str]:
    """
    Build ordered list of all BIO labels.
    Index 0 = 'O' (outside).  B/I pairs follow alphabetically.
    This ordering is deterministic and reproducible across runs.
    """
    labels = ["O"]
    for entity_type in NEREntityType:
        if entity_type == NEREntityType.O:
            continue
        labels.append(f"B-{entity_type.value}")
        labels.append(f"I-{entity_type.value}")
    return labels


BIO_LABELS: List[str] = _build_bio_labels()
NUM_LABELS: int = len(BIO_LABELS)
LABEL2ID: Dict[str, int] = {label: idx for idx, label in enumerate(BIO_LABELS)}
ID2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Entity → MechanismCategory mapping
# ─────────────────────────────────────────────────────────────────────────────

ENTITY_TO_MECHANISM: Dict[NEREntityType, MechanismCategory] = {
    NEREntityType.MODEL_ARCH:     MechanismCategory.representation,
    NEREntityType.PREPROCESSING:  MechanismCategory.preprocessing,
    NEREntityType.SAMPLING:       MechanismCategory.sampling,
    NEREntityType.FEATURE_REPR:   MechanismCategory.representation,
    NEREntityType.FUSION:         MechanismCategory.fusion,
    NEREntityType.LOSS:           MechanismCategory.loss,
    NEREntityType.OPTIMIZATION:   MechanismCategory.regularization,  # closest category
    NEREntityType.REGULARIZATION: MechanismCategory.regularization,
    NEREntityType.EVALUATION:     MechanismCategory.unmapped,         # no direct category
    NEREntityType.DATASET:        MechanismCategory.unmapped,         # no direct category
    NEREntityType.HYPERPARAMETER: MechanismCategory.unmapped,         # no direct category
    NEREntityType.O:              MechanismCategory.unmapped,
}


# ─────────────────────────────────────────────────────────────────────────────
# Confidence thresholds
# ─────────────────────────────────────────────────────────────────────────────

HIGH_CONFIDENCE_THRESHOLD: float = 0.80
LOW_CONFIDENCE_THRESHOLD: float = 0.60


def get_confidence_level(score: float) -> str:
    """Return human-readable confidence level for a given score."""
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return "HIGH"
    elif score >= LOW_CONFIDENCE_THRESHOLD:
        return "MEDIUM"
    else:
        return "LOW"


def requires_review(score: float) -> bool:
    """Return True if an entity requires human review (low confidence)."""
    return score < LOW_CONFIDENCE_THRESHOLD


# ─────────────────────────────────────────────────────────────────────────────
# Canonical examples for bootstrapping and documentation
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_EXAMPLES: Dict[NEREntityType, List[str]] = {
    NEREntityType.MODEL_ARCH: [
        "ResNet-18", "ResNet-50", "LSTM", "Transformer", "XGBoost",
        "Random Forest", "SVM", "Logistic Regression", "CNN", "autoencoder",
        "BERT", "BioBERT", "SciBERT", "MLP", "GRU", "ResNet",
        "VGG", "EfficientNet", "DenseNet",
    ],
    NEREntityType.PREPROCESSING: [
        "normalization", "standardization", "one-hot encoding",
        "missing value imputation", "MICE", "mean imputation",
        "median imputation", "min-max scaling", "z-score normalization",
        "log transformation", "label encoding", "winsorization",
        "data cleaning", "feature scaling",
    ],
    NEREntityType.SAMPLING: [
        "SMOTE", "oversampling", "undersampling", "stratified sampling",
        "random oversampling", "ADASYN", "class-balanced sampling",
        "weighted sampling", "bootstrap sampling",
    ],
    NEREntityType.FEATURE_REPR: [
        "PCA", "principal component analysis", "t-SNE", "UMAP",
        "word embeddings", "word2vec", "GloVe", "TF-IDF",
        "bag of words", "feature extraction", "HOG", "SIFT",
        "spectral features", "attention weights",
    ],
    NEREntityType.FUSION: [
        "late fusion", "early fusion", "intermediate fusion",
        "cross-attention", "concatenation", "weighted average",
        "gated fusion", "joint embedding", "feature-level fusion",
        "decision-level fusion",
    ],
    NEREntityType.LOSS: [
        "cross-entropy", "cross entropy", "binary cross-entropy",
        "binary cross entropy", "focal loss", "mean squared error",
        "MSE", "MAE", "mean absolute error", "hinge loss",
        "dice loss", "Kullback-Leibler divergence", "KL divergence",
        "contrastive loss", "triplet loss",
    ],
    NEREntityType.OPTIMIZATION: [
        "Adam", "SGD", "stochastic gradient descent", "AdamW",
        "RMSprop", "learning rate schedule", "cosine annealing",
        "warm-up schedule", "weight decay schedule",
        "gradient clipping", "learning rate decay",
    ],
    NEREntityType.REGULARIZATION: [
        "dropout", "L1 regularization", "L2 regularization",
        "weight decay", "early stopping", "batch normalization",
        "layer normalization", "data augmentation",
        "label smoothing", "gradient penalty",
    ],
    NEREntityType.EVALUATION: [
        "ROC-AUC", "AUROC", "AUC", "F1-score", "F1 score",
        "C-index", "concordance index", "accuracy", "precision",
        "recall", "sensitivity", "specificity", "AUPRC", "PR-AUC",
        "Brier score", "log-loss", "Matthews correlation coefficient",
        "MCC", "balanced accuracy",
    ],
    NEREntityType.DATASET: [
        "TCGA", "MIMIC-III", "MIMIC-IV", "ImageNet", "CheXpert",
        "NIH ChestX-ray", "MNIST", "CIFAR-10", "SEER", "UK Biobank",
        "Hancock", "ISIC", "PatchCamelyon", "LIDC-IDRI",
    ],
    NEREntityType.HYPERPARAMETER: [
        "batch size", "learning rate", "epochs", "hidden units",
        "dropout rate", "weight decay", "kernel size",
        "number of layers", "embedding dimension", "temperature",
    ],
}
