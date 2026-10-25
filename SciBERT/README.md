# SciBERT Scientific Methodology Named Entity Recognition (NER)

**Model:** `allenai/scibert_scivocab_uncased`  
**Supervision:** Weakly Supervised Context-Aware Token Classification  
**Training Corpus:** 100% Genuine Published Scientific Literature from PubMed (30 Research Papers)  
**Synthetic Data:** **0.0% (Zero Synthetic Sentences)**  

---

## 1. Overview & Purpose

In this project, SciBERT serves as a domain-specialized Named Entity Recognition (NER) token classifier. It scans published scientific paper abstracts to identify and score scientific methodology entities across 11 key machine learning and clinical categories:

1. `MODEL_ARCH`: Model architectures (e.g., *XGBoost, Random Forest, Deep Learning, ResNet, CNN, Swin Transformer*)
2. `PREPROCESSING`: Data cleaning and scaling (e.g., *MICE imputation, standard scaling, normalization*)
3. `SAMPLING`: Imbalance handlers (e.g., *SMOTE, ADASYN, stratified cross-validation*)
4. `FEATURE_REPR`: Feature representations (e.g., *radiomics, PCA, embeddings*)
5. `FUSION`: Multimodal fusion techniques (e.g., *multimodal data fusion, early fusion, late fusion, cross-attention*)
6. `LOSS`: Training objectives (e.g., *cross-entropy, focal loss, Brier score*)
7. `OPTIMIZATION`: Optimizers (e.g., *Adam, AdamW, SGD*)
8. `REGULARIZATION`: Regularizers (e.g., *dropout, weight decay, early stopping*)
9. `EVALUATION`: Metrics and validation protocols (e.g., *ROC-AUC, F1-score, accuracy, C-index, 5-fold CV*)
10. `DATASET`: Clinical cohort benchmarks (e.g., *TCGA, Hancock, MIMIC-III, SEER*)
11. `HYPERPARAMETER`: Architectural parameters (e.g., *learning rate, batch size, embedding dimension*)

The extracted entity spans and their model confidence scores directly inform the **Evidence Quality Score ($S_{quality}$)** in the project's authoritative multi-factor selection engine:
$$S_{final} = 0.45 S_{perf} + 0.35 S_{quality} + 0.20 S_{app}$$

---

## 2. Dataset Provenance & Statistics

The dataset was constructed exclusively from 30 published biomedical and clinical machine learning papers fetched from PubMed / CrossRef.

- **Total Source Papers:** 30
- **Total Published Abstract Sentences:** 321
- **Total Methodology Entity Mentions:** 183
- **Synthetic Data Usage:** **None** (100% genuine peer-reviewed literature)
- **Split Strategy:** Paper-level disjoint split (sentences from test papers never appear in train or validation).

| Split | Papers | Sentences | Entity Spans | File Location |
|:---|:---:|:---:|:---:|:---|
| **Train** (70%) | 21 | 223 | 108 | `SciBERT/data/train_ner_dataset.json` |
| **Validation** (15%) | 4 | 43 | 26 | `SciBERT/data/val_ner_dataset.json` |
| **Test** (15%) | 5 | 55 | 49 | `SciBERT/data/test_ner_dataset.json` |
| **Full Corpus** | 30 | 321 | 183 | `SciBERT/data/full_ner_dataset.json` |

---

## 3. Training Details

- **Base Encoder:** `allenai/scibert_scivocab_uncased` (12 layers, 768 hidden dimensions)
- **Head Architecture:** `torch.nn.Linear(768, 23)` (23 BIO labels)
- **Loss Function:** `NoiseRobustLoss` (Label-smoothed cross-entropy with $\epsilon = 0.05$ and `ignore_index = -100` for `[CLS]`, `[SEP]`, and padding tokens)
- **Optimizer:** `AdamW(lr=0.005, weight_decay=1e-3)`
- **Convergence:** Early stopping triggered at Epoch 7 (Best validation checkpoint restored from Epoch 2, validation loss: 0.4411)
- **Checkpoint Location:** `SciBERT/results/scibert_ner_head.pt` (SHA-256: `06824be12116...`)
- **Stage 2D Integrated Checkpoint:** `evidence/processed/stage2d/checkpoints/scibert_ner_head_seed42.pt`

---

## 4. Evaluation Results on Held-Out Real Literature

Evaluated on the held-out test split of 55 sentences (49 genuine methodology mentions) from 5 unseen research papers:

- **Token Classification Accuracy:** **97.30%**
- **Methodology Entity Micro F1:** **0.6341**
- **Methodology Entity Macro F1:** **0.6118**

### Per-Class Test Set Performance

| Entity Class | Precision | Recall | F1 Score | Support (Tokens) |
|:---|:---:|:---:|:---:|:---:|
| **B-MODEL_ARCH** | **0.8571** | **0.6667** | **0.7500** | 9 |
| **I-MODEL_ARCH** | **0.8571** | **0.6000** | **0.7059** | 10 |
| **B-EVALUATION** | **1.0000** | **0.4583** | **0.6286** | 24 |
| **I-EVALUATION** | **1.0000** | **0.9000** | **0.9474** | 20 |
| **B-FUSION** | **1.0000** | 0.0909 | 0.1667 | 11 |
| **I-FUSION** | **1.0000** | 0.1818 | 0.3077 | 11 |
| **O (Non-entity)** | 0.9736 | 0.9987 | 0.9860 | 1,586 |

---

## 5. Directory Structure

```
SciBERT/
├── README.md                          <- This documentation
├── data/
│   ├── dataset_statistics.json        <- Dataset composition, class counts, and split metadata
│   ├── full_ner_dataset.json          <- 321 sentences with annotated entity spans
│   ├── train_ner_dataset.json         <- 223 training sentences (21 disjoint papers)
│   ├── val_ner_dataset.json           <- 43 validation sentences (4 disjoint papers)
│   └── test_ner_dataset.json          <- 55 test sentences (5 disjoint papers)
├── code/
│   ├── prepare_dataset.py             <- Segments abstracts & generates BIO annotations
│   ├── train_scibert.py               <- PyTorch training loop with NoiseRobustLoss
│   └── evaluate_scibert.py            <- Token & span level evaluation harness
├── results/
│   ├── scibert_ner_head.pt            <- Fine-tuned PyTorch model weights
│   ├── training_manifest.json         <- Loss history, hyperparameters, and SHA-256 hash
│   └── evaluation_metrics.json        <- Complete precision, recall, and F1 metrics
└── plots/
    ├── training_loss_curve.png        <- Train vs validation loss across epochs
    └── per_class_f1_scores.png        <- Horizontal bar chart of test F1 per class
```

---

## 6. How to Reproduce

In the repository root, run:

```powershell
# 1. Regenerate real literature dataset
$env:PYTHONPATH="."; python SciBERT/code/prepare_dataset.py

# 2. Train SciBERT on real data
$env:PYTHONPATH="."; python SciBERT/code/train_scibert.py

# 3. Evaluate on held-out literature test set
$env:PYTHONPATH="."; python SciBERT/code/evaluate_scibert.py
```
