# SciBERT Scientific Methodology Named Entity Recognition (NER)

**Model:** `allenai/scibert_scivocab_uncased`  
**Supervision:** Weakly Supervised Context-Aware Token Classification  
**Training Corpus:** 100% Genuine Published Scientific Literature from PubMed (215 Research Papers, 2,481 Sentences)  
**Extraction Corpus:** 100% Genuine Disjoint Scientific Literature from PubMed (105 Extra Research Papers)  
**Synthetic Data:** **0.0% (Zero Synthetic Sentences)**  

---

## 1. Overview & Purpose

In this project, SciBERT serves as a domain-specialized Named Entity Recognition (NER) token classifier. It scans published scientific paper abstracts to identify, extract, and score scientific methodology entities and quantitative training constraints across 11 key machine learning and clinical categories:

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
11. `HYPERPARAMETER`: Architectural parameters & explicit experimental constraints (e.g., *epochs, batch size, learning rate, patience, tree depth*)

The extracted entity spans and their model confidence scores directly inform the **Evidence Quality Score ($S_{quality}$)** in the project's authoritative multi-factor selection engine:
$$S_{final} = 0.45 S_{perf} + 0.35 S_{quality} + 0.20 S_{app}$$

---

## 2. Dataset Provenance & Corpus Disjointness

Two completely separate and strictly disjoint literature corpora were retrieved dynamically from NCBI PubMed:

1. **SciBERT Training Corpus (`SciBERT/data/scibert_training_papers.jsonl`)**: **215 distinct research papers** used for training and evaluating the SciBERT token classification head.
2. **Pipeline Extraction Corpus (`evidence/processed/dynamic_extraction_papers.jsonl`)**: **105 extra distinct research papers** used exclusively for downstream evidence extraction and pipeline synthesis.

- **Corpus Overlap:** **0 papers (0.0% overlap; strictly disjoint PMIDs)**
- **Total Published Abstract Sentences (Training Set):** 2,481
- **Total Methodology Entity Mentions (Training Set):** 996
- **Synthetic Data Usage:** **None** (100% genuine peer-reviewed literature)
- **Split Strategy:** Paper-level disjoint split (sentences from test papers never appear in train or validation).

| Split | Papers | Sentences | Entity Spans | File Location |
|:---|:---:|:---:|:---:|:---|
| **Train** (70%) | 150 | 1,757 | 687 | `SciBERT/data/train_ner_dataset.json` |
| **Validation** (15%) | 32 | 357 | 163 | `SciBERT/data/val_ner_dataset.json` |
| **Test** (15%) | 33 | 367 | 146 | `SciBERT/data/test_ner_dataset.json` |
| **Full Corpus** | 215 | 2,481 | 996 | `SciBERT/data/full_ner_dataset.json` |

---

## 3. Training Details

- **Base Encoder:** `allenai/scibert_scivocab_uncased` (12 layers, 768 hidden dimensions)
- **Head Architecture:** `torch.nn.Linear(768, 23)` (23 BIO labels)
- **Loss Function:** `NoiseRobustLoss` (Label-smoothed cross-entropy with $\epsilon = 0.05$ and `ignore_index = -100` for `[CLS]`, `[SEP]`, and padding tokens)
- **Optimizer:** `AdamW(lr=0.005, weight_decay=1e-3)`
- **Total Epochs Trained:** **50 Epochs** (Full 50-epoch trajectory completed)
- **Best Validation Loss:** **0.4147**
- **Checkpoint Location:** `SciBERT/results/scibert_ner_head.pt` (SHA-256: `956daab174d9...`)
- **Stage 2D Integrated Checkpoint:** `evidence/processed/stage2d/checkpoints/scibert_ner_head_seed42.pt`

---

## 4. Evaluation Results on Held-Out Real Literature

Evaluated on the held-out test split of **367 sentences (11,932 tokens)** from **33 unseen research papers**:

- **Token Classification Accuracy:** **99.05%**
- **Methodology Entity Micro F1:** **0.8114**
- **Methodology Entity Macro F1:** **0.6409**

### Per-Class Test Set Performance Highlights

| Entity Class | Precision | Recall | F1 Score | Support (Tokens) |
|:---|:---:|:---:|:---:|:---:|
| **B-MODEL_ARCH** | **0.8103** | **0.9400** | **0.8704** | 50 |
| **I-MODEL_ARCH** | **0.8776** | **0.8776** | **0.8776** | 49 |
| **B-FEATURE_REPR** | **1.0000** | **0.8182** | **0.9000** | 22 |
| **I-FEATURE_REPR** | **0.9429** | **0.9429** | **0.9429** | 35 |
| **B-EVALUATION** | **0.6277** | **0.9516** | **0.7564** | 62 |
| **I-EVALUATION** | **0.6792** | **0.8780** | **0.7660** | 41 |
| **B-DATASET** | **0.7500** | **1.0000** | **0.8571** | 3 |
| **O (Non-entity)** | **0.9966** | **0.9937** | **0.9952** | 11,649 |

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
