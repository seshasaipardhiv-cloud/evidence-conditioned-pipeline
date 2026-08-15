# Scientific Methodology: Evidence-Conditioned Pipeline Synthesis

## 1. Overview and Problem Formulation
Biomedical predictive modeling commonly suffers from reproducibility failure, arbitrary machine learning defaults masquerading as informed choices, and subtle target leakage across preprocessing boundaries. This work presents an end-to-end **Evidence-Conditioned Pipeline Synthesis** framework that bridges published biomedical literature to executable, safe, and reproducible machine learning pipelines.

## 2. End-to-End Synthesis Architecture
The synthesis framework operates across structured, gated evolutionary stages:
1. **Literature Retrieval & Ingestion (Stage 2A–2C)**: Querying peer-reviewed literature across PubMed/PMC for domain-relevant mechanisms in cancer recurrence risk prediction.
2. **Evidence Extraction & Normalization (Stage 2D–2E)**: Extracting structured claims, grounding them in a standardized taxonomy, and verifying cryptographic provenance.
3. **Evidence Authenticity & Provenance Audit (Stage 2F-1)**: Validating source text sentences, author metadata, and PubMed IDs.
4. **Evidence Sufficiency Audit (Stage 2F-2)**: Auditing whether candidate mechanisms have empirical backing for the target task and tabular modality.
5. **Controlled Taxonomy Extension (Stage 2E-1/2F-3)**: Safely incorporating domain primitives without unconstrained taxonomy expansion.
6. **Implementation Primitive Audit (Stage 2F-4)**: Distinguishing evidence-backed mechanisms from unsupported slots.
7. **Explicit Configuration Boundary (Stage 3.4–3.6)**: Gating unresolved primitives and requiring human-controlled explicit project configuration rather than silent library defaults.
8. **Pipeline Materialization & Verification Gates (Stage 4)**: 10 independent verification gates ensuring zero data leakage, target isolation, and executable class mapping.
9. **Controlled Experimental Contract (Stage 5A)**: Freezing dataset splits, random seeds, and compute budgets into an immutable contract.
10. **Controlled Execution (Stage 5B)**: Deterministic model training and evaluation across isolated folds.
11. **Statistical Validation & Ablation Analysis (Stage 5C)**: Quantifying baseline margins, component contributions, and calibration without fabricated significance.
12. **Final Evidence Packaging & Manuscript Verification (Stage 5D–6C)**: Establishing conservative claim boundaries and complete research audit trails.

## 3. Strict Provenance Segregation
Every pipeline primitive is explicitly classified into one of two distinct categories:

### A. EVIDENCE_BACKED Primitives (Literature Grounded)
- **`feature_representation`**: `clinical_tabular_representation` (PMID: 42487970; grounded in structured clinical tabular predictors).
- **`modality_fusion`**: `cross_attention` (Literature-backed multimodal fusion mechanism).
- **`ensembling`**: `average_ensembling` (Literature-backed ensemble combination).
- **`missing_value_handling`**: `MissForest / MICE` (PMID: 41826845; iterative multivariate imputation).
- **`base_learner`**: `XGBoost` (PMID: 41775771; regularized gradient tree boosting).
- **`imbalance_handling`**: `SMOTE` (PMID: 41006422; synthetic minority over-sampling technique).

### B. EXPLICITLY_CONFIGURED Primitives (Human Project Gated)
- **`categorical_encoding`**: `one_hot_encoding` (Explicitly supplied in `experiment_config.json`; barred from being claimed as literature evidence).
- **`loss_function`**: `binary_logistic` (Explicitly supplied in `experiment_config.json`; barred from being claimed as literature evidence).

## 4. Methodological Invariant
Under no circumstances are explicitly configured primitives labeled as literature evidence, nor are unresolved pipeline slots filled with silent machine learning library defaults.
