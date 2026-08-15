# 3. Proposed Methodology

The Evidence-Conditioned Pipeline Synthesis framework operates across 13 structured stages that bridge biomedical literature citations to verified, executable experimental pipelines.

```
Literature Ingestion (PubMed/PMC)
       ↓
Information Extraction & Normalization
       ↓
Authenticity & Provenance Audit
       ↓
Evidence Sufficiency Assessment
       ↓
Controlled Taxonomy Mapping & Extension
       ↓
Implementation Primitive Resolution
       ↓
Explicit Configuration Gate (Human Control)
       ↓
Pipeline Composition & Verification Gates
       ↓
Executable Materialization
       ↓
Frozen Experimental Execution Contract
       ↓
Controlled Execution & Statistical Audit
```

## 3.1–3.5 Evidence Ingestion, Extraction, and Provenance Auditing
Biomedical literature is queried across PubMed/PMC for target clinical tasks. Candidate claims are parsed, and source text sentences, author metadata, and PubMed IDs are recorded in an immutable provenance ledger. Cryptographic SHA-256 hashes ensure the corpus remains untampered throughout synthesis.

## 3.6–3.8 Controlled Taxonomy Extension and Primitive Resolution
Extracted mechanisms are mapped into a standardized taxonomy spanning feature representation, missing value handling, categorical encoding, class imbalance handling, learner selection, modality fusion, ensembling, and loss functions. The taxonomy is extended in a controlled, auditable manner to prevent unconstrained primitive proliferation.

## 3.9 Provenance Boundary: Evidence-Backed vs. Explicitly Configured
A core governance principle of the framework is the strict segregation of pipeline primitives:

### A. EVIDENCE_BACKED Primitives (Literature Grounded)
- **`feature_representation`**: `clinical_tabular_representation` (PMID: 42487970; structured clinical tabular predictors).
- **`modality_fusion`**: `cross_attention` (Literature-backed multimodal fusion mechanism; *dormant during unimodal tabular benchmark*).
- **`ensembling`**: `average_ensembling` (Literature-backed ensemble combination; *dormant during single-model XGBoost benchmark*).
- **`missing_value_handling`**: `MissForest / MICE` (PMID: 41826845; taxonomy component family).
- **`base_learner`**: `XGBoost` (PMID: 41775771; regularized gradient tree boosting).
- **`imbalance_handling`**: `SMOTE` (PMID: 41006422; synthetic minority over-sampling).

### B. EXPLICITLY_CONFIGURED Primitives (Human Project Gated)
- **`categorical_encoding`**: `one_hot_encoding` (Explicitly supplied in `experiment_config.json`; barred from being claimed as literature evidence).
- **`loss_function`**: `binary_logistic` (Explicitly supplied in `experiment_config.json`; barred from being claimed as literature evidence).

Under no circumstances are explicitly configured primitives labeled as literature evidence, nor are unresolved pipeline slots populated with silent library defaults.

## 3.10 Operational Execution vs. Taxonomy Capabilities
It is essential to distinguish between the **general taxonomy capabilities** supported by the synthesis framework and the **actual executed experimental path** evaluated in a specific benchmark:
1. **Multimodal Primitives (`cross_attention` & `average_ensembling`)**: These mechanisms belong to the general multimodal synthesis taxonomy. Because the HANCOCK benchmark evaluates unimodal clinical tabular data, cross-attention and model ensembling were **dormant / not executed** in the empirical run.
2. **Missing-Value Imputation Primitive**: The evidence-conditioned taxonomy associated missing-value handling with MICE/MissForest-based approaches (PMID: 41826845); however, for the unimodal tabular executor evaluated here, the operational implementation used train-fitted univariate median imputation for numerical variables and most-frequent imputation for categorical variables. Accordingly, the reported empirical results should be interpreted as evaluating this operational tabular implementation rather than an iterative MICE/MissForest estimator.

## 3.11–3.13 Pipeline Materialization, Safety Gates, and Execution Contract
The synthesized pipeline specification is materialized into executable Python classes subject to 10 independent verification gates:
1. Candidate pipeline completeness (all 8 slots defined).
2. Executable implementation class mappings.
3. Train-only preprocessing fit contract.
4. Target isolation firewall (8 outcome variables barred from feature matrix $X$).
5. Patient-level split isolation (strictly zero patient overlap).
6. Multi-seed deterministic split generation.
7. Strict provenance boundary preservation.
8. Explicit configuration verification (no mislabeled defaults).
9. Baseline and compute budget validation (RAM < 4 GB, CPU device, 15 min limit).
10. Cryptographic corpus and pipeline hash consistency (`6b6bcb1b...`).

The execution contract is cryptographically frozen (`6eb6b035...`) prior to authorizing experimental training.

*(Refer to Figure 1 for the architectural flowchart and Figure 7 for the complete component provenance boundary).*
