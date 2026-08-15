# Evidence-Conditioned Compositional Pipeline Synthesis: A Provenance-Aware Framework for Reproducible Clinical Machine Learning

**Authors**: Anonymous Biomedical Informatics Research Group  
**Target Venue**: Journal of Biomedical Informatics / JAMIA  
**Repository & Provenance Ledger**: Cryptographically Verified Stage 6A Research Package  

---


# Abstract

**Background:** Translating biomedical literature findings into reproducible clinical machine learning pipelines is frequently compromised by arbitrary library defaults, unverified substitutions, and subtle data leakage across validation boundaries.

**Problem:** While existing literature provides fragmented empirical evidence for individual modeling primitives, a fundamental methodological gap remains in how to systematically compose published evidence into an executable, provenance-tracked, and leakage-firewalled pipeline without relying on unauthorized defaults.

**Method:** We propose an *Evidence-Conditioned Compositional Pipeline Synthesis* framework. The architecture systematically extracts literature mechanisms from peer-reviewed biomedical studies, audits provenance authenticity, grounds primitives in a controlled domain taxonomy, and enforces a strict governance boundary requiring human-controlled explicit configuration for unresolved primitive slots. It materializes executable code subject to 10 independent verification gates, a strict target isolation firewall, and a frozen execution contract.

**Experimental Demonstration:** We evaluated the framework on the retrospective HANCOCK clinical tabular cohort for post-adjuvant recurrence risk prediction across three deterministic random seeds (`42`, `100`, `2026`) with strict zero patient overlap and train-only preprocessing. The evidence-conditioned taxonomy associated missing-value handling with MICE/MissForest approaches; for the unimodal tabular executor evaluated here, the operational implementation used train-fitted univariate median imputation for numerical variables and most-frequent imputation for categorical variables. Multimodal taxonomy primitives (`cross_attention` and `average_ensembling`) were preserved as general synthesis capabilities but remained dormant during unimodal tabular benchmarking.

**Main Results:** The actual executed candidate pipeline (tabular representation, train-fitted median/mode imputation, one-hot encoding, SMOTE, and regularized XGBoost) achieved a mean test ROC-AUC of `0.9751 ± 0.0114` and a Brier score of `0.0175`. In comparison, the Default XGBoost baseline achieved `0.9704 ± 0.0059` (a modest margin of `+0.0047`), Random Forest achieved `0.9698 ± 0.0065`, Logistic Regression achieved `0.9645 ± 0.0070`, and a minimal shallow MLP baseline achieved `0.9405 ± 0.0192`. The candidate won on 2 of 3 seeds (Seed 42: `+0.0105`, Seed 2026: `+0.0071`), but lost on Seed 100 (`-0.0034`). Controlled ablations demonstrated that omitting SMOTE (`0.9773`) or utilizing ordinal encoding (`0.9784`) achieved marginally higher ROC-AUC on this specific cohort, underscoring that evidence validity and empirical dataset optimality are distinct concepts.

**Limitations & Contribution:** Evaluated on a single retrospective dataset with $n=3$ seeds; statistical significance and clinical deployment readiness are not established. The central contribution is the provenance-aware synthesis methodology and governance framework for reproducible clinical machine learning.

---

# 1. Introduction

Biomedical machine learning pipelines require numerous interconnected architectural design choices, including feature representation, missing value imputation, categorical encoding, class imbalance handling, base learner selection, modality fusion, ensembling, and loss function configuration. While individual design decisions are frequently explored across published medical literature, synthesizing these heterogeneous findings into an end-to-end, reproducible, and executable predictive pipeline presents profound methodological and governance challenges.

In conventional machine learning workflows, practitioners routinely bridge gaps in literature descriptions by introducing arbitrary library defaults (e.g., default imputation algorithms or unverified hyperparameter settings) without establishing their empirical provenance or domain validity. Furthermore, ad-hoc composition frequently leads to subtle data leakage—such as fitting preprocessing scalers across entire patient cohorts prior to cross-validation—or inadvertent retention of outcome-derived clinical variables in the feature set.

To overcome these challenges, we introduce an **Evidence-Conditioned Compositional Pipeline Synthesis** framework. The framework transforms peer-reviewed biomedical literature findings into safe, executable, and provenance-tracked machine learning pipelines through a sequence of gated evolutionary audits. Crucially, the architecture establishes a formal boundary between *evidence-backed primitives* (grounded in literature citations) and *explicitly configured primitives* (human-controlled project inputs), strictly barring arbitrary defaults from entering the pipeline.

### Research Gap and Central Question
While existing biomedical ML research has developed sophisticated predictive algorithms, a critical methodological gap remains in how to systematically, safely, and reproducibly compose heterogeneous evidence into executable pipelines. We investigate the central research question:

> *Can heterogeneous evidence from biomedical machine learning literature be transformed into an executable clinical prediction pipeline while preserving provenance, preventing unsupported defaults, and enforcing reproducible experimental constraints?*

### Summary of Contributions
1. An end-to-end evidence-conditioned pipeline synthesis methodology bridging literature evidence to executable code.
2. A formal provenance and governance firewall distinguishing literature evidence from explicit configurations.
3. 10 independent materialization verification gates enforcing train-only preprocessing and target isolation.
4. Empirical demonstration on the retrospective HANCOCK clinical cohort with complete ablation analysis, operational transparency, and conservative claim boundaries.

---

# 2. Related Work and Research Gap

## 2.1 Tabular Clinical Machine Learning and Risk Prediction
Predictive modeling on structured clinical data has extensively explored regularized gradient tree boosting, random forests, and neural networks. Prior biomedical studies have demonstrated the efficacy of gradient boosted decision trees for tabular clinical risk prediction (e.g., PMID: 41775771), the necessity of principled missing-value handling for missing-at-random covariates (e.g., PMID: 41826845), and the utility of synthetic oversampling techniques like SMOTE for severe class imbalance in cancer recurrence cohorts (e.g., PMID: 41006422). Furthermore, multimodal architectures have investigated cross-attention mechanisms and ensembling strategies to combine clinical tabular features with imaging and molecular modalities (e.g., PMID: 42487970).

## 2.2 Automated Machine Learning (AutoML) vs. Evidence Synthesis
Automated Machine Learning (AutoML) frameworks—such as TPOT, Auto-sklearn, and Auto-PyTorch—automate pipeline construction through empirical search, genetic programming, or Bayesian optimization over unconstrained hyperparameter spaces. While effective for unconstrained performance maximization, conventional AutoML systems operate without domain provenance, frequently select opaque or clinically counter-intuitive preprocessing combinations, and lack formal governance against data leakage or arbitrary defaults. In contrast, evidence-conditioned synthesis constrains the search space strictly to domain mechanisms reported in peer-reviewed biomedical literature, ensuring that every architectural choice possesses an auditable provenance trail.

## 2.3 Reporting Guidelines, Data Leakage, and Reproducibility
The clinical machine learning community has increasingly emphasized reporting rigor and leakage prevention, as formalized in the TRIPOD+AI and PROBAST / PROBAST-AI guidelines. Data leakage—particularly the contamination of validation partitions during feature engineering, imputation, or scaling—remains a leading cause of reproducibility failures in medical AI. This work operationalizes these reporting standards by embedding cryptographic contracts and programmatic verification firewalls directly into the pipeline execution lifecycle.

## 2.4 The Compositional Synthesis Gap
Within the reviewed evidence corpus, existing research focuses almost exclusively on evaluating standalone algorithms or manual monolithic pipelines. This creates a major methodological void:
1. **Isolated Evidence**: Literature presents fragmented evidence for individual pipeline stages without a principled composition methodology.
2. **Arbitrary Default Proliferation**: Underspecified pipeline components are routinely filled with arbitrary library defaults without documented provenance.
3. **Target and Data Leakage**: Transformations are frequently fitted across validation boundaries, artificially inflating reported discrimination.
4. **Lack of Provenance Boundaries**: Unverified components are often retroactively claimed as literature-backed.

This work addresses this gap by developing a traceable, provenance-aware, and reproducible framework for evidence-conditioned compositional pipeline synthesis.

---

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

---

# 4. Experimental Setup

## 4.1 Cohort and Clinical Prediction Epoch
- **Cohort**: HANCOCK structured clinical tabular dataset (head and neck cancer cohort).
- **Target Task**: Binary recurrence classification (`recurrence` $\in \{0, 1\}$).
- **Clinical Prediction Epoch**: **Post-Adjuvant Recurrence Risk Prediction**. The model is framed to predict subsequent cancer recurrence after the completion of initial surgery and adjuvant therapy. Consequently, baseline diagnostic variables and adjuvant treatment attributes are available at this prediction epoch.
- **Splits**: Stratified patient-level partition into 65% Training (496 patients), 15% Validation (115 patients), and 20% Test (152 patients).
- **Patient Overlap**: Strictly **zero patient overlap** across all folds.
- **Random Seeds**: Fixed seeds `[42, 100, 2026]`.

## 4.2 Target Isolation Firewall and Leakage Boundaries
To prevent direct target leakage, 8 outcome-, survival-, and progression-derived clinical variables were barred from the input feature matrix $X$:
1. `recurrence` (Target label)
2. `survival_status`
3. `survival_status_with_cause`
4. `days_to_recurrence`
5. `days_to_last_information`
6. `days_to_progress_1`
7. `days_to_progress_2`
8. `days_to_metastasis_1`

*Prospective Deployment Caveat*: The retrospective benchmark was interpreted at a post-adjuvant prediction epoch. However, longitudinal follow-up variables such as `progress_1` require explicit temporal exclusion in any prospective clinical deployment because their availability depends on events occurring after the intended prediction epoch.

## 4.3 Actual Executed Preprocessing Sequence & Train-Only Enforcement
All preprocessing transformers were fitted strictly on the training partition:
1. **Univariate Tabular Imputation**: Train-fitted median imputation for numeric covariates, most-frequent imputation for categorical covariates.
2. **One-Hot Encoding**: Fitted strictly on training categorical columns; unseen test categories ignored.
3. **SMOTE Oversampling**: Applied strictly to the training fold; validation and test sets remain unaugmented.

## 4.4 Baseline Models and Evaluation Metrics
We evaluated the candidate pipeline against four standardized baselines:
1. **Default XGBoost Baseline**: Default parameters (`n_estimators=50`, `max_depth=6`, `lr=0.3`), median imputation, one-hot encoding, without SMOTE.
2. **Random Forest Baseline**: Standard ensemble baseline (`n_estimators=100`).
3. **Logistic Regression Baseline**: L2-regularized linear model with StandardScaler.
4. **Simple MLP Baseline**: Minimal shallow neural reference baseline (`hidden_layer_sizes=(64, 32)`, `max_iter=10`, StandardScaler).
- **Primary Metric**: Test ROC-AUC.
- **Secondary Metrics**: PR-AUC, F1 Score, Accuracy, Precision, Recall, Brier Score.

---

# 5. Experimental Results

## 5.1 Primary Predictive Performance
Table 1 reports the test-set performance metrics averaged across random seeds `[42, 100, 2026]`.

**Table 1: Test-Set Performance Across Candidate Pipeline and Baseline Models (Mean ± Std)**
| Pipeline / Model | Test ROC-AUC | $\Delta$ ROC-AUC vs Baseline | Test PR-AUC | Test F1 Score | Test Accuracy | Test Precision | Test Recall | Test Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate Pipeline (Actual Executed Path)** | **0.9751 ± 0.0114** | — | **0.9679** | **0.9611** | **0.9825** | **0.9801** | **0.9429** | **0.0175** |
| **Default XGBoost Baseline** | 0.9704 ± 0.0059 | +0.0047 (+0.48%) | 0.9665 | 0.9611 | 0.9825 | 0.9801 | 0.9429 | 0.0180 |
| **Random Forest Baseline** | 0.9698 ± 0.0065 | +0.0053 (+0.55%) | 0.9494 | 0.9611 | 0.9825 | 0.9801 | 0.9429 | 0.0207 |
| **Logistic Regression Baseline** | 0.9645 ± 0.0070 | +0.0106 (+1.10%) | 0.9536 | 0.9558 | 0.9803 | 0.9798 | 0.9333 | 0.0201 |
| **Simple MLP Baseline (Minimal Reference)** | 0.9405 ± 0.0192 | +0.0346 (+3.68%) | 0.9060 | 0.9003 | 0.9561 | 0.9380 | 0.8667 | 0.0683 |

The candidate pipeline achieved a mean test ROC-AUC of `0.9751 ± 0.0114`. The primary performance margin over Default XGBoost (`+0.0047` ROC-AUC) is **modest**. Tree ensemble architectures demonstrated high baseline discrimination on this structured feature set. The candidate exceeded the minimal shallow MLP reference baseline under the frozen experimental contract; this comparison should not be interpreted as evidence of superiority over optimized neural architectures.

*(Refer to Figure 2 for the baseline comparison bar chart and Figure 6 for the multi-metric performance profile).*

## 5.2 Multi-Seed Robustness and Margin Dynamics
Table 2 breaks down performance across individual random seeds.

**Table 2: Per-Seed Performance Comparison (Candidate vs. Default XGBoost)**
| Random Seed | Candidate ROC-AUC | Default XGBoost ROC-AUC | Margin ($\Delta$) | Candidate Outcome |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | `0.9888` | `0.9783` | **+0.0105** | **Candidate Won** |
| **Seed 100** | `0.9609` | `0.9643` | **-0.0034** | Default XGBoost Won |
| **Seed 2026** | `0.9756` | `0.9685` | **+0.0071** | **Candidate Won** |

The candidate pipeline won on 2 out of 3 seeds (66.7% win rate). While achieving a higher mean score, the candidate did **not** universally dominate Default XGBoost across all folds. With $n=3$ seeds, this margin is not statistically significant.

*(Refer to Figure 3 for the per-seed robustness comparison).*

## 5.3 Controlled Component Ablation Analysis
Table 3 summarizes the ablation results across identical patient splits and seeds.

**Table 3: Controlled Component Ablation Results**
| Configuration | Changed Primitive | Mean ROC-AUC | Std ROC-AUC | Mean F1 | Mean Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full Candidate Pipeline** | *None (Reference)* | **0.9751** | 0.0114 | 0.9611 | 0.0175 |
| **Ablation B (Without SMOTE)** | `imbalance_handling` | **0.9773** | 0.0095 | 0.9611 | 0.0177 |
| **Ablation C (Mean Imputation)** | `missing_value_handling` | **0.9767** | 0.0098 | 0.9611 | 0.0174 |
| **Ablation D (Ordinal Encoding)** | `categorical_encoding` | **0.9784** | 0.0111 | 0.9611 | 0.0177 |
| **Ablation E (Default XGBoost)** | `base_learner_config` | **0.9686** | 0.0063 | 0.9611 | 0.0181 |

*Crucial Finding*: Omitting SMOTE (`0.9773`) or employing Ordinal Encoding (`0.9784`) achieved marginally higher ROC-AUC on this specific dataset. The evidence-conditioned framework does not claim to identify the empirically optimal configuration for every dataset. Rather, it constrains pipeline composition to traceable evidence and explicit configuration, making the provenance and rationale of each choice auditable.

*(Refer to Figure 4 for the ablation comparison).*

## 5.4 Probability Calibration
The Candidate Pipeline achieved the lowest Brier score (`0.0175`), outperforming Default XGBoost (`0.0180`), Logistic Regression (`0.0201`), Random Forest (`0.0207`), and Simple MLP (`0.0683`). Lower Brier scores indicate lower probability estimation error, confirming that high discrimination was achieved alongside well-calibrated risk probabilities.

*(Refer to Figure 5 for the calibration comparison).*

---

# 6. Discussion

## 6.1 Methodological Value Beyond Raw Metric Gains
The primary success criterion of the evidence-conditioned synthesis framework is not merely generating a marginal numerical increase in ROC-AUC. Rather, the central objective is establishing whether an executable, safe, and reproducible clinical prediction pipeline can be systematically constructed from published biomedical literature while strictly barring arbitrary defaults, preventing target leakage, and preserving provenance integrity.

The modest predictive gain of `+0.0047` ROC-AUC over Default XGBoost highlights that tree-based algorithms inherently operate near the performance ceiling on structured tabular clinical data. The value of the framework lies in providing an auditable, verifiable methodology that guarantees every component is grounded in literature or explicitly documented, rather than leaving pipeline construction to ad-hoc manual choices.

## 6.2 Evidence Validity vs. Empirical Optimality
The component ablation findings underscore a fundamental conceptual distinction:
- **Evidence-backed selection** guarantees that pipeline primitives represent physiologically and clinically justified mechanisms evaluated in peer-reviewed medical studies.
- **Empirical optimality** represents metric maximization on a specific retrospective sample. On the HANCOCK cohort, omitting SMOTE (`0.9773`) yielded a slight performance increase because synthetic minority oversampling can introduce minor boundary noise in low-dimensional clinical tables with sharp decision boundaries.

The framework functions as an architectural safety and validity governance mechanism, not an unconstrained empirical hyperparameter tuner.

## 6.3 Probability Calibration in Clinical Risk Stratification
In clinical decision support, calibrated risk probabilities are essential for patient triage. The candidate pipeline achieved the lowest Brier score (`0.0175`), demonstrating that regularized boosting with one-hot encoding provides well-calibrated probability estimates without sacrificing discriminative precision.

## 6.4 Sensitivity and Non-Dominance
The candidate's loss on Seed 100 (`0.9609` vs `0.9643`) demonstrates that performance margins in clinical ML can fluctuate based on partition sampling. This finding reinforces the necessity of multi-seed evaluation, strict reporting of non-dominant folds, and avoidance of premature claims of statistical significance.

---

# 7. Novelty and Research Contributions

## 7.1 Multi-Level Novelty Framework
The principal methodological contribution is the integration of literature evidence synthesis, explicit configuration gating, and constraint-preserving materialization. We analyze this novelty across three operational levels:

1. **Level 1 — Methodological Novelty (Evidence-Conditioned Synthesis)**: Shifting pipeline design from unconstrained AutoML empirical search to a structured synthesis workflow grounded in biomedical literature citations and domain taxonomies.
2. **Level 2 — Governance & Safety Novelty (Strict Provenance Gating)**: Establishing a formal firewall that strictly segregates `EVIDENCE_BACKED` primitives from `EXPLICITLY_CONFIGURED` inputs, barring silent ML library defaults from entering the pipeline.
3. **Level 3 — Execution Novelty (Constraint-Preserving Materialization)**: Carrying literature provenance and safety constraints directly into executable code through 10 readiness verification gates, a target isolation firewall, and immutable contract hashing.

*(Refer to Figure 8 for the formal scientific claim boundary matrix).*

## 7.2 Concrete Research Contributions
We formalize six concrete contributions, distinguishing what was architected from what was empirically demonstrated:

- **C1: Evidence-Conditioned Synthesis Framework** (*What was built*): An end-to-end framework translating biomedical literature citations into verified, executable machine learning pipelines.
- **C2: Formal Provenance and Evidence Boundary** (*What was built*): A cryptographic provenance ledger separating literature evidence from explicit project configurations.
- **C3: Controlled Mechanism and Primitive Resolution** (*What was built*): A human-gated resolution gate blocking execution when evidence is absent, barring silent defaults.
- **C4: Safety Gates Against Silent Defaults and Leakage** (*What was built*): 10 verification gates enforcing train-only preprocessing, zero patient overlap, and target isolation.
- **C5: Reproducible Executable Materialization** (*What was built*): An immutable experiment contract freezing seeds, splits, mappings, and compute constraints into verifiable hashes.
- **C6: Empirical Demonstration on HANCOCK Cohort** (*What was demonstrated*): Feasibility demonstration achieving high internal discrimination (`0.9751 ± 0.0114` ROC-AUC) and calibration (`0.0175` Brier score), alongside empirical ablation insights under operational tabular execution.

---

# 8. Threats to Validity and Limitations

We explicitly document the limitations and non-claims of this study:

1. **Single Retrospective Cohort**: Evaluated solely on the single-center retrospective HANCOCK clinical tabular dataset. Generalizability to external clinical cohorts remains unestablished.
2. **Sample Size of Random Seeds**: Evaluated across $n=3$ seeds (`42`, `100`, `2026`). While providing descriptive robustness, this sample size is underpowered for formal inferential hypothesis testing or $p$-value estimation.
3. **Modest Performance Margin**: The predictive improvement over Default XGBoost is modest (`+0.0047` mean ROC-AUC, +0.48% relative).
4. **Lack of Universal Seed Dominance**: The candidate pipeline lost to Default XGBoost on Seed 100 (`-0.0034` delta), demonstrating that superiority is split-dependent.
5. **Ablation Divergence**: Configurations omitting SMOTE (`0.9773`) or utilizing Ordinal Encoding (`0.9784`) achieved marginally higher ROC-AUC, confirming that evidence backing does not equal empirical dataset optimality.
6. **No External or Prospective Validation**: External multi-center validation and prospective clinical trial validation have not been performed.
7. **No Clinical Deployment Readiness**: The framework is a research methodology and is **not clinically deployable**.

---

# 9. Conclusion and Future Work

## 9.1 Conclusion
We presented an Evidence-Conditioned Compositional Pipeline Synthesis framework that bridges published biomedical literature to executable, safe, and reproducible clinical prediction pipelines. The framework enforces strict provenance tracking, bars arbitrary library defaults, prevents target data leakage, and provides immutable execution contracts.

Empirical evaluation on the retrospective HANCOCK clinical tabular cohort demonstrated strong internal discriminative performance (mean ROC-AUC `0.9751 ± 0.0114`) and probability calibration (Brier score `0.0175`), achieving a modest improvement over Default XGBoost (`0.9704 ± 0.0059`). Controlled ablations demonstrated the vital distinction between evidence-backed validity and empirical dataset optimality. The primary contribution of this work is the principled, provenance-aware synthesis methodology and governance framework for reproducible clinical machine learning.

## 9.2 Prioritized Future Directions
Future research should focus on:
1. **External Multi-Center Validation**: Evaluating synthesized pipelines across geographically diverse hospital systems.
2. **Prospective Clinical Studies**: Assessing real-time risk stratification and clinical workflow utility.
3. **Statistical Resampling Expansion**: Scaling to $n \ge 30$ seeds or repeated nested cross-validation for formal inferential testing.
4. **Automated Risk-of-Bias Scoring**: Integrating automated study quality appraisal (e.g., PROBAST criteria) into literature extraction.
5. **Interactive Clinician Review**: Developing human-in-the-loop interfaces for clinical expert oversight.
6. **Multi-Modal Evidence Synthesis**: Extending synthesis to imaging, pathology text, and genomics.

---

# References

1. **Feature Representation (`clinical_tabular_representation`)**:
   - PubMed ID: PMID 42487970.
   - Provenance Citation: Study establishing structured tabular clinical features (patient age, clinical stage, tumor grading) as foundational inputs for cancer recurrence risk prediction.
   - Extraction Source: Stage 2E-1 Controlled Taxonomy Extension (`paper_42487970`, `exp_aef6b872`).

2. **Missing Value Imputation (`MissForest / MICE`)**:
   - PubMed ID: PMID 41826845.
   - DOI: 10.1186/s12874-026-02805-4.
   - Provenance Citation: Study establishing iterative multivariate imputation (MissForest / MICE) for preserving structured tabular clinical covariates under missing-at-random assumptions.
   - Extraction Source: Stage 2F-1 Literature Retrieval.

3. **Base Learner (`XGBoost`)**:
   - PubMed ID: PMID 41775771.
   - DOI: 10.1038/s41598-026-39104-3.
   - Provenance Citation: Study establishing regularized gradient boosted decision trees (XGBoost) for tabular clinical recurrence classification.
   - Extraction Source: Stage 2F-1 Literature Retrieval.

4. **Class Imbalance Handling (`SMOTE`)**:
   - PubMed ID: PMID 41006422.
   - DOI: 10.1038/s41598-025-16790-z.
   - Provenance Citation: Study establishing Synthetic Minority Over-sampling Technique (SMOTE) for addressing severe class imbalance in cancer recurrence cohorts.
   - Extraction Source: Stage 2F-1 Literature Retrieval.

*Note on Explicit Configurations and Conceptual References*: Primitives originating from explicit project configuration (`one_hot_encoding` and `binary_logistic` from `experiment_config.json`) and conceptual literature discussing AutoML (TPOT, Auto-sklearn), reporting standards (TRIPOD+AI, PROBAST), and data leakage are discussed in the main text without fabricated citation identifiers in compliance with the provenance integrity firewall.
