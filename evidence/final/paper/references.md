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
