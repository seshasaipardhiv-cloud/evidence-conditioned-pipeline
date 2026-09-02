# Cover Letter

**Date:** September 02, 2026  
**To:** The Editor-in-Chief  
**Target Journal:** *Journal of Biomedical Informatics* / *JAMIA*  
**Submission Title:** Evidence-Conditioned Compositional Pipeline Synthesis: A Provenance-Aware Framework for Reproducible Clinical Machine Learning  
**Manuscript Type:** Original Research / Methodology  

Dear Editor-in-Chief and Editorial Board,

We are pleased to submit our original research manuscript titled **"Evidence-Conditioned Compositional Pipeline Synthesis: A Provenance-Aware Framework for Reproducible Clinical Machine Learning"** for consideration for publication in your journal.

### Problem and Context
Translating published biomedical literature findings into reproducible, leak-free clinical machine learning pipelines presents a major methodological challenge. In contemporary clinical AI workflows, researchers routinely bridge gaps in literature descriptions by introducing arbitrary library defaults without documented empirical provenance. Furthermore, uncoordinated composition frequently causes subtle data leakage across validation folds, severely undermining reproducibility and clinical translation.

### Methodological Contribution
To resolve this gap, we present an **Evidence-Conditioned Compositional Pipeline Synthesis** framework. The architecture systematically extracts literature mechanisms from peer-reviewed studies, audits provenance authenticity, grounds primitives in a controlled domain taxonomy, and enforces a strict governance firewall separating literature-grounded primitives from human-controlled explicit configurations. Executable pipelines are materialized under 10 independent verification gates and executed via cryptographically frozen contracts.

### Empirical Validation and Conservative Scientific Boundaries
We demonstrate the framework on the retrospective HANCOCK clinical tabular cohort for post-adjuvant recurrence risk prediction across 3 deterministic seeds (`42`, `100`, `2026`) with zero patient overlap and train-only preprocessing.
- The actual executed candidate pipeline achieved a mean test ROC-AUC of `0.9751 ± 0.0114` and a Brier score of `0.0175`, compared to `0.9704 ± 0.0059` for Default XGBoost (a modest margin of `+0.0047`) and `0.9405 ± 0.0192` for a minimal shallow MLP baseline (`max_iter=10`).
- The candidate won on 2 of 3 seeds but lost on Seed 100 (`-0.0034` delta).
- Controlled ablations revealed that omitting SMOTE (`0.9773`) or using ordinal encoding (`0.9784`) achieved marginally higher ROC-AUC on this specific sample, illustrating that evidence validity and empirical dataset optimality are distinct concepts.
- The manuscript strictly avoids hyperbolic claims: no claims of statistical significance, universal superiority, or clinical deployment readiness are made.

### Governance and Transparency Highlights
- **Exact Operational Imputation Disclosed:** The manuscript transparently notes that while the taxonomy associated missing-value handling with MICE/MissForest, the operational tabular executor used train-fitted median/mode imputation.
- **Dormant Multimodal Primitives:** Cross-attention and model ensembling are formally documented as dormant taxonomy capabilities inactive during unimodal tabular benchmarking.
- **Temporal Prediction Epoch:** Formally anchored to *Post-Adjuvant Recurrence Risk Prediction*, with an explicit prospective caveat regarding follow-up variables such as `progress_1`.

### Statements of Compliance
1. This manuscript represents original work and is not under consideration for publication elsewhere.
2. All authors have reviewed and approved the manuscript.
3. All code, synthesized pipeline configs, and verification test suites (720/720 passing) are made fully open-source and reproducible.

We thank you and the reviewers for your time and consideration of our work.

Sincerely,

**The Authors**  
*On behalf of the Research Collaboration*  
Corresponding Author Placeholder  
Department of Biomedical Informatics Placeholder  
Email: CORRESPONDING_EMAIL_PLACEHOLDER@institution.edu
