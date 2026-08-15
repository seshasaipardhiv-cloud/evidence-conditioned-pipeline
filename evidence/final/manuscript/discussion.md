# Section 5: Discussion and Empirical Analysis

## 5.1 Interpretation of Predictive Findings
The empirical results demonstrate that regularized gradient boosted decision trees, paired with structured clinical tabular features, achieve high internal discriminative capability (ROC-AUC > 0.97) for retrospective recurrence risk prediction. However, the performance margin of the evidence-conditioned candidate over Default XGBoost (`+0.0047` ROC-AUC) is modest. Tree-based learners inherently capture non-linear interactions among tabular clinical features, creating a performance ceiling on clean retrospective cohorts.

## 5.2 Evidence Validity vs. Empirical Optimality
A critical conceptual insight arising from the component ablations is the fundamental distinction between **evidence-backed validity** and **empirical performance optimality**:
1. **Evidence-backed validity** ensures that pipeline primitives are scientifically motivated, physiologically grounded, and sourced from peer-reviewed clinical studies rather than arbitrary trial-and-error.
2. **Empirical optimality** reflects performance on a specific dataset split. On the HANCOCK cohort, omitting SMOTE (`0.9773`) or using ordinal encoding (`0.9784`) slightly outperformed the candidate pipeline (`0.9751`). SMOTE synthesizes artificial samples along minority class boundaries, which can introduce minor boundary noise in low-dimensional clinical tables where decision boundaries are sharp.
3. Therefore, evidence-conditioned synthesis must be understood as an architectural safety and validity governance mechanism, rather than an automatic empirical hyperparameter optimizer.

## 5.3 Calibration and Clinical Risk Estimation
In biomedical risk estimation, discrimination (ROC-AUC) must not come at the expense of probability calibration. The candidate pipeline achieved the lowest Brier score (`0.0175`), outperforming both Default XGBoost (`0.0180`) and Logistic Regression (`0.0201`). This indicates that the combination of iterative MICE imputation, one-hot feature encoding, and tree regularization produces calibrated risk probabilities suitable for downstream risk stratification.

## 5.4 Seed Sensitivity and Non-Dominance
The loss on Seed 100 (`0.9609` vs `0.9643`) highlights the sensitivity of small sample clinical splits to patient distribution variations. While the candidate won on Seeds 42 and 2026, the lack of universal fold dominance underscores the necessity of multi-seed evaluation and conservative scientific reporting.
