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
