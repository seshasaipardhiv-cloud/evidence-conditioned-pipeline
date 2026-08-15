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
