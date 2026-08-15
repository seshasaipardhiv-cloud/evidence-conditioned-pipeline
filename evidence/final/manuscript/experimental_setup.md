# Experimental Setup and Empirical Evaluation

## 1. Dataset and Target Task
- **Dataset**: HANCOCK structured clinical tabular cohort (head and neck cancer clinical records).
- **Target Task**: Binary recurrence classification (`recurrence` $\in \{0, 1\}$).

## 2. Cohort Splitting Protocol
- **Partitioning**: 65% Training (496 patients), 15% Validation (115 patients), 20% Test (152 patients).
- **Patient Isolation**: Stratified patient-level splitting with strictly **0 patient overlap** across all folds.
- **Random Seeds**: Evaluated across 3 deterministic random seeds: `42`, `100`, `2026`.

## 3. Target Isolation Firewall
To prevent subtle data leakage, 8 outcome-, survival-, and progression-derived variables were barred from the feature matrix $X$:
1. `recurrence` (Target label)
2. `survival_status`
3. `survival_status_with_cause`
4. `days_to_recurrence`
5. `days_to_last_information`
6. `days_to_progress_1`
7. `days_to_progress_2`
8. `days_to_metastasis_1`

## 4. Train-Only Preprocessing Sequence
All transformations were strictly fitted on the training split and applied out-of-sample:
1. `MissForest / MICE` (Iterative tabular multivariate imputer fitted on training set)
2. `OneHotEncoder` (Fitted on training categorical features, unseen categories ignored)
3. `SMOTE` (Applied strictly to the training fold; validation and test splits remain unaugmented)

## 5. Evaluation Protocol & Baseline Models
- **Primary Metric**: Test ROC-AUC.
- **Secondary Metrics**: PR-AUC, F1 Score, Accuracy, Precision, Recall, Brier Score.
- **Baselines Evaluated**:
  1. Default XGBoost
  2. Random Forest
  3. Logistic Regression
  4. Simple Multi-Layer Perceptron (MLP)

## 6. Authoritative Empirical Results

### Multi-Seed Aggregate Performance
- **Candidate Pipeline (Evidence-Conditioned XGBoost)**:
  - **ROC-AUC**: `0.9751 ± 0.0114`
  - **PR-AUC**: `0.9679`
  - **F1 Score**: `0.9611`
  - **Accuracy**: `0.9825`
  - **Precision**: `0.9801`
  - **Recall**: `0.9429`
  - **Brier Score**: `0.0175`
- **Default XGBoost Baseline**: `0.9704 ± 0.0059` (Delta: `+0.0047`, +0.48% relative)
- **Random Forest Baseline**: `0.9698 ± 0.0065` (Delta: `+0.0053`, +0.55% relative)
- **Logistic Regression Baseline**: `0.9645 ± 0.0070` (Delta: `+0.0106`, +1.10% relative)
- **Simple MLP Baseline**: `0.9405 ± 0.0192` (Delta: `+0.0346`, +3.68% relative)

### Per-Seed Results (Candidate vs Default XGBoost)
- **Seed 42**: Candidate `0.9888` vs Default `0.9783` (**Candidate Won**, `+0.0105`)
- **Seed 100**: Candidate `0.9609` vs Default `0.9643` (**Candidate Lost**, `-0.0034`)
- **Seed 2026**: Candidate `0.9756` vs Default `0.9685` (**Candidate Won**, `+0.0071`)

### Component Ablation Findings
All ablations evaluated on identical patient splits and seeds:
- **Full Candidate Pipeline**: `0.9751`
- **Ablation B (Without SMOTE)**: `0.9773`
- **Ablation C (Mean Imputation)**: `0.9767`
- **Ablation D (Ordinal Encoding)**: `0.9784`
- **Ablation E (Default XGBoost)**: `0.9686`

*Crucial Scientific Insight*: Evidence-backed validity and empirical performance optimality on a single retrospective dataset are distinct concepts.
