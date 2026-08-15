# Stage 11: Formal Ensemble Model-Composition Manifest

**Generated:** 2026-08-15 16:24:39 UTC  
**Evaluation Seeds:** `[42, 100, 2026]`  
**Cohort:** Retrospective Hancock Clinical Cohort  

---

## Ensemble Composition & Architecture Table

| Ensemble | Method | Models Combined | Meta Model | Selection Rule | Seeds |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Candidate Bagging (10x Resamples)** | `Bootstrap Aggregation` | Candidate Pipeline (10x Bootstrapped Resamples) | `None (Direct Aggregation)` | Bootstrap resampling with replacement on train fold (N=10 bags) | `[42, 100, 2026]` |
| **Validation-Weighted Voting** | `Validation-Performance-Weighted Averaging` | Evidence-Conditioned Candidate + Default XGBoost + Random Forest | `None (Direct Aggregation)` | Softmax temperature=0.5 on validation fold ROC-AUC scores | `[42, 100, 2026]` |
| **Soft Voting** | `Probability Averaging` | Evidence-Conditioned Candidate + Default XGBoost + Random Forest | `None (Direct Aggregation)` | Uniform arithmetic mean of predicted probabilities (1/M) | `[42, 100, 2026]` |
| **Rank Averaging** | `Rank Averaging` | Evidence-Conditioned Candidate + Default XGBoost + Random Forest | `None (Direct Aggregation)` | Normalized percentile rank averaging on predicted probabilities [0, 1] | `[42, 100, 2026]` |
| **Stacking Ensemble** | `Stacking Meta-Learner` | Evidence-Conditioned Candidate + Default XGBoost + Random Forest | `LogisticRegression(penalty='l2', C=1.0, max_iter=1000)` | Logistic Regression meta-model fitted on validation fold probability vectors | `[42, 100, 2026]` |

---

## Detailed Component Specifications

### Candidate Bagging (10x Resamples)
- **Identifier:** `ensemble_bagging`
- **Ensemble Method:** `Bootstrap Aggregation`
- **Constituent Base Models:** Evidence-Conditioned Candidate
- **Meta-Classifier:** `None`
- **Selection Rule:** Bootstrap resampling with replacement on train fold (N=10 bags)
- **Validation Isolation:** Weights and meta-classifiers trained strictly on validation fold predictions. Test data completely isolated.
- **Per-Seed Weights / Fits:**
  - **Seed 42:** `{'bag_1': 0.1, 'bag_2': 0.1, 'bag_3': 0.1, 'bag_4': 0.1, 'bag_5': 0.1, 'bag_6': 0.1, 'bag_7': 0.1, 'bag_8': 0.1, 'bag_9': 0.1, 'bag_10': 0.1}`
  - **Seed 100:** `{'bag_1': 0.1, 'bag_2': 0.1, 'bag_3': 0.1, 'bag_4': 0.1, 'bag_5': 0.1, 'bag_6': 0.1, 'bag_7': 0.1, 'bag_8': 0.1, 'bag_9': 0.1, 'bag_10': 0.1}`
  - **Seed 2026:** `{'bag_1': 0.1, 'bag_2': 0.1, 'bag_3': 0.1, 'bag_4': 0.1, 'bag_5': 0.1, 'bag_6': 0.1, 'bag_7': 0.1, 'bag_8': 0.1, 'bag_9': 0.1, 'bag_10': 0.1}`

### Validation-Weighted Voting
- **Identifier:** `ensemble_weighted_voting`
- **Ensemble Method:** `Validation-Performance-Weighted Averaging`
- **Constituent Base Models:** Evidence-Conditioned Candidate, Default XGBoost, Random Forest
- **Meta-Classifier:** `None`
- **Selection Rule:** Softmax temperature=0.5 on validation fold ROC-AUC scores
- **Validation Isolation:** Weights and meta-classifiers trained strictly on validation fold predictions. Test data completely isolated.
- **Per-Seed Weights / Fits:**
  - **Seed 42:** `{'candidate_pipeline': 0.3353, 'xgboost_default': 0.3307, 'random_forest': 0.334}`
  - **Seed 100:** `{'candidate_pipeline': 0.3332, 'xgboost_default': 0.326, 'random_forest': 0.3409}`
  - **Seed 2026:** `{'candidate_pipeline': 0.3197, 'xgboost_default': 0.3246, 'random_forest': 0.3558}`

### Soft Voting
- **Identifier:** `ensemble_soft_voting`
- **Ensemble Method:** `Probability Averaging`
- **Constituent Base Models:** Evidence-Conditioned Candidate, Default XGBoost, Random Forest
- **Meta-Classifier:** `None`
- **Selection Rule:** Uniform arithmetic mean of predicted probabilities (1/M)
- **Validation Isolation:** Weights and meta-classifiers trained strictly on validation fold predictions. Test data completely isolated.
- **Per-Seed Weights / Fits:**
  - **Seed 42:** `{'candidate_pipeline': 0.3333, 'xgboost_default': 0.3333, 'random_forest': 0.3333}`
  - **Seed 100:** `{'candidate_pipeline': 0.3333, 'xgboost_default': 0.3333, 'random_forest': 0.3333}`
  - **Seed 2026:** `{'candidate_pipeline': 0.3333, 'xgboost_default': 0.3333, 'random_forest': 0.3333}`

### Rank Averaging
- **Identifier:** `ensemble_rank_averaging`
- **Ensemble Method:** `Rank Averaging`
- **Constituent Base Models:** Evidence-Conditioned Candidate, Default XGBoost, Random Forest
- **Meta-Classifier:** `None`
- **Selection Rule:** Normalized percentile rank averaging on predicted probabilities [0, 1]
- **Validation Isolation:** Weights and meta-classifiers trained strictly on validation fold predictions. Test data completely isolated.
- **Per-Seed Weights / Fits:**
  - **Seed 42:** `{'candidate_pipeline': 0.3333, 'xgboost_default': 0.3333, 'random_forest': 0.3333}`
  - **Seed 100:** `{'candidate_pipeline': 0.3333, 'xgboost_default': 0.3333, 'random_forest': 0.3333}`
  - **Seed 2026:** `{'candidate_pipeline': 0.3333, 'xgboost_default': 0.3333, 'random_forest': 0.3333}`

### Stacking Ensemble
- **Identifier:** `ensemble_stacking`
- **Ensemble Method:** `Stacking Meta-Learner`
- **Constituent Base Models:** Evidence-Conditioned Candidate, Default XGBoost, Random Forest
- **Meta-Classifier:** `LogisticRegression(penalty='l2', C=1.0, max_iter=1000)`
- **Selection Rule:** Logistic Regression meta-model fitted on validation fold probability vectors
- **Validation Isolation:** Weights and meta-classifiers trained strictly on validation fold predictions. Test data completely isolated.
- **Per-Seed Weights / Fits:**
  - **Seed 42:** `{'candidate_pipeline': 0.3224, 'xgboost_default': 0.3112, 'random_forest': 0.3663}`
  - **Seed 100:** `{'candidate_pipeline': 0.3177, 'xgboost_default': 0.326, 'random_forest': 0.3563}`
  - **Seed 2026:** `{'candidate_pipeline': 0.3048, 'xgboost_default': 0.3096, 'random_forest': 0.3857}`

