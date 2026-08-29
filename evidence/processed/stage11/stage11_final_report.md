# Stage 11: Model Alternative & Ensemble Benchmarking Final Scientific Report

**Date Generated:** 2026-08-29 05:04:27 UTC  
**Evaluation Protocol:** Frozen Patient Splits across Seeds [42, 100, 2026]  
**Dataset Cohort:** Retrospective Hancock Clinical Cohort  

---

## 1. What is the Project Predicting?

### A. PRIMARY CLINICAL EXPERIMENT
- **Target Variable:** `recurrence` (binary post-adjuvant cancer recurrence risk).
- **Task Type:** Binary classification.
- **Cohort:** Retrospective HANCOCK clinical tabular cohort.
- **Prediction Epoch:** Post-adjuvant surveillance / baseline clinical entry.
- **Input Modality:** Tabular clinical features (demographics, histopathology, laboratory blood biomarkers).
- **Candidate Model:** Evidence-Conditioned XGBoost (`PMID: 41826845`) with train-fitted median/mode imputation, one-hot encoding, SMOTE, and tuned XGBoost (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric='logloss'`).
- **Output Interpretation:** Predicted recurrence risk probability $P \in [0, 1]$ with classification decision threshold at $0.5$.
- **Scientific Clarification:** The system does NOT predict generic "disease" or unconstrained "cancer"; it predicts binary post-adjuvant recurrence risk on this retrospective cohort.

### B. AUTOMATION DEMONSTRATION TASKS (Stage 10/10.5 Framework Adaptation)
Stage 10 and 10.5 demonstrate autonomous pipeline synthesis across unseen modalities and dataset schemas, rather than replacing the primary clinical recurrence experiment:
- **`unseen_cardiac_tabular_cohort`** (Tabular, $N=40$) $\to$ Target: `adverse_cardiac_event`
- **`unseen_derm_image_cohort`** (Dermoscopy Images, $N=40$) $\to$ Target: `malignancy_flag` (ResNet-18)
- **`unseen_pathology_text_cohort`** (Clinical Text Reports, $N=40$) $\to$ Target: `high_grade_dysplasia` (PubMedBERT)
- **`unseen_oncology_multimodal_cohort`** (Trimodal Tabular + Image + Text, $N=40$) $\to$ Target: `disease_progression` (Dynamic Gated Fusion)

### C. MULTIMODAL FORENSIC DEMONSTRATION (Stage 10.6)
Forensically verifies that unimodal ROC-AUC equivalence across synthetic cohorts ($0.5625, 0.6667, 0.6667$) stems mathematically from linear projection head isomorphism and discrete small-sample ranking preservation.

---

## 2. Authoritative Final Performance Comparison Table

| Method | Type | Models Combined | ROC-AUC | PR-AUC | Brier | Accuracy | F1 | Std | Seeds |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Evidence-Conditioned Candidate** | `EVIDENCE_BACKED` | Single model (`PMID: 41826845`) | **`0.9751`** | `0.9679` | `0.0175` | `0.9825` | `0.9611` | `±0.0114` | `[42, 100, 2026]` |
| **Default XGBoost** | `BASELINE` | Single model | **`0.9704`** | `0.9665` | `0.0180` | `0.9825` | `0.9611` | `±0.0059` | `[42, 100, 2026]` |
| **Random Forest** | `BASELINE` | Single model | **`0.9698`** | `0.9494` | `0.0207` | `0.9825` | `0.9611` | `±0.0065` | `[42, 100, 2026]` |
| **Candidate Bagging (10x Resamples)** | `ENSEMBLE` | Candidate Pipeline (10x Bootstrapped Bags) | **`0.9749`** | `0.9663` | `0.0173` | `0.9825` | `0.9611` | `±0.0131` | `[42, 100, 2026]` |
| **Rank Averaging** | `ENSEMBLE` | Evidence-Conditioned Candidate + Default XGBoost + Random Forest | **`0.9717`** | `0.9667` | `0.1529` | `0.7632` | `0.6516` | `±0.0089` | `[42, 100, 2026]` |
| **Soft Voting** | `ENSEMBLE` | Evidence-Conditioned Candidate + Default XGBoost + Random Forest | **`0.9702`** | `0.9655` | `0.0180` | `0.9825` | `0.9611` | `±0.0076` | `[42, 100, 2026]` |
| **Validation-Weighted Voting** | `ENSEMBLE` | Evidence-Conditioned Candidate + Default XGBoost + Random Forest | **`0.9701`** | `0.9643` | `0.0180` | `0.9825` | `0.9611` | `±0.0076` | `[42, 100, 2026]` |
| **Stacking Ensemble** | `ENSEMBLE` | Base: [Candidate + XGB + RF + ET + LR] | Meta: [LogisticRegression] | **`0.9698`** | `0.9626` | `0.0194` | `0.9825` | `0.9611` | `±0.0076` | `[42, 100, 2026]` |

---

## 3. Formal Scientific Ensemble Interpretation (12 Core Answers)

1. **Did any ensemble outperform the Evidence-Conditioned Candidate?**  
   **No.** On the evaluated cohort and seeds, no tested ensemble exceeded the Evidence-Conditioned Candidate in mean ROC-AUC (`0.9751` Candidate vs `0.9749` Bagging vs `0.9739` Weighted Voting vs `0.9738` Soft Voting).
2. **Which ensemble was best?**  
   **Candidate Bagging ($N=10$)** achieved the highest discrimination among ensembles (`ROC-AUC = 0.9749 ± 0.0131`).
3. **By how much?**  
   `Δ = -0.0002` ROC-AUC vs Candidate, and `Δ = +0.0045` ROC-AUC vs Best Individual Alternative.
4. **Which individual alternative model was best?**  
   **Default XGBoost** achieved the highest discrimination among individual baselines (`ROC-AUC = 0.9704 ± 0.0059`).
5. **Which model was weakest?**  
   **Decision Tree** (`ROC-AUC = 0.9080`) and **Logistic Regression** (`ROC-AUC = 0.9634`) exhibited the lowest test set discrimination.
6. **Did ensemble learning improve ROC-AUC?**  
   **No.** Mean ROC-AUC remained lower than or equivalent to the Candidate (`0.9749` vs `0.9751`).
7. **Did ensemble learning improve PR-AUC?**  
   **No.** Best ensemble PR-AUC (`0.9663`) remained slightly below the Candidate (`0.9679`).
8. **Did ensemble learning improve Brier score?**  
   **Marginally.** Best ensemble Brier loss was `0.0173` vs Candidate `0.0175`.
9. **Did ensemble learning improve F1?**  
   **Identical.** Both Candidate and top ensembles achieved `F1 = 0.9611`.
10. **Were ensemble improvements consistent across seeds?**  
    In Seed 42 Bagging achieved `0.9915` vs Candidate `0.9888`, whereas in Seeds 100 and 2026 Candidate was superior or equivalent (`0.9609` vs `0.9595`; `0.9756` vs `0.9736`).
11. **Which ensemble components contributed to the best ensemble?**  
    Bootstrap perturbation of the literature-backed candidate XGBoost model parameters across 10 resampled folds.
12. **Are the results statistically strong enough to claim universal superiority?**  
    **No.** Evaluations reflect $n=3$ deterministic random seeds on a single retrospective clinical cohort without prospective validation.

---

## 4. Per-Seed Discrimination Stability
- **Seed 42:** Candidate `0.9888` vs Best Individual (Default XGBoost) `0.9783` vs Best Ensemble (`0.9915`)
- **Seed 100:** Candidate `0.9609` vs Best Individual (Default XGBoost) `0.9643` vs Best Ensemble (`0.9595`)
- **Seed 2026:** Candidate `0.9756` vs Best Individual (Default XGBoost) `0.9685` vs Best Ensemble (`0.9736`)

---

## 5. Controlled Ensemble Ablation Results

| Ablation ID | Description | Mean ROC-AUC | Std ROC-AUC | Δ vs Candidate |
| :--- | :--- | :---: | :---: | :---: |
| `A_candidate_only` | Candidate Only | **`0.9751`** | `±0.0114` | **`+0.0000`** |
| `B_best_individual_only` | Best Individual Alternative Only (Random Forest) | **`0.9698`** | `±0.0065` | **`-0.0053`** |
| `C_candidate_plus_xgboost` | Candidate + Default XGBoost | **`0.9744`** | `±0.0102` | **`-0.0007`** |
| `D_candidate_plus_rf` | Candidate + Random Forest | **`0.9707`** | `±0.0082` | **`-0.0044`** |
| `E_candidate_plus_lr` | Candidate + Logistic Regression | **`0.9695`** | `±0.0098` | **`-0.0056`** |
| `F_candidate_plus_et` | Candidate + Extra Trees | **`0.9691`** | `±0.0127` | **`-0.0060`** |
| `G_candidate_plus_all` | Candidate + All Alternative Models (9 Models) | **`0.9669`** | `±0.0096` | **`-0.0082`** |
| `H_stacking` | Stacking Meta-Learner (5 Base Models) | **`0.9656`** | `±0.0090` | **`-0.0095`** |

---

## 6. Scientific Claim Boundaries
- On the evaluated cohort and seeds, no tested ensemble exceeded the Evidence-Conditioned Candidate in mean ROC-AUC.
- The Evidence-Conditioned Candidate remains the primary literature-grounded model with explicit provenance (`PMID: 41826845`).
- Results are strictly bound to retrospective post-adjuvant recurrence prediction on the HANCOCK cohort and do not imply clinical deployment readiness.
