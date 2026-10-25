# Final Verified Multi-Cohort Results

> **Source**: All values computed programmatically from `canonical_predictions.jsonl`.
> **Data SHA-256**: `5f7f10a837f5d1c1b1fe5968014c668d7a30a2eb34f004bb46e8e73f29dbcb59`

| Cohort | Dataset Status | Modalities | Model | Ens. Members | ROC-AUC (mean±std) | PR-AUC | Brier | F1 | Ens. ROC-AUC |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Cohort_A_Authoritative_Hancock** | `REAL_CLINICAL_DATASET` | tabular | XGBoost | XGBoost + Random Forest + Logistic Regression | **0.9633 ± 0.0037** | 0.9437 | 0.0227 | 0.9524 | 0.9807 |
| **Cohort_B_Unseen_Cardiac_Tabular** | `REAL_CLINICAL_DATASET` | tabular | XGBoost | XGBoost + Random Forest + Logistic Regression | **0.7862 ± 0.0543** | 0.4069 | 0.2077 | 0.4522 | 0.7922 |
| **Cohort_C_Unseen_Derm_Image** | `REAL_CLINICAL_DATASET` | image | ResNet-18 | ResNet-18 + EfficientNet-B0 + Logistic Regression | **0.8068 ± 0.0141** | 0.5884 | 0.1231 | 0.4944 | 0.7479 |
| **Cohort_D_Unseen_Pathology_Text** | `REAL_CLINICAL_DATASET` | text | PubMedBERT | PubMedBERT + ClinicalBERT + TF-IDF + Linear Classifier | **0.5681 ± 0.0183** | 0.3083 | 0.206 | 0.3084 | 0.5425 |
| **Cohort_E_Unseen_Trimodal_Oncology** | `REAL_CLINICAL_DATASET` | tabular, image, text | Multimodal Pipeline (tabular + image + text) | XGBoost + Random Forest + Logistic Regression | **0.8373 ± 0.0530** | 0.6105 | 0.1285 | 0.5847 | 0.8686 |
