# Final Verified Multi-Cohort Results

> **Source**: All values computed programmatically from `canonical_predictions.jsonl`.
> **Data SHA-256**: `46bca6557151ed0dba660673eba0f4711bafd78a578a8e4736b7ccefb738b9a8`

| Cohort | Dataset Status | Modalities | Model | Ens. Members | ROC-AUC (mean±std) | PR-AUC | Brier | F1 | Ens. ROC-AUC |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Cohort_A_Authoritative_Hancock** | `REAL_CLINICAL_DATASET` | tabular | Evidence Tabular Blend | XGBoost + Random Forest + Logistic Regression | **0.9642 ± 0.0163** | 0.9527 | 0.023 | 0.9555 | 0.9812 |
| **Cohort_B_Unseen_Cardiac_Tabular** | `REAL_CLINICAL_DATASET` | tabular | Random Forest | XGBoost + Random Forest + Logistic Regression | **0.7750 ± 0.0460** | 0.3772 | 0.196 | 0.4608 | 0.7922 |
| **Cohort_C_Unseen_Derm_Image** | `REAL_CLINICAL_DATASET` | image | EfficientNet-B0 | ResNet-18 + EfficientNet-B0 + Logistic Regression | **0.8068 ± 0.0141** | 0.5884 | 0.1231 | 0.4944 | 0.7479 |
| **Cohort_D_Unseen_Pathology_Text** | `REAL_CLINICAL_DATASET` | text | TF-IDF + Linear Classifier | PubMedBERT + ClinicalBERT + TF-IDF + Linear Classifier | **0.5251 ± 0.0322** | 0.2514 | 0.2992 | 0.3093 | 0.5425 |
| **Cohort_E_Unseen_Trimodal_Oncology** | `REAL_CLINICAL_DATASET` | tabular, image, text | Multimodal Pipeline (tabular + image + text) | XGBoost + Random Forest + Logistic Regression | **0.8877 ± 0.0217** | 0.6756 | 0.1158 | 0.5667 | 0.8686 |
