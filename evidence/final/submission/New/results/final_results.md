# Final Verified Multi-Cohort Results

> **Source**: All values computed programmatically from `canonical_predictions.jsonl`.
> **Data SHA-256**: `570af01d18e8d2970c18bbf704edd010d71bef83cdde9d22d786139a2c1b0553`

| Cohort | Dataset Status | Modalities | Model | Ens. Members | ROC-AUC (mean±std) | PR-AUC | Brier | F1 | Ens. ROC-AUC |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Cohort_A_Authoritative_Hancock** | `CONTROLLED_SYNTHETIC_DEMONSTRATION` | tabular | XGBoost | XGBoost + Random Forest + Logistic Regression | **0.5536 ± 0.1312** | 0.4554 | 0.2076 | 0.3238 | 0.2840 |
| **Cohort_B_Unseen_Cardiac_Tabular** | `CONTROLLED_SYNTHETIC_DEMONSTRATION` | tabular | XGBoost | XGBoost + Random Forest + Logistic Regression | **0.6741 ± 0.1313** | 0.308 | 0.1845 | 0.1111 | 0.5500 |
| **Cohort_C_Unseen_Derm_Image** | `SYNTHETIC_DEMONSTRATION` | image | ResNet-18 | ResNet-18 + EfficientNet-B0 + Logistic Regression | **0.4125 ± 0.0707** | 0.4417 | 0.4304 | 0.4106 | 0.3241 |
| **Cohort_D_Unseen_Pathology_Text** | `SYNTHETIC_DEMONSTRATION` | text | TF-IDF + Linear Classifier | PubMedBERT + ClinicalBERT + TF-IDF + Linear Classifier | **1.0000 ± 0.0000** | 1.0 | 0.0032 | 1.0 | 1.0000 |
| **Cohort_E_Unseen_Trimodal_Oncology** | `SYNTHETIC_DEMONSTRATION` | tabular, image, text | Multimodal Pipeline (tabular + image + text) | Multimodal Candidate + Tabular-Only Baseline + Vision-Text Baseline | **0.8513 ± 0.2103** | 0.8586 | 0.1585 | 0.5833 | 0.5104 |
