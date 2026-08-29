# Final Verified Multi-Cohort Results

> **Source**: All values computed programmatically from `canonical_predictions.jsonl`.
> **Data SHA-256**: `b37b3910132b29b85f46a8e0c7186af62df5e642401d92a9b5b81d2140435906`

| Cohort | Dataset Status | Modalities | Model | Ens. Members | ROC-AUC (mean±std) | PR-AUC | Brier | F1 | Ens. ROC-AUC |
|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **Cohort_A_Authoritative_Hancock** | `CONTROLLED_SYNTHETIC_DEMONSTRATION` | tabular | XGBoost | XGBoost + Random Forest + Logistic Regression | **0.5536 ± 0.1312** | 0.4554 | 0.2076 | 0.3238 | 0.2840 |
| **Cohort_B_Unseen_Cardiac_Tabular** | `CONTROLLED_SYNTHETIC_DEMONSTRATION` | tabular | XGBoost | XGBoost + Random Forest + Logistic Regression | **0.6741 ± 0.1313** | 0.308 | 0.1845 | 0.1111 | 0.5500 |
| **Cohort_C_Unseen_Derm_Image** | `SYNTHETIC_DEMONSTRATION` | image | ResNet-18 | ResNet-18 + EfficientNet-B0 + Logistic Regression | **0.9957 ± 0.0061** | 0.994 | 0.0427 | 0.9267 | 1.0000 |
| **Cohort_D_Unseen_Pathology_Text** | `SYNTHETIC_DEMONSTRATION` | text | TF-IDF + Linear Classifier | PubMedBERT + ClinicalBERT + TF-IDF + Linear Classifier | **1.0000 ± 0.0000** | 1.0 | 0.0032 | 1.0 | 1.0000 |
| **Cohort_E_Unseen_Trimodal_Oncology** | `SYNTHETIC_DEMONSTRATION` | tabular, image, text | Multimodal Pipeline (tabular + image + text) | Multimodal Candidate + Tabular-Only Baseline + Vision-Text Baseline | **0.8646 ± 0.1915** | 0.8556 | 0.0909 | N/A | 0.7000 |
