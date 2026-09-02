# Evidence-Conditioned Pipeline — Submission Package (New2 / new2)

**Status:** `VERIFIED | SCIENTIFICALLY_RECONCILED | REAL_DATA_VALIDATED`  
**Latest Primary Real Multimodal Experiment:** `REAL MCR-SL MULTIMODAL EXPERIMENT`  
**Authoritative Real Tabular Experiment:** `Cohort A (Hancock Cystectomy Recurrence)`  
**Synthetic Control Forensics:** `Cohort C (Derm Image) & Cohort D (Pathology Text)`  
**Test Suite Status:** `938 PASSED | 0 FAILED`  

---

## Folder Navigation & Contents

```
evidence/final/submission/New2/ (new2/)
├── FINAL_PROJECT_COMPLETION_REPORT.md  # Comprehensive project report (MCR-SL + synthetic forensics)
├── README.md                           # Package navigation & quick-reference summary
├── plots.zip                           # Compressed archive containing all 24 plot assets
│
├── plots/                              # 24 high-resolution plot PNGs
│   ├── mcr_sl_fusion_comparison.png    # Real MCR-SL Image vs Context vs Fusion bar chart
│   ├── mcr_sl_per_seed_stability.png   # Real MCR-SL per-seed ROC-AUC across seeds [42, 100, 2026]
│   ├── mcr_sl_roc_pr_curves.png        # Real MCR-SL ROC and PR curves per architecture (Seed 42)
│   ├── cohort_C_baseline_forensics.png # Forensic baseline hierarchy for synthetic image cohort
│   ├── cohort_D_baseline_forensics.png # Forensic baseline hierarchy for synthetic text cohort
│   ├── 01_model_comparison_roc_auc.png # Multi-cohort benchmark ROC-AUC
│   └── ... (24 total plot PNGs)
│
├── predictions/                        # Sample-level canonical prediction records (.jsonl)
│   ├── Cohort_MCR_SL_Real_Multimodal_predictions.jsonl # Real MCR-SL multi-seed predictions
│   ├── Cohort_A_Authoritative_Hancock_predictions.jsonl
│   ├── Cohort_B_Unseen_Cardiac_Tabular_predictions.jsonl
│   ├── Cohort_C_Unseen_Derm_Image_predictions.jsonl
│   ├── Cohort_D_Unseen_Pathology_Text_predictions.jsonl
│   └── Cohort_E_Unseen_Trimodal_Oncology_predictions.jsonl
│
├── provenance/                         # Machine-readable forensic and audit artifacts
│   ├── mcr_sl_forensic_report.json     # Real MCR-SL image existence, leakage & split isolation audit
│   ├── cohort_C_D_forensic_report.json # Synthetic cohort C & D baseline forensic audit
│   ├── cohort_forensics.json           # Multi-cohort structural audit
│   ├── evidence_source_verification.json # SciBERT extraction provenance
│   └── provenance_manifest.json        # End-to-end checksum ledger
│
├── results/                            # Markdown reports & consolidated result JSONs
│   ├── MCR_SL_EXPERIMENT_REPORT.md     # Dedicated report for Real MCR-SL experiment
│   ├── mcr_sl_multimodal_results.json  # Machine-readable MCR-SL multi-seed metrics
│   ├── COHORT_C_D_FORENSIC_REPORT.md   # Forensic analysis of synthetic cohorts
│   ├── canonical_predictions.jsonl     # Consolidated 1,956 sample-level predictions
│   ├── final_results.json              # Canonical multi-cohort metrics
│   ├── final_results.md                # Multi-cohort benchmark summary table
│   └── RESULT_RECONCILIATION_REPORT.md # Post-hoc statistical reconciliation
│
├── evidence/                           # Evidence decision engine ledgers
│   ├── final_evidence_decision_ledger.json
│   └── old_vs_new_comparison.json
│
└── models/                             # Model registry & architecture specifications
    └── model_registry.json
```

---

## Summary of Results

### 1. Primary Real Multimodal Experiment: MCR-SL (Skin Lesion Malignancy)
- **Modality:** Real Dermoscopic PNG Images + Serialized Pre-Diagnostic Clinical Context (20 allowlisted fields)
- **Target:** `mcr_sl_malignancy` (42 malignant / 192 non-malignant, 234 total labeled lesions across 59 subjects)
- **Splits:** Deterministic Subject-Level Stratified Group Split (Zero patient & lesion overlap)

| Architecture | ROC-AUC (mean ± std) | PR-AUC | Brier Score | Accuracy | F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Image-Only (ResNet-18)** | **0.6143 ± 0.0189** | 0.2827 | 0.1474 | 0.8202 | 0.0000 |
| **Clinical-Context-Only (PubMedBERT)** | 0.5012 ± 0.0829 | 0.2099 | 0.1522 | 0.8202 | 0.0000 |
| **Feature Concatenation Fusion** | 0.5049 ± 0.0746 | 0.2248 | 0.1547 | 0.8202 | 0.0000 |
| **Late Fusion** | 0.4627 ± 0.0840 | 0.2048 | 0.2418 | 0.7376 | 0.0979 |
| **Cross-Attention Fusion (Candidate)** | **0.5525 ± 0.0719** | **0.2512** | 0.1525 | 0.8202 | 0.0000 |
| **Gated Multimodal Fusion** | 0.5021 ± 0.0667 | 0.2123 | 0.1595 | 0.8202 | 0.0000 |

### 2. Forensic Reconciliation of Synthetic Demonstration Cohorts
- **Cohort C (Synthetic Derm Image):** ROC-AUC 0.9957 scientifically explained by central Gaussian intensity offset (`TRIVIAL_SYNTHETIC_SIGNAL`; zero train/test leakage).
- **Cohort D (Synthetic Pathology Text):** ROC-AUC 1.0000 scientifically explained by disjoint diagnostic phrase dictionaries (`TRIVIAL_SYNTHETIC_SIGNAL`; zero train/test leakage).
