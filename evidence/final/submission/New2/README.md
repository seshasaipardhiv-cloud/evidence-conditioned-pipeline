# Evidence-Conditioned Pipeline — Submission Package (New2)

**Status:** `VERIFIED | SCIENTIFICALLY_RECONCILED | REAL_DATA_VALIDATED`  
**Latest Primary Real Multimodal Experiment:** `REAL MCR-SL MULTIMODAL EXPERIMENT`  
**Test Suite:** `938 PASSED | 0 FAILED`  

---

## Directory Structure

```
New2/
├── FINAL_PROJECT_COMPLETION_REPORT.md  # Exhaustive project completion & reconciliation report
├── README.md                           # Package navigation & manifest summary
├── plots.zip                           # Archive containing all 24 high-resolution plots
│
├── plots/                              # Individual PNG plot assets
│   ├── mcr_sl_fusion_comparison.png    # MCR-SL Image vs Context vs Fusion bar comparison
│   ├── mcr_sl_per_seed_stability.png   # MCR-SL per-seed ROC-AUC across seeds [42, 100, 2026]
│   ├── mcr_sl_roc_pr_curves.png        # MCR-SL ROC and PR curves per architecture (Seed 42)
│   ├── cohort_C_baseline_forensics.png # Forensic baseline comparison for synthetic image cohort
│   ├── cohort_D_baseline_forensics.png # Forensic baseline comparison for synthetic text cohort
│   ├── 01_model_comparison_roc_auc.png # Multi-cohort ROC-AUC benchmark
│   └── ... (24 total plot files)
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
│   ├── mcr_sl_forensic_report.json     # MCR-SL image existence, leakage & split isolation audit
│   ├── cohort_C_D_forensic_report.json # Synthetic cohort C & D baseline forensic audit
│   ├── cohort_forensics.json           # Multi-cohort structural audit
│   ├── evidence_source_verification.json # SciBERT extraction provenance
│   └── provenance_manifest.json        # End-to-end checksum ledger
│
├── results/                            # Markdown reports & consolidated result JSONs
│   ├── MCR_SL_EXPERIMENT_REPORT.md     # Dedicated report for Real MCR-SL experiment
│   ├── mcr_sl_multimodal_results.json  # Machine-readable MCR-SL multi-seed metrics
│   ├── COHORT_C_D_FORENSIC_REPORT.md   # Detailed forensic analysis of synthetic cohorts
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

## Key Experimental Results Summary

### 1. Primary Real Multimodal Experiment: MCR-SL
- **Dataset:** MCR-SL (Multimodal, Context-Rich Skin Lesion)
- **Sample Size:** 234 lesions across 59 subjects (42 malignant / 192 non-malignant)
- **Splits:** Deterministic Subject-Level Stratified Group Split (Zero patient & lesion overlap)
- **Image Modality:** Dermoscopic PNGs (`RGBA`, 1024×1024 or 1750×1750)
- **Clinical Modality:** Serialized pre-diagnostic structured clinical context (20 allowlisted fields)

| Architecture | ROC-AUC (mean ± std) | PR-AUC | Brier Score | Accuracy | F1 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Image-Only (ResNet-18)** | **0.6143 ± 0.0189** | 0.2827 | 0.1474 | 0.8202 | 0.0000 |
| **Clinical-Context-Only (PubMedBERT)** | 0.5012 ± 0.0829 | 0.2099 | 0.1522 | 0.8202 | 0.0000 |
| **Feature Concatenation Fusion** | 0.5049 ± 0.0746 | 0.2248 | 0.1547 | 0.8202 | 0.0000 |
| **Late Fusion** | 0.4627 ± 0.0840 | 0.2048 | 0.2418 | 0.7376 | 0.0979 |
| **Cross-Attention Fusion (Candidate)** | **0.5525 ± 0.0719** | **0.2512** | 0.1525 | 0.8202 | 0.0000 |
| **Gated Multimodal Fusion** | 0.5021 ± 0.0667 | 0.2123 | 0.1595 | 0.8202 | 0.0000 |

### 2. Forensic Reconciliation of Synthetic Demonstration Cohorts
- **Cohort C (Synthetic Derm Image):** High ROC-AUC (0.9957) traced to localized synthetic Gaussian patch offset against stationary noise (`TRIVIAL_SYNTHETIC_SIGNAL`). Zero code/train-test leakage.
- **Cohort D (Synthetic Pathology Text):** Perfect ROC-AUC (1.0000) traced to disjoint diagnostic vocabulary pools in synthetic text generator (`TRIVIAL_SYNTHETIC_SIGNAL`). Zero code/train-test leakage.

---

## Verification & Traceability
- **Test Suite:** 938 tests passing (`pytest backend/tests/`).
- **Prediction Store:** 1,956 individual predictions stored in `canonical_predictions.jsonl`.
- **Plot Generation:** All figures generated directly from canonical prediction JSONL/JSON stores without hardcoded arrays.
