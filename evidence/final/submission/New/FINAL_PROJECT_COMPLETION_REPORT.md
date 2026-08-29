# Final Project Completion Report: Stage 2D Scientific Integrity Repair

**Generated**: 2026-08-29T09:32:54.291009+00:00
**Canonical Data SHA-256**: `b37b3910132b29b85f46a8e0c7186af62df5e642401d92a9b5b81d2140435906`
**Source**: All metrics computed from `results/canonical_predictions.jsonl`

---

## ⚠️ SCIENTIFIC LIMITATIONS — READ FIRST

### What This System Does NOT Claim

- **NER is NOT gold-standard supervised**: Training supervision = `WEAKLY_SUPERVISED`.
  Labels were generated programmatically by `AdvancedWeakLabeler`. No human-annotated
  ground truth exists. `ground_truth_status = NOT_AVAILABLE_WITHOUT_GOLD_LABELS`.
  NER precision/recall/F1 are NOT reported as scientific NER metrics.

- **Relation extraction is HEURISTIC**: No trained neural relation extraction model.
  Relations are inferred from entity proximity and syntactic context.

- **All cohorts are SYNTHETIC/CONTROLLED DEMONSTRATIONS**. None are real clinical
  datasets. Results do NOT establish clinical superiority or deployment readiness.

- **Small cohorts (n=60)** severely limit statistical conclusions.
  Standard errors across 3 seeds reflect data variability, not true generalisability.

- **Vision and language "model" proxies**: ResNet-18 and PubMedBERT names reflect
  evidence-selected architectures; actual training uses sklearn MLP/LR proxies
  on pixel/TF-IDF features. Full deep model training would require GPU infrastructure.

- **Evidence routing**: 5/7 decisions were
  `RUNTIME_MATCHED` from actual SciBERT extraction. 2 used `FALLBACK_DEFAULT`
  (score=0.50). FALLBACK decisions are NOT literature-derived evidence.

---

## 1. Architecture

```
Research Papers (30 synthetic)
    → SciBERT NER (WEAKLY_SUPERVISED, conf=0.154 mean, ALL LOW tier)
    → Section-Aware Evidence Scoring
    → Runtime Evidence Decision Engine (RUNTIME_MATCHED or FALLBACK_DEFAULT)
    → Dataset Auto-Discovery (5 cohorts)
    → Model / Preprocessing / Fusion Selection
    → Safety Gates
    → Real Training (sklearn proxies, seeds=[42,100,2026])
    → Actual Predictions
    → canonical_predictions.jsonl (SHA-256=b37b3910132b29b8...)
    → Computed Metrics
    → 18 Plots (from canonical data)
    → This Report
```

---

## 2. SciBERT NER Training Status

| Field | Value |
|---|---|
| Model | `allenai/scibert_scivocab_uncased` |
| Encoder frozen? | No — top layers unfrozen for fine-tuning |
| Classification head | Trainable linear NER head |
| Training data | 30 synthetic papers (programmatic labels) |
| Supervision | `WEAKLY_SUPERVISED_WITH_NOISE_ROBUST_TRAINING` |
| Ground truth | `NOT_AVAILABLE_WITHOUT_GOLD_LABELS` |
| Entities extracted | 87 |
| Mean NER confidence | 0.154 (ALL classified as LOW) |
| Checkpoint SHA-256 | `405fc1be40760a25a2426bc6213072dd03deb1a46f72478c4f7f63683398eacf` |

---

## 3. Evidence Routing

- RUNTIME_MATCHED decisions: **5**
- FALLBACK_DEFAULT decisions: **2**

FALLBACK means no SciBERT-extracted entity matched the candidate in `evidence_scores.json`.
Score = 0.50 for all FALLBACK candidates → selection determined by priority order.

---

## 4. Real Performance Results (from `canonical_predictions.jsonl`)

| Cohort | Dataset Status | Model | ROC-AUC (mean±std) | PR-AUC | Brier | F1 |
|---|---|---|:---:|:---:|:---:|:---:|
| **Cohort_A_Authoritative_Hancock** | `CONTROLLED_SYNTHETIC_DEMONSTRATION` | XGBoost | **0.5536 ± 0.1312** | 0.4554 | 0.2076 | 0.3238 |
| **Cohort_B_Unseen_Cardiac_Tabular** | `CONTROLLED_SYNTHETIC_DEMONSTRATION` | XGBoost | **0.6741 ± 0.1313** | 0.308 | 0.1845 | 0.1111 |
| **Cohort_C_Unseen_Derm_Image** | `SYNTHETIC_DEMONSTRATION` | ResNet-18 | **0.9957 ± 0.0061** | 0.994 | 0.0427 | 0.9267 |
| **Cohort_D_Unseen_Pathology_Text** | `SYNTHETIC_DEMONSTRATION` | TF-IDF + Linear Classifier | **1.0000 ± 0.0000** | 1.0 | 0.0032 | 1.0 |
| **Cohort_E_Unseen_Trimodal_Oncology** | `SYNTHETIC_DEMONSTRATION` | Multimodal Pipeline (tabular + image + text) | **0.8646 ± 0.1915** | 0.8556 | 0.0909 | 0.0 |

---

## 5. Cohort Forensic Audit

### Cohort A (Hancock)
**Previous result (REJECTED)**: ROC-AUC = 1.000
**Reason**: `TARGET_ENCODED_FEATURE_LEAKAGE` — ki67_proliferation_index, tumor_size_mm,
and lymph_node_positive contained label-dependent offsets (`+ 15.0 if label==1 else 0`).
**Corrective action**: All label-derived offsets removed. Features now independent of target.
**New result**: See canonical_predictions.jsonl (expect realistic imperfect performance).

### Cohort C (Derm Image)
**Dataset**: 32×32 random noise PNG images with white square patch as the only signal.
**Expected**: Near-random performance (ROC-AUC ≈ 0.5–0.65).
**Reported**: Actual value from canonical_predictions.jsonl.

### Cohort D (Pathology Text)
**Dataset**: Template text strings (two fixed sentences per class).
**Expected**: Variable performance depending on TF-IDF feature extraction.
**Reported**: Actual value from canonical_predictions.jsonl.

### Cohort E (Trimodal Oncology)
**Previous bug**: Only 1/18 predictions stored per seed.
**Corrective action**: Multimodal executor now returns complete prediction arrays.

---

## 6. Software Tests vs Scientific Validation

### Software Tests (existing suite)
- Tests verified software behaviour (JSON schema, file existence, value ranges).
- **Do NOT constitute scientific validation.**

### Scientific Validation Tests (`test_scientific_validation.py`)
- `test_no_train_test_identifier_overlap` — leakage absence
- `test_metric_reproduction_from_predictions` — metric reproducibility
- `test_ensemble_reproduction_from_member_preds` — ensemble reproducibility
- `test_no_hardcoded_arrays_in_plot_generator` — no fabricated arrays
- `test_evidence_propagation_sensitivity` — evidence routing sensitivity
- `test_prediction_file_completeness` — no truncated prediction files
- `test_no_target_derived_features` — Cohort A leakage free
- `test_fallback_evidence_is_explicit` — FALLBACK status logged
- `test_pmid_verification_recorded` — verification status stored
- `test_plot_metadata_hash_matches_canonical` — plot traceability

---

## 7. Claims NOT Made

- ❌ "Clinically validated"
- ❌ "Clinical deployment ready"
- ❌ "Clinically superior"
- ❌ "NER F1 = X% (gold standard)"
- ❌ "Results generalise to real patients"
- ❌ "Ensemble outperforms all baselines" (report actual comparison)
