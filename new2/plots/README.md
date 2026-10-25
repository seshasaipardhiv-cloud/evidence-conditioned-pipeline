# Final Plots Index (`new2/plots/`)

This directory contains all 26 publication-grade figures, forensic charts, and performance plots for the **Evidence-Conditioned Automated Machine Learning Pipeline**.

---

## 1. Multi-Cohort Benchmark Performance (`01`–`08`)

| Filename | Description |
|:---|:---|
| [`01_model_comparison_roc_auc.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/01_model_comparison_roc_auc.png) | Cross-cohort ROC-AUC comparison across candidate models and baselines |
| [`02_model_comparison_pr_auc.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/02_model_comparison_pr_auc.png) | Precision-Recall AUC curves across evaluation cohorts |
| [`03_brier_score_comparison.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/03_brier_score_comparison.png) | Probability calibration assessment via Brier score (lower is better) |
| [`04_accuracy_comparison.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/04_accuracy_comparison.png) | Multi-cohort classification accuracy across candidate pipelines |
| [`05_f1_comparison.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/05_f1_comparison.png) | Macro and binary F1 scores across experimental cohorts |
| [`06_candidate_vs_ensemble.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/06_candidate_vs_ensemble.png) | Single best candidate pipeline versus validation-weighted ensemble |
| [`07_ensemble_member_comparison.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/07_ensemble_member_comparison.png) | Individual contributions of ensemble constituent models |
| [`08_ensemble_members.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/08_ensemble_members.png) | Detailed ensemble member weightings and diversity distributions |

---

## 2. Evidence Scoring & Selection Diagnostics (`09`–`15`)

| Filename | Description |
|:---|:---|
| [`09_pipeline_component_comparison.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/09_pipeline_component_comparison.png) | Impact of evidence-selected vs default preprocessing & sampling |
| [`10_evidence_model_ranking.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/10_evidence_model_ranking.png) | Multi-factor evidence ranking scores ($S_{perf}, S_{quality}, S_{app}$) |
| [`11_evidence_confidence_distribution.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/11_evidence_confidence_distribution.png) | SciBERT NER extraction confidence score distribution across corpus |
| [`12_entity_type_distribution.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/12_entity_type_distribution.png) | Extracted entity frequencies across methodology taxonomy categories |
| [`13_evidence_switching_validation.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/13_evidence_switching_validation.png) | Controlled switching experiment proving evidence controls executor model |
| [`14_provenance_coverage.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/14_provenance_coverage.png) | Paper-level provenance coverage across abstract and full-text tiers |
| [`15_modality_pipeline_comparison.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/15_modality_pipeline_comparison.png) | Performance across Tabular, Image, Text, and Multi-modal modalities |

---

## 3. Stability & Forensic Auditing (`16`–`20`)

| Filename | Description |
|:---|:---|
| [`16_per_seed_performance.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/16_per_seed_performance.png) | Multi-seed stability across random seeds `[42, 100, 2026]` |
| [`17_candidate_vs_default_xgboost.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/17_candidate_vs_default_xgboost.png) | Evidence-conditioned pipeline vs unconditioned standalone models across cohorts |
| [`18_end_to_end_pipeline_summary.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/18_end_to_end_pipeline_summary.png) | End-to-end execution summary showing paper $\to$ model $\to$ test results |
| [`19_unconditioned_vs_evidence_pipeline_benchmark.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/19_unconditioned_vs_evidence_pipeline_benchmark.png) | 4-panel comprehensive benchmark: ROC-AUC, F1, Brier score, and net gain |
| [`cohort_C_baseline_forensics.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/cohort_C_baseline_forensics.png) | Deep forensic analysis of Cohort C baselines |
| [`cohort_D_baseline_forensics.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/cohort_D_baseline_forensics.png) | Deep forensic analysis of Cohort D baselines |

---

## 4. Real Multimodal MCR-SL & User Demo Figures (`21`–`26`)

| Filename | Description |
|:---|:---|
| [`mcr_sl_fusion_comparison.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/mcr_sl_fusion_comparison.png) | Real MCR-SL Image vs Context vs Trimodal Early/Late Fusion comparison |
| [`mcr_sl_per_seed_stability.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/mcr_sl_per_seed_stability.png) | Real MCR-SL multi-seed metric distributions across seeds `[42, 100, 2026]` |
| [`mcr_sl_roc_pr_curves.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/mcr_sl_roc_pr_curves.png) | Real MCR-SL ROC and Precision-Recall curves per model architecture |
| [`user_demo_calibration_benchmark.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/user_demo_calibration_benchmark.png) | Calibration curve and reliability diagram from unified runner execution |
| [`user_demo_comparative_performance.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/user_demo_comparative_performance.png) | Comparative performance radar and bar chart from unified runner execution |

---

## 5. Fine-Tuned SciBERT Paper Extraction & Evaluation Suite

| Filename | Description |
|:---|:---|
| [`scibert_paper_extraction_accuracy.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/scibert_paper_extraction_accuracy.png) | Per-paper extraction accuracy across all 30 published biomedical research papers |
| [`scibert_entity_category_accuracy_radar.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/scibert_entity_category_accuracy_radar.png) | Radar diagram of extraction accuracy across 8 biomedical entity categories |
| [`scibert_confidence_by_entity_type.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/scibert_confidence_by_entity_type.png) | Posterior confidence violin and box distributions per entity category |
| [`scibert_normalized_confusion_matrix.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/scibert_normalized_confusion_matrix.png) | High-contrast normalized confusion matrix for token classification |
| [`scibert_paper_methodology_heatmap.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/scibert_paper_methodology_heatmap.png) | Full 30-paper $\times$ 8-category methodology extraction density heatmap |
| [`scibert_extraction_calibration_curve.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/scibert_extraction_calibration_curve.png) | Empirical reliability diagram of SciBERT entity confidence calibration |
| [`scibert_section_context_extraction_fidelity.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/scibert_section_context_extraction_fidelity.png) | Section-context extraction accuracy (Abstract vs Results vs Methods vs Tables) |
| [`per_class_f1_scores.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/per_class_f1_scores.png) | Detailed per-class precision, recall, and F1 bar chart with bold annotations |
| [`training_loss_curve.png`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/training_loss_curve.png) | SciBERT fine-tuning training loss trajectory across 5 epochs |
| [`plot_metadata.json`](file:///A:/Deep%20Learning/evidence-conditioned-pipeline/new2/plots/plot_metadata.json) | Machine-readable plot manifest with hashes, dimensions, and timestamps |
