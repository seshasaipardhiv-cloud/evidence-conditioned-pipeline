# Stage 2D Final Submission Package (`evidence/final/submission/New/`)

This directory contains the authoritative, publication-quality deliverables for the **Stage 2D End-to-End Scientific NER & Evidence-Conditioned Multimodal Pipeline Synthesis** project.

## Directory Structure:
- `plots/`: All 18 publication-quality figures with explicit model composition and ensemble member labelling.
- `results/`: `final_results.json` and `final_results.md` summarizing metrics across all 5 benchmark cohorts.
- `evidence/`: `final_evidence_decision_ledger.json` (traceable Paper -> Extraction -> Decision chains) and `old_vs_new_comparison.json`.
- `provenance/`: `provenance_manifest.json` recording cryptographic SHA-256 checkpoint hashes and verified literature PMIDs/DOIs.
- `models/`: `model_registry.json` detailing model hyperparameters and architectures.
- `predictions/`: Machine-readable per-cohort prediction logs (`.jsonl`) with true labels, predicted probabilities, and seed tags.
- `FINAL_PROJECT_COMPLETION_REPORT.md`: Comprehensive end-to-end scientific completion report.
