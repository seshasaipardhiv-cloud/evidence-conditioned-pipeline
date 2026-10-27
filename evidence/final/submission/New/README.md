# Stage 2D Final Submission Package

**Canonical Data SHA-256**: `46bca6557151ed0dba660673eba0f4711bafd78a578a8e4736b7ccefb738b9a8`

## Scientific Honesty Statement

- SciBERT NER is **WEAKLY_SUPERVISED** (no human-annotated gold labels exist).
- NER precision/recall/F1 are NOT reported as gold-standard metrics.
- All cohorts are **SYNTHETIC/CONTROLLED DEMONSTRATIONS**, not real clinical datasets.
- All reported metrics are computed programmatically from `results/canonical_predictions.jsonl`.
- No hardcoded performance values appear anywhere in this package.

## Directory Structure

- `plots/` — 18 publication figures generated from canonical results.
- `results/canonical_predictions.jsonl` — **SINGLE SOURCE OF TRUTH** for all metrics.
- `results/final_results.json` — computed from canonical_predictions.
- `results/final_results.md` — computed from canonical_predictions.
- `results/RESULT_RECONCILIATION_REPORT.md` — old vs new values with status.
- `evidence/` — decision ledger and old vs new comparison.
- `provenance/` — SHA-256 manifest, PMID verification, cohort forensics.
- `predictions/` — per-cohort prediction JSONL files.
- `FINAL_PROJECT_COMPLETION_REPORT.md` — all numbers from canonical data.
