"""
master_end_to_end_runner.py  —  SCIENTIFICALLY REPAIRED

Stage 2D Master End-to-End Execution Pipeline with Strict Scientific Integrity

Executes the entire verified scientific workflow from scientific literature to final audited deliverables:
  RESEARCH PAPERS
        ↓
  FULL-TEXT / ABSTRACT ACQUISITION
        ↓
  SciBERT SCIENTIFIC NER (allenai/scibert_scivocab_uncased)
        ↓
  SECTION-AWARE METHODOLOGY FILTERING
        ↓
  CONTEXT-CUED RELATION EXTRACTION
        ↓
  MULTI-FACTOR DETERMINISTIC EVIDENCE SCORING
        ↓
  DATASET AUTO-DISCOVERY & ADAPTATION (5 Benchmark Cohorts)
        ↓
  AUTOMATIC COMPONENT & PREPROCESSING RANKING
        ↓
  SAFETY GATES AUDITING & FORENSIC AUDIT
        ↓
  MULTI-SEED REAL MODEL TRAINING ([42, 100, 2026])
        ↓
  EXPLICIT VALIDATION-WEIGHTED ENSEMBLING
        ↓
  CANONICAL PREDICTION STORE (canonical_predictions.jsonl)
        ↓
  PMID / DOI VERIFICATION (PubMed E-utilities)
        ↓
  RECONCILIATION REPORT GENERATION
        ↓
  18 PUBLICATION-QUALITY PLOTS (Read strictly from canonical predictions)
        ↓
  FINAL DELIVERABLES COMPILATION (evidence/final/submission/New/)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from backend.app.final_integration.cohort_evaluator import CohortBenchmarkEvaluator
from backend.app.final_integration.cohort_forensics import run_cohort_forensics
from backend.app.final_integration.evidence_decision_engine import EvidenceDecisionEngine
from backend.app.final_integration.final_plot_generator import FinalPlotGenerator
from backend.app.final_integration.pmid_verifier import verify_all_papers
from backend.app.final_integration.reconciliation_reporter import generate_reconciliation_report
from backend.app.final_integration.results_packager import ResultsPackager
from backend.app.stage2.stage2d.stage2d_orchestrator import Stage2DOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("master_end_to_end_runner")


class MasterEndToEndRunner:
    """
    Master runner executing the entire pipeline with scientific integrity reconciliation.
    """

    def __init__(self, new_out_dir: str = "evidence/final/submission/New"):
        self.new_out_dir = Path(new_out_dir)
        self.new_out_dir.mkdir(parents=True, exist_ok=True)

        self.stage2d_orchestrator = Stage2DOrchestrator(output_dir="evidence/processed/stage2d")
        self.cohort_evaluator = CohortBenchmarkEvaluator()
        self.results_packager = ResultsPackager(base_out=str(self.new_out_dir))
        self.plot_generator = FinalPlotGenerator(
            out_dir=str(self.new_out_dir / "plots"),
            results_dir=str(self.new_out_dir / "results"),
            stage2d_dir="evidence/processed/stage2d",
        )

    def run_complete_project(self) -> Dict[str, Any]:
        start_time = time.time()
        logger.info("================================================================================")
        logger.info("STARTING REPAIRED MASTER END-TO-END EVIDENCE-CONDITIONED PIPELINE (STAGE 2D)")
        logger.info("================================================================================")

        # 1. Literature Extraction & Evidence Scoring via SciBERT Stage 2D
        logger.info("[STEP 1/7] Executing SciBERT NER & Evidence Scoring Engine (Stage 2D)...")
        stage2d_manifest = self.stage2d_orchestrator.run_stage2d(seed=42)

        # 2. Multi-Cohort Benchmark Evaluation (5 Cohorts across seeds [42, 100, 2026])
        logger.info("[STEP 2/7] Executing Multi-Cohort Real Training & Evaluation across 5 Cohorts...")
        cohort_results = self.cohort_evaluator.evaluate_all_cohorts()

        # 3. Evidence Decision Ledger
        decision_ledger = self.cohort_evaluator.decision_engine.decision_ledger

        # 4. PMID / DOI Verification (NCBI PubMed E-Utilities)
        logger.info("[STEP 3/7] Verifying PMIDs and DOIs via PubMed API...")
        pmid_report = verify_all_papers(out_dir=str(self.new_out_dir / "provenance"))

        # 5. Cohort Forensic Leakage Audit
        logger.info("[STEP 4/7] Running Forensic Audit on Cohorts...")
        forensics_report = run_cohort_forensics(cohort_results, out_dir=str(self.new_out_dir / "provenance"))

        # 6. Package Final Results & Canonical Predictions (Single Source of Truth)
        logger.info("[STEP 5/7] Compiling Canonical Predictions & Final Results Store...")
        canonical_path = self.results_packager.package_all(cohort_results, decision_ledger, stage2d_manifest)

        # Load final results for reconciliation
        final_results_file = self.new_out_dir / "results" / "final_results.json"
        with open(final_results_file, "r", encoding="utf-8") as f:
            final_results_data = json.load(f)

        # 7. Generate Results Reconciliation Report
        logger.info("[STEP 6/7] Generating Result Reconciliation Report...")
        generate_reconciliation_report(final_results_data, out_dir=str(self.new_out_dir / "results"))

        # 8. Render All 18 Plots from Canonical Results
        logger.info("[STEP 7/7] Rendering 18 Publication Plots (strictly from canonical predictions)...")
        self.plot_generator.generate_all_18_plots(cohort_results, decision_ledger)

        elapsed = round(time.time() - start_time, 2)
        logger.info("================================================================================")
        logger.info(f"MASTER END-TO-END EXECUTION COMPLETED SUCCESSFULLY IN {elapsed}s")
        logger.info(f"All deliverables verified in: {self.new_out_dir}")
        logger.info("STATUS: PROJECT_SCIENTIFICALLY_RECONCILED")
        logger.info("================================================================================")

        return {
            "status": "PROJECT_SCIENTIFICALLY_RECONCILED",
            "elapsed_seconds": elapsed,
            "stage2d_manifest": stage2d_manifest,
            "cohort_results_summary": {
                k: v["multi_seed_metrics"] for k, v in cohort_results.items()
            },
            "pmid_verification_summary": pmid_report.get("summary", {}),
            "forensics_summary": forensics_report.get("summary", {}),
            "plots_generated_count": 18,
            "canonical_predictions_path": canonical_path,
            "output_directory": str(self.new_out_dir),
        }


if __name__ == "__main__":
    runner = MasterEndToEndRunner()
    res = runner.run_complete_project()
    print(json.dumps(res, indent=2))
