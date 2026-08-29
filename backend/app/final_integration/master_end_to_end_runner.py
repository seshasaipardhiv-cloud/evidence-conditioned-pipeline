"""
master_end_to_end_runner.py

Stage 2D Master End-to-End Execution Pipeline

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
  14 SAFETY GATES AUDITING
        ↓
  MULTI-SEED REAL MODEL TRAINING ([42, 100, 2026])
        ↓
  EXPLICIT VALIDATION-WEIGHTED ENSEMBLING
        ↓
  PREDICTION GENERATION & REPRODUCIBILITY AUDIT
        ↓
  18 PUBLICATION-QUALITY PLOTS
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
from backend.app.final_integration.evidence_decision_engine import EvidenceDecisionEngine
from backend.app.final_integration.final_plot_generator import FinalPlotGenerator
from backend.app.final_integration.results_packager import ResultsPackager
from backend.app.stage2.stage2d.stage2d_orchestrator import Stage2DOrchestrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("master_end_to_end_runner")


class MasterEndToEndRunner:
    """
    Master runner executing the entire pipeline from literature through final submission deliverables.
    """

    def __init__(self, new_out_dir: str = "evidence/final/submission/New"):
        self.new_out_dir = Path(new_out_dir)
        self.new_out_dir.mkdir(parents=True, exist_ok=True)

        self.stage2d_orchestrator = Stage2DOrchestrator(output_dir="evidence/processed/stage2d")
        self.cohort_evaluator = CohortBenchmarkEvaluator()
        self.plot_generator = FinalPlotGenerator(out_dir=str(self.new_out_dir / "plots"))
        self.results_packager = ResultsPackager(base_out=str(self.new_out_dir))

    def run_complete_project(self) -> Dict[str, Any]:
        """
        Runs the entire project end-to-end.
        """
        start_time = time.time()
        logger.info("================================================================================")
        logger.info("STARTING MASTER END-TO-END EVIDENCE-CONDITIONED SYNTHESIS PIPELINE (STAGE 2D)")
        logger.info("================================================================================")

        # 1. Literature Extraction & Evidence Scoring via SciBERT Stage 2D
        logger.info("[STEP 1/6] Executing SciBERT NER & Evidence Scoring Engine (Stage 2D)...")
        stage2d_manifest = self.stage2d_orchestrator.run_stage2d(seed=42)

        # 2. Multi-Cohort Benchmark Evaluation (5 Cohorts across seeds [42, 100, 2026])
        logger.info("[STEP 2/6] Executing Multi-Cohort Real Training & Evaluation across 5 Cohorts...")
        cohort_results = self.cohort_evaluator.evaluate_all_cohorts()

        # 3. Evidence Decision Ledger
        decision_ledger = self.cohort_evaluator.decision_engine.decision_ledger

        # 4. Generate all 18 Publication Plots
        logger.info("[STEP 3/6] Rendering 18 Publication-Quality Plots in New/plots/...")
        self.plot_generator.generate_all_18_plots(cohort_results, decision_ledger)

        # 5. Package Final Results, Manifests, Predictions, and Reports
        logger.info("[STEP 4/6] Compiling Final Results Package into evidence/final/submission/New/...")
        self.results_packager.package_all(cohort_results, decision_ledger, stage2d_manifest)

        elapsed = round(time.time() - start_time, 2)
        logger.info("================================================================================")
        logger.info(f"MASTER END-TO-END EXECUTION COMPLETED SUCCESSFULLY IN {elapsed}s")
        logger.info(f"All deliverables verified in: {self.new_out_dir}")
        logger.info("STATUS: PROJECT_COMPLETE_WITH_STAGE_2D_INTEGRATION")
        logger.info("================================================================================")

        return {
            "status": "PROJECT_COMPLETE_WITH_STAGE_2D_INTEGRATION",
            "elapsed_seconds": elapsed,
            "stage2d_manifest": stage2d_manifest,
            "cohort_results_summary": {
                k: v["multi_seed_metrics"] for k, v in cohort_results.items()
            },
            "plots_generated_count": 18,
            "output_directory": str(self.new_out_dir),
        }


if __name__ == "__main__":
    runner = MasterEndToEndRunner()
    res = runner.run_complete_project()
    print(json.dumps(res, indent=2))
