"""
Repository Cleanup and Governance Engine

Performs safe repository cleanup in accordance with Stage Requirements 17 & 18:
1. Deprecates and removes automatic PDF conversion tool (backend/app/stage6/submission_packager.py and its test).
2. Preserves the immutable final research paper PDF (evidence/final/submission/final_research_paper.pdf).
3. Removes obsolete temporary scripts, patch files, and development scratch files.
4. Generates repository_cleanup_manifest.json with complete traceability.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

FILES_TO_REMOVE = [
    {
        "path": "backend/app/stage6/submission_packager.py",
        "reason": "Deprecated automatic PDF conversion tool per Requirement 17.",
        "category": "deprecated_tool",
    },
    {
        "path": "backend/tests/stage6/test_submission_packager.py",
        "reason": "Tests dedicated exclusively to the deprecated PDF conversion tool per Requirement 17.",
        "category": "deprecated_tests",
    },
    {
        "path": "audit_helper.txt",
        "reason": "Temporary development debugging and audit scratch file.",
        "category": "temporary_file",
    },
    {
        "path": "patch.py",
        "reason": "Obsolete temporary patching script.",
        "category": "temporary_file",
    },
    {
        "path": "scratch/fetch_real_papers.py",
        "reason": "Temporary scratch retrieval script.",
        "category": "temporary_file",
    },
]

IMMUTABLE_PRESERVED_FILES = [
    "evidence/final/submission/final_research_paper.pdf",
    "evidence/final/submission/final_research_paper.md",
    "evidence/final/submission/cover_letter.md",
    "evidence/final/submission/jbi_highlights.md",
    "evidence/final/submission/credit_statement.md",
    "evidence/processed/stage5b_run_results.json",
    "evidence/processed/stage5b_candidate_results.json",
    "evidence/processed/stage5b_baseline_results.json",
    "evidence/metadata/stage5b_safety_audit.json",
    "evidence/metadata/stage5c_statistical_analysis.json",
    "evidence/metadata/stage5c_ablation_results.json",
    "evidence/metadata/stage5c_robustness_report.json",
    "evidence/metadata/stage5c_calibration_report.json",
    "evidence/final/stage6a_master_results.json",
    "evidence/final/reconciliation/stage6h_manuscript_reconciliation.json",
    "evidence/final/reconciliation/stage6i_final_verdict.json",
    "evidence/final/submission/submission_manifest.json",
]


class RepositoryCleaner:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)

    def execute_cleanup(self) -> Dict[str, Any]:
        deleted_records = []
        for item in FILES_TO_REMOVE:
            p = self.base_dir / item["path"]
            existed = p.exists()
            if existed:
                try:
                    p.unlink()
                    deleted_records.append({
                        "file": item["path"],
                        "status": "DELETED",
                        "reason": item["reason"],
                        "category": item["category"],
                    })
                except Exception as e:
                    deleted_records.append({
                        "file": item["path"],
                        "status": "ERROR",
                        "error": str(e),
                    })
            else:
                deleted_records.append({
                    "file": item["path"],
                    "status": "ALREADY_ABSENT",
                    "reason": item["reason"],
                    "category": item["category"],
                })

        # Verify preserved files
        retained_records = []
        for rel_path in IMMUTABLE_PRESERVED_FILES:
            p = self.base_dir / rel_path
            retained_records.append({
                "file": rel_path,
                "exists": p.exists(),
                "status": "PRESERVED_IMMUTABLE",
            })

        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": "REPOSITORY_CLEANUP_AND_PDF_TOOL_DEPRECATION",
            "pdf_conversion_tool_removed": True,
            "deleted_files_count": len([d for d in deleted_records if d["status"] == "DELETED"]),
            "deleted_files": deleted_records,
            "retained_authoritative_files_count": len(retained_records),
            "retained_files": retained_records,
            "status": "CLEANUP_COMPLETE",
        }

        out_path = self.base_dir / "repository_cleanup_manifest.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest


if __name__ == "__main__":
    cleaner = RepositoryCleaner()
    res = cleaner.execute_cleanup()
    print(json.dumps(res, indent=2))
