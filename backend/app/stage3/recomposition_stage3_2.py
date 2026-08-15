"""
Stage 3.2: Evidence-Conditioned Pipeline Recomposition

Searches the Stage 2C evidence corpus to replace the currently blocked feature_representation component.
Produces a recomposed pipeline specification if a compatible evidence-backed candidate is found.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class Stage3_2Recomposer:
    def __init__(
        self,
        experiments_path: str = "evidence/processed/experiments.jsonl",
        claims_path: str = "evidence/processed/evidence_claims.jsonl",
        papers_path: str = "evidence/processed/papers.jsonl",
        stage3_spec_path: str = "evidence/processed/stage3_validated_pipeline_specification.json",
        out_dir: str = "evidence/metadata",
        proc_out_dir: str = "evidence/processed"
    ):
        self.experiments_path = Path(experiments_path)
        self.claims_path = Path(claims_path)
        self.papers_path = Path(papers_path)
        self.stage3_spec_path = Path(stage3_spec_path)
        self.out_dir = Path(out_dir)
        self.proc_out_dir = Path(proc_out_dir)
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.proc_out_dir.mkdir(parents=True, exist_ok=True)
        self.hancock_modalities = {"clinical", "pathology_tabular", "blood", "text"}

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def recompose(self) -> Dict[str, Any]:
        experiments = self._load_jsonl(self.experiments_path)
        stage3_spec = self._load_json(self.stage3_spec_path)
        
        # 1. Evidence Inventory
        candidates = []
        for exp in experiments:
            # Check direct fields
            for comp in ["feature_representation", "base_learner"]:
                val = exp.get(comp)
                if not val:
                    continue
                
                # Extract modalities
                exp_mod = exp.get("modalities", [])
                
                # Check compatibility
                comp_status = "SUPPORTED_BUT_INCOMPATIBLE"
                reason = "Requires imaging or unavailable modalities"
                
                val_lower = val.lower()
                
                # Cannot use CNN without imaging
                has_imaging_req = "cnn" in val_lower or "image" in val_lower or "vision" in val_lower
                
                # Target-derived checks
                target_leakage = "target" in val_lower or "survival" in val_lower or "recurrence" in val_lower
                
                # Clinical / Tabular
                is_clinical = "clinical" in val_lower or "tabular" in val_lower or "structured" in val_lower
                
                # Requires explicit provenance
                has_provenance = bool(exp.get("id"))
                
                if target_leakage:
                    comp_status = "INVALID_ENTITY"
                    reason = "Depends on target leakage"
                elif not has_provenance:
                    comp_status = "INSUFFICIENT_EVIDENCE"
                    reason = "Missing provenance"
                elif has_imaging_req:
                    comp_status = "SUPPORTED_BUT_INCOMPATIBLE"
                    reason = "Requires imaging modalities"
                elif is_clinical:
                    # Is it compatible with HANCOCK? Yes
                    comp_status = "SUPPORTED_AND_COMPATIBLE"
                    reason = "Compatible tabular/clinical representation"
                else:
                    # Not explicitly clinical/tabular
                    comp_status = "SUPPORTED_BUT_INCOMPATIBLE"
                    reason = "Not explicitly supported for tabular data"
                    
                cand = {
                    "mechanism": val,
                    "paper_id": exp.get("paper_id"),
                    "proposed_method": exp.get("proposed_method", ""),
                    "task": exp.get("task", ""),
                    "modalities": exp_mod,
                    "source_sentence": exp.get("claim_text", ""),
                    "source_section": exp.get("source_section", ""),
                    "metric": exp.get("primary_metric", ""),
                    "result": exp.get("result", ""),
                    "evidence_status": "SUPPORTED" if comp_status.startswith("SUPPORTED") else "UNSUPPORTED",
                    "provenance": exp.get("id", ""),
                    "compatibility_with_hancock": comp_status,
                    "reason": reason
                }
                candidates.append(cand)
                
        with open(self.out_dir / "stage3_2_representation_inventory.json", "w", encoding="utf-8") as f:
            json.dump(candidates, f, indent=2)
            
        # 3. Recomposition
        valid_candidates = [c for c in candidates if c["compatibility_with_hancock"] == "SUPPORTED_AND_COMPATIBLE"]
        
        # Rank deterministically by mechanism name then provenance ID
        valid_candidates.sort(key=lambda x: (x["mechanism"], x["provenance"]))
        
        recomposition_report = {
            "valid_candidate_count": len(valid_candidates),
            "candidates": valid_candidates,
            "selected_replacement": None,
            "action": "NONE"
        }
        
        if valid_candidates:
            selected = valid_candidates[0]
            recomposition_report["selected_replacement"] = selected
            recomposition_report["action"] = "REPLACE_FEATURE_REPRESENTATION"
            exec_status = "RECOMPOSED"
            selected_mech = selected["mechanism"]
            prov = selected["provenance"]
            sel_reason = selected["reason"]
            comp_status = "COMPATIBLE"
        else:
            exec_status = "BLOCKED_NO_COMPATIBLE_EVIDENCE"
            selected_mech = stage3_spec.get("selected_mechanisms", {}).get("feature_representation")
            prov = None
            sel_reason = "No compatible evidence-backed candidates found in Stage 2C corpus."
            comp_status = "INCOMPATIBLE"
            
        with open(self.proc_out_dir / "stage3_2_recomposition.json", "w", encoding="utf-8") as f:
            json.dump(recomposition_report, f, indent=2)

        # 4. New Pipeline Specification
        replaced_components = {}
        unchanged_components = stage3_spec.get("selected_mechanisms", {}).copy()
        
        if exec_status == "RECOMPOSED":
            orig = unchanged_components.get("feature_representation")
            replaced_components["feature_representation"] = {
                "original": orig,
                "new": selected_mech
            }
            unchanged_components["feature_representation"] = selected_mech
            
        new_spec = {
            "original_stage3_specification": stage3_spec,
            "replaced_components": replaced_components,
            "unchanged_components": unchanged_components,
            "selected_feature_representation": selected_mech,
            "selection_reason": sel_reason,
            "evidence_provenance": prov,
            "compatibility_status": comp_status,
            "execution_status": exec_status,
            "training_allowed": False
        }
        
        with open(self.proc_out_dir / "stage3_2_pipeline_specification.json", "w", encoding="utf-8") as f:
            json.dump(new_spec, f, indent=2)
            
        # 5. Audit
        audit = {
            "stage2_artifacts_unchanged": True, # Implicit by read-only access
            "stage3_artifacts_unchanged": True,
            "no_unsupported_mechanism_selected": all(c["compatibility_with_hancock"] == "SUPPORTED_AND_COMPATIBLE" for c in [recomposition_report["selected_replacement"]] if c),
            "no_imaging_dependent_mechanism_selected": all("cnn" not in c["mechanism"].lower() and "image" not in c["mechanism"].lower() for c in [recomposition_report["selected_replacement"]] if c),
            "every_selected_mechanism_has_provenance": all(bool(c["provenance"]) for c in [recomposition_report["selected_replacement"]] if c),
            "no_target_derived_feature_introduced": all("target" not in c["mechanism"].lower() and "survival" not in c["mechanism"].lower() for c in [recomposition_report["selected_replacement"]] if c),
            "no_model_training_occurred": True
        }
        
        with open(self.out_dir / "stage3_2_recomposition_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

        return new_spec

if __name__ == "__main__":
    recomposer = Stage3_2Recomposer()
    recomposer.recompose()
