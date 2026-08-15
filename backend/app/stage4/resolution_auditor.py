import json
from pathlib import Path
from typing import Dict, Any, List

class ResolutionAuditor:
    def __init__(
        self,
        mat_audit_path: str,
        mat_manifest_path: str,
        claims_path: str,
        experiments_path: str
    ):
        with open(mat_audit_path, "r", encoding="utf-8") as f:
            self.mat_audit = json.load(f)
        with open(mat_manifest_path, "r", encoding="utf-8") as f:
            self.mat_manifest = json.load(f)
            
        self.claims = []
        if Path(claims_path).exists():
            with open(claims_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.claims.append(json.loads(line))
                        
    def audit_resolution(self) -> Dict[str, Any]:
        resolutions = []
        
        for comp in self.mat_audit.get("components", []):
            status = comp.get("materialization_status")
            c_name = comp.get("component")
            
            if status in ["BLOCKED", "INCOMPATIBLE", "INSUFFICIENT_EVIDENCE"]:
                # Check for evidence
                evidence = self._find_evidence(c_name)
                
                if evidence:
                    # In this theoretical branch, we resolve it.
                    # However, Stage 2 actually has no evidence for these.
                    resolutions.append({
                        "component": c_name,
                        "original_status": status,
                        "resolved": True,
                        "new_status": "SUPPORTED",
                        "implementation_mapping": f"backend.models.{c_name}.Resolved{c_name}",
                        "provenance": evidence
                    })
                else:
                    resolutions.append({
                        "component": c_name,
                        "original_status": status,
                        "resolved": False,
                        "new_status": status, # Preserve the blocker
                        "implementation_mapping": None,
                        "provenance": "No explicit support found in Stage 2 evidence."
                    })
                    
        resolution_report = {
            "resolutions": resolutions,
            "pipeline_materializable": self.mat_audit["pipeline_materializable"],
            "execution_status": "CONFIGURATION_VALIDATED",
            "training_allowed": False
        }
        
        # Determine if any remain blocked
        all_resolved = all(r["resolved"] for r in resolutions)
        if not all_resolved or not self.mat_audit["pipeline_materializable"]:
            resolution_report["pipeline_materializable"] = False
            
        out_dir = Path("data/metadata/hancock")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stage4_blocker_resolution.json", "w", encoding="utf-8") as f:
            json.dump(resolution_report, f, indent=2)
            
        return resolution_report
        
    def _find_evidence(self, component_name: str) -> Dict[str, Any]:
        # Search claims for anything matching this component name
        for claim in self.claims:
            if claim.get("mechanism_id") == component_name and claim.get("direction") == "positive":
                return {
                    "paper_id": claim.get("paper_id"),
                    "claim_id": claim.get("claim_id", "unknown"),
                    "experiment_id": claim.get("experiment_id")
                }
        return {}
