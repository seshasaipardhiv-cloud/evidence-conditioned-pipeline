import json
from pathlib import Path
from typing import Dict, Any, List

class FeatureRepresentationResolutionAuditor:
    def __init__(
        self,
        mechanisms_path: str,
        impl_config_path: str,
        readiness_path: str,
        experiments_path: str,
        out_path: str = "data/metadata/hancock/stage4_feature_representation_resolution.json"
    ):
        self.mechanisms_path = Path(mechanisms_path)
        self.impl_config_path = Path(impl_config_path)
        self.readiness_path = Path(readiness_path)
        self.experiments_path = Path(experiments_path)
        self.out_path = Path(out_path)

    def audit(self) -> Dict[str, Any]:
        # Read implementation config
        impl_config = {}
        if self.impl_config_path.exists():
            with open(self.impl_config_path, "r", encoding="utf-8") as f:
                impl_config = json.load(f)

        # Read original readiness gate output to find the feature_representation status
        original_readiness = {}
        original_status = "UNAVAILABLE"
        original_selected = "cnn_representation"
        if self.readiness_path.exists():
            with open(self.readiness_path, "r", encoding="utf-8") as f:
                original_readiness = json.load(f)
                for c in original_readiness.get("required_components", []):
                    if c["component"] == "feature_representation":
                        original_selected = c.get("selected_value", "cnn_representation")
                        original_status = c.get("compatibility_status", "incompatible")

        # Parse mechanisms for Category = Representation
        evidence_backed_candidates = []
        if self.mechanisms_path.exists():
            with open(self.mechanisms_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    mech = json.loads(line)
                    if mech.get("category") == "Representation":
                        evidence_backed_candidates.append(mech.get("canonical_name"))
        
        # Determine candidate statuses
        candidates = []
        explicitly_configured = []
        incompatible = []
        unresolved = []
        evidence_backed = []

        # 1. Check explicit configuration
        explicit_rep = impl_config.get("feature_representation")
        if explicit_rep:
            # explicit config provided
            explicitly_configured.append(explicit_rep)
            candidates.append({
                "mechanism": explicit_rep,
                "evidence_status": "EXPLICITLY_CONFIGURED",
                "compatibility_status": "valid",
                "confidence_status": "explicit_configuration"
            })
        
        # 2. Check evidence-backed candidates
        for c in evidence_backed_candidates:
            if c == "cnn" or "cnn" in c.lower():
                # CNN requires imaging modality which is missing in Hancock
                incompatible.append(c)
                candidates.append({
                    "mechanism": c,
                    "evidence_status": "INCOMPATIBLE",
                    "compatibility_status": "incompatible",
                    "reason": "HANCOCK does not provide a validated imaging modality"
                })
            else:
                # Any other evidence backed representation (none currently exist for tabular)
                evidence_backed.append(c)
                candidates.append({
                    "mechanism": c,
                    "evidence_status": "EVIDENCE_BACKED",
                    "compatibility_status": "valid",
                    "confidence_status": "explicit"
                })

        # Evaluate final resolution
        final_selected = None
        selection_reason = None
        provenance = None
        final_resolution_status = "BLOCKED"

        if explicitly_configured:
            final_selected = explicitly_configured[0]
            selection_reason = "Explicitly configured by user."
            provenance = "explicit_configuration"
            final_resolution_status = "RESOLVED"
        elif evidence_backed:
            final_selected = evidence_backed[0]
            selection_reason = "Evidence-backed representation compatible with data."
            provenance = "stage2_corpus"
            final_resolution_status = "RESOLVED"
        else:
            selection_reason = "No explicitly configured or evidence-backed compatible representation found."
            final_resolution_status = "BLOCKED"
            unresolved.extend([c for c in evidence_backed_candidates if c not in incompatible])

        report = {
            "original_selected_representation": original_selected,
            "original_compatibility_status": original_status,
            "candidate_representations": candidates,
            "evidence_backed_candidates": evidence_backed,
            "explicitly_configured_candidates": explicitly_configured,
            "incompatible_candidates": incompatible,
            "unresolved_candidates": unresolved,
            "selected_replacement": final_selected,
            "selection_reason": selection_reason,
            "provenance": provenance,
            "final_resolution_status": final_resolution_status
        }

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

if __name__ == "__main__":
    auditor = FeatureRepresentationResolutionAuditor(
        "evidence/processed/mechanisms.jsonl",
        "data/config/implementation_config.json",
        "data/metadata/hancock/stage4_pretraining_readiness.json",
        "evidence/processed/experiments.jsonl"
    )
    auditor.audit()
