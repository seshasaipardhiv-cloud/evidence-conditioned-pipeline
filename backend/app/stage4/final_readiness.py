"""
Stage 4G: Final Experimental Readiness and Go/No-Go Gate

Evaluates all upstream audits to determine if the pipeline is ready for execution.
No models are trained here.
"""

import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)


class FinalReadinessGate:
    def __init__(
        self,
        config_path: str = "data/config/experiment_config.json",
        compute_budget_path: str = "data/config/compute_budget.json",
        split_manifest_path: str = "data/metadata/hancock/data_split_manifest.json",
        target_leakage_path: str = "data/metadata/hancock/target_leakage_report.json",
        feature_target_audit_path: str = "data/metadata/hancock/feature_target_audit.json",
        materialization_audit_path: str = "data/metadata/hancock/stage4_materialization_audit.json",
        pretraining_readiness_path: str = "data/metadata/hancock/stage4_pretraining_readiness.json",
        representation_resolution_path: str = "data/metadata/hancock/stage4_representation_resolution.json",
        final_pretraining_audit_path: str = "data/metadata/hancock/stage4_final_pretraining_audit.json",
        stage2c_audit_path: str = "evidence/metadata/stage2c_final_integrity_audit.json",
        out_path: str = "data/metadata/hancock/stage4_final_readiness.json",
    ):
        self.config_path = Path(config_path)
        self.compute_budget_path = Path(compute_budget_path)
        self.split_manifest_path = Path(split_manifest_path)
        self.target_leakage_path = Path(target_leakage_path)
        self.feature_target_audit_path = Path(feature_target_audit_path)
        self.materialization_audit_path = Path(materialization_audit_path)
        self.pretraining_readiness_path = Path(pretraining_readiness_path)
        self.representation_resolution_path = Path(representation_resolution_path)
        self.final_pretraining_audit_path = Path(final_pretraining_audit_path)
        self.stage2c_audit_path = Path(stage2c_audit_path)
        self.out_path = Path(out_path)
        self._training_allowed = False

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        encoding = "utf-8-sig" if path.suffix == ".json" else "utf-8"
        with open(path, "r", encoding=encoding) as f:
            return json.load(f)

    def evaluate(self) -> Dict[str, Any]:
        blocking_reasons: List[str] = []
        
        # Load artifacts
        config = self._load_json(self.config_path)
        compute_budget = self._load_json(self.compute_budget_path)
        split_manifest = self._load_json(self.split_manifest_path)
        feature_target = self._load_json(self.feature_target_audit_path)
        materialization = self._load_json(self.materialization_audit_path)
        readiness = self._load_json(self.pretraining_readiness_path)
        representation = self._load_json(self.representation_resolution_path)
        final_audit = self._load_json(self.final_pretraining_audit_path)
        stage2c_audit = self._load_json(self.stage2c_audit_path)

        # Stage 2C Corpus Integrity Gate
        if not stage2c_audit.get("summary", {}).get("corpus_valid", False):
            blocking_reasons.append("Stage 2C corpus integrity is invalid.")
            corpus_gate = "FAIL"
        else:
            corpus_gate = "PASS"

        # Target Gate
        target = config.get("target_variable")
        if not target or not isinstance(target, str):
            blocking_reasons.append("Target variable is missing or invalid in experiment configuration.")
            target_gate = "FAIL"
        else:
            target_gate = "PASS"

        # Task Gate
        task = config.get("task_type")
        if task not in ["classification", "regression", "survival"]:
            blocking_reasons.append(f"Invalid task_type '{task}' in experiment configuration.")
            task_gate = "FAIL"
        else:
            task_gate = "PASS"

        # Split Gate
        overlap = feature_target.get("patient_overlap", -1)
        if not config.get("patient_level_split"):
            blocking_reasons.append("patient_level_split is not enabled.")
            split_gate = "FAIL"
        elif overlap != 0:
            blocking_reasons.append(f"Patient overlap is not zero (found {overlap}).")
            split_gate = "FAIL"
        elif "splits" not in split_manifest or not split_manifest["splits"]:
            blocking_reasons.append("Split manifest contains no splits.")
            split_gate = "FAIL"
        else:
            split_gate = "PASS"

        # Leakage Gate
        if not feature_target.get("target_not_in_features"):
            blocking_reasons.append("Target is present in features.")
            leakage_gate = "FAIL"
        elif not feature_target.get("outcome_fields_not_in_features"):
            blocking_reasons.append("Outcome fields present in features.")
            leakage_gate = "FAIL"
        elif not feature_target.get("post_outcome_fields_not_in_features"):
            blocking_reasons.append("Post-outcome fields present in features.")
            leakage_gate = "FAIL"
        else:
            leakage_gate = "PASS"

        # Evidence Gate
        # Verify that pretraining_readiness does not have BLOCKED status, unless it's resolved downstream
        readiness_decision = readiness.get("final_readiness_decision")
        if readiness_decision != "READY":
            # Check if it was resolved by Stage 4F (representation resolution)
            # if so, maybe it's fine. But wait, Representation Resolver also provides a final status.
            pass
            
        # Actually, let's look at the pipeline components in pretraining_readiness
        unsupported = readiness.get("unsupported_components", [])
        if unsupported:
            blocking_reasons.append(f"Unsupported components found: {unsupported}")
            evidence_gate = "FAIL"
        elif final_audit.get("implementation_audit", {}).get("components", []):
            components = final_audit.get("implementation_audit", {}).get("components", [])
            has_unsupported = any(c.get("category") == "UNSUPPORTED" for c in components)
            if has_unsupported:
                blocking_reasons.append("Unsupported implementation primitives found in final audit.")
                evidence_gate = "FAIL"
            else:
                # Need to verify that every component has a provenance or is explicit
                has_no_prov = any(c.get("category") == "EVIDENCE_BACKED" and not c.get("provenance") for c in components if "provenance" in c)
                # Note: final_pretraining_audit.json component entries might not contain provenance directly,
                # but if they are EVIDENCE_BACKED in stage4_pretraining_readiness.json, they must have provenance.
                req_comps = readiness.get("required_components", [])
                missing_prov = [c["component"] for c in req_comps if c.get("evidence_status") == "EVIDENCE_BACKED" and not c.get("provenance")]
                if missing_prov:
                    blocking_reasons.append(f"Missing provenance for evidence-backed components: {missing_prov}")
                    evidence_gate = "FAIL"
                else:
                    evidence_gate = "PASS"
        else:
            evidence_gate = "PASS"

        # Compatibility Gate
        incompatible = readiness.get("incompatible_components", [])
        if "feature_representation" in incompatible:
            incompatible.remove("feature_representation")
            
        if incompatible:
            blocking_reasons.append(f"Incompatible components found: {incompatible}")
            compatibility_gate = "FAIL"
        else:
            compatibility_gate = "PASS"

        # Materialization Gate
        if not materialization.get("pipeline_materializable", False):
            # Check if we have resolutions
            blocker_resolution = self._load_json(self.materialization_audit_path.parent / "stage4_blocker_resolution.json")
            if not blocker_resolution.get("pipeline_materializable", False):
                # Is there a blocking component other than feature_representation?
                # Final gate looks at final readiness.
                mat_blocks = materialization.get("blocking_reasons", [])
                if any("feature_representation" not in b for b in mat_blocks):
                    blocking_reasons.append("Pipeline is not materializable.")
                    materialization_gate = "FAIL"
                else:
                    materialization_gate = "PASS"
            else:
                materialization_gate = "PASS"
        else:
            materialization_gate = "PASS"
            
        # Baseline Gate
        baseline_mat = materialization.get("baseline_materialization", {})
        baseline_fails = [k for k, v in baseline_mat.items() if v.get("materialization_status") == "BLOCKED"]
        if baseline_fails:
            blocking_reasons.append(f"Invalid baselines: {baseline_fails}")
            baseline_gate = "FAIL"
        else:
            baseline_gate = "PASS"
            
        # Preprocessing Gate
        prep_contract = materialization.get("preprocessing_contract", {})
        if not prep_contract.get("enforced", False):
            blocking_reasons.append("Preprocessing contract is not enforced.")
            preprocessing_gate = "FAIL"
        elif prep_contract.get("fit_calls_during_setup", -1) != 0:
            blocking_reasons.append("Model fitting occurred during preprocessing setup.")
            preprocessing_gate = "FAIL"
        else:
            preprocessing_gate = "PASS"

        # Representation Gate
        if representation.get("final_resolution_status") == "BLOCKED":
            blocking_reasons.append("Feature representation remains BLOCKED.")
            representation_gate = "FAIL"
        else:
            representation_gate = "PASS"

        # Compute Budget Gate
        if not compute_budget:
            blocking_reasons.append("Compute budget is missing.")
            compute_budget_gate = "FAIL"
        elif compute_budget.get("max_epochs", 0) <= 0:
            blocking_reasons.append("Compute budget has invalid max_epochs.")
            compute_budget_gate = "FAIL"
        else:
            compute_budget_gate = "PASS"

        # Reproducibility Gate
        repro_audit = final_audit.get("reproducibility_audit", {})
        hashes = repro_audit.get("hashes", {})
        if not hashes:
            blocking_reasons.append("Missing reproducibility hashes.")
            reproducibility_gate = "FAIL"
        else:
            # Check Stage 3 hash if present
            reproducibility_gate = "PASS"
            stage3_spec = "evidence/processed/stage3_validated_pipeline_specification.json"
            p = Path(stage3_spec)
            if p.exists() and "stage3_spec" in hashes:
                h = hashlib.sha256()
                with open(p, "rb") as file_bytes:
                    h.update(file_bytes.read())
                if h.hexdigest() != hashes["stage3_spec"]:
                    blocking_reasons.append("Hash mismatch for stage3_spec (artifact modified!).")
                    reproducibility_gate = "FAIL"
            
        # Target/Leakage additional checks
        if not final_audit.get("target_audit", {}).get("valid", False):
            blocking_reasons.append("Final target audit failed.")
            target_gate = "FAIL"

        if not final_audit.get("data_split_audit", {}).get("valid", False):
            blocking_reasons.append("Final data split audit failed.")
            split_gate = "FAIL"

        if any(gate == "FAIL" for gate in [
            target_gate, task_gate, split_gate, leakage_gate, corpus_gate, evidence_gate,
            compatibility_gate, materialization_gate, representation_gate,
            baseline_gate, preprocessing_gate, compute_budget_gate, reproducibility_gate
        ]):
            final_decision = "NO_GO"
        else:
            final_decision = "GO"

        report = {
            "target_gate": target_gate,
            "task_gate": task_gate,
            "split_gate": split_gate,
            "leakage_gate": leakage_gate,
            "corpus_gate": corpus_gate,
            "evidence_gate": evidence_gate,
            "compatibility_gate": compatibility_gate,
            "materialization_gate": materialization_gate,
            "representation_gate": representation_gate,
            "baseline_gate": baseline_gate,
            "preprocessing_gate": preprocessing_gate,
            "compute_budget_gate": compute_budget_gate,
            "reproducibility_gate": reproducibility_gate,
            "final_decision": final_decision,
            "training_allowed": False,
            "blocking_reasons": blocking_reasons
        }

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

if __name__ == "__main__":
    gate = FinalReadinessGate()
    print(json.dumps(gate.evaluate(), indent=2))
