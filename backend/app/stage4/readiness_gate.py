import json
from pathlib import Path
from typing import Dict, Any, List

class ReadinessGate:
    def __init__(
        self,
        resolution_path: str,
        materialization_path: str,
        execution_gate_path: str,
        spec_path: str,
        rankings_path: str,
        config_path: str,
        impl_config_audit_path: str = None,
        feat_rep_res_path: str = "data/metadata/hancock/stage4_feature_representation_resolution.json",
        out_path: str = "data/metadata/hancock/stage4_pretraining_readiness.json"
    ):
        with open(resolution_path, "r", encoding="utf-8") as f:
            self.resolution = json.load(f)
        with open(materialization_path, "r", encoding="utf-8") as f:
            self.materialization = json.load(f)
        with open(execution_gate_path, "r", encoding="utf-8") as f:
            self.execution_gate = json.load(f)
        with open(spec_path, "r", encoding="utf-8") as f:
            self.spec = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.out_path = Path(out_path)
        self.impl_config_audit = self._load_json(impl_config_audit_path)
        
        self.feat_rep_res_path = Path(feat_rep_res_path)
        self.feat_rep_res = self._load_json(str(self.feat_rep_res_path))

        self.IMPLEMENTATION_PRIMITIVES = [
            "missing_value_handling",
            "categorical_encoding",
            "base_learner",
            "loss_function",
            "imbalance_handling"
        ]

        self.EVIDENCE_CONDITIONED = [
            "modality_fusion",
            "feature_representation",
            "ensembling"
        ]

    def _load_json(self, path: str) -> Dict[str, Any]:
        if path and Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
        
    def check_readiness(self) -> Dict[str, Any]:
        required_components = []
        evidence_backed = []
        explicitly_configured = []
        unsupported = []
        incompatible = []
        primitives_requiring_config = []
        
        # Merge knowledge from materialization, resolution, and impl config
        mat_comps = {c["component"]: c for c in self.materialization.get("components", [])}
        res_comps = {c["component"]: c for c in self.resolution.get("resolutions", [])}
        impl_comps = {c["component"]: c for c in self.impl_config_audit.get("configured_primitives", [])}
        
        has_blocked_components = False
        
        # Check components
        for comp in self.IMPLEMENTATION_PRIMITIVES + self.EVIDENCE_CONDITIONED:
            cat = "implementation_primitive" if comp in self.IMPLEMENTATION_PRIMITIVES else "evidence_conditioned"
            
            # Extract resolution status if it was blocked initially
            res_info = res_comps.get(comp)
            mat_info = mat_comps.get(comp)
            
            val = None
            ev_status = "unsupported"
            comp_status = "valid"
            prov = None
            exec_status = "BLOCKED"
            reason = ""
            
            if res_info:
                val = self.spec.get("selected_mechanisms", {}).get(comp)
                if res_info["resolved"]:
                    ev_status = "EVIDENCE_BACKED"
                    comp_status = "valid"
                    prov = res_info["provenance"]
                    exec_status = "EXECUTABLE"
                    reason = "Resolved via explicit evidence."
                else:
                    # Check if it was explicitly configured
                    impl_info = impl_comps.get(comp)
                    if impl_info and impl_info["selected_value"] is not None:
                        val = impl_info["selected_value"]
                        if impl_info["compatibility_status"] == "valid":
                            ev_status = "EXPLICITLY_CONFIGURED"
                            comp_status = "valid"
                            prov = impl_info["configuration_source"]
                            exec_status = "EXECUTABLE"
                            reason = impl_info["rationale"]
                        else:
                            ev_status = "UNSUPPORTED"
                            comp_status = "incompatible"
                            prov = impl_info["configuration_source"]
                            exec_status = "BLOCKED"
                            reason = impl_info["rationale"]
                    else:
                        ev_status = "UNSUPPORTED"
                        exec_status = "BLOCKED"
                        reason = res_info["provenance"]
                        if cat == "implementation_primitive":
                            primitives_requiring_config.append(comp)
            
            # Feature representation resolution override
            if comp == "feature_representation" and self.feat_rep_res and self.feat_rep_res.get("final_resolution_status") == "RESOLVED":
                val = self.feat_rep_res.get("selected_replacement")
                ev_status = "EXPLICITLY_CONFIGURED" if self.feat_rep_res.get("provenance") == "explicit_configuration" else "EVIDENCE_BACKED"
                comp_status = "valid"
                exec_status = "EXECUTABLE"
                reason = self.feat_rep_res.get("selection_reason")
                prov = self.feat_rep_res.get("provenance")
            elif not res_info and mat_info:
                val = mat_info["selected_mechanism"]
                if mat_info["materialization_status"] == "SUPPORTED":
                    ev_status = "EVIDENCE_BACKED"
                    comp_status = "valid"
                    # For pre-resolved components in stage 3, just assume provenance from stage 3 spec
                    prov = "Inherited from Stage 3 validation."
                    exec_status = "EXECUTABLE"
                    reason = "Materialization successful."
                else:
                    ev_status = "UNSUPPORTED"
                    exec_status = "BLOCKED"
                    reason = "Materialization blocked."
            
            # Check compatibility explicitly from materializer only if it hasn't been explicitly configured as valid
            if mat_info and mat_info.get("dataset_compatibility") is False:
                if ev_status != "EXPLICITLY_CONFIGURED":
                    comp_status = "incompatible"
                    exec_status = "BLOCKED"
                    reason = "Dataset incompatibility (e.g. missing modality)."
                
            req_c = {
                "component": comp,
                "category": cat,
                "selected_value": val,
                "evidence_status": ev_status,
                "compatibility_status": comp_status,
                "provenance": prov,
                "execution_status": exec_status,
                "reason": reason
            }
            required_components.append(req_c)
            
            if ev_status == "EVIDENCE_BACKED":
                evidence_backed.append(comp)
            elif ev_status == "EXPLICITLY_CONFIGURED":
                explicitly_configured.append(comp)
            else:
                unsupported.append(comp)
                
            if comp_status == "incompatible":
                incompatible.append(comp)
                
            if exec_status == "BLOCKED":
                has_blocked_components = True

        # Validation blocks
        target_valid = self.config.get("target_variable") == "recurrence" and self.config.get("task_type") == "classification"
        leakage_valid = self.materialization.get("target_firewall", {}).get("enforced", False)
        split_valid = self.execution_gate.get("split_valid", False)
        materialization_valid = not has_blocked_components
        
        # Baselines
        b_mat = self.materialization.get("baseline_materialization", {})
        baseline_valid = True
        for b_name, b_info in b_mat.items():
            if b_info.get("materialization_status") == "BLOCKED":
                baseline_valid = False
                break
                
        # Final Readiness Decision
        if not target_valid or not split_valid:
            decision = "BLOCKED_CONFIGURATION"
        elif any(c["compatibility_status"] == "incompatible" for c in required_components):
            decision = "BLOCKED_COMPATIBILITY"
        elif has_blocked_components or not baseline_valid:
            decision = "BLOCKED_MISSING_EVIDENCE"
        else:
            decision = "READY_FOR_TRAINING"
            
        # Hard override for current state: missing evidence requires a BLOCKED_MISSING_EVIDENCE
        if len(unsupported) > 0:
            decision = "BLOCKED_MISSING_EVIDENCE"
            
        report = {
            "candidate_pipeline_status": decision,
            "required_components": required_components,
            "evidence_backed_components": evidence_backed,
            "explicitly_configured_components": explicitly_configured,
            "unsupported_components": unsupported,
            "incompatible_components": incompatible,
            "implementation_primitives_requiring_explicit_configuration": primitives_requiring_config,
            "target_task_validation": target_valid,
            "leakage_validation": leakage_valid,
            "split_validation": split_valid,
            "materialization_validation": materialization_valid,
            "baseline_validation": baseline_valid,
            "final_readiness_decision": decision,
            "training_allowed": False # Hard rule from requirements
        }
        
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        return report
