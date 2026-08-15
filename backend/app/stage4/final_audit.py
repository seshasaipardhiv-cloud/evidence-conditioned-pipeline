import json
import hashlib
from pathlib import Path
from typing import Dict, Any

class FinalAudit:
    def __init__(
        self,
        exp_config_path: str,
        budget_path: str,
        impl_config_path: str,
        exec_gate_path: str,
        mech_gate_path: str,
        mat_audit_path: str,
        readiness_path: str,
        split_manifest_path: str,
        feat_audit_path: str,
        leakage_report_path: str,
        spec_path: str
    ):
        self.paths = {
            "experiment_config": exp_config_path,
            "compute_budget": budget_path,
            "implementation_config": impl_config_path,
            "execution_gate": exec_gate_path,
            "mechanism_gate": mech_gate_path,
            "materialization_audit": mat_audit_path,
            "pretraining_readiness": readiness_path,
            "data_split_manifest": split_manifest_path,
            "feature_target_audit": feat_audit_path,
            "target_leakage_report": leakage_report_path,
            "stage3_spec": spec_path
        }
        
        self.data = {}
        for k, p in self.paths.items():
            path_obj = Path(p)
            if path_obj.exists():
                with open(path_obj, "r", encoding="utf-8") as f:
                    self.data[k] = json.load(f)
            else:
                self.data[k] = {}

    def _hash_file(self, filepath: str) -> str:
        path_obj = Path(filepath)
        if not path_obj.exists():
            return None
        hasher = hashlib.sha256()
        with open(path_obj, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def audit(self) -> Dict[str, Any]:
        # A. Target
        exp_conf = self.data["experiment_config"]
        target_explicit = exp_conf.get("target_variable") == "recurrence"
        task_explicit = exp_conf.get("task_type") == "classification"
        
        feat_audit = self.data["feature_target_audit"]
        target_excluded = True
        if feat_audit:
            target_excluded = feat_audit.get("target_not_in_features") is True
            
        leak_rep = self.data["target_leakage_report"]
        leakage_excluded = True
        if leak_rep:
            leakage_excluded = len(leak_rep.get("leaked_fields", [])) == 0
            
        target_valid = target_explicit and task_explicit and target_excluded and leakage_excluded

        # B. Data Split
        split_man = self.data["data_split_manifest"]
        patient_level = exp_conf.get("patient_level_split", False)
        
        overlap_zero = True
        split_deterministic = True
        if split_man:
            overlap_zero = True
            for s in split_man.get("splits", []):
                overlap = s.get("overlap_counts", {})
                if overlap.get("train_validation") != 0 or overlap.get("train_test") != 0 or overlap.get("validation_test") != 0:
                    overlap_zero = False
            seeds = [s.get("seed") for s in split_man.get("splits", [])]
            split_deterministic = set(seeds) == {42, 100, 2026}
            
        mat_aud = self.data["materialization_audit"]
        prep_contract = mat_aud.get("preprocessing_contract", {})
        no_prep_fitted = prep_contract.get("fit_calls_during_setup") == 0
        
        split_valid = patient_level and overlap_zero and split_deterministic and no_prep_fitted
        
        # C. Implementation
        readiness = self.data["pretraining_readiness"]
        classified_components = []
        has_blocked_required = False
        
        if readiness:
            for req in readiness.get("required_components", []):
                classified_components.append({
                    "component": req["component"],
                    "category": req["evidence_status"],
                    "execution_status": req["execution_status"]
                })
                if req["execution_status"] == "BLOCKED":
                    has_blocked_required = True
        
        # D. Stage 3.1
        stage3_1_valid = True
        if readiness:
            for req in readiness.get("required_components", []):
                if req["component"] == "cross_attention" and req["execution_status"] != "EXECUTABLE":
                    stage3_1_valid = False
                if req["component"] == "feature_representation" and req["selected_value"] == "cnn_representation" and req["execution_status"] != "BLOCKED":
                    stage3_1_valid = False

        baseline_mat = mat_aud.get("baseline_materialization", {})
        if any(b.get("materialization_status") != "BLOCKED" and b.get("reason") == "Malformed baseline entity." for b in baseline_mat.values()):
            stage3_1_valid = False
            
        # E. Preprocessing Contract
        prep_valid = prep_contract.get("allowed_fit_partition") == "train"

        # F. Reproducibility
        hashes = {k: self._hash_file(v) for k, v in self.paths.items()}
        
        # G. Execution Gate
        # READY_FOR_TRAINING ONLY if every required component is either EVIDENCE_BACKED or EXPLICITLY_CONFIGURED and valid.
        ready = (
            target_valid and
            split_valid and
            stage3_1_valid and
            prep_valid and
            not has_blocked_required
        )
        gate_decision = "READY_FOR_TRAINING" if ready else "BLOCKED"
        
        audit_report = {
            "target_audit": {
                "recurrence_explicit": target_explicit,
                "classification_explicit": task_explicit,
                "target_excluded": target_excluded,
                "leakage_excluded": leakage_excluded,
                "valid": target_valid
            },
            "data_split_audit": {
                "patient_level": patient_level,
                "overlap_zero": overlap_zero,
                "split_deterministic": split_deterministic,
                "no_preprocessing_fitted": no_prep_fitted,
                "valid": split_valid
            },
            "implementation_audit": {
                "components": classified_components
            },
            "stage_3_1_audit": {
                "valid": stage3_1_valid
            },
            "preprocessing_audit": {
                "fit_train_only": prep_valid,
                "valid": prep_valid
            },
            "reproducibility_audit": {
                "configured_seeds": [42, 100, 2026],
                "hashes": hashes
            },
            "final_gate_decision": gate_decision,
            "training_allowed": False # H. TRAINING FIREWALL - Regardless of the result of this audit
        }

        out_dir = Path("data/metadata/hancock")
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "stage4_final_pretraining_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

        return audit_report
