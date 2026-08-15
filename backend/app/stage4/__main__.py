import argparse
import json
import logging
from pathlib import Path
import hashlib

from backend.app.stage4.models import ExperimentConfig, ComputeBudget, DataSplitManifest
from backend.app.stage4.validator import Stage4Validator
from backend.app.stage4.splitter import PatientSplitter
from backend.app.stage4.feature_auditor import FeatureAuditor
from backend.app.stage4.materializer import Materializer
from backend.app.stage4.resolution_auditor import ResolutionAuditor
from backend.app.stage4.implementation_auditor import ImplementationAuditor
from backend.app.stage4.readiness_gate import ReadinessGate
from backend.app.stage4.feature_representation_resolution import FeatureRepresentationResolutionAuditor
from backend.app.stage4.representation_resolver import RepresentationResolver
from backend.app.stage4.final_audit import FinalAudit
from backend.app.stage4.final_readiness import FinalReadinessGate
from backend.app.stage4.blocker_resolver_stage4h import Stage4HBlockerResolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode without training models.")
    args = parser.parse_args()
    
    if not args.dry_run:
        logger.error("Stage 4A only allows --dry-run mode.")
        raise RuntimeError("Stage 4A only allows --dry-run mode.")

    logger.info("Starting Stage 4A Dry-Run")
    
    config_path = Path("data/config/experiment_config.json")
    budget_path = Path("data/config/compute_budget.json")
    stage3_spec_path = Path("evidence/processed/stage3_validated_pipeline_specification.json")
    data_dir = Path("data")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = ExperimentConfig.model_validate_json(f.read())
        
    with open(budget_path, "r", encoding="utf-8") as f:
        budget = ComputeBudget.model_validate_json(f.read())
        
    with open(stage3_spec_path, "r", encoding="utf-8") as f:
        stage3_spec = json.load(f)
        
    # Validation
    validator = Stage4Validator(config, budget, stage3_spec, {})
    gate = validator.run_all_gates()
    
    _, leakage_report, _ = validator.validate_leakage()
    _, mech_gate, _ = validator.validate_mechanisms()
    
    meta_out = Path("data/metadata/hancock")
    meta_out.mkdir(parents=True, exist_ok=True)
    
    gate_out = meta_out / "stage4_execution_gate.json"
    with open(gate_out, "w", encoding="utf-8") as f:
        f.write(gate.model_dump_json(indent=2))
        
    with open(meta_out / "target_leakage_report.json", "w", encoding="utf-8") as f:
        f.write(leakage_report.model_dump_json(indent=2))
        
    with open(meta_out / "stage4_mechanism_gate.json", "w", encoding="utf-8") as f:
        f.write(mech_gate.model_dump_json(indent=2))
        
    # Split Manifest
    manifest = DataSplitManifest()
    manifest.total_patients = 763
    manifest.excluded_missing_target = 0
    manifest.eligible_patients = 763
    manifest.train_validation_overlap = 0
    manifest.train_test_overlap = 0
    manifest.validation_test_overlap = 0
    
    if config.test_size and config.validation_size:
        manifest.test_count = int(763 * config.test_size)
        manifest.validation_count = int(763 * config.validation_size)
        manifest.train_count = 763 - manifest.test_count - manifest.validation_count
        manifest.split_method = "patient_level"
        manifest.split_seed = config.random_seeds[0] if config.random_seeds else 42
        manifest.split_hash = "mock_hash_for_dry_run"
        manifest.task_type = config.task_type
        manifest.stratification_policy = config.stratification_policy
    
    with open(meta_out / "data_split_manifest.json", "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))
        
    logger.info(f"Execution gate written to {gate_out}")
        
    if gate.execution_status.value == "CONFIGURATION_VALIDATED":
        logger.info("Configuration validated. Running Patient Splitter...")
        clinical_path = "data/raw/hancock/structured/StructuredData/clinical_data.json"
        splitter = PatientSplitter(str(config_path), clinical_path)
        manifest = splitter.run_splits()
        logger.info("Splits generated successfully.")
        
        logger.info("Running Feature/Target Auditor...")
        auditor = FeatureAuditor(str(config_path), clinical_path)
        audit = auditor.audit()
        logger.info("Feature/Target audit generated successfully.")
        
        logger.info("Running Pipeline Materialization Auditor...")
        materializer = Materializer(
            str(stage3_spec_path),
            "evidence/metadata/stage3_compatibility_audit.json",
            clinical_path
        )
        mat_audit, mat_manifest = materializer.audit_materialization()
        logger.info("Materialization audit completed successfully.")
        
        logger.info("Running Blocker Resolution Auditor...")
        resolution_auditor = ResolutionAuditor(
            str(meta_out / "stage4_materialization_audit.json"),
            str(meta_out / "stage4_materialization_manifest.json"),
            "evidence/processed/evidence_claims.jsonl",
            "evidence/processed/experiments.jsonl"
        )
        resolution_report = resolution_auditor.audit_resolution()
        logger.info("Blocker Resolution audit completed successfully.")
        
        logger.info("Running Implementation Auditor...")
        impl_auditor = ImplementationAuditor(
            "data/config/implementation_config.json",
            str(meta_out / "stage4_blocker_resolution.json"),
            clinical_path,
            str(config_path)
        )
        impl_audit_report = impl_auditor.audit()
        logger.info("Implementation Config audit completed successfully.")
        
        logger.info("Running Final Pre-Training Readiness Gate...")
        readiness_gate = ReadinessGate(
            str(meta_out / "stage4_blocker_resolution.json"),
            str(meta_out / "stage4_materialization_audit.json"),
            str(meta_out / "stage4_execution_gate.json"),
            str(stage3_spec_path),
            "evidence/processed/stage3_mechanism_rankings.json",
            str(config_path),
            str(meta_out / "stage4_implementation_config_audit.json")
        )
        readiness_report = readiness_gate.check_readiness()
        logger.info(f"Final Readiness Decision: {readiness_report['final_readiness_decision']}")
        
        logger.info("Running Feature Representation Resolution Auditor (Stage 4E)...")
        feat_rep_auditor = FeatureRepresentationResolutionAuditor(
            "evidence/processed/mechanisms.jsonl",
            "data/config/implementation_config.json",
            str(meta_out / "stage4_pretraining_readiness.json"),
            "evidence/processed/experiments.jsonl"
        )
        feat_rep_report = feat_rep_auditor.audit()
        
        logger.info("Running Evidence-Conditioned Representation Resolver (Stage 4F)...")
        rep_resolver = RepresentationResolver(
            stage2_experiments_path="evidence/processed/experiments.jsonl",
            stage2_mechanisms_path="evidence/processed/mechanisms.jsonl",
            stage3_spec_path=str(stage3_spec_path),
            stage3_rankings_path="evidence/processed/stage3_mechanism_rankings.json",
            stage1_profile_path=str(meta_out / "stage1_profile_report.json"),
            existing_resolution_path=str(meta_out / "stage4_feature_representation_resolution.json"),
            impl_config_path="data/config/implementation_config.json",
            out_path=str(meta_out / "stage4_representation_resolution.json"),
        )
        rep_res_report = rep_resolver.resolve()
        logger.info(
            "Stage 4F resolution: status=%s | replacement=%s | training_allowed=%s",
            rep_res_report.get("final_resolution_status"),
            rep_res_report.get("selected_replacement"),
            rep_res_report.get("training_allowed"),
        )
        
        # Re-run readiness gate only if Stage 4F resolved the blocker
        resolved_statuses = ("RESOLVED_EVIDENCE_BACKED", "RESOLVED_EXPLICIT")
        if rep_res_report.get("final_resolution_status") in resolved_statuses:
            logger.info("Stage 4F resolved the feature_representation blocker — re-running readiness gate.")
            readiness_report = readiness_gate.check_readiness()
        
        logger.info("Running Final Pre-Training Gate and Reproducibility Audit...")
        final_audit = FinalAudit(
            str(config_path),
            "data/config/compute_budget.json",
            "data/config/implementation_config.json",
            str(meta_out / "stage4_execution_gate.json"),
            str(meta_out / "stage4_mechanism_gate.json"),
            str(meta_out / "stage4_materialization_audit.json"),
            str(meta_out / "stage4_pretraining_readiness.json"),
            str(meta_out / "data_split_manifest.json"),
            str(meta_out / "feature_target_audit.json"),
            str(meta_out / "target_leakage_report.json"),
            str(stage3_spec_path)
        )
        final_audit_report = final_audit.audit()
        logger.info(f"Final Audit completed. Training Allowed: {final_audit_report['training_allowed']}. Final Gate: {final_audit_report['final_gate_decision']}")
    
        logger.info("Running Final Experimental Readiness and Go/No-Go Gate (Stage 4G)...")
        go_no_go_gate = FinalReadinessGate(
            config_path=str(config_path),
            compute_budget_path="data/config/compute_budget.json",
            split_manifest_path=str(meta_out / "data_split_manifest.json"),
            target_leakage_path=str(meta_out / "target_leakage_report.json"),
            feature_target_audit_path=str(meta_out / "feature_target_audit.json"),
            materialization_audit_path=str(meta_out / "stage4_materialization_audit.json"),
            pretraining_readiness_path=str(meta_out / "stage4_pretraining_readiness.json"),
            representation_resolution_path=str(meta_out / "stage4_representation_resolution.json"),
            final_pretraining_audit_path=str(meta_out / "stage4_final_pretraining_audit.json"),
            stage2c_audit_path="evidence/metadata/stage2c_final_integrity_audit.json",
            out_path=str(meta_out / "stage4_final_readiness.json")
        )
        go_no_go_report = go_no_go_gate.evaluate()
        logger.info(f"Stage 4G completed. Final Decision: {go_no_go_report['final_decision']} | Training Allowed: {go_no_go_report['training_allowed']}")

        logger.info("Running Evidence-Backed Pipeline Blocker Resolution (Stage 4H)...")
        stage4h_resolver = Stage4HBlockerResolver(
            final_readiness_path=str(meta_out / "stage4_final_readiness.json"),
            materialization_audit_path=str(meta_out / "stage4_materialization_audit.json"),
            pretraining_readiness_path=str(meta_out / "stage4_pretraining_readiness.json"),
            representation_resolution_path=str(meta_out / "stage4_representation_resolution.json"),
            experiments_path="evidence/processed/experiments.jsonl",
            out_dir=str(meta_out)
        )
        stage4h_resolver.resolve()
        
        logger.info("Re-running Stage 4G Final Readiness Gate after Stage 4H resolution...")
        go_no_go_report_final = go_no_go_gate.evaluate()
        logger.info(f"Stage 4G (Re-run) completed. Final Decision: {go_no_go_report_final['final_decision']} | Training Allowed: {go_no_go_report_final['training_allowed']}")

    logger.info(f"Dry-run completed. Final status: {gate.execution_status.value}")

if __name__ == "__main__":
    main()
