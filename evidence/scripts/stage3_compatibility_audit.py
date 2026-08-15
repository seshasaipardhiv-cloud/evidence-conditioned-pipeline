import json
import logging
import hashlib
from pathlib import Path
from backend.app.stage3.compatibility import CompatibilityAuditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hash_file(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def main():
    logger.info("Starting Stage 3.1 Compatibility Audit")
    
    data_dir = Path("data/processed/hancock")
    ev_dir = Path("evidence/processed")
    
    # 1. Fingerprint Stage 2 artifacts
    stage2_files = [
        ev_dir / "experiments.jsonl",
        ev_dir / "evidence_claims.jsonl",
        ev_dir / "evidence_graph.json"
    ]
    
    initial_hashes = {f: hash_file(f) for f in stage2_files}

    # 2. Load Artifacts
    with open(data_dir / "stage1_problem_representation.json", "r", encoding="utf-8") as f:
        context = json.load(f)
        
    with open(ev_dir / "stage3_pipeline_specification.json", "r", encoding="utf-8") as f:
        pipeline_spec = json.load(f)
        
    with open(ev_dir / "stage3_mechanism_rankings.json", "r", encoding="utf-8") as f:
        rankings = json.load(f)
        
    claims = []
    with open(ev_dir / "evidence_claims.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                claims.append(json.loads(line))
                
    experiments = []
    with open(ev_dir / "experiments.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))

    # 3. Audit
    auditor = CompatibilityAuditor(
        stage1_context=context,
        pipeline_spec=pipeline_spec,
        mechanism_rankings=rankings,
        evidence_claims=claims,
        experiments=experiments
    )
    
    report = auditor.audit()

    # 4. Save metadata audit output
    metadata_out = Path("evidence/metadata/stage3_compatibility_audit.json")
    with open(metadata_out, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2)
        
    # 5. Save validated pipeline specification
    validated_spec_out = ev_dir / "stage3_validated_pipeline_specification.json"
    with open(validated_spec_out, "w", encoding="utf-8") as f:
        json.dump(pipeline_spec, f, indent=2)
        
    # 6. Verify integrity
    final_hashes = {f: hash_file(f) for f in stage2_files}
    modified = any(initial_hashes[f] != final_hashes[f] for f in stage2_files)
    
    if modified:
        logger.error("Stage 2 artifacts were modified during the audit!")
        raise RuntimeError("Stage 2 artifacts modified")
        
    logger.info(f"Audit completed. Target blocked: {report.target_gate.blocked}")
    logger.info(f"Stage 2 artifacts modified: {modified}")
    
if __name__ == "__main__":
    main()
