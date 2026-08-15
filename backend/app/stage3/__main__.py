import json
from pathlib import Path
from backend.app.stage3.context_builder import ContextBuilder
from backend.app.stage3.evidence_matcher import EvidenceMatcher
from backend.app.stage3.belief_updater import BeliefUpdater
from backend.app.stage3.ranker import RankerComposer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_stage3():
    logger.info("Starting Stage 3: Evidence-Conditioned Mechanism Selection")
    
    # 1. Build Context
    builder = ContextBuilder()
    context = builder.build()
    logger.info(f"Context Built. Task: {context.task}, Modalities: {context.modalities}")
    
    # 2. Match Evidence
    matcher = EvidenceMatcher()
    matches = matcher.match_evidence(context)
    logger.info(f"Found {len(matches)} evidence matches.")
    
    # 3. Update Beliefs
    updater = BeliefUpdater()
    beliefs = updater.update_beliefs(matches)
    
    # 4. Rank and Compose
    ranker = RankerComposer(evidence_threshold=0.1)
    spec, rankings = ranker.rank_and_compose(context, beliefs)
    
    # 5. Output
    processed_dir = Path("evidence/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = Path("evidence/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    with open(processed_dir / "stage3_context.json", "w") as f:
        f.write(context.model_dump_json(indent=2))
        
    with open(processed_dir / "stage3_evidence_matches.jsonl", "w") as f:
        for match in matches:
            f.write(match.model_dump_json() + "\n")
            
    with open(processed_dir / "stage3_contextual_beliefs.jsonl", "w") as f:
        for belief in beliefs.values():
            f.write(belief.model_dump_json() + "\n")
            
    # Write rankings dictionary
    rankings_dict = {k.value: v.model_dump() for k, v in rankings.items()}
    with open(processed_dir / "stage3_mechanism_rankings.json", "w") as f:
        json.dump(rankings_dict, f, indent=2)
        
    with open(processed_dir / "stage3_pipeline_specification.json", "w") as f:
        f.write(spec.model_dump_json(indent=2))
        
    # Write report
    report = {
        "context": context.model_dump(),
        "evidence_matched": len(matches),
        "mechanism_rankings": rankings_dict,
        "selected_pipeline": spec.selected_mechanisms,
        "evidence_gaps": [k for k, v in spec.selected_mechanisms.items() if v is None]
    }
    with open(metadata_dir / "stage3_selection_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Stage 3 complete. Artifacts written to evidence/processed/")

if __name__ == "__main__":
    run_stage3()
