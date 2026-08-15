import json
import logging
from pathlib import Path

from backend.app.stage2.models import PaperRecord, SourceScope
from backend.app.stage2.experiment_extractor import ExperimentExtractor
from backend.app.stage2.section_parser import SectionParser
from backend.app.stage2.full_text_fetcher import FullTextFetcher
from backend.app.stage2.graph_builder import GraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def repair_provenance():
    logger.info("Starting Provenance Repair (Reprocessing all 30 papers)")
    
    # Load papers
    papers = []
    with open("evidence/processed/papers.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                papers.append(PaperRecord.model_validate_json(line))
                
    section_parser = SectionParser()
    experiment_extractor = ExperimentExtractor()
    fetcher = FullTextFetcher()
    
    all_experiments = []
    all_claims = []
    all_mechanisms = []
    
    for paper in papers:
        logger.info(f"Extracting evidence from {paper.paper_id}")
        
        # 1. Fetch text
        ft_text, paper = fetcher.fetch(paper)
        if ft_text and paper.full_text_available:
            text_to_parse = ft_text
            scope = SourceScope.full_text
        elif paper.abstract:
            text_to_parse = paper.abstract
            scope = SourceScope.abstract
        else:
            continue
            
        sections = section_parser.parse(text_to_parse)
        
        exps, abls = experiment_extractor.extract(
            paper_id=paper.paper_id,
            text=text_to_parse,
            source_scope=scope,
            sections=sections
        )
        
        all_experiments.extend(exps)
        
    # We will just reuse the existing claims and mechanisms, 
    # or just let the downstream modules be. The user only wanted to repair experiments' modalities/baselines gaps.
    
    from backend.app.stage2.models import EvidenceClaim, Mechanism
    
    # Load claims
    with open("evidence/processed/evidence_claims.jsonl", "r", encoding="utf-8") as f:
        all_claims = [EvidenceClaim.model_validate_json(line) for line in f if line.strip()]
    # Load mechanisms
    with open("evidence/processed/mechanisms.jsonl", "r", encoding="utf-8") as f:
        all_mechanisms = [Mechanism.model_validate_json(line) for line in f if line.strip()]
        
    # Write back
    out_dir = Path("evidence/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "experiments.jsonl", "w", encoding="utf-8") as f:
        for exp in all_experiments:
            f.write(exp.model_dump_json() + "\n")
            
    with open(out_dir / "evidence_claims.jsonl", "w", encoding="utf-8") as f:
        for claim in all_claims:
            f.write(claim.model_dump_json() + "\n")
            
    with open(out_dir / "mechanisms.jsonl", "w", encoding="utf-8") as f:
        for mech in all_mechanisms:
            f.write(mech.model_dump_json() + "\n")
            
    graph_builder = GraphBuilder()
    graph = graph_builder.build_graph(papers=papers, claims=all_claims, mechanisms=all_mechanisms, experiments=all_experiments)
    
    with open(out_dir / "evidence_graph.json", "w", encoding="utf-8") as f:
        json.dump(graph.model_dump(), f, indent=2)
        
    logger.info("Provenance Repair complete. All derived files regenerated.")

if __name__ == "__main__":
    repair_provenance()
