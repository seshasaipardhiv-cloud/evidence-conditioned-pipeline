import json
import logging
from datetime import datetime
from pathlib import Path
from backend.app.stage2.models import PaperRecord, SourceScope
from backend.app.stage2.orchestrator import Stage2Orchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def expand_corpus():
    selected_path = Path("evidence/metadata/selected_papers.jsonl")
    if not selected_path.exists():
        logger.error("No selected papers found.")
        return

    with open(selected_path) as f:
        selected = [json.loads(line) for line in f if line.strip()]

    # Load existing papers to protect them
    existing_papers_path = Path("evidence/processed/papers.jsonl")
    existing_pmids = set()
    existing_papers = []
    if existing_papers_path.exists():
        with open(existing_papers_path) as f:
            for line in f:
                if line.strip():
                    p = PaperRecord.model_validate_json(line)
                    existing_papers.append(p)
                    if p.pmid:
                        existing_pmids.add(p.pmid)
                        
    logger.info(f"Loaded {len(existing_papers)} existing seed papers.")

    # Create PaperRecords for new candidates
    new_papers = []
    for c in selected:
        if c["pmid"] in existing_pmids:
            continue
        pr = PaperRecord(
            paper_id=f"paper_{c['pmid']}",
            title=c["title"],
            authors=["Unknown"],
            publication_year=int(c["year"]) if str(c["year"]).isdigit() else 2024,
            doi=c.get("doi"),
            pmid=c["pmid"],
            source="PubMed",
            abstract=None,
            abstract_available=False,
            full_text_available=False,
            retrieval_date=datetime.now().isoformat()
        )
        new_papers.append(pr)

    logger.info(f"Adding {len(new_papers)} new papers to the corpus.")

    orch = Stage2Orchestrator()
    
    # 1. Stage 2A equivalent for new papers
    all_candidate_claims = []
    all_mechanisms = {}
    
    # Load existing mechanisms
    mech_path = Path("evidence/processed/mechanisms.jsonl")
    if mech_path.exists():
        from backend.app.stage2.models import Mechanism
        with open(mech_path) as f:
            for line in f:
                if line.strip():
                    m = Mechanism.model_validate_json(line)
                    all_mechanisms[m.mechanism_id] = m

    for paper in new_papers:
        # Fetch abstract if missing (we don't have it from esummary, but full text might give us something later)
        # We will rely heavily on full-text for these.
        candidates = orch.parser.parse_paper(paper)
        for claim, mechanism in candidates:
            all_candidate_claims.append(claim)
            if mechanism.mechanism_id not in all_mechanisms:
                all_mechanisms[mechanism.mechanism_id] = mechanism
            elif mechanism.mapping_status == "MAPPED":
                all_mechanisms[mechanism.mechanism_id] = mechanism
                
    validated_new_claims = orch.validator.validate_claims(all_candidate_claims)
    
    # Load existing claims
    existing_claims = []
    claims_path = Path("evidence/processed/evidence_claims.jsonl")
    if claims_path.exists():
        from backend.app.stage2.models import EvidenceClaim
        with open(claims_path) as f:
            for line in f:
                if line.strip():
                    existing_claims.append(EvidenceClaim.model_validate_json(line))
                    
    all_claims = existing_claims + validated_new_claims
    
    # 2. Stage 2B equivalent for new papers
    all_experiments = []
    # Load existing experiments
    exp_path = Path("evidence/processed/experiments.jsonl")
    if exp_path.exists():
        from backend.app.stage2.models import ExperimentRecord
        with open(exp_path) as f:
            for line in f:
                if line.strip():
                    all_experiments.append(ExperimentRecord.model_validate_json(line))

    all_ablations = []
    abl_path = Path("evidence/processed/ablations.jsonl")
    if abl_path.exists():
        from backend.app.stage2.models import AblationRecord
        with open(abl_path) as f:
            for line in f:
                if line.strip():
                    all_ablations.append(AblationRecord.model_validate_json(line))

    # Process new papers for full text
    import time
    updated_new_papers = []
    for paper in new_papers:
        logger.info(f"Fetching full text for {paper.paper_id}")
        text, paper = orch.fetcher.fetch(paper)
        updated_new_papers.append(paper)
            
        time.sleep(1.5)
        
        exps = []
        abls = []
        if paper.full_text_available and text:
            exps, abls = orch.exp_extractor.extract(paper.paper_id, text, SourceScope.full_text)
        elif paper.abstract_available and paper.abstract:
            exps, abls = orch.exp_extractor.extract(paper.paper_id, paper.abstract, SourceScope.abstract)
            
        all_experiments.extend(exps)
        all_ablations.extend(abls)
        
    combined_papers = existing_papers + updated_new_papers
    
    # Build updated graph
    graph = orch.graph_builder.build_graph(combined_papers, all_claims, list(all_mechanisms.values()))
    
    # Save everything back
    out_dir = Path("evidence/processed")
    with open(out_dir / "papers.jsonl", "w", encoding="utf-8") as f:
        for p in combined_papers:
            f.write(p.model_dump_json() + "\n")
    with open(out_dir / "evidence_claims.jsonl", "w", encoding="utf-8") as f:
        for c in all_claims:
            f.write(c.model_dump_json() + "\n")
    with open(out_dir / "mechanisms.jsonl", "w", encoding="utf-8") as f:
        for m in all_mechanisms.values():
            f.write(m.model_dump_json() + "\n")
    with open(out_dir / "experiments.jsonl", "w", encoding="utf-8") as f:
        for e in all_experiments:
            f.write(e.model_dump_json() + "\n")
    with open(out_dir / "ablations.jsonl", "w", encoding="utf-8") as f:
        for a in all_ablations:
            f.write(a.model_dump_json() + "\n")
    with open(out_dir / "evidence_graph.json", "w", encoding="utf-8") as f:
        f.write(graph.model_dump_json(indent=2))
        
    # Generate report
    report = {
        "seed_papers": len(existing_papers),
        "new_candidates": len(selected),
        "new_selected": len(new_papers),
        "final_corpus": len(combined_papers),
        "full_text_available": sum(1 for p in combined_papers if p.full_text_available),
        "abstract_only": sum(1 for p in combined_papers if p.abstract_available and not p.full_text_available),
        "unavailable": sum(1 for p in combined_papers if not p.full_text_available and not p.abstract_available),
        "tasks": {},
        "modalities": {},
        "fusion": {},
        "evidence": {
            "positive": sum(1 for c in all_claims if c.evidence_status and c.evidence_status.value == "positive"),
            "negative": sum(1 for c in all_claims if c.evidence_status and c.evidence_status.value == "negative"),
            "neutral": sum(1 for c in all_claims if c.evidence_status and c.evidence_status.value in ("neutral", "inconclusive")),
            "ablation": len(all_ablations)
        },
        "duplicates": len(selected) - len(new_papers),
        "rejected": 0
    }
    
    for e in all_experiments:
        if e.task: report["tasks"][e.task] = report["tasks"].get(e.task, 0) + 1
        if e.fusion_strategy: report["fusion"][e.fusion_strategy] = report["fusion"].get(e.fusion_strategy, 0) + 1
        for m in (e.modalities or []):
            report["modalities"][m] = report["modalities"].get(m, 0) + 1
            
    with open("evidence/metadata/corpus_expansion_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    logger.info("Corpus expansion complete.")

if __name__ == "__main__":
    expand_corpus()
