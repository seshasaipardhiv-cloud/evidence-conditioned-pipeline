import json

def generate_report():
    with open("evidence/processed/papers.jsonl") as f:
        papers = [json.loads(l) for l in f if l.strip()]
    with open("evidence/processed/evidence_claims.jsonl") as f:
        claims = [json.loads(l) for l in f if l.strip()]
    with open("evidence/processed/experiments.jsonl") as f:
        exps = [json.loads(l) for l in f if l.strip()]
    
    abls = []
    try:
        with open("evidence/processed/ablations.jsonl") as f:
            abls = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        pass

    with open("evidence/metadata/selected_papers.jsonl") as f:
        selected = [json.loads(l) for l in f if l.strip()]

    report = {
        "seed_papers": 8,
        "new_candidates": len(selected),
        "new_selected": len(papers) - 8,
        "final_corpus": len(papers),
        "full_text_available": sum(1 for p in papers if p.get("full_text_available")),
        "abstract_only": sum(1 for p in papers if p.get("abstract_available") and not p.get("full_text_available")),
        "unavailable": sum(1 for p in papers if not p.get("full_text_available") and not p.get("abstract_available")),
        "tasks": {},
        "modalities": {},
        "fusion": {},
        "evidence": {
            "positive": sum(1 for c in claims if c.get("evidence_status") == "positive"),
            "negative": sum(1 for c in claims if c.get("evidence_status") == "negative"),
            "neutral": sum(1 for c in claims if c.get("evidence_status") in ("neutral", "inconclusive")),
            "ablation": len(abls)
        },
        "duplicates": len(selected) - (len(papers) - 8),
        "rejected": 0
    }
    
    for e in exps:
        if e.get("task"): report["tasks"][e.get("task")] = report["tasks"].get(e.get("task"), 0) + 1
        if e.get("fusion_strategy"): report["fusion"][e.get("fusion_strategy")] = report["fusion"].get(e.get("fusion_strategy"), 0) + 1
        for m in (e.get("modalities") or []):
            report["modalities"][m] = report["modalities"].get(m, 0) + 1
            
    with open("evidence/metadata/corpus_expansion_report.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    generate_report()
