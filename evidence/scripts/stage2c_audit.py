import json
import re
from pathlib import Path

def normalize_title(title):
    if not title: return ""
    return re.sub(r'[^a-z0-9]', '', str(title).lower())

def validate_corpus():
    out_dir = Path("evidence/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    papers = []
    with open("evidence/processed/papers.jsonl") as f:
        papers = [json.loads(l) for l in f if l.strip()]
        
    claims = []
    with open("evidence/processed/evidence_claims.jsonl") as f:
        claims = [json.loads(l) for l in f if l.strip()]
        
    exps = []
    with open("evidence/processed/experiments.jsonl") as f:
        exps = [json.loads(l) for l in f if l.strip()]
        
    mechs = []
    with open("evidence/processed/mechanisms.jsonl") as f:
        mechs = [json.loads(l) for l in f if l.strip()]
        
    # Helpers
    suspicious_dataset = {"cancer", "raw", "data", "patients", "clinical", "model", "method", "framework", "approach", "images", "imaging"}
    suspicious_method = {"offers", "laying", "using", "shows", "improves", "approach", "framework", "model", "method"}
    
    audit_log = []
    
    # 1. Corpus Counts
    seed_papers = [p for p in papers if p["paper_id"].startswith("paper_10.") or p["paper_id"] in ["paper_38396486", "paper_39074400", "paper_40325104", "paper_40449048", "paper_41131352", "paper_41353186"]]
    # A bit more robust seed check: the original 8. 
    # Let's just find exactly the 8 known ones, but counting by length is fine for now, we'll verify it's 8.
    # The actual seeds are PMIDs: 38396486, 39074400, 40325104, 40449048, 41131352, 41353186
    # and DOIs: 10.1038/s42256-023-00633-5, 10.3390/bioengineering11010013
    known_seeds = {"paper_38396486", "paper_39074400", "paper_40325104", "paper_40449048", "paper_41131352", "paper_41353186", "paper_10.1038_s42256-023-00633-5", "paper_10.3390_bioengineering11010013"}
    
    seed_count = sum(1 for p in papers if p["paper_id"] in known_seeds)
    new_count = len(papers) - seed_count
    
    full_text_count = sum(1 for p in papers if p.get("full_text_available"))
    abstract_only_count = sum(1 for p in papers if p.get("abstract_available") and not p.get("full_text_available"))
    unavailable_count = sum(1 for p in papers if not p.get("full_text_available") and not p.get("abstract_available"))
    
    # 2. Duplicate Audit
    seen_doi, seen_pmid, seen_title = set(), set(), set()
    duplicate_papers = 0
    for p in papers:
        doi = p.get("doi")
        pmid = p.get("pmid")
        t_yr = f"{normalize_title(p.get('title'))}_{p.get('publication_year')}"
        
        dup = False
        if doi and doi in seen_doi: dup = True
        elif pmid and pmid in seen_pmid: dup = True
        elif p.get("title") and t_yr in seen_title: dup = True
        
        if dup:
            duplicate_papers += 1
            audit_log.append({"type": "DUPLICATE_PAPER", "id": p["paper_id"]})
        else:
            if doi: seen_doi.add(doi)
            if pmid: seen_pmid.add(pmid)
            if p.get("title"): seen_title.add(t_yr)

    # 3. Provenance Coverage
    total_fields = 0
    prov_fields = 0
    missing_prov_fields = []
    
    for c in claims:
        provs = c.get("provenance")
        for f in ["claim", "task", "dataset_characteristics", "baseline", "metric", "result", "experimental_conditions", "limitations"]:
            val = c.get(f)
            if val:
                total_fields += 1
                if provs:
                    prov_fields += 1
                else:
                    missing_prov_fields.append({"type": "CLAIM_MISSING_PROV", "id": c.get("evidence_id"), "field": f})
                    
    for e in exps:
        for f in ["dataset", "task", "modalities", "fusion_strategy", "proposed_method", "baselines", "results"]:
            val = e.get(f)
            if val:
                total_fields += 1
                provs = e.get("field_provenance", {})
                if f in provs:
                    prov_fields += 1
                else:
                    missing_prov_fields.append({"type": "EXP_MISSING_PROV", "id": e.get("experiment_id"), "field": f})
                    
    for m in mechs:
        val = m.get("name")
        if val:
            total_fields += 1
            provs = m.get("field_provenance", {})
            if "name" in provs:
                prov_fields += 1
            else:
                missing_prov_fields.append({"type": "MECH_MISSING_PROV", "id": m.get("mechanism_id"), "field": "name"})

    # 4. Entity Sanity Audit
    suspicious_count = 0
    for e in exps:
        d = str(e.get("dataset") or "").lower()
        if d in suspicious_dataset:
            suspicious_count += 1
            audit_log.append({"type": "SUSPICIOUS_DATASET", "id": e["experiment_id"], "value": d})
            
        pm = str(e.get("proposed_method") or "").lower()
        if pm in suspicious_method:
            suspicious_count += 1
            audit_log.append({"type": "SUSPICIOUS_METHOD", "id": e["experiment_id"], "value": pm})

    # PCA regression
    for m in mechs:
        if m.get("name") == "PCA" and "csPCa" in str(m.get("description", "")):
             suspicious_count += 1
             audit_log.append({"type": "MECHANISM_PCA_REGRESSION", "id": m["mechanism_id"]})
             
    # 5. Entity Type Confusion
    entity_confusion = 0
    for e in exps:
        pm = e.get("proposed_method")
        ds = e.get("dataset")
        bls = e.get("baselines", [])
        
        if pm and ds and str(pm).lower() == str(ds).lower():
            entity_confusion += 1
            audit_log.append({"type": "CONFUSION_METHOD_DATASET", "id": e["experiment_id"]})
        if pm and bls:
            for b in bls:
                if str(pm).lower() == str(b).lower():
                    entity_confusion += 1
                    audit_log.append({"type": "CONFUSION_METHOD_BASELINE", "id": e["experiment_id"]})
                    
    # 6. Numerical Evidence
    numerical_errors = 0
    for e in exps:
        for r in e.get("results", []):
            val = r.get("method_value")
            if val is not None:
                # Find provenance
                provs = e.get("field_provenance", {})
                found_prov = provs.get("results")
                if found_prov:
                    sentence = str(found_prov.get("source_sentence", ""))
                    if str(val) not in sentence:
                        numerical_errors += 1
                        audit_log.append({"type": "FABRICATED_NUMERICAL", "id": e.get("experiment_id"), "value": val, "sentence": sentence})

    # 7. Negative Evidence & 9. Corpus Coverage
    coverage = {
        "tasks": {},
        "modalities": {},
        "fusion": {},
        "evidence": {"positive": 0, "negative": 0, "neutral": 0, "unknown": 0, "ablation": 0, "comparative": 0}
    }
    
    for c in claims:
        # Pydantic may serialize enums as dict or string. Handle both
        status = c.get("evidence_status")
        if isinstance(status, dict) and "value" in status:
             status = status["value"]
        elif hasattr(status, "value"):
             status = status.value
        
        if status == "positive": coverage["evidence"]["positive"] += 1
        elif status == "negative": coverage["evidence"]["negative"] += 1
        elif status in ["neutral", "inconclusive"]: coverage["evidence"]["neutral"] += 1
        else: coverage["evidence"]["unknown"] += 1
        
        if c.get("baseline"): coverage["evidence"]["comparative"] += 1
        
    try:
        with open("evidence/processed/ablations.jsonl") as f:
            coverage["evidence"]["ablation"] = sum(1 for _ in f if _.strip())
    except FileNotFoundError:
        pass

    for e in exps:
        if e.get("task"): coverage["tasks"][e.get("task")] = coverage["tasks"].get(e.get("task"), 0) + 1
        if e.get("fusion_strategy"): coverage["fusion"][e.get("fusion_strategy")] = coverage["fusion"].get(e.get("fusion_strategy"), 0) + 1
        for m in (e.get("modalities") or []):
            coverage["modalities"][m] = coverage["modalities"].get(m, 0) + 1

    # 10. Output & 11. Acceptance
    corpus_valid = (
        len(papers) == 30 and 
        seed_count == 8 and
        duplicate_papers == 0 and
        len(missing_prov_fields) == 0 and
        suspicious_count == 0 and 
        numerical_errors == 0 and
        entity_confusion == 0
    )
    
    summary = {
        "corpus_valid": corpus_valid,
        "critical_errors": len([x for x in missing_prov_fields]) + duplicate_papers + suspicious_count + numerical_errors + entity_confusion,
        "warnings": 0,
        "corpus_counts": {
            "total_papers": len(papers),
            "seed_papers": seed_count,
            "new_papers": new_count,
            "full_text_papers": full_text_count,
            "abstract_only_papers": abstract_only_count,
            "unavailable_papers": unavailable_count
        },
        "duplicate_count": {
            "duplicate_papers": duplicate_papers,
            "duplicate_experiments": 0, # not tracked explicitly here yet
            "duplicate_claims": 0
        },
        "provenance_coverage": {
            "total_non_null_fields": total_fields,
            "fields_with_provenance": prov_fields,
            "fields_without_provenance": len(missing_prov_fields),
            "provenance_coverage_percent": round(prov_fields / total_fields * 100, 2) if total_fields else 100.0
        },
        "suspicious_entity_count": suspicious_count,
        "numerical_evidence_errors": numerical_errors,
        "entity_confusion_errors": entity_confusion,
        "coverage": coverage
    }
    
    full_audit = {
        "summary": summary,
        "missing_provenance": missing_prov_fields,
        "audit_log": audit_log
    }
    
    with open(out_dir / "stage2c_final_integrity_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    with open(out_dir / "stage2c_final_integrity_audit.json", "w") as f:
        json.dump(full_audit, f, indent=2)
        
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    validate_corpus()
