import json
import os
import uuid

def main():
    experiments_path = "evidence/processed/experiments.jsonl"
    claims_path = "evidence/processed/evidence_claims.jsonl"
    out_path = "evidence/metadata/evidence_gap_analysis.json"

    if not os.path.exists(experiments_path):
        return
        
    with open(experiments_path) as f:
        exps = [json.loads(line) for line in f if line.strip()]

    with open(claims_path) as f:
        claims = [json.loads(line) for line in f if line.strip()]

    counts = {
        "task": {"diagnosis": 0, "classification": 0, "subtyping": 0, "prognosis": 0, "survival_prediction": 0, "recurrence": 0},
        "modality": {"clinical": 0, "imaging": 0, "pathology": 0, "text": 0, "omics": 0, "laboratory/blood": 0},
        "fusion": {"early_fusion": 0, "intermediate_fusion": 0, "late_fusion": 0, "cross_attention": 0, "gated_fusion": 0, "joint_embedding": 0, "ensemble_fusion": 0},
        "mechanism": {"representation": 0, "preprocessing": 0, "feature selection": 0, "attention": 0, "fusion": 0, "regularization": 0, "classifier": 0, "ensembling": 0, "calibration": 0},
        "evidence_type": {"positive": 0, "negative": 0, "neutral/inconclusive": 0, "ablation": 0, "comparative": 0}
    }

    # Count tasks, modalities, fusions
    for e in exps:
        if e.get("task") in counts["task"]:
            counts["task"][e.get("task")] += 1
        for m in e.get("modalities", []):
            if m in counts["modality"]:
                counts["modality"][m] += 1
        if e.get("fusion_strategy") in counts["fusion"]:
            counts["fusion"][e.get("fusion_strategy")] += 1

    # Count evidence types
    for c in claims:
        # Simplistic mapping based on claim metadata
        if c.get("direction_of_improvement") == "positive":
            counts["evidence_type"]["positive"] += 1
        elif c.get("direction_of_improvement") == "negative":
            counts["evidence_type"]["negative"] += 1
        elif c.get("direction_of_improvement") == "neutral":
            counts["evidence_type"]["neutral/inconclusive"] += 1
            
        if c.get("comparison_context"):
            counts["evidence_type"]["comparative"] += 1
        # If it's an ablation claim, we check if baseline is the same method without something
        # For now, just mark ablation if baseline is empty or mentions 'unimodal' or 'w/o'
        bls = c.get("baseline", "")
        if bls and ("unimodal" in str(bls).lower() or "only" in str(bls).lower() or "w/o" in str(bls).lower()):
             counts["evidence_type"]["ablation"] += 1

    gaps = []
    
    def add_gap(cat, desc, count, priority, rationale):
        gaps.append({
            "gap_id": f"gap_{uuid.uuid4().hex[:8]}",
            "category": cat,
            "description": desc,
            "existing_evidence_count": count,
            "priority": priority,
            "search_rationale": rationale
        })

    # A. Tasks
    if counts["task"]["diagnosis"] + counts["task"]["classification"] < 2:
        add_gap("task", "Diagnosis/classification", counts["task"]["diagnosis"], "HIGH", "multimodal cancer diagnosis classification")
    
    if counts["task"]["subtyping"] < 1:
        add_gap("task", "Cancer subtyping", counts["task"]["subtyping"], "MEDIUM", "multimodal cancer subtyping clustering")

    if counts["task"]["recurrence"] < 1:
        add_gap("task", "Disease recurrence prediction", counts["task"]["recurrence"], "MEDIUM", "multimodal cancer recurrence prediction")

    # B. Modalities
    if counts["modality"]["pathology"] < 1:
         add_gap("modality", "Pathology/WSI images", counts["modality"]["pathology"], "HIGH", "multimodal cancer pathology whole slide image")
    
    if counts["modality"]["text"] < 2:
         add_gap("modality", "Clinical notes/reports", counts["modality"]["text"], "HIGH", "multimodal cancer NLP clinical text report")
         
    if counts["modality"]["omics"] < 2:
         add_gap("modality", "Omics (genomic/transcriptomic)", counts["modality"]["omics"], "HIGH", "multimodal cancer omics genomic integration")

    if counts["modality"]["clinical"] < 3:
         add_gap("modality", "Clinical tabular data", counts["modality"]["clinical"], "HIGH", "multimodal cancer clinical tabular features")

    # C. Fusion
    for f_type in ["early_fusion", "intermediate_fusion", "late_fusion", "cross_attention", "gated_fusion", "joint_embedding"]:
         if counts["fusion"][f_type] < 2:
             add_gap("fusion", f_type.replace('_', ' ').title(), counts["fusion"][f_type], "HIGH", f"multimodal cancer {f_type.replace('_', ' ')}")

    # E. Evidence Type
    if counts["evidence_type"]["negative"] < 2:
         add_gap("evidence_type", "Negative findings", counts["evidence_type"]["negative"], "MEDIUM", "multimodal cancer \"did not improve\" OR \"no significant\"")
         
    if counts["evidence_type"]["ablation"] < 2:
         add_gap("evidence_type", "Ablation/component analysis", counts["evidence_type"]["ablation"], "HIGH", "multimodal cancer ablation study component")

    with open(out_path, "w") as f:
        json.dump(gaps, f, indent=2)
        
    print(f"Wrote {len(gaps)} gaps to {out_path}")

if __name__ == "__main__":
    main()
