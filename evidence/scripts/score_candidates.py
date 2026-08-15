import json
import random

def score_and_select():
    candidates_path = "evidence/metadata/candidate_papers.jsonl"
    out_path = "evidence/metadata/selected_papers.jsonl"
    
    with open(candidates_path) as f:
        candidates = [json.loads(line) for line in f if line.strip()]
        
    # Group by gap_target
    gaps_map = {}
    for c in candidates:
        g = c.get("gap_target", "Unknown")
        if g not in gaps_map:
            gaps_map[g] = []
        gaps_map[g].append(c)
        
    selected = []
    seen_pmids = set()
    
    # Target 2-3 per gap, total around 20
    for gap, cand_list in gaps_map.items():
        # Score relevance (simplistic for this expansion: 10 if multimodal/fusion/cancer in title)
        for c in cand_list:
            t = c.get("title", "").lower()
            score = 1
            if "cancer" in t or "tumor" in t or "carcinoma" in t:
                score += 3
            if "multimodal" in t or "multi-modal" in t or "fusion" in t:
                score += 3
            if "deep learning" in t or "machine learning" in t or "artificial intelligence" in t:
                score += 2
            c["relevance_score"] = score
            
        cand_list.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Select top 2
        count = 0
        for c in cand_list:
            if c["pmid"] not in seen_pmids:
                selected.append(c)
                seen_pmids.add(c["pmid"])
                count += 1
            if count >= 2:
                break
                
    # We want 12-22 papers to add to our 8 seed papers (total 20-30).
    if len(selected) > 22:
        selected = selected[:22]
        
    print(f"Selected {len(selected)} papers from {len(candidates)} candidates.")
    
    with open(out_path, "w") as f:
        for s in selected:
            f.write(json.dumps(s) + "\n")
            
if __name__ == "__main__":
    score_and_select()
